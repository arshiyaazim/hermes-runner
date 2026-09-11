"""config.fazle_ai.retry_policy

Pure-Python bounded retry + fallback executor for AI model calls.

This module owns:
  * FailureClass — the closed enum of failure classes callers must use.
  * RetryDecision — what to do next (retry current target, skip to next,
    fail-closed).
  * decide_retry() — given (failure_class, attempts_made_so_far,
    retry_rule), returns the next decision. Pure function. No I/O.
  * execute_with_retries() — generic executor that walks a workload's
    fallback_chain, applies the policy's retry rules, and is bounded by:
      - max_attempts_per_fallback_target from the workload
      - total_max_attempts_across_chain from the workload
      - a visited-set invariant: no fallback target is selected twice

The executor NEVER:
  * logs or echoes the auth header,
  * logs or echoes any secret value,
  * retries after auth_failed / authorization_failed / invalid_request /
    policy_rejection / safety_refusal,
  * selects a fallback target it has already tried (per the visited set).

Auth-failure detection is intentionally broad: any failure matching the
hard-fail classes causes an immediate PolicyHardFailError that aborts
the chain. The runner can surface that to ops via the
`emit_health_signal_on_auth_failure` shared-policy flag.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from config.fazle_ai.policy_loader import (
    Policy,
    PolicyValidationError,
    RetryRule,
    WorkloadConfig,
    redact_for_log,
)


# ── Public types ────────────────────────────────────────────────────────────


class FailureClass(str, Enum):
    """Closed set of failure classes callers must use.

    Values are strings (not int) so they can be logged safely and matched
    against Policy.retry_policy keys. The set is closed by design: adding
    a new failure class requires extending the enum AND adding a
    corresponding retry rule in policy.yaml.
    """

    NETWORK_TIMEOUT = "network_timeout"
    TRANSIENT_5XX = "transient_5xx"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    PROVIDER_OR_MODEL_UNAVAILABLE = "provider_or_model_unavailable"
    CONTEXT_TOO_LARGE = "context_too_large"
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    INVALID_REQUEST = "invalid_request"
    POLICY_REJECTION = "policy_rejection"
    SAFETY_REFUSAL = "safety_refusal"
    UNKNOWN = "unknown"


# Hard-fail classes: never retry, never substitute fallback. The runner
# must fail closed and emit a health signal. These map 1:1 onto
# policy.yaml's auth_or_authorization_failed + invalid_request_or_policy_or_safety_refusal
# retry-policy keys (which are both configured with max_attempts_same_target=0).
_HARD_FAIL_CLASSES = frozenset({
    FailureClass.AUTHENTICATION_FAILED,
    FailureClass.AUTHORIZATION_FAILED,
    FailureClass.INVALID_REQUEST,
    FailureClass.POLICY_REJECTION,
    FailureClass.SAFETY_REFUSAL,
})


class PolicyHardFailError(RuntimeError):
    """Raised by execute_with_retries when a hard-fail class is encountered.

    Carries no secret values. The runner is expected to surface this to
    ops (alert / health endpoint) and NOT to silently substitute a
    fallback route.
    """


# Hard-retry classes: retry exactly once then move on (matches
# policy.yaml's `max_attempts_same_target=2` for these classes, where the
# first attempt is the original call and the second is the retry).
_TRANSIENT_RETRY_CLASSES = frozenset({
    FailureClass.NETWORK_TIMEOUT,
    FailureClass.TRANSIENT_5XX,
    FailureClass.RATE_LIMITED,
    FailureClass.CONTEXT_TOO_LARGE,
})


@dataclass(frozen=True)
class RetryDecision:
    action: str  # one of: "retry" | "next_fallback" | "fail_closed"
    backoff_s: float = 0.0
    reason: str = ""

    @property
    def is_retry(self) -> bool:
        return self.action == "retry"

    @property
    def is_next_fallback(self) -> bool:
        return self.action == "next_fallback"

    @property
    def is_fail_closed(self) -> bool:
        return self.action == "fail_closed"


# ── Pure-function decision (the heart of the policy) ──────────────────────


def _policy_retry_rule_key(failure_class: FailureClass) -> str:
    """Map a FailureClass to the corresponding policy.yaml retry_policy key.

    Centralized so the mapping can be inspected and unit-tested. Each
    FailureClass maps to exactly one policy key.
    """
    if failure_class in _HARD_FAIL_CLASSES:
        # The policy uses two keys for hard-fail classes:
        #   * auth_or_authorization_failed
        #   * invalid_request_or_policy_or_safety_refusal
        # Both are configured with max_attempts_same_target=0.
        if failure_class in {
            FailureClass.AUTHENTICATION_FAILED,
            FailureClass.AUTHORIZATION_FAILED,
        }:
            return "auth_or_authorization_failed"
        return "invalid_request_or_policy_or_safety_refusal"
    if failure_class in _TRANSIENT_RETRY_CLASSES:
        # Two policy keys cover the transient cluster:
        #   * network_timeout_or_transient_5xx  (covers NETWORK_TIMEOUT + TRANSIENT_5XX)
        #   * rate_limited                        (covers RATE_LIMITED)
        # CONTEXT_TOO_LARGE has its own key.
        if failure_class == FailureClass.RATE_LIMITED:
            return "rate_limited"
        if failure_class == FailureClass.CONTEXT_TOO_LARGE:
            return "context_too_large"
        return "network_timeout_or_transient_5xx"
    # Non-retry hard classes: quota_exhausted, provider_or_model_unavailable.
    # policy.yaml has two keys for these (each configured max_attempts_same_target=0):
    #   * quota_exhausted
    #   * provider_or_model_unavailable
    if failure_class == FailureClass.QUOTA_EXHAUSTED:
        return "quota_exhausted"
    if failure_class == FailureClass.PROVIDER_OR_MODEL_UNAVAILABLE:
        return "provider_or_model_unavailable"
    return ""


def decide_retry(
    failure_class: FailureClass,
    attempts_made_on_target: int,
    policy: Policy,
    retry_after_s: float | None = None,
) -> RetryDecision:
    """Decide the next action after one failure on a target.

    Pure function. No I/O. No clock. No side effects.

    `attempts_made_on_target` is the count of attempts so far INCLUDING
    the original call. First call: 1. After 1st retry: 2. Etc.

    `retry_after_s` is the server-provided Retry-After value (in seconds)
    for a RATE_LIMITED response. Ignored for all other failure classes.

    Owner rule (task #22, 2026-08-25):
      * NETWORK_TIMEOUT and TRANSIENT_5XX: attempt ONCE, retry AT MOST ONCE.
        The retry uses the only backoff_s value configured (2 seconds).
        After one retry fails, advance to next fallback. NO second retry.
      * RATE_LIMITED: if Retry-After is present AND is within the policy's
        retry_after_safe_maximum_s, wait that long and retry ONCE.
        Otherwise skip directly to next fallback. After one rate-limit
        retry fails, advance to next fallback. No repeated 429 retry loop.

    Hard-fail classes ALWAYS return `fail_closed` regardless of
    attempts_made_on_target — there is no safe retry path for them.

    The contract:
      * attempts_made_on_target < max_attempts_same_target + 1
        => returns `retry` (with the next backoff_s value).
      * attempts_made_on_target >= max_attempts_same_target + 1
        => returns `next_fallback` (or `fail_closed` for hard-fail classes).
      * `failure_class` not in policy.retry_policy => returns `next_fallback`
        (defensive default; policy should enumerate every class it cares about).
    """
    # Hard-fail classes ALWAYS fail closed, regardless of attempts so far.
    if failure_class in _HARD_FAIL_CLASSES:
        return RetryDecision(
            action="fail_closed",
            reason=f"hard-fail class {failure_class.value!r}: no retry, no fallback substitution",
        )

    # Rate-limited special case: respect Retry-After only if it's within safe max.
    # Owner rule (task #22, 2026-08-25): "If Retry-After is absent, invalid,
    # or exceeds configured maximum: skip directly to next approved fallback."
    # So absent-or-too-large -> next_fallback on the FIRST failure too (no retry).
    if failure_class == FailureClass.RATE_LIMITED:
        rl = policy.rate_limit_policy
        if retry_after_s is None:
            return RetryDecision(
                action="next_fallback",
                reason=(
                    f"Retry-After absent; skipping to next fallback (no 429 retry loop; "
                    f"honor_retry_after_if_within_max={rl.honor_retry_after_if_within_max})"
                ),
            )
        if not rl.honor_retry_after_if_within_max:
            return RetryDecision(
                action="next_fallback",
                reason="honor_retry_after_if_within_max is disabled in policy",
            )
        if not (0 <= float(retry_after_s) <= rl.retry_after_safe_maximum_s):
            return RetryDecision(
                action="next_fallback",
                reason=(
                    f"Retry-After={retry_after_s!r} exceeds safe max "
                    f"{rl.retry_after_safe_maximum_s}s (or is negative); skip to next fallback"
                ),
            )
        # Retry-After is present, in range, and policy honors it.
        # One-shot retry, regardless of attempts_made_on_target above 1.
        if rl.one_shot and attempts_made_on_target >= 2:
            return RetryDecision(
                action="next_fallback",
                reason=(
                    f"Retry-After={retry_after_s!r} honored once already; "
                    f"rate_limit_policy.one_shot=true -> advance to next fallback"
                ),
            )
        return RetryDecision(
            action="retry",
            backoff_s=float(retry_after_s),
            reason=(
                f"Retry-After={float(retry_after_s):.1f}s within safe max "
                f"{rl.retry_after_safe_maximum_s}s; one-shot retry on this target"
            ),
        )

    key = _policy_retry_rule_key(failure_class)
    if not key:
        return RetryDecision(
            action="next_fallback",
            reason=f"failure_class {failure_class.value!r} has no policy rule; advancing to next fallback",
        )
    rule = policy.retry_policy.get(key)
    if rule is None:
        return RetryDecision(
            action="next_fallback",
            reason=f"policy.retry_policy missing key {key!r}; advancing to next fallback",
        )

    # attempts_made_on_target is 1-based. The rule's max_attempts is also
    # 1-based (max_attempts_same_target=2 means "1 original + 1 retry").
    # Owner rule (task #22, 2026-08-25):
    #   For NETWORK_TIMEOUT and TRANSIENT_5XX (and other transient classes
    #   that use this generic path): retry EXACTLY ONCE. After one retry
    #   fails, advance to next fallback. NO second retry, NO third call
    #   to the same provider/model.
    # The comparison is strict less-than: when attempts_made_on_target
    # reaches max_attempts_same_target, we have already used our one
    # retry; the next decision is next_fallback.
    if attempts_made_on_target < rule.max_attempts_same_target:
        # Compute the backoff. For NETWORK_TIMEOUT/TRANSIENT_5XX the
        # policy now mandates backoff_s: [2] (single value, used for the
        # one retry). For other classes, use the configured list.
        idx = max(0, min(attempts_made_on_target - 1, len(rule.backoff_s) - 1))
        backoff = float(rule.backoff_s[idx]) if rule.backoff_s else 0.0
        return RetryDecision(
            action="retry",
            backoff_s=backoff,
            reason=(
                f"retry class {failure_class.value!r} attempt "
                f"{attempts_made_on_target} of {rule.max_attempts_same_target} "
                f"(one-shot; next failure advances to next fallback)"
            ),
        )

    return RetryDecision(
        action="next_fallback",
        reason=(
            f"retry budget exhausted for class {failure_class.value!r} on this target; "
            f"advancing to next fallback (no third call to same provider/model)"
        ),
    )


# ── Bounded executor (the policy in motion) ───────────────────────────────


@dataclass
class AttemptRecord:
    """A single attempt's outcome for logging/debug (no secret values).

    Used by execute_with_retries() to build the trailing chain report.
    """

    provider: str
    model: str
    failure_class: str
    action_taken: str
    backoff_s: float
    elapsed_s: float


def execute_with_retries(
    workload: WorkloadConfig,
    policy: Policy,
    call: Callable[[str, str], "_CallResult"],
    *,
    truncate_for_context: Callable[["_CallResult"], "_CallResult"] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> "_CallResult":
    """Walk workload.fallback_chain, applying retry_policy per target.

    `call(provider, model)` returns a `_CallResult` populated with
    content (str), success (bool), failure_class (FailureClass), and
    raw_error (str). The executor uses these to decide what to do next.

    `truncate_for_context` is invoked when a call fails with
    CONTEXT_TOO_LARGE. The truncated result is re-tried exactly once on
    the same target (per policy.context_too_large), then skipped.

    `sleep` is injected so tests can avoid real wall-clock waits.

    Bounds:
      * max_attempts_per_fallback_target (from workload.retry_budget)
      * total_max_attempts_across_chain (from workload.retry_budget)
      * visited-set invariant: no (provider, model) target is selected twice.

    Raises PolicyHardFailError on any hard-fail class.
    """
    attempts_total = 0
    max_total = int(workload.retry_budget.get("total_max_attempts_across_chain", 10))
    max_per_target = int(workload.retry_budget.get("max_attempts_per_fallback_target", 2))
    visited: set[tuple[str, str]] = set()
    history: list[AttemptRecord] = []

    for provider, model in workload.fallback_chain:
        if (provider, model) in visited:
            # Visited-set invariant: never select the same fallback twice.
            return _final_failure(
                history,
                reason=(
                    f"fallback chain contained duplicate ({provider!r}, {model!r}); "
                    f"visited-set invariant triggered"
                ),
            )
        visited.add((provider, model))

        attempts_on_target = 0
        while True:
            attempts_on_target += 1
            attempts_total += 1
            if attempts_total > max_total:
                return _final_failure(
                    history,
                    reason=f"total attempt budget {max_total} exhausted across the chain",
                )

            t0 = time.monotonic()
            result = call(provider, model)
            elapsed = time.monotonic() - t0
            result = _coerce_call_result(result)

            if result.success:
                return result

            fc = result.failure_class or FailureClass.UNKNOWN

            # Handle context_too_large with a one-shot truncation retry on
            # the same target. The truncated retry does NOT count as a
            # separate attempt against the per-target budget when the
            # policy says so (it currently does; max_attempts_same_target
            # for context_too_large = 1 == 1 original + 0 retries; we
            # therefore try the truncation, but if it ALSO fails, we
            # honor the visited set and move on).
            if (
                fc == FailureClass.CONTEXT_TOO_LARGE
                and truncate_for_context is not None
                and attempts_on_target == 1
            ):
                # Single truncation retry (one-shot).
                truncated = truncate_for_context(result)
                t1 = time.monotonic()
                truncated = _coerce_call_result(truncated)
                truncated_elapsed = time.monotonic() - t1
                attempts_total += 1
                if truncated.success:
                    return truncated
                history.append(AttemptRecord(
                    provider=provider, model=model,
                    failure_class=fc.value,
                    action_taken="truncate_retry_failed",
                    backoff_s=0.0,
                    elapsed_s=truncated_elapsed,
                ))
                # After one truncate retry that also failed, fall through
                # to the standard decide_retry path with attempts_on_target=2.

            # Decide what to do next. Pass retry_after_s only when the
            # failure was RATE_LIMITED (decide_retry ignores it otherwise).
            retry_after = (
                result.retry_after_s
                if fc == FailureClass.RATE_LIMITED
                else None
            )
            decision = decide_retry(
                failure_class=fc,
                attempts_made_on_target=attempts_on_target,
                policy=policy,
                retry_after_s=retry_after,
            )

            history.append(AttemptRecord(
                provider=provider, model=model,
                failure_class=fc.value,
                action_taken=decision.action,
                backoff_s=decision.backoff_s,
                elapsed_s=elapsed,
            ))

            if decision.is_fail_closed:
                # Hard-fail: never substitute a fallback. Raise so the
                # runner can surface a health signal.
                raise PolicyHardFailError(
                    f"hard-fail class {fc.value!r} on ({provider!r}, {model!r}): "
                    f"{decision.reason}; not attempting fallback to another credential. "
                    f"All attempts so far: {len(history)}."
                )

            if decision.is_retry:
                if attempts_on_target >= max_per_target:
                    # Per-target budget exhausted; advance.
                    break
                if decision.backoff_s > 0:
                    sleep(decision.backoff_s)
                continue

            # next_fallback: advance to the next entry in the workload
            # chain.
            break

        # Loop exhausted this target; advance to the next.
        continue

    # Exhausted every entry in the fallback chain (no break-and-throw
    # above; all reached `next_fallback`).
    return _final_failure(
        history,
        reason="all configured fallback routes exhausted without a successful response",
    )


# ── Call-result contract ───────────────────────────────────────────────────


@dataclass
class _CallResult:
    success: bool
    content: str = ""
    failure_class: FailureClass | None = None
    raw_error: str = ""
    # Server-provided Retry-After (seconds) for a RATE_LIMITED response.
    # Optional; defaults to None (treated as "absent"). Caller-supplied;
    # only honored when failure_class == RATE_LIMITED AND the value is
    # within the policy's retry_after_safe_maximum_s.
    retry_after_s: float | None = None


def _coerce_call_result(value: Any) -> _CallResult:
    """Normalize whatever the caller returned into a _CallResult.

    Accepts:
      * _CallResult (returned as-is)
      * dict with keys success/content/failure_class/raw_error/retry_after_s
      * any object with a `success` boolean attribute
    """
    if isinstance(value, _CallResult):
        return value
    if isinstance(value, dict):
        fc = value.get("failure_class")
        if isinstance(fc, str):
            try:
                fc = FailureClass(fc)
            except ValueError:
                fc = FailureClass.UNKNOWN
        retry_after = value.get("retry_after_s")
        if retry_after is not None:
            try:
                retry_after = float(retry_after)
            except (TypeError, ValueError):
                retry_after = None
        return _CallResult(
            success=bool(value.get("success", False)),
            content=str(value.get("content", "")),
            failure_class=fc,
            raw_error=str(value.get("raw_error", "")),
            retry_after_s=retry_after,
        )
    # Best-effort duck typing.
    success = bool(getattr(value, "success", False))
    return _CallResult(
        success=success,
        content=str(getattr(value, "content", "")),
        failure_class=getattr(value, "failure_class", None),
        raw_error=str(getattr(value, "raw_error", "")),
        retry_after_s=getattr(value, "retry_after_s", None),
    )


def _final_failure(history: list[AttemptRecord], *, reason: str) -> _CallResult:
    """Wrap an all-routes-exhausted result. No secrets in the message."""
    summary = "; ".join(
        f"({a.provider},{a.model}) class={a.failure_class} action={a.action_taken}"
        for a in history
    )
    return _CallResult(
        success=False,
        content="",
        failure_class=FailureClass.UNKNOWN,
        raw_error=f"{reason}. attempts={redact_for_log(len(history))} chain=[{summary}]",
    )
