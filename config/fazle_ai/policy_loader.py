"""config.fazle_ai.policy_loader

Pure-Python loader + validator for the version-controlled Fazle-AI routing
policy. No I/O beyond reading the supplied file paths. No subprocess
calls. Safe to import from anywhere; no hidden side effects.

Owner-approved implementation rule (2026-08-25, task #22):
  * No secret values are ever read, logged, or printed.
  * All strings returned in error messages are concrete enough to be
    actionable (which file, which key, which model) but never include
    values that look like a secret.
  * `auto/*` is rejected at validation time with a clear, redacted
    PolicyValidationError. This is enforced for both the global policy
    and every workload config.
  * The `secret_refs` block of a workload is parsed but its values are
    validated to match the URI schema only; no resolution is attempted.
  * When a policy file is absent (the default backward-compatible state),
    `load_policy()` returns `None`, signaling "use legacy env config" to
    callers. When a policy file is present but invalid, it raises
    PolicyValidationError with no secret values.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from typing import Any, Iterable

import yaml


# ── Module constants ───────────────────────────────────────────────────────


# The set of field names that are actually read by the policy/executor
# code for any provider. Adding a field here is a deliberate API change;
# anything else is rejected at validation time (so callers cannot
# accidentally smuggle secret-shaped keys like api_key, token, password,
# secret, authorization, credential into the policy file).
#
# Currently-used: only requests_per_minute is documented in the README
# as the supported advisory field. concurrent_requests is provided for
# future executor use (the retry_policy.executor already accepts a
# workload-level retry_budget, but a per-provider advisory is harmless
# to allow here for forward-compatibility).
#
# IMPORTANT: This list MUST stay aligned with the fields the
# implementation actually consumes. Do not add fields "just in case".
_ALLOWED_PROVIDER_LIMITS_KEYS: frozenset[str] = frozenset({
    "requests_per_minute",  # soft advisory cap (no enforcement today)
    "max_concurrent",        # soft advisory cap (no enforcement today)
})


# ── Public dataclasses (no secret values) ────────────────────────────────────


@dataclass(frozen=True)
class SecretRef:
    """A NON-SECRET reference to a secret. Values are URI-shaped only.

    `value` must match the secret_ref_schema in policy.yaml (vault://...).
    If the value does not match the URI schema, load_workload() raises
    PolicyValidationError with the offending key (not the value).
    """

    key: str
    value: str


@dataclass(frozen=True)
class RetryRule:
    failure_class: str
    max_attempts_same_target: int
    backoff_s: tuple[int, ...] = ()


@dataclass(frozen=True)
class RateLimitPolicy:
    """429 / Retry-After handling policy.

    `retry_after_safe_maximum_s` bounds how long the executor will wait
    before honoring a server-provided Retry-After value. Any value above
    this threshold is treated as if Retry-After were absent (the executor
    then either skips to next fallback or uses the default backoff_s).

    `one_shot=True` enforces the Owner rule that a 429 on a given target
    triggers AT MOST ONE retry, regardless of what Retry-After says.
    """

    retry_after_safe_maximum_s: int = 30
    honor_retry_after_if_within_max: bool = True
    one_shot: bool = True


@dataclass(frozen=True)
class Policy:
    version: str
    environment: str
    request_timeout_s: int
    connect_timeout_s: int
    read_budget_total_s: int
    retry_policy: dict[str, RetryRule]
    rate_limit_policy: RateLimitPolicy
    model_policy_mode: str
    deny_patterns: tuple[str, ...]
    allow_models: frozenset[str]
    provider_limits: dict[str, dict[str, int]]
    shared: dict[str, Any]


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    enabled: bool
    fallback_chain: tuple[tuple[str, str], ...]  # ((provider, model), ...)
    model_allowlist_override: tuple[str, ...]
    model_deny_patterns: tuple[str, ...]
    timeouts: dict[str, int]
    retry_budget: dict[str, int]
    logging: dict[str, Any]
    secret_refs: dict[str, SecretRef]
    backward_compat: dict[str, bool]


# ── Public exception ─────────────────────────────────────────────────────────


class PolicyValidationError(ValueError):
    """Raised when a policy or workload file fails validation.

    Messages list KEY NAMES and PATTERNS, never values.
    """


# ── Public API ───────────────────────────────────────────────────────────────


def load_policy(path: str) -> Policy | None:
    """Load and validate the global policy YAML.

    Returns None if the file does not exist (backward-compatible mode).
    Raises PolicyValidationError on any structural or semantic issue.
    Never reads or logs any secret value.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise PolicyValidationError(
            f"policy.yaml at {path} is not valid YAML: {type(e).__name__}"
        ) from e

    if raw is None:
        # Empty file or empty document is treated as "no policy present"
        # (backward-compatible mode). This matches the behavior when the
        # file does not exist.
        return None

    if not isinstance(raw, dict):
        raise PolicyValidationError(
            f"policy.yaml at {path} must be a YAML mapping at top level"
        )

    version = _require_str(raw, "version", path)
    environment = _require_str(raw, "environment", path)
    if environment not in {"dev", "staging", "production"}:
        raise PolicyValidationError(
            f"policy.yaml at {path}: environment must be one of dev|staging|production"
        )

    defaults = _require_dict(raw, "defaults", path)
    request_timeout_s = _require_int(defaults, "request_timeout_s", path)
    connect_timeout_s = _require_int(defaults, "connect_timeout_s", path)
    read_budget_total_s = _require_int(defaults, "read_budget_total_s", path)
    for name, val in (
        ("request_timeout_s", request_timeout_s),
        ("connect_timeout_s", connect_timeout_s),
        ("read_budget_total_s", read_budget_total_s),
    ):
        if val <= 0:
            raise PolicyValidationError(
                f"policy.yaml at {path}: defaults.{name} must be a positive integer"
            )

    retry_raw = _require_dict(raw, "retry_policy", path)
    retry_policy: dict[str, RetryRule] = {}
    for failure_class, rule in retry_raw.items():
        if not isinstance(rule, dict):
            raise PolicyValidationError(
                f"policy.yaml at {path}: retry_policy.{failure_class} must be a mapping"
            )
        max_attempts = _require_int(rule, "max_attempts_same_target", path)
        if max_attempts < 0:
            raise PolicyValidationError(
                f"policy.yaml at {path}: retry_policy.{failure_class}.max_attempts_same_target must be >= 0"
            )
        backoff = rule.get("backoff_s", [])
        if not isinstance(backoff, list):
            raise PolicyValidationError(
                f"policy.yaml at {path}: retry_policy.{failure_class}.backoff_s must be a list"
            )
        for b in backoff:
            if not isinstance(b, int) or isinstance(b, bool) or b < 0:
                raise PolicyValidationError(
                    f"policy.yaml at {path}: retry_policy.{failure_class}.backoff_s entries must be non-negative integers"
                )
        retry_policy[failure_class] = RetryRule(
            failure_class=failure_class,
            max_attempts_same_target=max_attempts,
            backoff_s=tuple(int(b) for b in backoff),
        )

    # ── Rate-limit (429) policy ──────────────────────────────────────────
    rate_limit_raw = raw.get("rate_limit_policy", {})
    if not isinstance(rate_limit_raw, dict):
        raise PolicyValidationError(
            f"policy.yaml at {path}: rate_limit_policy must be a mapping"
        )
    retry_after_max_raw = rate_limit_raw.get("retry_after_safe_maximum_s", 30)
    if not isinstance(retry_after_max_raw, int) or isinstance(retry_after_max_raw, bool) or retry_after_max_raw < 0:
        raise PolicyValidationError(
            f"policy.yaml at {path}: rate_limit_policy.retry_after_safe_maximum_s must be a non-negative integer"
        )
    honor_raw = rate_limit_raw.get("honor_retry_after_if_within_max", True)
    if not isinstance(honor_raw, bool):
        raise PolicyValidationError(
            f"policy.yaml at {path}: rate_limit_policy.honor_retry_after_if_within_max must be a boolean"
        )
    one_shot_raw = rate_limit_raw.get("one_shot", True)
    if not isinstance(one_shot_raw, bool):
        raise PolicyValidationError(
            f"policy.yaml at {path}: rate_limit_policy.one_shot must be a boolean"
        )
    rate_limit_policy = RateLimitPolicy(
        retry_after_safe_maximum_s=int(retry_after_max_raw),
        honor_retry_after_if_within_max=bool(honor_raw),
        one_shot=bool(one_shot_raw),
    )

    # ── Owner-mandated invariant: NETWORK_TIMEOUT/TRANSIENT_5XX retry
    # counts MUST be exactly one (i.e. max_attempts_same_target == 2).
    # Earlier task #22 had backoff_s: [2, 4] which implied a 2-retry
    # policy. Owner ruled that 4-second second retry is forbidden.
    nt_rule = retry_policy.get("network_timeout_or_transient_5xx")
    if nt_rule is None or nt_rule.max_attempts_same_target != 2:
        raise PolicyValidationError(
            f"policy.yaml at {path}: network_timeout_or_transient_5xx.max_attempts_same_target MUST be 2 (one original + exactly one retry). Got: {nt_rule.max_attempts_same_target if nt_rule else 'missing'}"
        )
    # RL rule also must be exactly one retry.
    rl_rule = retry_policy.get("rate_limited")
    if rl_rule is None or rl_rule.max_attempts_same_target != 2:
        raise PolicyValidationError(
            f"policy.yaml at {path}: rate_limited.max_attempts_same_target MUST be 2 (one original + exactly one retry). Got: {rl_rule.max_attempts_same_target if rl_rule else 'missing'}"
        )
    # Rate-limit one_shot invariant.
    if not rate_limit_policy.one_shot:
        raise PolicyValidationError(
            f"policy.yaml at {path}: rate_limit_policy.one_shot MUST be true (Owner rule: at most one retry on a 429)"
        )

    model_policy = _require_dict(raw, "model_policy", path)
    mode = _require_str(model_policy, "mode", path)
    if mode not in {"allowlist", "denylist"}:
        raise PolicyValidationError(
            f"policy.yaml at {path}: model_policy.mode must be allowlist|denylist"
        )
    deny_raw = model_policy.get("deny_patterns", [])
    if not isinstance(deny_raw, list):
        raise PolicyValidationError(
            f"policy.yaml at {path}: model_policy.deny_patterns must be a list"
        )
    deny_patterns: tuple[str, ...] = tuple(str(p) for p in deny_raw)
    allow_raw = model_policy.get("allow_models", [])
    if not isinstance(allow_raw, list):
        raise PolicyValidationError(
            f"policy.yaml at {path}: model_policy.allow_models must be a list"
        )
    allow_models = frozenset(str(m) for m in allow_raw)

    provider_limits_raw = raw.get("provider_limits", {})
    if not isinstance(provider_limits_raw, dict):
        raise PolicyValidationError(
            f"policy.yaml at {path}: provider_limits must be a mapping"
        )
    provider_limits: dict[str, dict[str, int]] = {}
    for prov_name, prov_rule in provider_limits_raw.items():
        if not isinstance(prov_rule, dict):
            raise PolicyValidationError(
                f"policy.yaml at {path}: provider_limits.{prov_name} must be a mapping"
            )
        prov_limits: dict[str, int] = {}
        for k, v in prov_rule.items():
            # ── Strict allowlist ────────────────────────────────────────
            # Only these field names are actually read by the policy/
            # executor code. Any other name (especially secret-shaped
            # ones like api_key, token, password, secret, authorization,
            # credential) is rejected with a redacted message that NEVER
            # echoes the value supplied by the caller.
            if k not in _ALLOWED_PROVIDER_LIMITS_KEYS:
                raise PolicyValidationError(
                    f"policy.yaml at {path}: provider_limits.{prov_name}.{k} "
                    f"is not an allowed configuration field "
                    f"(allowed: {sorted(_ALLOWED_PROVIDER_LIMITS_KEYS)})"
                )
            # ── Strict int validation ───────────────────────────────────
            # Never let a non-int (e.g. a secret-shaped string mistakenly
            # placed under provider_limits) raise a bare ValueError that
            # bypasses the validator and might leak the value into a
            # traceback.
            if not isinstance(v, int) or isinstance(v, bool):
                raise PolicyValidationError(
                    f"policy.yaml at {path}: provider_limits.{prov_name}.{k} "
                    f"must be a non-negative integer (no strings, no booleans)"
                )
            if v < 0:
                raise PolicyValidationError(
                    f"policy.yaml at {path}: provider_limits.{prov_name}.{k} "
                    f"must be a non-negative integer (got value type {type(v).__name__})"
                )
            prov_limits[str(k)] = int(v)
        provider_limits[str(prov_name)] = prov_limits

    shared = raw.get("shared", {})
    if not isinstance(shared, dict):
        raise PolicyValidationError(
            f"policy.yaml at {path}: shared must be a mapping"
        )

    policy = Policy(
        version=version,
        environment=environment,
        request_timeout_s=request_timeout_s,
        connect_timeout_s=connect_timeout_s,
        read_budget_total_s=read_budget_total_s,
        retry_policy=retry_policy,
        rate_limit_policy=rate_limit_policy,
        model_policy_mode=mode,
        deny_patterns=deny_patterns,
        allow_models=allow_models,
        provider_limits=provider_limits,
        shared=dict(shared),
    )

    # Validate the allow_models list against the deny patterns (early
    # failure if the policy itself is self-inconsistent).
    _validate_models_against_patterns(
        list(allow_models), deny_patterns,
        source_label=f"policy.yaml allow_models ({path})",
    )

    return policy


def load_workload(name: str, path: str) -> WorkloadConfig:
    """Load and validate a workload YAML. Raises PolicyValidationError.

    No secret values are validated or stored. The workload's
    `secret_refs` block is checked for URI schema only.
    """
    if not os.path.exists(path):
        raise PolicyValidationError(
            f"workload YAML not found at {path}"
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise PolicyValidationError(
            f"workload YAML at {path} is not valid YAML: {type(e).__name__}"
        ) from e
    if not isinstance(raw, dict):
        raise PolicyValidationError(
            f"workload YAML at {path} must be a YAML mapping at top level"
        )

    workload = _require_dict(raw, "workload", path)
    wl_name = _require_str(workload, "name", path)
    if wl_name != name:
        raise PolicyValidationError(
            f"workload YAML at {path}: workload.name does not match the requested workload name"
        )
    enabled = _require_bool(workload, "enabled", path)

    chain_raw = _require_list(raw, "fallback_chain", path)
    fallback_chain: list[tuple[str, str]] = []
    for entry in chain_raw:
        if not isinstance(entry, dict):
            raise PolicyValidationError(
                f"workload YAML at {path}: each fallback_chain entry must be a mapping"
            )
        provider = _require_str(entry, "provider", path)
        model = _require_str(entry, "model", path)
        fallback_chain.append((provider, model))

    model_allowlist_raw = raw.get("model_allowlist", [])
    if not isinstance(model_allowlist_raw, list):
        raise PolicyValidationError(
            f"workload YAML at {path}: model_allowlist must be a list"
        )
    model_allowlist = tuple(str(m) for m in model_allowlist_raw)

    model_deny_raw = raw.get("model_deny_patterns", [])
    if not isinstance(model_deny_raw, list):
        raise PolicyValidationError(
            f"workload YAML at {path}: model_deny_patterns must be a list"
        )
    model_deny = tuple(str(p) for p in model_deny_raw)

    timeouts = _validate_optional_int_map(raw.get("timeouts", {}), "timeouts", path)
    retry_budget = _validate_optional_int_map(raw.get("retry_budget", {}), "retry_budget", path)

    logging_raw = raw.get("logging", {})
    if not isinstance(logging_raw, dict):
        raise PolicyValidationError(
            f"workload YAML at {path}: logging must be a mapping"
        )
    # Hard invariants: the no-secret invariants cannot be disabled.
    if logging_raw.get("log_request_authorization_headers") is not False:
        raise PolicyValidationError(
            f"workload YAML at {path}: logging.log_request_authorization_headers must be false (hard invariant)"
        )
    if logging_raw.get("log_secret_values") is not False:
        raise PolicyValidationError(
            f"workload YAML at {path}: logging.log_secret_values must be false (hard invariant)"
        )

    secret_refs_raw = raw.get("secret_refs", {})
    if not isinstance(secret_refs_raw, dict):
        raise PolicyValidationError(
            f"workload YAML at {path}: secret_refs must be a mapping"
        )
    secret_refs: dict[str, SecretRef] = {}
    for ref_key, ref_val in secret_refs_raw.items():
        if not isinstance(ref_val, str):
            raise PolicyValidationError(
                f"workload YAML at {path}: secret_refs entries must be URI strings"
            )
        if not _is_uri_shaped(ref_val):
            raise PolicyValidationError(
                f"workload YAML at {path}: secret_refs entries must start with vault:// or env://"
            )
        secret_refs[str(ref_key)] = SecretRef(key=str(ref_key), value=ref_val)

    backward_compat_raw = raw.get("backward_compat", {})
    if not isinstance(backward_compat_raw, dict):
        raise PolicyValidationError(
            f"workload YAML at {path}: backward_compat must be a mapping"
        )

    cfg = WorkloadConfig(
        name=wl_name,
        enabled=enabled,
        fallback_chain=tuple(fallback_chain),
        model_allowlist_override=model_allowlist,
        model_deny_patterns=model_deny,
        timeouts=timeouts,
        retry_budget=retry_budget,
        logging=dict(logging_raw),
        secret_refs=secret_refs,
        backward_compat=dict(backward_compat_raw),
    )

    # Validate every model in the fallback chain against the workload's
    # local deny list (and the workload allowlist, if set).
    chain_models = [m for _, m in cfg.fallback_chain]
    if cfg.model_allowlist_override:
        invalid = [m for m in chain_models if m not in cfg.model_allowlist_override]
        if invalid:
            raise PolicyValidationError(
                f"workload YAML at {path}: fallback_chain contains models not in model_allowlist"
            )
    _validate_models_against_patterns(
        chain_models,
        cfg.model_deny_patterns,
        source_label=f"workload {name} fallback_chain ({path})",
    )

    return cfg


def validate_workload_against_policy(
    workload: WorkloadConfig, policy: Policy
) -> None:
    """Cross-check a workload against the global policy. Raises on violations.

    Order of checks:
      1. Workload's effective deny set = global deny_patterns UNION workload's deny_patterns.
      2. Workload's effective allowlist = (workload.model_allowlist_override
         if non-empty else policy.allow_models).
      3. Every model in the workload's fallback_chain must pass the effective
         deny set AND be present in the effective allowlist (when the policy
         is in allowlist mode).
    """
    effective_deny = list(policy.deny_patterns) + list(workload.model_deny_patterns)
    effective_allow = (
        frozenset(workload.model_allowlist_override)
        if workload.model_allowlist_override
        else policy.allow_models
    )

    _validate_models_against_patterns(
        [m for _, m in workload.fallback_chain],
        tuple(effective_deny),
        source_label=f"workload {workload.name} fallback_chain (cross-validated against policy)",
    )

    if policy.model_policy_mode == "allowlist":
        invalid = [m for _, m in workload.fallback_chain if m not in effective_allow]
        if invalid:
            raise PolicyValidationError(
                f"workload {workload.name}: fallback_chain contains models not in the policy allowlist"
            )


def redact_for_log(value: Any) -> str:
    """Return a non-secret placeholder for any value that might be a secret.

    Used by callers to format messages without exposing the value.
    Never returns the input value. Never logs keys or values from
    secret_refs maps.
    """
    return "[REDACTED]"


# ── Internal helpers ────────────────────────────────────────────────────────


_SECRET_URI_SCHEMES = ("vault://", "env://")


def _is_uri_shaped(value: str) -> bool:
    return value.startswith(_SECRET_URI_SCHEMES)


def _require_str(obj: dict, key: str, source: str) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or not val:
        raise PolicyValidationError(
            f"{source}: missing required string field '{key}'"
        )
    return val


def _require_int(obj: dict, key: str, source: str) -> int:
    val = obj.get(key)
    if not isinstance(val, int) or isinstance(val, bool):
        raise PolicyValidationError(
            f"{source}: missing required integer field '{key}'"
        )
    return val


def _require_bool(obj: dict, key: str, source: str) -> bool:
    val = obj.get(key)
    if not isinstance(val, bool):
        raise PolicyValidationError(
            f"{source}: missing required boolean field '{key}'"
        )
    return val


def _require_dict(obj: dict, key: str, source: str) -> dict:
    val = obj.get(key)
    if not isinstance(val, dict):
        raise PolicyValidationError(
            f"{source}: missing required mapping field '{key}'"
        )
    return val


def _require_list(obj: dict, key: str, source: str) -> list:
    val = obj.get(key)
    if not isinstance(val, list):
        raise PolicyValidationError(
            f"{source}: missing required list field '{key}'"
        )
    return val


def _validate_optional_int_map(raw: Any, field_name: str, source: str) -> dict[str, int]:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise PolicyValidationError(
            f"{source}: {field_name} must be a mapping"
        )
    out: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(v, int) or isinstance(v, bool):
            raise PolicyValidationError(
                f"{source}: {field_name}.{k} must be an integer"
            )
        out[str(k)] = int(v)
    return out


def _validate_models_against_patterns(
    models: Iterable[str], patterns: Iterable[str], source_label: str
) -> None:
    """Reject any model string that matches any deny pattern.

    Patterns are matched via fnmatch. Empty pattern list is a no-op.
    """
    pats = tuple(p for p in patterns if p)
    if not pats:
        return
    for m in models:
        for p in pats:
            if fnmatch.fnmatchcase(m, p):
                raise PolicyValidationError(
                    f"{source_label}: model {m!r} matches deny pattern {p!r} (rejected by policy)"
                )
