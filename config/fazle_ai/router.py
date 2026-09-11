"""Deterministic Phase 3 routing executor.

The engine is transport-agnostic: adapters supply one callable that invokes a
candidate route.  It returns exactly one final content value and never performs
delivery, which keeps provider fallback inside one logical reply lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

from config.fazle_ai.contracts import (
    AttemptOutcome,
    FailureAction,
    FailureClass,
    InferenceRequest,
    RouteCandidate,
    build_attempt_audit,
    failure_action,
    resolve_eligible_routes,
)


@dataclass(frozen=True)
class ProviderResult:
    ok: bool
    content: str = ""
    failure_class: FailureClass | None = None
    retry_after_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    safe_detail: str = ""

    @classmethod
    def success(
        cls, content: str, *, input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> "ProviderResult":
        if not content or not content.strip():
            raise ValueError("successful provider result requires non-empty content")
        return cls(True, content.strip(), input_tokens=input_tokens, output_tokens=output_tokens)

    @classmethod
    def failure(
        cls, failure_class: FailureClass, *, retry_after_s: float | None = None,
        safe_detail: str = "",
    ) -> "ProviderResult":
        return cls(False, failure_class=failure_class, retry_after_s=retry_after_s, safe_detail=safe_detail[:500])

    def __post_init__(self) -> None:
        if self.ok and self.failure_class is not None:
            raise ValueError("successful provider result cannot contain a failure class")
        if not self.ok and self.failure_class is None:
            raise ValueError("failed provider result requires a failure class")


@dataclass(frozen=True)
class RoutingResult:
    content: str
    selected_route_id: str
    attempt_audits: tuple[dict[str, object], ...]

    @property
    def final_route_id(self) -> str:
        return self.selected_route_id


class RoutingExhausted(RuntimeError):
    def __init__(
        self, message: str, *, failure_class: FailureClass,
        attempt_audits: Iterable[dict[str, object]],
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.attempt_audits = tuple(attempt_audits)
        self.content = ""


class RoutingEngine:
    """Execute an ordered, compatibility-filtered route policy."""

    def __init__(
        self,
        routes: Iterable[RouteCandidate],
        *,
        sleep: Callable[[float], None] = time.sleep,
        transient_backoff_s: float = 2.0,
        retry_after_safe_maximum_s: float = 30.0,
        environment: str = "development",
        available_credential_refs: frozenset[str] | None = None,
    ) -> None:
        self.routes = tuple(routes)
        self.sleep = sleep
        self.transient_backoff_s = transient_backoff_s
        self.retry_after_safe_maximum_s = retry_after_safe_maximum_s
        self.environment = environment
        self.available_credential_refs = available_credential_refs

    def execute(
        self,
        request: InferenceRequest,
        call: Callable[[RouteCandidate, int], ProviderResult],
    ) -> RoutingResult:
        routes = resolve_eligible_routes(
            request, self.routes, environment=self.environment,
            available_credential_refs=self.available_credential_refs,
        )
        audits: list[dict[str, object]] = []
        last_failure = FailureClass.UNKNOWN

        for fallback_number, route in enumerate(routes):
            attempt_number = 0
            while True:
                attempt_number += 1
                started = time.monotonic()
                try:
                    result = call(route, attempt_number)
                    if not isinstance(result, ProviderResult):
                        result = ProviderResult.failure(FailureClass.INTERNAL_APPLICATION_DEFECT)
                except Exception:
                    result = ProviderResult.failure(FailureClass.INTERNAL_APPLICATION_DEFECT)
                latency_ms = max(0, int((time.monotonic() - started) * 1000))

                if result.ok:
                    audits.append(build_attempt_audit(
                        request=request, route=route,
                        fallback_number=fallback_number,
                        attempt_number=attempt_number, latency_ms=latency_ms,
                        outcome=AttemptOutcome.SUCCESS, failure_class=None,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                    ))
                    return RoutingResult(result.content, route.route_id, tuple(audits))

                last_failure = result.failure_class or FailureClass.UNKNOWN_UNCLASSIFIED
                action = failure_action(last_failure, retry_eligible=request.retry_eligible)
                audits.append(build_attempt_audit(
                    request=request, route=route,
                    fallback_number=fallback_number,
                    attempt_number=attempt_number, latency_ms=latency_ms,
                    outcome=AttemptOutcome.FAILURE, failure_class=last_failure,
                    fallback_decision=action,
                    diagnostic=result.safe_detail or None,
                ))

                if action in {FailureAction.FAIL_CLOSED, FailureAction.INTERNAL_ERROR}:
                    raise RoutingExhausted(
                        f"routing stopped on {last_failure.value}",
                        failure_class=last_failure, attempt_audits=audits,
                    )
                if action is FailureAction.RETRY_THEN_NEXT and attempt_number < route.max_attempts_same_route:
                    backoff = route.transient_backoff_s
                    if backoff:
                        self.sleep(backoff)
                    continue
                if action is FailureAction.BOUNDED_RETRY_THEN_NEXT and attempt_number < route.max_attempts_same_route:
                    retry_after = result.retry_after_s
                    maximum = min(self.retry_after_safe_maximum_s, route.retry_after_safe_maximum_s)
                    if retry_after is not None and 0 <= retry_after <= maximum:
                        if retry_after:
                            self.sleep(retry_after)
                        continue
                break

        raise RoutingExhausted(
            "all compatible routes exhausted",
            failure_class=last_failure, attempt_audits=audits,
        )
