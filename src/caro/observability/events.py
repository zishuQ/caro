"""Optional Langfuse observability for Caro.

The integration is deliberately inert unless explicitly enabled. Langfuse is an
optional dependency, so this module keeps all SDK imports behind the factory and
turns initialization or export failures into local diagnostics instead of agent
failures.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Literal, Protocol, TypeAlias, cast

logger = logging.getLogger(__name__)

LANGFUSE_ENABLED_ENV = "CARO_LANGFUSE_ENABLED"
"""Explicit opt-in controlling the optional Langfuse integration."""

_LEGACY_LANGFUSE_ENABLED_ENV = "DEEPAGENTS_CODE_LANGFUSE_ENABLED"


def strip_langfuse_environment(env: dict[str, str]) -> None:
    """Remove Langfuse control, credential, and endpoint variables in place.

    The server process retains these variables so the optional callback can
    initialize. Agent-executed shell commands receive a curated environment and
    must not be able to read Langfuse credentials or redirect telemetry.

    Args:
        env: Shell environment mapping to modify in place.
    """
    for key in tuple(env):
        if key in {LANGFUSE_ENABLED_ENV, _LEGACY_LANGFUSE_ENABLED_ENV} or key.startswith("LANGFUSE_"):
            env.pop(key, None)


REDACTED_VALUE = "[REDACTED]"
"""Replacement stored in observations instead of sensitive values."""

_REDACTION_FAILED_VALUE = "[REDACTED:UNAVAILABLE]"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_SENSITIVE_KEYS = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "awssecretaccesskey",
        "clientsecret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "privatekey",
        "proxyauthorization",
        "refreshtoken",
        "secret",
        "secretkey",
        "setcookie",
        "token",
    }
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_AUTHORIZATION_VALUE = re.compile(r"(?i)\b(bearer|basic)(\s+)[^\s,;]+")

ApprovalDecision: TypeAlias = Literal[
    "requested",
    "approved",
    "auto_approved",
    "rejected",
    "blocked",
    "cancelled",
]
ObservationLevel: TypeAlias = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalObservationEvent:
    """Structured record of one tool-approval decision.

    Args:
        thread_id: LangGraph thread that requested approval.
        tool_name: Tool covered by the decision.
        decision: Terminal decision or the initial request marker.
        mode: Approval mode active when the decision was made.
        run_id: Caro run identifier, when run-state management is active.
        risk_level: Registry-assigned risk level, when available.
        reason: Sanitized human or policy reason.
        latency_ms: Time spent waiting for or computing the decision.
        metadata: Additional structured fields. Values are copied and redacted
            before export.
        trace_id: Langfuse trace identifier for cross-context correlation.
        parent_span_id: Optional Langfuse parent observation identifier.
    """

    thread_id: str
    tool_name: str
    decision: ApprovalDecision
    mode: str
    run_id: str | None = None
    risk_level: str | None = None
    reason: str | None = None
    latency_ms: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    trace_id: str | None = None
    parent_span_id: str | None = None

    def __post_init__(self) -> None:
        """Validate identifiers and duration at the event boundary."""
        _require_nonblank("thread_id", self.thread_id)
        _require_nonblank("tool_name", self.tool_name)
        _require_nonblank("mode", self.mode)
        _require_nonnegative("latency_ms", self.latency_ms)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunObservationEvent:
    """Structured record of a Caro run lifecycle or model-stage transition.

    Args:
        thread_id: Owning LangGraph thread identifier.
        run_id: Caro run identifier.
        phase: Current lifecycle phase, such as `planning` or `testing`.
        previous_phase: Previous lifecycle phase, when this is a transition.
        stage: Model-routing stage (`planning`, `coding`, `review`, or `summary`).
        attempt: One-based run or routed-model attempt number.
        model: Effective model identifier, when known.
        provider: Effective model provider, when known.
        input_tokens: Input tokens attributed to this event.
        output_tokens: Output tokens attributed to this event.
        latency_ms: Elapsed operation time.
        error_type: Stable exception type name for failed operations.
        error_message: Sanitized failure detail.
        metadata: Additional structured fields. Values are copied and redacted
            before export.
        trace_id: Langfuse trace identifier for cross-context correlation.
        parent_span_id: Optional Langfuse parent observation identifier.
    """

    thread_id: str
    run_id: str
    phase: str
    previous_phase: str | None = None
    stage: str | None = None
    attempt: int = 1
    model: str | None = None
    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    trace_id: str | None = None
    parent_span_id: str | None = None

    def __post_init__(self) -> None:
        """Validate identifiers, counters, and duration at the event boundary.

        Raises:
            ValueError: If a required field is invalid or a counter is negative.
        """
        _require_nonblank("thread_id", self.thread_id)
        _require_nonblank("run_id", self.run_id)
        _require_nonblank("phase", self.phase)
        if self.attempt <= 0:
            msg = "attempt must be positive"
            raise ValueError(msg)
        _require_nonnegative("input_tokens", self.input_tokens)
        _require_nonnegative("output_tokens", self.output_tokens)
        _require_nonnegative("latency_ms", self.latency_ms)


class ObservationEventSink(Protocol):
    """Fail-open destination for approval and run observations."""

    def record_approval(self, event: ApprovalObservationEvent) -> None:
        """Record a tool-approval event without affecting agent execution."""

    def record_run(self, event: RunObservationEvent) -> None:
        """Record a run lifecycle event without affecting agent execution."""

    def flush(self) -> None:
        """Best-effort flush of buffered observations."""


class NullObservationEventSink:
    """No-op sink used while Langfuse is disabled or unavailable."""

    @staticmethod
    def record_approval(event: ApprovalObservationEvent) -> None:
        """Discard an approval event."""
        del event

    @staticmethod
    def record_run(event: RunObservationEvent) -> None:
        """Discard a run event."""
        del event

    def flush(self) -> None:
        """Return immediately because no events are buffered."""


class _LangfuseClient(Protocol):
    """Narrow SDK surface used by `LangfuseObservationEventSink`."""

    def create_event(self, **kwargs: object) -> object:
        """Create an instantaneous Langfuse event observation."""

    def flush(self) -> None:
        """Flush buffered observations."""


class _ClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _LangfuseClient:
        """Construct and register the process-default Langfuse client."""


class _CallbackFactory(Protocol):
    def __call__(self) -> object:
        """Construct the process-default LangChain callback handler."""


class _LangfuseModule(Protocol):
    Langfuse: _ClientFactory


class _LangfuseLangchainModule(Protocol):
    CallbackHandler: _CallbackFactory


class LangfuseObservationEventSink:
    """Best-effort structured event exporter backed by Langfuse."""

    def __init__(self, client: _LangfuseClient) -> None:
        """Initialize the sink with an already configured Langfuse client.

        Args:
            client: Langfuse client created by the lazy integration factory.
        """
        self._client = client
        self._export_failure_reported = False

    def record_approval(self, event: ApprovalObservationEvent) -> None:
        """Export one approval event as `caro.approval`.

        Args:
            event: Structured approval decision to export.
        """
        payload: dict[str, object] = {
            "event_kind": "approval",
            "thread_id": event.thread_id,
            "run_id": event.run_id,
            "tool_name": event.tool_name,
            "decision": event.decision,
            "mode": event.mode,
            "risk_level": event.risk_level,
            "reason": event.reason,
            "latency_ms": event.latency_ms,
            "metadata": dict(event.metadata),
        }
        level: ObservationLevel = (
            "WARNING"
            if event.decision in {"rejected", "blocked", "cancelled"}
            else "DEFAULT"
        )
        self._record(
            name="caro.approval",
            payload=payload,
            level=level,
            trace_id=event.trace_id,
            parent_span_id=event.parent_span_id,
        )

    def record_run(self, event: RunObservationEvent) -> None:
        """Export one run transition as `caro.run`.

        Args:
            event: Structured run or model-stage transition to export.
        """
        payload: dict[str, object] = {
            "event_kind": "run",
            "thread_id": event.thread_id,
            "run_id": event.run_id,
            "phase": event.phase,
            "previous_phase": event.previous_phase,
            "stage": event.stage,
            "attempt": event.attempt,
            "model": event.model,
            "provider": event.provider,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "latency_ms": event.latency_ms,
            "error_type": event.error_type,
            "error_message": event.error_message,
            "metadata": dict(event.metadata),
        }
        level: ObservationLevel = "DEFAULT"
        if event.phase == "failed" or event.error_type is not None:
            level = "ERROR"
        elif event.phase == "cancelled":
            level = "WARNING"
        self._record(
            name="caro.run",
            payload=payload,
            level=level,
            trace_id=event.trace_id,
            parent_span_id=event.parent_span_id,
        )

    def flush(self) -> None:
        """Flush buffered events without surfacing exporter failures."""
        try:
            self._client.flush()
        except Exception as exc:  # Telemetry must not fail a run.
            self._report_export_failure("flush", exc)

    def _record(
        self,
        *,
        name: str,
        payload: Mapping[str, object],
        level: ObservationLevel,
        trace_id: str | None,
        parent_span_id: str | None,
    ) -> None:
        safe_payload = redact_observation_payload(payload)
        trace_context: dict[str, str] | None = None
        if trace_id:
            trace_context = {"trace_id": trace_id}
            if parent_span_id:
                trace_context["parent_span_id"] = parent_span_id
        try:
            self._client.create_event(
                name=name,
                metadata=safe_payload,
                level=level,
                trace_context=trace_context,
            )
        except Exception as exc:  # Telemetry must not fail a run.
            self._report_export_failure(name, exc)

    def _report_export_failure(self, operation: str, error: Exception) -> None:
        if self._export_failure_reported:
            return
        self._export_failure_reported = True
        logger.warning(
            "Langfuse observation export failed during %s (%s); "
            "agent execution will continue",
            operation,
            type(error).__name__,
        )


@dataclass(frozen=True, slots=True)
class LangfuseObservability:
    """Optional Langfuse callback and structured-event bundle."""

    callback_handler: object | None
    event_sink: ObservationEventSink

    @property
    def enabled(self) -> bool:
        """Whether the Langfuse SDK initialized successfully."""
        return self.callback_handler is not None

    @property
    def callbacks(self) -> tuple[object, ...]:
        """Callbacks ready to merge into a LangGraph runnable config."""
        if self.callback_handler is None:
            return ()
        return (self.callback_handler,)


def is_langfuse_enabled(*, env: Mapping[str, str] | None = None) -> bool:
    """Return whether Caro explicitly enables Langfuse.

    Args:
        env: Environment mapping to inspect. Defaults to `os.environ`.

    Returns:
        `True` only for an explicit truthy opt-in. Missing and invalid values are
        disabled, preserving the integration's default-off contract.
    """
    values = os.environ if env is None else env
    return any(
        values.get(name, "").strip().lower() in _TRUTHY
        for name in (LANGFUSE_ENABLED_ENV, _LEGACY_LANGFUSE_ENABLED_ENV)
    )


def create_langfuse_observability(
    *,
    enabled: bool | None = None,
) -> LangfuseObservability:
    """Lazily construct Langfuse callbacks and event export.

    Args:
        enabled: Explicit override. When `None`, read
        `CARO_LANGFUSE_ENABLED`; the default is disabled.

    Returns:
        An enabled integration bundle, or a no-op bundle when disabled, the
        optional dependency is missing, or SDK initialization fails.
    """
    active = is_langfuse_enabled() if enabled is None else enabled
    if not active:
        return _disabled_observability()

    try:
        langfuse_module = cast(
            "_LangfuseModule",
            import_module("langfuse"),
        )
        langchain_module = cast(
            "_LangfuseLangchainModule",
            import_module("langfuse.langchain"),
        )
        client = langfuse_module.Langfuse(mask=_langfuse_mask)
        callback_handler = langchain_module.CallbackHandler()
    except Exception:  # Optional telemetry must fail open.
        logger.warning(
            "Langfuse initialization failed; observability is disabled",
            exc_info=True,
        )
        return _disabled_observability()

    return LangfuseObservability(
        callback_handler=callback_handler,
        event_sink=LangfuseObservationEventSink(client),
    )


def redact_observation_payload(value: object) -> object:
    """Copy and recursively redact a value before observation export.

    Args:
        value: Arbitrary event metadata or payload value.

    Returns:
        A detached copy with sensitive mapping values and common inline secret
        forms replaced. If copying fails, returns a fail-closed placeholder rather
        than exporting the original object.
    """
    try:
        cloned = copy.deepcopy(value)
    except Exception:  # Never fall back to a possibly secret object.
        logger.warning("Could not copy observation payload; redacting it entirely")
        return _REDACTION_FAILED_VALUE
    return _redact_value(cloned, seen=set())


def _langfuse_mask(*, data: object) -> object:
    """Adapt recursive redaction to Langfuse's keyword-only mask contract.

    Returns:
        Detached and recursively redacted observation data.
    """
    return redact_observation_payload(data)


def _disabled_observability() -> LangfuseObservability:
    return LangfuseObservability(
        callback_handler=None,
        event_sink=NullObservationEventSink(),
    )


def _require_nonblank(name: str, value: str) -> None:
    if not value.strip():
        msg = f"{name} must be non-empty"
        raise ValueError(msg)


def _require_nonnegative(name: str, value: int | None) -> None:
    if value is not None and value < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _is_sensitive_key(key: object) -> bool:
    normalized = _normalized_key(key)
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("apikey", "accesstoken", "refreshtoken", "secretkey", "privatekey")
    )


def _redact_text(value: str) -> str:
    value = _SECRET_ASSIGNMENT.sub(r"\1\2" + REDACTED_VALUE, value)
    return _AUTHORIZATION_VALUE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED_VALUE}",
        value,
    )


def _redact_value(value: object, *, seen: set[int]) -> object:
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float, bytes)):
        return value

    value_id = id(value)
    if value_id in seen:
        return _REDACTION_FAILED_VALUE

    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            return {
                key: (
                    REDACTED_VALUE
                    if _is_sensitive_key(key)
                    else _redact_value(item, seen=seen)
                )
                for key, item in value.items()
            }
        finally:
            seen.remove(value_id)
    if isinstance(value, list):
        seen.add(value_id)
        try:
            return [_redact_value(item, seen=seen) for item in value]
        finally:
            seen.remove(value_id)
    if isinstance(value, tuple):
        seen.add(value_id)
        try:
            return tuple(_redact_value(item, seen=seen) for item in value)
        finally:
            seen.remove(value_id)
    if isinstance(value, (set, frozenset)):
        seen.add(value_id)
        try:
            return type(value)(_redact_value(item, seen=seen) for item in value)
        finally:
            seen.remove(value_id)
    return value
