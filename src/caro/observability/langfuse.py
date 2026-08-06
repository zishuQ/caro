"""Langfuse v4 callback integration for Caro LangGraph runs."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler

from caro.config import CaroSettings
from caro.errors import CaroError
from caro.observability.events import (
    LangfuseObservationEventSink,
    NullObservationEventSink,
    ObservationEventSink,
    _langfuse_mask,
    _LangfuseClient,
)
from caro.state.runs import RunRecord

HandlerFactory = Callable[[], BaseCallbackHandler]
FlushCallback = Callable[[], None]
ObservationSinkFactory = Callable[[], ObservationEventSink]
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TraceContext:
    """Callbacks and invocation metadata associated with one Caro run."""

    callbacks: list[BaseCallbackHandler] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    observation_sink: ObservationEventSink = field(default_factory=NullObservationEventSink)
    _flush: FlushCallback | None = None

    def close(self) -> None:
        """Flush short-lived CLI telemetry without masking application errors."""

        try:
            if self._flush is not None:
                self._flush()
        except Exception as error:  # Telemetry must never fail a run.
            logger.warning("Langfuse callback flush failed (%s); continuing", type(error).__name__)
        finally:
            try:
                self.observation_sink.flush()
            except Exception as error:  # Telemetry must never fail a run.
                logger.warning("Langfuse event flush failed (%s); continuing", type(error).__name__)


def _default_handler(*, public_key: str | None = None) -> BaseCallbackHandler:
    """Create a Langfuse callback bound to Caro's masked client."""

    from langfuse.langchain import CallbackHandler

    if public_key is None:
        return CallbackHandler()
    return CallbackHandler(public_key=public_key)


def _default_flush() -> None:
    from langfuse import get_client

    get_client().flush()


def _default_observation_sink() -> ObservationEventSink:
    """Create a structured event sink using the process Langfuse client."""

    from langfuse import Langfuse

    # Keep the SDK-specific callable typing at this integration boundary. The
    # recursive Caro redactor is shared by callbacks and structured events.
    return LangfuseObservationEventSink(
        cast("_LangfuseClient", Langfuse(mask=cast("Any", _langfuse_mask)))
    )


def build_trace_context(
    settings: CaroSettings,
    run: RunRecord,
    *,
    environ: Mapping[str, str] | None = None,
    handler_factory: HandlerFactory = _default_handler,
    flush: FlushCallback = _default_flush,
    observation_sink_factory: ObservationSinkFactory = _default_observation_sink,
) -> TraceContext:
    """Create a Langfuse callback context, or a no-op context when disabled."""

    if not settings.langfuse.enabled:
        return TraceContext()
    environment = os.environ if environ is None else environ
    missing = [name for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not environment.get(name)]
    if missing:
        raise CaroError(f"Langfuse is enabled but required environment variable(s) are missing: {', '.join(missing)}")

    tags = [*settings.langfuse.tags, "deepagents", "langgraph"]
    memory_enabled = settings.memory.enabled and settings.memory.structured_enabled
    if memory_enabled:
        tags.append("structured-memory")
    metadata: dict[str, Any] = {
        "caro_run_id": run.run_id,
        "caro_config_hash": run.config_hash,
        "caro_environment": settings.langfuse.environment,
        "caro_memory_enabled": memory_enabled,
        "caro_memory_auto_retrieve": memory_enabled and settings.memory.auto_retrieve,
        "langfuse_session_id": run.thread_id,
        "langfuse_tags": tags,
    }
    observation_sink_ready = True
    try:
        observation_sink = observation_sink_factory()
    except Exception as error:  # Optional telemetry fails open.
        logger.warning("Langfuse structured events unavailable (%s); continuing", type(error).__name__)
        observation_sink = NullObservationEventSink()
        observation_sink_ready = False
    if handler_factory is _default_handler:
        # The v4 LangChain callback resolves its client through Langfuse's
        # singleton registry. Bind the configured public key explicitly so it
        # cannot select another project's client. If Caro could not create the
        # masked client, disable the default callback instead of risking raw
        # prompt/tool data through an unmasked global client.
        callbacks: list[BaseCallbackHandler] = (
            [_default_handler(public_key=str(environment["LANGFUSE_PUBLIC_KEY"]))]
            if observation_sink_ready
            else []
        )
    else:
        callbacks = [handler_factory()]
    return TraceContext(
        callbacks=callbacks,
        metadata=metadata,
        observation_sink=observation_sink,
        _flush=flush,
    )
