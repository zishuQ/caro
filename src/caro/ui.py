"""Adapter from the independent Caro product to the maintained Textual UI.

The terminal experience belongs to `libs/code/deepagents_code`. Caro owns the
governed backend and configuration, so this module only translates the model,
approval, workspace, and routing settings needed by the public Textual launch
function. The graph is assembled by `caro.code` through `libs/code`'s public
factory, while `CaroRuntime` still owns all Caro resources. Keeping this
boundary explicit prevents the two products from being silently merged again
while still letting Textual drive a Caro-backed graph.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from caro.code import build_code_agent_session
from caro.config import CaroSettings
from caro.runtime import CaroRuntime
from caro.state.runs import RunStatus


async def _graph_state(graph: Any, thread_id: str) -> Any:
    """Read the final graph state through either async or sync graph APIs."""

    config = {"configurable": {"thread_id": thread_id}}
    async_getter = getattr(graph, "aget_state", None)
    if callable(async_getter):
        return await async_getter(config)
    sync_getter = getattr(graph, "get_state", None)
    if callable(sync_getter):
        return await asyncio.to_thread(sync_getter, config)
    raise RuntimeError("Caro could not inspect the Textual graph state")


async def _resolve_local_model(
    runtime: CaroRuntime,
    model_spec: str,
    model_params: dict[str, Any] | None,
) -> str:
    """Validate and warm a TUI model selection through Caro's model pool."""

    del model_params
    pool = runtime.assembly.pool
    canonical = pool.canonical_identifier(model_spec)
    # Warm the instance here so provider import/credential errors are reported
    # by `/model`, before the next user turn starts.
    pool.get_for(canonical)
    return canonical


def _configured_model_rows(settings: CaroSettings) -> tuple[tuple[str, str], ...]:
    """Return unique `(provider:model, provider)` rows for the Caro selector."""

    from caro.models.providers import parse_model_identifier

    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for definition in settings.routing.models.values():
        if definition.model in seen:
            continue
        identifier = parse_model_identifier(definition.model)
        seen.add(definition.model)
        rows.append((definition.model, identifier.provider))
    return tuple(rows)


def _status_from_graph_state(snapshot: Any, return_code: int) -> tuple[RunStatus, str | None]:
    """Map final LangGraph state to the durable Caro lifecycle status."""

    if getattr(snapshot, "interrupts", ()):
        return RunStatus.WAITING_APPROVAL, None
    if getattr(snapshot, "next", ()):
        return RunStatus.PAUSED, None
    if return_code == 0:
        return RunStatus.COMPLETED, None
    return RunStatus.FAILED, f"Textual exited with code {return_code}"


async def _finalize_textual_run(runtime: CaroRuntime, session: Any, result: Any) -> None:
    """Persist a TUI run status from the final graph state, not only exit code."""

    run = session.run
    if run is None:
        return
    current = runtime.runs.get(run.run_id)
    if current.status not in {RunStatus.RUNNING, RunStatus.WAITING_APPROVAL}:
        return
    final_thread_id = getattr(result, "thread_id", None) or run.thread_id
    # A thread switch belongs to the interactive session, not the original
    # RunRecord. Inspect the original thread and retain an audit marker rather
    # than attributing another thread's completion to this run.
    inspected_thread_id = run.thread_id
    try:
        snapshot = await _graph_state(session.graph, inspected_thread_id)
        target, error = _status_from_graph_state(snapshot, int(result.return_code))
    except Exception as inspection_error:
        target = RunStatus.FAILED
        error = f"could not inspect final Textual graph state: {inspection_error}"
    payload: dict[str, object] = {}
    if final_thread_id != run.thread_id:
        payload["textual_thread_id"] = final_thread_id
    try:
        runtime.runs.transition(run.run_id, target, error=error, payload=payload)
    except ValueError:
        # An operator may cancel concurrently while Textual is shutting down.
        if runtime.runs.get(run.run_id).status is not RunStatus.CANCELLED:
            raise


async def _run_local_textual(runtime: CaroRuntime, settings: CaroSettings) -> int:
    """Run the existing Textual app against a Caro-backed Code graph."""

    from deepagents_code.app import run_textual_app
    from deepagents_code.approval_mode import ApprovalMode

    session = await build_code_agent_session(runtime, settings)
    approval_mode = (
        ApprovalMode.MANUAL if settings.approvals.enabled else ApprovalMode.YOLO
    )
    try:
        result = await run_textual_app(
            agent=session.graph,
            assistant_id="caro",
            backend=session.backend,
            approval_mode=approval_mode,
            cwd=settings.workspace.root,
            thread_id=session.run.thread_id if session.run is not None else None,
            mcp_server_info=session.mcp_server_info,
            model_explicitly_set=True,
            title="Caro",
            sub_title="DeepAgents + LangGraph",
            banner_brand="Caro",
            banner_show_version=False,
            banner_show_model=True,
            local_model_resolver=lambda model_spec, model_params: _resolve_local_model(
                runtime,
                model_spec,
                model_params,
            ),
            configured_models=_configured_model_rows(settings),
            # Keep Caro's effort preference beside its run/checkpoint state.
            # This avoids making a project session depend on write access to
            # the global DeepAgents config directory.
            model_config_path=settings.state.directory / "deepagents-code.toml",
        )
        if session.run is not None:
            await _finalize_textual_run(runtime, session, result)
        return int(result.return_code)
    except Exception as error:
        if session.run is not None:
            current = runtime.runs.get(session.run.run_id)
            if current.status in {RunStatus.RUNNING, RunStatus.WAITING_APPROVAL}:
                runtime.runs.transition(session.run.run_id, RunStatus.FAILED, error=str(error))
        raise
    finally:
        await session.close()


def run_textual(settings: CaroSettings) -> int:
    """Run the interactive Textual terminal for one Caro workspace.

    Args:
        settings: Validated Caro configuration used to select the model and
        workspace. Textual remains responsible for rendering and keyboard
            handling; Caro owns the local graph and its resource lifecycle.

    Returns:
        The Textual application's process-style return code.

    Raises:
        OSError: If the configured workspace cannot be entered.
        RuntimeError: If the Textual application cannot be launched.
    """

    original_cwd = Path.cwd()
    workspace = settings.workspace.root
    os.chdir(workspace)
    try:
        with CaroRuntime.open(settings) as runtime:
            return asyncio.run(_run_local_textual(runtime, settings))
    finally:
        os.chdir(original_cwd)
