"""Typer + Rich terminal interface for Caro."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any, Never
from urllib.parse import urlsplit

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from caro import __version__
from caro.config import CaroSettings, load_settings
from caro.errors import CaroError, RunNotFoundError
from caro.memory import ProjectMemory
from caro.memory.benchmark import load_benchmark_cases, run_memory_benchmark
from caro.memory.graph import MemoryConsolidator
from caro.memory.models import ConsolidationOutcome, MemoryAuthorization, MemoryKind, MemoryProposal
from caro.memory.policy import MemoryPolicy
from caro.memory.store import MemoryStore
from caro.models.providers import parse_model_identifier, provider_status
from caro.runner import CaroRunner
from caro.runtime import CaroRuntime
from caro.state.checkpoints import CheckpointView, coerce_state_mapping
from caro.state.events import GraphEvent, GraphEventKind
from caro.state.runs import RunRecord, RunStatus, RunStore
from caro.tools.preview import build_tool_preview
from caro.tools.registry import RiskLevel, ToolRegistry

app = typer.Typer(
    name="caro",
    help="Caro — governed terminal coding agent on DeepAgents + LangGraph.",
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode="markdown",
)
graph_app = typer.Typer(help="Inspect, fork, replay, and recover LangGraph checkpoints.", no_args_is_help=True)
memory_app = typer.Typer(help="Inspect and manage project-scoped long-term memory.", no_args_is_help=True)
app.add_typer(graph_app, name="graph")
app.add_typer(memory_app, name="memory")
console = Console()

_RUNTIME_VERSIONS: dict[str, str] = {
    "deepagents": "0.7.1",
    "deepagents-code": "0.1.50",
    "langchain": "1.3.14",
    "langgraph": "1.2.10",
    "langgraph-checkpoint-sqlite": "3.1.1",
    "langfuse": "4.14.2",
}


@dataclass(frozen=True, slots=True)
class CliContext:
    """Global CLI path overrides."""

    config: Path | None
    workspace: Path | None

    def settings(self) -> CaroSettings:
        """Load settings only when a command needs them."""

        config = self.config
        if config is None:
            for conventional in (Path("caro.yaml"), Path("caro.local.yaml")):
                if conventional.is_file():
                    config = conventional
                    break
        elif not config.is_file():
            raise CaroError(f"configuration file not found: {config}")
        return load_settings(config, workspace=self.workspace)


def _ctx(context: typer.Context) -> CliContext:
    value = context.obj
    if not isinstance(value, CliContext):
        raise RuntimeError("Caro CLI context is not initialized")
    return value


def _fail(error: Exception, *, code: int = 1) -> Never:
    console.print(f"[bold red]Error:[/bold red] {error}")
    raise typer.Exit(code) from error


def _load(context: typer.Context) -> CaroSettings:
    try:
        return _ctx(context).settings()
    except (CaroError, OSError, ValueError, ValidationError) as error:
        _fail(error)


def _run_store(settings: CaroSettings) -> RunStore:
    return RunStore(settings.state.directory / settings.state.runs_database)


def _structured_memory_store(settings: CaroSettings) -> MemoryStore:
    if not settings.memory.enabled or not settings.memory.structured_enabled:
        raise CaroError("structured memory is disabled")
    return MemoryStore(
        settings.state.directory / settings.memory.structured_database,
        allowed_root=settings.workspace.root,
    )


def _memory_consolidator(settings: CaroSettings, store: MemoryStore) -> MemoryConsolidator:
    return MemoryConsolidator(
        store,
        MemoryPolicy(
            minimum_confidence=settings.memory.minimum_confidence,
            max_content_chars=settings.memory.max_item_chars,
            episodic_ttl_days=settings.memory.episodic_ttl_days,
        ),
    )


def _short(value: Any, limit: int = 100) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def _proxy_status() -> str:
    """Return a redacted summary of proxy variables visible to Caro."""

    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "OPENAI_PROXY")
    configured: list[str] = []
    for name in names:
        value = os.environ.get(name) or os.environ.get(name.lower())
        if not value:
            continue
        if name == "NO_PROXY":
            configured.append(f"{name}=configured")
            continue
        try:
            parsed = urlsplit(value)
            host = parsed.hostname or "?"
        except ValueError:
            configured.append(f"{name}=configured")
            continue
        try:
            port = f":{parsed.port}" if parsed.port is not None else ""
        except ValueError:
            port = ""
        scheme = parsed.scheme or "unknown"
        configured.append(f"{name}={scheme}://{host}{port}")
    if not configured:
        return "none; provider clients will attempt a direct connection"
    return ", ".join(configured)


class TerminalRenderer:
    """Minimal Rich renderer for normalized LangGraph stream events."""

    def __init__(self) -> None:
        self._streaming_text = False

    def __call__(self, event: GraphEvent) -> None:
        if event.kind is GraphEventKind.MESSAGE and event.text:
            console.print(event.text, end="", markup=False, highlight=False)
            self._streaming_text = True
            return
        if event.kind is GraphEventKind.UPDATE and event.node:
            if self._streaming_text:
                console.print()
                self._streaming_text = False
            if event.node == "orchestration_finalize":
                final_text = self._orchestration_text(event.data)
                if final_text:
                    console.print(Panel(Text(final_text), title="Caro orchestration result", border_style="green"))
                return
            if event.node not in {"model"}:
                console.print(f"[dim]graph update · {event.node}[/dim]")
        elif event.kind is GraphEventKind.TASK and isinstance(event.data, dict):
            name = event.data.get("name")
            if name:
                console.print(f"[dim]graph task · {name}[/dim]")

    @staticmethod
    def _orchestration_text(value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        messages = value.get("messages")
        if not isinstance(messages, list) or not messages:
            return ""
        message = messages[-1]
        if hasattr(message, "text"):
            return str(message.text)
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return str(message["content"])
        return ""

    def finish(self) -> None:
        if self._streaming_text:
            console.print()
            self._streaming_text = False


def _terminal_approval(
    request: dict[str, Any],
    *,
    registry: ToolRegistry | None = None,
    workspace: Path | None = None,
) -> list[dict[str, Any]]:
    """Prompt for LangGraph HITL decisions with a concrete side-effect preview."""

    action_requests = request.get("action_requests", [])
    if not isinstance(action_requests, list):
        raise TypeError("approval interrupt action_requests must be a list")
    resolved_registry = ToolRegistry.default() if registry is None else registry
    resolved_workspace = Path.cwd() if workspace is None else workspace
    decisions: list[dict[str, Any]] = []
    for action in action_requests:
        if not isinstance(action, dict):
            raise TypeError("approval action must be a mapping")
        name = str(action.get("name", "unknown"))
        raw_args = action.get("args", {})
        args = raw_args if isinstance(raw_args, dict) else {}
        description = action.get("description")
        assessment = resolved_registry.assess(name, args)
        body = build_tool_preview(name, args, resolved_workspace, assessment)
        if description:
            body = f"{description}\n\n{body}"
        console.print(Panel(body, title=f"Approval required · {name}", border_style="yellow"))
        choice = typer.prompt(
            "Decision: [a]pprove, [e]dit, or [r]eject",
            default="r",
            show_default=True,
        ).strip().lower()
        if choice in {"a", "approve", "y", "yes"}:
            decisions.append({"type": "approve"})
        elif choice in {"e", "edit"}:
            edited = typer.prompt(
                "Edited arguments as JSON",
                default=json.dumps(args, ensure_ascii=False, default=_json_default),
            )
            try:
                edited_args = json.loads(edited)
            except json.JSONDecodeError as error:
                raise ValueError(f"edited tool arguments must be valid JSON: {error}") from error
            if not isinstance(edited_args, dict):
                raise ValueError("edited tool arguments must be a JSON object")
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {"name": name, "args": edited_args},
                }
            )
        else:
            decisions.append({"type": "reject", "message": "Rejected by the Caro terminal operator."})
    return decisions


def _run_summary(run: RunRecord) -> None:
    style = {
        RunStatus.COMPLETED: "green",
        RunStatus.FAILED: "red",
        RunStatus.WAITING_APPROVAL: "yellow",
        RunStatus.PAUSED: "yellow",
    }.get(run.status, "cyan")
    console.print(
        Panel(
            f"run: {run.run_id}\nthread: {run.thread_id}\nstatus: [{style}]{run.status.value}[/{style}]"
            + (f"\ncheckpoint: {run.checkpoint_id}" if run.checkpoint_id else ""),
            title="Caro",
        )
    )


@app.callback()
def _configure(
    context: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML configuration path.")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w", help="Override workspace root.")] = None,
) -> None:
    """Configure paths shared by all Caro commands.

    With no subcommand in a real terminal, Caro opens the existing DeepAgents
    Textual REPL through the adapter in `caro.ui`. The governed Caro Runtime
    remains the backend for `caro run` and the lifecycle commands.
    """

    context.obj = CliContext(config=config, workspace=workspace)
    if context.invoked_subcommand is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            launch_tui(context)
        else:
            typer.echo(context.get_help())


def main_entry() -> None:
    """Run the Caro command-line application."""

    app()


@app.command("tui")
def tui(context: typer.Context) -> None:
    """Open the interactive Textual terminal backed by Caro settings."""

    launch_tui(context)


def launch_tui(context: typer.Context) -> None:
    """Launch the maintained `deepagents_code` Textual interface.

    This adapter intentionally keeps the UI in `libs/code`; it only maps the
    Caro model/workspace configuration into that public launch API. Direct
    `caro run` uses Caro's full Memory v2, orchestration, and HITL runtime.
    """

    try:
        settings = _ctx(context).settings()
        from caro.ui import run_textual

        result = run_textual(settings)
    except (CaroError, OSError, ValueError, ValidationError) as error:
        _fail(error)
        return
    raise typer.Exit(result)


@app.command()
def version() -> None:
    """Print the Caro version."""

    console.print(f"caro {__version__}")


@app.command("init")
def init_config(
    provider: Annotated[
        str,
        typer.Option(help="Initial provider: openai, deepseek, anthropic, google_genai, openrouter, or ollama."),
    ] = "openai",
    output: Annotated[Path, typer.Option("--output", "-o", help="Destination YAML file.")] = Path("caro.yaml"),
    force: Annotated[bool, typer.Option("--force", help="Replace an existing destination.")] = False,
) -> None:
    """Create a minimal provider-specific Caro YAML file."""

    models = {
        "openai": "openai:gpt-5.6-sol",
        "deepseek": "deepseek:deepseek-v4-flash",
        "anthropic": "anthropic:claude-sonnet-4-6",
        "google_genai": "google_genai:gemini-2.5-pro",
        "openrouter": "openrouter:openai/gpt-5.5",
        "ollama": "ollama:gpt-oss:20b",
    }
    if provider not in models:
        _fail(ValueError(f"unsupported init provider: {provider}"), code=2)
    if output.exists() and not force:
        _fail(FileExistsError(f"destination exists; use --force to replace it: {output}"), code=2)
    payload = {
        "routing": {"default": "primary", "models": {"primary": {"model": models[provider]}}},
        "workspace": {"root": ".", "shell_enabled": True},
        "approvals": {"enabled": True, "minimum_risk": "high"},
        "state": {"directory": ".caro"},
        "memory": {
            "enabled": True,
            "sources": [".caro/memory/AGENTS.md"],
            "initialize_missing": True,
            "structured_enabled": True,
            "structured_database": "memory.sqlite3",
            "auto_retrieve": True,
        },
        "orchestration": {"enabled": False},
        "code": {
            "enabled": True,
            "skills_enabled": True,
            "mcp_enabled": True,
            "trust_project_mcp": False,
            "ask_user_enabled": True,
            "interpreter_enabled": False,
        },
        "langfuse": {"enabled": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    console.print(f"Created [bold]{output}[/bold] for provider [cyan]{provider}[/cyan].")


@app.command()
def providers(context: typer.Context) -> None:
    """Show configured model providers, packages, credentials, and routing roles."""

    settings = _load(context)
    table = Table(title="Caro model providers")
    for column in ("Pool", "Provider", "Model", "Role", "Integration", "Credentials"):
        table.add_column(column, overflow="fold")
    for name, definition in settings.routing.models.items():
        identifier = parse_model_identifier(definition.model)
        status = provider_status(identifier.provider)
        roles: list[str] = []
        if name == settings.routing.default:
            roles.append("default")
        roles.extend(f"rule:{rule.name}" for rule in settings.routing.rules if rule.model == name)
        if name in settings.routing.fallbacks:
            roles.append(f"fallback:{settings.routing.fallbacks.index(name) + 1}")
        if status.spec is None:
            integration = "unmanaged"
        elif status.installed_version is None:
            integration = f"[red]missing[/red] ({status.spec.extra} extra)"
        else:
            integration = f"{status.spec.distribution} {status.installed_version}"
        credential_style = "green" if status.credential_state in {"configured", "local"} else "yellow"
        table.add_row(
            name,
            status.spec.display_name if status.spec else identifier.provider,
            identifier.model,
            ", ".join(roles) or "unused",
            integration,
            f"[{credential_style}]{status.credential_state}[/{credential_style}]",
        )
    console.print(table)


@app.command()
def tools(context: typer.Context) -> None:
    """Show Caro's unified Tool Registry and effective approval policy."""

    settings = _load(context)
    registry = ToolRegistry.default()
    minimum = RiskLevel(settings.approvals.minimum_risk)
    table = Table(title="Caro Tool Registry")
    table.add_column("Tool")
    table.add_column("Risk")
    table.add_column("Approval")
    table.add_column("Timeout")
    table.add_column("Retry")
    table.add_column("Source")
    table.add_column("Purpose")
    for spec in registry.all():
        approval = registry.requires_approval(spec, enabled=settings.approvals.enabled, minimum_risk=minimum)
        timeout = f"{spec.timeout_seconds:g}s" if spec.timeout_seconds is not None else "backend"
        table.add_row(
            spec.name,
            spec.risk.value,
            "yes" if approval else "no",
            timeout,
            str(spec.retry.max_attempts),
            spec.source,
            spec.description,
        )
    console.print(table)


@app.command()
def agents(context: typer.Context) -> None:
    """Show effective single-agent or LangGraph multi-agent orchestration."""

    settings = _load(context)
    table = Table(title="Caro agent orchestration")
    table.add_column("Stage", style="cyan")
    table.add_column("Mode", no_wrap=True)
    table.add_column("Agents")
    if not settings.orchestration.enabled:
        table.add_row("dynamic", "automatic", "general-purpose")
        console.print(table)
        definition = settings.routing.models[settings.routing.default]
        console.print(
            f"[bold]Main agent[/bold] · {settings.routing.default} ({definition.model}) · "
            f"{'shell' if settings.workspace.shell_enabled else 'files only'} · "
            f"{'memory' if settings.memory.enabled else 'no memory'}\n"
            "  DeepAgents automatic task-tool delegation; deterministic orchestration is disabled."
        )
        return

    for stage in settings.orchestration.stages:
        table.add_row(stage.name, "parallel" if len(stage.agents) > 1 else "sequential", ", ".join(stage.agents))
    console.print(table)
    console.print("\n[bold]Role policies[/bold]")
    for agent_name, role in settings.orchestration.agents.items():
        model_name = role.model or settings.routing.default
        definition = settings.routing.models[model_name]
        shell_enabled = settings.workspace.shell_enabled and role.shell_enabled is not False
        memory_enabled = settings.memory.enabled and role.memory
        delegate_summary = ""
        if role.delegates:
            delegate_summary = "\n  Delegates: " + ", ".join(
                f"{delegate.name} ({delegate.model})" for delegate in role.delegates
            )
        console.print(
            f"[cyan]{agent_name}[/cyan] · {model_name} ({definition.model}) · "
            f"{'shell' if shell_enabled else 'files only'} · "
            f"{'memory' if memory_enabled else 'no memory'}\n  {role.description}"
            f"{delegate_summary}"
        )


@app.command()
def doctor(context: typer.Context) -> None:
    """Validate runtime pins, configured providers, workspace, SQLite, and Langfuse."""

    settings = _load(context)
    table = Table(title="Caro doctor")
    table.add_column("Check", overflow="fold")
    table.add_column("Result", overflow="fold")
    failures = 0
    for distribution, expected in _RUNTIME_VERSIONS.items():
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            actual = "missing"
        ok = actual == expected
        failures += not ok
        table.add_row(distribution, f"[green]{actual}[/green]" if ok else f"[red]{actual} (expected {expected})[/red]")

    seen: set[str] = set()
    for definition in settings.routing.models.values():
        provider = parse_model_identifier(definition.model).provider
        if provider in seen:
            continue
        seen.add(provider)
        status = provider_status(provider)
        if status.spec is None:
            table.add_row(f"provider:{provider}", "[yellow]unmanaged; LangChain validates at initialization[/yellow]")
            continue
        if status.installed_version is None:
            failures += 1
            table.add_row(f"provider:{provider}", f"[red]missing; {status.spec.install_hint}[/red]")
        elif status.credential_state == "missing":
            failures += 1
            variables = "/".join(status.credential_variables)
            table.add_row(f"provider:{provider}", f"[red]missing credentials: {variables}[/red]")
        else:
            table.add_row(f"provider:{provider}", f"[green]{status.credential_state}[/green]")

    table.add_row("network proxy", _proxy_status())

    workspace = settings.workspace.root
    workspace_ok = workspace.is_dir() and os.access(workspace, os.R_OK | os.W_OK | os.X_OK)
    failures += not workspace_ok
    table.add_row("workspace", f"[green]{workspace}[/green]" if workspace_ok else f"[red]not usable: {workspace}[/red]")
    if not settings.code.enabled:
        table.add_row("code adapter", "disabled")
    else:
        mcp_config = settings.code.mcp_config
        if mcp_config is not None:
            resolved_mcp = mcp_config.expanduser()
            if not resolved_mcp.is_absolute():
                resolved_mcp = workspace / resolved_mcp
            if not resolved_mcp.is_file():
                failures += 1
                table.add_row("code MCP config", f"[red]missing: {resolved_mcp}[/red]")
            else:
                table.add_row("code MCP config", f"[green]{resolved_mcp}[/green]")
        else:
            table.add_row(
                "code adapter",
                "[green]enabled; Textual/Skills/MCP discovery available[/green]",
            )
        try:
            from deepagents_code.config import settings as code_settings

            table.add_row(
                "web search",
                "[green]enabled (Tavily)[/green]"
                if code_settings.has_tavily
                else "disabled; set TAVILY_API_KEY",
            )
        except (ImportError, AttributeError):
            table.add_row("web search", "disabled; optional Code tools unavailable")
    if settings.langfuse.enabled:
        missing_langfuse = [
            name for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not os.environ.get(name)
        ]
        failures += bool(missing_langfuse)
        table.add_row(
            "langfuse",
            "[green]configured[/green]"
            if not missing_langfuse
            else f"[red]missing: {', '.join(missing_langfuse)}[/red]",
        )
    else:
        table.add_row("langfuse", "disabled")
    if not settings.workspace.shell_enabled:
        shell_policy = "shell disabled"
    elif settings.workspace.sandbox_backend == "bubblewrap":
        shell_policy = "bubblewrap sandbox" if shutil.which("bwrap") else "bubblewrap unavailable"
        if not shutil.which("bwrap"):
            failures += 1
    else:
        shell_policy = "guarded local shell"
    table.add_row("shell policy", shell_policy)
    if settings.orchestration.enabled:
        parallel_width = max(len(stage.agents) for stage in settings.orchestration.stages)
        table.add_row(
            "orchestration",
            f"[green]{len(settings.orchestration.agents)} agent(s), "
            f"{len(settings.orchestration.stages)} stage(s), max parallel {parallel_width}[/green]",
        )
    else:
        table.add_row("orchestration", "single DeepAgent + automatic general-purpose subagent")
    if not settings.memory.enabled:
        table.add_row("project memory", "disabled")
    else:
        try:
            memory_statuses = ProjectMemory(settings.workspace.root, settings.memory).statuses()
            missing_memory = [item for item in memory_statuses if not item.exists]
            if missing_memory and not settings.memory.initialize_missing:
                failures += 1
                table.add_row("project memory", "[red]missing and automatic initialization is disabled[/red]")
            elif missing_memory:
                table.add_row("project memory", "[yellow]ready; missing sources initialize on first run[/yellow]")
            else:
                total_bytes = sum(item.size_bytes for item in memory_statuses)
                table.add_row("project memory", f"[green]{len(memory_statuses)} source(s), {total_bytes} bytes[/green]")
        except (CaroError, OSError) as error:
            failures += 1
            table.add_row("project memory", f"[red]{error}[/red]")
        if settings.memory.structured_enabled:
            database = settings.state.directory / settings.memory.structured_database
            if database.is_file():
                try:
                    stats = MemoryStore(database, allowed_root=settings.workspace.root).stats()
                    table.add_row(
                        "structured memory",
                        f"[green]{stats.active} active, {stats.archived} archived[/green]",
                    )
                except (OSError, ValueError) as error:
                    failures += 1
                    table.add_row("structured memory", f"[red]{error}[/red]")
            else:
                table.add_row("structured memory", "[yellow]ready; initializes on first run[/yellow]")
    console.print(table)
    if failures:
        raise typer.Exit(1)


@memory_app.command("status")
def memory_status(context: typer.Context) -> None:
    """Show configured project-memory sources without initializing a model."""

    settings = _load(context)
    if not settings.memory.enabled:
        console.print("Project memory is disabled.")
        return
    try:
        statuses = ProjectMemory(settings.workspace.root, settings.memory).statuses()
    except (CaroError, OSError) as error:
        _fail(error)
        return
    table = Table(title="Caro project memory")
    for column in ("Source", "Backend path", "State", "Bytes"):
        table.add_column(column)
    for item in statuses:
        table.add_row(
            item.source.configured_path,
            item.source.backend_path,
            "ready" if item.exists else "not initialized",
            str(item.size_bytes),
        )
    console.print(table)
    if settings.memory.structured_enabled:
        database = settings.state.directory / settings.memory.structured_database
        if database.is_file():
            try:
                stats = MemoryStore(database, allowed_root=settings.workspace.root).stats()
            except (OSError, ValueError) as error:
                _fail(error)
                return
            console.print(
                f"Structured memory · [green]ready[/green] · {stats.active} active · "
                f"{stats.archived} archived · {database}"
            )
        else:
            console.print(f"Structured memory · [yellow]not initialized[/yellow] · {database}")


@memory_app.command("init")
def memory_init(context: typer.Context) -> None:
    """Initialize missing project-memory files from Caro's safe template."""

    settings = _load(context)
    if not settings.memory.enabled:
        _fail(ValueError("project memory is disabled"), code=2)
    try:
        manager = ProjectMemory(settings.workspace.root, settings.memory)
        manager.prepare()
        statuses = manager.statuses()
        if settings.memory.structured_enabled:
            _structured_memory_store(settings)
    except (CaroError, OSError, ValueError) as error:
        _fail(error)
        return
    for item in statuses:
        console.print(f"Ready [bold]{item.source.host_path}[/bold] ({item.size_bytes} bytes).")


@memory_app.command("show")
def memory_show(
    context: typer.Context,
    source: Annotated[str | None, typer.Option(help="Configured memory source to show.")] = None,
) -> None:
    """Print project memory without initializing a model."""

    settings = _load(context)
    if not settings.memory.enabled:
        _fail(ValueError("project memory is disabled"), code=2)
    try:
        memories = ProjectMemory(settings.workspace.root, settings.memory).read(source)
    except (CaroError, OSError) as error:
        _fail(error)
        return
    for memory_source, content in memories:
        console.print(Panel(Text(content), title=f"Project memory · {memory_source.configured_path}"))


@memory_app.command("add")
def memory_add(
    context: typer.Context,
    text: Annotated[str, typer.Argument(help="Confirmed durable project learning to append.")],
    source: Annotated[str | None, typer.Option(help="Configured destination source.")] = None,
) -> None:
    """Append an operator-confirmed project memory entry."""

    settings = _load(context)
    if not settings.memory.enabled:
        _fail(ValueError("project memory is disabled"), code=2)
    try:
        status = ProjectMemory(settings.workspace.root, settings.memory).append_operator_memory(text, source)
    except (CaroError, OSError) as error:
        _fail(error)
        return
    console.print(
        f"Updated [bold]{status.source.host_path}[/bold] ({status.size_bytes} bytes). "
        "Do not store credentials or transient task data in project memory."
    )


@memory_app.command("remember")
def memory_remember(
    context: typer.Context,
    text: Annotated[str, typer.Argument(help="Verified durable coding knowledge.")],
    kind: Annotated[MemoryKind, typer.Option(help="episodic, semantic, or procedural.")] = MemoryKind.SEMANTIC,
    scope: Annotated[str, typer.Option(help="Structured scope, e.g. project or role:reviewer.")] = "project",
    importance: Annotated[float, typer.Option(min=0, max=1)] = 0.7,
    confidence: Annotated[float, typer.Option(min=0, max=1)] = 1.0,
    evidence: Annotated[str, typer.Option(help="Verifiable source or reason for this memory.")] = "Operator confirmed.",
    expires_days: Annotated[int | None, typer.Option(min=1, max=3650)] = None,
    supersedes_id: Annotated[str | None, typer.Option(help="Same-kind/scope memory replaced atomically.")] = None,
) -> None:
    """Submit operator-authorized knowledge to the LangGraph consolidation workflow."""

    settings = _load(context)
    try:
        store = _structured_memory_store(settings)
        expiry = None if expires_days is None else datetime.now(UTC) + timedelta(days=expires_days)
        result = _memory_consolidator(settings, store).consolidate(
            MemoryProposal(
                kind=kind,
                scope=scope,
                content=text,
                importance=importance,
                confidence=confidence,
                evidence=evidence,
                source_agent="operator",
                metadata={"origin": "caro memory remember"},
                expires_at=expiry,
                supersedes_id=supersedes_id,
            ),
            authorization=MemoryAuthorization.OPERATOR,
        )
    except (CaroError, KeyError, OSError, PermissionError, ValueError) as error:
        _fail(error)
        return
    if result.outcome is ConsolidationOutcome.REJECTED:
        _fail(CaroError(result.reason))
    if result.item is None:
        _fail(CaroError("memory consolidation returned no item"))
    console.print(
        f"Structured memory [bold]{result.outcome.value}[/bold]\n"
        f"memory_id: {result.item.memory_id}\n"
        f"kind: {result.item.kind.value}\n"
        f"scope: {result.item.scope}"
    )
    if result.similar_memory_ids:
        console.print(f"Similar existing IDs: {', '.join(result.similar_memory_ids)}")


@memory_app.command("search")
def memory_search(
    context: typer.Context,
    query: Annotated[str, typer.Argument(help="Full-text memory query.")],
    scope: Annotated[list[str] | None, typer.Option("--scope", help="Repeat to search multiple scopes.")] = None,
    kind: Annotated[MemoryKind | None, typer.Option(help="Optional kind filter.")] = None,
    limit: Annotated[int | None, typer.Option(min=1, max=50)] = None,
) -> None:
    """Search active structured memory without initializing a model."""

    settings = _load(context)
    try:
        store = _structured_memory_store(settings)
        results = store.search(
            query,
            scopes=tuple(scope or ["project"]),
            kinds=None if kind is None else (kind,),
            limit=limit or settings.memory.retrieval_limit,
            min_importance=settings.memory.minimum_importance,
        )
    except (CaroError, OSError, ValueError) as error:
        _fail(error)
        return
    if not results:
        console.print("No active structured memories matched.")
        return
    table = Table(title="Caro structured-memory search")
    table.add_column("ID", overflow="fold")
    table.add_column("Kind")
    table.add_column("Scope")
    table.add_column("Score")
    table.add_column("Memory", overflow="fold")
    for result in results:
        table.add_row(
            result.item.memory_id,
            result.item.kind.value,
            result.item.scope,
            f"{result.score:.3f}",
            result.item.content,
        )
    console.print(table)


@memory_app.command("stats")
def memory_stats(context: typer.Context) -> None:
    """Show structured-memory lifecycle and scope counts."""

    settings = _load(context)
    try:
        stats = _structured_memory_store(settings).stats()
    except (CaroError, OSError) as error:
        _fail(error)
        return
    table = Table(title="Caro structured-memory stats")
    table.add_column("Metric")
    table.add_column("Count")
    table.add_row("active", str(stats.active))
    table.add_row("archived", str(stats.archived))
    for kind, count in sorted(stats.by_kind.items(), key=lambda item: item[0].value):
        table.add_row(f"kind:{kind.value}", str(count))
    for item_scope, count in sorted(stats.by_scope.items()):
        table.add_row(f"scope:{item_scope}", str(count))
    console.print(table)


@memory_app.command("history")
def memory_history(
    context: typer.Context,
    limit: Annotated[int, typer.Option(min=1, max=1000)] = 50,
    run_id: Annotated[str | None, typer.Option(help="Optional source Run ID.")] = None,
) -> None:
    """Show content-free consolidation audit records."""

    settings = _load(context)
    try:
        operations = _structured_memory_store(settings).operations(limit=limit, source_run_id=run_id)
    except (CaroError, OSError, ValueError) as error:
        _fail(error)
        return
    table = Table(title="Caro memory consolidation history")
    for column in ("Time", "Outcome", "Kind", "Scope", "Authority", "Run", "Memory", "Reason"):
        table.add_column(column, overflow="fold")
    for operation in operations:
        table.add_row(
            operation.created_at.isoformat(),
            operation.outcome,
            operation.kind.value,
            operation.scope,
            operation.authorization.value,
            operation.source_run_id or "—",
            operation.memory_id or "—",
            operation.reason,
        )
    console.print(table)


@memory_app.command("archive")
def memory_archive(
    context: typer.Context,
    memory_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option(help="Verified reason for soft archival.")],
    scope: Annotated[list[str] | None, typer.Option("--scope", help="Allowed scope; repeat as needed.")] = None,
) -> None:
    """Soft-archive structured memory while retaining its audit history."""

    settings = _load(context)
    try:
        item = _structured_memory_store(settings).archive(
            memory_id,
            allowed_scopes=tuple(scope or ["project"]),
            reason=reason,
            actor="operator",
        )
    except (CaroError, KeyError, OSError, PermissionError, ValueError) as error:
        _fail(error)
        return
    console.print(f"Structured memory [bold]archived[/bold]: {item.memory_id} ({item.scope}).")


@memory_app.command("prune")
def memory_prune(context: typer.Context) -> None:
    """Soft-archive expired episodic memories without physical deletion."""

    settings = _load(context)
    try:
        count = _structured_memory_store(settings).prune_expired(actor="operator")
    except (CaroError, OSError) as error:
        _fail(error)
        return
    console.print(f"Archived {count} expired structured memory item(s).")


@memory_app.command("benchmark")
def memory_benchmark(
    context: typer.Context,
    dataset: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    limit: Annotated[int | None, typer.Option(min=1, max=50)] = None,
    minimum_hit_rate: Annotated[float, typer.Option("--min-hit-rate", min=0, max=1)] = 0.0,
    minimum_recall: Annotated[float, typer.Option("--min-recall", min=0, max=1)] = 0.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Evaluate retrieval quality and scope isolation without model calls or access side effects."""

    settings = _load(context)
    try:
        report = run_memory_benchmark(
            _structured_memory_store(settings),
            load_benchmark_cases(dataset),
            limit=limit or settings.memory.retrieval_limit,
            minimum_importance=settings.memory.minimum_importance,
        )
    except (CaroError, OSError, ValueError) as error:
        _fail(error)
        return
    payload = {
        "case_count": report.case_count,
        "hit_rate": report.hit_rate,
        "recall_at_k": report.recall_at_k,
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "cases": [
            {
                "name": case.name,
                "passed": case.passed,
                "retrieved_memory_ids": list(case.retrieved_memory_ids),
                "missing_expected": list(case.missing_expected),
                "leaked_forbidden": list(case.leaked_forbidden),
                "reciprocal_rank": case.reciprocal_rank,
            }
            for case in report.cases
        ],
    }
    if as_json:
        console.print(json.dumps(payload, ensure_ascii=False, sort_keys=True), markup=False, highlight=False)
    else:
        table = Table(title="Caro structured-memory benchmark")
        for column in ("Case", "Pass", "RR", "Missing", "Forbidden leakage"):
            table.add_column(column, overflow="fold")
        for case in report.cases:
            table.add_row(
                case.name,
                "yes" if case.passed else "no",
                f"{case.reciprocal_rank:.3f}",
                ", ".join(case.missing_expected) or "—",
                ", ".join(case.leaked_forbidden) or "—",
            )
        console.print(table)
        console.print(
            f"Hit rate: {report.hit_rate:.3f} · Recall@K: {report.recall_at_k:.3f} · "
            f"MRR: {report.mean_reciprocal_rank:.3f}"
        )
    if report.hit_rate < minimum_hit_rate or report.recall_at_k < minimum_recall:
        raise typer.Exit(1)


@app.command("run")
def run_task(
    context: typer.Context,
    task: Annotated[str, typer.Argument(help="Coding task for Caro.")],
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Pause at approval interrupts instead of prompting."),
    ] = False,
) -> None:
    """Run a new coding task and stream LangGraph events in the terminal."""

    settings = _load(context)
    renderer = TerminalRenderer()
    try:
        with CaroRuntime.open(settings) as runtime:
            runner = CaroRunner(
                runtime.assembly.graph,
                runtime.runs,
                settings,
                approval_handler=(
                    None
                    if non_interactive
                    else lambda request: _terminal_approval(
                        request,
                        registry=runtime.assembly.registry,
                        workspace=settings.workspace.root,
                    )
                ),
                event_sink=renderer,
                registry=runtime.assembly.registry,
            )
            result = runner.run(task)
    except (CaroError, OSError, ValueError, ValidationError) as error:
        renderer.finish()
        _fail(error)
        return
    renderer.finish()
    _run_summary(result)


@app.command()
def resume(
    context: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Run ID to resume.")],
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Leave approval interrupts paused."),
    ] = False,
) -> None:
    """Resume a paused, failed, or approval-waiting run."""

    settings = _load(context)
    renderer = TerminalRenderer()
    try:
        with CaroRuntime.open(settings) as runtime:
            result = CaroRunner(
                runtime.assembly.graph,
                runtime.runs,
                settings,
                approval_handler=(
                    None
                    if non_interactive
                    else lambda request: _terminal_approval(
                        request,
                        registry=runtime.assembly.registry,
                        workspace=settings.workspace.root,
                    )
                ),
                event_sink=renderer,
                registry=runtime.assembly.registry,
            ).resume(run_id)
    except (CaroError, OSError, ValueError, ValidationError) as error:
        renderer.finish()
        _fail(error)
        return
    renderer.finish()
    _run_summary(result)


@app.command("runs")
def list_runs(
    context: typer.Context,
    limit: Annotated[int, typer.Option(min=1, max=1000, help="Maximum rows.")] = 50,
    status: Annotated[RunStatus | None, typer.Option(help="Filter by lifecycle status.")] = None,
) -> None:
    """List durable Caro runs without initializing a model."""

    settings = _load(context)
    try:
        records = _run_store(settings).list_runs(limit=limit, status=status)
    except (OSError, ValueError) as error:
        _fail(error)
        return
    table = Table(title="Caro runs")
    for column in ("Run", "Status", "Updated", "Checkpoint", "Task"):
        table.add_column(column)
    for record in records:
        table.add_row(
            record.run_id,
            record.status.value,
            record.updated_at,
            record.checkpoint_id or "—",
            _short(record.task),
        )
    console.print(table)


@app.command()
def retry(context: typer.Context, run_id: Annotated[str, typer.Argument(help="Failed run ID to retry.")]) -> None:
    """Retry a failed run as a new child run and preserve its lineage."""

    settings = _load(context)
    renderer = TerminalRenderer()
    try:
        with CaroRuntime.open(settings) as runtime:
            result = CaroRunner(
                runtime.assembly.graph,
                runtime.runs,
                settings,
                approval_handler=lambda request: _terminal_approval(
                    request,
                    registry=runtime.assembly.registry,
                    workspace=settings.workspace.root,
                ),
                event_sink=renderer,
                registry=runtime.assembly.registry,
            ).retry(run_id)
    except (CaroError, OSError, ValueError, ValidationError) as error:
        renderer.finish()
        _fail(error)
        return
    renderer.finish()
    _run_summary(result)


@app.command()
def cancel(context: typer.Context, run_id: Annotated[str, typer.Argument(help="Run ID to cancel.")]) -> None:
    """Cancel a non-terminal run in Caro's durable lifecycle store."""

    settings = _load(context)
    try:
        store = _run_store(settings)
        current = store.get(run_id)
        if current.status in {RunStatus.COMPLETED, RunStatus.CANCELLED}:
            raise ValueError(f"run cannot be cancelled from status {current.status.value}")
        result = store.transition(
            run_id,
            RunStatus.CANCELLED,
            payload={"reason": "cancelled by Caro operator"},
        )
    except (CaroError, OSError, ValueError, ValidationError) as error:
        _fail(error)
        return
    _run_summary(result)


@app.command("inspect")
def inspect_run(context: typer.Context, run_id: Annotated[str, typer.Argument()]) -> None:
    """Inspect one run and its append-only lifecycle events."""

    settings = _load(context)
    try:
        store = _run_store(settings)
        record = store.get(run_id)
        events = store.events(run_id)
    except (OSError, RunNotFoundError) as error:
        _fail(error)
        return
    details = {
        "run_id": record.run_id,
        "thread_id": record.thread_id,
        "status": record.status.value,
        "task": record.task,
        "workspace": record.workspace,
        "checkpoint_id": record.checkpoint_id,
        "parent_run_id": record.parent_run_id,
        "config_hash": record.config_hash,
        "error": record.error,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    console.print(Panel(JSON.from_data(details), title="Caro run"))
    table = Table(title="Lifecycle events")
    table.add_column("ID")
    table.add_column("Time")
    table.add_column("Type")
    table.add_column("Payload")
    for event in events:
        table.add_row(str(event.event_id), event.created_at, event.event_type, _short(event.payload))
    console.print(table)


def _checkpoint_table(items: list[CheckpointView]) -> Table:
    table = Table(title="LangGraph checkpoint history (newest first)")
    for column in ("Checkpoint", "Parent", "Step", "Source", "Next", "Interrupts", "Errors", "Created"):
        table.add_column(column)
    for item in items:
        table.add_row(
            item.checkpoint_id,
            item.parent_checkpoint_id or "—",
            str(item.step) if item.step is not None else "—",
            item.source or "—",
            ", ".join(item.next_nodes) or "END",
            str(item.interrupt_count),
            "; ".join(item.task_errors) or "—",
            item.created_at or "—",
        )
    return table


@graph_app.command("history")
def graph_history(
    context: typer.Context,
    run_id: Annotated[str, typer.Argument()],
    limit: Annotated[int | None, typer.Option(min=1, max=1000)] = None,
) -> None:
    """List a run's LangGraph checkpoint history."""

    settings = _load(context)
    try:
        with CaroRuntime.open(settings) as runtime:
            run = runtime.runs.get(run_id)
            items = runtime.controller.history(run.thread_id, limit=limit or settings.graph.history_limit)
    except (CaroError, OSError, ValueError) as error:
        _fail(error)
        return
    console.print(_checkpoint_table(items))


@graph_app.command("state")
def graph_state(
    context: typer.Context,
    run_id: Annotated[str, typer.Argument()],
    checkpoint: Annotated[str | None, typer.Option(help="Specific checkpoint ID.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print full JSON-serializable state.")] = False,
) -> None:
    """Inspect latest or historical LangGraph state."""

    settings = _load(context)
    try:
        with CaroRuntime.open(settings) as runtime:
            run = runtime.runs.get(run_id)
            snapshot = runtime.controller.state(run.thread_id, checkpoint=checkpoint)
            state = coerce_state_mapping(snapshot)
    except (CaroError, OSError, TypeError, ValueError) as error:
        _fail(error)
        return
    if as_json:
        console.print_json(json.dumps(state, ensure_ascii=False, default=_json_default))
        return
    table = Table(title=f"LangGraph state · {checkpoint or 'latest'}")
    table.add_column("Channel")
    table.add_column("Type")
    table.add_column("Summary")
    for key, value in state.items():
        summary = f"{len(value)} item(s)" if isinstance(value, (list, tuple, dict)) else _short(value)
        table.add_row(str(key), type(value).__name__, summary)
    console.print(table)


def _parse_updates(values: list[str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ValueError(f"state update must use KEY=JSON: {value}")
        if key in updates:
            raise ValueError(f"duplicate state update: {key}")
        try:
            updates[key] = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON for state channel {key}: {error}") from error
    if not updates:
        raise ValueError("at least one --set KEY=JSON update is required")
    return updates


@graph_app.command("fork")
def graph_fork(
    context: typer.Context,
    run_id: Annotated[str, typer.Argument()],
    checkpoint: Annotated[str, typer.Argument()],
    values: Annotated[list[str], typer.Option("--set", help="Top-level state update as KEY=JSON.")],
    as_node: Annotated[str | None, typer.Option(help="Optional LangGraph as_node override.")] = None,
) -> None:
    """Create a branch checkpoint with modified state; continue it with ``caro resume``."""

    settings = _load(context)
    try:
        updates = _parse_updates(values)
        with CaroRuntime.open(settings) as runtime:
            parent = runtime.runs.get(run_id)
            fork = runtime.controller.fork(parent.thread_id, checkpoint, updates, as_node=as_node)
            child = runtime.runs.fork(
                parent.run_id,
                checkpoint_id=fork.checkpoint_id,
                config_hash=settings.fingerprint(),
            )
    except (CaroError, OSError, TypeError, ValueError) as error:
        _fail(error)
        return
    console.print(
        Panel(
            f"child run: {child.run_id}\nbranch checkpoint: {fork.checkpoint_id}\n"
            f"resume: caro resume {child.run_id}",
            title="LangGraph fork created",
        )
    )


@graph_app.command("replay")
def graph_replay(
    context: typer.Context,
    run_id: Annotated[str, typer.Argument()],
    checkpoint: Annotated[str, typer.Argument()],
    yes: Annotated[bool, typer.Option("--yes", help="Confirm replay side effects non-interactively.")] = False,
) -> None:
    """Replay all downstream graph nodes from a selected checkpoint."""

    if not yes:
        approved = typer.confirm(
            "Replay re-executes downstream LLM calls, tools, interrupts, and side effects. Continue?",
            default=False,
        )
        if not approved:
            raise typer.Abort()
    settings = _load(context)
    renderer = TerminalRenderer()
    try:
        with CaroRuntime.open(settings) as runtime:
            result = CaroRunner(
                runtime.assembly.graph,
                runtime.runs,
                settings,
                approval_handler=lambda request: _terminal_approval(
                    request,
                    registry=runtime.assembly.registry,
                    workspace=settings.workspace.root,
                ),
                event_sink=renderer,
                registry=runtime.assembly.registry,
            ).replay(run_id, checkpoint)
    except (CaroError, OSError, TypeError, ValueError) as error:
        renderer.finish()
        _fail(error)
        return
    renderer.finish()
    _run_summary(result)
