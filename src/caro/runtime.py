"""Own Caro's SQLite and compiled-agent resources."""

from __future__ import annotations

from dataclasses import dataclass

from caro.agent import AgentAssembly, assemble_agent
from caro.config import CaroSettings, apply_network_environment
from caro.models.pool import ModelFactory
from caro.state.checkpoints import CheckpointDatabase, GraphController
from caro.state.runs import RunStore


@dataclass(slots=True)
class CaroRuntime:
    """Process-local resource owner for CLI commands."""

    settings: CaroSettings
    runs: RunStore
    checkpoints: CheckpointDatabase
    assembly: AgentAssembly

    @classmethod
    def open(cls, settings: CaroSettings, *, model_factory: ModelFactory | None = None) -> CaroRuntime:
        """Open databases and compile the governed DeepAgents graph."""

        apply_network_environment(settings.network)
        settings.state.directory.mkdir(parents=True, exist_ok=True)
        runs = RunStore(settings.state.directory / settings.state.runs_database)
        checkpoints = CheckpointDatabase(settings.state.directory / settings.state.checkpoints_database)
        try:
            assembly = assemble_agent(settings, checkpoints.saver, model_factory=model_factory)
        except Exception:
            checkpoints.close()
            raise
        return cls(settings=settings, runs=runs, checkpoints=checkpoints, assembly=assembly)

    @property
    def controller(self) -> GraphController:
        """Return LangGraph recovery operations for this compiled graph."""

        return GraphController(self.assembly.graph)

    def close(self) -> None:
        """Close SQLite checkpoint resources."""

        self.checkpoints.close()

    def __enter__(self) -> CaroRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
