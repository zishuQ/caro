"""Nested DeepAgent-shaped subgraph interrupt persistence test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from caro.config import OrchestrationSettings
from caro.orchestration import compile_orchestration
from caro.state.checkpoints import CheckpointDatabase, graph_config


def approval_node(_: MessagesState) -> dict[str, list[AIMessage]]:
    response = interrupt(
        {
            "action_requests": [{"name": "write_file", "args": {"path": "README.md"}}],
            "review_configs": [
                {"action_name": "write_file", "allowed_decisions": ["approve", "reject"]}
            ],
        }
    )
    decision = response["decisions"][0]["type"]
    return {"messages": [AIMessage(content=f"worker decision: {decision}")]}


def build_interrupting_role() -> Any:
    builder = StateGraph(MessagesState)
    builder.add_node("approval", approval_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    return builder.compile()


def test_parent_checkpointer_persists_and_resumes_nested_interrupt(tmp_path: Path) -> None:
    settings = OrchestrationSettings(
        enabled=True,
        agents={"worker": {"description": "Modify files", "system_prompt": "Make the approved change."}},
        stages=[{"name": "work", "agents": ["worker"]}],
    )
    config = graph_config("nested-interrupt")
    with CheckpointDatabase(tmp_path / "checkpoints.sqlite3") as database:
        graph = compile_orchestration(
            settings,
            {"worker": build_interrupting_role()},
            {"worker": "primary"},
            database.saver,
        )

        graph.invoke({"messages": [HumanMessage(content="Update README.")]}, config)
        paused = graph.get_state(config)
        assert paused.interrupts
        assert paused.next == ("stage_00__worker",)

        result = graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), paused.config)

        assert result["messages"][-1].text == "worker decision: approve"
        assert not graph.get_state(config).interrupts
