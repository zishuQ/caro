"""Real LangGraph SQLite checkpoint, replay, and fork integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from caro.state.checkpoints import CheckpointDatabase, GraphController, checkpoint_id, graph_config


class NumberState(TypedDict):
    value: int


def build_graph(database: CheckpointDatabase):
    builder = StateGraph(NumberState)
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_node("double", lambda state: {"value": state["value"] * 2})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", "double")
    builder.add_edge("double", END)
    return builder.compile(checkpointer=database.saver)


def test_checkpoint_history_replay_fork_and_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "checkpoints.sqlite3"
    thread_id = "thread-1"
    with CheckpointDatabase(database_path) as database:
        graph = build_graph(database)
        assert graph.invoke({"value": 1}, graph_config(thread_id))["value"] == 4
        controller = GraphController(graph)
        history = controller.history(thread_id)
        assert history
        before_double = next(item for item in history if item.next_nodes == ("double",))

        replayed = controller.replay(thread_id, before_double.checkpoint_id)
        assert replayed["value"] == 4

        fork = controller.fork(thread_id, before_double.checkpoint_id, {"value": 10})
        assert fork.parent_checkpoint_id == before_double.checkpoint_id
        assert checkpoint_id(fork.config) == fork.checkpoint_id
        forked_result = graph.invoke(None, fork.config)
        assert forked_result["value"] == 20
        assert any(item.checkpoint_id == before_double.checkpoint_id for item in controller.history(thread_id))

    with CheckpointDatabase(database_path) as reopened:
        graph = build_graph(reopened)
        latest = GraphController(graph).state(thread_id)
        assert latest.values["value"] in {4, 20}

