"""Configured delegate execution and nested approval persistence tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from caro.agent import assemble_agent
from caro.config import CaroSettings
from caro.state.checkpoints import CheckpointDatabase, graph_config


class BindableFakeMessagesModel(FakeMessagesListChatModel):
    """Cycle scripted messages while advertising the tool-binding API."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> BindableFakeMessagesModel:
        """Accept tool schemas and keep the scripted model unchanged."""

        del tools, kwargs
        return self


def test_delegate_uses_cheap_model_and_resumes_nested_approval(tmp_path: Path) -> None:
    models = {
        "openai:gpt-5.6-sol": BindableFakeMessagesModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "Create the requested file.",
                                "subagent_type": "cheap_writer",
                            },
                            "id": "task-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Delegation complete."),
            ]
        ),
        "deepseek:deepseek-v4-flash": BindableFakeMessagesModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "delegated.txt", "content": "delegated"},
                            "id": "write-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="File created."),
                AIMessage(content="File created."),
            ]
        ),
    }

    def model_factory(definition: Any) -> BindableFakeMessagesModel:
        return models[definition.model]

    settings = CaroSettings(
        workspace={"root": tmp_path},
        memory={"enabled": False},
        routing={
            "default": "high",
            "models": {
                "high": {"model": "openai:gpt-5.6-sol"},
                "cheap": {"model": "deepseek:deepseek-v4-flash"},
            },
        },
        orchestration={
            "enabled": True,
            "allow_nested_delegation": True,
            "agents": {
                "architect": {
                    "description": "Design the change.",
                    "system_prompt": "Analyze the architecture.",
                    "model": "high",
                    "delegates": [
                        {
                            "name": "cheap_writer",
                            "description": "Create a focused file change.",
                            "system_prompt": "Make only the requested file change.",
                            "model": "cheap",
                        }
                    ],
                }
            },
            "stages": [{"name": "implementation", "agents": ["architect"]}],
        },
    )

    with CheckpointDatabase(tmp_path / "checkpoints.sqlite3") as database:
        assembly = assemble_agent(settings, database.saver, model_factory=model_factory)
        config = graph_config("delegate-interrupt")

        assembly.graph.invoke(
            {"messages": [HumanMessage(content="Create the delegated file.")]},
            config,
        )
        first_pause = assembly.graph.get_state(config)
        assert first_pause.interrupts[0].value["action_requests"][0]["name"] == "task"

        assembly.graph.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            first_pause.config,
        )
        second_pause = assembly.graph.get_state(config)
        assert second_pause.interrupts[0].value["action_requests"][0]["name"] == "write_file"

        result = assembly.graph.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            second_pause.config,
        )

    assert result["messages"][-1].text == "Delegation complete."
    assert models["deepseek:deepseek-v4-flash"].i == 2
    assert (tmp_path / "delegated.txt").read_text(encoding="utf-8") == "delegated"
