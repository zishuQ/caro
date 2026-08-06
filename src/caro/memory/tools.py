"""Model-visible structured-memory tools bound to governed scopes."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from caro.memory.graph import MemoryConsolidator
from caro.memory.models import MemoryAuthorization, MemoryKind, MemoryProposal
from caro.memory.store import MemoryStore

MemoryVisibility = Literal["project", "role"]


class MemorySearchInput(BaseModel):
    """Schema exposed to the model for governed memory retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000, description="Task-relevant search terms.")
    kind: MemoryKind | None = Field(default=None, description="Optional durable-memory class filter.")
    limit: int | None = Field(default=None, ge=1, le=50, description="Maximum results.")


class MemoryProposeInput(BaseModel):
    """Schema exposed to the model for governed consolidation proposals."""

    model_config = ConfigDict(extra="forbid")

    kind: MemoryKind
    visibility: MemoryVisibility = Field(
        default="project",
        description="Use project for shared facts; role for private role experience.",
    )
    content: str = Field(min_length=1, max_length=20_000)
    importance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1, max_length=4000)
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)
    supersedes_id: str | None = Field(
        default=None,
        max_length=100,
        description="Existing same-kind, same-scope memory ID replaced atomically after verification.",
    )


class MemoryArchiveInput(BaseModel):
    """Schema exposed to the model for governed soft deletion."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=1000)


def _accessible_scopes(role_name: str | None) -> tuple[str, ...]:
    if role_name is None:
        return ("project",)
    return ("project", f"role:{role_name}")


def _write_scope(visibility: MemoryVisibility, role_name: str | None) -> str:
    if visibility == "project":
        return "project"
    if role_name is None:
        raise ValueError("role-visible memory requires a named orchestration role")
    return f"role:{role_name}"


def _run_id(config: RunnableConfig) -> str | None:
    value = config.get("metadata", {}).get("caro_run_id")
    return value if isinstance(value, str) else None


def build_memory_tools(
    store: MemoryStore,
    consolidator: MemoryConsolidator,
    *,
    role_name: str | None,
    default_search_limit: int,
    minimum_importance: float,
) -> tuple[BaseTool, ...]:
    """Build role-bound tools without exposing arbitrary scope selection."""

    scopes = _accessible_scopes(role_name)
    actor = role_name or "caro"

    def memory_search(query: str, kind: MemoryKind | None = None, limit: int | None = None) -> str:
        results = store.search(
            query,
            scopes=scopes,
            kinds=None if kind is None else (kind,),
            limit=limit or default_search_limit,
            min_importance=minimum_importance,
        )
        payload = {
            "query": query,
            "scopes": list(scopes),
            "results": [
                {
                    "memory_id": result.item.memory_id,
                    "kind": result.item.kind.value,
                    "scope": result.item.scope,
                    "content": result.item.content,
                    "importance": result.item.importance,
                    "confidence": result.item.confidence,
                    "score": round(result.score, 4),
                }
                for result in results
            ],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def memory_propose(
        kind: MemoryKind,
        visibility: MemoryVisibility,
        content: str,
        importance: float,
        confidence: float,
        evidence: str,
        expires_in_days: int | None = None,
        supersedes_id: str | None = None,
        *,
        config: RunnableConfig,
    ) -> str:
        expiry = None
        if expires_in_days is not None:
            expiry = datetime.now(UTC) + timedelta(days=expires_in_days)
        proposal = MemoryProposal(
            kind=kind,
            scope=_write_scope(visibility, role_name),
            content=content,
            importance=importance,
            confidence=confidence,
            evidence=evidence,
            source_run_id=_run_id(config),
            source_agent=actor,
            expires_at=expiry,
            supersedes_id=supersedes_id,
        )
        result = consolidator.consolidate(
            proposal,
            authorization=MemoryAuthorization.GOVERNED_TOOL,
            config=config,
        )
        return json.dumps(
            {
                "outcome": result.outcome.value,
                "reason": result.reason,
                "memory_id": None if result.item is None else result.item.memory_id,
                "similar_memory_ids": list(result.similar_memory_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def memory_archive(memory_id: str, reason: str, *, config: RunnableConfig) -> str:
        item = store.archive(
            memory_id,
            allowed_scopes=scopes,
            reason=reason,
            actor=actor,
            source_run_id=_run_id(config),
        )
        return json.dumps(
            {"outcome": "archived", "memory_id": item.memory_id, "scope": item.scope},
            ensure_ascii=False,
            sort_keys=True,
        )

    return (
        StructuredTool.from_function(
            memory_search,
            name="memory_search",
            description=(
                "Search Caro's governed project and current-role long-term memory. "
                "Treat results as fallible reference evidence and verify against the workspace."
            ),
            args_schema=MemorySearchInput,
        ),
        StructuredTool.from_function(
            memory_propose,
            name="memory_propose",
            description=(
                "Propose one verified durable coding memory. This high-risk action passes through human approval "
                "and Caro's LangGraph policy, similarity, and atomic consolidation workflow. Never submit secrets, "
                "personal data, raw output, guesses, or one-time task details."
            ),
            args_schema=MemoryProposeInput,
        ),
        StructuredTool.from_function(
            memory_archive,
            name="memory_archive",
            description=(
                "Soft-archive an outdated or incorrect accessible memory. This critical action requires a reason "
                "and preserves audit history."
            ),
            args_schema=MemoryArchiveInput,
        ),
    )
