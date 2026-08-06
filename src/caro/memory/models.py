"""Typed records shared by Caro's structured-memory components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryKind(StrEnum):
    """Durable memory classes tailored to coding-agent work."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStatus(StrEnum):
    """Lifecycle state for a structured memory item."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class MemoryAuthorization(StrEnum):
    """Explicit authority under which a consolidation may commit."""

    OPERATOR = "operator"
    GOVERNED_TOOL = "governed_tool"


class ConsolidationOutcome(StrEnum):
    """Terminal outcome from the memory consolidation graph."""

    CREATED = "created"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class MemoryProposal(BaseModel):
    """Validated candidate submitted to the governed consolidation graph."""

    model_config = ConfigDict(extra="forbid")

    kind: MemoryKind
    scope: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=20_000)
    importance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: str | None = Field(default=None, max_length=4000)
    source_run_id: str | None = Field(default=None, max_length=200)
    source_agent: str | None = Field(default=None, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime | None = None
    supersedes_id: str | None = Field(default=None, max_length=100)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        """Normalize whitespace before policy checks and hashing."""

        return " ".join(value.split())

    @field_validator("evidence")
    @classmethod
    def normalize_evidence(cls, value: str | None) -> str | None:
        """Treat blank evidence as absent."""

        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One provenance-rich durable memory record."""

    memory_id: str
    kind: MemoryKind
    scope: str
    content: str
    importance: float
    confidence: float
    source_run_id: str | None
    source_agent: str | None
    status: MemoryStatus
    content_hash: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None
    expires_at: datetime | None
    access_count: int


@dataclass(frozen=True, slots=True)
class MemoryPutResult:
    """Idempotent write result."""

    item: MemoryItem
    created: bool


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """One ranked retrieval result."""

    item: MemoryItem
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """Append-only memory audit event."""

    event_id: int
    memory_id: str
    event_type: str
    actor: str | None
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """Aggregate structured-memory counts."""

    active: int
    archived: int
    by_kind: dict[MemoryKind, int]
    by_scope: dict[str, int]


@dataclass(frozen=True, slots=True)
class MemoryOperation:
    """Content-free audit record for a consolidation attempt."""

    operation_id: int
    content_hash: str
    kind: MemoryKind
    scope: str
    outcome: str
    reason: str
    authorization: MemoryAuthorization
    source_run_id: str | None
    source_agent: str | None
    memory_id: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Public result returned by the LangGraph consolidation workflow."""

    outcome: ConsolidationOutcome
    item: MemoryItem | None
    reason: str
    similar_memory_ids: tuple[str, ...]
