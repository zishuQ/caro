"""LangGraph workflow for governed memory consolidation."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, NotRequired, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from caro.memory.models import (
    ConsolidationOutcome,
    ConsolidationResult,
    MemoryAuthorization,
    MemoryItem,
    MemoryProposal,
)
from caro.memory.policy import MemoryPolicy
from caro.memory.store import MemoryStore


class ConsolidationState(TypedDict):
    """Serializable state passed through deterministic consolidation nodes."""

    proposal: dict[str, Any]
    authorization: str
    content_hash: NotRequired[str]
    approved: NotRequired[bool]
    reason: NotRequired[str]
    expires_at: NotRequired[Any]
    similar_memory_ids: NotRequired[list[str]]
    outcome: NotRequired[str]
    memory_id: NotRequired[str]


class MemoryConsolidator:
    """Compile and invoke the policy, retrieval, and commit graph."""

    def __init__(self, store: MemoryStore, policy: MemoryPolicy) -> None:
        self.store = store
        self.policy = policy
        builder = StateGraph(ConsolidationState)
        builder.add_node("encode", self._encode)
        builder.add_node("policy", self._policy)
        builder.add_node("audit_rejection", self._audit_rejection)
        builder.add_node("retrieve_existing", self._retrieve_existing)
        builder.add_node("commit", self._commit)
        builder.add_edge(START, "encode")
        builder.add_edge("encode", "policy")
        builder.add_conditional_edges(
            "policy",
            self._route_policy,
            {"approved": "retrieve_existing", "rejected": "audit_rejection"},
        )
        builder.add_edge("retrieve_existing", "commit")
        builder.add_edge("audit_rejection", END)
        builder.add_edge("commit", END)
        self.graph = builder.compile(name="caro_memory_consolidation")

    @staticmethod
    def _proposal(state: ConsolidationState) -> MemoryProposal:
        return MemoryProposal.model_validate(state["proposal"])

    @staticmethod
    def _authorization(state: ConsolidationState) -> MemoryAuthorization:
        return MemoryAuthorization(state["authorization"])

    def _encode(self, state: ConsolidationState) -> dict[str, str]:
        proposal = self._proposal(state)
        digest = hashlib.sha256(proposal.content.encode("utf-8")).hexdigest()
        return {"content_hash": digest}

    def _policy(self, state: ConsolidationState) -> dict[str, Any]:
        decision = self.policy.evaluate(self._proposal(state))
        return {
            "approved": decision.approved,
            "reason": decision.reason,
            "expires_at": decision.expires_at,
        }

    @staticmethod
    def _route_policy(state: ConsolidationState) -> Literal["approved", "rejected"]:
        return "approved" if state.get("approved") else "rejected"

    def _audit_rejection(self, state: ConsolidationState) -> dict[str, str]:
        proposal = self._proposal(state)
        reason = state.get("reason", "candidate rejected")
        self.store.record_operation(
            content_hash=state["content_hash"],
            kind=proposal.kind,
            scope=proposal.scope,
            outcome=ConsolidationOutcome.REJECTED.value,
            reason=reason,
            authorization=self._authorization(state),
            source_run_id=proposal.source_run_id,
            source_agent=proposal.source_agent,
            memory_id=None,
        )
        return {"outcome": ConsolidationOutcome.REJECTED.value}

    def _retrieve_existing(self, state: ConsolidationState) -> dict[str, list[str]]:
        proposal = self._proposal(state)
        matches = self.store.search(
            proposal.content,
            scopes=(proposal.scope,),
            kinds=(proposal.kind,),
            limit=3,
            record_access=False,
        )
        return {"similar_memory_ids": [match.item.memory_id for match in matches]}

    def _commit(self, state: ConsolidationState) -> dict[str, str]:
        proposal = self._proposal(state)
        authorization = self._authorization(state)
        metadata = dict(proposal.metadata)
        if proposal.evidence is not None:
            metadata["evidence"] = proposal.evidence
        result = self.store.put(
            kind=proposal.kind,
            scope=proposal.scope,
            content=proposal.content,
            importance=proposal.importance,
            confidence=proposal.confidence,
            source_run_id=proposal.source_run_id,
            source_agent=proposal.source_agent,
            metadata=metadata,
            expires_at=state.get("expires_at"),
            supersedes_id=proposal.supersedes_id,
            actor=authorization.value,
            operation_authorization=authorization,
        )
        outcome = ConsolidationOutcome.CREATED if result.created else ConsolidationOutcome.DUPLICATE
        reason = "memory committed" if result.created else "exact active memory already exists"
        return {"outcome": outcome.value, "reason": reason, "memory_id": result.item.memory_id}

    def consolidate(
        self,
        proposal: MemoryProposal,
        *,
        authorization: MemoryAuthorization,
        config: RunnableConfig | None = None,
    ) -> ConsolidationResult:
        """Run one authorized candidate through the compiled LangGraph."""

        raw = self.graph.invoke(
            {
                "proposal": proposal.model_dump(mode="python"),
                "authorization": authorization.value,
            },
            config=config,
        )
        state = cast("ConsolidationState", raw)
        outcome = ConsolidationOutcome(state["outcome"])
        item: MemoryItem | None = None
        memory_id = state.get("memory_id")
        if memory_id is not None:
            item = self.store.get(memory_id)
        return ConsolidationResult(
            outcome=outcome,
            item=item,
            reason=state.get("reason", ""),
            similar_memory_ids=tuple(state.get("similar_memory_ids", [])),
        )
