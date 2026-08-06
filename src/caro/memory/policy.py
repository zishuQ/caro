"""Deterministic safety and durability policy for memory candidates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from caro.memory.models import MemoryKind, MemoryProposal

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])-[-A-Za-z0-9_]{12,}\b", re.IGNORECASE),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(
        r"['\"]?\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\b['\"]?"
        r"\s*[:=]\s*['\"]?[^\s'\"]{8,}",
        re.IGNORECASE,
    ),
)
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_TRANSIENT_PATTERN = re.compile(
    r"\b(?:today|tonight|right now|currently|this one task|this session)\b|今天|今晚|当前临时|这次任务",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Result from deterministic memory-policy evaluation."""

    approved: bool
    reason: str
    expires_at: datetime | None


class MemoryPolicy:
    """Reject unsafe, transient, unverified, or oversized durable memory."""

    def __init__(self, *, minimum_confidence: float, max_content_chars: int, episodic_ttl_days: int) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum memory confidence must be between 0 and 1")
        if max_content_chars < 100:
            raise ValueError("maximum memory content must be at least 100 characters")
        if episodic_ttl_days < 1:
            raise ValueError("episodic TTL must be positive")
        self.minimum_confidence = minimum_confidence
        self.max_content_chars = max_content_chars
        self.episodic_ttl_days = episodic_ttl_days

    def evaluate(self, proposal: MemoryProposal, *, now: datetime | None = None) -> PolicyDecision:
        """Evaluate a proposal without performing any persistence."""

        current = (now or datetime.now(UTC)).astimezone(UTC)
        content = proposal.content
        inspection = "\n".join(
            part
            for part in (
                content,
                proposal.evidence,
                json.dumps(proposal.metadata, ensure_ascii=False, sort_keys=True),
            )
            if part
        )
        if len(content) > self.max_content_chars:
            return PolicyDecision(False, "candidate exceeds the configured durable-memory size limit", None)
        if any(pattern.search(inspection) for pattern in _SECRET_PATTERNS):
            return PolicyDecision(False, "candidate contains credential-like secret material", None)
        if _EMAIL_PATTERN.search(inspection):
            return PolicyDecision(False, "candidate contains a personal identifier", None)
        if _TRANSIENT_PATTERN.search(content):
            return PolicyDecision(False, "candidate describes transient context", None)
        if proposal.confidence < self.minimum_confidence:
            return PolicyDecision(False, "candidate confidence is below the configured threshold", None)
        if proposal.evidence is None and proposal.source_run_id is None:
            return PolicyDecision(False, "candidate has no evidence or source run", None)
        expires_at = proposal.expires_at
        if proposal.kind is MemoryKind.EPISODIC and expires_at is None:
            expires_at = current + timedelta(days=self.episodic_ttl_days)
        if expires_at is not None:
            if expires_at.tzinfo is None:
                return PolicyDecision(False, "candidate expiry must be timezone-aware", None)
            expires_at = expires_at.astimezone(UTC)
            if expires_at <= current:
                return PolicyDecision(False, "candidate expiry is not in the future", None)
        return PolicyDecision(True, "candidate passed deterministic memory policy", expires_at)
