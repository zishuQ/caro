"""Caro's public tool inventory and execution-governance adapter.

The registry is intentionally the only place where Caro maps model-visible
DeepAgents tools to risk, permissions, approval, timeout, and retry policy.
The implementation delegates execution details to :mod:`caro.tools.policy`
while retaining the small ``ToolSpec`` API used by the CLI and integrations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from deepagents import FilesystemPermission
from langchain.agents.middleware import InterruptOnConfig

from caro.tools.policy import (
    ApprovalPolicy,
    RetryPolicy,
    ToolAssessment,
    ToolPermission,
    ToolPolicy,
    ToolPolicyMiddleware,
    ToolRisk,
)
from caro.tools.policy import (
    ToolRegistry as ExecutionRegistry,
)

RiskLevel = ToolRisk
"""Compatibility alias for the canonical execution-risk enum."""


_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


def _execution_risk(value: RiskLevel) -> ToolRisk:
    return ToolRisk(value.value)


def _permissions_for(name: str) -> frozenset[ToolPermission]:
    if name in {"ls", "read_file", "glob", "grep", "memory_search", "list_async_tasks"}:
        return frozenset({ToolPermission.FILESYSTEM_READ})
    if name in {"write_file", "edit_file", "delete", "memory_archive"}:
        return frozenset({ToolPermission.FILESYSTEM_WRITE})
    if name == "execute":
        return frozenset({ToolPermission.SHELL})
    if name in {"task"}:
        return frozenset({ToolPermission.DELEGATE})
    if name in {"memory_propose", "update_goal", "compact_conversation"}:
        return frozenset({ToolPermission.STATE_WRITE})
    return frozenset()


def _policy_for_spec(spec: ToolSpec) -> ToolPolicy:
    """Convert the stable display spec into the execution policy model."""

    permission_set = spec.permissions if spec.permissions is not None else _permissions_for(spec.name)
    side_effect = bool(
        permission_set
        & {
            ToolPermission.FILESYSTEM_WRITE,
            ToolPermission.SHELL,
            ToolPermission.EXTERNAL_WRITE,
            ToolPermission.DELEGATE,
            ToolPermission.STATE_WRITE,
            ToolPermission.UNSPECIFIED,
        }
    )
    approval = (
        ApprovalPolicy.ALWAYS
        if side_effect or spec.risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}
        else ApprovalPolicy.NEVER
    )
    return ToolPolicy(
        name=spec.name,
        risk=_execution_risk(spec.risk),
        permissions=permission_set,
        approval=approval,
        reason=spec.description,
        idempotent=spec.idempotent,
        timeout_seconds=spec.timeout_seconds,
        retry=spec.retry,
    )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Governance metadata shown for one model-visible tool.

    ``timeout_seconds`` and ``retry`` are execution controls. Retries are
    validated by the underlying policy and therefore only apply to declared
    read-only, idempotent tools.
    """

    name: str
    description: str
    risk: RiskLevel
    source: str = "deepagents"
    timeout_seconds: float | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    idempotent: bool = False
    permissions: frozenset[ToolPermission] | None = None


class ToolRegistry:
    """Authoritative Caro tool inventory and LangGraph approval adapter."""

    def __init__(self, specs: tuple[ToolSpec, ...] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._policies = ExecutionRegistry()
        for spec in specs:
            self.register(spec)

    @classmethod
    def default(cls) -> ToolRegistry:
        """Return governance metadata for DeepAgents and Caro memory tools."""

        read_retry = RetryPolicy(
            max_attempts=2,
            backoff_seconds=0.1,
            retry_on_timeout=True,
            retry_on_error=True,
        )
        return cls(
            (
                ToolSpec(
                    "ls",
                    "List workspace paths without modifying them.",
                    RiskLevel.LOW,
                    timeout_seconds=30,
                    retry=read_retry,
                    idempotent=True,
                ),
                ToolSpec(
                    "read_file",
                    "Read a workspace file without modifying it.",
                    RiskLevel.LOW,
                    timeout_seconds=30,
                    retry=read_retry,
                    idempotent=True,
                ),
                ToolSpec(
                    "glob",
                    "Find workspace paths without modifying them.",
                    RiskLevel.LOW,
                    timeout_seconds=30,
                    retry=read_retry,
                    idempotent=True,
                ),
                ToolSpec(
                    "grep",
                    "Search workspace text without modifying it.",
                    RiskLevel.LOW,
                    timeout_seconds=30,
                    retry=read_retry,
                    idempotent=True,
                ),
                ToolSpec(
                    "write_file",
                    "Create or replace a workspace file; review the resulting diff before execution.",
                    RiskLevel.HIGH,
                ),
                ToolSpec(
                    "edit_file",
                    "Edit a workspace file; review the resulting diff before execution.",
                    RiskLevel.HIGH,
                ),
                ToolSpec(
                    "delete",
                    "Delete a workspace path and potentially unrecoverable data.",
                    RiskLevel.CRITICAL,
                ),
                ToolSpec(
                    "execute",
                    "Execute a guarded local shell command.",
                    RiskLevel.CRITICAL,
                    timeout_seconds=120,
                ),
                ToolSpec("task", "Delegate work to a Caro subagent.", RiskLevel.HIGH),
                ToolSpec("write_todos", "Update the agent task plan.", RiskLevel.LOW),
                ToolSpec(
                    "memory_search",
                    "Search governed structured project memory.",
                    RiskLevel.LOW,
                    source="caro",
                    timeout_seconds=15,
                    retry=read_retry,
                    idempotent=True,
                ),
                ToolSpec(
                    "memory_propose",
                    "Propose verified durable memory through the consolidation graph.",
                    RiskLevel.HIGH,
                    source="caro",
                ),
                ToolSpec(
                    "memory_archive",
                    "Soft-archive an accessible structured memory.",
                    RiskLevel.CRITICAL,
                    source="caro",
                ),
            )
        )

    @classmethod
    def with_builtins(cls, *, require_compact_approval: bool = True) -> ToolRegistry:
        """Compatibility constructor matching the lower-level policy registry."""

        del require_compact_approval
        return cls.default()

    def copy(self) -> ToolRegistry:
        """Return an independent registry with the same public specifications."""

        return ToolRegistry(self.all())

    def register(self, spec: ToolSpec) -> None:
        """Register a tool and reject ambiguous duplicate names."""

        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        policy = _policy_for_spec(spec)
        self._specs[spec.name] = spec
        self._policies.register(policy)

    def get(self, name: str) -> ToolSpec:
        """Return one registered tool or fail closed for unknown names."""

        try:
            return self._specs[name]
        except KeyError as error:
            raise KeyError(f"tool is not governed by Caro: {name}") from error

    def all(self) -> tuple[ToolSpec, ...]:
        """Return tools in registration order."""

        return tuple(self._specs.values())

    @property
    def policies(self) -> tuple[ToolPolicy, ...]:
        """Return execution policies in registration order."""

        return self._policies.policies

    def contains(self, name: str) -> bool:
        """Return whether a tool has an explicit Caro specification."""

        return name in self._specs

    def update(self, other: ToolRegistry, *, replace_existing: bool = True) -> None:
        """Merge another registry while preserving insertion order."""

        for spec in other.all():
            if spec.name in self._specs:
                if not replace_existing:
                    raise ValueError(f"tool already registered: {spec.name}")
                del self._specs[spec.name]
                self._policies.register(_policy_for_spec(spec), replace_existing=True)
                self._specs[spec.name] = spec
                continue
            self.register(spec)

    def register_mcp(self, name: str, *, read_only: bool) -> ToolSpec:
        """Register an MCP tool from a trusted read-only annotation."""

        spec = ToolSpec(
            name=name,
            description=(
                "MCP operation declared read-only by its server."
                if read_only
                else "MCP operation may mutate or access an external system."
            ),
            risk=RiskLevel.LOW if read_only else RiskLevel.HIGH,
            source="mcp",
            idempotent=read_only,
            permissions=frozenset(
                {
                    ToolPermission.EXTERNAL_READ
                    if read_only
                    else ToolPermission.EXTERNAL_WRITE
                }
            ),
        )
        if name in self._specs:
            self._specs.pop(name)
            self._policies.register(_policy_for_spec(spec), replace_existing=True)
            self._specs[name] = spec
        else:
            self.register(spec)
        return spec

    def interrupt_inventory(self) -> tuple[ToolSpec, ...]:
        """Return registered tools that intrinsically belong to HITL."""

        return tuple(spec for spec in self.all() if self.policy_for(spec.name).requires_approval)

    def policy_for(self, name: str) -> ToolPolicy:
        """Return an explicit policy, or a high-risk fail-closed policy."""

        return self._policies.policy_for(name)

    def assess(self, name: str, args: Mapping[str, object] | None = None) -> ToolAssessment:
        """Assess one invocation, including dynamic shell and Git hazards."""

        return self._policies.assess(name, args)

    def approval_description(
        self,
        name: str,
        args: Mapping[str, object] | None = None,
        *,
        action: str | None = None,
    ) -> str:
        """Render policy risk, reason, and preview metadata for HITL."""

        return self._policies.approval_description(name, args, action=action)

    @staticmethod
    def requires_approval(spec: ToolSpec, *, enabled: bool, minimum_risk: RiskLevel) -> bool:
        """Return whether a tool meets the configured approval threshold."""

        return enabled and _RISK_ORDER[spec.risk] >= _RISK_ORDER[minimum_risk]

    def filesystem_permissions(
        self,
        *,
        enabled: bool,
        minimum_risk: RiskLevel,
    ) -> list[FilesystemPermission]:
        """Adapt Caro write risk to DeepAgents' path-aware permission API."""

        write_specs = (self.get("write_file"), self.get("edit_file"), self.get("delete"))
        if any(self.requires_approval(spec, enabled=enabled, minimum_risk=minimum_risk) for spec in write_specs):
            return [FilesystemPermission(operations=["write"], paths=["/**"], mode="interrupt")]
        return []

    def interrupt_on(
        self,
        *,
        enabled: bool,
        minimum_risk: RiskLevel,
    ) -> dict[str, bool | InterruptOnConfig]:
        """Build LangGraph HITL configs for tools at the approval threshold."""

        result: dict[str, bool | InterruptOnConfig] = {}
        for spec in self._specs.values():
            if not self.requires_approval(spec, enabled=enabled, minimum_risk=minimum_risk):
                continue
            result[spec.name] = InterruptOnConfig(
                allowed_decisions=["approve", "edit", "reject"],
                description=(
                    f"Caro {spec.risk.value}-risk approval: {spec.description}"
                ),
            )
        return result

    def middleware(self) -> ToolPolicyMiddleware:
        """Return execution middleware bound to an immutable registry snapshot."""

        return ToolPolicyMiddleware(self._policies)
