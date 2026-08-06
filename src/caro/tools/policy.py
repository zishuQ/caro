"""Central tool policy registry for execution and approval governance."""

from __future__ import annotations

import asyncio
import math
import re
import shlex
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from langchain_core.messages import ToolMessage
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command


class ToolRisk(StrEnum):
    """Risk level associated with a tool invocation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolPermission(StrEnum):
    """Capability a tool needs in order to execute."""

    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    SHELL = "shell"
    NETWORK = "network"
    EXTERNAL_READ = "external_read"
    EXTERNAL_WRITE = "external_write"
    DELEGATE = "delegate"
    STATE_WRITE = "state_write"
    UNSPECIFIED = "unspecified"


class ApprovalPolicy(StrEnum):
    """Whether a tool call belongs in the HITL approval inventory."""

    NEVER = "never"
    ALWAYS = "always"


_SIDE_EFFECT_PERMISSIONS = frozenset(
    {
        ToolPermission.FILESYSTEM_WRITE,
        ToolPermission.SHELL,
        ToolPermission.EXTERNAL_WRITE,
        ToolPermission.DELEGATE,
        ToolPermission.STATE_WRITE,
        ToolPermission.UNSPECIFIED,
    }
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded retry behavior for a tool call.

    `max_attempts` includes the initial attempt. Retries are intentionally
    opt-in and may only be attached to read-only, idempotent tools.
    """

    max_attempts: int = 1
    backoff_seconds: float = 0.0
    retry_on_timeout: bool = True
    retry_on_error: bool = False

    def __post_init__(self) -> None:
        """Validate retry bounds.

        Raises:
            ValueError: If attempts or backoff settings are invalid.
        """
        if self.max_attempts < 1:
            msg = "max_attempts must be at least 1"
            raise ValueError(msg)
        if not math.isfinite(self.backoff_seconds) or self.backoff_seconds < 0:
            msg = "backoff_seconds must be finite and non-negative"
            raise ValueError(msg)
        if self.max_attempts > 1 and not (self.retry_on_timeout or self.retry_on_error):
            msg = "a retrying policy must select timeout and/or error retries"
            raise ValueError(msg)

    def should_retry(self, exc: Exception) -> bool:
        """Return whether `exc` is retryable under this policy.

        Args:
            exc: Error raised by the tool handler.

        Returns:
            Whether another bounded attempt may be made.
        """
        if isinstance(exc, TimeoutError):
            return self.retry_on_timeout
        return self.retry_on_error


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Execution and approval policy for one canonical tool name."""

    name: str
    risk: ToolRisk
    permissions: frozenset[ToolPermission]
    approval: ApprovalPolicy
    reason: str
    idempotent: bool = False
    timeout_seconds: float | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        """Enforce fail-safe timeout and retry invariants.

        Raises:
            ValueError: If fields are invalid or unsafe retries are requested.
        """
        if not self.name.strip():
            msg = "tool policy name must not be empty"
            raise ValueError(msg)
        if not self.reason.strip():
            msg = f"tool policy {self.name!r} must explain its risk"
            raise ValueError(msg)
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            msg = "timeout_seconds must be finite and positive"
            raise ValueError(msg)
        if self.retry.max_attempts > 1 and (not self.idempotent or not self.read_only):
            msg = (
                f"tool policy {self.name!r} may retry only when the tool is "
                "read-only and idempotent"
            )
            raise ValueError(msg)

    @property
    def read_only(self) -> bool:
        """Whether the declared permissions exclude side effects."""
        return not bool(self.permissions & _SIDE_EFFECT_PERMISSIONS)

    @property
    def requires_approval(self) -> bool:
        """Whether calls belong in the HITL approval inventory."""
        return self.approval is ApprovalPolicy.ALWAYS


@dataclass(frozen=True, slots=True)
class ToolAssessment:
    """Invocation-specific risk assessment and approval metadata."""

    risk: ToolRisk
    reason: str
    metadata: tuple[tuple[str, str], ...] = ()


_RISK_ORDER = {
    ToolRisk.LOW: 0,
    ToolRisk.MEDIUM: 1,
    ToolRisk.HIGH: 2,
    ToolRisk.CRITICAL: 3,
}


def _higher_risk(left: ToolRisk, right: ToolRisk) -> ToolRisk:
    return left if _RISK_ORDER[left] >= _RISK_ORDER[right] else right


def _unknown_policy(name: str) -> ToolPolicy:
    return ToolPolicy(
        name=name,
        risk=ToolRisk.HIGH,
        permissions=frozenset({ToolPermission.UNSPECIFIED}),
        approval=ApprovalPolicy.ALWAYS,
        reason="No explicit policy is registered; approval is required by default.",
    )


def _builtin_policies(*, require_compact_approval: bool) -> tuple[ToolPolicy, ...]:
    read = frozenset({ToolPermission.FILESYSTEM_READ})
    write = frozenset({ToolPermission.FILESYSTEM_WRITE})
    external_read = frozenset({ToolPermission.EXTERNAL_READ})
    external_write = frozenset({ToolPermission.EXTERNAL_WRITE})
    policies = [
        *(
            ToolPolicy(
                name=name,
                risk=ToolRisk.LOW,
                permissions=read,
                approval=ApprovalPolicy.NEVER,
                reason="Reads repository data without modifying it.",
                idempotent=True,
            )
            for name in ("ls", "read_file", "glob", "grep")
        ),
        ToolPolicy(
            name="write_file",
            risk=ToolRisk.HIGH,
            permissions=write,
            approval=ApprovalPolicy.ALWAYS,
            reason="Creates or overwrites repository content.",
        ),
        ToolPolicy(
            name="edit_file",
            risk=ToolRisk.HIGH,
            permissions=write,
            approval=ApprovalPolicy.ALWAYS,
            reason="Changes existing repository content.",
        ),
        ToolPolicy(
            name="delete",
            risk=ToolRisk.CRITICAL,
            permissions=write,
            approval=ApprovalPolicy.ALWAYS,
            reason="Deletes a file or directory and may remove unrecoverable data.",
        ),
        ToolPolicy(
            name="execute",
            risk=ToolRisk.MEDIUM,
            permissions=frozenset({ToolPermission.SHELL}),
            approval=ApprovalPolicy.ALWAYS,
            reason="Runs a shell command in the tool environment.",
        ),
        ToolPolicy(
            name="web_search",
            risk=ToolRisk.MEDIUM,
            permissions=frozenset({ToolPermission.NETWORK}),
            approval=ApprovalPolicy.ALWAYS,
            reason=(
                "Sends a query to an external search service and may consume credits."
            ),
            idempotent=True,
        ),
        ToolPolicy(
            name="fetch_url",
            risk=ToolRisk.MEDIUM,
            permissions=frozenset({ToolPermission.NETWORK}),
            approval=ApprovalPolicy.ALWAYS,
            reason="Fetches content from an external network location.",
            idempotent=True,
        ),
        ToolPolicy(
            name="task",
            risk=ToolRisk.HIGH,
            permissions=frozenset({ToolPermission.DELEGATE}),
            approval=ApprovalPolicy.ALWAYS,
            reason="Delegates work to an agent that can use file and shell tools.",
        ),
        *(
            ToolPolicy(
                name=name,
                risk=ToolRisk.HIGH,
                permissions=external_write,
                approval=ApprovalPolicy.ALWAYS,
                reason=(
                    "Launches, changes, or cancels work in an external agent runtime."
                ),
            )
            for name in ("start_async_task", "update_async_task", "cancel_async_task")
        ),
        ToolPolicy(
            name="list_async_tasks",
            risk=ToolRisk.LOW,
            permissions=external_read,
            approval=ApprovalPolicy.NEVER,
            reason="Reads async task status without changing it.",
            idempotent=True,
        ),
        ToolPolicy(
            name="compact_conversation",
            risk=ToolRisk.MEDIUM,
            permissions=frozenset({ToolPermission.STATE_WRITE}),
            approval=(
                ApprovalPolicy.ALWAYS
                if require_compact_approval
                else ApprovalPolicy.NEVER
            ),
            reason=(
                "Rewrites active conversation context while preserving "
                "archived history."
            ),
        ),
        ToolPolicy(
            name="get_current_thread_id",
            risk=ToolRisk.LOW,
            permissions=frozenset(),
            approval=ApprovalPolicy.NEVER,
            reason="Reads the current thread identifier.",
            idempotent=True,
        ),
        ToolPolicy(
            name="ask_user",
            risk=ToolRisk.LOW,
            permissions=frozenset(),
            approval=ApprovalPolicy.NEVER,
            reason="Requests user input without performing an external side effect.",
        ),
        ToolPolicy(
            name="get_goal",
            risk=ToolRisk.LOW,
            permissions=frozenset(),
            approval=ApprovalPolicy.NEVER,
            reason="Reads goal state.",
            idempotent=True,
        ),
        ToolPolicy(
            name="get_rubric",
            risk=ToolRisk.LOW,
            permissions=frozenset(),
            approval=ApprovalPolicy.NEVER,
            reason="Reads rubric state.",
            idempotent=True,
        ),
        ToolPolicy(
            name="update_goal",
            risk=ToolRisk.MEDIUM,
            permissions=frozenset({ToolPermission.STATE_WRITE}),
            approval=ApprovalPolicy.NEVER,
            reason="Updates constrained local goal state.",
        ),
        ToolPolicy(
            name="js_eval",
            risk=ToolRisk.MEDIUM,
            permissions=frozenset(),
            approval=ApprovalPolicy.NEVER,
            reason="Runs code in the bounded QuickJS interpreter.",
        ),
    ]
    return tuple(policies)


_SHELL_SPLIT_RE = re.compile(r"(?:&&|\|\||[;|\n])")
_COMMAND_SUBSTITUTION_RE = re.compile(r"\$\(|`[^`]*`")
_REDIRECTION_RE = re.compile(r"(?:^|\s)(?:\d*>|>>|<)(?:\s|[^=])")
_DOWNLOAD_PIPE_SHELL_RE = re.compile(
    r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|)sh\b",
    re.IGNORECASE,
)
_GIT_OPERATIONS = frozenset({"commit", "push", "reset", "clean"})


def _shell_tokens(command: str) -> list[list[str]]:
    segments: list[list[str]] = []
    for segment in _SHELL_SPLIT_RE.split(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return []
        if tokens:
            segments.append(tokens)
    return segments


def _git_operation(tokens: Sequence[str]) -> tuple[str, int] | None:
    try:
        git_index = tokens.index("git")
    except ValueError:
        return None
    for index in range(git_index + 1, len(tokens)):
        token = tokens[index]
        if token in _GIT_OPERATIONS:
            return token, index
    return None


def _option_value(tokens: Sequence[str], *options: str) -> str | None:
    for index, token in enumerate(tokens):
        for option in options:
            if token == option and index + 1 < len(tokens):
                return tokens[index + 1]
            prefix = f"{option}="
            if token.startswith(prefix):
                return token.removeprefix(prefix)
    return None


def _git_operands(tokens: Sequence[str], operation_index: int) -> list[str]:
    operands: list[str] = []
    skip_next = False
    value_options = {"-m", "--message", "--repo", "--receive-pack"}
    for token in tokens[operation_index + 1 :]:
        if skip_next:
            skip_next = False
            continue
        if token in value_options:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        operands.append(token)
    return operands


def _git_assessment(
    tokens: Sequence[str], operation: str, index: int
) -> ToolAssessment:
    if operation == "commit":
        metadata: list[tuple[str, str]] = [
            ("Git operation", "commit"),
            ("Preview", "staged changes (`git diff --cached`)"),
        ]
        if message := _option_value(tokens, "-m", "--message"):
            metadata.append(("Commit message", message))
        return ToolAssessment(
            risk=ToolRisk.HIGH,
            reason="Creates a durable repository commit from the staged snapshot.",
            metadata=tuple(metadata),
        )

    if operation == "push":
        operands = _git_operands(tokens, index)
        force = any(
            token in {"-f", "--force", "--force-with-lease", "--force-if-includes"}
            or token.startswith(("--force=", "--force-with-lease="))
            or (
                token.startswith("-")
                and not token.startswith("--")
                and "f" in token[1:]
            )
            for token in tokens[index + 1 :]
        ) or any(operand.startswith("+") for operand in operands)
        metadata = [
            ("Git operation", "force push" if force else "push"),
            ("Remote", operands[0] if operands else "default upstream"),
            ("Refspec", operands[1] if len(operands) > 1 else "current branch"),
            ("Preview", "outgoing commits and target remote/ref"),
        ]
        return ToolAssessment(
            risk=ToolRisk.CRITICAL if force else ToolRisk.HIGH,
            reason=(
                "Force-pushing can rewrite shared remote history."
                if force
                else "Publishes local commits to a remote repository."
            ),
            metadata=tuple(metadata),
        )

    if operation == "reset":
        mode = next(
            (
                token
                for token in tokens[index + 1 :]
                if token in {"--soft", "--mixed", "--hard", "--merge", "--keep"}
            ),
            "--mixed",
        )
        operands = _git_operands(tokens, index)
        destructive = mode == "--hard"
        return ToolAssessment(
            risk=ToolRisk.CRITICAL if destructive else ToolRisk.HIGH,
            reason=(
                "A hard reset discards tracked working-tree and index changes."
                if destructive
                else "Moves repository state and may rewrite the index."
            ),
            metadata=(
                ("Git operation", "reset"),
                ("Mode", mode),
                ("Target", operands[0] if operands else "HEAD"),
                ("Preview", "current status and diff before reset"),
            ),
        )

    flags = " ".join(token for token in tokens[index + 1 :] if token.startswith("-"))
    forced = any("f" in token.removeprefix("--") for token in tokens[index + 1 :])
    return ToolAssessment(
        risk=ToolRisk.CRITICAL if forced else ToolRisk.HIGH,
        reason="Removes untracked repository files and directories.",
        metadata=(
            ("Git operation", "clean"),
            ("Flags", flags or "none"),
            ("Preview", "dry run (`git clean -ndx`)"),
        ),
    )


def assess_execute_command(command: object) -> ToolAssessment:
    """Classify shell and sensitive Git operations for approval display.

    Args:
        command: Raw `execute.command` argument.

    Returns:
        Invocation-specific risk, reason, and preview metadata.
    """
    if not isinstance(command, str) or not command.strip():
        return ToolAssessment(
            ToolRisk.HIGH,
            (
                "The shell command is missing or malformed and cannot be "
                "classified safely."
            ),
        )

    if _COMMAND_SUBSTITUTION_RE.search(command):
        return ToolAssessment(
            ToolRisk.CRITICAL,
            (
                "Command substitution hides additional shell execution from a "
                "simple review."
            ),
            (("Shell hazard", "command substitution"),),
        )
    if _DOWNLOAD_PIPE_SHELL_RE.search(command):
        return ToolAssessment(
            ToolRisk.CRITICAL,
            "Piping downloaded content into a shell executes unreviewed remote code.",
            (("Shell hazard", "download piped to shell"),),
        )
    if _REDIRECTION_RE.search(command):
        return ToolAssessment(
            ToolRisk.HIGH,
            (
                "Shell redirection can overwrite files outside a tool-specific "
                "diff preview."
            ),
            (("Shell hazard", "input/output redirection"),),
        )

    segments = _shell_tokens(command)
    if not segments:
        return ToolAssessment(
            ToolRisk.HIGH,
            "The shell command could not be parsed and requires conservative review.",
        )

    assessments: list[ToolAssessment] = []
    for tokens in segments:
        if git := _git_operation(tokens):
            assessments.append(_git_assessment(tokens, *git))
        executable = tokens[0].lower()
        flags = tokens[1:]
        if executable in {"sudo", "doas", "su"}:
            assessments.append(
                ToolAssessment(
                    ToolRisk.CRITICAL,
                    (
                        "Privilege escalation can modify host resources outside "
                        "the project."
                    ),
                    (("Shell hazard", "privilege escalation"),),
                )
            )
        elif executable in {"mkfs", "fdisk", "shutdown", "reboot", "dd"}:
            assessments.append(
                ToolAssessment(
                    ToolRisk.CRITICAL,
                    "The command can damage devices or disrupt the host system.",
                    (("Shell hazard", executable),),
                )
            )
        elif executable in {"rm", "rmdir"}:
            recursive = any(
                flag in {"-r", "-R", "--recursive", "-rf", "-fr", "-Rf", "-fR"}
                for flag in flags
            )
            assessments.append(
                ToolAssessment(
                    ToolRisk.CRITICAL if recursive else ToolRisk.HIGH,
                    (
                        "Recursive removal can delete an entire directory tree."
                        if recursive
                        else (
                            "Removes filesystem content without a structured file diff."
                        )
                    ),
                    (("Shell hazard", "recursive delete" if recursive else "delete"),),
                )
            )
        elif executable in {"bash", "sh", "zsh", "python", "python3", "node"} and any(
            flag in {"-c", "-e"} for flag in flags
        ):
            assessments.append(
                ToolAssessment(
                    ToolRisk.HIGH,
                    (
                        "An interpreter command can execute behavior embedded in "
                        "its arguments."
                    ),
                    (("Shell hazard", "inline interpreter"),),
                )
            )

    if not assessments:
        return ToolAssessment(
            ToolRisk.MEDIUM,
            "Runs a shell command in the tool environment.",
        )
    return max(assessments, key=lambda item: _RISK_ORDER[item.risk])


class ToolRegistry:
    """Mutable registry that is copied and frozen into each agent build."""

    def __init__(self, policies: Iterable[ToolPolicy] = ()) -> None:
        """Initialize the registry from unique policies.

        Args:
            policies: Policies to register in insertion order.
        """
        self._policies: dict[str, ToolPolicy] = {}
        for policy in policies:
            self.register(policy)

    @classmethod
    def with_builtins(cls, *, require_compact_approval: bool = True) -> ToolRegistry:
        """Return a registry populated with the built-in tool policies."""
        return cls(_builtin_policies(require_compact_approval=require_compact_approval))

    def copy(self) -> ToolRegistry:
        """Return an independent registry with the same policies."""
        return ToolRegistry(self._policies.values())

    def register(self, policy: ToolPolicy, *, replace_existing: bool = False) -> None:
        """Register one policy.

        Args:
            policy: Validated policy to register.
            replace_existing: Whether an existing name may be replaced.

        Raises:
            ValueError: If the name is already registered without replacement.
        """
        if policy.name in self._policies and not replace_existing:
            msg = f"tool policy already registered for {policy.name!r}"
            raise ValueError(msg)
        self._policies[policy.name] = policy

    def update(self, other: ToolRegistry, *, replace_existing: bool = True) -> None:
        """Merge policies from `other` in its registration order."""
        for policy in other.policies:
            self.register(policy, replace_existing=replace_existing)

    def contains(self, name: str) -> bool:
        """Return whether `name` has an explicit policy."""
        return name in self._policies

    def policy_for(self, name: str) -> ToolPolicy:
        """Return the explicit policy or a fail-closed unknown policy."""
        return self._policies.get(name, _unknown_policy(name))

    @property
    def policies(self) -> tuple[ToolPolicy, ...]:
        """Registered policies in stable insertion order."""
        return tuple(self._policies.values())

    def register_unknown(self, name: str) -> ToolPolicy:
        """Materialize a fail-closed policy for an otherwise unknown tool.

        Returns:
            Registered high-risk policy.
        """
        policy = _unknown_policy(name)
        self.register(policy, replace_existing=True)
        return policy

    def register_mcp(self, name: str, *, read_only: bool) -> ToolPolicy:
        """Register a policy derived from trusted MCP annotations.

        Returns:
            Registered MCP policy.
        """
        policy = ToolPolicy(
            name=name,
            risk=ToolRisk.LOW if read_only else ToolRisk.HIGH,
            permissions=frozenset(
                {
                    ToolPermission.EXTERNAL_READ
                    if read_only
                    else ToolPermission.EXTERNAL_WRITE
                }
            ),
            approval=(ApprovalPolicy.NEVER if read_only else ApprovalPolicy.ALWAYS),
            reason=(
                "MCP annotations coherently declare this operation read-only."
                if read_only
                else "The MCP action may mutate or access an external system."
            ),
            idempotent=read_only,
        )
        self.register(policy, replace_existing=True)
        return policy

    def interrupt_inventory(self) -> tuple[ToolPolicy, ...]:
        """Return policies that require HITL, in registration order."""
        return tuple(policy for policy in self.policies if policy.requires_approval)

    def assess(self, name: str, args: Mapping[str, object] | None) -> ToolAssessment:
        """Return invocation-specific risk without lowering the base policy risk."""
        policy = self.policy_for(name)
        if name != "execute":
            return ToolAssessment(policy.risk, policy.reason)
        command = args.get("command") if args is not None else None
        command_assessment = assess_execute_command(command)
        risk = _higher_risk(policy.risk, command_assessment.risk)
        reason = (
            command_assessment.reason
            if _RISK_ORDER[command_assessment.risk] >= _RISK_ORDER[policy.risk]
            else policy.reason
        )
        return replace(command_assessment, risk=risk, reason=reason)

    def approval_description(
        self,
        name: str,
        args: Mapping[str, object] | None,
        *,
        action: str | None = None,
    ) -> str:
        """Render the common risk block appended to an approval description.

        Returns:
            Human-readable action, risk reason, and preview metadata.
        """
        assessment = self.assess(name, args)
        lines = [] if not action else [action.rstrip(), ""]
        lines.extend(
            [
                f"Risk: {assessment.risk.value.upper()}",
                f"Risk reason: {assessment.reason}",
            ]
        )
        lines.extend(f"{label}: {value}" for label, value in assessment.metadata)
        return "\n".join(lines)


def tool_name(tool: object) -> str | None:
    """Return the canonical name LangChain will use for a tool-like object."""
    if isinstance(tool, Mapping):
        raw = tool.get("name")
        if isinstance(raw, str) and raw:
            return raw
        function = tool.get("function")
        if isinstance(function, Mapping):
            raw = function.get("name")
            if isinstance(raw, str) and raw:
                return raw
        raw = tool.get("type")
        if isinstance(raw, str) and raw and raw != "function":
            return raw
        return None
    raw = getattr(tool, "name", None)
    if isinstance(raw, str) and raw:
        return raw
    raw = getattr(tool, "__name__", None)
    return raw if isinstance(raw, str) and raw else None


class ToolPolicyMiddleware(AgentMiddleware):
    """Apply opt-in timeout and retry settings from a `ToolRegistry`."""

    def __init__(self, registry: ToolRegistry) -> None:
        """Initialize with an agent-build-local registry snapshot."""
        super().__init__()
        self._registry = registry.copy()

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Retry configured synchronous read-only calls.

        Synchronous handlers cannot be cancelled safely, so wall-clock timeout
        enforcement is limited to the async server path.

        Returns:
            Tool result from the first successful attempt.

        Raises:
            RuntimeError: If an impossible empty retry loop is reached.
        """
        name = request.tool_call["name"]
        if not self._registry.contains(name):
            raise PermissionError(f"tool is not allowed by Caro Tool Registry: {name}")
        policy = self._registry.policy_for(name)
        attempts = policy.retry.max_attempts
        for attempt in range(attempts):
            try:
                return handler(request)
            except Exception as exc:
                if attempt + 1 >= attempts or not policy.retry.should_retry(exc):
                    raise
                if policy.retry.backoff_seconds:
                    time.sleep(policy.retry.backoff_seconds)
        msg = "unreachable retry loop"
        raise RuntimeError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Enforce timeout and bounded retries on the async server path.

        Returns:
            Tool result from the first successful attempt.

        Raises:
            RuntimeError: If an impossible empty retry loop is reached.
        """
        name = request.tool_call["name"]
        if not self._registry.contains(name):
            raise PermissionError(f"tool is not allowed by Caro Tool Registry: {name}")
        policy = self._registry.policy_for(name)
        attempts = policy.retry.max_attempts
        for attempt in range(attempts):
            try:
                result = handler(request)
                if policy.timeout_seconds is None:
                    return await result
                return await asyncio.wait_for(result, timeout=policy.timeout_seconds)
            except Exception as exc:
                if attempt + 1 >= attempts or not policy.retry.should_retry(exc):
                    raise
                if policy.retry.backoff_seconds:
                    await asyncio.sleep(policy.retry.backoff_seconds)
        msg = "unreachable retry loop"
        raise RuntimeError(msg)
