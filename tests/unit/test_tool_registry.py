"""Tool Registry risk and approval tests."""

from __future__ import annotations

import pytest

from caro.tools.registry import RiskLevel, ToolRegistry, ToolSpec


def test_default_registry_covers_deepagents_harness_tools() -> None:
    registry = ToolRegistry.default()
    names = {spec.name for spec in registry.all()}
    assert {"ls", "read_file", "write_file", "edit_file", "execute", "task"}.issubset(names)
    assert registry.get("execute").risk is RiskLevel.CRITICAL


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry((ToolSpec("x", "first", RiskLevel.LOW),))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ToolSpec("x", "second", RiskLevel.HIGH))


def test_high_threshold_interrupts_mutations_but_not_reads() -> None:
    registry = ToolRegistry.default()
    interrupt_on = registry.interrupt_on(enabled=True, minimum_risk=RiskLevel.HIGH)
    permissions = registry.filesystem_permissions(enabled=True, minimum_risk=RiskLevel.HIGH)
    assert "execute" in interrupt_on
    assert "write_file" in interrupt_on
    assert "memory_propose" in interrupt_on
    assert "memory_archive" in interrupt_on
    assert "memory_search" not in interrupt_on
    assert "read_file" not in interrupt_on
    assert len(permissions) == 1
    assert permissions[0].mode == "interrupt"


def test_approvals_can_be_disabled_without_disabling_shell_guard() -> None:
    registry = ToolRegistry.default()
    assert registry.interrupt_on(enabled=False, minimum_risk=RiskLevel.LOW) == {}
    assert registry.filesystem_permissions(enabled=False, minimum_risk=RiskLevel.LOW) == []


def test_default_registry_exposes_bounded_read_controls_and_edit_hitl() -> None:
    registry = ToolRegistry.default()
    read = registry.get("read_file")

    assert read.timeout_seconds == 30
    assert read.retry.max_attempts == 2
    assert "edit" in registry.interrupt_on(enabled=True, minimum_risk=RiskLevel.HIGH)["write_file"][
        "allowed_decisions"
    ]


def test_execute_assessment_escalates_force_push_and_unknown_tools_fail_closed() -> None:
    registry = ToolRegistry.default()

    assessment = registry.assess("execute", {"command": "git push --force origin main"})

    assert assessment.risk.value == RiskLevel.CRITICAL.value
    assert "force" in assessment.reason.lower()
    assert registry.policy_for("unregistered_tool").requires_approval is True
