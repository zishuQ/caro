"""Workspace and dangerous-shell interception tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from caro.errors import UnsafeCommandError
from caro.workspace import BubblewrapShellBackend, GuardedShellBackend, ShellGuard, filtered_environment


@pytest.mark.parametrize(
    "command",
    [
        "sudo apt update",
        "rm -rf /",
        "rm -rf .",
        "cd ../outside",
        "cat $HOME/.ssh/id_rsa",
        "echo $(cat secret)",
        "git push origin main",
        "git -C/tmp/outside status",
        "git -c core.sshCommand='ssh -o ProxyCommand=evil' push origin main",
        "git -c alias.publish='!sh -c evil' publish",
        "git config --global alias.publish '!sh -c evil'",
        "git reset --hard HEAD~1",
        "dd if=/dev/zero of=/dev/sda",
        "cat /etc/passwd",
    ],
)
def test_dangerous_or_escaping_commands_are_blocked(tmp_path: Path, command: str) -> None:
    inspection = ShellGuard(tmp_path).inspect(command)
    assert inspection.allowed is False
    assert inspection.reason


@pytest.mark.parametrize("command", ["pytest -q", "git status", "rm -rf build", "python -m compileall src"])
def test_normal_workspace_commands_are_allowed(tmp_path: Path, command: str) -> None:
    assert ShellGuard(tmp_path).inspect(command).allowed is True


def test_assert_safe_raises_domain_error(tmp_path: Path) -> None:
    with pytest.raises(UnsafeCommandError):
        ShellGuard(tmp_path).assert_safe("reboot")


def test_backend_executes_in_workspace_and_blocks_before_subprocess(tmp_path: Path) -> None:
    backend = GuardedShellBackend(tmp_path)
    allowed = backend.execute("pwd")
    blocked = backend.execute("rm -rf /")
    assert allowed.exit_code == 0
    assert allowed.output.strip() == str(tmp_path.resolve())
    assert blocked.exit_code == 126
    assert "Blocked by Caro" in blocked.output


def test_virtual_filesystem_rejects_parent_and_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    backend = GuardedShellBackend(workspace)

    with pytest.raises(ValueError, match="traversal"):
        backend.read("../outside/secret.txt")
    with pytest.raises(ValueError, match="outside root"):
        backend.read("/escape/secret.txt")


def test_environment_is_allowlisted() -> None:
    result = filtered_environment(["PATH"], environ={"PATH": "/bin", "SECRET": "do-not-copy"})
    assert result == {"PATH": "/bin"}


def test_backend_always_removes_langfuse_credentials_from_explicit_environment(
    tmp_path: Path,
) -> None:
    backend = GuardedShellBackend(
        tmp_path,
        allowed_environment=["PATH", "LANGFUSE_SECRET_KEY"],
    )
    assert "LANGFUSE_SECRET_KEY" not in backend._env


def test_backend_removes_langfuse_credentials_when_inheriting_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "should-not-reach-shell")
    backend = GuardedShellBackend(tmp_path, inherit_environment=True)
    assert "LANGFUSE_SECRET_KEY" not in backend._env


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
def test_bubblewrap_backend_hides_host_paths_and_writes_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "caro-bwrap-outside-secret"
    outside.write_text("host-only", encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from pathlib import Path\n"
        f"print(Path({str(outside)!r}).exists())\n"
        "print(Path('/etc/passwd').exists())\n"
        "Path('inside.txt').write_text('sandboxed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    backend = BubblewrapShellBackend(tmp_path)
    result = backend.execute("python probe.py")
    assert result.exit_code == 0
    assert "False" in result.output
    assert result.output.count("False") >= 2
    assert (tmp_path / "inside.txt").read_text(encoding="utf-8") == "sandboxed"
