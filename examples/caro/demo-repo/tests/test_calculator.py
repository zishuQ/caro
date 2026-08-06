"""Calculator behavior expected by the demo."""

from calculator import add, divide


def test_add() -> None:
    assert add(2, 3) == 5


def test_divide_preserves_fraction() -> None:
    assert divide(5, 2) == 2.5

