"""Caro domain exceptions."""


class CaroError(Exception):
    """Base exception for expected Caro failures."""


class ProviderInitializationError(CaroError):
    """A configured model provider could not be initialized."""


class ProjectMemoryError(CaroError):
    """Project memory is outside policy or cannot be safely loaded."""


class RunNotFoundError(CaroError):
    """A requested persisted run does not exist."""


class UnsafeCommandError(CaroError):
    """A shell command violates Caro's non-bypassable deny rules."""
