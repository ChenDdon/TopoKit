"""Shared exceptions; resource refusals never change the mathematical object."""

class ResourceLimitError(RuntimeError):
    """A requested computation exceeds an explicit resource budget."""
