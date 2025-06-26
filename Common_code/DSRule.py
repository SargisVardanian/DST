from typing import Callable, Any

class DSRule:
    """Tiny wrapper around a boolean predicate with a human-friendly caption.

    Parameters
    ----------
    predicate : Callable[[Any], bool]
        Function that takes a 1-D row (numpy array or sequence) and returns True/False.
    name : str, optional
        Caption to show in logs/exports. If omitted, ``repr(predicate)`` is used.
    """
    def __init__(self, predicate: Callable[..., bool], name: str = "") -> None:
        self.predicate = predicate
        self.name = name or repr(predicate)

    def __call__(self, x: Any) -> bool:
        """Return ``bool(predicate(x))`` for convenience."""
        return bool(self.predicate(x))

    def __str__(self) -> str:
        return self.name

    # Backward compatibility: some code refers to `rule.caption`
    @property
    def caption(self) -> str:
        return self.name

    @caption.setter
    def caption(self, value: str) -> None:
        self.name = value
