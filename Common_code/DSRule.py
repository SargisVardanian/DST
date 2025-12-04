"""
Helper structure for storing rule specifications and diagnostics.

This module provides the DSRule dataclass for representing classification rules
with their conditions, labels, and statistical properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

Literal = Tuple[str, str, Any]
Condition = Tuple[Literal, ...]
_EPS = 1e-9



def _literal_mask(column: np.ndarray, op: str, value: Any) -> np.ndarray:
    """
    Evaluate a literal condition against a data column.
    
    Args:
        column (ndarray): Data column to evaluate
        op (str): Comparison operator ('==', '<', '>')
        value: Value to compare against
        
    Returns:
        ndarray: Boolean mask indicating which rows satisfy the condition
    """
    if op == "==":
        if np.issubdtype(column.dtype, np.number):
            # If column is numeric but value is string, we can't directly compare
            # unless we try to cast or if the user meant to compare against decoded values.
            # Here we assume strict type matching or float comparison.
            try:
                val_float = float(value)
                return np.isfinite(column) & np.isclose(column, val_float, atol=_EPS)
            except (ValueError, TypeError):
                # Value is not numeric, comparison impossible
                return np.zeros(column.shape, dtype=bool)
        # String/object column: direct equality
        return column == value
    
    # Inequality operators require numeric comparison
    try:
        val_float = float(value)
    except (ValueError, TypeError):
        # Cannot compare non-numeric values with inequality operators
        return np.zeros(column.shape, dtype=bool)

    if op == "<":
        return np.isfinite(column) & (column < val_float)
    if op == ">":
        return np.isfinite(column) & (column > val_float)
    raise ValueError(f"Unsupported operator {op}")


@dataclass
class DSRule:
    """
    Represents a classification rule with conditions and metadata.
    
    Attributes:
        specs (Iterable[Literal]): Sequence of (feature_name, operator, value) tuples
        label (int, optional): Target class label for this rule
        caption (str): Human-readable description of the rule
        stats (dict, optional): Statistical properties (precision, recall, etc.)
    """
    specs: Iterable[Literal]
    label: Optional[int] = None
    caption: str = ""
    stats: Optional[dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize specs and generate default caption if needed."""
        self.specs = tuple((str(name), str(op), value) for name, op, value in self.specs)
        if self.caption == "" and self.label is not None:
            text = " & ".join(f"{name} {op} {val}" for name, op, val in self.specs)
            self.caption = f"Class {self.label}: {text}" if text else f"Class {self.label}"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """
        Convert rule to dictionary format.
        
        Returns:
            dict: Dictionary with specs, label, caption, and stats
        """
        return {
            "specs": tuple(self.specs),
            "label": self.label,
            "caption": self.caption,
            "stats": None if self.stats is None else dict(self.stats),
        }

    def mask(self, X: np.ndarray, feature_to_idx: Mapping[str, int]) -> np.ndarray:
        """
        Compute boolean mask indicating which samples satisfy this rule.
        
        Args:
            X (ndarray): Data matrix (N, D)
            feature_to_idx (dict): Mapping from feature names to column indices
            
        Returns:
            ndarray: Boolean mask (N,) indicating which samples match the rule
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array for rule masking")
        mask = np.ones(X.shape[0], dtype=bool)
        for name, op, value in self.specs:
            idx = feature_to_idx.get(name)
            if idx is None:
                raise KeyError(f"Unknown feature '{name}' in rule")
            mask &= _literal_mask(X[:, idx], op, value)
        return mask

    def applies(self, sample: Sequence[Any], feature_to_idx: Mapping[str, int]) -> bool:
        """
        Check if this rule applies to a single sample.
        
        Args:
            sample (Sequence): Single data sample (1D)
            feature_to_idx (dict): Mapping from feature names to indices
            
        Returns:
            bool: True if all conditions in this rule are satisfied
        """
        arr = np.asarray(sample)
        if arr.ndim != 1:
            raise ValueError("sample must be a 1D sequence")
        for name, op, value in self.specs:
            idx = feature_to_idx.get(name)
            if idx is None:
                raise KeyError(f"Unknown feature '{name}' in rule")
            if not _literal_mask(arr[idx : idx + 1], op, value)[0]:
                return False
        return True


__all__ = ["DSRule", "Literal", "Condition"]
