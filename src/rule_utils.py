from __future__ import annotations

import math

import numpy as np


_FLOAT_FMT = "{:.6g}"


def normalise_rule_value(value):
    if isinstance(value, (np.floating, float)):
        value_float = float(value)
        if math.isfinite(value_float):
            value_round = round(value_float)
            if abs(value_float - value_round) < 1e-9:
                return float(value_round)
        return value_float
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def format_literal_value(value):
    value = normalise_rule_value(value)
    if isinstance(value, float):
        return _FLOAT_FMT.format(value)
    return str(value)


def resolve_rule_value(name, raw_value, value_names):
    mapping = (value_names or {}).get(name, {})
    for candidate in (
        raw_value,
        int(raw_value) if str(raw_value).isdigit() else None,
        str(int(raw_value)) if str(raw_value).isdigit() else None,
        str(raw_value),
    ):
        if candidate in mapping:
            return mapping[candidate]
    return raw_value


def condition_caption(condition, value_names, label=None):
    parts = []
    for name, op, raw_value in condition:
        value = resolve_rule_value(name, raw_value, value_names)
        value_text = _FLOAT_FMT.format(float(value)) if isinstance(value, float) else str(value)
        parts.append(f"{name} {op} {value_text}")
    base = " & ".join(parts) or "<empty>"
    return f"{base} → class {label}" if label is not None else base
