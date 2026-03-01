from __future__ import annotations

import math
from typing import Any


def as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def fmt_metric(value: Any) -> str:
    parsed = as_optional_float(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.3f}"
