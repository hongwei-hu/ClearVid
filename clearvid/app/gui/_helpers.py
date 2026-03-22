"""Shared GUI helper functions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def populate_combo(
    combo: Any, labels: dict[Any, str], values: Iterable[Any], default_value: Any
) -> None:
    """Fill a QComboBox with labeled enum items."""
    for item in values:
        combo.addItem(labels[item], item)
    combo.setCurrentText(labels[default_value])


def coerce_enum(enum_type: type, value: Any, default: Any) -> Any:
    """Safely convert a value to an enum member."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            return default
    return default


def set_combo_by_value(combo: Any, value: str) -> None:
    """Set a QComboBox to the item whose data (enum value) matches *value*."""
    for i in range(combo.count()):
        data = combo.itemData(i)
        item_value = data.value if hasattr(data, "value") else str(data)
        if item_value == value:
            combo.setCurrentIndex(i)
            return
