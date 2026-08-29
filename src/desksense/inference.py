"""Pure LEFT/RIGHT inference for the frozen DeskSense peak-ratio baseline."""

from __future__ import annotations

import math
from typing import Any

from desksense.features import PRIMARY_FEATURE_NAME


_EXPECTED_ZONES = {"LEFT", "RIGHT"}


def classify_peak_ratio_value(
    feature_value_db: Any,
    threshold_db: Any,
    lower_feature_zone: Any,
    higher_feature_zone: Any,
) -> dict[str, Any]:
    """Classify one feature value without requiring a known label.

    Values below the threshold predict ``lower_feature_zone``. An exact tie
    predicts ``higher_feature_zone``, matching the frozen Phase 2C rule.
    Reported margins are distances on the dB feature axis, not probabilities.
    """

    value = _finite_float(feature_value_db, "primary feature")
    threshold = _finite_float(threshold_db, "threshold")
    lower_zone, higher_zone = _validate_direction(
        lower_feature_zone, higher_feature_zone
    )

    predicted = lower_zone if value < threshold else higher_zone
    threshold_offset = float(value - threshold)
    signed_toward_right = (
        threshold_offset if higher_zone == "RIGHT" else -threshold_offset
    )
    return {
        "feature_name": PRIMARY_FEATURE_NAME,
        "feature_value_db": value,
        "predicted_label": predicted,
        "threshold_db": threshold,
        "threshold_offset_db": threshold_offset,
        "absolute_margin_db": float(abs(threshold_offset)),
        "signed_margin_toward_RIGHT_db": float(signed_toward_right),
    }


def _validate_direction(lower_zone: Any, higher_zone: Any) -> tuple[str, str]:
    if (
        not isinstance(lower_zone, str)
        or not isinstance(higher_zone, str)
        or {lower_zone, higher_zone} != _EXPECTED_ZONES
    ):
        raise ValueError(
            "Lower and higher feature zones must be distinct LEFT and RIGHT labels."
        )
    return lower_zone, higher_zone


def _finite_float(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label.capitalize()} must be finite.") from error
    if not math.isfinite(converted):
        raise ValueError(f"{label.capitalize()} must be finite.")
    return converted
