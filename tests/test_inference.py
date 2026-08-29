from __future__ import annotations

import math

import pytest

from desksense.analysis import DatasetAnalysisError, predict_peak_ratio
from desksense.features import PRIMARY_FEATURE_NAME
from desksense.inference import classify_peak_ratio_value


def _baseline(*, reverse: bool = False) -> dict[str, object]:
    return {
        "threshold_db": 1.25,
        "lower_feature_zone": "RIGHT" if reverse else "LEFT",
        "higher_feature_zone": "LEFT" if reverse else "RIGHT",
    }


def _record(actual: str, value: float) -> dict[str, object]:
    return {
        "sample_id": "sample-001",
        "zone": actual,
        "accepted_sample_number": 1,
        "collection_order_index": 1,
        "features": {PRIMARY_FEATURE_NAME: value},
    }


@pytest.mark.parametrize(
    ("value", "predicted", "offset", "toward_right"),
    [
        (0.25, "LEFT", -1.0, -1.0),
        (1.25, "RIGHT", 0.0, 0.0),
        (3.25, "RIGHT", 2.0, 2.0),
    ],
)
def test_normal_direction_and_higher_zone_tie_rule(
    value: float, predicted: str, offset: float, toward_right: float
) -> None:
    result = classify_peak_ratio_value(value, 1.25, "LEFT", "RIGHT")

    assert result == {
        "feature_name": PRIMARY_FEATURE_NAME,
        "feature_value_db": value,
        "predicted_label": predicted,
        "threshold_db": 1.25,
        "threshold_offset_db": offset,
        "absolute_margin_db": abs(offset),
        "signed_margin_toward_RIGHT_db": toward_right,
    }


@pytest.mark.parametrize(
    ("value", "predicted", "offset", "toward_right"),
    [
        (0.25, "RIGHT", -1.0, 1.0),
        (1.25, "LEFT", 0.0, 0.0),
        (3.25, "LEFT", 2.0, -2.0),
    ],
)
def test_reverse_direction_and_higher_zone_tie_rule(
    value: float, predicted: str, offset: float, toward_right: float
) -> None:
    result = classify_peak_ratio_value(value, 1.25, "RIGHT", "LEFT")

    assert result["predicted_label"] == predicted
    assert result["threshold_offset_db"] == pytest.approx(offset)
    assert result["absolute_margin_db"] == pytest.approx(abs(offset))
    assert result["signed_margin_toward_RIGHT_db"] == pytest.approx(
        toward_right
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("value", [0.25, 1.25, 3.25])
def test_offline_predictor_delegation_preserves_shared_fields(
    reverse: bool, value: float
) -> None:
    baseline = _baseline(reverse=reverse)
    direct = classify_peak_ratio_value(
        value,
        baseline["threshold_db"],
        baseline["lower_feature_zone"],
        baseline["higher_feature_zone"],
    )
    wrapped = predict_peak_ratio(_record("LEFT", value), baseline)

    for key, expected in direct.items():
        assert wrapped[key] == expected
    assert wrapped["sample_id"] == "sample-001"
    assert wrapped["actual_label"] == "LEFT"
    assert "actual_class_margin_db" in wrapped
    assert isinstance(wrapped["correct"], bool)


def test_label_free_inference_requires_no_sample_or_actual_label() -> None:
    result = classify_peak_ratio_value(-2.0, 0.0, "LEFT", "RIGHT")

    assert result["predicted_label"] == "LEFT"
    assert "actual_label" not in result
    assert "correct" not in result
    assert "sample_id" not in result
    assert "actual_class_margin_db" not in result


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, None, "bad"])
def test_nonfinite_or_nonnumeric_feature_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="Primary feature must be finite"):
        classify_peak_ratio_value(value, 0.0, "LEFT", "RIGHT")


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf, None, "bad"])
def test_nonfinite_or_nonnumeric_threshold_is_rejected(threshold: object) -> None:
    with pytest.raises(ValueError, match="Threshold must be finite"):
        classify_peak_ratio_value(0.0, threshold, "LEFT", "RIGHT")


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("LEFT", "LEFT"),
        ("RIGHT", "RIGHT"),
        ("left", "RIGHT"),
        ("LEFT", "OTHER"),
        (1, "RIGHT"),
    ],
)
def test_direction_requires_distinct_exact_left_and_right_labels(
    lower: object, higher: object
) -> None:
    with pytest.raises(ValueError, match="distinct LEFT and RIGHT"):
        classify_peak_ratio_value(0.0, 1.0, lower, higher)


def test_predictor_preserves_sample_specific_unavailable_feature_error() -> None:
    record = _record("LEFT", 0.0)
    record["features"] = {PRIMARY_FEATURE_NAME: None}

    with pytest.raises(
        DatasetAnalysisError,
        match="Primary feature is unavailable for sample sample-001",
    ):
        predict_peak_ratio(record, _baseline())


def test_predictor_translates_inference_validation_to_analysis_error() -> None:
    baseline = _baseline()
    baseline["threshold_db"] = math.nan

    with pytest.raises(DatasetAnalysisError, match="Threshold must be finite"):
        predict_peak_ratio(_record("LEFT", -1.0), baseline)
