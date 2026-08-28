from __future__ import annotations

from collections.abc import Iterable

import pytest

from desksense.analysis import (
    DatasetAnalysisError,
    chronological_holdout_analysis,
    chronological_split,
    confusion_matrix,
    fit_peak_ratio_baseline,
    leave_one_pair_out_cross_validation,
    predict_peak_ratio,
)
from desksense.features import PRIMARY_FEATURE_NAME


def _record(zone: str, number: int, value_db: float) -> dict[str, object]:
    zone_order = 0 if zone == "LEFT" else 1
    return {
        "sample_id": f"{zone.lower()}-{number:03d}",
        "zone": zone,
        "accepted_sample_number": number,
        "collection_order_index": ((number - 1) * 2) + zone_order + 1,
        "features": {PRIMARY_FEATURE_NAME: value_db},
    }


def _twenty_pairs(
    *,
    left_values: Iterable[float] | None = None,
    right_values: Iterable[float] | None = None,
) -> list[dict[str, object]]:
    left = list(left_values if left_values is not None else [-6.0] * 20)
    right = list(right_values if right_values is not None else [2.0] * 20)
    assert len(left) == len(right) == 20
    records = [
        _record(zone, number, left[number - 1] if zone == "LEFT" else right[number - 1])
        for number in range(1, 21)
        for zone in ("LEFT", "RIGHT")
    ]
    return records


def test_chronological_split_uses_accepted_number_not_input_order() -> None:
    records = list(reversed(_twenty_pairs()))

    training, test = chronological_split(records)

    assert len(training) == 30
    assert len(test) == 10
    assert {
        int(record["accepted_sample_number"]) for record in training
    } == set(range(1, 16))
    assert {
        int(record["accepted_sample_number"]) for record in test
    } == set(range(16, 21))
    assert [record["sample_id"] for record in test] == [
        f"{zone.lower()}-{number:03d}"
        for number in range(16, 21)
        for zone in ("LEFT", "RIGHT")
    ]


def test_chronological_split_rejects_overlapping_training_and_test_ranges() -> None:
    with pytest.raises(DatasetAnalysisError, match="training_end < test_start"):
        chronological_split(
            _twenty_pairs(),
            training_end=16,
            test_start=16,
            test_end=20,
        )


def test_midpoint_and_direction_are_learned_from_training_class_means() -> None:
    training = [
        _record("LEFT", 1, -8.0),
        _record("LEFT", 2, -4.0),
        _record("RIGHT", 1, 2.0),
        _record("RIGHT", 2, 6.0),
    ]

    fitted = fit_peak_ratio_baseline(training)

    assert fitted["training_class_means_db"] == {"LEFT": -6.0, "RIGHT": 4.0}
    assert fitted["threshold_db"] == pytest.approx(-1.0)
    assert fitted["lower_feature_zone"] == "LEFT"
    assert fitted["higher_feature_zone"] == "RIGHT"
    assert fitted["training_count_by_class"] == {"LEFT": 2, "RIGHT": 2}


def test_threshold_direction_reverses_when_left_training_mean_is_higher() -> None:
    training = [
        _record("LEFT", 1, 4.0),
        _record("LEFT", 2, 8.0),
        _record("RIGHT", 1, -4.0),
        _record("RIGHT", 2, 0.0),
    ]

    fitted = fit_peak_ratio_baseline(training)

    assert fitted["threshold_db"] == pytest.approx(2.0)
    assert fitted["lower_feature_zone"] == "RIGHT"
    assert fitted["higher_feature_zone"] == "LEFT"


def test_holdout_values_do_not_influence_fitted_threshold() -> None:
    ordinary = _twenty_pairs()
    altered = _twenty_pairs(
        left_values=[-6.0] * 15 + [1_000.0] * 5,
        right_values=[2.0] * 15 + [-1_000.0] * 5,
    )

    ordinary_result = chronological_holdout_analysis(ordinary)
    altered_result = chronological_holdout_analysis(altered)

    assert ordinary_result["fitted_baseline"]["threshold_db"] == pytest.approx(
        -2.0
    )
    assert altered_result["fitted_baseline"]["threshold_db"] == pytest.approx(
        -2.0
    )
    assert all(
        sample_id.endswith(tuple(f"-{number:03d}" for number in range(1, 16)))
        for sample_id in altered_result["fitted_baseline"]["training_sample_ids"]
    )


def test_prediction_reports_threshold_offsets_and_class_margin() -> None:
    fitted = fit_peak_ratio_baseline(
        [_record("LEFT", 1, -6.0), _record("RIGHT", 1, 2.0)]
    )

    left_prediction = predict_peak_ratio(_record("LEFT", 2, -3.5), fitted)
    wrong_left_prediction = predict_peak_ratio(
        _record("LEFT", 3, 0.5), fitted
    )

    assert fitted["threshold_db"] == pytest.approx(-2.0)
    assert left_prediction["predicted_label"] == "LEFT"
    assert left_prediction["threshold_offset_db"] == pytest.approx(-1.5)
    assert left_prediction["absolute_margin_db"] == pytest.approx(1.5)
    assert left_prediction["signed_margin_toward_RIGHT_db"] == pytest.approx(
        -1.5
    )
    assert left_prediction["actual_class_margin_db"] == pytest.approx(1.5)
    assert left_prediction["correct"] is True
    assert wrong_left_prediction["predicted_label"] == "RIGHT"
    assert wrong_left_prediction["actual_class_margin_db"] == pytest.approx(
        -2.5
    )
    assert wrong_left_prediction["correct"] is False


def test_exact_threshold_uses_fitted_higher_zone_tie_rule() -> None:
    fitted = fit_peak_ratio_baseline(
        [_record("LEFT", 1, 8.0), _record("RIGHT", 1, -4.0)]
    )

    prediction = predict_peak_ratio(_record("LEFT", 2, 2.0), fitted)

    assert fitted["higher_feature_zone"] == "LEFT"
    assert prediction["predicted_label"] == "LEFT"
    assert prediction["absolute_margin_db"] == pytest.approx(0.0)


def test_confusion_matrix_has_explicit_actual_and_predicted_axes() -> None:
    predictions = [
        {"actual_label": "LEFT", "predicted_label": "LEFT"},
        {"actual_label": "LEFT", "predicted_label": "RIGHT"},
        {"actual_label": "RIGHT", "predicted_label": "LEFT"},
        {"actual_label": "RIGHT", "predicted_label": "RIGHT"},
        {"actual_label": "RIGHT", "predicted_label": "RIGHT"},
    ]

    assert confusion_matrix(predictions) == {
        "actual_LEFT": {"predicted_LEFT": 1, "predicted_RIGHT": 1},
        "actual_RIGHT": {"predicted_LEFT": 1, "predicted_RIGHT": 2},
    }


def test_chronological_holdout_reports_known_perfect_predictions() -> None:
    result = chronological_holdout_analysis(_twenty_pairs())

    assert result["label"] == "within-session chronological holdout"
    assert result["protocol"]["training_count_by_class"] == {
        "LEFT": 15,
        "RIGHT": 15,
    }
    assert result["protocol"]["test_count_by_class"] == {
        "LEFT": 5,
        "RIGHT": 5,
    }
    assert result["correct_count"] == result["total_count"] == 10
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == {
        "actual_LEFT": {"predicted_LEFT": 5, "predicted_RIGHT": 0},
        "actual_RIGHT": {"predicted_LEFT": 0, "predicted_RIGHT": 5},
    }


def test_leave_one_pair_out_excludes_exact_pair_and_fits_remaining_samples() -> None:
    left_values = [-10.0 + number * 0.1 for number in range(1, 21)]
    right_values = [2.0 + number * 0.2 for number in range(1, 21)]
    records = _twenty_pairs(
        left_values=left_values,
        right_values=right_values,
    )

    result = leave_one_pair_out_cross_validation(records)
    fold = result["folds"][6]
    expected_left_mean = sum(left_values[:6] + left_values[7:]) / 19
    expected_right_mean = sum(right_values[:6] + right_values[7:]) / 19

    assert fold["fold"] == 7
    assert fold["held_out_sample_ids"] == ["left-007", "right-007"]
    assert "left-007" not in fold["fitted_baseline"]["training_sample_ids"]
    assert "right-007" not in fold["fitted_baseline"]["training_sample_ids"]
    assert len(fold["fitted_baseline"]["training_sample_ids"]) == 38
    assert fold["fitted_baseline"]["threshold_db"] == pytest.approx(
        (expected_left_mean + expected_right_mean) / 2.0
    )


def test_leave_one_pair_out_aggregates_all_folds_and_predictions() -> None:
    result = leave_one_pair_out_cross_validation(_twenty_pairs())

    assert result["label"] == (
        "within-session leave-one-pair-out cross-validation"
    )
    assert result["protocol"]["fold_count"] == 20
    assert result["protocol"]["held_out_samples_per_fold"] == 2
    assert len(result["folds"]) == 20
    assert all(len(fold["predictions"]) == 2 for fold in result["folds"])
    assert len(result["predictions"]) == 40
    assert result["correct_count"] == result["total_count"] == 40
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == {
        "actual_LEFT": {"predicted_LEFT": 20, "predicted_RIGHT": 0},
        "actual_RIGHT": {"predicted_LEFT": 0, "predicted_RIGHT": 20},
    }
