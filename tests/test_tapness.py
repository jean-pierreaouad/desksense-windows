from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from desksense.robustness import extract_descriptive_tapness_metrics
import desksense.robustness as robustness_module
import desksense.tapness as tapness_module
from desksense.streaming import StreamingDetectorConfig
from desksense.tapness import (
    TAPNESS_FEATURE_NAMES,
    TapnessBaselineError,
    classify_tapness_metrics,
    create_tapness_baseline,
    extract_descriptive_tapness_metrics as pure_extract_tapness_metrics,
    fit_l2_logistic,
    grouped_development_folds,
    load_tapness_baseline,
    stage1_policy_metadata,
    select_development_threshold,
    tapness_feature_schema,
    transform_tapness_metrics,
    validate_tapness_artifact,
    validate_tapness_stage1_config,
    write_tapness_baseline,
)


def _metrics(scale: float = 1.0) -> dict[str, float]:
    return {
        "pre_onset_rms": 0.001 * scale,
        "impact_window_rms": 0.02 * scale,
        "impact_peak_absolute": 0.08 * scale,
        "onset_contrast_rms_ratio": 20.0,
        "peak_dominant_contrast_ratio": 80.0,
        "effective_energy_duration_seconds": 0.012,
        "early_energy_fraction": 0.8,
        "late_to_impact_rms_ratio": 0.1,
        "strong_sample_fraction": 0.05,
    }


def _artifact() -> dict:
    artifact = json.loads(
        Path("baselines/lenovo-tapness-v1.json").read_text(encoding="utf-8")
    )
    count = len(TAPNESS_FEATURE_NAMES)
    artifact["model"]["standardization"]["means"] = [0.0] * count
    artifact["model"]["standardization"]["scales"] = [1.0] * count
    artifact["model"]["coefficients"] = [0.0] * count
    artifact["model"]["intercept"] = 0.0
    artifact["model"]["decision_threshold"] = 0.5
    selected = artifact["development_cross_validation"][
        "selected_threshold_metrics"
    ]
    selected["threshold"] = 0.5
    artifact["model"]["threshold_selection"]["selected_threshold"] = 0.5
    artifact["model"]["threshold_selection"][
        "selected_out_of_fold_metrics"
    ] = deepcopy(selected)
    selected_l2 = next(
        item
        for item in artifact["development_cross_validation"][
            "l2_development_comparison"
        ]
        if item["l2_regularization"] == artifact["model"]["l2_regularization"]
    )
    selected_l2["selected_threshold"] = 0.5
    selected_l2["positive_accepted_count"] = selected[
        "positive_accepted_count"
    ]
    selected_l2["negative_false_accept_count"] = selected[
        "negative_false_accept_count"
    ]
    selected_l2["maximum_false_accepts_in_one_negative_group"] = selected[
        "maximum_false_accepts_in_one_negative_group"
    ]
    for fold in artifact["development_cross_validation"]["folds"]:
        fold["held_out_metrics_at_selected_threshold"]["threshold"] = 0.5
    return artifact


def test_feature_schema_freezes_exact_order_and_transforms() -> None:
    schema = tapness_feature_schema()

    assert tuple(schema["ordered_feature_names"]) == TAPNESS_FEATURE_NAMES
    transforms = tuple(item["transformation"] for item in schema["features"])
    assert transforms == (
        "natural_log_with_fixed_epsilon",
        "natural_log_with_fixed_epsilon",
        "natural_log_with_fixed_epsilon",
        "natural_log_with_fixed_epsilon",
        "natural_log_with_fixed_epsilon",
        "natural_log_with_fixed_epsilon",
        "linear",
        "natural_log_with_fixed_epsilon",
        "linear",
    )


def test_tapness_extraction_is_one_shared_implementation() -> None:
    assert robustness_module.extract_descriptive_tapness_metrics is pure_extract_tapness_metrics


def test_transform_is_finite_deterministic_and_rejects_invalid_values() -> None:
    first = transform_tapness_metrics(_metrics())
    second = transform_tapness_metrics(_metrics())
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))

    for value in (float("nan"), float("inf"), -1.0):
        invalid = _metrics()
        invalid["impact_window_rms"] = value
        with pytest.raises(TapnessBaselineError):
            transform_tapness_metrics(invalid)


def test_feature_extraction_preserves_candidate_and_opposite_polarities() -> None:
    candidate = np.zeros((9_600, 2), dtype=np.float32)
    candidate[4_800:4_805, 0] = 0.2
    candidate[4_800:4_805, 1] = -0.2
    before = candidate.tobytes()

    metrics = extract_descriptive_tapness_metrics(
        candidate, sample_rate_hz=48_000.0, onset_offset_frames=4_800
    )

    assert metrics["impact_peak_absolute"] == pytest.approx(0.2)
    assert metrics["impact_window_rms"] > 0.0
    assert candidate.tobytes() == before


def test_l2_fitting_and_inference_are_deterministic_with_exact_tie_rule() -> None:
    matrix = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    first = fit_l2_logistic(matrix, labels)
    second = fit_l2_logistic(matrix, labels)
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]

    artifact = _artifact()
    decision = classify_tapness_metrics(_metrics(), artifact)
    assert decision["tap_accepted"] is True
    assert decision["predicted_label"] == "TAP"
    assert decision["model_margin"] == pytest.approx(0.0)
    assert decision["score_is_calibrated_probability"] is False


def test_logistic_fit_fails_explicitly_when_iteration_budget_does_not_converge() -> None:
    matrix = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])

    with pytest.raises(TapnessBaselineError, match="did not converge"):
        fit_l2_logistic(
            matrix,
            labels,
            maximum_iterations=1,
            tolerance=1.0e-30,
        )


def test_artifact_round_trip_is_deterministic_and_overwrite_is_refused(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    path = tmp_path / "tapness.json"
    write_tapness_baseline(artifact, path)

    assert load_tapness_baseline(path) == artifact
    with pytest.raises(TapnessBaselineError, match="overwrite"):
        write_tapness_baseline(artifact, path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: artifact.update(artifact_schema_version=2),
        lambda artifact: artifact.update(created_at_utc="not-a-timestamp"),
        lambda artifact: artifact["feature_schema"].update(version=2),
        lambda artifact: artifact["stage1_policy"]["strong_impact_recovery"].update(rms_ratio_minimum_inclusive=5.0),
        lambda artifact: artifact["model"].update(coefficients=[0.0]),
        lambda artifact: artifact["model"].update(decision_threshold=float("nan")),
        lambda artifact: artifact["model"].update(l2_regularization=0.0),
        lambda artifact: artifact["model"].update(score_is_calibrated_probability=True),
        lambda artifact: artifact["model"].update(tie_rule="score > threshold predicts TAP"),
        lambda artifact: artifact["source_development_dataset"].update(dataset_fingerprint={"sha256": "bad"}),
        lambda artifact: artifact["source_development_dataset"]["training_candidate_counts"].update(total=80),
        lambda artifact: artifact["source_development_dataset"]["positive_membership_identifiers"].pop(),
        lambda artifact: artifact["development_cross_validation"].update(external_validation=True),
        lambda artifact: artifact["development_cross_validation"]["selected_threshold_metrics"].update(threshold=0.25),
        lambda artifact: artifact["model"]["threshold_selection"].update(session_a_tuned=False),
        lambda artifact: artifact["development_cross_validation"]["folds"][0]["held_out_metrics_at_selected_threshold"].update(positive_accepted_count=0),
        lambda artifact: artifact.update(candidate_window=[1, 2, 3]),
        lambda artifact: artifact["target"].update(spatial_labels_or_features_used=True),
    ],
)
def test_artifact_validation_is_strict(mutation) -> None:
    artifact = deepcopy(_artifact())
    mutation(artifact)
    with pytest.raises(TapnessBaselineError):
        validate_tapness_artifact(artifact)


def test_malformed_json_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(TapnessBaselineError):
        load_tapness_baseline(path)


def test_grouped_folds_keep_each_recording_together() -> None:
    examples = [
        {"source_group_id": "p1", "target_label": "TAP"},
        {"source_group_id": "p2", "target_label": "TAP"},
        {"source_group_id": "p3", "target_label": "TAP"},
        {"source_group_id": "n1", "target_label": "NON_TAP"},
        {"source_group_id": "n1", "target_label": "NON_TAP"},
        {"source_group_id": "n2", "target_label": "NON_TAP"},
        {"source_group_id": "n3", "target_label": "NON_TAP"},
    ]
    folds = grouped_development_folds(examples, 3)
    locations = {
        index: fold_number
        for fold_number, fold in enumerate(folds)
        for index in fold
    }
    assert locations[3] == locations[4]
    assert sorted(index for fold in folds for index in fold) == list(range(7))


def test_grouped_oof_preprocessing_is_fit_on_training_fold_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.asarray(
        [[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [100.0, 1_000.0]],
        dtype=np.float64,
    )
    labels = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    groups = ["n1", "n2", "p1", "p2"]
    folds = ((0, 2), (1, 3))
    fitted_training_matrices: list[np.ndarray] = []

    def fake_fit(training_matrix, _labels, *, l2_regularization):
        fitted_training_matrices.append(np.asarray(training_matrix).copy())
        return np.zeros(training_matrix.shape[1]), 0.0

    monkeypatch.setattr(tapness_module, "fit_l2_logistic", fake_fit)
    scores, reports = tapness_module._grouped_oof_fit(
        matrix,
        labels,
        groups,
        folds,
        l2_regularization=0.01,
    )

    assert len(fitted_training_matrices) == 2
    assert all(
        np.allclose(np.mean(training, axis=0), 0.0)
        for training in fitted_training_matrices
    )
    assert np.array_equal(scores, np.full(4, 0.5))
    assert all(report["training_preprocessing_only"] for report in reports)


def test_threshold_selection_and_exact_tie_are_deterministic() -> None:
    scores = np.asarray([0.2, 0.4, 0.6, 0.8])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    groups = ["n1", "n2", "p1", "p2"]

    first = select_development_threshold(scores, labels, groups)
    second = select_development_threshold(scores, labels, groups)

    assert first == second
    threshold, _ = first
    assert int(np.count_nonzero(scores >= threshold)) >= 2


def test_artifact_contains_no_spatial_feature_dependency() -> None:
    serialized = json.dumps(_artifact()).casefold()
    assert "peak_ratio_db_ch2_minus_ch1" not in serialized
    assert "spatial_margin" not in serialized


@pytest.mark.parametrize(
    "changed_config",
    [
        replace(StreamingDetectorConfig(), internal_block_seconds=0.010),
        replace(StreamingDetectorConfig(), startup_learning_seconds=0.500),
        replace(StreamingDetectorConfig(), minimum_noise_floor_rms=2.0e-7),
        replace(StreamingDetectorConfig(), onset_rms_noise_multiplier=4.0),
        replace(StreamingDetectorConfig(), onset_peak_noise_multiplier=7.0),
        replace(StreamingDetectorConfig(), minimum_crest_factor=2.5),
        replace(StreamingDetectorConfig(), noise_floor_adaptation_alpha=0.03),
        replace(StreamingDetectorConfig(), center_search_pre_onset_seconds=0.010),
        replace(StreamingDetectorConfig(), tap_window_seconds=0.180),
        replace(StreamingDetectorConfig(), refractory_seconds=0.300),
        replace(StreamingDetectorConfig(), clipping_threshold=0.95),
        replace(StreamingDetectorConfig(), history_safety_seconds=0.030),
    ],
)
def test_material_stage1_config_changes_are_incompatible(changed_config) -> None:
    with pytest.raises(TapnessBaselineError, match="incompatible"):
        validate_tapness_stage1_config(changed_config, _artifact())


def test_missing_stage1_config_fails_as_tapness_error() -> None:
    with pytest.raises(TapnessBaselineError, match="configuration is required"):
        validate_tapness_stage1_config(None, _artifact())


def test_tracked_session_a_artifact_records_grouped_fold_only_preprocessing() -> None:
    artifact = load_tapness_baseline(
        Path("baselines/lenovo-tapness-v1.json")
    )

    assert artifact["source_development_dataset"]["training_candidate_counts"] == {
        "positive": 29,
        "negative": 50,
        "total": 79,
    }
    assert all(
        fold["training_preprocessing_only"] is True
        for fold in artifact["development_cross_validation"]["folds"]
    )
    assert artifact["development_cross_validation"]["external_validation"] is False
    assert artifact["source_development_dataset"]["no_external_samples_used"] is True


SESSION_A = Path("datasets/20260901T131308.362205Z-f9e2b1ec")


@pytest.mark.skipif(not SESSION_A.is_dir(), reason="local ignored Session A is unavailable")
def test_session_a_training_membership_regression_is_29_tap_50_non_tap() -> None:
    artifact = create_tapness_baseline(
        SESSION_A,
        now_fn=lambda: datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    assert artifact["source_development_dataset"]["training_candidate_counts"] == {
        "positive": 29,
        "negative": 50,
        "total": 79,
    }
    assert artifact["model"]["decision_threshold"] == 0.4495211534633274
    assert artifact["source_development_dataset"]["dataset_fingerprint"]["sha256"] == (
        "88a003966141d15858a2afec390c42e567439eb5f538b958e7aa60ee7638ab83"
    )
