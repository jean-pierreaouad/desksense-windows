"""Versioned Phase 3B tap-vs-non-tap baseline fitting and inference.

The model is deliberately independent of the frozen LEFT/RIGHT classifier.
It consumes only fixed time-domain candidate descriptors and never waveform
channel-ratio, spatial-label, or spatial-margin information.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desksense.robustness_dataset import (
    load_robustness_dataset,
    validate_robustness_capture,
)
from desksense.streaming import StreamingDetectorConfig


TAPNESS_ARTIFACT_SCHEMA_VERSION = 1
TAPNESS_ARTIFACT_TYPE = "desksense_frozen_tapness_logistic_baseline"
TAPNESS_FEATURE_SCHEMA_VERSION = 1
TAPNESS_STAGE1_POLICY_VERSION = 1
TAPNESS_POSITIVE_LABEL = "TAP"
TAPNESS_NEGATIVE_LABEL = "NON_TAP"
TAPNESS_L2_REGULARIZATION = 0.01
TAPNESS_CV_FOLD_COUNT = 5
TAPNESS_MINIMUM_CV_POSITIVE_RECALL = 0.90
TAPNESS_LOG_EPSILON = 1.0e-12

_METRIC_PRE_ONSET_SECONDS = 0.012
_METRIC_IMPACT_SECONDS = 0.025
_METRIC_EARLY_SECONDS = 0.025
_METRIC_LATE_START_SECONDS = 0.040
_METRIC_FRAME_SECONDS = 0.005
_METRIC_STRONG_FRACTION_OF_PEAK = 0.50

TAPNESS_FEATURE_NAMES = (
    "pre_onset_rms",
    "impact_window_rms",
    "impact_peak_absolute",
    "onset_contrast_rms_ratio",
    "peak_dominant_contrast_ratio",
    "effective_energy_duration_seconds",
    "early_energy_fraction",
    "late_to_impact_rms_ratio",
    "strong_sample_fraction",
)
TAPNESS_FEATURE_TRANSFORMS = (
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


class TapnessBaselineError(RuntimeError):
    """A tapness training, artifact, compatibility, or inference failure."""


def stage1_policy_metadata(
    config: StreamingDetectorConfig | None = None,
) -> dict[str, Any]:
    """Return the exact production Stage 1 policy required by this model."""

    detector = config or StreamingDetectorConfig()
    return {
        "policy_version": TAPNESS_STAGE1_POLICY_VERSION,
        "candidate_routes": {
            "ordinary": "RMS AND peak AND crest",
            "strong_impact_recovery": "RMS ratio AND peak ratio; no crest requirement",
            "combination": "ordinary OR strong_impact_recovery",
            "ordinary_route_diagnostic_precedence": True,
            "duplicate_candidate_when_both_routes_pass": False,
        },
        "audio_domain": {
            "sample_rate_hz": float(detector.sample_rate_hz),
            "channel_count": int(detector.channel_count),
        },
        "fixed_block": {
            "duration_seconds": float(detector.internal_block_seconds),
            "frames": int(detector.internal_block_frames),
            "overlapping_blocks": False,
        },
        "startup_learning": {
            "duration_seconds_requested": float(detector.startup_learning_seconds),
            "frames_effective": int(detector.startup_learning_frames),
        },
        "ordinary_gate": {
            "minimum_noise_floor_rms": float(detector.minimum_noise_floor_rms),
            "rms_noise_multiplier": float(detector.onset_rms_noise_multiplier),
            "peak_noise_multiplier": float(detector.onset_peak_noise_multiplier),
            "minimum_onset_rms": float(detector.minimum_onset_rms),
            "minimum_onset_peak": float(detector.minimum_onset_peak),
            "minimum_crest_factor": float(detector.minimum_crest_factor),
        },
        "strong_impact_recovery": {
            "route_name": "strong_impact_recovery",
            "rms_ratio_minimum_inclusive": float(
                detector.strong_impact_recovery_rms_ratio
            ),
            "peak_ratio_minimum_inclusive": float(
                detector.strong_impact_recovery_peak_ratio
            ),
            "crest_gate_required": False,
        },
        "noise_floor": {
            "adaptation_alpha_per_fixed_block": float(
                detector.noise_floor_adaptation_alpha
            ),
            "update_policy": "existing EWMA behavior unchanged",
        },
        "center_search": {
            "pre_onset_seconds": float(detector.center_search_pre_onset_seconds),
            "pre_onset_frames": int(detector.center_search_pre_onset_frames),
            "post_onset_seconds": float(detector.center_search_post_onset_seconds),
            "post_onset_frames": int(detector.center_search_post_onset_frames),
            "transient_energy_window_seconds": float(
                detector.transient_energy_window_seconds
            ),
            "transient_energy_window_frames": int(
                detector.transient_energy_window_frames
            ),
        },
        "candidate_window_geometry": {
            "duration_seconds": float(detector.tap_window_seconds),
            "frames": int(detector.tap_window_frames),
            "pre_center_frames": int(detector.tap_window_pre_center_frames),
        },
        "refractory": {
            "duration_seconds": float(detector.refractory_seconds),
            "frames": int(detector.refractory_frames),
        },
        "clipping": {
            "threshold": float(detector.clipping_threshold),
            "reject_near_clipping": bool(detector.reject_near_clipping),
        },
        "history": {
            "safety_seconds": float(detector.history_safety_seconds),
            "safety_frames": int(detector.history_safety_frames),
            "capacity_frames": int(detector.history_capacity_frames),
        },
    }


def tapness_feature_schema() -> dict[str, Any]:
    """Return the immutable ordered feature/transformation definition."""

    descriptions = {
        "pre_onset_rms": "Pooled RMS over up to 12 ms immediately before detector onset.",
        "impact_window_rms": "Maximum pooled RMS among 5 ms frames in the first 25 ms from onset.",
        "impact_peak_absolute": "Maximum absolute sample over both channels in the first 25 ms from onset.",
        "onset_contrast_rms_ratio": "Impact RMS divided by the pre-onset/learned numerical reference floor.",
        "peak_dominant_contrast_ratio": "Impact peak divided by the same reference floor.",
        "effective_energy_duration_seconds": "Post-onset energy divided by peak instantaneous power and sample rate.",
        "early_energy_fraction": "Fraction of post-onset energy in the first 25 ms.",
        "late_to_impact_rms_ratio": "RMS from 40 ms after onset divided by impact RMS.",
        "strong_sample_fraction": "Fraction of impact frames at least half the impact peak.",
    }
    units = {
        "pre_onset_rms": "linear_full_scale",
        "impact_window_rms": "linear_full_scale",
        "impact_peak_absolute": "linear_full_scale",
        "onset_contrast_rms_ratio": "ratio",
        "peak_dominant_contrast_ratio": "ratio",
        "effective_energy_duration_seconds": "seconds",
        "early_energy_fraction": "fraction",
        "late_to_impact_rms_ratio": "ratio",
        "strong_sample_fraction": "fraction",
    }
    return {
        "version": TAPNESS_FEATURE_SCHEMA_VERSION,
        "ordered_feature_names": list(TAPNESS_FEATURE_NAMES),
        "features": [
            {
                "name": name,
                "units": units[name],
                "definition": descriptions[name],
                "transformation": transform,
                **(
                    {"log_epsilon": TAPNESS_LOG_EPSILON}
                    if transform == "natural_log_with_fixed_epsilon"
                    else {}
                ),
            }
            for name, transform in zip(
                TAPNESS_FEATURE_NAMES, TAPNESS_FEATURE_TRANSFORMS, strict=True
            )
        ],
        "finite_value_handling": (
            "all raw descriptors must be finite and nonnegative; logarithmic "
            "features use ln(max(value, 1e-12)); invalid values are rejected"
        ),
        "channel_processing": "pooled squared multichannel energy; no waveform averaging",
        "candidate_processing": "float64 processing copy; raw 9600x2 candidate unchanged",
    }


def extract_descriptive_tapness_metrics(
    candidate_window: Any,
    *,
    sample_rate_hz: float,
    onset_offset_frames: int,
    learned_noise_floor_rms: float | None = None,
) -> dict[str, Any]:
    """Extract the exact immutable Stage 2 v1 descriptor set.

    Computation uses a float64 processing copy and pooled squared channel
    energy, so opposite-polarity channels cannot cancel. The raw classifier
    candidate is never modified.
    """

    audio = validate_robustness_capture(candidate_window).astype(np.float64)
    sample_rate = float(sample_rate_hz)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("Tapness metric sample rate must be positive and finite.")
    if (
        isinstance(onset_offset_frames, bool)
        or not isinstance(onset_offset_frames, int)
        or onset_offset_frames < 0
        or onset_offset_frames >= audio.shape[0]
    ):
        raise ValueError("Tapness metric onset offset is outside the candidate.")

    pre_frames = max(1, round(sample_rate * _METRIC_PRE_ONSET_SECONDS))
    pre_start = max(0, onset_offset_frames - pre_frames)
    pre = audio[pre_start:onset_offset_frames]
    dc_reference = pre if pre.shape[0] else audio
    centered = audio - np.mean(dc_reference, axis=0, keepdims=True)
    pre_centered = centered[pre_start:onset_offset_frames]
    pre_rms = _pooled_rms(pre_centered)

    impact_end = min(
        audio.shape[0], onset_offset_frames + round(sample_rate * _METRIC_IMPACT_SECONDS)
    )
    impact = centered[onset_offset_frames:impact_end]
    frame_count = max(1, round(sample_rate * _METRIC_FRAME_SECONDS))
    impact_frame_rms = [
        _pooled_rms(impact[start : start + frame_count])
        for start in range(0, impact.shape[0], frame_count)
    ]
    impact_rms = max(impact_frame_rms, default=0.0)
    impact_peak = float(np.max(np.abs(impact))) if impact.size else 0.0
    noise_floor = _finite_nonnegative(learned_noise_floor_rms)
    reference_floor = max(pre_rms, noise_floor or 0.0, np.finfo(np.float64).tiny)

    post = centered[onset_offset_frames:]
    post_power = np.mean(np.square(post), axis=1) if post.size else np.empty(0)
    peak_power = float(np.max(post_power)) if post_power.size else 0.0
    effective_duration = (
        float(np.sum(post_power) / peak_power / sample_rate)
        if peak_power > 0.0
        else 0.0
    )
    early_end = min(
        audio.shape[0], onset_offset_frames + round(sample_rate * _METRIC_EARLY_SECONDS)
    )
    early_power = centered[onset_offset_frames:early_end]
    total_energy = float(np.sum(np.square(post)))
    early_fraction = (
        float(np.sum(np.square(early_power)) / total_energy)
        if total_energy > 0.0
        else 0.0
    )
    late_start = min(
        audio.shape[0], onset_offset_frames + round(sample_rate * _METRIC_LATE_START_SECONDS)
    )
    late_rms = _pooled_rms(centered[late_start:])
    late_ratio = float(late_rms / impact_rms) if impact_rms > 0.0 else 0.0
    instantaneous_peak = (
        np.max(np.abs(impact), axis=1) if impact.shape[0] else np.empty(0)
    )
    strong_fraction = (
        float(
            np.count_nonzero(
                instantaneous_peak
                >= impact_peak * _METRIC_STRONG_FRACTION_OF_PEAK
            )
            / instantaneous_peak.size
        )
        if instantaneous_peak.size and impact_peak > 0.0
        else 0.0
    )
    return {
        "metric_schema_version": TAPNESS_FEATURE_SCHEMA_VERSION,
        "processing": (
            "float64 copy; per-channel DC reference from immediate pre-onset "
            "audio; pooled squared multichannel power; raw candidate unchanged"
        ),
        "pre_onset_seconds": _METRIC_PRE_ONSET_SECONDS,
        "impact_search_seconds": _METRIC_IMPACT_SECONDS,
        "energy_frame_seconds": _METRIC_FRAME_SECONDS,
        "early_energy_seconds": _METRIC_EARLY_SECONDS,
        "late_start_seconds": _METRIC_LATE_START_SECONDS,
        "strong_sample_peak_fraction": _METRIC_STRONG_FRACTION_OF_PEAK,
        "pre_onset_rms": float(pre_rms),
        "impact_window_rms": float(impact_rms),
        "impact_peak_absolute": float(impact_peak),
        "reference_floor_rms": float(reference_floor),
        "onset_contrast_rms_ratio": float(impact_rms / reference_floor),
        "peak_dominant_contrast_ratio": float(impact_peak / reference_floor),
        "effective_energy_duration_seconds": float(effective_duration),
        "early_energy_fraction": float(early_fraction),
        "late_to_impact_rms_ratio": float(late_ratio),
        "strong_sample_fraction": float(strong_fraction),
        "decision_use": "descriptive_only_or_frozen_tapness_v1_input",
    }


def transform_tapness_metrics(metrics: Mapping[str, Any]) -> np.ndarray:
    """Validate and transform one fixed tapness descriptor vector."""

    transformed: list[float] = []
    for name, transform in zip(
        TAPNESS_FEATURE_NAMES, TAPNESS_FEATURE_TRANSFORMS, strict=True
    ):
        if name not in metrics:
            raise TapnessBaselineError(f"Missing tapness feature: {name}")
        try:
            value = float(metrics[name])
        except (TypeError, ValueError, OverflowError) as error:
            raise TapnessBaselineError(
                f"Tapness feature {name} must be numeric."
            ) from error
        if not math.isfinite(value) or value < 0.0:
            raise TapnessBaselineError(
                f"Tapness feature {name} must be finite and nonnegative."
            )
        if transform == "natural_log_with_fixed_epsilon":
            value = math.log(max(value, TAPNESS_LOG_EPSILON))
        transformed.append(value)
    return np.asarray(transformed, dtype=np.float64)


def fit_l2_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    l2_regularization: float = TAPNESS_L2_REGULARIZATION,
    maximum_iterations: int = 200,
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, float]:
    """Fit deterministic L2 logistic regression with Newton/IRLS updates."""

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] == 0:
        raise TapnessBaselineError("Logistic training arrays have invalid shapes.")
    if not np.all(np.isfinite(x)) or not np.all(np.isin(y, (0.0, 1.0))):
        raise TapnessBaselineError("Logistic training arrays contain invalid values.")
    if not math.isfinite(float(l2_regularization)) or l2_regularization <= 0.0:
        raise TapnessBaselineError("L2 regularization must be positive and finite.")

    augmented = np.column_stack((np.ones(x.shape[0]), x))
    parameters = np.zeros(augmented.shape[1], dtype=np.float64)
    regularizer = np.diag(
        np.r_[0.0, np.full(x.shape[1], float(l2_regularization))]
    )
    converged = False
    for _ in range(maximum_iterations):
        logits = augmented @ parameters
        probabilities = _stable_sigmoid(logits)
        weights = np.maximum(probabilities * (1.0 - probabilities), 1.0e-12)
        gradient = augmented.T @ (probabilities - y) / x.shape[0]
        gradient += regularizer @ parameters
        hessian = (
            augmented.T @ (augmented * weights[:, None]) / x.shape[0]
            + regularizer
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise TapnessBaselineError("Logistic fitting Hessian was singular.") from error
        parameters -= step
        if float(np.max(np.abs(step))) <= tolerance:
            converged = True
            break
    if not converged:
        raise TapnessBaselineError(
            f"Logistic fitting did not converge within {maximum_iterations} iterations."
        )
    if not np.all(np.isfinite(parameters)):
        raise TapnessBaselineError("Logistic fitting produced non-finite parameters.")
    return parameters[1:].copy(), float(parameters[0])


def classify_tapness_metrics(
    metrics: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply a frozen tapness artifact without fitting or spatial information."""

    validate_tapness_artifact(artifact)
    model = artifact["model"]
    vector = transform_tapness_metrics(metrics)
    means = np.asarray(model["standardization"]["means"], dtype=np.float64)
    scales = np.asarray(model["standardization"]["scales"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    standardized = (vector - means) / scales
    decision_score = float(model["intercept"] + standardized @ coefficients)
    score = float(_stable_sigmoid(np.asarray([decision_score]))[0])
    threshold = float(model["decision_threshold"])
    accepted = score >= threshold
    return {
        "predicted_label": TAPNESS_POSITIVE_LABEL if accepted else TAPNESS_NEGATIVE_LABEL,
        "tap_accepted": accepted,
        "uncalibrated_model_output": score,
        "decision_score": decision_score,
        "decision_threshold": threshold,
        "model_margin": float(score - threshold),
        "tie_rule": "score >= threshold predicts TAP",
        "score_is_calibrated_probability": False,
    }


def create_tapness_baseline(
    session_path: Path,
    *,
    now_fn: Any | None = None,
) -> dict[str, Any]:
    """Fit the fixed Phase 3B.2 model from one development robustness session."""

    # Lazy import avoids coupling pure artifact loading/inference to replay code.
    from desksense.robustness import replay_robustness_dataset

    dataset = load_robustness_dataset(Path(session_path))
    if dataset.session.get("evidence_role") == "external_validation":
        raise TapnessBaselineError(
            "Refusing to fit a tapness baseline from an external_validation "
            "robustness session. External evidence must never enter fitting."
        )
    report = replay_robustness_dataset(Path(session_path))
    examples = _training_examples_from_replay(report)
    positive_count = sum(item["target_label"] == TAPNESS_POSITIVE_LABEL for item in examples)
    negative_count = len(examples) - positive_count
    if positive_count < 2 or negative_count < 2:
        raise TapnessBaselineError("Tapness fitting requires both candidate classes.")

    matrix = np.vstack(
        [transform_tapness_metrics(item["metrics"]) for item in examples]
    )
    labels = np.asarray(
        [item["target_label"] == TAPNESS_POSITIVE_LABEL for item in examples],
        dtype=np.float64,
    )
    groups = [str(item["source_group_id"]) for item in examples]
    folds = grouped_development_folds(examples, TAPNESS_CV_FOLD_COUNT)
    l2_comparison: list[dict[str, Any]] = []
    selected_fit: tuple[np.ndarray, list[dict[str, Any]], float, dict[str, Any]] | None = None
    for l2_value in (1.0, 0.1, 0.01):
        candidate_scores, candidate_folds = _grouped_oof_fit(
            matrix,
            labels,
            groups,
            folds,
            l2_regularization=l2_value,
        )
        candidate_threshold, candidate_selection = select_development_threshold(
            candidate_scores, labels, groups
        )
        candidate_metrics = _score_summary(
            candidate_scores, labels, groups, candidate_threshold
        )
        l2_comparison.append(
            {
                "l2_regularization": l2_value,
                "selected_threshold": candidate_threshold,
                "positive_accepted_count": candidate_metrics["positive_accepted_count"],
                "positive_count": candidate_metrics["positive_count"],
                "negative_false_accept_count": candidate_metrics["negative_false_accept_count"],
                "negative_count": candidate_metrics["negative_count"],
                "maximum_false_accepts_in_one_negative_group": candidate_metrics[
                    "maximum_false_accepts_in_one_negative_group"
                ],
            }
        )
        if l2_value == TAPNESS_L2_REGULARIZATION:
            selected_fit = (
                candidate_scores,
                candidate_folds,
                candidate_threshold,
                candidate_selection,
            )
    if selected_fit is None:
        raise TapnessBaselineError("Configured L2 value was absent from the audit comparison.")
    oof_scores, fold_reports, threshold, selection = selected_fit
    for fold_report, held_out_indices in zip(fold_reports, folds, strict=True):
        held_out = np.asarray(held_out_indices, dtype=int)
        fold_report["held_out_metrics_at_selected_threshold"] = _score_summary(
            oof_scores[held_out],
            labels[held_out],
            [groups[index] for index in held_out],
            threshold,
        )
    cv_metrics = _score_summary(oof_scores, labels, groups, threshold)
    means, scales = _standardization(matrix)
    coefficients, intercept = fit_l2_logistic(
        (matrix - means) / scales,
        labels,
    )
    fingerprint = robustness_dataset_fingerprint(Path(session_path))
    clock = now_fn or (lambda: datetime.now(timezone.utc))
    created = clock()
    if not isinstance(created, datetime):
        raise TapnessBaselineError("Tapness artifact clock must return datetime.")
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    artifact = {
        "artifact_schema_version": TAPNESS_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": TAPNESS_ARTIFACT_TYPE,
        "project_phase": "3B.2",
        "created_at_utc": created.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target": {
            "positive_label": TAPNESS_POSITIVE_LABEL,
            "negative_label": TAPNESS_NEGATIVE_LABEL,
            "task": "tap_vs_non_tap",
            "spatial_labels_or_features_used": False,
        },
        "feature_schema": tapness_feature_schema(),
        "stage1_policy": stage1_policy_metadata(),
        "model": {
            "type": "l2_regularized_binary_logistic_regression",
            "fitting_algorithm": "deterministic Newton/IRLS; intercept unregularized",
            "l2_regularization": TAPNESS_L2_REGULARIZATION,
            "l2_selection": (
                "Session A development-only three-value ablation over "
                "{1.0, 0.1, 0.01}; selected 0.01 for the strongest grouped-CV "
                "positive recall with comparable negative-source burden"
            ),
            "standardization": {
                "means": means.tolist(),
                "scales": scales.tolist(),
                "zero_scale_rule": "replace scale <= 1e-12 with 1.0",
            },
            "coefficients": coefficients.tolist(),
            "intercept": intercept,
            "score": "uncalibrated sigmoid of standardized linear decision score",
            "score_is_calibrated_probability": False,
            "decision_threshold": threshold,
            "tie_rule": "score >= threshold predicts TAP",
            "threshold_selection": selection,
        },
        "source_development_dataset": {
            "session_id": dataset.session["session_id"],
            "evidence_role": "development_only_not_external_validation",
            "dataset_fingerprint": fingerprint,
            "training_candidate_counts": {
                "positive": positive_count,
                "negative": negative_count,
                "total": len(examples),
            },
            "positive_membership_identifiers": [
                item["membership_id"] for item in examples
                if item["target_label"] == TAPNESS_POSITIVE_LABEL
            ],
            "negative_membership_identifiers": [
                item["membership_id"] for item in examples
                if item["target_label"] == TAPNESS_NEGATIVE_LABEL
            ],
            "negative_source_groups": sorted({
                item["source_group_id"] for item in examples
                if item["target_label"] == TAPNESS_NEGATIVE_LABEL
            }),
            "all_candidates_generated_by_required_stage1_policy": True,
            "no_external_samples_used": True,
        },
        "development_cross_validation": {
            "protocol": "five-fold grouped out-of-fold development estimate",
            "group_rule": "all events from one recording remain in one fold",
            "preprocessing_rule": "means/scales/model fit use training fold only",
            "l2_development_comparison": l2_comparison,
            "folds": fold_reports,
            "selected_threshold_metrics": cv_metrics,
            "external_validation": False,
            "post_model_selection_development_evidence": True,
        },
        "evidence_limitations": [
            "Session A was used for feature/model/threshold development.",
            "Grouped cross-validation is development evidence, not external validation.",
            "The model output is not a calibrated probability.",
            "No spatial feature, threshold, margin, or predicted zone is a tapness input.",
            "A new untouched Session B is required after this artifact is frozen.",
        ],
        "privacy": {
            "derived_metadata_only": True,
            "raw_waveforms_stored_in_artifact": False,
            "processing": "local_offline",
        },
    }
    validate_tapness_artifact(artifact)
    json.dumps(artifact, allow_nan=False)
    return artifact


def _grouped_oof_fit(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    folds: Sequence[Sequence[int]],
    *,
    l2_regularization: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit grouped folds with preprocessing learned from each training fold."""

    oof_scores = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    all_indices = set(range(matrix.shape[0]))
    for fold_index, held_out_indices in enumerate(folds):
        held_out_set = set(held_out_indices)
        held_out = np.asarray(tuple(held_out_indices), dtype=int)
        training = np.asarray(sorted(all_indices - held_out_set), dtype=int)
        train_means, train_scales = _standardization(matrix[training])
        coefficients, intercept = fit_l2_logistic(
            (matrix[training] - train_means) / train_scales,
            labels[training],
            l2_regularization=l2_regularization,
        )
        oof_scores[held_out] = _stable_sigmoid(
            intercept
            + ((matrix[held_out] - train_means) / train_scales) @ coefficients
        )
        fold_reports.append(
            {
                "fold_index": fold_index,
                "training_example_count": int(training.size),
                "held_out_example_count": int(held_out.size),
                "held_out_group_ids": sorted({groups[index] for index in held_out}),
                "training_preprocessing_only": True,
            }
        )
    if not np.all(np.isfinite(oof_scores)):
        raise TapnessBaselineError("Grouped development CV left examples unevaluated.")
    return oof_scores, fold_reports


def grouped_development_folds(
    examples: Sequence[Mapping[str, Any]], fold_count: int = TAPNESS_CV_FOLD_COUNT
) -> tuple[tuple[int, ...], ...]:
    """Assign whole source recordings to deterministic class-stratified folds."""

    if fold_count < 2:
        raise TapnessBaselineError("Grouped CV requires at least two folds.")
    groups_by_class: dict[str, list[str]] = defaultdict(list)
    group_label: dict[str, str] = {}
    for example in examples:
        group = str(example["source_group_id"])
        label = str(example["target_label"])
        if group in group_label and group_label[group] != label:
            raise TapnessBaselineError("A CV source group contains mixed labels.")
        group_label[group] = label
    for group, label in group_label.items():
        groups_by_class[label].append(group)
    assignment: dict[str, int] = {}
    for label in (TAPNESS_POSITIVE_LABEL, TAPNESS_NEGATIVE_LABEL):
        for position, group in enumerate(sorted(groups_by_class[label])):
            assignment[group] = position % fold_count
    folds = tuple(
        tuple(
            index for index, example in enumerate(examples)
            if assignment[str(example["source_group_id"])] == fold
        )
        for fold in range(fold_count)
    )
    if any(not fold for fold in folds):
        raise TapnessBaselineError("Grouped CV produced an empty held-out fold.")
    return folds


def select_development_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
) -> tuple[float, dict[str, Any]]:
    """Select a deterministic recall-constrained, repeated-negative-aware threshold."""

    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    unique = sorted(set(float(value) for value in values))
    candidates = [0.0, 1.0]
    candidates.extend(unique)
    candidates.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    required_positive = math.ceil(
        TAPNESS_MINIMUM_CV_POSITIVE_RECALL * int(np.count_nonzero(y == 1.0))
    )
    evaluated: list[tuple[tuple[Any, ...], float, dict[str, Any]]] = []
    for threshold in sorted(set(candidates)):
        summary = _score_summary(values, y, groups, threshold)
        feasible = summary["positive_accepted_count"] >= required_positive
        key = (
            0 if feasible else 1,
            summary["maximum_false_accepts_in_one_negative_group"] if feasible else -summary["positive_accepted_count"],
            summary["negative_false_accept_count"],
            -summary["positive_accepted_count"],
            -threshold,
        )
        evaluated.append((key, threshold, summary))
    _, selected, summary = min(evaluated, key=lambda item: item[0])
    return float(selected), {
        "method": (
            "out-of-fold grouped scores; require at least 90% positive recall "
            "when feasible, then minimize maximum false accepts from one "
            "negative recording, total negative false accepts, maximize positive "
            "accepts, and choose the highest remaining threshold"
        ),
        "minimum_positive_recall_constraint": TAPNESS_MINIMUM_CV_POSITIVE_RECALL,
        "required_positive_accept_count": required_positive,
        "selected_threshold": float(selected),
        "selected_out_of_fold_metrics": summary,
        "session_a_tuned": True,
    }


def robustness_dataset_fingerprint(session_path: Path) -> dict[str, Any]:
    """Bind exact session, manifest, and manifest-ordered NPZ bytes with SHA-256."""

    dataset = load_robustness_dataset(Path(session_path))
    components = [dataset.directory / "session.json", dataset.directory / "manifest.jsonl"]
    components.extend(record.path for record in dataset.records)
    digest = hashlib.sha256()
    digest.update(b"DeskSense Phase3 robustness fingerprint v1\0")
    reported: list[dict[str, Any]] = []
    for path in components:
        relative = path.relative_to(dataset.directory).as_posix()
        data = path.read_bytes()
        name = relative.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        reported.append(
            {
                "relative_path": relative,
                "byte_count": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "algorithm": "SHA-256",
        "framing": "v1 domain prefix then uint64be name length/name/uint64be byte length/exact bytes",
        "logical_order": "session.json, manifest.jsonl, then manifest record order",
        "sha256": digest.hexdigest(),
        "component_count": len(reported),
        "components": reported,
    }


def write_tapness_baseline(artifact: Mapping[str, Any], path: Path) -> Path:
    """Strictly validate and exclusively persist one compact JSON artifact."""

    validate_tapness_artifact(artifact)
    target = Path(path)
    if target.suffix.casefold() != ".json":
        raise TapnessBaselineError("Tapness baseline path must end in .json.")
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as output:
            output.write(serialized)
            output.write("\n")
    except FileExistsError as error:
        raise TapnessBaselineError(
            f"Refusing to overwrite existing tapness baseline: {target}"
        ) from error
    return target.resolve()


def load_tapness_baseline(path: Path) -> dict[str, Any]:
    """Load and strictly validate one tapness baseline JSON artifact."""

    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TapnessBaselineError(f"Could not load tapness baseline: {error}") from error
    if not isinstance(artifact, dict):
        raise TapnessBaselineError("Tapness baseline must be a JSON object.")
    validate_tapness_artifact(artifact)
    return artifact


def validate_tapness_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject malformed, inconsistent, non-finite, or spatially coupled artifacts."""

    if not isinstance(artifact, Mapping):
        raise TapnessBaselineError("Tapness artifact must be a mapping.")
    _require_exact_keys(
        artifact,
        {
            "artifact_schema_version",
            "artifact_type",
            "project_phase",
            "created_at_utc",
            "target",
            "feature_schema",
            "stage1_policy",
            "model",
            "source_development_dataset",
            "development_cross_validation",
            "evidence_limitations",
            "privacy",
        },
        "tapness artifact",
    )
    if artifact.get("artifact_schema_version") != TAPNESS_ARTIFACT_SCHEMA_VERSION:
        raise TapnessBaselineError("Unsupported tapness artifact schema version.")
    if artifact.get("artifact_type") != TAPNESS_ARTIFACT_TYPE:
        raise TapnessBaselineError("Unexpected tapness artifact type.")
    if artifact.get("project_phase") != "3B.2":
        raise TapnessBaselineError("Tapness artifact project phase must be 3B.2.")
    _require_utc_timestamp(artifact.get("created_at_utc"), "created_at_utc")
    if artifact.get("feature_schema") != tapness_feature_schema():
        raise TapnessBaselineError("Tapness feature schema does not match version 1.")
    if artifact.get("stage1_policy") != stage1_policy_metadata():
        raise TapnessBaselineError("Tapness artifact Stage 1 policy is incompatible.")
    target = artifact.get("target")
    if not isinstance(target, Mapping):
        raise TapnessBaselineError("Tapness target metadata is missing.")
    _require_exact_keys(
        target,
        {"positive_label", "negative_label", "task", "spatial_labels_or_features_used"},
        "tapness target",
    )
    if target != {
        "positive_label": TAPNESS_POSITIVE_LABEL,
        "negative_label": TAPNESS_NEGATIVE_LABEL,
        "task": "tap_vs_non_tap",
        "spatial_labels_or_features_used": False,
    }:
        raise TapnessBaselineError("Tapness target metadata is invalid or spatially coupled.")
    model = artifact.get("model")
    if not isinstance(model, Mapping) or model.get("type") != "l2_regularized_binary_logistic_regression":
        raise TapnessBaselineError("Tapness model metadata is invalid.")
    _require_exact_keys(
        model,
        {
            "type",
            "fitting_algorithm",
            "l2_regularization",
            "l2_selection",
            "standardization",
            "coefficients",
            "intercept",
            "score",
            "score_is_calibrated_probability",
            "decision_threshold",
            "tie_rule",
            "threshold_selection",
        },
        "tapness model",
    )
    if model.get("score_is_calibrated_probability") is not False:
        raise TapnessBaselineError("Tapness score must be declared uncalibrated.")
    if model.get("tie_rule") != "score >= threshold predicts TAP":
        raise TapnessBaselineError("Tapness tie rule is invalid.")
    count = len(TAPNESS_FEATURE_NAMES)
    standardization = model.get("standardization")
    if not isinstance(standardization, Mapping):
        raise TapnessBaselineError("Tapness standardization metadata is missing.")
    _require_exact_keys(
        standardization, {"means", "scales", "zero_scale_rule"}, "standardization"
    )
    for name, values, positive in (
        ("means", standardization.get("means"), False),
        ("scales", standardization.get("scales"), True),
        ("coefficients", model.get("coefficients"), False),
    ):
        if not isinstance(values, list) or len(values) != count:
            raise TapnessBaselineError(f"Tapness {name} must contain {count} values.")
        if any(not _finite_number(value) or (positive and float(value) <= 0.0) for value in values):
            raise TapnessBaselineError(f"Tapness {name} contains invalid values.")
    for name in ("intercept", "decision_threshold", "l2_regularization"):
        if not _finite_number(model.get(name)):
            raise TapnessBaselineError(f"Tapness model {name} must be finite.")
    if float(model["l2_regularization"]) <= 0.0:
        raise TapnessBaselineError("Tapness L2 regularization must be positive.")
    threshold = float(model["decision_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise TapnessBaselineError("Tapness decision threshold must be in [0, 1].")
    source = artifact.get("source_development_dataset")
    if not isinstance(source, Mapping) or not isinstance(source.get("session_id"), str) or not source["session_id"]:
        raise TapnessBaselineError("Tapness source session identity is missing.")
    _require_exact_keys(
        source,
        {
            "session_id",
            "evidence_role",
            "dataset_fingerprint",
            "training_candidate_counts",
            "positive_membership_identifiers",
            "negative_membership_identifiers",
            "negative_source_groups",
            "all_candidates_generated_by_required_stage1_policy",
            "no_external_samples_used",
        },
        "source development dataset",
    )
    if source.get("evidence_role") != "development_only_not_external_validation":
        raise TapnessBaselineError("Tapness source evidence role must be development-only.")
    fingerprint = source.get("dataset_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise TapnessBaselineError("Tapness source dataset fingerprint is invalid.")
    _require_exact_keys(
        fingerprint,
        {"algorithm", "framing", "logical_order", "sha256", "component_count", "components"},
        "dataset fingerprint",
    )
    if fingerprint.get("algorithm") != "SHA-256" or not _valid_sha256(fingerprint.get("sha256")):
        raise TapnessBaselineError("Tapness source dataset fingerprint is invalid.")
    if fingerprint.get("framing") != "v1 domain prefix then uint64be name length/name/uint64be byte length/exact bytes":
        raise TapnessBaselineError("Tapness fingerprint framing is unsupported.")
    if fingerprint.get("logical_order") != "session.json, manifest.jsonl, then manifest record order":
        raise TapnessBaselineError("Tapness fingerprint logical order is invalid.")
    components = fingerprint.get("components")
    component_count = fingerprint.get("component_count")
    if (
        isinstance(component_count, bool)
        or not isinstance(component_count, int)
        or component_count < 2
        or not isinstance(components, list)
        or len(components) != component_count
    ):
        raise TapnessBaselineError("Tapness fingerprint component count is invalid.")
    component_paths: set[str] = set()
    for component in components:
        if not isinstance(component, Mapping):
            raise TapnessBaselineError("Tapness fingerprint component is invalid.")
        _require_exact_keys(
            component, {"relative_path", "byte_count", "sha256"}, "fingerprint component"
        )
        relative_path = component.get("relative_path")
        byte_count = component.get("byte_count")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path in component_paths
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not _valid_sha256(component.get("sha256"))
        ):
            raise TapnessBaselineError("Tapness fingerprint component is invalid.")
        component_paths.add(relative_path)
    if components[0]["relative_path"] != "session.json" or components[1]["relative_path"] != "manifest.jsonl":
        raise TapnessBaselineError("Tapness fingerprint component order is invalid.")

    counts = source.get("training_candidate_counts")
    if not isinstance(counts, Mapping):
        raise TapnessBaselineError("Tapness training candidate counts are missing.")
    _require_exact_keys(counts, {"positive", "negative", "total"}, "training counts")
    if any(
        isinstance(counts.get(name), bool)
        or not isinstance(counts.get(name), int)
        or int(counts[name]) <= 0
        for name in ("positive", "negative", "total")
    ) or counts["positive"] + counts["negative"] != counts["total"]:
        raise TapnessBaselineError("Tapness training candidate counts are inconsistent.")
    positives = _membership_list(source.get("positive_membership_identifiers"), "positive")
    negatives = _membership_list(source.get("negative_membership_identifiers"), "negative")
    if len(positives) != counts["positive"] or len(negatives) != counts["negative"]:
        raise TapnessBaselineError("Tapness membership lengths do not match training counts.")
    if set(positives) & set(negatives):
        raise TapnessBaselineError("Tapness positive and negative memberships overlap.")
    negative_groups = _membership_list(source.get("negative_source_groups"), "negative source group")
    derived_negative_groups = {
        membership.split(":onset:", 1)[0]
        for membership in negatives
        if ":onset:" in membership
    }
    if len(derived_negative_groups) != len(negative_groups) or derived_negative_groups != set(negative_groups):
        raise TapnessBaselineError("Tapness negative source-group provenance is inconsistent.")
    if source.get("no_external_samples_used") is not True:
        raise TapnessBaselineError("Tapness artifact must declare no external samples used.")
    if source.get("all_candidates_generated_by_required_stage1_policy") is not True:
        raise TapnessBaselineError("Tapness training candidates must match the required Stage 1 policy.")

    cross_validation = artifact.get("development_cross_validation")
    if not isinstance(cross_validation, Mapping):
        raise TapnessBaselineError("Tapness development cross-validation metadata is missing.")
    _require_exact_keys(
        cross_validation,
        {
            "protocol",
            "group_rule",
            "preprocessing_rule",
            "l2_development_comparison",
            "folds",
            "selected_threshold_metrics",
            "external_validation",
            "post_model_selection_development_evidence",
        },
        "development cross-validation",
    )
    if cross_validation.get("external_validation") is not False or cross_validation.get("post_model_selection_development_evidence") is not True:
        raise TapnessBaselineError("Tapness CV must be explicitly development-only.")
    selected = cross_validation.get("selected_threshold_metrics")
    if not isinstance(selected, Mapping):
        raise TapnessBaselineError("Tapness selected-threshold metrics are missing.")
    _validate_score_summary(selected, threshold, counts)
    threshold_selection = model.get("threshold_selection")
    if not isinstance(threshold_selection, Mapping):
        raise TapnessBaselineError("Tapness threshold-selection metadata is missing.")
    _require_exact_keys(
        threshold_selection,
        {
            "method",
            "minimum_positive_recall_constraint",
            "required_positive_accept_count",
            "selected_threshold",
            "selected_out_of_fold_metrics",
            "session_a_tuned",
        },
        "threshold selection",
    )
    if (
        not _finite_number(threshold_selection.get("selected_threshold"))
        or float(threshold_selection["selected_threshold"]) != threshold
        or threshold_selection.get("session_a_tuned") is not True
        or threshold_selection.get("minimum_positive_recall_constraint")
        != TAPNESS_MINIMUM_CV_POSITIVE_RECALL
    ):
        raise TapnessBaselineError("Tapness threshold-selection metadata is inconsistent.")
    required_positive = threshold_selection.get("required_positive_accept_count")
    if (
        isinstance(required_positive, bool)
        or not isinstance(required_positive, int)
        or required_positive != math.ceil(
            TAPNESS_MINIMUM_CV_POSITIVE_RECALL * counts["positive"]
        )
    ):
        raise TapnessBaselineError(
            "Tapness threshold-selection positive constraint is inconsistent."
        )
    selection_metrics = threshold_selection.get("selected_out_of_fold_metrics")
    if selection_metrics != selected:
        raise TapnessBaselineError("Tapness selected OOF metrics disagree across artifact sections.")

    l2_comparison = cross_validation.get("l2_development_comparison")
    if not isinstance(l2_comparison, list) or [item.get("l2_regularization") for item in l2_comparison if isinstance(item, Mapping)] != [1.0, 0.1, 0.01]:
        raise TapnessBaselineError("Tapness L2 development comparison is invalid.")
    for item in l2_comparison:
        _require_exact_keys(
            item,
            {
                "l2_regularization",
                "selected_threshold",
                "positive_accepted_count",
                "positive_count",
                "negative_false_accept_count",
                "negative_count",
                "maximum_false_accepts_in_one_negative_group",
            },
            "L2 development comparison entry",
        )
        if (
            not _finite_number(item["selected_threshold"])
            or not 0.0 <= float(item["selected_threshold"]) <= 1.0
            or item["positive_count"] != counts["positive"]
            or item["negative_count"] != counts["negative"]
            or isinstance(item["positive_accepted_count"], bool)
            or not isinstance(item["positive_accepted_count"], int)
            or not 0 <= item["positive_accepted_count"] <= item["positive_count"]
            or isinstance(item["negative_false_accept_count"], bool)
            or not isinstance(item["negative_false_accept_count"], int)
            or not 0 <= item["negative_false_accept_count"] <= item["negative_count"]
            or isinstance(item["maximum_false_accepts_in_one_negative_group"], bool)
            or not isinstance(item["maximum_false_accepts_in_one_negative_group"], int)
            or not 0
            <= item["maximum_false_accepts_in_one_negative_group"]
            <= item["negative_false_accept_count"]
        ):
            raise TapnessBaselineError("Tapness L2 comparison threshold is invalid.")
    matching_l2 = [
        item
        for item in l2_comparison
        if item["l2_regularization"] == float(model["l2_regularization"])
    ]
    if len(matching_l2) != 1:
        raise TapnessBaselineError(
            "Frozen L2 value is absent from the development comparison."
        )
    selected_l2 = matching_l2[0]
    if (
        float(selected_l2["selected_threshold"]) != threshold
        or selected_l2["positive_accepted_count"]
        != selected["positive_accepted_count"]
        or selected_l2["negative_false_accept_count"]
        != selected["negative_false_accept_count"]
        or selected_l2["maximum_false_accepts_in_one_negative_group"]
        != selected["maximum_false_accepts_in_one_negative_group"]
    ):
        raise TapnessBaselineError(
            "Selected L2 comparison does not match the frozen development result."
        )

    folds = cross_validation.get("folds")
    if not isinstance(folds, list) or len(folds) != TAPNESS_CV_FOLD_COUNT:
        raise TapnessBaselineError("Tapness grouped CV fold list is invalid.")
    held_out_total = 0
    fold_positive_count = 0
    fold_positive_accepted = 0
    fold_negative_count = 0
    fold_negative_false_accepts = 0
    held_out_groups: set[str] = set()
    seen_negative_groups: set[str] = set()
    for expected_index, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            raise TapnessBaselineError("Tapness grouped CV fold is invalid.")
        _require_exact_keys(
            fold,
            {
                "fold_index",
                "training_example_count",
                "held_out_example_count",
                "held_out_group_ids",
                "training_preprocessing_only",
                "held_out_metrics_at_selected_threshold",
            },
            "grouped CV fold",
        )
        held_count = fold.get("held_out_example_count")
        train_count = fold.get("training_example_count")
        if (
            fold.get("fold_index") != expected_index
            or fold.get("training_preprocessing_only") is not True
            or isinstance(held_count, bool)
            or not isinstance(held_count, int)
            or held_count <= 0
            or isinstance(train_count, bool)
            or not isinstance(train_count, int)
            or train_count + held_count != counts["total"]
        ):
            raise TapnessBaselineError("Tapness grouped CV fold counts are inconsistent.")
        groups = _membership_list(fold.get("held_out_group_ids"), "held-out group")
        if held_out_groups & set(groups):
            raise TapnessBaselineError("A recording group appears in multiple held-out folds.")
        held_out_groups.update(groups)
        fold_negative_groups = set(groups) & set(negative_groups)
        if seen_negative_groups & fold_negative_groups:
            raise TapnessBaselineError("A negative recording group crosses CV folds.")
        seen_negative_groups.update(fold_negative_groups)
        fold_metrics = fold.get("held_out_metrics_at_selected_threshold")
        if not isinstance(fold_metrics, Mapping):
            raise TapnessBaselineError("Tapness held-out fold metrics are missing.")
        _validate_score_summary(fold_metrics, threshold, None)
        if fold_metrics["positive_count"] + fold_metrics["negative_count"] != held_count:
            raise TapnessBaselineError("Tapness held-out fold metrics do not match its count.")
        held_out_total += held_count
        fold_positive_count += fold_metrics["positive_count"]
        fold_positive_accepted += fold_metrics["positive_accepted_count"]
        fold_negative_count += fold_metrics["negative_count"]
        fold_negative_false_accepts += fold_metrics["negative_false_accept_count"]
    if held_out_total != counts["total"]:
        raise TapnessBaselineError("Tapness held-out folds do not cover all candidates exactly once.")
    if seen_negative_groups != set(negative_groups):
        raise TapnessBaselineError("Tapness negative groups are not fully represented in held-out folds.")
    positive_groups = {
        membership.split(":onset:", 1)[0]
        for membership in positives
        if ":onset:" in membership
    }
    if held_out_groups != positive_groups | set(negative_groups):
        raise TapnessBaselineError(
            "Tapness held-out folds do not contain the exact training source groups."
        )
    if (
        fold_positive_count != selected["positive_count"]
        or fold_positive_accepted != selected["positive_accepted_count"]
        or fold_negative_count != selected["negative_count"]
        or fold_negative_false_accepts
        != selected["negative_false_accept_count"]
    ):
        raise TapnessBaselineError(
            "Tapness held-out fold metrics do not aggregate to the selected result."
        )

    limitations = artifact.get("evidence_limitations")
    if not isinstance(limitations, list) or not limitations or any(
        not isinstance(item, str) or not item for item in limitations
    ):
        raise TapnessBaselineError("Tapness evidence limitations are invalid.")
    privacy = artifact.get("privacy")
    if not isinstance(privacy, Mapping):
        raise TapnessBaselineError("Tapness privacy metadata is invalid.")
    _require_exact_keys(
        privacy,
        {"derived_metadata_only", "raw_waveforms_stored_in_artifact", "processing"},
        "tapness privacy metadata",
    )
    if privacy.get("derived_metadata_only") is not True or privacy.get("raw_waveforms_stored_in_artifact") is not False or privacy.get("processing") != "local_offline":
        raise TapnessBaselineError("Tapness privacy metadata is inconsistent.")

    forbidden = {"candidate_window", "capture", "waveform", "audio_samples"}
    if _contains_forbidden_key(artifact, forbidden):
        raise TapnessBaselineError("Tapness artifact contains a forbidden waveform key.")
    try:
        json.dumps(artifact, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TapnessBaselineError("Tapness artifact is not finite JSON.") from error


def validate_tapness_stage1_config(config: Any, artifact: Mapping[str, Any]) -> None:
    """Fail before inference when a detector does not match the artifact policy."""

    validate_tapness_artifact(artifact)
    if config is None:
        raise TapnessBaselineError("Detector configuration is required for Stage 1 compatibility.")
    try:
        actual = stage1_policy_metadata(config)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise TapnessBaselineError(
            "Detector configuration is missing or has invalid material Stage 1 fields."
        ) from error
    expected = artifact["stage1_policy"]
    if actual != expected:
        differences = _mapping_differences(expected, actual)
        raise TapnessBaselineError(
            "Detector is incompatible with the frozen tapness Stage 1 policy: "
            + ", ".join(differences[:8])
        )


def format_tapness_baseline_summary(artifact: Mapping[str, Any]) -> str:
    counts = artifact["source_development_dataset"]["training_candidate_counts"]
    cv = artifact["development_cross_validation"]["selected_threshold_metrics"]
    return "\n".join(
        [
            "DeskSense Phase 3B.2 tapness baseline (development evidence)",
            f"Session: {artifact['source_development_dataset']['session_id']}",
            f"Candidates: {counts['positive']} TAP, {counts['negative']} NON_TAP",
            f"Grouped-CV TAP accepted: {cv['positive_accepted_count']}/{cv['positive_count']}",
            f"Grouped-CV NON_TAP false accepts: {cv['negative_false_accept_count']}/{cv['negative_count']}",
            f"Frozen uncalibrated-score threshold: {artifact['model']['decision_threshold']:.12g}",
            "This is Session A development fitting, not external validation.",
        ]
    )


def _training_examples_from_replay(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for attempt in report["positive"]["attempts"]:
        for event_index, event in enumerate(attempt["completed_events"]):
            if event["interval_relation"] != "associated" or event["status"] != "detected":
                continue
            metrics = event["descriptive_tapness_metrics"]
            if metrics is None:
                continue
            examples.append(
                {
                    "membership_id": f"{attempt['record_id']}:onset:{event['onset_frame_index']}",
                    "source_group_id": attempt["record_id"],
                    "target_label": TAPNESS_POSITIVE_LABEL,
                    "metrics": metrics,
                    "zone": attempt["intended_zone"],
                    "strength": attempt["intended_strength"],
                }
            )
    for segment in report["negative"]["segments"]:
        for event in segment["completed_events"]:
            if event["interval_relation"] != "activity" or event["status"] != "detected":
                continue
            metrics = event["descriptive_tapness_metrics"]
            if metrics is None:
                continue
            examples.append(
                {
                    "membership_id": f"{segment['record_id']}:onset:{event['onset_frame_index']}",
                    "source_group_id": segment["record_id"],
                    "target_label": TAPNESS_NEGATIVE_LABEL,
                    "metrics": metrics,
                    "activity": segment["activity"],
                }
            )
    return examples


def _standardization(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales = np.where(scales <= 1.0e-12, 1.0, scales)
    return means, scales


def _score_summary(
    scores: np.ndarray, labels: np.ndarray, groups: Sequence[str], threshold: float
) -> dict[str, Any]:
    accepted = np.asarray(scores) >= float(threshold)
    positive = np.asarray(labels) == 1.0
    negative_indices = np.flatnonzero(~positive & accepted)
    group_counts = Counter(groups[index] for index in negative_indices)
    positive_count = int(np.count_nonzero(positive))
    positive_accepted = int(np.count_nonzero(positive & accepted))
    negative_count = int(np.count_nonzero(~positive))
    negative_accepted = int(negative_indices.size)
    return {
        "threshold": float(threshold),
        "positive_count": positive_count,
        "positive_accepted_count": positive_accepted,
        "positive_recall": float(positive_accepted / positive_count),
        "negative_count": negative_count,
        "negative_false_accept_count": negative_accepted,
        "negative_false_accept_rate": float(negative_accepted / negative_count),
        "negative_groups_with_false_accepts": len(group_counts),
        "maximum_false_accepts_in_one_negative_group": max(group_counts.values(), default=0),
        "false_accepts_by_negative_group": dict(sorted(group_counts.items())),
    }


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    nonnegative = array >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))
    exponential = np.exp(array[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TapnessBaselineError(
            f"{label.capitalize()} keys are invalid; missing={missing}, extra={extra}."
        )


def _membership_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise TapnessBaselineError(
            f"Tapness {label} identifiers must be non-empty and unique."
        )
    return value


def _validate_score_summary(
    summary: Mapping[str, Any],
    threshold: float,
    source_counts: Mapping[str, Any] | None,
) -> None:
    expected_keys = {
        "threshold",
        "positive_count",
        "positive_accepted_count",
        "positive_recall",
        "negative_count",
        "negative_false_accept_count",
        "negative_false_accept_rate",
        "negative_groups_with_false_accepts",
        "maximum_false_accepts_in_one_negative_group",
        "false_accepts_by_negative_group",
    }
    _require_exact_keys(summary, expected_keys, "score summary")
    if not _finite_number(summary.get("threshold")) or float(summary["threshold"]) != threshold:
        raise TapnessBaselineError("Tapness score-summary threshold is inconsistent.")
    integer_names = (
        "positive_count",
        "positive_accepted_count",
        "negative_count",
        "negative_false_accept_count",
        "negative_groups_with_false_accepts",
        "maximum_false_accepts_in_one_negative_group",
    )
    if any(
        isinstance(summary.get(name), bool)
        or not isinstance(summary.get(name), int)
        or int(summary[name]) < 0
        for name in integer_names
    ):
        raise TapnessBaselineError("Tapness score-summary counts are invalid.")
    if summary["positive_accepted_count"] > summary["positive_count"] or summary["negative_false_accept_count"] > summary["negative_count"]:
        raise TapnessBaselineError("Tapness score-summary accepted counts are invalid.")
    if source_counts is not None and (
        summary["positive_count"] != source_counts["positive"]
        or summary["negative_count"] != source_counts["negative"]
    ):
        raise TapnessBaselineError("Tapness selected-threshold counts disagree with source counts.")
    for name in ("positive_recall", "negative_false_accept_rate"):
        if not _finite_number(summary.get(name)) or not 0.0 <= float(summary[name]) <= 1.0:
            raise TapnessBaselineError("Tapness score-summary rate is invalid.")
    group_counts = summary.get("false_accepts_by_negative_group")
    if not isinstance(group_counts, Mapping) or any(
        not isinstance(group, str)
        or not group
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        for group, count in group_counts.items()
    ):
        raise TapnessBaselineError("Tapness score-summary group counts are invalid.")
    if sum(group_counts.values()) != summary["negative_false_accept_count"]:
        raise TapnessBaselineError("Tapness score-summary group totals are inconsistent.")
    if len(group_counts) != summary["negative_groups_with_false_accepts"] or max(group_counts.values(), default=0) != summary["maximum_false_accepts_in_one_negative_group"]:
        raise TapnessBaselineError("Tapness score-summary group burden is inconsistent.")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_utc_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise TapnessBaselineError(f"Tapness {label} timestamp is missing.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise TapnessBaselineError(
            f"Tapness {label} is not a valid ISO timestamp."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise TapnessBaselineError(f"Tapness {label} must be UTC.")


def _contains_forbidden_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).casefold() in forbidden or _contains_forbidden_key(item, forbidden) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False


def _mapping_differences(
    expected: Any, actual: Any, prefix: str = "stage1_policy"
) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{prefix}.{key}"
            if key not in expected or key not in actual:
                differences.append(child)
            else:
                differences.extend(
                    _mapping_differences(expected[key], actual[key], child)
                )
        return differences
    return [] if expected == actual else [prefix]


def _pooled_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(math.sqrt(max(0.0, float(np.mean(np.square(samples))))))


def _finite_nonnegative(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) and converted >= 0.0 else None
