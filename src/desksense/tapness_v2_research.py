"""Phase 3B.3 research-only Stage 2 v2 extraction and development evaluation.

This module reproduces the two-feature representation selected in Experiment
19.  It is deliberately not wired into live sensing and cannot write a frozen
model artifact.  Sessions A and B are development evidence for this work; a
changed pipeline requires a new untouched Session C.

The deferred 100--180 ms late-context feature reduced development false
accepts, but also reduced tap survival and would add roughly 100 ms of runtime
history/latency.  It is intentionally absent from the v2 feature vector.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from desksense.robustness_dataset import (
    LoadedRobustnessDataset,
    NEGATIVE_RECORD_TYPE,
    POSITIVE_RECORD_TYPE,
    load_robustness_dataset,
)
from desksense.streaming import StreamingTapDetector
from desksense.tapness import fit_l2_logistic, robustness_dataset_fingerprint


TAPNESS_V2_RESEARCH_FEATURE_SCHEMA_VERSION = 1
TAPNESS_V2_RESEARCH_REPORT_SCHEMA_VERSION = 1
TAPNESS_V2_RESEARCH_FEATURE_NAMES = (
    "spectral_bandwidth_hz",
    "post_0_100_zero_crossing_rate",
)
TAPNESS_V2_RESEARCH_FEATURE_TRANSFORMS = (
    "natural_log_with_fixed_epsilon",
    "natural_log_with_fixed_epsilon",
)
TAPNESS_V2_RESEARCH_LOG_EPSILON = 1.0e-12
TAPNESS_V2_RESEARCH_L2 = 0.01
TAPNESS_V2_RESEARCH_MINIMUM_POSITIVE_RECALL = 0.95
TAPNESS_V2_RESEARCH_FOLD_COUNT = 5

_SAMPLE_RATE_HZ = 48_000.0
_CHANNEL_COUNT = 2
_CANDIDATE_FRAMES = 9_600
_CENTER_OFFSET_FRAMES = 4_800
_SPECTRAL_SECONDS = 0.050
_POST_ONSET_ZCR_SECONDS = 0.100
_DC_REFERENCE_SECONDS = 0.012
_SPECTRAL_TOTAL_EPSILON = 1.0e-18

_EXPERIMENT_19_SESSION_A_ID = "20260901T131308.362205Z-f9e2b1ec"
_EXPERIMENT_19_SESSION_B_ID = "20260904T171453.458221Z-4ecb16ad"
_EXPERIMENT_19_SESSION_A_FINGERPRINT = (
    "88a003966141d15858a2afec390c42e567439eb5f538b958e7aa60ee7638ab83"
)
_EXPERIMENT_19_SESSION_B_FINGERPRINT = (
    "6b6b7cff6ff4f597d0c4c9bc8ccea3544cda99e4d25bf9be1cc732ee6b9ee5ac"
)


class TapnessV2ResearchError(RuntimeError):
    """A research extraction, replay, or grouped-evaluation failure."""


@dataclass(frozen=True)
class ResearchLogisticModel:
    """One in-memory development fit; this is not a frozen artifact."""

    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    l2_regularization: float

    def score(self, matrix: Any) -> np.ndarray[Any, Any]:
        """Return uncalibrated sigmoid outputs for transformed feature rows."""

        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.means):
            raise TapnessV2ResearchError("Research scoring matrix has invalid shape.")
        if not np.all(np.isfinite(values)):
            raise TapnessV2ResearchError("Research scoring matrix must be finite.")
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        decisions = self.intercept + ((values - means) / scales) @ coefficients
        return _stable_sigmoid(decisions)

    def to_report(self) -> dict[str, Any]:
        """Return JSON-safe development parameters with an explicit boundary."""

        return {
            "status": "research_only_not_a_frozen_inference_artifact",
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "l2_regularization": self.l2_regularization,
            "score": "uncalibrated sigmoid output",
        }


def tapness_v2_research_feature_schema() -> dict[str, Any]:
    """Return the immutable Experiment 19 feature definitions and order."""

    spectral_frames = round(_SAMPLE_RATE_HZ * _SPECTRAL_SECONDS)
    zcr_frames = round(_SAMPLE_RATE_HZ * _POST_ONSET_ZCR_SECONDS)
    return {
        "version": TAPNESS_V2_RESEARCH_FEATURE_SCHEMA_VERSION,
        "status": "research_only_not_live_or_frozen",
        "ordered_feature_names": list(TAPNESS_V2_RESEARCH_FEATURE_NAMES),
        "features": [
            {
                "name": "spectral_bandwidth_hz",
                "units": "Hz",
                "candidate_interval": {
                    "start": "refined center, inclusive",
                    "end": "refined center + 50 ms, exclusive",
                    "start_offset_frames": _CENTER_OFFSET_FRAMES,
                    "end_offset_frames_exclusive": (
                        _CENTER_OFFSET_FRAMES + spectral_frames
                    ),
                    "frame_count": spectral_frames,
                },
                "dc_reference": (
                    "subtract each channel's mean over up to 12 ms immediately "
                    "before detector onset; use the complete candidate only when "
                    "that reference is empty"
                ),
                "window": (
                    "NumPy symmetric Hann: np.hanning(2400), equivalent to "
                    "0.5 - 0.5*cos(2*pi*n/(M-1))"
                ),
                "fft": (
                    "np.fft.rfft independently by channel with n=2400 and no "
                    "normalization; bins from np.fft.rfftfreq(2400, 1/48000)"
                ),
                "power": "sum(abs(channel_rfft)**2) across the two channels",
                "dc_treatment": "set the pooled zero-frequency power bin to zero",
                "centroid": "sum(frequency * power) / max(sum(power), 1e-18)",
                "definition": (
                    "sqrt(sum((frequency-centroid)**2 * power) / "
                    "max(sum(power), 1e-18))"
                ),
                "zero_energy_result": 0.0,
                "transformation": "natural_log_with_fixed_epsilon",
                "log_epsilon": TAPNESS_V2_RESEARCH_LOG_EPSILON,
            },
            {
                "name": "post_0_100_zero_crossing_rate",
                "units": "fraction_of_adjacent_samples",
                "candidate_interval": {
                    "start": "detector onset, inclusive",
                    "end": "min(candidate end, onset + 100 ms), exclusive",
                    "maximum_frame_count": zcr_frames,
                },
                "definition": (
                    "for each channel compare np.signbit(sample[i+1]) != "
                    "np.signbit(sample[i]); average all adjacent comparisons "
                    "across time and the two channels"
                ),
                "zero_rule": (
                    "positive zero is nonnegative under np.signbit; IEEE negative "
                    "zero retains its negative sign bit"
                ),
                "denominator": (
                    "(interval_frame_count - 1) * 2 channels; return zero when "
                    "fewer than two interval frames are available"
                ),
                "transformation": "natural_log_with_fixed_epsilon",
                "log_epsilon": TAPNESS_V2_RESEARCH_LOG_EPSILON,
            },
        ],
        "candidate_domain": {
            "sample_rate_hz": _SAMPLE_RATE_HZ,
            "channel_count": _CHANNEL_COUNT,
            "frame_count": _CANDIDATE_FRAMES,
            "center_offset_frames": _CENTER_OFFSET_FRAMES,
            "raw_candidate_modified": False,
        },
        "spatial_inputs_used": False,
        "excluded_inputs": [
            "LEFT/RIGHT labels",
            "signed channel amplitude ratio",
            "spatial threshold",
            "predicted spatial zone",
            "spatial margin",
        ],
        "finite_value_handling": (
            "candidate samples and extracted features must be finite; invalid "
            "values are rejected"
        ),
        "late_context_ablation": {
            "included": False,
            "reason": (
                "100-180 ms context improved development negative rejection but "
                "reduced tap recall and adds roughly 100 ms history/latency"
            ),
        },
    }


def extract_tapness_v2_research_features(
    candidate_window: Any,
    *,
    sample_rate_hz: float,
    onset_frame_index: int,
    center_frame_index: int,
    window_start_frame_index: int,
    window_end_frame_index_exclusive: int,
) -> dict[str, float]:
    """Extract the exact two Experiment 19 candidate-only descriptors.

    Absolute timing metadata is required so the detector onset can be mapped
    into the retained candidate.  The refined center must remain at candidate
    index 4,800.  Work occurs on a float64 copy and never mutates the caller's
    raw candidate.
    """

    audio = _validate_candidate(candidate_window)
    sample_rate = _finite_float(sample_rate_hz, "sample rate")
    if sample_rate != _SAMPLE_RATE_HZ:
        raise ValueError("Stage 2 v2 research requires exactly 48000 Hz audio.")
    start = _integer_frame(window_start_frame_index, "window start")
    end = _integer_frame(window_end_frame_index_exclusive, "window end")
    onset = _integer_frame(onset_frame_index, "onset")
    center = _integer_frame(center_frame_index, "center")
    if end - start != _CANDIDATE_FRAMES:
        raise ValueError("Candidate frame metadata must describe exactly 9600 frames.")
    if center - start != _CENTER_OFFSET_FRAMES:
        raise ValueError("Refined center must be at candidate index 4800.")
    if not start <= onset < end:
        raise ValueError("Detector onset must lie inside the candidate window.")

    onset_offset = onset - start
    center_offset = center - start
    reference_frames = round(sample_rate * _DC_REFERENCE_SECONDS)
    reference = audio[max(0, onset_offset - reference_frames) : onset_offset]
    if reference.shape[0] == 0:
        reference = audio
    centered = audio - np.mean(reference, axis=0, keepdims=True)

    spectral_frames = round(sample_rate * _SPECTRAL_SECONDS)
    spectral_audio = centered[center_offset : center_offset + spectral_frames]
    if spectral_audio.shape != (spectral_frames, _CHANNEL_COUNT):
        raise ValueError("The complete fixed 50 ms spectral interval is unavailable.")
    hann = np.hanning(spectral_frames)[:, None]
    spectrum = np.fft.rfft(spectral_audio * hann, axis=0)
    spectral_power = np.sum(np.square(np.abs(spectrum)), axis=1)
    frequencies = np.fft.rfftfreq(spectral_frames, 1.0 / sample_rate)
    spectral_power[0] = 0.0
    total = max(float(np.sum(spectral_power)), _SPECTRAL_TOTAL_EPSILON)
    centroid = float(np.sum(frequencies * spectral_power) / total)
    bandwidth = float(
        np.sqrt(
            np.sum(np.square(frequencies - centroid) * spectral_power) / total
        )
    )

    zcr_end = min(
        audio.shape[0], onset_offset + round(sample_rate * _POST_ONSET_ZCR_SECONDS)
    )
    zcr_audio = centered[onset_offset:zcr_end]
    if zcr_audio.shape[0] >= 2:
        zcr = float(
            np.mean(np.signbit(zcr_audio[1:]) != np.signbit(zcr_audio[:-1]))
        )
    else:
        zcr = 0.0

    features = {
        "spectral_bandwidth_hz": bandwidth,
        "post_0_100_zero_crossing_rate": zcr,
    }
    if tuple(features) != TAPNESS_V2_RESEARCH_FEATURE_NAMES:
        raise TapnessV2ResearchError("Internal v2 feature ordering changed.")
    if any(not math.isfinite(value) or value < 0.0 for value in features.values()):
        raise TapnessV2ResearchError("Stage 2 v2 extraction produced invalid values.")
    return features


def transform_tapness_v2_research_features(
    features: Mapping[str, Any],
) -> np.ndarray[Any, Any]:
    """Validate and log-transform the ordered Experiment 19 feature vector."""

    values: list[float] = []
    for name in TAPNESS_V2_RESEARCH_FEATURE_NAMES:
        if name not in features:
            raise TapnessV2ResearchError(f"Missing Stage 2 v2 feature: {name}")
        value = _finite_float(features[name], f"Stage 2 v2 feature {name}")
        if value < 0.0:
            raise TapnessV2ResearchError(
                f"Stage 2 v2 feature {name} must be nonnegative."
            )
        values.append(math.log(max(value, TAPNESS_V2_RESEARCH_LOG_EPSILON)))
    return np.asarray(values, dtype=np.float64)


def fit_research_logistic_model(
    matrix: Any,
    labels: Any,
    *,
    l2_regularization: float = TAPNESS_V2_RESEARCH_L2,
) -> ResearchLogisticModel:
    """Fit one deterministic in-memory research model with local standardization."""

    values = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != len(TAPNESS_V2_RESEARCH_FEATURE_NAMES)
        or targets.shape != (values.shape[0],)
    ):
        raise TapnessV2ResearchError("Research fitting arrays have invalid shapes.")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(targets, (0.0, 1.0))):
        raise TapnessV2ResearchError("Research fitting arrays contain invalid values.")
    means = np.mean(values, axis=0)
    scales = np.std(values, axis=0)
    scales = np.where(scales <= 1.0e-12, 1.0, scales)
    coefficients, intercept = fit_l2_logistic(
        (values - means) / scales,
        targets,
        l2_regularization=l2_regularization,
    )
    return ResearchLogisticModel(
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
        l2_regularization=float(l2_regularization),
    )


def grouped_v2_research_folds(
    candidates: Sequence[Mapping[str, Any]],
    fold_count: int = TAPNESS_V2_RESEARCH_FOLD_COUNT,
) -> tuple[tuple[int, ...], ...]:
    """Create deterministic session/class-stratified whole-record folds."""

    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise TapnessV2ResearchError("Grouped research CV requires at least two folds.")
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    group_identity: dict[str, tuple[str, int]] = {}
    for row in candidates:
        session = str(row["session_label"])
        target = int(row["target"])
        group = str(row["source_group_id"])
        identity = (session, target)
        if group in group_identity and group_identity[group] != identity:
            raise TapnessV2ResearchError(
                "A source group crosses a session or target class."
            )
        group_identity[group] = identity
        if group not in strata[identity]:
            strata[identity].append(group)
    assignment: dict[str, int] = {}
    for key in sorted(strata):
        for position, group in enumerate(sorted(strata[key])):
            assignment[group] = position % fold_count
    folds = tuple(
        tuple(
            index
            for index, row in enumerate(candidates)
            if assignment[str(row["source_group_id"])] == fold
        )
        for fold in range(fold_count)
    )
    if any(not fold for fold in folds):
        raise TapnessV2ResearchError("Grouped research CV produced an empty fold.")
    return folds


def select_v2_research_threshold(
    scores: Any,
    labels: Any,
    *,
    minimum_positive_recall: float = TAPNESS_V2_RESEARCH_MINIMUM_POSITIVE_RECALL,
) -> tuple[float, dict[str, Any]]:
    """Reproduce Experiment 19's deterministic 95%-recall threshold rule."""

    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or targets.shape != values.shape or values.size == 0:
        raise TapnessV2ResearchError("Threshold-selection arrays have invalid shapes.")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(targets, (0, 1))):
        raise TapnessV2ResearchError("Threshold-selection arrays contain invalid values.")
    recall = _finite_float(minimum_positive_recall, "minimum positive recall")
    if not 0.0 <= recall <= 1.0:
        raise TapnessV2ResearchError("Minimum positive recall must be in [0, 1].")
    positive_count = int(np.count_nonzero(targets == 1))
    if positive_count == 0:
        raise TapnessV2ResearchError("Threshold selection requires positive examples.")
    required = math.ceil(recall * positive_count)
    unique = sorted(set(float(value) for value in values))
    candidates = [0.0, 1.0, *unique]
    candidates.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    evaluated: list[tuple[tuple[Any, ...], float, int, int]] = []
    for threshold in sorted(set(candidates)):
        accepted = values >= threshold
        true_positive = int(np.count_nonzero(accepted & (targets == 1)))
        false_positive = int(np.count_nonzero(accepted & (targets == 0)))
        key = (
            0 if true_positive >= required else 1,
            false_positive if true_positive >= required else -true_positive,
            -true_positive,
            -threshold,
        )
        evaluated.append((key, threshold, true_positive, false_positive))
    _, selected, true_positive, false_positive = min(evaluated, key=lambda item: item[0])
    return float(selected), {
        "method": (
            "grouped out-of-fold development scores; require at least 95% "
            "positive-candidate recall when feasible, minimize negative false "
            "accepts, maximize positive accepts, then choose the highest threshold"
        ),
        "minimum_positive_candidate_recall": recall,
        "required_positive_candidate_count": required,
        "selected_threshold": float(selected),
        "selected_positive_candidates": true_positive,
        "selected_negative_false_accepts": false_positive,
    }


def evaluate_tapness_v2_research_pair(
    session_a_path: Path,
    session_b_path: Path,
    *,
    detector_factory: Callable[[], Any] = StreamingTapDetector,
) -> dict[str, Any]:
    """Replay two sessions and reproduce the grouped Experiment 19 study.

    This function performs development fitting only in memory.  It never writes
    a frozen artifact, never invokes spatial inference, and fingerprints both
    datasets before and after replay.
    """

    paths = {"A": Path(session_a_path), "B": Path(session_b_path)}
    fingerprints_before = {
        label: robustness_dataset_fingerprint(path)["sha256"]
        for label, path in paths.items()
    }
    datasets = {
        label: load_robustness_dataset(path) for label, path in paths.items()
    }
    if fingerprints_before["A"] == fingerprints_before["B"]:
        raise TapnessV2ResearchError("Research evaluation requires two distinct datasets.")

    all_rows: list[dict[str, Any]] = []
    populations: dict[str, Any] = {}
    contexts: dict[str, Any] = {}
    for label in ("A", "B"):
        rows, population, context = _collect_research_candidates(
            label, datasets[label], detector_factory
        )
        all_rows.extend(rows)
        populations[label] = population
        contexts[label] = context

    feature_matrix = np.vstack(
        [
            transform_tapness_v2_research_features(row["features"])
            for row in all_rows
        ]
    )
    results: dict[str, Any] = {}
    for train_label, test_label in (("A", "B"), ("B", "A")):
        train_indexes = np.asarray(
            [i for i, row in enumerate(all_rows) if row["session_label"] == train_label],
            dtype=int,
        )
        test_indexes = np.asarray(
            [i for i, row in enumerate(all_rows) if row["session_label"] == test_label],
            dtype=int,
        )
        train_rows = [all_rows[index] for index in train_indexes]
        model, threshold, selection, folds = _fit_with_grouped_oof_threshold(
            train_rows, feature_matrix[train_indexes]
        )
        scores = model.score(feature_matrix[test_indexes])
        results[f"train_{train_label}_test_{test_label}"] = {
            "training_only_selected_threshold": threshold,
            "threshold_selection": selection,
            "training_oof_folds": folds,
            "development_model": model.to_report(),
            "metrics": _evaluation_metrics(
                [all_rows[index] for index in test_indexes],
                scores >= threshold,
                {test_label: contexts[test_label]},
            ),
        }

    labels = np.asarray([row["target"] for row in all_rows], dtype=np.int64)
    predictions = np.zeros(labels.size, dtype=bool)
    outer_reports: list[dict[str, Any]] = []
    all_indexes = set(range(labels.size))
    for outer_index, held_out_values in enumerate(grouped_v2_research_folds(all_rows)):
        held_out = np.asarray(held_out_values, dtype=int)
        training = np.asarray(sorted(all_indexes - set(held_out_values)), dtype=int)
        train_rows = [all_rows[index] for index in training]
        model, threshold, selection, _ = _fit_with_grouped_oof_threshold(
            train_rows, feature_matrix[training]
        )
        predictions[held_out] = model.score(feature_matrix[held_out]) >= threshold
        outer_reports.append(
            {
                "fold_index": outer_index,
                "held_out_group_ids": sorted(
                    {str(all_rows[index]["source_group_id"]) for index in held_out}
                ),
                "training_group_ids": sorted(
                    {str(all_rows[index]["source_group_id"]) for index in training}
                ),
                "selected_threshold_from_inner_grouped_oof": threshold,
                "threshold_selection": selection,
                "training_only_preprocessing": True,
            }
        )
    results["pooled_nested_grouped_oof"] = {
        "outer_fold_count": TAPNESS_V2_RESEARCH_FOLD_COUNT,
        "whole_source_record_groups": True,
        "training_fold_only_preprocessing_model_and_threshold": True,
        "folds": outer_reports,
        "metrics": _evaluation_metrics(all_rows, predictions, contexts),
    }

    fingerprints_after = {
        label: robustness_dataset_fingerprint(path)["sha256"]
        for label, path in paths.items()
    }
    if fingerprints_after != fingerprints_before:
        raise TapnessV2ResearchError("A robustness dataset changed during research replay.")

    report = {
        "report_schema_version": TAPNESS_V2_RESEARCH_REPORT_SCHEMA_VERSION,
        "report_type": "desksense_phase3b3_stage2_v2_research_reproduction",
        "evidence_role": "development_only_not_external_validation",
        "production_or_live_integration": False,
        "frozen_artifact_written": False,
        "stage1_behavior_changed": False,
        "stage3_used_as_input": False,
        "feature_schema": tapness_v2_research_feature_schema(),
        "development_methodology": {
            "model": "L2-regularized binary logistic regression",
            "implementation": "existing deterministic NumPy Newton/IRLS fitter",
            "l2_regularization": TAPNESS_V2_RESEARCH_L2,
            "transforms": (
                "ln(max(value, 1e-12)) for both ordered descriptors before "
                "standardization"
            ),
            "standardization": (
                "population mean and population standard deviation (ddof=0) "
                "from each training fold only; replace scale <=1e-12 with 1"
            ),
            "grouping": (
                "all candidates from one source recording stay together; folds "
                "are stratified deterministically by session and target class"
            ),
            "cross_session": (
                "train one complete session; select threshold from grouped OOF "
                "scores within that session; evaluate the other session"
            ),
            "pooled": (
                "five outer grouped folds; each outer training split performs "
                "its own grouped inner OOF threshold selection"
            ),
            "threshold_selection": (
                "95% positive-candidate-recall constraint, then minimum negative "
                "false accepts, maximum positive accepts, highest threshold"
            ),
            "score_is_calibrated_probability": False,
        },
        "datasets": {
            label: {
                "session_id": datasets[label].session["session_id"],
                "fingerprint_before": fingerprints_before[label],
                "fingerprint_after": fingerprints_after[label],
                "verified_unchanged": True,
            }
            for label in ("A", "B")
        },
        "candidate_populations": populations,
        "candidate_memberships": all_rows,
        "development_results": results,
        "limitations": [
            "Sessions A and B are both development evidence for this representation.",
            "Feature screening used A and B, so these estimates are optimistic.",
            "No result in this report is external validation.",
            "No final v2 artifact or production threshold is selected here.",
            "A revised frozen pipeline requires untouched Session C.",
        ],
    }
    if _is_experiment_19_pair(report):
        _verify_experiment_19_reproduction(report)
        report["experiment_19_reproduction_gate"] = {
            "applicable": True,
            "passed": True,
            "expected": {
                "A_population": "29 positive / 50 negative candidates",
                "B_population": "30 positive / 74 negative candidates",
                "A_to_B": "30/30 intended taps / 2 of 74 negative false accepts",
                "B_to_A": "29/30 intended taps / 1 of 50 negative false accepts",
                "pooled": "59/60 intended taps / 3 of 124 negative false accepts",
            },
        }
    else:
        report["experiment_19_reproduction_gate"] = {
            "applicable": False,
            "passed": None,
            "reason": "dataset identities differ from Experiment 19 Sessions A/B",
        }
    json.dumps(report, allow_nan=False)
    return report


def write_tapness_v2_research_report(report: Mapping[str, Any], path: Path) -> Path:
    """Exclusively write a waveform-free JSON development report."""

    if report.get("report_type") != "desksense_phase3b3_stage2_v2_research_reproduction":
        raise TapnessV2ResearchError("Unexpected Stage 2 v2 research report type.")
    destination = Path(path)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            output.write(payload)
    except FileExistsError as error:
        raise TapnessV2ResearchError(
            f"Refusing to overwrite an existing research report: {destination}"
        ) from error
    return destination.resolve()


def _collect_research_candidates(
    session_label: str,
    dataset: LoadedRobustnessDataset,
    detector_factory: Callable[[], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    associated_start_count = 0
    extraneous_start_count = 0
    extraneous_completed_count = 0
    associated_without_window = 0
    for record in dataset.records:
        detector = detector_factory()
        results = tuple(detector.process_chunk(record.capture))
        starts = tuple(detector.drain_candidate_start_diagnostics())
        if record.metadata["record_type"] == POSITIVE_RECORD_TYPE:
            association = record.metadata["guided_cue"]["intended_event_association"]
            interval_start = int(association["start_frame_index_inclusive"])
            interval_end = int(association["end_frame_index_exclusive"])
            target = 1
        elif record.metadata["record_type"] == NEGATIVE_RECORD_TYPE:
            interval_start = int(record.metadata["activity_start_frame_index"])
            interval_end = int(record.metadata["activity_end_frame_index_exclusive"])
            target = 0
        else:
            raise TapnessV2ResearchError("Unsupported robustness record type.")

        associated_starts = {
            int(item.onset_frame_index): str(item.route)
            for item in starts
            if interval_start <= int(item.onset_frame_index) < interval_end
        }
        associated_start_count += len(associated_starts)
        extraneous_start_count += len(starts) - len(associated_starts)
        for result in results:
            onset = int(result.onset_frame_index)
            if not interval_start <= onset < interval_end:
                extraneous_completed_count += 1
                continue
            if result.candidate_window is None:
                associated_without_window += 1
                continue
            if (
                result.center_frame_index is None
                or result.window_start_frame_index is None
                or result.window_end_frame_index_exclusive is None
            ):
                raise TapnessV2ResearchError(
                    "A completed candidate has incomplete frame metadata."
                )
            if onset not in associated_starts:
                raise TapnessV2ResearchError(
                    "A completed candidate lacks its Stage 1 start diagnostic."
                )
            features = extract_tapness_v2_research_features(
                result.candidate_window,
                sample_rate_hz=_SAMPLE_RATE_HZ,
                onset_frame_index=onset,
                center_frame_index=int(result.center_frame_index),
                window_start_frame_index=int(result.window_start_frame_index),
                window_end_frame_index_exclusive=int(
                    result.window_end_frame_index_exclusive
                ),
            )
            rows.append(
                {
                    "candidate_id": f"{record.metadata['record_id']}@{onset}",
                    "session_label": session_label,
                    "session_id": dataset.session["session_id"],
                    "source_record_id": record.metadata["record_id"],
                    "source_group_id": record.metadata["record_id"],
                    "target": target,
                    "target_label": "TAP" if target else "NON_TAP",
                    "intended_zone": record.metadata.get("intended_zone"),
                    "intended_strength": record.metadata.get("intended_strength"),
                    "negative_activity": record.metadata.get("activity"),
                    "stage1_route": associated_starts[onset],
                    "onset_frame_index": onset,
                    "center_frame_index": int(result.center_frame_index),
                    "window_start_frame_index": int(result.window_start_frame_index),
                    "window_end_frame_index_exclusive": int(
                        result.window_end_frame_index_exclusive
                    ),
                    "features": features,
                }
            )

    positive_candidate_records = {
        row["source_record_id"] for row in rows if row["target"] == 1
    }
    population = {
        "positive_intended_record_count": len(dataset.positive_records),
        "positive_candidate_count": sum(row["target"] == 1 for row in rows),
        "positive_records_with_candidate": len(positive_candidate_records),
        "negative_candidate_count": sum(row["target"] == 0 for row in rows),
        "positive_routes": dict(
            sorted(Counter(row["stage1_route"] for row in rows if row["target"] == 1).items())
        ),
        "negative_routes": dict(
            sorted(Counter(row["stage1_route"] for row in rows if row["target"] == 0).items())
        ),
        "associated_candidate_start_count": associated_start_count,
        "associated_completed_without_window": associated_without_window,
        "extraneous_candidate_start_count": extraneous_start_count,
        "extraneous_completed_result_count": extraneous_completed_count,
    }
    activity_seconds: Counter[str] = Counter()
    for record in dataset.negative_records:
        activity_seconds[str(record.metadata["activity"])] += float(
            record.metadata["activity_duration_seconds"]
        )
    context = {
        "session_id": dataset.session["session_id"],
        "positive_records": [
            {
                "record_id": record.metadata["record_id"],
                "zone": record.metadata["intended_zone"],
                "strength": record.metadata["intended_strength"],
            }
            for record in dataset.positive_records
        ],
        "negative_labeled_seconds_by_activity": dict(sorted(activity_seconds.items())),
        "negative_labeled_seconds_total": float(sum(activity_seconds.values())),
    }
    return rows, population, context


def _fit_with_grouped_oof_threshold(
    rows: Sequence[Mapping[str, Any]], matrix: np.ndarray[Any, Any]
) -> tuple[ResearchLogisticModel, float, dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray([row["target"] for row in rows], dtype=np.int64)
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    reports: list[dict[str, Any]] = []
    all_indexes = set(range(labels.size))
    for fold_index, held_out_values in enumerate(grouped_v2_research_folds(rows)):
        held_out = np.asarray(held_out_values, dtype=int)
        training = np.asarray(sorted(all_indexes - set(held_out_values)), dtype=int)
        model = fit_research_logistic_model(matrix[training], labels[training])
        scores[held_out] = model.score(matrix[held_out])
        reports.append(
            {
                "fold_index": fold_index,
                "held_out_group_ids": sorted(
                    {str(rows[index]["source_group_id"]) for index in held_out}
                ),
                "training_group_ids": sorted(
                    {str(rows[index]["source_group_id"]) for index in training}
                ),
                "training_only_standardization": {
                    "means": list(model.means),
                    "scales": list(model.scales),
                },
            }
        )
    if not np.all(np.isfinite(scores)):
        raise TapnessV2ResearchError("Grouped OOF evaluation left examples unscored.")
    threshold, selection = select_v2_research_threshold(scores, labels)
    model = fit_research_logistic_model(matrix, labels)
    return model, threshold, selection, reports


def _evaluation_metrics(
    rows: Sequence[Mapping[str, Any]],
    accepted: Any,
    contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    decisions = np.asarray(accepted, dtype=bool)
    if decisions.shape != (len(rows),):
        raise TapnessV2ResearchError("Research decision vector has invalid shape.")
    positive_indexes = [index for index, row in enumerate(rows) if row["target"] == 1]
    negative_indexes = [index for index, row in enumerate(rows) if row["target"] == 0]
    accepted_positive_records = {
        str(rows[index]["source_record_id"])
        for index in positive_indexes
        if decisions[index]
    }
    all_positive_records = [
        record
        for context in contexts.values()
        for record in context["positive_records"]
    ]
    candidate_positive_records = {
        str(rows[index]["source_record_id"]) for index in positive_indexes
    }

    def positive_breakdown(key: str) -> dict[str, Any]:
        values = sorted({str(record[key]) for record in all_positive_records})
        output: dict[str, Any] = {}
        for value in values:
            intended_ids = {
                str(record["record_id"])
                for record in all_positive_records
                if str(record[key]) == value
            }
            output[value] = {
                "intended_attempts": len(intended_ids),
                "stage1_candidate_attempts": len(intended_ids & candidate_positive_records),
                "stage2_research_accepted_attempts": len(
                    intended_ids & accepted_positive_records
                ),
            }
        return output

    cell_breakdown: dict[str, Any] = {}
    for record in all_positive_records:
        cell = f"{record['zone']}-{record['strength']}"
        cell_breakdown.setdefault(cell, {"record_ids": []})["record_ids"].append(
            str(record["record_id"])
        )
    for cell, item in cell_breakdown.items():
        intended_ids = set(item.pop("record_ids"))
        item.update(
            {
                "intended_attempts": len(intended_ids),
                "stage1_candidate_attempts": len(intended_ids & candidate_positive_records),
                "stage2_research_accepted_attempts": len(
                    intended_ids & accepted_positive_records
                ),
            }
        )

    false_accept_by_activity = Counter(
        str(rows[index]["negative_activity"])
        for index in negative_indexes
        if decisions[index]
    )
    candidate_by_activity = Counter(
        str(rows[index]["negative_activity"]) for index in negative_indexes
    )
    seconds_by_activity: Counter[str] = Counter()
    for context in contexts.values():
        seconds_by_activity.update(context["negative_labeled_seconds_by_activity"])
    per_activity = {}
    for activity in sorted(seconds_by_activity):
        seconds = float(seconds_by_activity[activity])
        per_activity[activity] = {
            "labeled_seconds": seconds,
            "stage1_candidates": candidate_by_activity[activity],
            "research_false_accepts": false_accept_by_activity[activity],
            "research_false_accepts_per_minute": (
                false_accept_by_activity[activity] / (seconds / 60.0)
            ),
        }
    negative_seconds = float(sum(seconds_by_activity.values()))
    false_accept_count = int(np.count_nonzero(decisions[negative_indexes]))
    return {
        "positive_intended_attempt_count": len(all_positive_records),
        "positive_candidate_count": len(positive_indexes),
        "positive_attempts_with_stage1_candidate": len(candidate_positive_records),
        "positive_attempts_accepted": len(accepted_positive_records),
        "positive_acceptance_over_all_intended": (
            len(accepted_positive_records) / len(all_positive_records)
        ),
        "by_zone": positive_breakdown("zone"),
        "by_strength": positive_breakdown("strength"),
        "by_zone_and_strength": dict(sorted(cell_breakdown.items())),
        "negative_candidate_count": len(negative_indexes),
        "negative_false_accept_count": false_accept_count,
        "negative_labeled_seconds": negative_seconds,
        "negative_false_accepts_per_minute": (
            false_accept_count / (negative_seconds / 60.0)
        ),
        "false_accepts_by_activity": dict(sorted(false_accept_by_activity.items())),
        "per_activity": per_activity,
    }


def _is_experiment_19_pair(report: Mapping[str, Any]) -> bool:
    datasets = report["datasets"]
    return (
        datasets["A"]["session_id"] == _EXPERIMENT_19_SESSION_A_ID
        and datasets["B"]["session_id"] == _EXPERIMENT_19_SESSION_B_ID
        and datasets["A"]["fingerprint_before"]
        == _EXPERIMENT_19_SESSION_A_FINGERPRINT
        and datasets["B"]["fingerprint_before"]
        == _EXPERIMENT_19_SESSION_B_FINGERPRINT
    )


def _verify_experiment_19_reproduction(report: Mapping[str, Any]) -> None:
    population = report["candidate_populations"]
    results = report["development_results"]
    actual = {
        "A_positive": population["A"]["positive_candidate_count"],
        "A_negative": population["A"]["negative_candidate_count"],
        "B_positive": population["B"]["positive_candidate_count"],
        "B_negative": population["B"]["negative_candidate_count"],
        "A_to_B_positive": results["train_A_test_B"]["metrics"][
            "positive_attempts_accepted"
        ],
        "A_to_B_negative": results["train_A_test_B"]["metrics"][
            "negative_false_accept_count"
        ],
        "B_to_A_positive": results["train_B_test_A"]["metrics"][
            "positive_attempts_accepted"
        ],
        "B_to_A_negative": results["train_B_test_A"]["metrics"][
            "negative_false_accept_count"
        ],
        "pooled_positive": results["pooled_nested_grouped_oof"]["metrics"][
            "positive_attempts_accepted"
        ],
        "pooled_negative": results["pooled_nested_grouped_oof"]["metrics"][
            "negative_false_accept_count"
        ],
    }
    expected = {
        "A_positive": 29,
        "A_negative": 50,
        "B_positive": 30,
        "B_negative": 74,
        "A_to_B_positive": 30,
        "A_to_B_negative": 2,
        "B_to_A_positive": 29,
        "B_to_A_negative": 1,
        "pooled_positive": 59,
        "pooled_negative": 3,
    }
    if actual != expected:
        raise TapnessV2ResearchError(
            "Experiment 19 reproduction gate failed; refusing to tune or hide "
            f"the discrepancy. expected={expected!r}, actual={actual!r}"
        )


def _validate_candidate(candidate_window: Any) -> np.ndarray[Any, Any]:
    try:
        audio = np.array(candidate_window, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Stage 2 v2 candidate must be numeric audio.") from error
    if audio.shape != (_CANDIDATE_FRAMES, _CHANNEL_COUNT):
        raise ValueError("Stage 2 v2 candidate must have shape (9600, 2).")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Stage 2 v2 candidate contains non-finite samples.")
    return audio


def _integer_frame(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name.capitalize()} frame index must be an integer.")
    return int(value)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TapnessV2ResearchError(f"{name.capitalize()} must be numeric.") from error
    if not math.isfinite(result):
        raise TapnessV2ResearchError(f"{name.capitalize()} must be finite.")
    return result


def _stable_sigmoid(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output
