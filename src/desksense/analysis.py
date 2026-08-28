"""Reproducible offline analysis of Phase 2A LEFT/RIGHT datasets."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile

import numpy as np

from desksense.features import (
    DESCRIPTIVE_FEATURE_NAMES,
    PRIMARY_FEATURE_NAME,
    extract_two_channel_features,
    feature_definitions,
)


ANALYSIS_SCHEMA_VERSION = 1
SUPPORTED_DATASET_SCHEMA_VERSION = 1
EXPECTED_DATASET_KIND = "guided_labeled_multichannel_tap_dataset"
EXPECTED_ZONES = ("LEFT", "RIGHT")
HOLDOUT_TRAINING_END = 15
HOLDOUT_TEST_START = 16
HOLDOUT_TEST_END = 20
INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED = "hand-location-confounded"
INTERACTION_CONTEXT_SAME_HAND = "same-hand"
INTERACTION_CONTEXT_CHOICES = (
    INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED,
    INTERACTION_CONTEXT_SAME_HAND,
)
_REQUIRED_NPZ_MEMBERS = ("capture", "tap_window", "metadata_json")
_FORBIDDEN_REPORT_KEYS = {"capture", "tap_window", "metadata_json", "raw_audio"}


class DatasetAnalysisError(RuntimeError):
    """An offline dataset analysis failure."""


class DatasetIntegrityError(DatasetAnalysisError):
    """A dataset failed structural or content validation."""


@dataclass(frozen=True)
class LoadedSample:
    """Validated sample metadata plus only the retained analysis window."""

    metadata: dict[str, Any]
    sample_path: Path
    tap_window: np.ndarray[Any, Any]


@dataclass(frozen=True)
class LoadedDataset:
    """A read-only, validated Phase 2A session."""

    session_path: Path
    session_metadata: dict[str, Any]
    samples: tuple[LoadedSample, ...]
    rejected_attempt_count: int
    integrity_validation: dict[str, Any]


def load_dataset_session(session_path: Path) -> LoadedDataset:
    """Load and strictly validate a Phase 2A session without modifying it."""

    directory = Path(session_path).resolve()
    if not directory.exists():
        raise DatasetIntegrityError(f"Dataset session does not exist: {directory}")
    if not directory.is_dir():
        raise DatasetIntegrityError(
            f"Dataset session path is not a directory: {directory}"
        )

    session_file = directory / "session.json"
    manifest_file = directory / "manifest.jsonl"
    samples_directory = directory / "samples"
    if not session_file.is_file():
        raise DatasetIntegrityError(f"Missing session.json: {session_file}")
    if not manifest_file.is_file():
        raise DatasetIntegrityError(f"Missing manifest.jsonl: {manifest_file}")
    if not samples_directory.is_dir():
        raise DatasetIntegrityError(
            f"Missing samples directory: {samples_directory}"
        )

    session = _load_json_file(session_file, "session.json")
    _validate_session_metadata(session)
    records = _load_manifest(manifest_file)
    normalized_paths = _validate_manifest_records(records, session)

    loaded_samples: list[LoadedSample] = []
    referenced_paths: set[Path] = set()
    for record, normalized_path in zip(records, normalized_paths, strict=True):
        sample_path = _resolve_sample_path(
            directory, samples_directory, normalized_path
        )
        referenced_paths.add(sample_path)
        loaded_samples.append(
            _load_sample_artifact(
                sample_path,
                record,
                session,
            )
        )

    discovered_paths = {
        path.resolve()
        for path in samples_directory.rglob("*.npz")
        if path.is_file()
    }
    orphan_paths = sorted(discovered_paths - referenced_paths, key=str)
    if orphan_paths:
        formatted = ", ".join(str(path) for path in orphan_paths)
        raise DatasetIntegrityError(
            "Unreferenced NPZ artifact(s) were found under samples/: " + formatted
        )

    rejected_attempt_count = _load_rejected_attempt_count(
        directory / "rejected_attempts.jsonl"
    )
    counts_by_zone = _counts_by_zone(records, session["zones"])
    requested_per_zone = int(session["requested_samples_per_zone"])
    complete_against_request = all(
        count == requested_per_zone for count in counts_by_zone.values()
    )
    integrity = {
        "status": "passed",
        "dataset_files_modified": False,
        "accepted_manifest_records_validated": len(records),
        "npz_artifacts_validated": len(loaded_samples),
        "rejected_attempt_records_validated": rejected_attempt_count,
        "orphan_npz_artifact_count": 0,
        "counts_by_zone": counts_by_zone,
        "complete_against_session_request": complete_against_request,
        "checks": [
            "session metadata schema and configuration",
            "strict manifest JSON and accepted-record identity",
            "unique sample IDs, zone numbering, paths, and collection order",
            "NPZ members, float32 shapes, finiteness, and metadata equality",
            "sample rate, channel count, duration, endpoint, and storage metadata",
            "tap window equality with its declared full-capture slice",
            "absence of unreferenced NPZ artifacts",
            "rejected-attempt metadata excluded from labeled samples",
        ],
    }
    return LoadedDataset(
        session_path=directory,
        session_metadata=session,
        samples=tuple(loaded_samples),
        rejected_attempt_count=rejected_attempt_count,
        integrity_validation=integrity,
    )


def extract_dataset_features(
    dataset: LoadedDataset,
) -> list[dict[str, Any]]:
    """Extract identity and fixed numerical features from validated windows."""

    records: list[dict[str, Any]] = []
    for sample in dataset.samples:
        metadata = sample.metadata
        try:
            features = extract_two_channel_features(sample.tap_window)
        except ValueError as error:
            raise DatasetAnalysisError(
                f"Feature extraction failed for {metadata['sample_id']}: {error}"
            ) from error
        records.append(
            {
                "sample_id": str(metadata["sample_id"]),
                "zone": str(metadata["zone"]),
                "accepted_sample_number": int(
                    metadata["accepted_sample_number"]
                ),
                "collection_order_index": _optional_int(
                    metadata.get("collection_order_index")
                ),
                "attempt_index": _optional_int(metadata.get("attempt_index")),
                "captured_at_utc": _optional_string(
                    metadata.get("captured_at_utc")
                ),
                "sample_path": str(metadata["saved_sample_path"]),
                "sample_rate_hz": float(metadata["sample_rate_hz"]),
                "channel_count": int(metadata["channel_count"]),
                "features": features,
                "unavailable_features": [
                    name for name, value in features.items() if value is None
                ],
            }
        )
    return sorted(
        records,
        key=lambda item: (
            item["collection_order_index"]
            if item["collection_order_index"] is not None
            else math.inf,
            item["accepted_sample_number"],
            item["zone"],
        ),
    )


def descriptive_feature_statistics(
    feature_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the fixed secondary features by observed class."""

    result: dict[str, Any] = {}
    for feature_name in DESCRIPTIVE_FEATURE_NAMES:
        by_zone: dict[str, Any] = {}
        for zone in EXPECTED_ZONES:
            class_records = [
                record for record in feature_records if record["zone"] == zone
            ]
            values = [
                float(record["features"][feature_name])
                for record in class_records
                if record["features"].get(feature_name) is not None
            ]
            by_zone[zone] = _summarize_values(values, len(class_records))
        result[feature_name] = {
            "role": "descriptive_only",
            "population_standard_deviation_ddof": 0,
            "by_zone": by_zone,
        }
    return result


def chronological_split(
    feature_records: Sequence[Mapping[str, Any]],
    *,
    training_end: int = HOLDOUT_TRAINING_END,
    test_start: int = HOLDOUT_TEST_START,
    test_end: int = HOLDOUT_TEST_END,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split by accepted sample number, never by file or manifest ordering."""

    bounds = (training_end, test_start, test_end)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in bounds):
        raise DatasetAnalysisError(
            "Chronological split bounds must be integers."
        )
    if not 1 <= training_end < test_start <= test_end:
        raise DatasetAnalysisError(
            "Chronological split bounds must satisfy "
            "1 <= training_end < test_start <= test_end."
        )
    _require_exact_evaluation_numbers(feature_records, test_end)
    training = [
        record
        for record in feature_records
        if 1 <= int(record["accepted_sample_number"]) <= training_end
    ]
    test = [
        record
        for record in feature_records
        if test_start <= int(record["accepted_sample_number"]) <= test_end
    ]
    expected_training = training_end * len(EXPECTED_ZONES)
    expected_test = (test_end - test_start + 1) * len(EXPECTED_ZONES)
    if len(training) != expected_training or len(test) != expected_test:
        raise DatasetAnalysisError(
            "The chronological holdout requires LEFT/RIGHT accepted samples "
            f"1-{training_end} for training and {test_start}-{test_end} for test."
        )
    return _sort_by_number_and_zone(training), _sort_by_number_and_zone(test)


def fit_peak_ratio_baseline(
    training_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit the predeclared midpoint threshold using training data only."""

    values_by_zone = _primary_values_by_zone(training_records)
    class_means = {
        zone: float(math.fsum(values_by_zone[zone]) / len(values_by_zone[zone]))
        for zone in EXPECTED_ZONES
    }
    left_mean = class_means["LEFT"]
    right_mean = class_means["RIGHT"]
    tolerance = 1e-12 * max(1.0, abs(left_mean), abs(right_mean))
    if abs(left_mean - right_mean) <= tolerance:
        raise DatasetAnalysisError(
            "The training class means are numerically indistinguishable; a "
            "deterministic threshold direction cannot be learned."
        )
    lower_zone, higher_zone = (
        ("LEFT", "RIGHT") if left_mean < right_mean else ("RIGHT", "LEFT")
    )
    threshold = float((left_mean + right_mean) / 2.0)
    return {
        "feature_name": PRIMARY_FEATURE_NAME,
        "feature_units": "dB",
        "training_count_by_class": {
            zone: len(values_by_zone[zone]) for zone in EXPECTED_ZONES
        },
        "training_class_means_db": class_means,
        "threshold_db": threshold,
        "lower_feature_zone": lower_zone,
        "higher_feature_zone": higher_zone,
        "direction": (
            f"{lower_zone} below threshold; {higher_zone} at or above threshold"
        ),
        "tie_rule": f"An exact-threshold value predicts {higher_zone}.",
        "threshold_rule": (
            "midpoint between LEFT and RIGHT training means; held-out values "
            "are not used"
        ),
        "training_sample_ids": [
            str(record["sample_id"])
            for record in _sort_by_number_and_zone(training_records)
        ],
    }


def predict_peak_ratio(
    record: Mapping[str, Any], fitted_baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply a frozen fitted threshold and return an auditable margin."""

    feature_value = record["features"].get(PRIMARY_FEATURE_NAME)
    if feature_value is None:
        raise DatasetAnalysisError(
            f"Primary feature is unavailable for sample {record['sample_id']}."
        )
    value = _finite_float(feature_value, "primary feature")
    threshold = _finite_float(fitted_baseline["threshold_db"], "threshold")
    lower_zone = str(fitted_baseline["lower_feature_zone"])
    higher_zone = str(fitted_baseline["higher_feature_zone"])
    predicted = lower_zone if value < threshold else higher_zone
    threshold_offset = float(value - threshold)
    signed_toward_right = (
        threshold_offset if higher_zone == "RIGHT" else -threshold_offset
    )
    actual = str(record["zone"])
    actual_class_margin = (
        signed_toward_right if actual == "RIGHT" else -signed_toward_right
    )
    return {
        "sample_id": str(record["sample_id"]),
        "accepted_sample_number": int(record["accepted_sample_number"]),
        "collection_order_index": _optional_int(
            record.get("collection_order_index")
        ),
        "actual_label": actual,
        "predicted_label": predicted,
        "feature_name": PRIMARY_FEATURE_NAME,
        "feature_value_db": value,
        "threshold_db": threshold,
        "threshold_offset_db": threshold_offset,
        "absolute_margin_db": float(abs(threshold_offset)),
        "signed_margin_toward_RIGHT_db": float(signed_toward_right),
        "actual_class_margin_db": float(actual_class_margin),
        "correct": predicted == actual,
    }


def confusion_matrix(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a named LEFT/RIGHT matrix with no implicit axis ordering."""

    matrix = {
        "actual_LEFT": {"predicted_LEFT": 0, "predicted_RIGHT": 0},
        "actual_RIGHT": {"predicted_LEFT": 0, "predicted_RIGHT": 0},
    }
    for prediction in predictions:
        actual = str(prediction["actual_label"])
        predicted = str(prediction["predicted_label"])
        if actual not in EXPECTED_ZONES or predicted not in EXPECTED_ZONES:
            raise DatasetAnalysisError(
                f"Unsupported confusion-matrix label: {actual!r} -> {predicted!r}"
            )
        matrix[f"actual_{actual}"][f"predicted_{predicted}"] += 1
    return matrix


def chronological_holdout_analysis(
    feature_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the fixed accepted-number 1-15 / 16-20 protocol."""

    training, test = chronological_split(feature_records)
    fitted = fit_peak_ratio_baseline(training)
    predictions = [predict_peak_ratio(record, fitted) for record in test]
    evaluation = _evaluation_summary(predictions)
    return {
        "label": "within-session chronological holdout",
        "protocol": {
            "split_key": "accepted_sample_number within each zone",
            "training_accepted_sample_numbers": list(
                range(1, HOLDOUT_TRAINING_END + 1)
            ),
            "test_accepted_sample_numbers": list(
                range(HOLDOUT_TEST_START, HOLDOUT_TEST_END + 1)
            ),
            "training_count_by_class": {
                zone: sum(1 for record in training if record["zone"] == zone)
                for zone in EXPECTED_ZONES
            },
            "test_count_by_class": {
                zone: sum(1 for record in test if record["zone"] == zone)
                for zone in EXPECTED_ZONES
            },
            "held_out_samples_used_to_fit_threshold": False,
            "split_defined_after_dataset_collection": True,
        },
        "fitted_baseline": fitted,
        **evaluation,
        "predictions": predictions,
    }


def leave_one_pair_out_cross_validation(
    feature_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run deterministic within-session leave-one-pair-out validation."""

    _require_exact_evaluation_numbers(feature_records, HOLDOUT_TEST_END)
    pair_numbers = sorted(
        {
            int(record["accepted_sample_number"])
            for record in feature_records
            if record["zone"] == "LEFT"
        }
    )
    if len(pair_numbers) < 2:
        raise DatasetAnalysisError(
            "Leave-one-pair-out cross-validation requires at least two pairs."
        )

    folds: list[dict[str, Any]] = []
    aggregate_predictions: list[dict[str, Any]] = []
    for pair_number in pair_numbers:
        held_out = [
            record
            for record in feature_records
            if int(record["accepted_sample_number"]) == pair_number
        ]
        training = [
            record
            for record in feature_records
            if int(record["accepted_sample_number"]) != pair_number
        ]
        if {str(record["zone"]) for record in held_out} != set(EXPECTED_ZONES):
            raise DatasetAnalysisError(
                f"Accepted sample number {pair_number} is not a complete pair."
            )
        fitted = fit_peak_ratio_baseline(training)
        predictions = [
            predict_peak_ratio(record, fitted)
            for record in _sort_by_number_and_zone(held_out)
        ]
        aggregate_predictions.extend(predictions)
        folds.append(
            {
                "fold": int(pair_number),
                "held_out_accepted_sample_number": int(pair_number),
                "held_out_sample_ids": [
                    str(record["sample_id"])
                    for record in _sort_by_number_and_zone(held_out)
                ],
                "fitted_baseline": fitted,
                "predictions": predictions,
                **_evaluation_summary(predictions),
            }
        )

    return {
        "label": "within-session leave-one-pair-out cross-validation",
        "protocol": {
            "pairing_key": "accepted_sample_number",
            "fold_count": len(folds),
            "held_out_samples_per_fold": 2,
            "description": (
                "Each fold excludes LEFT #N and RIGHT #N, fits only on all "
                "remaining pairs, and predicts the excluded pair."
            ),
            "independent_external_test": False,
        },
        "folds": folds,
        **_evaluation_summary(aggregate_predictions),
        "predictions": aggregate_predictions,
    }


def analyze_dataset(
    session_path: Path,
    *,
    interaction_context: str,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Validate, analyze, and evaluate a dataset entirely offline."""

    if interaction_context not in INTERACTION_CONTEXT_CHOICES:
        raise DatasetAnalysisError(
            "Interaction context must be one of: "
            + ", ".join(INTERACTION_CONTEXT_CHOICES)
        )
    dataset = load_dataset_session(Path(session_path))
    zones = tuple(str(zone) for zone in dataset.session_metadata["zones"])
    if zones != EXPECTED_ZONES:
        raise DatasetAnalysisError(
            "Phase 2B currently requires session zones exactly LEFT, RIGHT in "
            "that order."
        )
    feature_records = extract_dataset_features(dataset)
    holdout = chronological_holdout_analysis(feature_records)
    cross_validation = leave_one_pair_out_cross_validation(feature_records)
    descriptive = descriptive_feature_statistics(feature_records)
    generated_at = _utc_timestamp(now_fn)
    session = dataset.session_metadata
    counts = _counts_by_zone(
        [sample.metadata for sample in dataset.samples], session["zones"]
    )

    report = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "report_type": "offline_left_right_spatial_feasibility",
        "status": "ok",
        "generated_at_utc": generated_at,
        "dataset": {
            "session_id": str(session["session_id"]),
            "session_path": str(dataset.session_path),
            "dataset_schema_version": int(session["schema_version"]),
            "session_created_at_utc": str(session["created_at_utc"]),
            "endpoint": _json_copy(session["selected_endpoint"]),
            "capture_configuration": _json_copy(
                session["capture_configuration"]
            ),
        },
        "integrity_validation": dataset.integrity_validation,
        "dataset_counts": {
            "accepted_total": len(feature_records),
            "accepted_by_zone": counts,
            "rejected_attempt_records": dataset.rejected_attempt_count,
        },
        "zones": list(EXPECTED_ZONES),
        "analysis_parameters": {
            "source_window": (
                "the complete retained tap_window array for each accepted sample"
            ),
            "alignment_or_filtering_applied": False,
            "primary_feature": PRIMARY_FEATURE_NAME,
            "interaction_context": interaction_context,
            "interaction_context_source": (
                "explicit analyst-supplied parameter; not inferred from audio "
                "or Phase 2A metadata"
            ),
            "holdout_training_accepted_sample_numbers": list(
                range(1, HOLDOUT_TRAINING_END + 1)
            ),
            "holdout_test_accepted_sample_numbers": list(
                range(HOLDOUT_TEST_START, HOLDOUT_TEST_END + 1)
            ),
            "cross_validation": "leave one matched LEFT/RIGHT number pair out",
        },
        "feature_definitions": feature_definitions(),
        "per_sample_features": feature_records,
        "descriptive_statistics": descriptive,
        "primary_baseline": {
            "feature_name": PRIMARY_FEATURE_NAME,
            "selection_provenance": (
                "Predeclared from the earlier 2+2 pilot before analysis of the "
                "main 20+20 dataset."
            ),
            "model": "one-dimensional deterministic midpoint threshold",
            "fit_rule": (
                "Compute LEFT and RIGHT means on training data only; use their "
                "midpoint as the threshold and infer direction from their order."
            ),
            "margin_definition": {
                "threshold_offset_db": "feature value minus fitted threshold",
                "absolute_margin_db": "absolute distance from threshold in dB",
                "signed_margin_toward_RIGHT_db": (
                    "positive supports RIGHT and negative supports LEFT after "
                    "applying the learned direction"
                ),
                "calibrated_probability": False,
            },
        },
        "chronological_holdout": holdout,
        "cross_validation": cross_validation,
        "exploratory_spectral": {
            "status": "not_implemented",
            "reason": (
                "Spectral features were intentionally deferred to avoid "
                "post-hoc feature searching in the primary evaluation."
            ),
        },
        "lag_analysis": {
            "status": "not_implemented",
            "reason": (
                "Wider lag remains a possible descriptive follow-up and is not "
                "used by the primary classifier."
            ),
            "prior_evidence_caution": (
                "Earlier +/-1 ms results were boundary-limited; lag must not be "
                "treated as proven physical microphone time of arrival, and "
                "driver or DSP processing may contribute."
            ),
        },
        "evidence_boundary": _evidence_boundary(interaction_context),
        "report_privacy": {
            "analysis_is_local_and_offline": True,
            "microphone_accessed": False,
            "dataset_files_modified": False,
            "raw_waveforms_included": False,
            "network_upload_performed": False,
        },
    }
    _validate_report_payload(report)
    return report


def write_analysis_report(report: Mapping[str, Any], path: Path) -> Path:
    """Write an analysis report exclusively and never inside its dataset."""

    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("Analysis report paths must end in .json.")
    dataset_path = Path(str(report["dataset"]["session_path"])).resolve()
    resolved_destination = destination.resolve()
    if resolved_destination == dataset_path or resolved_destination.is_relative_to(
        dataset_path
    ):
        raise ValueError(
            "Analysis reports must be written outside the source dataset session."
        )
    _validate_report_payload(report)
    serialized = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        sort_keys=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(serialized)
        output.write("\n")
    return destination.resolve()


def format_analysis_summary(report: Mapping[str, Any]) -> str:
    """Render a readable summary without overstating the evidence."""

    dataset = report["dataset"]
    counts = report["dataset_counts"]
    holdout = report["chronological_holdout"]
    holdout_fit = holdout["fitted_baseline"]
    cross_validation = report["cross_validation"]
    matrix = holdout["confusion_matrix"]
    evidence = report["evidence_boundary"]
    confound = evidence["hand_location_confound"]
    lines = [
        "DeskSense Phase 2B offline LEFT/RIGHT feasibility analysis",
        f"Session: {dataset['session_id']}",
        f"Session path: {dataset['session_path']}",
        (
            "Integrity validation: passed "
            f"({counts['accepted_total']} accepted sample(s); "
            f"LEFT={counts['accepted_by_zone']['LEFT']}, "
            f"RIGHT={counts['accepted_by_zone']['RIGHT']}; "
            f"rejected-attempt records={counts['rejected_attempt_records']})"
        ),
        "",
        "Descriptive feature summaries (all accepted samples):",
    ]
    for feature_name in DESCRIPTIVE_FEATURE_NAMES:
        statistics = report["descriptive_statistics"][feature_name]["by_zone"]
        lines.append(f"  {feature_name}:")
        for zone in EXPECTED_ZONES:
            summary = statistics[zone]
            lines.append(
                f"    {zone}: n={summary['available_count']}, "
                f"mean={_format_optional(summary['mean'])}, "
                f"std={_format_optional(summary['standard_deviation'])}, "
                f"min={_format_optional(summary['minimum'])}, "
                f"median={_format_optional(summary['median'])}, "
                f"max={_format_optional(summary['maximum'])}"
            )

    lines.extend(
        [
            "",
            "Primary baseline: Ch2 / Ch1 peak-ratio in dB",
            (
                "  Training means: "
                f"LEFT={holdout_fit['training_class_means_db']['LEFT']:.6f} dB, "
                f"RIGHT={holdout_fit['training_class_means_db']['RIGHT']:.6f} dB"
            ),
            f"  Learned threshold: {holdout_fit['threshold_db']:.6f} dB",
            f"  Learned direction: {holdout_fit['direction']}",
            "",
            "within-session chronological holdout:",
            (
                f"  Correct: {holdout['correct_count']}/{holdout['total_count']} "
                f"({holdout['accuracy'] * 100.0:.2f}%)"
            ),
            (
                "  Confusion: actual LEFT -> "
                f"LEFT {matrix['actual_LEFT']['predicted_LEFT']}, "
                f"RIGHT {matrix['actual_LEFT']['predicted_RIGHT']}; "
                "actual RIGHT -> "
                f"LEFT {matrix['actual_RIGHT']['predicted_LEFT']}, "
                f"RIGHT {matrix['actual_RIGHT']['predicted_RIGHT']}"
            ),
            "  Held-out predictions:",
        ]
    )
    for prediction in holdout["predictions"]:
        lines.append(
            f"    {prediction['sample_id']}: actual={prediction['actual_label']}, "
            f"predicted={prediction['predicted_label']}, "
            f"feature={prediction['feature_value_db']:.6f} dB, "
            f"threshold offset={prediction['threshold_offset_db']:+.6f} dB, "
            f"absolute margin={prediction['absolute_margin_db']:.6f} dB"
        )
    lines.extend(
        [
            "",
            "within-session leave-one-pair-out cross-validation:",
            (
                f"  Folds: {cross_validation['protocol']['fold_count']}; "
                f"correct: {cross_validation['correct_count']}/"
                f"{cross_validation['total_count']} "
                f"({cross_validation['accuracy'] * 100.0:.2f}%)"
            ),
            "",
            "Evidence boundary:",
            f"  LEFT = {confound['LEFT']}.",
            f"  RIGHT = {confound['RIGHT']}.",
            f"  {confound['consequence']}",
            (
                "  All samples share one user, laptop, desk/setup, and session. "
                "The holdout was defined after collection."
            ),
            (
                "  Held-out examples did not fit the threshold. "
                + evidence["future_validation"]
            ),
            "Raw waveform arrays are not included in the analysis report.",
        ]
    )
    return "\n".join(lines)


def _validate_session_metadata(session: dict[str, Any]) -> None:
    schema_version = _required_positive_int(session, "schema_version", "session")
    if schema_version != SUPPORTED_DATASET_SCHEMA_VERSION:
        raise DatasetIntegrityError(
            f"Unsupported dataset schema version: {schema_version}"
        )
    if _required_string(session, "dataset_kind", "session") != EXPECTED_DATASET_KIND:
        raise DatasetIntegrityError(
            "session.json does not describe a guided labeled tap dataset."
        )
    if _required_string(session, "project_phase", "session") != "2A":
        raise DatasetIntegrityError("session.json project_phase must be '2A'.")
    _required_string(session, "session_id", "session")
    _require_utc_timestamp(
        _required_string(session, "created_at_utc", "session"),
        "session created_at_utc",
    )
    zones = session.get("zones")
    if not isinstance(zones, list) or not zones:
        raise DatasetIntegrityError("session zones must be a non-empty JSON list.")
    if any(not isinstance(zone, str) or not zone for zone in zones):
        raise DatasetIntegrityError("session zone labels must be non-empty strings.")
    if len({zone.casefold() for zone in zones}) != len(zones):
        raise DatasetIntegrityError("session zone labels must be unique.")

    requested_per_zone = _required_positive_int(
        session, "requested_samples_per_zone", "session"
    )
    requested_total = _required_positive_int(
        session, "requested_total_samples", "session"
    )
    if requested_total != requested_per_zone * len(zones):
        raise DatasetIntegrityError(
            "session requested_total_samples disagrees with zones and "
            "requested_samples_per_zone."
        )
    if _required_string(session, "collection_order", "session") != (
        "round_robin_alternating_zones"
    ):
        raise DatasetIntegrityError(
            "Phase 2B requires round_robin_alternating_zones collection order."
        )

    configuration = _required_mapping(
        session, "capture_configuration", "session"
    )
    _required_positive_float(configuration, "sample_rate_hz", "capture configuration")
    _required_positive_int(configuration, "channel_count", "capture configuration")
    if _required_string(configuration, "dtype", "capture configuration") != "float32":
        raise DatasetIntegrityError("Session capture dtype must be float32.")
    _validate_endpoint(
        _required_mapping(session, "selected_endpoint", "session"),
        "session selected endpoint",
    )


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise DatasetIntegrityError(f"Could not read manifest.jsonl: {error}") from error
    if not lines:
        raise DatasetIntegrityError("manifest.jsonl contains no accepted samples.")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DatasetIntegrityError(
                f"manifest.jsonl line {line_number} is blank."
            )
        records.append(
            _load_json_text(line, f"manifest.jsonl line {line_number}")
        )
    return records


def _validate_manifest_records(
    records: Sequence[dict[str, Any]], session: Mapping[str, Any]
) -> list[PurePosixPath]:
    zones = [str(zone) for zone in session["zones"]]
    requested_per_zone = int(session["requested_samples_per_zone"])
    session_rate = float(session["capture_configuration"]["sample_rate_hz"])
    session_channels = int(session["capture_configuration"]["channel_count"])
    session_endpoint = session["selected_endpoint"]
    seen_ids: set[str] = set()
    seen_numbers: set[tuple[str, int]] = set()
    seen_paths: set[str] = set()
    seen_attempts: set[int] = set()
    attempt_index_presence: list[bool] = []
    normalized_paths: list[PurePosixPath] = []

    for expected_order, record in enumerate(records, start=1):
        context = f"manifest accepted record {expected_order}"
        if record.get("accepted") is not True:
            raise DatasetIntegrityError(f"{context} is not marked accepted=true.")
        sample_id = _required_string(record, "sample_id", context)
        normalized_id = sample_id.casefold()
        if normalized_id in seen_ids:
            raise DatasetIntegrityError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(normalized_id)

        zone = _required_string(record, "zone", context)
        if zone not in zones:
            raise DatasetIntegrityError(
                f"{context} has invalid zone {zone!r}; expected one of {zones}."
            )
        accepted_number = _required_positive_int(
            record, "accepted_sample_number", context
        )
        if accepted_number > requested_per_zone:
            raise DatasetIntegrityError(
                f"{context} accepted number exceeds the session request."
            )
        identity = (zone, accepted_number)
        if identity in seen_numbers:
            raise DatasetIntegrityError(
                f"Duplicate accepted sample number for {zone}: {accepted_number}"
            )
        seen_numbers.add(identity)

        order_index = _required_positive_int(
            record, "collection_order_index", context
        )
        if order_index != expected_order:
            raise DatasetIntegrityError(
                f"{context} collection_order_index must be {expected_order}, "
                f"not {order_index}."
            )
        expected_zone = zones[(expected_order - 1) % len(zones)]
        expected_number = (expected_order - 1) // len(zones) + 1
        if zone != expected_zone or accepted_number != expected_number:
            raise DatasetIntegrityError(
                f"{context} does not match the session round-robin order; "
                f"expected {expected_zone} #{expected_number}."
            )

        attempt_index_presence.append("attempt_index" in record)
        if "attempt_index" in record:
            attempt_index = _required_positive_int(record, "attempt_index", context)
            if attempt_index in seen_attempts:
                raise DatasetIntegrityError(
                    f"Duplicate accepted attempt_index: {attempt_index}"
                )
            seen_attempts.add(attempt_index)

        saved_path = _validate_saved_sample_path(
            _required_string(record, "saved_sample_path", context), context
        )
        path_key = saved_path.as_posix().casefold()
        if path_key in seen_paths:
            raise DatasetIntegrityError(
                f"Duplicate accepted sample path: {saved_path.as_posix()}"
            )
        seen_paths.add(path_key)
        normalized_paths.append(saved_path)

        sample_rate = _required_positive_float(record, "sample_rate_hz", context)
        if not _same_float(sample_rate, session_rate):
            raise DatasetIntegrityError(
                f"{context} sample rate disagrees with session.json."
            )
        if _required_positive_int(record, "channel_count", context) != session_channels:
            raise DatasetIntegrityError(
                f"{context} channel count disagrees with session.json."
            )
        if _canonical_json(
            _required_mapping(record, "device", context)
        ) != _canonical_json(session_endpoint):
            raise DatasetIntegrityError(
                f"{context} endpoint metadata disagrees with session.json."
            )
        _require_utc_timestamp(
            _required_string(record, "captured_at_utc", context),
            f"{context} captured_at_utc",
        )
        quality = _required_mapping(record, "capture_quality", context)
        if quality.get("accepted") is not True or quality.get("status") != "accepted":
            raise DatasetIntegrityError(
                f"{context} capture_quality is not marked accepted."
            )

    for zone in zones:
        numbers = sorted(number for item_zone, number in seen_numbers if item_zone == zone)
        if numbers != list(range(1, len(numbers) + 1)):
            raise DatasetIntegrityError(
                f"Accepted numbering for {zone} is not contiguous from 1."
            )
    if any(attempt_index_presence) and not all(attempt_index_presence):
        raise DatasetIntegrityError(
            "attempt_index must be present on every accepted record or none."
        )
    if seen_attempts:
        ordered_attempts = [int(record["attempt_index"]) for record in records]
        if ordered_attempts != sorted(ordered_attempts):
            raise DatasetIntegrityError(
                "Accepted attempt indices must increase with collection order."
            )
    return normalized_paths


def _load_sample_artifact(
    path: Path,
    record: dict[str, Any],
    session: Mapping[str, Any],
) -> LoadedSample:
    context = f"sample {record['sample_id']} ({path})"
    try:
        with np.load(path, allow_pickle=False) as archive:
            for member in _REQUIRED_NPZ_MEMBERS:
                if archive.files.count(member) != 1:
                    raise DatasetIntegrityError(
                        f"{context} must contain exactly one {member!r} array."
                    )
            capture = np.asarray(archive["capture"])
            tap_window = np.asarray(archive["tap_window"])
            metadata_array = np.asarray(archive["metadata_json"])
    except DatasetIntegrityError:
        raise
    except (OSError, TypeError, ValueError, KeyError, EOFError, BadZipFile) as error:
        raise DatasetIntegrityError(f"Could not read {context}: {error}") from error

    if metadata_array.shape != () or metadata_array.dtype.kind not in {"U", "S"}:
        raise DatasetIntegrityError(
            f"{context} metadata_json must be one scalar string."
        )
    metadata_value = metadata_array.item()
    if isinstance(metadata_value, bytes):
        try:
            metadata_text = metadata_value.decode("utf-8")
        except UnicodeError as error:
            raise DatasetIntegrityError(
                f"{context} metadata_json is not UTF-8."
            ) from error
    else:
        metadata_text = str(metadata_value)
    embedded = _load_json_text(metadata_text, f"{context} metadata_json")
    if _canonical_json(embedded) != _canonical_json(record):
        raise DatasetIntegrityError(
            f"{context} embedded metadata does not agree with manifest.jsonl."
        )

    _validate_audio_array(capture, "capture", context)
    _validate_audio_array(tap_window, "tap_window", context)
    session_channels = int(session["capture_configuration"]["channel_count"])
    if capture.shape[1] != session_channels:
        raise DatasetIntegrityError(
            f"{context} capture channel count disagrees with session.json."
        )
    if tap_window.shape[1] != session_channels:
        raise DatasetIntegrityError(
            f"{context} tap-window channel count disagrees with session.json."
        )
    if capture.shape[1] != tap_window.shape[1]:
        raise DatasetIntegrityError(
            f"{context} capture and tap-window channel counts differ."
        )
    capture_frames = _required_positive_int(record, "capture_frames", context)
    if capture.shape[0] != capture_frames:
        raise DatasetIntegrityError(
            f"{context} capture frame count disagrees with manifest metadata."
        )
    if _required_string(record, "capture_dtype", context) != "float32":
        raise DatasetIntegrityError(f"{context} capture_dtype must be float32.")
    sample_rate = float(record["sample_rate_hz"])
    expected_duration = capture.shape[0] / sample_rate
    recorded_duration = _required_positive_float(
        record, "capture_duration_seconds", context
    )
    if not math.isclose(
        recorded_duration,
        expected_duration,
        rel_tol=1e-12,
        abs_tol=0.5 / sample_rate,
    ):
        raise DatasetIntegrityError(
            f"{context} capture duration disagrees with frames/sample rate."
        )

    storage = _required_mapping(record, "storage", context)
    if storage.get("capture_array") != "capture":
        raise DatasetIntegrityError(f"{context} storage capture array name is invalid.")
    if storage.get("tap_window_array") != "tap_window":
        raise DatasetIntegrityError(f"{context} storage tap-window array name is invalid.")
    if storage.get("metadata_array") != "metadata_json":
        raise DatasetIntegrityError(f"{context} storage metadata array name is invalid.")
    if storage.get("dtype") != "float32":
        raise DatasetIntegrityError(f"{context} storage dtype must be float32.")
    if storage.get("capture_shape") != [int(value) for value in capture.shape]:
        raise DatasetIntegrityError(f"{context} storage capture_shape is incorrect.")
    if storage.get("tap_window_shape") != [int(value) for value in tap_window.shape]:
        raise DatasetIntegrityError(f"{context} storage tap_window_shape is incorrect.")

    quality = _required_mapping(record, "capture_quality", context)
    transient = _required_mapping(quality, "transient", f"{context} capture_quality")
    start = _required_nonnegative_int(
        transient, "exact_tap_window_start_sample", f"{context} transient"
    )
    end = _required_positive_int(
        transient,
        "exact_tap_window_end_sample_exclusive",
        f"{context} transient",
    )
    if transient.get("exact_tap_window_available") is not True:
        raise DatasetIntegrityError(
            f"{context} accepted tap window is not marked available."
        )
    if start >= end or end > capture.shape[0]:
        raise DatasetIntegrityError(f"{context} tap-window slice indices are invalid.")
    if end - start != tap_window.shape[0]:
        raise DatasetIntegrityError(
            f"{context} tap-window length disagrees with transient metadata."
        )
    if not np.array_equal(tap_window, capture[start:end]):
        raise DatasetIntegrityError(
            f"{context} tap_window does not equal its declared capture slice."
        )

    retained_window = np.ascontiguousarray(tap_window, dtype=np.float32)
    retained_window.setflags(write=False)
    return LoadedSample(
        metadata=dict(record),
        sample_path=path,
        tap_window=retained_window,
    )


def _load_rejected_attempt_count(path: Path) -> int:
    if not path.exists():
        return 0
    if not path.is_file():
        raise DatasetIntegrityError(f"Rejected-attempt path is not a file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise DatasetIntegrityError(
            f"Could not read rejected_attempts.jsonl: {error}"
        ) from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DatasetIntegrityError(
                f"rejected_attempts.jsonl line {line_number} is blank."
            )
        record = _load_json_text(
            line, f"rejected_attempts.jsonl line {line_number}"
        )
        if record.get("accepted") is not False:
            raise DatasetIntegrityError(
                f"Rejected-attempt line {line_number} is not marked accepted=false."
            )
        if record.get("waveform_saved") is not False:
            raise DatasetIntegrityError(
                f"Rejected-attempt line {line_number} is not marked waveform_saved=false."
            )
        if _FORBIDDEN_REPORT_KEYS.intersection(record):
            raise DatasetIntegrityError(
                f"Rejected-attempt line {line_number} unexpectedly contains waveform data."
            )
    return len(lines)


def _validate_saved_sample_path(value: str, context: str) -> PurePosixPath:
    if "\\" in value:
        raise DatasetIntegrityError(
            f"{context} saved_sample_path must use portable forward slashes."
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != "samples"
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[1]
        or path.suffix.casefold() != ".npz"
    ):
        raise DatasetIntegrityError(
            f"{context} saved_sample_path must be samples/<name>.npz."
        )
    return path


def _resolve_sample_path(
    session_directory: Path,
    samples_directory: Path,
    relative_path: PurePosixPath,
) -> Path:
    candidate = (session_directory / Path(*relative_path.parts)).resolve()
    resolved_samples = samples_directory.resolve()
    if candidate.parent != resolved_samples or not candidate.is_relative_to(
        resolved_samples
    ):
        raise DatasetIntegrityError(
            f"Sample path escapes the session samples directory: {relative_path}"
        )
    if not candidate.is_file():
        raise DatasetIntegrityError(f"Missing accepted sample NPZ: {candidate}")
    return candidate


def _validate_audio_array(array: np.ndarray[Any, Any], label: str, context: str) -> None:
    if array.dtype != np.float32:
        raise DatasetIntegrityError(f"{context} {label} dtype must be float32.")
    if array.ndim != 2:
        raise DatasetIntegrityError(f"{context} {label} must be two-dimensional.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DatasetIntegrityError(f"{context} {label} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise DatasetIntegrityError(f"{context} {label} contains non-finite values.")


def _primary_values_by_zone(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {zone: [] for zone in EXPECTED_ZONES}
    for record in records:
        zone = str(record["zone"])
        if zone not in values:
            raise DatasetAnalysisError(f"Unsupported baseline zone: {zone!r}")
        feature_value = record["features"].get(PRIMARY_FEATURE_NAME)
        if feature_value is None:
            raise DatasetAnalysisError(
                f"Primary feature is unavailable for sample {record['sample_id']}."
            )
        values[zone].append(_finite_float(feature_value, "primary feature"))
    for zone in EXPECTED_ZONES:
        if not values[zone]:
            raise DatasetAnalysisError(f"No {zone} samples are available for fitting.")
    return values


def _evaluation_summary(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    if total == 0:
        raise DatasetAnalysisError("Evaluation requires at least one prediction.")
    correct = sum(bool(prediction["correct"]) for prediction in predictions)
    per_class = {
        zone: {
            "correct_count": sum(
                bool(prediction["correct"])
                for prediction in predictions
                if prediction["actual_label"] == zone
            ),
            "total_count": sum(
                1
                for prediction in predictions
                if prediction["actual_label"] == zone
            ),
        }
        for zone in EXPECTED_ZONES
    }
    return {
        "confusion_matrix": confusion_matrix(predictions),
        "correct_count": int(correct),
        "total_count": int(total),
        "accuracy": float(correct / total),
        "per_class": per_class,
    }


def _require_exact_evaluation_numbers(
    records: Sequence[Mapping[str, Any]], expected_end: int
) -> None:
    for zone in EXPECTED_ZONES:
        numbers = sorted(
            int(record["accepted_sample_number"])
            for record in records
            if record["zone"] == zone
        )
        expected = list(range(1, expected_end + 1))
        if numbers != expected:
            raise DatasetAnalysisError(
                f"Phase 2B evaluation requires exactly {zone} accepted sample "
                f"numbers 1-{expected_end}; found {numbers}."
            )


def _sort_by_number_and_zone(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    zone_order = {zone: index for index, zone in enumerate(EXPECTED_ZONES)}
    return sorted(
        records,
        key=lambda record: (
            int(record["accepted_sample_number"]),
            zone_order[str(record["zone"])],
        ),
    )


def _summarize_values(values: Sequence[float], total_count: int) -> dict[str, Any]:
    if not values:
        return {
            "total_count": int(total_count),
            "available_count": 0,
            "unavailable_count": int(total_count),
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "median": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise DatasetAnalysisError("Descriptive statistics received non-finite data.")
    return {
        "total_count": int(total_count),
        "available_count": int(array.size),
        "unavailable_count": int(total_count - array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "median": float(np.median(array)),
    }


def _counts_by_zone(
    records: Sequence[Mapping[str, Any]], zones: Sequence[str]
) -> dict[str, int]:
    return {
        str(zone): sum(1 for record in records if record["zone"] == zone)
        for zone in zones
    }


def _evidence_boundary(interaction_context: str) -> dict[str, Any]:
    if interaction_context == INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED:
        hand_location_confound = {
            "present": True,
            "LEFT": "left hand + left location",
            "RIGHT": "right hand + right location",
            "consequence": "This dataset cannot isolate spatial location alone.",
            "provenance": (
                "Explicit analyst-supplied interaction context; tapping hand "
                "was not inferred from waveform or Phase 2A sample metadata."
            ),
        }
        scope_label = "within-session LEFT/RIGHT interaction-condition separation"
        future_validation = (
            "Use an untouched external session in which the same hand/finger "
            "taps both LEFT and RIGHT zones."
        )
    elif interaction_context == INTERACTION_CONTEXT_SAME_HAND:
        hand_location_confound = {
            "present": False,
            "LEFT": "same hand/finger + left location",
            "RIGHT": "same hand/finger + right location",
            "consequence": (
                "Tapping hand is controlled, but other within-session setup "
                "limitations still apply."
            ),
            "provenance": (
                "Explicit analyst-supplied interaction context; tapping hand "
                "was not inferred from waveform or Phase 2A sample metadata."
            ),
        }
        scope_label = "within-session same-hand LEFT/RIGHT separation"
        future_validation = (
            "Use additional untouched sessions before making external or "
            "general localization claims."
        )
    else:
        raise DatasetAnalysisError(f"Unsupported interaction context: {interaction_context}")
    return {
        "interaction_context": interaction_context,
        "interaction_context_required": True,
        "interaction_context_inferred_from_dataset": False,
        "scope_label": scope_label,
        "hand_location_confound": hand_location_confound,
        "single_session": True,
        "same_laptop": True,
        "same_desk_setup": True,
        "same_user": True,
        "holdout_split_defined_after_collection": True,
        "held_out_samples_used_to_fit_threshold": False,
        "not_claimed": [
            "pure spatial localization accuracy",
            "hand-independent localization accuracy",
            "external validation accuracy",
            "cross-device or general DeskSense accuracy",
        ],
        "future_validation": future_validation,
    }


def _validate_report_payload(value: Any, *, path: str = "report") -> None:
    if isinstance(value, np.ndarray) or isinstance(value, np.generic):
        raise ValueError(f"{path} contains a NumPy value that is not JSON safe.")
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_REPORT_KEYS.intersection(str(key) for key in value)
        if forbidden:
            raise ValueError(
                f"{path} contains forbidden waveform key(s): {sorted(forbidden)}"
            )
        for key, child in value.items():
            _validate_report_payload(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_report_payload(child, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite float.")


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DatasetIntegrityError(f"Could not read {label}: {error}") from error
    return _load_json_text(content, label)


def _load_json_text(content: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda token: _reject_json_constant(token),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise DatasetIntegrityError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetIntegrityError(f"{label} must contain a JSON object.")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"non-standard numeric constant {token!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _required_mapping(
    parent: Mapping[str, Any], key: str, context: str
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise DatasetIntegrityError(f"{context} field {key!r} must be an object.")
    return value


def _required_string(parent: Mapping[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetIntegrityError(
            f"{context} field {key!r} must be a non-empty string."
        )
    return value


def _required_positive_int(
    parent: Mapping[str, Any], key: str, context: str
) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetIntegrityError(
            f"{context} field {key!r} must be a positive integer."
        )
    return int(value)


def _required_nonnegative_int(
    parent: Mapping[str, Any], key: str, context: str
) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetIntegrityError(
            f"{context} field {key!r} must be a non-negative integer."
        )
    return int(value)


def _required_positive_float(
    parent: Mapping[str, Any], key: str, context: str
) -> float:
    value = parent.get(key)
    if isinstance(value, bool):
        raise DatasetIntegrityError(
            f"{context} field {key!r} must be positive and finite."
        )
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DatasetIntegrityError(
            f"{context} field {key!r} must be positive and finite."
        ) from error
    if not math.isfinite(converted) or converted <= 0.0:
        raise DatasetIntegrityError(
            f"{context} field {key!r} must be positive and finite."
        )
    return converted


def _validate_endpoint(endpoint: Mapping[str, Any], context: str) -> None:
    index = endpoint.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise DatasetIntegrityError(f"{context} index must be non-negative.")
    _required_string(endpoint, "name", context)
    host_api = _required_mapping(endpoint, "host_api", context)
    _required_string(host_api, "name", f"{context} host_api")
    host_index = host_api.get("index")
    if host_index is not None and (
        isinstance(host_index, bool)
        or not isinstance(host_index, int)
        or host_index < 0
    ):
        raise DatasetIntegrityError(
            f"{context} host_api index must be non-negative or null."
        )


def _require_utc_timestamp(value: str, context: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise DatasetIntegrityError(f"{context} is not a valid ISO timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DatasetIntegrityError(f"{context} must include a UTC offset.")
    if parsed.utcoffset().total_seconds() != 0:
        raise DatasetIntegrityError(f"{context} must be expressed in UTC.")


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def _finite_float(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DatasetAnalysisError(f"{label.capitalize()} must be finite.") from error
    if not math.isfinite(converted):
        raise DatasetAnalysisError(f"{label.capitalize()} must be finite.")
    return converted


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetAnalysisError("Optional integer metadata is invalid.")
    return int(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetAnalysisError("Optional string metadata is invalid.")
    return value


def _utc_timestamp(now_fn: Callable[[], datetime] | None) -> str:
    timestamp = datetime.now(timezone.utc) if now_fn is None else now_fn()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise DatasetAnalysisError("Analysis timestamps must be timezone-aware.")
    return timestamp.astimezone(timezone.utc).isoformat()


def _format_optional(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6f}"
