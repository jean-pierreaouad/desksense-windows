"""Frozen LEFT/RIGHT baseline artifacts and cross-session evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from desksense.analysis import (
    EXPECTED_ZONES,
    INTERACTION_CONTEXT_CHOICES,
    INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED,
    INTERACTION_CONTEXT_SAME_HAND,
    SOURCE_DATASET_FINGERPRINT_FRAMING,
    SOURCE_DATASET_FINGERPRINT_ORDER,
    SOURCE_DATASET_FINGERPRINT_SCHEMA_VERSION,
    SOURCE_DATASET_FINGERPRINT_SCHEME,
    DatasetAnalysisError,
    confusion_matrix,
    descriptive_feature_statistics,
    extract_dataset_features,
    fit_peak_ratio_baseline,
    load_dataset_session,
    predict_peak_ratio,
    source_dataset_fingerprint,
)
from desksense.features import (
    PRIMARY_FEATURE_DEFINITION,
    PRIMARY_FEATURE_DEFINITION_VERSION,
    PRIMARY_FEATURE_NAME,
    PRIMARY_FEATURE_UNITS,
)


FROZEN_BASELINE_SCHEMA_VERSION = 1
FROZEN_BASELINE_ARTIFACT_TYPE = (
    "desksense_frozen_left_right_midpoint_baseline"
)
EXTERNAL_EVALUATION_SCHEMA_VERSION = 1
EXTERNAL_EVALUATION_REPORT_TYPE = (
    "frozen_baseline_cross_session_external_evaluation"
)
CLASSIFIER_TYPE = "one_dimensional_midpoint_threshold"
FITTING_RULE = (
    "Compute the arithmetic mean of the primary feature for every accepted "
    "LEFT sample and every accepted RIGHT sample in the source development "
    "session; set threshold_db to the midpoint of those two means; infer the "
    "lower- and higher-feature zones from the ordering of the means."
)
TIE_RULE = "feature_value >= threshold_db predicts higher_feature_zone"
FEATURE_SOURCE_WINDOW = "complete retained tap_window array"
WILSON_95_Z = 1.96
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_WAVEFORM_KEYS = {
    "audio_samples",
    "capture",
    "metadata_json",
    "raw_audio",
    "tap_window",
    "waveform",
    "waveforms",
}


class FrozenBaselineError(RuntimeError):
    """A Phase 2C baseline or external-evaluation operation failed."""


class FrozenBaselineValidationError(FrozenBaselineError):
    """A frozen-baseline artifact failed strict validation."""


def create_frozen_baseline(
    session_path: Path,
    *,
    interaction_context: str,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Fit the fixed midpoint rule using every accepted development sample."""

    _validate_interaction_context(interaction_context, "source development")
    try:
        dataset = load_dataset_session(Path(session_path))
    except DatasetAnalysisError as error:
        raise FrozenBaselineError(
            f"Source development dataset validation failed: {error}"
        ) from error
    _require_left_right_dataset(dataset.session_metadata, phase="frozen fitting")

    try:
        feature_records = extract_dataset_features(dataset)
        fitted = fit_peak_ratio_baseline(feature_records)
    except DatasetAnalysisError as error:
        raise FrozenBaselineError(
            f"Frozen baseline fitting failed: {error}"
        ) from error
    fingerprint = source_dataset_fingerprint(dataset)
    sample_component_digests = {
        component["path"]: component["sha256_hex"]
        for component in fingerprint["components"]
        if component["role"] == "accepted_sample_artifact"
    }
    membership: list[dict[str, Any]] = []
    for record in feature_records:
        sample_path_value = str(record["sample_path"])
        feature_value = record["features"].get(PRIMARY_FEATURE_NAME)
        if feature_value is None:
            raise FrozenBaselineError(
                f"Primary feature is unavailable for sample {record['sample_id']}."
            )
        try:
            npz_digest = sample_component_digests[sample_path_value]
        except KeyError as error:
            raise FrozenBaselineError(
                f"Fingerprint membership is missing sample {sample_path_value}."
            ) from error
        membership.append(
            {
                "sample_id": str(record["sample_id"]),
                "zone": str(record["zone"]),
                "accepted_sample_number": int(record["accepted_sample_number"]),
                "collection_order_index": int(record["collection_order_index"]),
                "sample_path": sample_path_value,
                "npz_sha256_hex": npz_digest,
                "primary_feature_value_db": _finite_float(
                    feature_value, "primary feature"
                ),
            }
        )

    session = dataset.session_metadata
    counts = {
        zone: sum(1 for record in feature_records if record["zone"] == zone)
        for zone in EXPECTED_ZONES
    }
    tap_window_frames = {sample.tap_window.shape[0] for sample in dataset.samples}
    if len(tap_window_frames) != 1:
        raise FrozenBaselineError(
            "Frozen fitting requires a consistent tap-window frame count."
        )
    window_frames = int(next(iter(tap_window_frames)))
    sample_rate_hz = float(session["capture_configuration"]["sample_rate_hz"])
    endpoint = session["selected_endpoint"]
    class_means = fitted["training_class_means_db"]
    created_at_utc = _utc_timestamp(now_fn)

    artifact = {
        "baseline_schema_version": FROZEN_BASELINE_SCHEMA_VERSION,
        "artifact_type": FROZEN_BASELINE_ARTIFACT_TYPE,
        "project_phase": "2C",
        "created_at_utc": created_at_utc,
        "feature": {
            "name": PRIMARY_FEATURE_NAME,
            "units": PRIMARY_FEATURE_UNITS,
            "definition": PRIMARY_FEATURE_DEFINITION,
            "definition_version": PRIMARY_FEATURE_DEFINITION_VERSION,
            "source_window": FEATURE_SOURCE_WINDOW,
        },
        "classifier": {
            "type": CLASSIFIER_TYPE,
            "fitting_rule": FITTING_RULE,
            "training_class_means_db": {
                zone: float(class_means[zone]) for zone in EXPECTED_ZONES
            },
            "threshold_db": float(fitted["threshold_db"]),
            "lower_feature_zone": str(fitted["lower_feature_zone"]),
            "higher_feature_zone": str(fitted["higher_feature_zone"]),
            "direction": str(fitted["direction"]),
            "tie_rule": TIE_RULE,
        },
        "source_development_dataset": {
            "session_id": str(session["session_id"]),
            "interaction_context": interaction_context,
            "accepted_sample_count": len(feature_records),
            "accepted_count_by_zone": counts,
            "all_source_accepted_samples_used": True,
            "external_samples_used": False,
            "training_membership_order": (
                "manifest collection_order_index (round-robin LEFT/RIGHT order)"
            ),
            "training_membership": membership,
            "fingerprint": fingerprint,
            "compatibility": {
                "dataset_schema_version": int(session["schema_version"]),
                "endpoint_name": str(endpoint["name"]),
                "host_api_name": str(endpoint["host_api"]["name"]),
                "device_index_at_collection": int(endpoint["index"]),
                "sample_rate_hz": sample_rate_hz,
                "channel_count": int(
                    session["capture_configuration"]["channel_count"]
                ),
                "tap_window_frames": window_frames,
                "tap_window_duration_seconds": float(
                    window_frames / sample_rate_hz
                ),
            },
        },
        "evidence_limitations": _baseline_evidence_limitations(
            interaction_context
        ),
    }
    validate_frozen_baseline(artifact)
    return artifact


def write_frozen_baseline(
    artifact: Mapping[str, Any],
    path: Path,
    *,
    source_session_path: Path,
) -> Path:
    """Write a validated baseline exclusively outside its source dataset."""

    validate_frozen_baseline(artifact)
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("Frozen baseline paths must end in .json.")
    _require_outside_directory(
        destination,
        Path(source_session_path),
        "Frozen baselines must be written outside the source dataset session.",
    )
    return _write_json_exclusive(artifact, destination)


def load_frozen_baseline(path: Path) -> dict[str, Any]:
    """Load a frozen-baseline JSON artifact with strict schema validation."""

    artifact, _ = _load_frozen_baseline_with_file_digest(Path(path))
    return artifact


def validate_frozen_baseline(artifact: Mapping[str, Any]) -> None:
    """Strictly validate a frozen-baseline mapping without repairing it."""

    if not isinstance(artifact, Mapping):
        raise FrozenBaselineValidationError(
            "Frozen baseline must be a JSON object."
        )
    _validate_json_safe_waveform_free(artifact, path="frozen_baseline")
    _require_exact_keys(
        artifact,
        {
            "baseline_schema_version",
            "artifact_type",
            "project_phase",
            "created_at_utc",
            "feature",
            "classifier",
            "source_development_dataset",
            "evidence_limitations",
        },
        "frozen baseline",
    )
    if _positive_int(
        artifact["baseline_schema_version"], "baseline_schema_version"
    ) != FROZEN_BASELINE_SCHEMA_VERSION:
        raise FrozenBaselineValidationError("Unsupported baseline schema version.")
    _require_exact_string(
        artifact["artifact_type"],
        FROZEN_BASELINE_ARTIFACT_TYPE,
        "artifact_type",
    )
    _require_exact_string(artifact["project_phase"], "2C", "project_phase")
    _require_utc_timestamp(artifact["created_at_utc"], "created_at_utc")

    feature = _mapping(artifact["feature"], "feature")
    _require_exact_keys(
        feature,
        {"name", "units", "definition", "definition_version", "source_window"},
        "feature",
    )
    _require_exact_string(feature["name"], PRIMARY_FEATURE_NAME, "feature.name")
    _require_exact_string(feature["units"], PRIMARY_FEATURE_UNITS, "feature.units")
    _require_exact_string(
        feature["definition"], PRIMARY_FEATURE_DEFINITION, "feature.definition"
    )
    if _positive_int(feature["definition_version"], "feature.definition_version") != (
        PRIMARY_FEATURE_DEFINITION_VERSION
    ):
        raise FrozenBaselineValidationError(
            "Unsupported primary feature definition version."
        )
    _require_exact_string(
        feature["source_window"], FEATURE_SOURCE_WINDOW, "feature.source_window"
    )

    classifier = _mapping(artifact["classifier"], "classifier")
    _require_exact_keys(
        classifier,
        {
            "type",
            "fitting_rule",
            "training_class_means_db",
            "threshold_db",
            "lower_feature_zone",
            "higher_feature_zone",
            "direction",
            "tie_rule",
        },
        "classifier",
    )
    _require_exact_string(classifier["type"], CLASSIFIER_TYPE, "classifier.type")
    _require_exact_string(
        classifier["fitting_rule"], FITTING_RULE, "classifier.fitting_rule"
    )
    _require_exact_string(classifier["tie_rule"], TIE_RULE, "classifier.tie_rule")
    means = _mapping(
        classifier["training_class_means_db"],
        "classifier.training_class_means_db",
    )
    _require_exact_keys(means, set(EXPECTED_ZONES), "training class means")
    class_means = {
        zone: _finite_float(means[zone], f"{zone} training mean")
        for zone in EXPECTED_ZONES
    }
    threshold = _finite_float(classifier["threshold_db"], "threshold_db")
    midpoint = (class_means["LEFT"] + class_means["RIGHT"]) / 2.0
    if not _same_float(threshold, midpoint):
        raise FrozenBaselineValidationError(
            "Frozen threshold is inconsistent with the recorded midpoint rule."
        )
    tolerance = 1e-12 * max(
        1.0, abs(class_means["LEFT"]), abs(class_means["RIGHT"])
    )
    if abs(class_means["LEFT"] - class_means["RIGHT"]) <= tolerance:
        raise FrozenBaselineValidationError(
            "Frozen class means do not define a usable direction."
        )
    expected_lower, expected_higher = (
        ("LEFT", "RIGHT")
        if class_means["LEFT"] < class_means["RIGHT"]
        else ("RIGHT", "LEFT")
    )
    _require_exact_string(
        classifier["lower_feature_zone"],
        expected_lower,
        "classifier.lower_feature_zone",
    )
    _require_exact_string(
        classifier["higher_feature_zone"],
        expected_higher,
        "classifier.higher_feature_zone",
    )
    _require_exact_string(
        classifier["direction"],
        f"{expected_lower} below threshold; {expected_higher} at or above threshold",
        "classifier.direction",
    )

    source = _mapping(
        artifact["source_development_dataset"], "source_development_dataset"
    )
    _validate_source_dataset_metadata(source, class_means)

    limitations = artifact["evidence_limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item for item in limitations)
    ):
        raise FrozenBaselineValidationError(
            "evidence_limitations must be a non-empty list of strings."
        )
    expected_limitations = _baseline_evidence_limitations(
        str(source["interaction_context"])
    )
    if limitations != expected_limitations:
        raise FrozenBaselineValidationError(
            "evidence_limitations are inconsistent with the source context."
        )


def evaluate_frozen_baseline(
    external_session_path: Path,
    baseline_path: Path,
    *,
    interaction_context: str,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Apply one already-frozen baseline to every accepted external sample."""

    _validate_interaction_context(interaction_context, "external")
    baseline, baseline_file_digest = _load_frozen_baseline_with_file_digest(
        Path(baseline_path)
    )
    try:
        external = load_dataset_session(Path(external_session_path))
    except DatasetAnalysisError as error:
        raise FrozenBaselineError(
            f"External dataset validation failed: {error}"
        ) from error
    _require_left_right_dataset(
        external.session_metadata, phase="external evaluation"
    )

    source = baseline["source_development_dataset"]
    external_session_id = str(external.session_metadata["session_id"])
    if external_session_id == source["session_id"]:
        raise FrozenBaselineError(
            "External dataset session ID matches the frozen source development "
            "session; cross-session evaluation requires a different session."
        )
    external_fingerprint = source_dataset_fingerprint(external)
    if external_fingerprint["digest_hex"] == source["fingerprint"]["digest_hex"]:
        raise FrozenBaselineError(
            "External dataset fingerprint matches the frozen source dataset."
        )

    baseline_created = _parse_utc_timestamp(
        baseline["created_at_utc"], "baseline created_at_utc"
    )
    external_created = _parse_utc_timestamp(
        external.session_metadata["created_at_utc"],
        "external session created_at_utc",
    )
    baseline_precedes_external = baseline_created < external_created

    _validate_external_compatibility(baseline, external)
    try:
        feature_records = extract_dataset_features(external)
    except DatasetAnalysisError as error:
        raise FrozenBaselineError(
            f"External feature extraction failed: {error}"
        ) from error
    classifier = baseline["classifier"]
    frozen_predictor = {
        "threshold_db": classifier["threshold_db"],
        "lower_feature_zone": classifier["lower_feature_zone"],
        "higher_feature_zone": classifier["higher_feature_zone"],
    }
    try:
        predictions = [
            predict_peak_ratio(record, frozen_predictor)
            for record in feature_records
        ]
        metrics = _external_evaluation_summary(predictions)
        descriptive = descriptive_feature_statistics(feature_records)
    except DatasetAnalysisError as error:
        raise FrozenBaselineError(
            f"External frozen prediction failed: {error}"
        ) from error
    generated_at = _utc_timestamp(now_fn)
    counts = {
        zone: sum(1 for record in feature_records if record["zone"] == zone)
        for zone in EXPECTED_ZONES
    }
    report = {
        "external_evaluation_schema_version": EXTERNAL_EVALUATION_SCHEMA_VERSION,
        "report_type": EXTERNAL_EVALUATION_REPORT_TYPE,
        "project_phase": "2C",
        "status": "ok",
        "generated_at_utc": generated_at,
        "frozen_baseline": {
            "artifact_path": str(Path(baseline_path).resolve()),
            "artifact_file_sha256_hex": baseline_file_digest,
            "artifact": baseline,
        },
        "source_development_session": {
            "session_id": source["session_id"],
            "interaction_context": source["interaction_context"],
            "dataset_fingerprint": source["fingerprint"],
            "accepted_sample_count": source["accepted_sample_count"],
            "accepted_count_by_zone": source["accepted_count_by_zone"],
        },
        "external_session": {
            "session_id": external_session_id,
            "session_path": str(external.session_path),
            "interaction_context": interaction_context,
            "interaction_context_source": (
                "explicit analyst-supplied parameter; not inferred from audio "
                "or Phase 2A metadata"
            ),
            "dataset_fingerprint": external_fingerprint,
            "accepted_sample_count": len(feature_records),
            "accepted_count_by_zone": counts,
            "rejected_attempt_records": external.rejected_attempt_count,
            "integrity_validation": external.integrity_validation,
        },
        "precommitment_timing": {
            "baseline_created_at_utc": baseline["created_at_utc"],
            "external_session_created_at_utc": external.session_metadata[
                "created_at_utc"
            ],
            "baseline_timestamp_precedes_external_session": (
                baseline_precedes_external
            ),
            "enforced_as_hard_failure": False,
            "interpretation": (
                "Timestamps are recorded for provenance but are not treated as "
                "trusted proof of experimental ordering. The baseline should be "
                "reviewed and checkpointed before external collection."
            ),
        },
        "evaluation_protocol": {
            "label": "frozen-baseline cross-session external evaluation",
            "primary_feature": PRIMARY_FEATURE_NAME,
            "frozen_threshold_db": float(classifier["threshold_db"]),
            "frozen_lower_feature_zone": classifier["lower_feature_zone"],
            "frozen_higher_feature_zone": classifier["higher_feature_zone"],
            "frozen_tie_rule": classifier["tie_rule"],
            "all_accepted_external_samples_classified": True,
            "external_samples_used_to_fit_threshold": False,
            "threshold_refit_performed": False,
            "direction_relearned": False,
            "feature_selection_performed": False,
            "external_normalization_fitted": False,
        },
        "metrics": metrics,
        "predictions": predictions,
        "post_prediction_descriptive_statistics": {
            "used_to_fit_or_change_predictions": False,
            "statistics": descriptive,
        },
        "evidence_boundary": _external_evidence_boundary(
            str(source["interaction_context"]), interaction_context
        ),
        "report_privacy": {
            "analysis_is_local_and_offline": True,
            "microphone_accessed": False,
            "source_dataset_files_modified": False,
            "external_dataset_files_modified": False,
            "baseline_file_modified": False,
            "raw_waveforms_included": False,
            "network_upload_performed": False,
        },
    }
    _validate_json_safe_waveform_free(report, path="external_report")
    json.dumps(report, allow_nan=False)
    return report


def write_external_evaluation_report(
    report: Mapping[str, Any],
    path: Path,
    *,
    external_session_path: Path,
    baseline_path: Path,
) -> Path:
    """Write an external report exclusively without touching input artifacts."""

    if report.get("report_type") != EXTERNAL_EVALUATION_REPORT_TYPE:
        raise ValueError("Unexpected external-evaluation report type.")
    _validate_json_safe_waveform_free(report, path="external_report")
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("External evaluation report paths must end in .json.")
    _require_outside_directory(
        destination,
        Path(external_session_path),
        "External reports must be written outside the evaluated dataset session.",
    )
    if destination.resolve() == Path(baseline_path).resolve():
        raise ValueError("External reports must not replace the frozen baseline.")
    return _write_json_exclusive(report, destination)


def wilson_score_interval(
    correct_count: int,
    total_count: int,
) -> dict[str, Any]:
    """Return a two-sided Wilson interval for finite-sample accuracy."""

    if (
        isinstance(correct_count, bool)
        or not isinstance(correct_count, int)
        or isinstance(total_count, bool)
        or not isinstance(total_count, int)
        or total_count <= 0
        or correct_count < 0
        or correct_count > total_count
    ):
        raise ValueError(
            "Wilson counts must satisfy 0 <= correct <= total and total > 0."
        )
    z_value = WILSON_95_Z
    proportion = correct_count / total_count
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / total_count
    center = (proportion + z_squared / (2.0 * total_count)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / total_count
            + z_squared / (4.0 * total_count * total_count)
        )
        / denominator
    )
    return {
        "method": "two-sided Wilson score interval",
        "confidence_level": 0.95,
        "z_value": z_value,
        "correct_count": correct_count,
        "total_count": total_count,
        "lower_bound": float(max(0.0, center - half_width)),
        "upper_bound": float(min(1.0, center + half_width)),
        "interpretation": (
            "Finite-sample accuracy uncertainty interval; not a guarantee of "
            "future performance."
        ),
    }


def format_frozen_baseline_summary(artifact: Mapping[str, Any]) -> str:
    """Render the auditable fields of a newly fitted baseline."""

    validate_frozen_baseline(artifact)
    source = artifact["source_development_dataset"]
    classifier = artifact["classifier"]
    means = classifier["training_class_means_db"]
    fingerprint = source["fingerprint"]
    return "\n".join(
        [
            "DeskSense Phase 2C frozen LEFT/RIGHT midpoint baseline",
            f"Source development session: {source['session_id']}",
            f"Source interaction context: {source['interaction_context']}",
            (
                "Training membership: all "
                f"{source['accepted_sample_count']} accepted sample(s); "
                f"LEFT={source['accepted_count_by_zone']['LEFT']}, "
                f"RIGHT={source['accepted_count_by_zone']['RIGHT']}"
            ),
            (
                f"Primary feature: {artifact['feature']['name']} "
                f"({artifact['feature']['units']})"
            ),
            (
                f"Training means: LEFT={means['LEFT']:.6f} dB, "
                f"RIGHT={means['RIGHT']:.6f} dB"
            ),
            f"Frozen threshold: {classifier['threshold_db']:.6f} dB",
            f"Frozen direction: {classifier['direction']}",
            f"Source dataset SHA-256: {fingerprint['digest_hex']}",
            (
                "No external samples were used. Raw waveforms are not stored "
                "in the baseline."
            ),
        ]
    )


def format_external_evaluation_summary(report: Mapping[str, Any]) -> str:
    """Render a scoped external-evaluation summary for the terminal."""

    source = report["source_development_session"]
    external = report["external_session"]
    protocol = report["evaluation_protocol"]
    metrics = report["metrics"]
    matrix = metrics["confusion_matrix"]
    interval = metrics["accuracy_wilson_95_interval"]
    evidence = report["evidence_boundary"]
    lines = [
        "DeskSense frozen-baseline cross-session external evaluation",
        f"Source development session: {source['session_id']}",
        f"External session: {external['session_id']}",
        (
            "Interaction contexts: source="
            f"{source['interaction_context']}; external="
            f"{external['interaction_context']}"
        ),
        (
            f"Frozen threshold: {protocol['frozen_threshold_db']:.6f} dB; "
            f"{protocol['frozen_lower_feature_zone']} below, "
            f"{protocol['frozen_higher_feature_zone']} at or above"
        ),
        "Threshold and direction were loaded unchanged; no external refit occurred.",
        (
            f"Correct: {metrics['correct_count']}/{metrics['total_count']} "
            f"({metrics['accuracy'] * 100.0:.2f}%)"
        ),
        (
            "95% Wilson interval: "
            f"[{interval['lower_bound'] * 100.0:.2f}%, "
            f"{interval['upper_bound'] * 100.0:.2f}%]"
        ),
        (
            "Confusion: actual LEFT -> LEFT "
            f"{matrix['actual_LEFT']['predicted_LEFT']}, RIGHT "
            f"{matrix['actual_LEFT']['predicted_RIGHT']}; actual RIGHT -> LEFT "
            f"{matrix['actual_RIGHT']['predicted_LEFT']}, RIGHT "
            f"{matrix['actual_RIGHT']['predicted_RIGHT']}"
        ),
        "Predictions:",
    ]
    for prediction in report["predictions"]:
        lines.append(
            f"  {prediction['sample_id']}: actual={prediction['actual_label']}, "
            f"predicted={prediction['predicted_label']}, "
            f"feature={prediction['feature_value_db']:.6f} dB, "
            f"absolute margin={prediction['absolute_margin_db']:.6f} dB, "
            f"actual-class margin={prediction['actual_class_margin_db']:+.6f} dB"
        )
    lines.extend(
        [
            "Evidence boundary:",
            f"  {evidence['source_context_statement']}",
            f"  {evidence['external_context_statement']}",
            f"  {evidence['scope_statement']}",
            "Raw waveform arrays are not included in this report.",
        ]
    )
    return "\n".join(lines)


def _validate_source_dataset_metadata(
    source: Mapping[str, Any], class_means: Mapping[str, float]
) -> None:
    _require_exact_keys(
        source,
        {
            "session_id",
            "interaction_context",
            "accepted_sample_count",
            "accepted_count_by_zone",
            "all_source_accepted_samples_used",
            "external_samples_used",
            "training_membership_order",
            "training_membership",
            "fingerprint",
            "compatibility",
        },
        "source_development_dataset",
    )
    _nonempty_string(source["session_id"], "source session_id")
    _validate_interaction_context(
        source["interaction_context"], "source development"
    )
    total = _positive_int(
        source["accepted_sample_count"], "source accepted_sample_count"
    )
    counts = _mapping(
        source["accepted_count_by_zone"], "source accepted_count_by_zone"
    )
    _require_exact_keys(counts, set(EXPECTED_ZONES), "source accepted counts")
    zone_counts = {
        zone: _positive_int(counts[zone], f"source {zone} count")
        for zone in EXPECTED_ZONES
    }
    if total != sum(zone_counts.values()):
        raise FrozenBaselineValidationError(
            "Source accepted sample count disagrees with per-zone counts."
        )
    if source["all_source_accepted_samples_used"] is not True:
        raise FrozenBaselineValidationError(
            "Frozen baseline must state that all source accepted samples were used."
        )
    if source["external_samples_used"] is not False:
        raise FrozenBaselineValidationError(
            "Frozen baseline must state that no external samples were used."
        )
    _require_exact_string(
        source["training_membership_order"],
        "manifest collection_order_index (round-robin LEFT/RIGHT order)",
        "source training_membership_order",
    )

    membership = source["training_membership"]
    if not isinstance(membership, list) or len(membership) != total:
        raise FrozenBaselineValidationError(
            "Source training_membership length disagrees with accepted count."
        )
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_numbers: set[tuple[str, int]] = set()
    values_by_zone: dict[str, list[float]] = {
        zone: [] for zone in EXPECTED_ZONES
    }
    for expected_order, item in enumerate(membership, start=1):
        member = _mapping(item, f"training_membership[{expected_order - 1}]")
        _require_exact_keys(
            member,
            {
                "sample_id",
                "zone",
                "accepted_sample_number",
                "collection_order_index",
                "sample_path",
                "npz_sha256_hex",
                "primary_feature_value_db",
            },
            f"training_membership[{expected_order - 1}]",
        )
        sample_id = _nonempty_string(member["sample_id"], "training sample_id")
        if sample_id.casefold() in seen_ids:
            raise FrozenBaselineValidationError(
                f"Duplicate training sample ID: {sample_id}"
            )
        seen_ids.add(sample_id.casefold())
        zone = _nonempty_string(member["zone"], "training zone")
        if zone not in EXPECTED_ZONES:
            raise FrozenBaselineValidationError(
                f"Unsupported training zone: {zone!r}"
            )
        accepted_number = _positive_int(
            member["accepted_sample_number"], "training accepted sample number"
        )
        identity = (zone, accepted_number)
        if identity in seen_numbers:
            raise FrozenBaselineValidationError(
                f"Duplicate training accepted number: {zone} #{accepted_number}"
            )
        seen_numbers.add(identity)
        order = _positive_int(
            member["collection_order_index"], "training collection order"
        )
        if order != expected_order:
            raise FrozenBaselineValidationError(
                "Training membership is not in manifest collection order."
            )
        expected_zone = EXPECTED_ZONES[(expected_order - 1) % 2]
        expected_number = (expected_order - 1) // 2 + 1
        if zone != expected_zone or accepted_number != expected_number:
            raise FrozenBaselineValidationError(
                "Training membership does not follow round-robin LEFT/RIGHT order."
            )
        sample_path = _validate_portable_sample_path(member["sample_path"])
        if sample_path.casefold() in seen_paths:
            raise FrozenBaselineValidationError(
                f"Duplicate training sample path: {sample_path}"
            )
        seen_paths.add(sample_path.casefold())
        _validate_sha256(member["npz_sha256_hex"], "training NPZ SHA-256")
        values_by_zone[zone].append(
            _finite_float(
                member["primary_feature_value_db"],
                "training primary feature value",
            )
        )

    observed_counts = {zone: len(values_by_zone[zone]) for zone in EXPECTED_ZONES}
    if observed_counts != zone_counts:
        raise FrozenBaselineValidationError(
            "Training membership counts disagree with source counts."
        )
    for zone in EXPECTED_ZONES:
        expected_numbers = list(range(1, zone_counts[zone] + 1))
        actual_numbers = sorted(
            number for member_zone, number in seen_numbers if member_zone == zone
        )
        if actual_numbers != expected_numbers:
            raise FrozenBaselineValidationError(
                f"Training membership numbering for {zone} is not contiguous."
            )
        membership_mean = math.fsum(values_by_zone[zone]) / zone_counts[zone]
        if not _same_float(membership_mean, class_means[zone]):
            raise FrozenBaselineValidationError(
                f"Recorded {zone} mean disagrees with training membership values."
            )

    _validate_source_fingerprint(
        _mapping(source["fingerprint"], "source fingerprint"), membership
    )
    _validate_compatibility_metadata(
        _mapping(source["compatibility"], "source compatibility")
    )


def _validate_source_fingerprint(
    fingerprint: Mapping[str, Any], membership: Sequence[Mapping[str, Any]]
) -> None:
    _require_exact_keys(
        fingerprint,
        {
            "schema_version",
            "algorithm",
            "scheme",
            "logical_order",
            "framing",
            "digest_hex",
            "components",
            "rejected_attempts_included",
        },
        "source fingerprint",
    )
    if _positive_int(
        fingerprint["schema_version"], "source fingerprint schema version"
    ) != SOURCE_DATASET_FINGERPRINT_SCHEMA_VERSION:
        raise FrozenBaselineValidationError(
            "Unsupported source dataset fingerprint schema version."
        )
    _require_exact_string(
        fingerprint["algorithm"], "SHA-256", "source fingerprint algorithm"
    )
    _require_exact_string(
        fingerprint["scheme"],
        SOURCE_DATASET_FINGERPRINT_SCHEME,
        "source fingerprint scheme",
    )
    _require_exact_string(
        fingerprint["logical_order"],
        SOURCE_DATASET_FINGERPRINT_ORDER,
        "source fingerprint logical order",
    )
    _require_exact_string(
        fingerprint["framing"],
        SOURCE_DATASET_FINGERPRINT_FRAMING,
        "source fingerprint framing",
    )
    _validate_sha256(fingerprint["digest_hex"], "source fingerprint digest")
    if fingerprint["rejected_attempts_included"] is not False:
        raise FrozenBaselineValidationError(
            "Rejected-attempt metadata must not be part of the model-source "
            "fingerprint."
        )
    components = fingerprint["components"]
    if not isinstance(components, list) or len(components) != len(membership) + 2:
        raise FrozenBaselineValidationError(
            "Source fingerprint component count is inconsistent."
        )
    expected = [
        ("session_metadata", "session.json", None),
        ("accepted_manifest", "manifest.jsonl", None),
        *[
            (
                "accepted_sample_artifact",
                str(member["sample_path"]),
                str(member["npz_sha256_hex"]),
            )
            for member in membership
        ],
    ]
    seen_paths: set[str] = set()
    for index, (item, expected_item) in enumerate(
        zip(components, expected, strict=True)
    ):
        component = _mapping(item, f"source fingerprint component {index}")
        _require_exact_keys(
            component,
            {"role", "path", "byte_size", "sha256_hex"},
            f"source fingerprint component {index}",
        )
        role, path, expected_digest = expected_item
        _require_exact_string(component["role"], role, "fingerprint component role")
        _require_exact_string(component["path"], path, "fingerprint component path")
        byte_size = _positive_int(
            component["byte_size"], "fingerprint component byte_size"
        )
        if byte_size <= 0:
            raise FrozenBaselineValidationError(
                "Fingerprint component byte_size must be positive."
            )
        digest = _validate_sha256(
            component["sha256_hex"], "fingerprint component SHA-256"
        )
        if expected_digest is not None and digest != expected_digest:
            raise FrozenBaselineValidationError(
                "Training membership NPZ digest disagrees with source fingerprint."
            )
        if path.casefold() in seen_paths:
            raise FrozenBaselineValidationError(
                f"Duplicate source fingerprint component path: {path}"
            )
        seen_paths.add(path.casefold())


def _validate_compatibility_metadata(compatibility: Mapping[str, Any]) -> None:
    _require_exact_keys(
        compatibility,
        {
            "dataset_schema_version",
            "endpoint_name",
            "host_api_name",
            "device_index_at_collection",
            "sample_rate_hz",
            "channel_count",
            "tap_window_frames",
            "tap_window_duration_seconds",
        },
        "source compatibility",
    )
    _positive_int(
        compatibility["dataset_schema_version"], "dataset schema version"
    )
    _nonempty_string(compatibility["endpoint_name"], "endpoint name")
    _nonempty_string(compatibility["host_api_name"], "host API name")
    _nonnegative_int(
        compatibility["device_index_at_collection"],
        "device index at collection",
    )
    sample_rate = _positive_float(
        compatibility["sample_rate_hz"], "sample rate"
    )
    channels = _positive_int(compatibility["channel_count"], "channel count")
    if channels < 2:
        raise FrozenBaselineValidationError(
            "Frozen peak-ratio baseline requires at least two channels."
        )
    frames = _positive_int(
        compatibility["tap_window_frames"], "tap-window frame count"
    )
    duration = _positive_float(
        compatibility["tap_window_duration_seconds"], "tap-window duration"
    )
    if not _same_float(duration, frames / sample_rate):
        raise FrozenBaselineValidationError(
            "Tap-window duration disagrees with frames and sample rate."
        )


def _load_frozen_baseline_with_file_digest(
    path: Path,
) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FrozenBaselineValidationError(
            f"Frozen baseline does not exist or is not a file: {resolved}"
        )
    try:
        content = resolved.read_bytes()
    except OSError as error:
        raise FrozenBaselineValidationError(
            f"Could not read frozen baseline: {error}"
        ) from error
    try:
        text = content.decode("utf-8")
    except UnicodeError as error:
        raise FrozenBaselineValidationError(
            f"Frozen baseline is not UTF-8: {error}"
        ) from error
    artifact = _load_strict_json_object(text, "frozen baseline")
    validate_frozen_baseline(artifact)
    return artifact, hashlib.sha256(content).hexdigest()


def _validate_external_compatibility(
    baseline: Mapping[str, Any], external: Any
) -> None:
    compatibility = baseline["source_development_dataset"]["compatibility"]
    session = external.session_metadata
    endpoint = session["selected_endpoint"]
    checks = [
        (
            int(session["schema_version"])
            == int(compatibility["dataset_schema_version"]),
            "dataset schema version",
        ),
        (
            str(endpoint["name"]) == str(compatibility["endpoint_name"]),
            "endpoint name",
        ),
        (
            str(endpoint["host_api"]["name"])
            == str(compatibility["host_api_name"]),
            "host API name",
        ),
        (
            int(session["capture_configuration"]["channel_count"])
            == int(compatibility["channel_count"]),
            "channel count",
        ),
        (
            _same_float(
                float(session["capture_configuration"]["sample_rate_hz"]),
                float(compatibility["sample_rate_hz"]),
            ),
            "sample rate",
        ),
    ]
    mismatches = [label for matched, label in checks if not matched]
    window_frames = {sample.tap_window.shape[0] for sample in external.samples}
    if window_frames != {int(compatibility["tap_window_frames"])}:
        mismatches.append("tap-window frame count")
    if mismatches:
        raise FrozenBaselineError(
            "External dataset is incompatible with the frozen feature domain: "
            + ", ".join(mismatches)
            + ". Device index is intentionally not used as a stable identity."
        )


def _external_evaluation_summary(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not predictions:
        raise FrozenBaselineError("External evaluation requires predictions.")
    total = len(predictions)
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
    actual_class_margins = [
        _finite_float(
            prediction["actual_class_margin_db"], "actual-class margin"
        )
        for prediction in predictions
    ]
    correct_absolute_margins = [
        _finite_float(prediction["absolute_margin_db"], "absolute margin")
        for prediction in predictions
        if prediction["correct"]
    ]
    return {
        "confusion_matrix": confusion_matrix(predictions),
        "correct_count": int(correct),
        "total_count": int(total),
        "accuracy": float(correct / total),
        "per_class": per_class,
        "accuracy_wilson_95_interval": wilson_score_interval(correct, total),
        "signed_margin_toward_actual_class_summary_db": {
            "definition": (
                "signed distance from the frozen threshold toward the actual "
                "class; positive is correct-side support, negative is wrong-side "
                "support, and an exact zero is resolved by the frozen tie rule"
            ),
            **_summarize_finite_values(actual_class_margins),
        },
        "correct_prediction_absolute_margin_summary_db": {
            "definition": (
                "absolute threshold distance for correctly classified samples "
                "only; margin is not a calibrated probability"
            ),
            **_summarize_finite_values(correct_absolute_margins),
        },
        "margin_is_calibrated_probability": False,
    }


def _summarize_finite_values(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
        }
    ordered = sorted(_finite_float(value, "summary value") for value in values)
    count = len(ordered)
    midpoint = count // 2
    median = (
        ordered[midpoint]
        if count % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    return {
        "count": count,
        "minimum": float(ordered[0]),
        "median": float(median),
        "mean": float(math.fsum(ordered) / count),
    }


def _external_evidence_boundary(
    source_context: str, external_context: str
) -> dict[str, Any]:
    _validate_interaction_context(source_context, "source development")
    _validate_interaction_context(external_context, "external")
    source_statement = (
        "Source development context: LEFT = left hand + left location and "
        "RIGHT = right hand + right location; it did not isolate location alone."
        if source_context == INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED
        else "Source development context used the same hand/finger for both zones."
    )
    external_statement = (
        "External context: the same hand/finger was used for LEFT and RIGHT, "
        "controlling the previous tapping-hand variable more cleanly."
        if external_context == INTERACTION_CONTEXT_SAME_HAND
        else (
            "External context remains hand/location-confounded: LEFT and RIGHT "
            "also differ by tapping hand."
        )
    )
    return {
        "source_interaction_context": source_context,
        "external_interaction_context": external_context,
        "contexts_explicitly_supplied_not_inferred": True,
        "source_context_statement": source_statement,
        "external_context_statement": external_statement,
        "same_hand_controls_previous_hand_variable_more_cleanly": (
            source_context == INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED
            and external_context == INTERACTION_CONTEXT_SAME_HAND
        ),
        "no_refit_declaration": (
            "No external sample changed the feature, threshold, class direction, "
            "tie rule, or normalization; every prediction used the frozen artifact."
        ),
        "scope_statement": (
            "Even successful same-hand cross-session performance remains scoped "
            "to one user, laptop, and desk/setup unless separately documented; "
            "it is not cross-device generalization."
        ),
        "not_claimed": [
            "final general DeskSense accuracy",
            "cross-device accuracy",
            "automatic cross-laptop adaptation",
            "proven physical microphone time-of-arrival",
        ],
        "residual_independence_limit": (
            "Different session IDs and full-dataset fingerprints prevent obvious "
            "whole-session reuse but cannot prove every aspect of experimental "
            "independence."
        ),
    }


def _baseline_evidence_limitations(interaction_context: str) -> list[str]:
    _validate_interaction_context(interaction_context, "source development")
    context_statement = (
        "The source interaction context is hand-location-confounded: LEFT = "
        "left hand + left location and RIGHT = right hand + right location, so "
        "the source dataset cannot isolate location alone."
        if interaction_context == INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED
        else (
            "The source interaction context used the same hand/finger for both "
            "LEFT and RIGHT; this controls tapping hand but does not establish "
            "generalization."
        )
    )
    return [
        (
            "The source development session is a single-session dataset from "
            "one user, laptop, desk, and setup."
        ),
        context_statement,
        (
            "Interaction context is explicitly supplied; the artifact does not "
            "infer tapping hand from audio or Phase 2A metadata."
        ),
        (
            "A future external result must use this feature, threshold, "
            "direction, and tie rule unchanged and must not refit on external "
            "samples."
        ),
        "The frozen baseline is not a cross-device or final DeskSense model.",
    ]


def _require_left_right_dataset(session: Mapping[str, Any], *, phase: str) -> None:
    zones = tuple(str(zone) for zone in session["zones"])
    if zones != EXPECTED_ZONES:
        raise FrozenBaselineError(
            f"Phase 2C {phase} requires session zones exactly LEFT, RIGHT."
        )


def _validate_interaction_context(value: Any, label: str) -> str:
    if not isinstance(value, str) or value not in INTERACTION_CONTEXT_CHOICES:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} interaction context must be one of: "
            + ", ".join(INTERACTION_CONTEXT_CHOICES)
        )
    return value


def _write_json_exclusive(value: Mapping[str, Any], destination: Path) -> Path:
    _validate_json_safe_waveform_free(value, path="JSON artifact")
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            created = True
            written = output.write(serialized)
            if written != len(serialized):
                raise OSError("The complete JSON artifact could not be written.")
    except BaseException:
        if created:
            try:
                destination.unlink()
            except OSError as cleanup_error:
                raise OSError(
                    "JSON artifact writing failed and the partial file could not "
                    f"be removed: {destination} ({cleanup_error})"
                ) from cleanup_error
        raise
    return destination.resolve()


def _require_outside_directory(
    destination: Path, directory: Path, message: str
) -> None:
    resolved_destination = destination.resolve()
    resolved_directory = directory.resolve()
    if (
        resolved_destination == resolved_directory
        or resolved_destination.is_relative_to(resolved_directory)
    ):
        raise ValueError(message)


def _load_strict_json_object(content: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda token: _reject_json_constant(token),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise FrozenBaselineValidationError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise FrozenBaselineValidationError(f"{label.capitalize()} must be an object.")
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


def _validate_json_safe_waveform_free(value: Any, *, path: str) -> None:
    if isinstance(value, np.ndarray) or isinstance(value, np.generic):
        raise FrozenBaselineValidationError(
            f"{path} contains a NumPy value that is not JSON safe."
        )
    if isinstance(value, Mapping):
        forbidden = sorted(
            str(key)
            for key in value
            if str(key).casefold() in _FORBIDDEN_WAVEFORM_KEYS
        )
        if forbidden:
            raise FrozenBaselineValidationError(
                f"{path} contains forbidden waveform key(s): {forbidden}"
            )
        for key, child in value.items():
            _validate_json_safe_waveform_free(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_safe_waveform_free(child, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise FrozenBaselineValidationError(f"{path} contains a non-finite float.")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise FrozenBaselineValidationError(
            f"{path} contains unsupported JSON value {type(value).__name__}."
        )


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = {str(key) for key in value}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise FrozenBaselineValidationError(
            f"{context.capitalize()} fields are invalid: " + "; ".join(details)
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrozenBaselineValidationError(f"{label.capitalize()} must be an object.")
    return value


def _require_exact_string(value: Any, expected: str, label: str) -> str:
    converted = _nonempty_string(value, label)
    if converted != expected:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} must be {expected!r}."
        )
    return converted


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} must be a non-empty string."
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} must be a positive integer."
        )
    return int(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} must be a non-negative integer."
        )
    return int(value)


def _positive_float(value: Any, label: str) -> float:
    converted = _finite_float(value, label)
    if converted <= 0.0:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} must be positive."
        )
    return converted


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FrozenBaselineValidationError(f"{label.capitalize()} must be finite.")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} must be finite."
        ) from error
    if not math.isfinite(converted):
        raise FrozenBaselineValidationError(f"{label.capitalize()} must be finite.")
    return converted


def _validate_sha256(value: Any, label: str) -> str:
    converted = _nonempty_string(value, label)
    if _HEX_SHA256.fullmatch(converted) is None:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} must be 64 lowercase hexadecimal characters."
        )
    return converted


def _validate_portable_sample_path(value: Any) -> str:
    converted = _nonempty_string(value, "training sample path")
    if "\\" in converted:
        raise FrozenBaselineValidationError(
            "Training sample paths must use portable forward slashes."
        )
    path = PurePosixPath(converted)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != "samples"
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[1]
        or path.suffix.casefold() != ".npz"
    ):
        raise FrozenBaselineValidationError(
            "Training sample paths must use samples/<name>.npz."
        )
    return path.as_posix()


def _require_utc_timestamp(value: Any, label: str) -> None:
    _parse_utc_timestamp(value, label)


def _parse_utc_timestamp(value: Any, label: str) -> datetime:
    converted = _nonempty_string(value, label)
    try:
        parsed = datetime.fromisoformat(converted.replace("Z", "+00:00"))
    except ValueError as error:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} is not a valid ISO timestamp."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} must include a UTC offset."
        )
    if parsed.utcoffset().total_seconds() != 0:
        raise FrozenBaselineValidationError(
            f"{label.capitalize()} must be expressed in UTC."
        )
    return parsed.astimezone(timezone.utc)


def _utc_timestamp(now_fn: Callable[[], datetime] | None) -> str:
    timestamp = datetime.now(timezone.utc) if now_fn is None else now_fn()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise FrozenBaselineError("Phase 2C timestamps must be timezone-aware.")
    return timestamp.astimezone(timezone.utc).isoformat()


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
