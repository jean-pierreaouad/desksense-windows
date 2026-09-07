"""Strict provisional Stage 2 v2 research artifact support.

The artifact produced here is development-only research metadata.  It is not a
production baseline, is not used by live sensing, and has not been externally
validated.  Generation is intentionally restricted to the exact consumed
Phase 3B Sessions A and B selected in Experiment 19.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desksense.robustness_dataset import load_robustness_dataset
from desksense.tapness import stage1_policy_metadata
from desksense.tapness_v2_research import (
    TAPNESS_V2_RESEARCH_FEATURE_NAMES,
    TAPNESS_V2_RESEARCH_FEATURE_SCHEMA_VERSION,
    TAPNESS_V2_RESEARCH_L2,
    TapnessV2ResearchError,
    ResearchLogisticModel,
    evaluate_tapness_v2_research_pair,
    fit_grouped_v2_research_model,
    tapness_v2_research_feature_schema,
    transform_tapness_v2_research_features,
)


PROVISIONAL_V2_ARTIFACT_SCHEMA_VERSION = 1
PROVISIONAL_V2_ARTIFACT_TYPE = (
    "desksense_provisional_tapness_v2_research_baseline"
)
PROVISIONAL_V2_POSITIVE_LABEL = "TAP"
PROVISIONAL_V2_NEGATIVE_LABEL = "NON_TAP"
PROVISIONAL_V2_EXPECTED_CANDIDATE_COUNT = 183
PROVISIONAL_V2_EXPECTED_POSITIVE_COUNT = 59
PROVISIONAL_V2_EXPECTED_NEGATIVE_COUNT = 124

SESSION_A_ID = "20260901T131308.362205Z-f9e2b1ec"
SESSION_A_FINGERPRINT = (
    "88a003966141d15858a2afec390c42e567439eb5f538b958e7aa60ee7638ab83"
)
SESSION_B_ID = "20260904T171453.458221Z-4ecb16ad"
SESSION_B_FINGERPRINT = (
    "6b6b7cff6ff4f597d0c4c9bc8ccea3544cda99e4d25bf9be1cc732ee6b9ee5ac"
)


class ProvisionalTapnessV2Error(RuntimeError):
    """A provisional artifact generation, validation, or inference failure."""


def create_provisional_tapness_v2_artifact(
    session_a_path: Path,
    session_b_path: Path,
    *,
    now_fn: Any | None = None,
) -> dict[str, Any]:
    """Fit the fixed provisional model from exact consumed Sessions A+B only."""

    session_a = load_robustness_dataset(Path(session_a_path))
    session_b = load_robustness_dataset(Path(session_b_path))
    _validate_exact_training_sources(session_a.session, session_b.session)
    try:
        reproduction = evaluate_tapness_v2_research_pair(
            Path(session_a_path), Path(session_b_path)
        )
    except TapnessV2ResearchError as error:
        raise ProvisionalTapnessV2Error(str(error)) from error
    if reproduction["experiment_19_reproduction_gate"] != {
        "applicable": True,
        "passed": True,
        "expected": {
            "A_population": "29 positive / 50 negative candidates",
            "B_population": "30 positive / 74 negative candidates",
            "A_to_B": "30/30 intended taps / 2 of 74 negative false accepts",
            "B_to_A": "29/30 intended taps / 1 of 50 negative false accepts",
            "pooled": "59/60 intended taps / 3 of 124 negative false accepts",
        },
    }:
        raise ProvisionalTapnessV2Error(
            "Experiment 19 reproduction must pass before provisional fitting."
        )
    candidates = reproduction["candidate_memberships"]
    try:
        model, threshold, selection, folds = fit_grouped_v2_research_model(
            candidates
        )
    except TapnessV2ResearchError as error:
        raise ProvisionalTapnessV2Error(str(error)) from error
    labels = [str(row["target_label"]) for row in candidates]
    positive_ids = [
        str(row["candidate_id"])
        for row in candidates
        if row["target_label"] == PROVISIONAL_V2_POSITIVE_LABEL
    ]
    negative_ids = [
        str(row["candidate_id"])
        for row in candidates
        if row["target_label"] == PROVISIONAL_V2_NEGATIVE_LABEL
    ]
    if (
        len(candidates) != PROVISIONAL_V2_EXPECTED_CANDIDATE_COUNT
        or len(positive_ids) != PROVISIONAL_V2_EXPECTED_POSITIVE_COUNT
        or len(negative_ids) != PROVISIONAL_V2_EXPECTED_NEGATIVE_COUNT
        or len(set(positive_ids + negative_ids)) != len(candidates)
    ):
        raise ProvisionalTapnessV2Error(
            "Exact A+B provisional candidate membership was not reproduced."
        )
    created = (now_fn or (lambda: datetime.now(timezone.utc)))()
    if not isinstance(created, datetime):
        raise ProvisionalTapnessV2Error("Artifact clock must return datetime.")
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    created_text = created.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    source_reports = reproduction["datasets"]
    endpoint_a = session_a.session["selected_endpoint"]
    endpoint_b = session_b.session["selected_endpoint"]
    if (
        endpoint_a["name"] != endpoint_b["name"]
        or endpoint_a["host_api"]["name"] != endpoint_b["host_api"]["name"]
    ):
        raise ProvisionalTapnessV2Error(
            "Sessions A and B do not share one endpoint identity."
        )
    artifact = {
        "artifact_schema_version": PROVISIONAL_V2_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": PROVISIONAL_V2_ARTIFACT_TYPE,
        "project_phase": "3B.3",
        "created_at_utc": created_text,
        "research_status": {
            "development_only": True,
            "provisional": True,
            "not_production": True,
            "not_external_validated": True,
            "not_live_default": True,
        },
        "target": {
            "positive_label": PROVISIONAL_V2_POSITIVE_LABEL,
            "negative_label": PROVISIONAL_V2_NEGATIVE_LABEL,
            "task": "tap_vs_non_tap",
            "spatial_inputs_used": False,
        },
        "feature_schema": tapness_v2_research_feature_schema(),
        "required_stage1_policy": stage1_policy_metadata(),
        "compatibility": {
            "sample_rate_hz": 48_000.0,
            "channel_count": 2,
            "candidate_frames": 9_600,
            "endpoint_name": endpoint_a["name"],
            "host_api_name": endpoint_a["host_api"]["name"],
        },
        "model": {
            "type": "l2_regularized_binary_logistic_regression",
            "fitting_algorithm": (
                "deterministic NumPy Newton/IRLS; intercept unregularized"
            ),
            "l2_regularization": TAPNESS_V2_RESEARCH_L2,
            "ordered_feature_names": list(TAPNESS_V2_RESEARCH_FEATURE_NAMES),
            "standardization": {
                "means": list(model.means),
                "scales": list(model.scales),
                "rule": (
                    "all-A+B transformed-feature population mean/std (ddof=0); "
                    "replace scale <=1e-12 with 1"
                ),
            },
            "coefficients": list(model.coefficients),
            "intercept": model.intercept,
            "uncalibrated_score": "sigmoid of standardized linear decision score",
            "score_is_calibrated_probability": False,
            "decision_threshold": threshold,
            "tie_rule": "score >= decision_threshold predicts TAP",
            "threshold_selection": selection,
        },
        "development_selection": {
            "grouping": (
                "whole source recordings; deterministic folds stratified by "
                "session and target"
            ),
            "preprocessing": (
                "each grouped OOF fold learns transforms' standardization and "
                "model only from its training rows"
            ),
            "operating_threshold": (
                "one grouped-OOF score vector over pooled A+B; require >=95% "
                "positive-candidate recall, minimize negative false accepts, "
                "maximize positive accepts, then choose highest threshold"
            ),
            "final_fit": (
                "after selecting the threshold, means/scales/coefficients/intercept "
                "are refit on all 183 A+B development candidates"
            ),
            "grouped_oof_folds": folds,
            "experiment_19_nested_development_metrics": reproduction[
                "development_results"
            ],
            "external_validation": False,
        },
        "source_development_datasets": [
            {
                "session_label": label,
                "session_id": source_reports[label]["session_id"],
                "dataset_fingerprint_sha256": source_reports[label][
                    "fingerprint_before"
                ],
                "verified_unchanged_during_generation": True,
                "historical_evidence_role": (
                    session_a.session.get("evidence_role")
                    if label == "A"
                    else session_b.session.get("evidence_role")
                ),
                "v2_evidence_role": "development",
            }
            for label in ("A", "B")
        ],
        "training_membership": {
            "positive_candidate_count": len(positive_ids),
            "negative_candidate_count": len(negative_ids),
            "total_candidate_count": len(candidates),
            "positive_candidate_ids": positive_ids,
            "negative_candidate_ids": negative_ids,
            "source_group_ids": sorted(
                {str(row["source_group_id"]) for row in candidates}
            ),
            "target_labels_sha256": hashlib.sha256(
                json.dumps(labels, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "r1_candidates_used": False,
        },
        "evidence_boundary": {
            "session_b_historical_external_result_preserved": True,
            "session_b_now_development_for_v2": True,
            "development_replication_r1_used_for_fitting": False,
            "untouched_session_c_required_after_final_v2_freeze": True,
        },
        "privacy": {
            "derived_metadata_only": True,
            "raw_waveform_arrays_stored": False,
            "local_processing": True,
        },
    }
    validate_provisional_tapness_v2_artifact(artifact)
    json.dumps(artifact, allow_nan=False)
    return artifact


def validate_provisional_tapness_v2_artifact(
    artifact: Mapping[str, Any],
) -> None:
    """Strictly validate one provisional v2 artifact."""

    if artifact.get("artifact_schema_version") != PROVISIONAL_V2_ARTIFACT_SCHEMA_VERSION:
        raise ProvisionalTapnessV2Error("Wrong provisional v2 artifact schema version.")
    if artifact.get("artifact_type") != PROVISIONAL_V2_ARTIFACT_TYPE:
        raise ProvisionalTapnessV2Error("Wrong provisional v2 artifact type.")
    if artifact.get("project_phase") != "3B.3":
        raise ProvisionalTapnessV2Error("Wrong provisional v2 project phase.")
    if artifact.get("research_status") != {
        "development_only": True,
        "provisional": True,
        "not_production": True,
        "not_external_validated": True,
        "not_live_default": True,
    }:
        raise ProvisionalTapnessV2Error("Provisional research status is invalid.")
    target = artifact.get("target")
    if target != {
        "positive_label": PROVISIONAL_V2_POSITIVE_LABEL,
        "negative_label": PROVISIONAL_V2_NEGATIVE_LABEL,
        "task": "tap_vs_non_tap",
        "spatial_inputs_used": False,
    }:
        raise ProvisionalTapnessV2Error("Provisional target metadata is invalid.")
    if artifact.get("feature_schema") != tapness_v2_research_feature_schema():
        raise ProvisionalTapnessV2Error("Provisional v2 feature schema mismatch.")
    if artifact.get("required_stage1_policy") != stage1_policy_metadata():
        raise ProvisionalTapnessV2Error("Provisional v2 Stage 1 policy mismatch.")
    compatibility = _mapping(artifact, "compatibility")
    if (
        compatibility.get("sample_rate_hz") != 48_000.0
        or compatibility.get("channel_count") != 2
        or compatibility.get("candidate_frames") != 9_600
        or not isinstance(compatibility.get("endpoint_name"), str)
        or not isinstance(compatibility.get("host_api_name"), str)
    ):
        raise ProvisionalTapnessV2Error("Provisional compatibility is invalid.")
    model = _mapping(artifact, "model")
    if (
        model.get("type") != "l2_regularized_binary_logistic_regression"
        or model.get("l2_regularization") != TAPNESS_V2_RESEARCH_L2
        or model.get("ordered_feature_names")
        != list(TAPNESS_V2_RESEARCH_FEATURE_NAMES)
        or model.get("score_is_calibrated_probability") is not False
        or model.get("tie_rule") != "score >= decision_threshold predicts TAP"
    ):
        raise ProvisionalTapnessV2Error("Provisional model metadata is invalid.")
    standardization = _mapping(model, "standardization")
    means = _finite_vector(standardization.get("means"), "means")
    scales = _finite_vector(standardization.get("scales"), "scales")
    coefficients = _finite_vector(model.get("coefficients"), "coefficients")
    if any(value <= 0.0 for value in scales):
        raise ProvisionalTapnessV2Error("Provisional model scales must be positive.")
    intercept = _finite_number(model.get("intercept"), "intercept")
    threshold = _finite_number(model.get("decision_threshold"), "decision threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ProvisionalTapnessV2Error("Decision threshold must be in [0, 1].")
    selection = _mapping(model, "threshold_selection")
    if (
        selection.get("minimum_positive_candidate_recall") != 0.95
        or selection.get("selected_threshold") != threshold
    ):
        raise ProvisionalTapnessV2Error("Threshold-selection metadata is inconsistent.")
    if len(means) != 2 or len(scales) != 2 or len(coefficients) != 2:
        raise ProvisionalTapnessV2Error("Provisional model vector shapes are invalid.")
    _ = intercept

    sources = artifact.get("source_development_datasets")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ProvisionalTapnessV2Error("Exactly two source datasets are required.")
    expected_sources = (("A", SESSION_A_ID, SESSION_A_FINGERPRINT), ("B", SESSION_B_ID, SESSION_B_FINGERPRINT))
    for source, (label, session_id, fingerprint) in zip(sources, expected_sources, strict=True):
        if (
            not isinstance(source, Mapping)
            or source.get("session_label") != label
            or source.get("session_id") != session_id
            or source.get("dataset_fingerprint_sha256") != fingerprint
            or source.get("verified_unchanged_during_generation") is not True
            or source.get("v2_evidence_role") != "development"
        ):
            raise ProvisionalTapnessV2Error("Provisional source identity mismatch.")
    membership = _mapping(artifact, "training_membership")
    positive_ids = membership.get("positive_candidate_ids")
    negative_ids = membership.get("negative_candidate_ids")
    if (
        membership.get("positive_candidate_count") != 59
        or membership.get("negative_candidate_count") != 124
        or membership.get("total_candidate_count") != 183
        or not isinstance(positive_ids, list)
        or not isinstance(negative_ids, list)
        or len(positive_ids) != 59
        or len(negative_ids) != 124
        or len(set(positive_ids + negative_ids)) != 183
        or membership.get("r1_candidates_used") is not False
    ):
        raise ProvisionalTapnessV2Error("Provisional training membership is invalid.")
    boundary = _mapping(artifact, "evidence_boundary")
    if (
        boundary.get("session_b_historical_external_result_preserved") is not True
        or boundary.get("session_b_now_development_for_v2") is not True
        or boundary.get("development_replication_r1_used_for_fitting") is not False
        or boundary.get("untouched_session_c_required_after_final_v2_freeze") is not True
    ):
        raise ProvisionalTapnessV2Error("Provisional evidence boundary is invalid.")
    privacy = _mapping(artifact, "privacy")
    if (
        privacy.get("derived_metadata_only") is not True
        or privacy.get("raw_waveform_arrays_stored") is not False
    ):
        raise ProvisionalTapnessV2Error("Provisional privacy metadata is invalid.")
    forbidden = {"candidate_window", "capture", "waveform", "raw_audio"}
    if _contains_forbidden_key(artifact, forbidden):
        raise ProvisionalTapnessV2Error("Provisional artifact contains waveform fields.")
    try:
        json.dumps(artifact, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProvisionalTapnessV2Error(
            "Provisional artifact is not finite JSON metadata."
        ) from error


def classify_with_provisional_tapness_v2(
    features: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the provisional research model without fitting or spatial inputs."""

    validate_provisional_tapness_v2_artifact(artifact)
    model = artifact["model"]
    fitted = ResearchLogisticModel(
        means=tuple(float(value) for value in model["standardization"]["means"]),
        scales=tuple(float(value) for value in model["standardization"]["scales"]),
        coefficients=tuple(float(value) for value in model["coefficients"]),
        intercept=float(model["intercept"]),
        l2_regularization=float(model["l2_regularization"]),
    )
    transformed = transform_tapness_v2_research_features(features)[None, :]
    score = float(fitted.score(transformed)[0])
    threshold = float(model["decision_threshold"])
    accepted = score >= threshold
    return {
        "predicted_label": (
            PROVISIONAL_V2_POSITIVE_LABEL
            if accepted
            else PROVISIONAL_V2_NEGATIVE_LABEL
        ),
        "tap_accepted": accepted,
        "uncalibrated_model_output": score,
        "decision_threshold": threshold,
        "model_margin": score - threshold,
        "tie_rule": "score >= decision_threshold predicts TAP",
        "score_is_calibrated_probability": False,
        "research_only": True,
    }


def load_provisional_tapness_v2_artifact(path: Path) -> dict[str, Any]:
    """Load and strictly validate one provisional JSON artifact."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvisionalTapnessV2Error(
            f"Could not load provisional v2 artifact: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ProvisionalTapnessV2Error("Provisional artifact root must be an object.")
    validate_provisional_tapness_v2_artifact(value)
    return value


def write_provisional_tapness_v2_artifact(
    artifact: Mapping[str, Any], path: Path
) -> Path:
    """Strictly validate and exclusively write a provisional artifact."""

    validate_provisional_tapness_v2_artifact(artifact)
    destination = Path(path)
    payload = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            output.write(payload)
    except FileExistsError as error:
        raise ProvisionalTapnessV2Error(
            f"Refusing to overwrite provisional v2 artifact: {destination}"
        ) from error
    return destination.resolve()


def provisional_tapness_v2_file_sha256(path: Path) -> str:
    """Return the exact artifact file SHA-256."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_exact_training_sources(
    session_a: Mapping[str, Any], session_b: Mapping[str, Any]
) -> None:
    if session_a.get("session_id") != SESSION_A_ID:
        raise ProvisionalTapnessV2Error("Provisional source A must be exact Session A.")
    if session_b.get("session_id") != SESSION_B_ID:
        raise ProvisionalTapnessV2Error("Provisional source B must be exact Session B.")
    if session_a.get("evidence_role") in {
        "external_validation",
        "development_replication",
    }:
        raise ProvisionalTapnessV2Error(
            "Unexpected protected evidence role on provisional source A."
        )
    if session_b.get("evidence_role") != "external_validation":
        raise ProvisionalTapnessV2Error(
            "Historical Session B identity/evidence role is not preserved."
        )
    if session_a.get("source_mode") == "guided_phase3b_v2_development_replication_collection":
        raise ProvisionalTapnessV2Error("R1 may never enter provisional fitting.")
    if session_b.get("source_mode") == "guided_phase3b_v2_development_replication_collection":
        raise ProvisionalTapnessV2Error("R1 may never enter provisional fitting.")


def _mapping(parent: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise ProvisionalTapnessV2Error(f"Provisional {name} metadata is missing.")
    return value


def _finite_vector(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ProvisionalTapnessV2Error(f"Provisional model {name} must be a list.")
    return tuple(_finite_number(item, name) for item in value)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ProvisionalTapnessV2Error(f"Provisional {name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProvisionalTapnessV2Error(
            f"Provisional {name} must be numeric."
        ) from error
    if not math.isfinite(result):
        raise ProvisionalTapnessV2Error(f"Provisional {name} must be finite.")
    return result


def _contains_forbidden_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in forbidden or _contains_forbidden_key(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False
