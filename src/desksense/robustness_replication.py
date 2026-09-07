"""Phase 3B.3 Development Replication R1 collection and offline evaluation.

R1 is prediction-independent development replication evidence, never external
validation and never Session C.  Collection binds exact provisional v2 bytes
before recording, but performs no prediction.  Evaluation never fits a model
and Stage 3 is deliberately absent from the replication criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desksense.collection import (
    DatasetCollectionError,
    record_collection_attempt,
    terminal_countdown,
)
from desksense.robustness import (
    NEGATIVE_ACTIVITIES,
    POSITIVE_STRENGTHS,
    POSITIVE_ZONES,
    RobustnessCollectionParameters,
    RobustnessError,
    _capture_metadata,
    _endpoint_metadata,
    _guided_cue_metadata,
    _save_robustness_record,
    _selected_robustness_device,
    _utc_timestamp,
    _validate_fixed_capture_domain,
    _validated_exact_capture,
    record_robustness_segment,
    replay_robustness_dataset,
)
from desksense.robustness_dataset import (
    NEGATIVE_RECORD_TYPE,
    POSITIVE_RECORD_TYPE,
    ROBUSTNESS_CHANNEL_COUNT,
    ROBUSTNESS_DTYPE,
    ROBUSTNESS_SAMPLE_RATE_HZ,
    LoadedRobustnessDataset,
    RobustnessDatasetError,
    create_robustness_dataset_session,
    load_robustness_dataset,
)
from desksense.robustness_external import (
    EXTERNAL_NEGATIVE_LABELED_SECONDS,
    EXTERNAL_NEGATIVE_TAIL_SECONDS,
    EXTERNAL_NEGATIVE_WARMUP_SECONDS,
    EXTERNAL_POSITIVE_ASSOCIATION_SECONDS,
    EXTERNAL_POSITIVE_ATTEMPTS_PER_CONDITION,
    EXTERNAL_POSITIVE_CAPTURE_SECONDS,
    EXTERNAL_POSITIVE_COUNT,
    EXTERNAL_POSITIVE_PRE_CUE_SECONDS,
    EXTERNAL_STRENGTH_DEFINITIONS,
    external_negative_plan,
    external_positive_interaction_protocol,
    external_positive_plan,
)
from desksense.streaming import StreamingTapDetector
from desksense.tapness import TapnessBaselineError, load_tapness_baseline
from desksense.tapness_v2_provisional import (
    PROVISIONAL_V2_ARTIFACT_TYPE,
    ProvisionalTapnessV2Error,
    SESSION_A_FINGERPRINT,
    SESSION_A_ID,
    SESSION_B_FINGERPRINT,
    SESSION_B_ID,
    classify_with_provisional_tapness_v2,
    load_provisional_tapness_v2_artifact,
    provisional_tapness_v2_file_sha256,
)
from desksense.tapness_v2_research import replay_tapness_v2_research_candidates


R1_EVIDENCE_ROLE = "development_replication"
R1_PROTOCOL_NAME = "phase3b3_v2_development_replication_r1"
R1_PROTOCOL_VERSION = 1
R1_PROJECT_PHASE = "3B.3 development replication"
R1_SOURCE_MODE = "guided_phase3b_v2_development_replication_collection"
R1_REPORT_SCHEMA_VERSION = 1
R1_REPORT_TYPE = "phase3b3_v2_development_replication_r1_evaluation"
R1_POSITIVE_COUNT = 30
R1_NEGATIVE_SEGMENT_COUNT = 7
R1_NEGATIVE_LABELED_SECONDS = 300.0
R1_STAGE1_MINIMUM_CANDIDATE_ATTEMPTS = 27
R1_V2_MINIMUM_ACCEPTED_ATTEMPTS = 29
R1_V2_MAXIMUM_FALSE_ACCEPTS = 2
R1_CELL_PREFERENCE_MINIMUM_ACCEPTED = 4
EXPECTED_R1_PROVISIONAL_V2_FILE_SHA256 = (
    "c69937640fc1b438550defa02e213d751348d46f4e21468f8afce5091e127157"
)
EXPECTED_R1_PROVISIONAL_V2_THRESHOLD = 0.6786616454344989
EXPECTED_R1_V1_COMPARATOR_FILE_SHA256 = (
    "2c980a0d3ae05bf8f74d9bf3b9ba4a2f35cb9e679e9273f6f55eaf4e6778dbde"
)


class DevelopmentReplicationError(RuntimeError):
    """An invalid R1 protocol, dataset, artifact, evaluation, or report."""


def r1_positive_plan() -> tuple[Any, ...]:
    """Return the exact balanced 30-attempt R1 positive plan."""

    return external_positive_plan()


def r1_negative_plan() -> tuple[Any, ...]:
    """Return the exact seven-segment, 300-second R1 negative plan."""

    return external_negative_plan()


def r1_replication_criteria() -> dict[str, Any]:
    """Return criteria frozen before any R1 collection."""

    return {
        "required": {
            "stage1_candidate_attempts_minimum": R1_STAGE1_MINIMUM_CANDIDATE_ATTEMPTS,
            "provisional_v2_accepted_attempts_minimum": R1_V2_MINIMUM_ACCEPTED_ATTEMPTS,
            "provisional_v2_false_accepts_maximum_over_300_seconds": R1_V2_MAXIMUM_FALSE_ACCEPTS,
        },
        "cell_preference": {
            "minimum_v2_accepted_per_side_strength_cell": R1_CELL_PREFERENCE_MINIMUM_ACCEPTED,
            "part_of_required_replication_pass": False,
        },
        "overall_rule": "development_replication_pass requires all three required criteria",
        "criteria_may_change_after_collection": False,
        "stage3_part_of_replication_pass": False,
        "v1_comparator_part_of_replication_pass": False,
    }


def run_guided_v2_development_replication_collection(
    audio_backend: Any,
    *,
    device_index: int,
    provisional_v2_artifact_path: Path,
    dataset_root: Path = Path("datasets"),
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
    countdown_fn: Callable[[Callable[[str], None]], None] | None = None,
    positive_capture_fn: Callable[..., np.ndarray[Any, Any]] | None = None,
    negative_capture_fn: Callable[..., np.ndarray[Any, Any]] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Collect fixed R1 evidence without running Stage 1 or either Stage 2 model."""

    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn
    clock = (lambda: datetime.now(timezone.utc)) if now_fn is None else now_fn
    positive_record = (
        record_collection_attempt if positive_capture_fn is None else positive_capture_fn
    )
    negative_record = (
        record_robustness_segment if negative_capture_fn is None else negative_capture_fn
    )
    provisional = _load_exact_r1_provisional_artifact(
        Path(provisional_v2_artifact_path)
    )
    identity = _provisional_identity(Path(provisional_v2_artifact_path), provisional)
    inventory, device = _selected_robustness_device(audio_backend, device_index)
    _validate_fixed_capture_domain(audio_backend, device)
    _validate_replication_endpoint(device, provisional)
    positives = r1_positive_plan()
    negatives = r1_negative_plan()
    metadata = _r1_session_metadata(
        inventory=inventory,
        device=device,
        positive_plan=positives,
        negative_plan=negatives,
        provisional=provisional,
        provisional_identity=identity,
    )
    try:
        session = create_robustness_dataset_session(
            Path(dataset_root),
            metadata,
            session_id=session_id,
            created_at_utc=_utc_timestamp(clock),
        )
    except (OSError, TypeError, ValueError) as error:
        raise DevelopmentReplicationError(
            f"Could not create R1 dataset: {error}"
        ) from error

    write_output(f"Phase 3B.3 Development Replication R1: {session.session_id}")
    write_output(
        f"Endpoint {device['index']}: {device['name']} ({device['host_api']['name']})"
    )
    write_output("Fixed domain: 48000 Hz, 2 channels, float32")
    write_output(
        "R1 is development replication, not external validation or Session C. "
        "The provisional artifact is identity-checked only; no predictions are "
        "displayed or used for retries/storage."
    )
    write_output(
        "Reviewed provisional artifact: "
        f"SHA-256 {EXPECTED_R1_PROVISIONAL_V2_FILE_SHA256}, "
        f"threshold {EXPECTED_R1_PROVISIONAL_V2_THRESHOLD:.16g}"
    )
    write_output(
        "Use the same RIGHT index finger and fleshy fingertip pad for BOTH zones, "
        "exactly one tap per cue, at the established lower/front locations 7-10 cm "
        "outside the corresponding edge toward the touchpad side."
    )
    write_output(
        "Keep the same established Lenovo placement and wooden-desk setup; use "
        "no fingernail, knuckle, banging, or exaggerated force."
    )
    write_output(
        "For a clear procedural error, press Ctrl+C immediately. The partial R1 "
        "session remains incomplete; never retry or delete based on model output."
    )

    for plan in positives:
        write_output("")
        write_output(
            f"Positive {plan.collection_order_index}/{len(positives)}: "
            f"{plan.zone} {plan.strength}, attempt "
            f"{plan.attempt_number_within_condition}/5"
        )
        write_output(
            f"{plan.strength.upper()} definition: "
            f"{EXTERNAL_STRENGTH_DEFINITIONS[plan.strength]}"
        )
        read_input("Press Enter when positioned for this intended tap: ")
        (terminal_countdown if countdown_fn is None else countdown_fn)(write_output)
        captured_at = _utc_timestamp(clock)
        try:
            capture = positive_record(
                audio_backend,
                device_index=int(device["index"]),
                sample_rate_hz=ROBUSTNESS_SAMPLE_RATE_HZ,
                channels=ROBUSTNESS_CHANNEL_COUNT,
                duration_seconds=EXTERNAL_POSITIVE_CAPTURE_SECONDS,
                pre_cue_duration_seconds=EXTERNAL_POSITIVE_PRE_CUE_SECONDS,
                tap_cue_fn=lambda: write_output("TAP NOW"),
                sleep_fn=sleep_fn,
            )
            capture = _validated_exact_capture(
                capture, EXTERNAL_POSITIVE_CAPTURE_SECONDS
            )
        except (DatasetCollectionError, RobustnessError) as error:
            raise DevelopmentReplicationError(str(error)) from error
        stem = (
            f"{plan.zone.casefold()}_{plan.strength}_"
            f"{plan.attempt_number_within_condition:03d}"
        )
        _save_robustness_record(
            session,
            record_id=f"{session.session_id}-{stem}",
            filename_stem=stem,
            record_type=POSITIVE_RECORD_TYPE,
            capture=capture,
            metadata={
                "session_id": session.session_id,
                "collection_order_index": plan.collection_order_index,
                "intended_zone": plan.zone,
                "intended_strength": plan.strength,
                "attempt_number_within_condition": plan.attempt_number_within_condition,
                "captured_at_utc": captured_at,
                "semantic_label": "intended_desk_tap",
                "detector_acceptance_required_for_storage": False,
                "model_prediction_used_for_retry_or_storage": False,
                "provisional_v2_prediction_performed_during_collection": False,
                **_capture_metadata(capture, device),
                "guided_cue": _r1_guided_cue_metadata(),
            },
        )
        write_output(f"Saved intended attempt: {stem}.npz")

    instructions = _negative_activity_instructions()
    for plan in negatives:
        write_output("")
        write_output(f"Negative activity: {plan.activity}")
        write_output(instructions[plan.activity])
        write_output(
            "Remain quiet for 1 s, then perform the activity naturally for "
            f"{plan.activity_duration_seconds:g} s."
        )
        read_input("Press Enter when ready to begin this segment: ")
        started_at = _utc_timestamp(clock)
        capture = negative_record(
            audio_backend,
            device_index=int(device["index"]),
            sample_rate_hz=ROBUSTNESS_SAMPLE_RATE_HZ,
            channels=ROBUSTNESS_CHANNEL_COUNT,
            activity_duration_seconds=plan.activity_duration_seconds,
            warmup_duration_seconds=EXTERNAL_NEGATIVE_WARMUP_SECONDS,
            post_activity_tail_seconds=EXTERNAL_NEGATIVE_TAIL_SECONDS,
            activity_cue_fn=lambda activity=plan.activity: write_output(
                f"BEGIN {activity.upper()}"
            ),
            activity_end_cue_fn=lambda activity=plan.activity: write_output(
                f"END {activity.upper()} - REMAIN QUIET"
            ),
            sleep_fn=sleep_fn,
        )
        ended_at = _utc_timestamp(clock)
        stored_seconds = (
            EXTERNAL_NEGATIVE_WARMUP_SECONDS
            + plan.activity_duration_seconds
            + EXTERNAL_NEGATIVE_TAIL_SECONDS
        )
        capture = _validated_exact_capture(capture, stored_seconds)
        warmup_end = round(
            ROBUSTNESS_SAMPLE_RATE_HZ * EXTERNAL_NEGATIVE_WARMUP_SECONDS
        )
        activity_end = warmup_end + round(
            ROBUSTNESS_SAMPLE_RATE_HZ * plan.activity_duration_seconds
        )
        tail_end = activity_end + round(
            ROBUSTNESS_SAMPLE_RATE_HZ * EXTERNAL_NEGATIVE_TAIL_SECONDS
        )
        stem = plan.activity
        _save_robustness_record(
            session,
            record_id=f"{session.session_id}-{stem}-001",
            filename_stem=f"{stem}_001",
            record_type=NEGATIVE_RECORD_TYPE,
            capture=capture,
            metadata={
                "session_id": session.session_id,
                "collection_order_index": plan.collection_order_index,
                "activity": plan.activity,
                "repetition_index": 1,
                "segment_started_at_utc": started_at,
                "segment_ended_at_utc": ended_at,
                "warmup_duration_seconds": EXTERNAL_NEGATIVE_WARMUP_SECONDS,
                "warmup_end_frame_index": warmup_end,
                "activity_start_frame_index": warmup_end,
                "activity_end_frame_index_exclusive": activity_end,
                "activity_duration_seconds": plan.activity_duration_seconds,
                "post_activity_tail_duration_seconds": EXTERNAL_NEGATIVE_TAIL_SECONDS,
                "post_activity_start_frame_index": activity_end,
                "post_activity_end_frame_index_exclusive": tail_end,
                "activity_cue": {
                    "begin_cue_text": f"BEGIN {plan.activity.upper()}",
                    "end_cue_text": f"END {plan.activity.upper()} - REMAIN QUIET",
                    "cue_is_audible": False,
                    "timing_caution": (
                        "Terminal cues define an engineering schedule; they are "
                        "not measured physical timestamps."
                    ),
                },
                "semantic_label": "no_intended_desk_tap",
                "model_prediction_used_for_retry_or_storage": False,
                "provisional_v2_prediction_performed_during_collection": False,
                **_capture_metadata(capture, device),
            },
        )
        write_output(f"Saved negative activity segment: {stem}_001.npz")

    write_output("")
    write_output(f"R1 collection complete: 37 records in {session.directory}")
    return {
        "status": "completed",
        "session_id": session.session_id,
        "session_directory": str(session.directory),
        "evidence_role": R1_EVIDENCE_ROLE,
        "positive_attempts": R1_POSITIVE_COUNT,
        "negative_segments": R1_NEGATIVE_SEGMENT_COUNT,
        "labeled_negative_duration_seconds": R1_NEGATIVE_LABELED_SECONDS,
        "predictions_performed_during_collection": False,
    }


def load_v2_development_replication_dataset(
    session_path: Path, *, require_complete: bool = False
) -> tuple[LoadedRobustnessDataset, dict[str, Any]]:
    """Strictly load R1 while allowing a valid planned prefix when incomplete."""

    try:
        dataset = load_robustness_dataset(Path(session_path))
    except RobustnessDatasetError as error:
        raise DevelopmentReplicationError(str(error)) from error
    completeness = _validate_r1_dataset(dataset)
    if require_complete and not completeness["development_replication_complete"]:
        raise DevelopmentReplicationError(
            "R1 collection is incomplete; evaluation requires the exact 30/7 plan."
        )
    return dataset, completeness


def evaluate_v2_development_replication(
    session_path: Path,
    *,
    provisional_v2_artifact_path: Path,
    v1_comparator_path: Path | None = None,
    detector_factory: Callable[[], Any] = StreamingTapDetector,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Evaluate complete R1 without fitting, Stage 3, or artifact mutation."""

    dataset, completeness = load_v2_development_replication_dataset(
        Path(session_path), require_complete=True
    )
    provisional = _load_exact_r1_provisional_artifact(
        Path(provisional_v2_artifact_path)
    )
    identity = _provisional_identity(Path(provisional_v2_artifact_path), provisional)
    bound = dataset.session["provisional_v2_pipeline"]["artifact"]
    if bound != identity:
        raise DevelopmentReplicationError(
            "R1 is not bound to the supplied provisional v2 artifact bytes."
        )
    if dataset.session["provisional_v2_pipeline"]["stage1_policy"] != provisional[
        "required_stage1_policy"
    ]:
        raise DevelopmentReplicationError("R1 Stage 1 policy binding is invalid.")

    integrity_before = {
        "dataset_fingerprint": _dataset_fingerprint(Path(session_path)),
        "provisional_artifact_sha256": _file_sha256(
            Path(provisional_v2_artifact_path)
        ),
        "v1_comparator_sha256": (
            _file_sha256(Path(v1_comparator_path))
            if v1_comparator_path is not None
            else None
        ),
    }
    candidate_replay = replay_tapness_v2_research_candidates(
        Path(session_path),
        session_label="R1",
        detector_factory=detector_factory,
    )
    stage1_replay = replay_robustness_dataset(
        Path(session_path),
        detector_factory=detector_factory,
        now_fn=now_fn,
    )
    population_cross_check = _cross_check_r1_replay_populations(
        candidate_replay, stage1_replay
    )
    rows = candidate_replay["candidate_memberships"]
    predictions = [
        classify_with_provisional_tapness_v2(row["features"], provisional)
        for row in rows
    ]
    positive = _r1_positive_metrics(
        dataset, rows, predictions, stage1_replay["positive"]["attempts"]
    )
    negative = _r1_negative_metrics(
        dataset, rows, predictions, stage1_replay["negative"]["segments"]
    )
    criteria = _r1_criteria_result(positive, negative)

    comparator: dict[str, Any] | None = None
    if v1_comparator_path is not None:
        _load_exact_r1_v1_comparator(Path(v1_comparator_path))
        comparison = replay_robustness_dataset(
            Path(session_path),
            tapness_baseline_path=Path(v1_comparator_path),
            detector_factory=detector_factory,
            now_fn=now_fn,
        )
        comparator = {
            "status": "historical_comparator_only",
            "affects_v2_replication_pass": False,
            "artifact_sha256": integrity_before["v1_comparator_sha256"],
            "positive_stage2": comparison["positive"]["summary"]["stage2_tapness"],
            "positive_by_zone": comparison["positive"]["summary"]["by_zone"],
            "positive_by_strength": comparison["positive"]["summary"]["by_strength"],
            "negative_stage2": comparison["negative"]["summary"]["stage2_tapness"],
            "negative_per_activity": comparison["negative"]["summary"]["per_activity"],
        }

    integrity_after = {
        "dataset_fingerprint": _dataset_fingerprint(Path(session_path)),
        "provisional_artifact_sha256": _file_sha256(
            Path(provisional_v2_artifact_path)
        ),
        "v1_comparator_sha256": (
            _file_sha256(Path(v1_comparator_path))
            if v1_comparator_path is not None
            else None
        ),
    }
    if integrity_after != integrity_before:
        raise DevelopmentReplicationError(
            "R1 dataset or a supplied artifact changed during evaluation."
        )
    generated = (now_fn or (lambda: datetime.now(timezone.utc)))()
    if not isinstance(generated, datetime):
        raise DevelopmentReplicationError("R1 report clock must return datetime.")
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    report = {
        "report_schema_version": R1_REPORT_SCHEMA_VERSION,
        "report_type": R1_REPORT_TYPE,
        "project_phase": R1_PROJECT_PHASE,
        "generated_at_utc": generated.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "evidence_role": R1_EVIDENCE_ROLE,
        "development_only": True,
        "external_validation": False,
        "session_c": False,
        "session_id": dataset.session["session_id"],
        "collection_completeness": completeness,
        "protocol": dataset.session["development_replication_protocol"],
        "physical_positive_interaction_protocol": dataset.session[
            "physical_positive_interaction_protocol"
        ],
        "collection_context": {
            "selected_endpoint": dataset.session["selected_endpoint"],
            "capture_domain": dataset.session["capture_domain"],
            "negative_denominator": (
                "exactly 300 labeled seconds across seven separately recorded "
                "segments; each warm-up and tail is excluded"
            ),
        },
        "provisional_v2_artifact": identity,
        "model_fitting_performed": False,
        "replay_population_cross_check": population_cross_check,
        "stage3_evaluation": {
            "status": "not_performed",
            "affects_replication_pass": False,
        },
        "positive_metrics": positive,
        "negative_metrics": negative,
        "development_replication_criteria": criteria,
        "v1_comparator": comparator,
        "integrity": {
            "before": integrity_before,
            "after": integrity_after,
            "verified_unchanged": True,
        },
        "evidence_boundary": {
            "r1_not_used_to_fit_provisional_artifact": True,
            "r1_result_must_be_preserved": True,
            "r1_may_become_later_development_evidence": True,
            "session_c_reserved_for_final_external_validation": True,
        },
        "privacy": {
            "raw_waveforms_in_report": False,
            "local_offline_processing": True,
        },
    }
    json.dumps(report, allow_nan=False)
    return report


def write_v2_development_replication_report(
    report: Mapping[str, Any], path: Path
) -> Path:
    """Exclusively write one finite waveform-free R1 JSON report."""

    if report.get("report_type") != R1_REPORT_TYPE:
        raise DevelopmentReplicationError("Unexpected R1 report type.")
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise DevelopmentReplicationError("R1 report path must end in .json.")
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            output.write(payload)
    except FileExistsError as error:
        raise DevelopmentReplicationError(
            f"Refusing to overwrite an R1 report: {destination}"
        ) from error
    return destination.resolve()


def _r1_session_metadata(
    *,
    inventory: Mapping[str, Any],
    device: Mapping[str, Any],
    positive_plan: Sequence[Any],
    negative_plan: Sequence[Any],
    provisional: Mapping[str, Any],
    provisional_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "project_phase": R1_PROJECT_PHASE,
        "source_mode": R1_SOURCE_MODE,
        "evidence_role": R1_EVIDENCE_ROLE,
        "not_external_validation": True,
        "not_session_c": True,
        "no_fitting_during_collection_or_evaluation": True,
        "not_used_to_fit_provisional_artifact": True,
        "development_replication_protocol": _r1_protocol_metadata(),
        "physical_positive_interaction_protocol": (
            external_positive_interaction_protocol()
        ),
        "audio_backend": dict(inventory["backend"]),
        "selected_endpoint": _endpoint_metadata(device),
        "capture_domain": {
            "sample_rate_hz": ROBUSTNESS_SAMPLE_RATE_HZ,
            "channel_count": ROBUSTNESS_CHANNEL_COUNT,
            "dtype": ROBUSTNESS_DTYPE,
        },
        "positive_design": {
            "zones": list(POSITIVE_ZONES),
            "strengths": list(POSITIVE_STRENGTHS),
            "attempts_per_condition": EXTERNAL_POSITIVE_ATTEMPTS_PER_CONDITION,
            "requested_attempt_count": R1_POSITIVE_COUNT,
            "capture_duration_seconds": EXTERNAL_POSITIVE_CAPTURE_SECONDS,
            "pre_cue_seconds": EXTERNAL_POSITIVE_PRE_CUE_SECONDS,
            "association_duration_seconds": EXTERNAL_POSITIVE_ASSOCIATION_SECONDS,
            "ordering": "repetition-major round robin over strength then zone",
            "plan": [asdict(item) for item in positive_plan],
            "detector_acceptance_required_for_storage": False,
        },
        "negative_design": {
            "activities": list(NEGATIVE_ACTIVITIES),
            "segments_per_activity": 1,
            "requested_segment_count": R1_NEGATIVE_SEGMENT_COUNT,
            "warmup_duration_seconds": EXTERNAL_NEGATIVE_WARMUP_SECONDS,
            "post_activity_tail_duration_seconds": EXTERNAL_NEGATIVE_TAIL_SECONDS,
            "labeled_activity_duration_seconds": R1_NEGATIVE_LABELED_SECONDS,
            "ordering": "one predeclared segment per activity",
            "plan": [asdict(item) for item in negative_plan],
        },
        "provisional_v2_pipeline": {
            "stage1_policy": provisional["required_stage1_policy"],
            "artifact": dict(provisional_identity),
            "artifact_loaded_for_identity_only_during_collection": True,
            "predictions_performed_during_collection": False,
            "artifact_may_be_modified_by_collection": False,
        },
        "privacy": {
            "waveforms_retained_locally": True,
            "network_upload_performed": False,
        },
    }


def _r1_protocol_metadata() -> dict[str, Any]:
    return {
        "name": R1_PROTOCOL_NAME,
        "version": R1_PROTOCOL_VERSION,
        "predeclared_before_collection": True,
        "evidence_role": R1_EVIDENCE_ROLE,
        "collection_independent_of_model_predictions": True,
        "labeled_negative_denominator_seconds": 300.0,
        "negative_segment_count": 7,
        "negative_recording_structure": (
            "seven separately recorded and replayed segments; each has its own "
            "excluded 1.0 s warm-up and 0.25 s completion tail"
        ),
        "replication_criteria": r1_replication_criteria(),
        "fitting_allowed_during_evaluation": False,
    }


def _r1_guided_cue_metadata() -> dict[str, Any]:
    return _guided_cue_metadata(
        RobustnessCollectionParameters(
            attempts_per_condition=EXTERNAL_POSITIVE_ATTEMPTS_PER_CONDITION,
            positive_capture_seconds=EXTERNAL_POSITIVE_CAPTURE_SECONDS,
            pre_cue_seconds=EXTERNAL_POSITIVE_PRE_CUE_SECONDS,
            positive_association_seconds=EXTERNAL_POSITIVE_ASSOCIATION_SECONDS,
        )
    )


def _validate_r1_dataset(dataset: LoadedRobustnessDataset) -> dict[str, Any]:
    session = dataset.session
    required = {
        "project_phase": R1_PROJECT_PHASE,
        "source_mode": R1_SOURCE_MODE,
        "evidence_role": R1_EVIDENCE_ROLE,
        "not_external_validation": True,
        "not_session_c": True,
        "no_fitting_during_collection_or_evaluation": True,
        "not_used_to_fit_provisional_artifact": True,
    }
    for key, expected in required.items():
        if session.get(key) != expected:
            raise DevelopmentReplicationError(f"R1 session {key} is invalid.")
    if session.get("development_replication_protocol") != _r1_protocol_metadata():
        raise DevelopmentReplicationError("R1 protocol metadata does not match v1.")
    if session.get("physical_positive_interaction_protocol") != (
        external_positive_interaction_protocol()
    ):
        raise DevelopmentReplicationError("R1 physical protocol does not match Session B.")
    _validate_r1_design(session)
    pipeline = session.get("provisional_v2_pipeline")
    if not isinstance(pipeline, Mapping):
        raise DevelopmentReplicationError("R1 lacks provisional pipeline metadata.")
    artifact = pipeline.get("artifact")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("artifact_type") != PROVISIONAL_V2_ARTIFACT_TYPE
        or not _valid_sha256(artifact.get("file_sha256"))
        or pipeline.get("predictions_performed_during_collection") is not False
        or pipeline.get("artifact_may_be_modified_by_collection") is not False
    ):
        raise DevelopmentReplicationError("R1 provisional artifact binding is invalid.")

    expected_positive = [asdict(item) for item in r1_positive_plan()]
    observed_positive = []
    for record in dataset.positive_records:
        metadata = record.metadata
        if (
            metadata.get("model_prediction_used_for_retry_or_storage") is not False
            or metadata.get("provisional_v2_prediction_performed_during_collection")
            is not False
            or metadata.get("detector_acceptance_required_for_storage") is not False
        ):
            raise DevelopmentReplicationError(
                "R1 positive record is not prediction-independent."
            )
        observed_positive.append(
            {
                "collection_order_index": metadata["collection_order_index"],
                "zone": metadata["intended_zone"],
                "strength": metadata["intended_strength"],
                "attempt_number_within_condition": metadata[
                    "attempt_number_within_condition"
                ],
            }
        )
        association = metadata["guided_cue"]["intended_event_association"]
        if (
            record.capture.shape != (96_000, 2)
            or metadata["guided_cue"]["intended_cue_offset_frames"] != 43_200
            or association["start_frame_index_inclusive"] != 43_200
            or association["end_frame_index_exclusive"] != 79_200
        ):
            raise DevelopmentReplicationError("R1 positive boundaries are invalid.")
    if observed_positive != expected_positive[: len(observed_positive)]:
        raise DevelopmentReplicationError("R1 positive records violate the planned prefix.")

    expected_negative = [asdict(item) for item in r1_negative_plan()]
    observed_negative = []
    for index, record in enumerate(dataset.negative_records):
        metadata = record.metadata
        if len(observed_positive) != R1_POSITIVE_COUNT:
            raise DevelopmentReplicationError(
                "R1 negative records cannot precede all 30 positives."
            )
        if index >= len(expected_negative):
            raise DevelopmentReplicationError("R1 has excess negative records.")
        if (
            metadata.get("model_prediction_used_for_retry_or_storage") is not False
            or metadata.get("provisional_v2_prediction_performed_during_collection")
            is not False
        ):
            raise DevelopmentReplicationError(
                "R1 negative record is not prediction-independent."
            )
        plan = r1_negative_plan()[index]
        observed_negative.append(
            {
                "collection_order_index": metadata["collection_order_index"],
                "activity": metadata["activity"],
                "repetition_index": metadata["repetition_index"],
                "activity_duration_seconds": metadata["activity_duration_seconds"],
            }
        )
        activity_start = 48_000
        activity_end = activity_start + round(
            ROBUSTNESS_SAMPLE_RATE_HZ * plan.activity_duration_seconds
        )
        tail_end = activity_end + 12_000
        if (
            metadata["warmup_end_frame_index"] != activity_start
            or metadata["activity_start_frame_index"] != activity_start
            or metadata["activity_end_frame_index_exclusive"] != activity_end
            or metadata["post_activity_start_frame_index"] != activity_end
            or metadata["post_activity_end_frame_index_exclusive"] != tail_end
            or record.capture.shape != (tail_end, 2)
        ):
            raise DevelopmentReplicationError("R1 negative boundaries are invalid.")
    if observed_negative != expected_negative[: len(observed_negative)]:
        raise DevelopmentReplicationError("R1 negative records violate the planned prefix.")
    positive_complete = observed_positive == expected_positive
    negative_complete = observed_negative == expected_negative
    return {
        "expected_positive_attempts": 30,
        "observed_positive_attempts": len(observed_positive),
        "positive_plan_complete": positive_complete,
        "expected_negative_segments": 7,
        "observed_negative_segments": len(observed_negative),
        "negative_plan_complete": negative_complete,
        "expected_labeled_negative_seconds": 300.0,
        "development_replication_complete": positive_complete and negative_complete,
    }


def _validate_r1_design(session: Mapping[str, Any]) -> None:
    positive = session.get("positive_design")
    negative = session.get("negative_design")
    if not isinstance(positive, Mapping) or not isinstance(negative, Mapping):
        raise DevelopmentReplicationError("R1 design metadata is missing.")
    if (
        positive.get("plan") != [asdict(item) for item in r1_positive_plan()]
        or positive.get("requested_attempt_count") != 30
        or positive.get("attempts_per_condition") != 5
        or positive.get("capture_duration_seconds") != 2.0
        or positive.get("pre_cue_seconds") != 0.9
        or positive.get("association_duration_seconds") != 0.75
    ):
        raise DevelopmentReplicationError("R1 positive design is invalid.")
    if (
        negative.get("plan") != [asdict(item) for item in r1_negative_plan()]
        or negative.get("requested_segment_count") != 7
        or negative.get("segments_per_activity") != 1
        or negative.get("warmup_duration_seconds") != 1.0
        or negative.get("post_activity_tail_duration_seconds") != 0.25
        or negative.get("labeled_activity_duration_seconds") != 300.0
    ):
        raise DevelopmentReplicationError("R1 negative design is invalid.")


def _r1_positive_metrics(
    dataset: LoadedRobustnessDataset,
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    stage1_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    positive_pairs = [
        (row, prediction)
        for row, prediction in zip(rows, predictions, strict=True)
        if row["target"] == 1
    ]
    candidate_records = {
        str(attempt["record_id"])
        for attempt in stage1_attempts
        if attempt["stage1_candidate_started"]
    }
    accepted_records = {
        str(row["source_record_id"])
        for row, prediction in positive_pairs
        if prediction["tap_accepted"]
    }
    records = [record.metadata for record in dataset.positive_records]

    def breakdown(field: str) -> dict[str, Any]:
        values = sorted({str(record[field]) for record in records})
        output = {}
        for value in values:
            identifiers = {
                str(record["record_id"])
                for record in records
                if str(record[field]) == value
            }
            output[value] = {
                "intended_attempts": len(identifiers),
                "stage1_candidate_attempts": len(identifiers & candidate_records),
                "provisional_v2_accepted_attempts": len(
                    identifiers & accepted_records
                ),
            }
        return output

    cells = {}
    for zone in POSITIVE_ZONES:
        for strength in POSITIVE_STRENGTHS:
            identifiers = {
                str(record["record_id"])
                for record in records
                if record["intended_zone"] == zone
                and record["intended_strength"] == strength
            }
            cells[f"{zone}-{strength}"] = {
                "intended_attempts": len(identifiers),
                "stage1_candidate_attempts": len(identifiers & candidate_records),
                "provisional_v2_accepted_attempts": len(
                    identifiers & accepted_records
                ),
            }
    return {
        "intended_attempts": 30,
        "stage1_candidate_attempts": len(candidate_records),
        "stage1_candidate_count": sum(
            int(attempt["associated_candidate_start_count"])
            for attempt in stage1_attempts
        ),
        "associated_completed_detector_detections": sum(
            int(attempt["associated_completed_detection_count"])
            for attempt in stage1_attempts
        ),
        "associated_completed_detector_rejections": sum(
            int(attempt["associated_completed_rejection_count"])
            for attempt in stage1_attempts
        ),
        "stage1_routes": dict(
            sorted(
                Counter(
                    str(start["candidate_start_route"])
                    for attempt in stage1_attempts
                    for start in attempt["candidate_starts"]
                    if start["interval_relation"] == "associated"
                ).items()
            )
        ),
        "provisional_v2_accepted_attempts": len(accepted_records),
        "provisional_v2_rejected_attempts": 30 - len(accepted_records),
        "by_zone": breakdown("intended_zone"),
        "by_strength": breakdown("intended_strength"),
        "by_zone_and_strength": cells,
        "score_distribution": _score_distribution(
            [float(prediction["uncalibrated_model_output"]) for _, prediction in positive_pairs]
        ),
        "margin_distribution": _score_distribution(
            [float(prediction["model_margin"]) for _, prediction in positive_pairs]
        ),
        "candidate_results": [
            {
                "candidate_id": row["candidate_id"],
                "source_record_id": row["source_record_id"],
                "intended_zone": row["intended_zone"],
                "intended_strength": row["intended_strength"],
                "stage1_route": row["stage1_route"],
                "features": row["features"],
                "uncalibrated_model_output": prediction[
                    "uncalibrated_model_output"
                ],
                "model_margin": prediction["model_margin"],
                "tap_accepted": prediction["tap_accepted"],
            }
            for row, prediction in positive_pairs
        ],
    }


def _r1_negative_metrics(
    dataset: LoadedRobustnessDataset,
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    stage1_segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    negative_pairs = [
        (row, prediction)
        for row, prediction in zip(rows, predictions, strict=True)
        if row["target"] == 0
    ]
    seconds = Counter(
        {
            str(record.metadata["activity"]): float(
                record.metadata["activity_duration_seconds"]
            )
            for record in dataset.negative_records
        }
    )
    per_activity = {}
    stage1_by_activity = {
        str(segment["activity"]): segment for segment in stage1_segments
    }
    for activity in NEGATIVE_ACTIVITIES:
        pairs = [
            pair for pair in negative_pairs if pair[0]["negative_activity"] == activity
        ]
        stage1_segment = stage1_by_activity[activity]
        stage1_count = int(stage1_segment["activity_candidate_start_count"])
        false_accepts = sum(
            bool(prediction["tap_accepted"]) for _, prediction in pairs
        )
        per_activity[activity] = {
            "labeled_seconds": float(seconds[activity]),
            "stage1_candidate_count": stage1_count,
            "stage1_candidates_per_minute": (
                stage1_count / (float(seconds[activity]) / 60.0)
            ),
            "provisional_v2_false_accepts": false_accepts,
            "false_accepts_per_minute": (
                false_accepts / (float(seconds[activity]) / 60.0)
            ),
            "stage1_routes": dict(
                sorted(Counter(str(row["stage1_route"]) for row, _ in pairs).items())
            ),
        }
    false_accept_count = sum(
        bool(prediction["tap_accepted"]) for _, prediction in negative_pairs
    )
    labeled_seconds = float(sum(seconds.values()))
    labeled_minutes = labeled_seconds / 60.0
    activity_starts = sum(
        int(segment["activity_candidate_start_count"])
        for segment in stage1_segments
    )
    activity_routes = Counter(
        str(start["candidate_start_route"])
        for segment in stage1_segments
        for start in segment["candidate_starts"]
        if start["interval_relation"] == "activity"
    )
    return {
        "labeled_negative_seconds": labeled_seconds,
        "stage1_candidate_count": activity_starts,
        "stage1_candidates_per_minute": activity_starts / labeled_minutes,
        "stage1_routes": dict(sorted(activity_routes.items())),
        "activity_completed_detector_detections": sum(
            int(segment["activity_completed_detection_count"])
            for segment in stage1_segments
        ),
        "activity_completed_detector_rejections": sum(
            int(segment["activity_completed_rejection_count"])
            for segment in stage1_segments
        ),
        "provisional_v2_false_accepts": false_accept_count,
        "false_accepts_per_minute": false_accept_count / labeled_minutes,
        "per_activity": per_activity,
        "score_distribution": _score_distribution(
            [float(prediction["uncalibrated_model_output"]) for _, prediction in negative_pairs]
        ),
        "margin_distribution": _score_distribution(
            [float(prediction["model_margin"]) for _, prediction in negative_pairs]
        ),
        "false_accept_results": [
            {
                "candidate_id": row["candidate_id"],
                "source_record_id": row["source_record_id"],
                "activity": row["negative_activity"],
                "stage1_route": row["stage1_route"],
                "features": row["features"],
                "uncalibrated_model_output": prediction[
                    "uncalibrated_model_output"
                ],
                "model_margin": prediction["model_margin"],
            }
            for row, prediction in negative_pairs
            if prediction["tap_accepted"]
        ],
    }


def _r1_criteria_result(
    positive: Mapping[str, Any], negative: Mapping[str, Any]
) -> dict[str, Any]:
    stage1_pass = (
        positive["stage1_candidate_attempts"]
        >= R1_STAGE1_MINIMUM_CANDIDATE_ATTEMPTS
    )
    v2_survival_pass = (
        positive["provisional_v2_accepted_attempts"]
        >= R1_V2_MINIMUM_ACCEPTED_ATTEMPTS
    )
    false_accept_pass = (
        negative["provisional_v2_false_accepts"]
        <= R1_V2_MAXIMUM_FALSE_ACCEPTS
    )
    cell_results = {
        name: values["provisional_v2_accepted_attempts"]
        >= R1_CELL_PREFERENCE_MINIMUM_ACCEPTED
        for name, values in positive["by_zone_and_strength"].items()
    }
    return {
        "predeclared": r1_replication_criteria(),
        "required_results": {
            "stage1_candidate_attempts": positive["stage1_candidate_attempts"],
            "stage1_pass": stage1_pass,
            "provisional_v2_accepted_attempts": positive[
                "provisional_v2_accepted_attempts"
            ],
            "provisional_v2_survival_pass": v2_survival_pass,
            "provisional_v2_false_accepts_over_300_seconds": negative[
                "provisional_v2_false_accepts"
            ],
            "provisional_v2_false_accept_pass": false_accept_pass,
        },
        "cell_preference_results": cell_results,
        "cell_preference_pass": all(cell_results.values()),
        "cell_preference_affects_required_pass": False,
        "development_replication_pass": (
            stage1_pass and v2_survival_pass and false_accept_pass
        ),
    }


def _cross_check_r1_replay_populations(
    candidate_replay: Mapping[str, Any], stage1_replay: Mapping[str, Any]
) -> dict[str, Any]:
    """Require count and canonical event-identity agreement between replays."""

    population = candidate_replay["candidate_population"]
    positive_attempts = stage1_replay["positive"]["attempts"]
    negative_segments = stage1_replay["negative"]["segments"]
    expected = {
        "positive_detected_candidates": sum(
            int(item["associated_completed_detection_count"])
            for item in positive_attempts
        ),
        "negative_detected_candidates": sum(
            int(item["activity_completed_detection_count"])
            for item in negative_segments
        ),
        "associated_candidate_starts": sum(
            int(item["associated_candidate_start_count"])
            for item in positive_attempts
        )
        + sum(
            int(item["activity_candidate_start_count"])
            for item in negative_segments
        ),
    }
    observed = {
        "positive_detected_candidates": int(population["positive_candidate_count"]),
        "negative_detected_candidates": int(population["negative_candidate_count"]),
        "associated_candidate_starts": int(
            population["associated_candidate_start_count"]
        ),
    }
    if observed != expected:
        raise DevelopmentReplicationError(
            "Independent R1 replay paths disagree on candidate populations: "
            f"expected={expected!r}, observed={observed!r}."
        )

    generic_identities = {
        "positive_detected_candidates": sorted(
            (
                str(attempt["record_id"]),
                int(event["onset_frame_index"]),
            )
            for attempt in positive_attempts
            for event in attempt["completed_events"]
            if event["interval_relation"] == "associated"
            and event["status"] == "detected"
        ),
        "negative_detected_candidates": sorted(
            (
                str(segment["record_id"]),
                int(event["onset_frame_index"]),
            )
            for segment in negative_segments
            for event in segment["completed_events"]
            if event["interval_relation"] == "activity"
            and event["status"] == "detected"
        ),
        "positive_candidate_starts": sorted(
            (
                str(attempt["record_id"]),
                int(start["onset_frame_index"]),
                str(start["candidate_start_route"]),
            )
            for attempt in positive_attempts
            for start in attempt["candidate_starts"]
            if start["interval_relation"] == "associated"
        ),
        "negative_candidate_starts": sorted(
            (
                str(segment["record_id"]),
                int(start["onset_frame_index"]),
                str(start["candidate_start_route"]),
            )
            for segment in negative_segments
            for start in segment["candidate_starts"]
            if start["interval_relation"] == "activity"
        ),
    }
    memberships = candidate_replay["candidate_memberships"]
    v2_identities = {
        "positive_detected_candidates": sorted(
            (str(item["source_record_id"]), int(item["onset_frame_index"]))
            for item in memberships
            if int(item["target"]) == 1
        ),
        "negative_detected_candidates": sorted(
            (str(item["source_record_id"]), int(item["onset_frame_index"]))
            for item in memberships
            if int(item["target"]) == 0
        ),
        "positive_candidate_starts": _canonical_start_identities(
            population["positive_associated_candidate_start_identities"]
        ),
        "negative_candidate_starts": _canonical_start_identities(
            population["negative_activity_candidate_start_identities"]
        ),
    }
    for name, generic_values in generic_identities.items():
        v2_values = v2_identities[name]
        if generic_values != v2_values:
            missing = list((Counter(generic_values) - Counter(v2_values)).elements())
            extra = list((Counter(v2_values) - Counter(generic_values)).elements())
            raise DevelopmentReplicationError(
                "Independent R1 replay paths disagree on canonical identities "
                f"for {name}: missing_from_v2={missing!r}, extra_in_v2={extra!r}."
            )

    evidence = {
        name: _identity_evidence(values)
        for name, values in generic_identities.items()
    }
    return {
        "status": "passed",
        "count_match": True,
        "identity_match": True,
        "canonicalization": (
            "lexicographic sort by source_record_id, onset_frame_index, and "
            "candidate_start_route where applicable"
        ),
        "stage1_replay": expected,
        "v2_replay": observed,
        "populations": evidence,
    }


def _canonical_start_identities(
    values: Sequence[Mapping[str, Any]],
) -> list[tuple[str, int, str]]:
    return sorted(
        (
            str(item["source_record_id"]),
            int(item["onset_frame_index"]),
            str(item["candidate_start_route"]),
        )
        for item in values
    )


def _identity_evidence(values: Sequence[tuple[Any, ...]]) -> dict[str, Any]:
    canonical_json = json.dumps(
        [list(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "count": len(values),
        "identity_sha256": hashlib.sha256(canonical_json).hexdigest(),
        "canonical_identities": [list(value) for value in values],
    }


def _score_distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    if not np.all(np.isfinite(array)):
        raise DevelopmentReplicationError("R1 score distribution is non-finite.")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _provisional_identity(
    path: Path, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    sources = artifact["source_development_datasets"]
    return {
        "artifact_type": artifact["artifact_type"],
        "artifact_schema_version": artifact["artifact_schema_version"],
        "file_sha256": provisional_tapness_v2_file_sha256(path),
        "created_at_utc": artifact["created_at_utc"],
        "feature_schema_version": artifact["feature_schema"]["version"],
        "ordered_feature_names": artifact["feature_schema"]["ordered_feature_names"],
        "decision_threshold": artifact["model"]["decision_threshold"],
        "source_session_ids": [source["session_id"] for source in sources],
        "source_dataset_fingerprints": [
            source["dataset_fingerprint_sha256"] for source in sources
        ],
        "research_only": True,
    }


def _load_exact_r1_provisional_artifact(path: Path) -> dict[str, Any]:
    """Require the byte-exact reviewed provisional artifact for R1."""

    artifact_path = Path(path)
    actual_hash = _file_sha256(artifact_path)
    if actual_hash != EXPECTED_R1_PROVISIONAL_V2_FILE_SHA256:
        raise DevelopmentReplicationError(
            "R1 requires the exact reviewed provisional v2 artifact bytes; "
            f"expected {EXPECTED_R1_PROVISIONAL_V2_FILE_SHA256}, got {actual_hash}."
        )
    try:
        artifact = load_provisional_tapness_v2_artifact(artifact_path)
    except ProvisionalTapnessV2Error as error:
        raise DevelopmentReplicationError(str(error)) from error
    sources = artifact["source_development_datasets"]
    expected_sources = [
        (SESSION_A_ID, SESSION_A_FINGERPRINT),
        (SESSION_B_ID, SESSION_B_FINGERPRINT),
    ]
    actual_sources = [
        (item["session_id"], item["dataset_fingerprint_sha256"])
        for item in sources
    ]
    if (
        artifact["model"]["decision_threshold"]
        != EXPECTED_R1_PROVISIONAL_V2_THRESHOLD
        or actual_sources != expected_sources
    ):
        raise DevelopmentReplicationError(
            "R1 provisional artifact semantic identity is invalid."
        )
    return artifact


def _load_exact_r1_v1_comparator(path: Path) -> dict[str, Any]:
    """Require the byte-exact frozen v1 artifact when used as comparator."""

    artifact_path = Path(path)
    actual_hash = _file_sha256(artifact_path)
    if actual_hash != EXPECTED_R1_V1_COMPARATOR_FILE_SHA256:
        raise DevelopmentReplicationError(
            "R1 v1 comparator must be the exact frozen artifact; "
            f"expected {EXPECTED_R1_V1_COMPARATOR_FILE_SHA256}, got {actual_hash}."
        )
    try:
        return load_tapness_baseline(artifact_path)
    except TapnessBaselineError as error:
        raise DevelopmentReplicationError(
            f"Could not load the exact R1 v1 comparator: {error}"
        ) from error


def _validate_replication_endpoint(
    device: Mapping[str, Any], provisional: Mapping[str, Any]
) -> None:
    compatibility = provisional["compatibility"]
    if (
        device["name"] != compatibility["endpoint_name"]
        or device["host_api"]["name"] != compatibility["host_api_name"]
    ):
        raise DevelopmentReplicationError(
            "Selected endpoint identity does not match the provisional artifact."
        )


def _dataset_fingerprint(path: Path) -> str:
    from desksense.tapness import robustness_dataset_fingerprint

    return str(robustness_dataset_fingerprint(path)["sha256"])


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _negative_activity_instructions() -> dict[str, str]:
    return {
        "quiet": "Remain still and silent; do not intentionally interact with the desk.",
        "typing": "Type naturally at an ordinary pace.",
        "speech": "Speak continuously at a normal conversational level.",
        "trackpad": "Use normal pointer movement, scrolling, and clicks.",
        "hand_movement": (
            "Reposition, rest, and lift hands naturally around the keyboard, "
            "palm rest, and desk without intended desk taps."
        ),
        "laptop_movement": (
            "Naturally make small laptop repositioning/handling movements; "
            "restore the established setup afterward."
        ),
        "desk_object_interaction": (
            "Naturally handle and place ordinary desk objects. Ordinary impacts "
            "are valid hard negatives; do not imitate the tap-test gesture."
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the isolated research-only R1 command parser."""

    parser = argparse.ArgumentParser(
        prog="python -m desksense.robustness_replication",
        description=(
            "Research-only Phase 3B.3 Development Replication R1 harness; "
            "not production, external validation, or Session C."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser(
        "collect", help="collect prediction-independent local R1 evidence"
    )
    collect.add_argument("--device", type=int, required=True)
    collect.add_argument(
        "--provisional-artifact", type=Path, required=True
    )
    collect.add_argument("--dataset-root", type=Path, default=Path("datasets"))

    evaluate = commands.add_parser(
        "evaluate", help="evaluate a complete R1 dataset fully offline"
    )
    evaluate.add_argument("session", type=Path)
    evaluate.add_argument(
        "--provisional-artifact", type=Path, required=True
    )
    evaluate.add_argument("--save-report", type=Path, required=True)
    evaluate.add_argument("--v1-comparator", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the isolated R1 collector or offline evaluator."""

    args = build_argument_parser().parse_args(argv)
    try:
        if args.command == "collect":
            artifact = _load_exact_r1_provisional_artifact(
                args.provisional_artifact
            )
            print("Evidence role: DEVELOPMENT REPLICATION ONLY")
            print("This is NOT Session C and NOT external validation.")
            print(
                "Reviewed provisional artifact SHA-256: "
                f"{EXPECTED_R1_PROVISIONAL_V2_FILE_SHA256}"
            )
            print(
                "Fixed provisional threshold: "
                f"{artifact['model']['decision_threshold']:.16g}"
            )
            result = run_guided_v2_development_replication_collection(
                _load_collect_audio_backend(),
                device_index=args.device,
                provisional_v2_artifact_path=args.provisional_artifact,
                dataset_root=args.dataset_root,
            )
            print(f"R1 session ID: {result['session_id']}")
            print(f"R1 session path: {result['session_directory']}")
            return 0

        report = evaluate_v2_development_replication(
            args.session,
            provisional_v2_artifact_path=args.provisional_artifact,
            v1_comparator_path=args.v1_comparator,
        )
        saved = write_v2_development_replication_report(
            report, args.save_report
        )
        _print_r1_evaluation_summary(report)
        print(f"R1 report saved to: {saved}")
        return 0
    except KeyboardInterrupt:
        print("\nR1 operation interrupted; no retry decision was inferred.", file=sys.stderr)
        return 130
    except (
        DevelopmentReplicationError,
        ProvisionalTapnessV2Error,
        RobustnessError,
        RobustnessDatasetError,
        OSError,
        ValueError,
    ) as error:
        print(f"R1 operation failed: {error}", file=sys.stderr)
        return 1


def _load_collect_audio_backend() -> Any:
    """Load sounddevice only after the explicit collect command is selected."""

    import sounddevice

    return sounddevice


def _print_r1_evaluation_summary(report: Mapping[str, Any]) -> None:
    positive = report["positive_metrics"]
    negative = report["negative_metrics"]
    criteria = report["development_replication_criteria"]
    required = criteria["required_results"]
    print("DEVELOPMENT REPLICATION ONLY — not external validation or Session C")
    print(
        f"Stage 1 candidate attempts: {positive['stage1_candidate_attempts']}/30 "
        f"({'PASS' if required['stage1_pass'] else 'FAIL'})"
    )
    print(
        "Provisional v2 accepted intended attempts: "
        f"{positive['provisional_v2_accepted_attempts']}/30 "
        f"({'PASS' if required['provisional_v2_survival_pass'] else 'FAIL'})"
    )
    print(
        "Provisional v2 false accepts: "
        f"{negative['provisional_v2_false_accepts']}/300 labeled seconds "
        f"({'PASS' if required['provisional_v2_false_accept_pass'] else 'FAIL'})"
    )
    print(
        "development_replication_pass="
        f"{criteria['development_replication_pass']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
