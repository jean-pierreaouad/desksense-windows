"""Untouched Phase 3B Session B collection and external evaluation.

This module deliberately separates the predeclared external protocol from the
development replay/fitting path.  It reuses the Phase 3 robustness storage
format and frozen pipeline, but never fits or modifies either model artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desksense.collection import DatasetCollectionError, record_collection_attempt, terminal_countdown
from desksense.frozen_baseline import FrozenBaselineError, load_frozen_baseline
from desksense.robustness import (
    NEGATIVE_ACTIVITIES,
    POSITIVE_STRENGTHS,
    POSITIVE_ZONES,
    RobustnessError,
    _capture_metadata,
    _endpoint_metadata,
    _guided_cue_metadata,
    _save_robustness_record,
    _selected_robustness_device,
    _utc_timestamp,
    _validate_fixed_capture_domain,
    _validated_exact_capture,
    positive_robustness_plan,
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
from desksense.streaming import StreamingDetectorConfig, StreamingTapDetector
from desksense.tapness import (
    TapnessBaselineError,
    load_tapness_baseline,
    robustness_dataset_fingerprint,
    stage1_policy_metadata,
    validate_tapness_stage1_config,
)


EXTERNAL_EVIDENCE_ROLE = "external_validation"
EXTERNAL_PROTOCOL_NAME = "phase3b_session_b_untouched_external_validation"
EXTERNAL_PROTOCOL_VERSION = 1
EXTERNAL_PROJECT_PHASE = "3B external validation"
EXTERNAL_SOURCE_MODE = "guided_phase3b_external_validation_collection"
EXTERNAL_REPORT_SCHEMA_VERSION = 1
EXTERNAL_REPORT_TYPE = "phase3b_frozen_pipeline_external_validation"

EXTERNAL_POSITIVE_ATTEMPTS_PER_CONDITION = 5
EXTERNAL_POSITIVE_COUNT = 30
EXTERNAL_POSITIVE_CAPTURE_SECONDS = 2.0
EXTERNAL_POSITIVE_PRE_CUE_SECONDS = 0.9
EXTERNAL_POSITIVE_ASSOCIATION_SECONDS = 0.75
EXTERNAL_NEGATIVE_WARMUP_SECONDS = 1.0
EXTERNAL_NEGATIVE_TAIL_SECONDS = 0.25
EXTERNAL_NEGATIVE_ACTIVITY_SECONDS = {
    "quiet": 30.0,
    "typing": 45.0,
    "speech": 45.0,
    "trackpad": 45.0,
    "hand_movement": 45.0,
    "laptop_movement": 45.0,
    "desk_object_interaction": 45.0,
}
EXTERNAL_NEGATIVE_SEGMENT_COUNT = 7
EXTERNAL_NEGATIVE_LABELED_SECONDS = 300.0

EXPECTED_TAPNESS_SOURCE_SESSION_ID = "20260901T131308.362205Z-f9e2b1ec"
EXPECTED_TAPNESS_SOURCE_DATASET_SHA256 = (
    "88a003966141d15858a2afec390c42e567439eb5f538b958e7aa60ee7638ab83"
)
EXPECTED_TAPNESS_THRESHOLD = 0.4495211534633274
EXPECTED_TAPNESS_FILE_SHA256 = (
    "2c980a0d3ae05bf8f74d9bf3b9ba4a2f35cb9e679e9273f6f55eaf4e6778dbde"
)
EXPECTED_SPATIAL_THRESHOLD_DB = 0.12793235855251162
EXPECTED_SPATIAL_FILE_SHA256 = (
    "42ca902b06cc840b19df409d2869bad21db3f0d6b78c3850510938b31e13b588"
)

EXTERNAL_STRENGTH_DEFINITIONS = {
    "light": (
        "Gentle but deliberate natural tap that the user would reasonably want "
        "recognized; do not artificially strengthen it."
    ),
    "normal": "Comfortable ordinary natural tap.",
    "firm": (
        "Clearly stronger than normal but still a normal fingertip-pad tap; no "
        "nail, knuckle, banging, or exaggerated force."
    ),
}


class ExternalRobustnessError(RuntimeError):
    """An invalid Session B protocol, artifact, evaluation, or report."""


@dataclass(frozen=True)
class ExternalNegativePlanEntry:
    collection_order_index: int
    activity: str
    repetition_index: int
    activity_duration_seconds: float


def external_positive_plan() -> tuple[Any, ...]:
    """Return the predeclared balanced 30-attempt Session B plan."""

    return positive_robustness_plan(EXTERNAL_POSITIVE_ATTEMPTS_PER_CONDITION)


def external_negative_plan(
    *, start_order_index: int = EXTERNAL_POSITIVE_COUNT + 1
) -> tuple[ExternalNegativePlanEntry, ...]:
    """Return the predeclared seven-segment, 300-second negative plan."""

    return tuple(
        ExternalNegativePlanEntry(
            collection_order_index=start_order_index + index,
            activity=activity,
            repetition_index=1,
            activity_duration_seconds=EXTERNAL_NEGATIVE_ACTIVITY_SECONDS[activity],
        )
        for index, activity in enumerate(NEGATIVE_ACTIVITIES)
    )


def external_positive_interaction_protocol() -> dict[str, Any]:
    """Return the exact physical positive protocol frozen before Session B."""

    return {
        "hand_and_finger": {
            "same_for_both_zones": True,
            "required_hand": "RIGHT",
            "required_finger": "RIGHT index finger",
        },
        "contact_method": {
            "required": "fleshy fingertip pad",
            "forbidden": ["fingernail", "knuckle"],
        },
        "intended_taps_per_cue": 1,
        "setup": {
            "desk": "same established wooden desk",
            "laptop_setup": "same established laptop placement and setup",
        },
        "zone_geometry": {
            "LEFT": {
                "location": "established lower/front LEFT tap location",
                "distance_outside_corresponding_laptop_edge_centimeters": [7, 10],
                "orientation": "toward user/touchpad side, away from screen",
            },
            "RIGHT": {
                "location": "established lower/front RIGHT tap location",
                "distance_outside_corresponding_laptop_edge_centimeters": [7, 10],
                "orientation": "toward user/touchpad side, away from screen",
            },
        },
        "strength_definitions": dict(EXTERNAL_STRENGTH_DEFINITIONS),
        "procedural_error_policy": {
            "abort_action": "Press Ctrl+C immediately.",
            "abort_only_for_observable_protocol_error": True,
            "qualifying_errors": [
                "wrong zone",
                "wrong finger or contact method",
                "cue missed entirely",
                "another event made the intended protocol invalid",
            ],
            "partial_session_remains_incomplete": True,
            "partial_session_must_not_be_evaluated": True,
            "fresh_complete_session_allowed_after_procedural_abort": True,
            "model_output_must_not_influence_abort_retry_or_deletion": True,
            "per_attempt_retry_or_deletion_supported": False,
        },
    }


def run_guided_external_robustness_collection(
    audio_backend: Any,
    *,
    device_index: int,
    tapness_baseline_path: Path,
    spatial_baseline_path: Path,
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
    """Collect the fixed Session B protocol without evaluating any capture."""

    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn
    clock = (lambda: datetime.now(timezone.utc)) if now_fn is None else now_fn
    positive_record = record_collection_attempt if positive_capture_fn is None else positive_capture_fn
    negative_record = record_robustness_segment if negative_capture_fn is None else negative_capture_fn
    tapness, spatial, artifacts = _load_and_validate_frozen_artifacts(
        tapness_baseline_path, spatial_baseline_path
    )
    inventory, device = _selected_robustness_device(audio_backend, device_index)
    _validate_fixed_capture_domain(audio_backend, device)
    _validate_endpoint_for_spatial_baseline(device, spatial)
    positives = external_positive_plan()
    negatives = external_negative_plan()
    metadata = _external_session_metadata(
        inventory=inventory,
        device=device,
        positive_plan=positives,
        negative_plan=negatives,
        artifacts=artifacts,
        tapness=tapness,
    )
    try:
        session = create_robustness_dataset_session(
            Path(dataset_root),
            metadata,
            session_id=session_id,
            created_at_utc=_utc_timestamp(clock),
        )
    except (OSError, TypeError, ValueError) as error:
        raise ExternalRobustnessError(
            f"Could not create external robustness dataset: {error}"
        ) from error

    write_output(f"Phase 3B untouched external session: {session.session_id}")
    write_output(
        f"Endpoint {device['index']}: {device['name']} "
        f"({device['host_api']['name']})"
    )
    write_output("Fixed domain: 48000 Hz, 2 channels, float32")
    write_output(
        "Collection is prediction-independent: no Stage 1, Stage 2, or Stage 3 "
        "outcome is displayed or used for retries."
    )
    write_output("Positive physical protocol:")
    write_output(
        "- Use the same RIGHT index finger and fleshy fingertip pad for BOTH "
        "LEFT and RIGHT; never use a nail or knuckle."
    )
    write_output("- Perform exactly one intended tap per TAP NOW cue.")
    write_output(
        "- Keep the established wooden-desk/laptop setup and lower/front LEFT "
        "and RIGHT locations, approximately 7–10 cm outside the corresponding "
        "laptop edge toward the user/touchpad side, away from the screen."
    )
    write_output(
        "If you clearly use the wrong zone, finger/contact method, miss the cue "
        "entirely, or another event invalidates the protocol, press Ctrl+C "
        "immediately. The partial session remains incomplete and must not be "
        "evaluated. Start a fresh complete session only for such an observable "
        "procedural error—never because of model output."
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
            raise ExternalRobustnessError(str(error)) from error
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
                **_capture_metadata(capture, device),
                "guided_cue": _external_guided_cue_metadata(),
            },
        )
        write_output(f"Saved intended attempt: {stem}.npz")

    instructions = _negative_activity_instructions()
    for plan in negatives:
        write_output("")
        write_output(f"Negative activity: {plan.activity}")
        write_output(instructions[plan.activity])
        write_output(
            "Remain quiet for 1 s, then perform the cued activity naturally for "
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
                f"END {activity.upper()} — REMAIN QUIET"
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
                    "end_cue_text": f"END {plan.activity.upper()} — REMAIN QUIET",
                    "cue_is_audible": False,
                    "timing_caution": (
                        "Terminal cues define an engineering schedule; they are "
                        "not measured physical timestamps."
                    ),
                },
                "semantic_label": "no_intended_desk_tap",
                "model_prediction_used_for_retry_or_storage": False,
                **_capture_metadata(capture, device),
            },
        )
        write_output(f"Saved negative activity segment: {stem}_001.npz")

    write_output("")
    write_output(f"External collection complete: 37 records in {session.directory}")
    return {
        "status": "completed",
        "session_id": session.session_id,
        "session_directory": str(session.directory),
        "manifest_path": str(session.manifest_path),
        "evidence_role": EXTERNAL_EVIDENCE_ROLE,
        "positive_attempts": EXTERNAL_POSITIVE_COUNT,
        "negative_segments": EXTERNAL_NEGATIVE_SEGMENT_COUNT,
        "labeled_negative_duration_seconds": EXTERNAL_NEGATIVE_LABELED_SECONDS,
    }


def load_external_robustness_dataset(
    session_path: Path, *, require_complete: bool = False
) -> tuple[LoadedRobustnessDataset, dict[str, Any]]:
    """Load and strictly validate the fixed external protocol and role."""

    try:
        dataset = load_robustness_dataset(Path(session_path))
    except RobustnessDatasetError as error:
        raise ExternalRobustnessError(str(error)) from error
    completeness = _validate_external_dataset(dataset)
    if require_complete and not completeness["external_collection_complete"]:
        raise ExternalRobustnessError(
            "External Session B collection is incomplete; evaluation requires "
            "exactly 30 positives and seven negative segments."
        )
    return dataset, completeness


def evaluate_robustness_external(
    session_path: Path,
    *,
    tapness_baseline_path: Path,
    spatial_baseline_path: Path,
    detector_factory: Callable[[], StreamingTapDetector] = StreamingTapDetector,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Evaluate a complete external session without fitting any component."""

    dataset, completeness = load_external_robustness_dataset(
        session_path, require_complete=True
    )
    tapness, spatial, artifacts = _load_and_validate_frozen_artifacts(
        tapness_baseline_path, spatial_baseline_path
    )
    _validate_dataset_artifact_binding(dataset, artifacts, tapness)
    _validate_dataset_domain(dataset, spatial)
    integrity_before = _external_integrity_snapshot(
        Path(session_path),
        Path(tapness_baseline_path),
        Path(spatial_baseline_path),
    )
    replay = replay_robustness_dataset(
        Path(session_path),
        baseline_path=Path(spatial_baseline_path),
        tapness_baseline_path=Path(tapness_baseline_path),
        detector_factory=detector_factory,
        now_fn=now_fn,
    )
    integrity_after = _external_integrity_snapshot(
        Path(session_path),
        Path(tapness_baseline_path),
        Path(spatial_baseline_path),
    )
    _require_external_integrity_unchanged(integrity_before, integrity_after)
    positive = _external_positive_metrics(replay)
    negative = _external_negative_metrics(replay)
    gates = _engineering_gates(positive, negative)
    report = {
        "report_schema_version": EXTERNAL_REPORT_SCHEMA_VERSION,
        "report_type": EXTERNAL_REPORT_TYPE,
        "project_phase": "3B external validation",
        "generated_at_utc": replay["generated_at_utc"],
        "evidence_role": EXTERNAL_EVIDENCE_ROLE,
        "session_id": dataset.session["session_id"],
        "session_path": str(dataset.directory),
        "protocol": dict(dataset.session["external_validation_protocol"]),
        "collection_context": {
            "selected_endpoint": dict(dataset.session["selected_endpoint"]),
            "capture_domain": dict(dataset.session["capture_domain"]),
            "frozen_stage1_policy": dict(
                dataset.session["frozen_pipeline"]["stage1_policy"]
            ),
            "frozen_stage2_tapness_artifact": dict(
                artifacts["tapness_baseline"]
            ),
            "frozen_stage3_spatial_artifact": dict(
                artifacts["spatial_baseline"]
            ),
            "physical_positive_interaction_protocol": dict(
                dataset.session["physical_positive_interaction_protocol"]
            ),
            "negative_activity_durations_seconds": dict(
                EXTERNAL_NEGATIVE_ACTIVITY_SECONDS
            ),
            "negative_denominator": {
                "labeled_seconds": EXTERNAL_NEGATIVE_LABELED_SECONDS,
                "segment_count": EXTERNAL_NEGATIVE_SEGMENT_COUNT,
                "semantics": (
                    "300 labeled seconds across seven separately recorded and "
                    "replayed activity segments, each with its own excluded "
                    "quiet warm-up and completion tail; not one continuous "
                    "five-minute live run."
                ),
            },
        },
        "collection_completeness": completeness,
        "artifact_identity": artifacts,
        "artifact_paths": {
            "tapness_baseline": str(Path(tapness_baseline_path).resolve()),
            "spatial_baseline": str(Path(spatial_baseline_path).resolve()),
        },
        "dataset_integrity": {
            "status": "passed",
            "record_count": len(dataset.records),
            "fingerprint_before_replay": integrity_before["dataset_fingerprint"],
            "fingerprint_after_replay": integrity_after["dataset_fingerprint"],
            "dataset_files_modified": False,
        },
        "artifact_integrity": {
            "tapness_sha256_before_replay": integrity_before[
                "tapness_artifact_sha256"
            ],
            "tapness_sha256_after_replay": integrity_after[
                "tapness_artifact_sha256"
            ],
            "spatial_sha256_before_replay": integrity_before[
                "spatial_artifact_sha256"
            ],
            "spatial_sha256_after_replay": integrity_after[
                "spatial_artifact_sha256"
            ],
            "verified_unchanged": True,
        },
        "positive_metrics": positive,
        "negative_metrics": negative,
        "engineering_gates": gates,
        "positive_attempts": replay["positive"]["attempts"],
        "negative_segments": replay["negative"]["segments"],
        "no_refit_declaration": {
            "stage1_policy_changed": False,
            "tapness_model_fitted": False,
            "tapness_threshold_changed": False,
            "tapness_features_selected_or_changed": False,
            "spatial_model_fitted": False,
            "spatial_threshold_changed": False,
            "external_samples_used_for_fitting": False,
        },
        "evidence_boundary": {
            "tapness_development_session_id": EXPECTED_TAPNESS_SOURCE_SESSION_ID,
            "external_session_id": dataset.session["session_id"],
            "failure_discipline": (
                "If this Session B result influences any later detector, Stage 1, "
                "Stage 2, feature, threshold, or model change, Session B becomes "
                "development evidence and a modified pipeline requires a new "
                "untouched Session C for another external-validation claim."
            ),
            "scope": (
                "One predeclared cross-session robustness protocol; not a product, "
                "cross-device, or general reliability claim."
            ),
        },
        "privacy": {
            "processing": "local_offline",
            "network_upload_performed": False,
            "waveforms_embedded_in_report": False,
        },
    }
    json.dumps(report, allow_nan=False)
    return report


def write_external_robustness_report(
    report: Mapping[str, Any], path: Path
) -> Path:
    """Exclusively write one waveform-free external-validation report."""

    if report.get("report_type") != EXTERNAL_REPORT_TYPE:
        raise ExternalRobustnessError("Not a Phase 3B external-validation report.")
    _reject_waveform_fields(report)
    target = Path(path)
    if target.suffix.casefold() != ".json":
        raise ExternalRobustnessError("External report path must end in .json.")
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as output:
            output.write(serialized)
            output.write("\n")
    except FileExistsError as error:
        raise ExternalRobustnessError(
            f"Refusing to overwrite existing external report: {target}"
        ) from error
    return target.resolve()


def format_external_robustness_summary(report: Mapping[str, Any]) -> str:
    """Render a concise, evidence-scoped external evaluation summary."""

    positive = report["positive_metrics"]
    negative = report["negative_metrics"]
    stage3 = positive["stage3"]
    return "\n".join(
        [
            "DeskSense Phase 3B frozen-pipeline external validation",
            f"Session: {report['session_id']}",
            (
                "Stage 1 intended-tap candidates: "
                f"{positive['stage1']['attempts_with_candidate_start']}/30"
            ),
            (
                "Stage 2 intended taps accepted: "
                f"{positive['stage2']['attempts_accepted_as_tap']}/30"
            ),
            (
                "Conditional frozen spatial result: "
                f"{stage3['correct_count']}/{stage3['classified_count']}"
            ),
            (
                "End-to-end intended-zone outputs: "
                f"{positive['end_to_end']['correct_final_zone_outputs']}/30"
            ),
            (
                "Negative Stage 2 false accepts: "
                f"{negative['stage2_false_accepts']} over exactly 300 labeled s "
                f"({negative['stage2_false_accepts_per_minute']:.3f}/min)"
            ),
            (
                "Predeclared required engineering gates: "
                + ("PASS" if report["engineering_gates"]["engineering_gate_pass"] else "FAIL")
            ),
            "Cell >=4/5 is reported separately as a preference, not a hard gate.",
        ]
    )


def _external_session_metadata(
    *,
    inventory: Mapping[str, Any],
    device: Mapping[str, Any],
    positive_plan: Sequence[Any],
    negative_plan: Sequence[ExternalNegativePlanEntry],
    artifacts: Mapping[str, Any],
    tapness: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "project_phase": EXTERNAL_PROJECT_PHASE,
        "source_mode": EXTERNAL_SOURCE_MODE,
        "evidence_role": EXTERNAL_EVIDENCE_ROLE,
        "no_fitting_allowed": True,
        "external_validation_protocol": _external_validation_protocol_metadata(),
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
            "requested_attempt_count": EXTERNAL_POSITIVE_COUNT,
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
            "requested_segment_count": EXTERNAL_NEGATIVE_SEGMENT_COUNT,
            "warmup_duration_seconds": EXTERNAL_NEGATIVE_WARMUP_SECONDS,
            "post_activity_tail_duration_seconds": EXTERNAL_NEGATIVE_TAIL_SECONDS,
            "labeled_activity_duration_seconds": EXTERNAL_NEGATIVE_LABELED_SECONDS,
            "semantic_label": "no_intended_desk_tap",
            "ordering": "one predeclared segment per activity",
            "plan": [asdict(item) for item in negative_plan],
        },
        "frozen_pipeline": {
            "stage1_policy": dict(tapness["stage1_policy"]),
            "tapness_baseline": dict(artifacts["tapness_baseline"]),
            "spatial_baseline": dict(artifacts["spatial_baseline"]),
            "artifacts_may_be_modified_by_collection": False,
        },
        "privacy": {
            "waveforms_retained_locally": True,
            "network_upload_performed": False,
            "ordinary_sense_audio_saved": False,
        },
    }


def _external_validation_protocol_metadata() -> dict[str, Any]:
    return {
        "name": EXTERNAL_PROTOCOL_NAME,
        "version": EXTERNAL_PROTOCOL_VERSION,
        "predeclared_before_collection": True,
        "collection_independent_of_model_predictions": True,
        "labeled_negative_denominator_seconds": EXTERNAL_NEGATIVE_LABELED_SECONDS,
        "negative_segment_count": EXTERNAL_NEGATIVE_SEGMENT_COUNT,
        "negative_recording_structure": (
            "seven separately recorded and replayed activity segments; each has "
            "its own excluded quiet warm-up and completion tail"
        ),
        "fitting_allowed": False,
    }


def _external_guided_cue_metadata() -> dict[str, Any]:
    # Reuse the established metadata definition with the fixed protocol values.
    from desksense.robustness import RobustnessCollectionParameters

    return _guided_cue_metadata(
        RobustnessCollectionParameters(
            attempts_per_condition=EXTERNAL_POSITIVE_ATTEMPTS_PER_CONDITION,
            positive_capture_seconds=EXTERNAL_POSITIVE_CAPTURE_SECONDS,
            pre_cue_seconds=EXTERNAL_POSITIVE_PRE_CUE_SECONDS,
            positive_association_seconds=EXTERNAL_POSITIVE_ASSOCIATION_SECONDS,
        )
    )


def _validate_external_dataset(dataset: LoadedRobustnessDataset) -> dict[str, Any]:
    session = dataset.session
    if session.get("project_phase") != EXTERNAL_PROJECT_PHASE:
        raise ExternalRobustnessError(
            "External session project_phase does not match protocol v1."
        )
    if session.get("source_mode") != EXTERNAL_SOURCE_MODE:
        raise ExternalRobustnessError(
            "External session source_mode does not match the collection mode."
        )
    if session.get("evidence_role") != EXTERNAL_EVIDENCE_ROLE:
        raise ExternalRobustnessError(
            "Robustness session is not marked evidence_role=external_validation."
        )
    if session.get("no_fitting_allowed") is not True:
        raise ExternalRobustnessError("External session must explicitly forbid fitting.")
    if session.get("session_id") == EXPECTED_TAPNESS_SOURCE_SESSION_ID:
        raise ExternalRobustnessError(
            "External session ID must differ from the tapness development session."
        )
    if session.get("external_validation_protocol") != (
        _external_validation_protocol_metadata()
    ):
        raise ExternalRobustnessError(
            "External validation protocol metadata does not exactly match v1."
        )
    if session.get("physical_positive_interaction_protocol") != (
        external_positive_interaction_protocol()
    ):
        raise ExternalRobustnessError(
            "External physical positive interaction protocol does not match v1."
        )
    _validate_external_design(session)
    frozen = session.get("frozen_pipeline")
    if not isinstance(frozen, Mapping):
        raise ExternalRobustnessError("External session lacks frozen pipeline metadata.")
    if frozen.get("stage1_policy") != stage1_policy_metadata():
        raise ExternalRobustnessError("External session has the wrong frozen Stage 1 policy.")
    for artifact_key in ("tapness_baseline", "spatial_baseline"):
        identity = frozen.get(artifact_key)
        if not isinstance(identity, Mapping):
            raise ExternalRobustnessError(
                f"External session lacks {artifact_key} identity metadata."
            )
        digest = identity.get("file_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ExternalRobustnessError(
                f"External session {artifact_key} SHA-256 is invalid."
            )

    expected_positive = external_positive_plan()
    expected_negative = external_negative_plan()
    observed_positive = dataset.positive_records
    observed_negative = dataset.negative_records
    observed_positive_plan = []
    for record in observed_positive:
        metadata = record.metadata
        if metadata.get("model_prediction_used_for_retry_or_storage") is not False:
            raise ExternalRobustnessError(
                "External positive record lacks prediction-independence metadata."
            )
        if metadata.get("detector_acceptance_required_for_storage") is not False:
            raise ExternalRobustnessError(
                "External positive record must not require detector acceptance."
            )
        observed_positive_plan.append(
            {
                "collection_order_index": metadata["collection_order_index"],
                "zone": metadata["intended_zone"],
                "strength": metadata["intended_strength"],
                "attempt_number_within_condition": metadata[
                    "attempt_number_within_condition"
                ],
            }
        )
        if record.capture.shape != (96_000, 2):
            raise ExternalRobustnessError("External positive capture must be 96000x2.")
        cue = metadata["guided_cue"]
        association = cue["intended_event_association"]
        if (
            cue["intended_cue_offset_frames"] != 43_200
            or association["start_frame_index_inclusive"] != 43_200
            or association["end_frame_index_exclusive"] != 79_200
        ):
            raise ExternalRobustnessError("External positive cue boundaries are invalid.")

    expected_positive_plan = [asdict(item) for item in expected_positive]
    if observed_positive_plan != expected_positive_plan[: len(observed_positive_plan)]:
        raise ExternalRobustnessError(
            "External positive records do not exactly match the planned prefix."
        )

    observed_negative_plan = []
    for record in observed_negative:
        metadata = record.metadata
        if metadata.get("model_prediction_used_for_retry_or_storage") is not False:
            raise ExternalRobustnessError(
                "External negative record lacks prediction-independence metadata."
            )
        if len(observed_positive_plan) != EXTERNAL_POSITIVE_COUNT:
            raise ExternalRobustnessError(
                "External negative records cannot precede the complete positive plan."
            )
        negative_index = len(observed_negative_plan)
        if negative_index >= len(expected_negative):
            raise ExternalRobustnessError(
                "External dataset contains more negative records than planned."
            )
        plan = expected_negative[negative_index]
        observed_negative_plan.append(
            {
                "collection_order_index": metadata["collection_order_index"],
                "activity": metadata["activity"],
                "repetition_index": metadata["repetition_index"],
                "activity_duration_seconds": metadata["activity_duration_seconds"],
            }
        )
        expected_start = 48_000
        expected_end = expected_start + round(
            ROBUSTNESS_SAMPLE_RATE_HZ * plan.activity_duration_seconds
        )
        expected_tail_end = expected_end + 12_000
        if (
            metadata["collection_order_index"] != plan.collection_order_index
            or metadata["activity"] != plan.activity
            or metadata["repetition_index"] != 1
            or metadata["warmup_end_frame_index"] != expected_start
            or metadata["activity_start_frame_index"] != expected_start
            or metadata["activity_end_frame_index_exclusive"] != expected_end
            or metadata["post_activity_start_frame_index"] != expected_end
            or metadata["post_activity_end_frame_index_exclusive"] != expected_tail_end
            or record.capture.shape != (expected_tail_end, 2)
        ):
            raise ExternalRobustnessError("External negative record boundaries are invalid.")

    expected_negative_plan = [asdict(item) for item in expected_negative]
    if observed_negative_plan != expected_negative_plan[: len(observed_negative_plan)]:
        raise ExternalRobustnessError(
            "External negative records do not exactly match the planned prefix."
        )
    positive_plan_complete = observed_positive_plan == expected_positive_plan
    negative_plan_complete = observed_negative_plan == expected_negative_plan
    return {
        "expected_positive_attempts": EXTERNAL_POSITIVE_COUNT,
        "observed_positive_attempts": len(observed_positive),
        "expected_negative_segments": EXTERNAL_NEGATIVE_SEGMENT_COUNT,
        "observed_negative_segments": len(observed_negative),
        "expected_labeled_negative_seconds": EXTERNAL_NEGATIVE_LABELED_SECONDS,
        "positive_plan_complete": positive_plan_complete,
        "negative_plan_complete": negative_plan_complete,
        "external_collection_complete": (
            positive_plan_complete and negative_plan_complete
        ),
    }


def _validate_external_design(session: Mapping[str, Any]) -> None:
    positive = session["positive_design"]
    negative = session["negative_design"]
    expected_positive = [asdict(item) for item in external_positive_plan()]
    expected_negative = [asdict(item) for item in external_negative_plan()]
    if (
        positive.get("requested_attempt_count") != EXTERNAL_POSITIVE_COUNT
        or positive.get("attempts_per_condition") != 5
        or positive.get("plan") != expected_positive
    ):
        raise ExternalRobustnessError("External positive design is not the fixed 30-attempt plan.")
    for key, expected in {
        "capture_duration_seconds": 2.0,
        "pre_cue_seconds": 0.9,
        "association_duration_seconds": 0.75,
    }.items():
        _require_exact_number(positive, key, expected)
    if (
        negative.get("requested_segment_count") != 7
        or negative.get("segments_per_activity") != 1
        or negative.get("plan") != expected_negative
    ):
        raise ExternalRobustnessError("External negative design is not the fixed seven-segment plan.")
    for key, expected in {
        "warmup_duration_seconds": 1.0,
        "post_activity_tail_duration_seconds": 0.25,
        "labeled_activity_duration_seconds": 300.0,
    }.items():
        _require_exact_number(negative, key, expected)


def _load_and_validate_frozen_artifacts(
    tapness_path: Path, spatial_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        tapness = load_tapness_baseline(Path(tapness_path))
        spatial = load_frozen_baseline(Path(spatial_path))
        validate_tapness_stage1_config(StreamingDetectorConfig(), tapness)
    except (
        FrozenBaselineError,
        TapnessBaselineError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise ExternalRobustnessError(f"Frozen artifact validation failed: {error}") from error
    source = tapness["source_development_dataset"]
    if source["session_id"] != EXPECTED_TAPNESS_SOURCE_SESSION_ID:
        raise ExternalRobustnessError("Wrong tapness source development session.")
    if source["dataset_fingerprint"]["sha256"] != EXPECTED_TAPNESS_SOURCE_DATASET_SHA256:
        raise ExternalRobustnessError("Wrong tapness source dataset fingerprint.")
    if not math.isclose(
        float(tapness["model"]["decision_threshold"]),
        EXPECTED_TAPNESS_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ExternalRobustnessError("Wrong frozen tapness decision threshold.")
    if not math.isclose(
        float(spatial["classifier"]["threshold_db"]),
        EXPECTED_SPATIAL_THRESHOLD_DB,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ExternalRobustnessError("Wrong frozen spatial threshold.")
    compatibility = spatial["source_development_dataset"]["compatibility"]
    if (
        float(compatibility["sample_rate_hz"]) != 48_000.0
        or compatibility["channel_count"] != 2
        or compatibility["tap_window_frames"] != 9_600
    ):
        raise ExternalRobustnessError("Frozen spatial domain is not 48000 Hz, 2ch, 9600 frames.")
    tap_hash = _file_sha256(Path(tapness_path))
    spatial_hash = _file_sha256(Path(spatial_path))
    if tap_hash != EXPECTED_TAPNESS_FILE_SHA256:
        raise ExternalRobustnessError("Tapness artifact file identity is not the frozen v1 artifact.")
    if spatial_hash != EXPECTED_SPATIAL_FILE_SHA256:
        raise ExternalRobustnessError("Spatial artifact file identity is not the frozen v1 artifact.")
    return tapness, spatial, {
        "tapness_baseline": {
            "artifact_type": tapness["artifact_type"],
            "schema_version": tapness["artifact_schema_version"],
            "file_sha256": tap_hash,
            "source_development_session_id": source["session_id"],
            "source_dataset_sha256": source["dataset_fingerprint"]["sha256"],
            "decision_threshold": tapness["model"]["decision_threshold"],
        },
        "spatial_baseline": {
            "artifact_type": spatial["artifact_type"],
            "schema_version": spatial["baseline_schema_version"],
            "file_sha256": spatial_hash,
            "source_development_session_id": spatial["source_development_dataset"]["session_id"],
            "source_dataset_sha256": spatial["source_development_dataset"]["fingerprint"]["digest_hex"],
            "decision_threshold_db": spatial["classifier"]["threshold_db"],
        },
    }


def _validate_dataset_artifact_binding(
    dataset: LoadedRobustnessDataset,
    artifacts: Mapping[str, Any],
    tapness: Mapping[str, Any],
) -> None:
    frozen = dataset.session.get("frozen_pipeline")
    if not isinstance(frozen, Mapping):
        raise ExternalRobustnessError("External session lacks frozen pipeline identity.")
    if frozen.get("artifacts_may_be_modified_by_collection") is not False:
        raise ExternalRobustnessError("External session artifact immutability declaration is invalid.")
    if frozen.get("tapness_baseline") != artifacts["tapness_baseline"]:
        raise ExternalRobustnessError("Tapness artifact differs from the one bound at collection.")
    if frozen.get("spatial_baseline") != artifacts["spatial_baseline"]:
        raise ExternalRobustnessError("Spatial artifact differs from the one bound at collection.")
    if frozen.get("stage1_policy") != tapness["stage1_policy"]:
        raise ExternalRobustnessError(
            "External session Stage 1 policy differs from the frozen tapness artifact."
        )


def _validate_dataset_domain(
    dataset: LoadedRobustnessDataset, spatial: Mapping[str, Any]
) -> None:
    compatibility = spatial["source_development_dataset"]["compatibility"]
    endpoint = dataset.session["selected_endpoint"]
    if (
        endpoint["name"] != compatibility["endpoint_name"]
        or endpoint["host_api"]["name"] != compatibility["host_api_name"]
    ):
        raise ExternalRobustnessError("External endpoint identity does not match spatial baseline.")


def _validate_endpoint_for_spatial_baseline(
    device: Mapping[str, Any], spatial: Mapping[str, Any]
) -> None:
    compatibility = spatial["source_development_dataset"]["compatibility"]
    if (
        device["name"] != compatibility["endpoint_name"]
        or device["host_api"]["name"] != compatibility["host_api_name"]
    ):
        raise ExternalRobustnessError(
            "Selected endpoint identity/host API does not match the frozen spatial baseline."
        )


def _external_positive_metrics(replay: Mapping[str, Any]) -> dict[str, Any]:
    attempts = replay["positive"]["attempts"]
    summary = replay["positive"]["summary"]
    cells: dict[str, Any] = {}
    for zone in POSITIVE_ZONES:
        for strength in POSITIVE_STRENGTHS:
            selected = [
                item
                for item in attempts
                if item["intended_zone"] == zone
                and item["intended_strength"] == strength
            ]
            stage1 = sum(bool(item["stage1_candidate_started"]) for item in selected)
            stage2 = sum(_attempt_stage2_accepted(item) for item in selected)
            cells[f"{zone}-{strength}"] = {
                "intended_attempts": len(selected),
                "stage1_candidate_attempts": stage1,
                "stage2_accepted_attempts": stage2,
                "preferred_at_least_4_of_5": stage2 >= 4,
            }
    predictions = [
        event["frozen_spatial_prediction"]
        for attempt in attempts
        for event in attempt["completed_events"]
        if event["interval_relation"] == "associated"
        and event["frozen_spatial_prediction"] is not None
    ]
    confusion = {
        actual: {predicted: 0 for predicted in POSITIVE_ZONES}
        for actual in POSITIVE_ZONES
    }
    for prediction in predictions:
        confusion[prediction["actual_zone"]][prediction["predicted_zone"]] += 1
    return {
        "intended_attempt_count": 30,
        "stage1": {
            "associated_candidate_start_count": sum(
                int(item["associated_candidate_start_count"]) for item in attempts
            ),
            "attempts_with_candidate_start": summary["attempts_with_associated_candidate_start"],
            "candidate_recall": summary["stage1_candidate_start_recall"],
            "by_zone": summary["by_zone"],
            "by_strength": summary["by_strength"],
            "by_side_and_strength": cells,
        },
        "stage2": {
            "attempts_accepted_as_tap": summary["stage2_tapness"]["attempts_with_accepted_tap"],
            "intended_tap_acceptance_rate": summary["stage2_tapness"]["intended_attempt_acceptance_rate"],
            "by_zone": summary["by_zone"],
            "by_strength": summary["by_strength"],
            "by_side_and_strength": cells,
        },
        "stage3": {
            "classified_count": summary["conditional_frozen_spatial"]["classified_count"],
            "correct_count": summary["conditional_frozen_spatial"]["correct_count"],
            "conditional_spatial_correctness": summary["conditional_frozen_spatial"]["accuracy"],
            "confusion_matrix": confusion,
        },
        "end_to_end": {
            "correct_final_zone_outputs": summary["end_to_end"]["intended_taps_accepted_and_spatially_correct"],
            "total_intended_attempts": 30,
            "success_rate": summary["end_to_end"]["success_rate"],
        },
    }


def _external_negative_metrics(replay: Mapping[str, Any]) -> dict[str, Any]:
    summary = replay["negative"]["summary"]
    if not math.isclose(
        float(summary["labeled_activity_duration_seconds"]),
        EXTERNAL_NEGATIVE_LABELED_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ExternalRobustnessError("Replay negative denominator is not exactly 300 seconds.")
    per_activity = {
        activity: {
            "labeled_seconds": values["labeled_activity_duration_seconds"],
            "stage1_candidate_starts": values["activity_associated_candidate_starts"],
            "stage1_candidates_per_minute": values["candidate_start_rate_per_labeled_minute"],
            "stage2_false_accepts": values["stage2_false_accepted_events"],
            "stage2_false_accepts_per_minute": values["stage2_false_accept_rate_per_labeled_minute"],
        }
        for activity, values in summary["per_activity"].items()
    }
    return {
        "labeled_activity_seconds": 300.0,
        "stage1_candidate_starts": summary["activity_associated_candidate_starts"],
        "stage1_candidates_per_minute": summary["candidate_start_rate_per_labeled_minute"],
        "stage2_false_accepts": summary["stage2_false_accepted_events"],
        "stage2_false_accepts_per_minute": summary["stage2_false_accept_rate_per_labeled_minute"],
        "per_activity": per_activity,
        "excluded_context": {
            "warmup_seconds": summary["warmup_duration_seconds"],
            "warmup_candidate_starts": summary["warmup_candidate_starts"],
            "warmup_completed_detections": summary["warmup_completed_detections"],
            "post_activity_tail_seconds": summary["post_activity_tail_duration_seconds"],
            "post_activity_candidate_starts": summary["post_activity_candidate_starts"],
            "post_activity_completed_detections": summary["post_activity_completed_detections"],
        },
        "stage3_negative_evaluation": {
            "status": "not_applicable",
            "reason": (
                "Negative activities have no valid LEFT/RIGHT ground-truth "
                "label; Stage 2 false acceptance is the robustness error."
            ),
            "left_right_accuracy_reported": False,
        },
    }


def _engineering_gates(
    positive: Mapping[str, Any], negative: Mapping[str, Any]
) -> dict[str, Any]:
    stage1_pass = positive["stage1"]["attempts_with_candidate_start"] >= 27
    stage2_pass = positive["stage2"]["attempts_accepted_as_tap"] >= 27
    end_to_end_pass = positive["end_to_end"]["correct_final_zone_outputs"] >= 27
    negative_pass = negative["stage2_false_accepts"] <= 1
    cells = positive["stage2"]["by_side_and_strength"]
    cell_preference = all(item["preferred_at_least_4_of_5"] for item in cells.values())
    return {
        "predeclared_required_gates": {
            "stage1_at_least_27_of_30": stage1_pass,
            "stage2_at_least_27_of_30": stage2_pass,
            "end_to_end_correct_zone_outputs_at_least_27_of_30": (
                end_to_end_pass
            ),
            "stage2_negative_false_accepts_at_most_1_of_300_seconds": negative_pass,
        },
        "engineering_gate_pass": (
            stage1_pass and stage2_pass and end_to_end_pass and negative_pass
        ),
        "cell_preference": {
            "criterion": "preferably at least 4/5 Stage 2 accepted in every side-strength cell",
            "all_cells_meet_preference": cell_preference,
            "included_in_required_overall_gate": False,
        },
        "stage3_conditional_performance_is_reported_separately": True,
    }


def _attempt_stage2_accepted(attempt: Mapping[str, Any]) -> bool:
    return any(
        event["interval_relation"] == "associated"
        and event["frozen_tapness_prediction"] is not None
        and event["frozen_tapness_prediction"]["tap_accepted"]
        for event in attempt["completed_events"]
    )


def _negative_activity_instructions() -> dict[str, str]:
    return {
        "quiet": "Remain still and silent; perform no intentional desk interaction.",
        "typing": "Type naturally and ordinarily.",
        "speech": "Use normal continuous conversational speech.",
        "trackpad": "Use normal pointer movement, scrolling, and clicks.",
        "hand_movement": (
            "Reposition, rest, and lift hands normally around the keyboard, "
            "palm rest, and desk; do not perform intended desk taps."
        ),
        "laptop_movement": (
            "Naturally reposition or handle the laptop slightly, then restore "
            "the original setup before the next segment."
        ),
        "desk_object_interaction": (
            "Naturally handle and place ordinary desk objects; ordinary impacts "
            "are valid hard negatives, but do not imitate tap-test behavior."
        ),
    }


def _require_exact_number(
    mapping: Mapping[str, Any], key: str, expected: float
) -> None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExternalRobustnessError(f"External protocol field {key} is invalid.")
    if not math.isfinite(float(value)) or not math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ExternalRobustnessError(f"External protocol field {key} is invalid.")


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise ExternalRobustnessError(f"Could not hash frozen artifact: {error}") from error


def _external_integrity_snapshot(
    session_path: Path, tapness_path: Path, spatial_path: Path
) -> dict[str, Any]:
    try:
        dataset_fingerprint = robustness_dataset_fingerprint(Path(session_path))
    except (RobustnessDatasetError, OSError, TypeError, ValueError) as error:
        raise ExternalRobustnessError(
            f"Could not fingerprint external dataset: {error}"
        ) from error
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "tapness_artifact_sha256": _file_sha256(Path(tapness_path)),
        "spatial_artifact_sha256": _file_sha256(Path(spatial_path)),
    }


def _require_external_integrity_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    if before["dataset_fingerprint"] != after["dataset_fingerprint"]:
        raise ExternalRobustnessError(
            "External dataset bytes changed during read-only evaluation."
        )
    if before["tapness_artifact_sha256"] != after["tapness_artifact_sha256"]:
        raise ExternalRobustnessError(
            "Frozen tapness artifact bytes changed during evaluation."
        )
    if before["spatial_artifact_sha256"] != after["spatial_artifact_sha256"]:
        raise ExternalRobustnessError(
            "Frozen spatial artifact bytes changed during evaluation."
        )


def _reject_waveform_fields(value: Any, *, path: str = "report") -> None:
    forbidden = {
        "candidate_window",
        "capture",
        "raw_audio",
        "tap_window",
        "waveform",
        "waveforms",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in forbidden:
                raise ExternalRobustnessError(
                    f"External report contains forbidden waveform field at {path}.{key}."
                )
            _reject_waveform_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_waveform_fields(item, path=f"{path}[{index}]")
