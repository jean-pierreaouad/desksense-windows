"""Phase 3B.0 robustness evidence collection and unchanged-detector replay.

This module intentionally adds no production onset or tap-validation policy.
It stores explicitly guided evidence, replays the current detector offline, and
computes descriptive research metrics that do not affect any detector result.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desksense.collection import (
    DatasetCollectionError,
    record_collection_attempt,
    terminal_countdown,
)
from desksense.diagnostics import (
    collect_audio_inventory,
    describe_audio_error,
    system_information,
)
from desksense.features import PRIMARY_FEATURE_NAME, extract_two_channel_features
from desksense.frozen_baseline import load_frozen_baseline
from desksense.inference import classify_peak_ratio_value
from desksense.robustness_dataset import (
    NEGATIVE_RECORD_TYPE,
    POSITIVE_RECORD_TYPE,
    ROBUSTNESS_CHANNEL_COUNT,
    ROBUSTNESS_DTYPE,
    ROBUSTNESS_NEGATIVE_ACTIVITIES,
    ROBUSTNESS_POSITIVE_STRENGTHS,
    ROBUSTNESS_POSITIVE_ZONES,
    ROBUSTNESS_SAMPLE_RATE_HZ,
    LoadedRobustnessDataset,
    RobustnessDatasetError,
    RobustnessRecord,
    create_robustness_dataset_session,
    load_robustness_dataset,
    validate_robustness_capture,
)
from desksense.streaming import DetectionResult, StreamingTapDetector


POSITIVE_ZONES = ROBUSTNESS_POSITIVE_ZONES
POSITIVE_STRENGTHS = ROBUSTNESS_POSITIVE_STRENGTHS
NEGATIVE_ACTIVITIES = ROBUSTNESS_NEGATIVE_ACTIVITIES
DEFAULT_ATTEMPTS_PER_CONDITION = 5
DEFAULT_POSITIVE_CAPTURE_SECONDS = 2.0
DEFAULT_PRE_CUE_SECONDS = 0.9
DEFAULT_POSITIVE_ASSOCIATION_SECONDS = 0.75
DEFAULT_NEGATIVE_SEGMENTS_PER_ACTIVITY = 2
DEFAULT_NEGATIVE_SEGMENT_SECONDS = 10.0
DEFAULT_NEGATIVE_WARMUP_SECONDS = 1.0
DEFAULT_NEGATIVE_POST_ACTIVITY_TAIL_SECONDS = 0.25
ROBUSTNESS_REPORT_SCHEMA_VERSION = 1
ROBUSTNESS_REPORT_TYPE = "phase3_robustness_offline_replay"

_METRIC_PRE_ONSET_SECONDS = 0.012
_METRIC_IMPACT_SECONDS = 0.025
_METRIC_EARLY_SECONDS = 0.025
_METRIC_LATE_START_SECONDS = 0.040
_METRIC_FRAME_SECONDS = 0.005
_METRIC_STRONG_FRACTION_OF_PEAK = 0.50


class RobustnessError(RuntimeError):
    """A Phase 3B.0 collection, replay, or reporting failure."""

    def __init__(self, message: str, *, category: str = "robustness_error") -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class PositivePlanEntry:
    collection_order_index: int
    zone: str
    strength: str
    attempt_number_within_condition: int


@dataclass(frozen=True)
class NegativePlanEntry:
    collection_order_index: int
    activity: str
    repetition_index: int


@dataclass(frozen=True)
class RobustnessCollectionParameters:
    attempts_per_condition: int = DEFAULT_ATTEMPTS_PER_CONDITION
    positive_capture_seconds: float = DEFAULT_POSITIVE_CAPTURE_SECONDS
    pre_cue_seconds: float = DEFAULT_PRE_CUE_SECONDS
    positive_association_seconds: float = DEFAULT_POSITIVE_ASSOCIATION_SECONDS
    negative_segments_per_activity: int = DEFAULT_NEGATIVE_SEGMENTS_PER_ACTIVITY
    negative_segment_seconds: float = DEFAULT_NEGATIVE_SEGMENT_SECONDS
    negative_warmup_seconds: float = DEFAULT_NEGATIVE_WARMUP_SECONDS
    negative_post_activity_tail_seconds: float = (
        DEFAULT_NEGATIVE_POST_ACTIVITY_TAIL_SECONDS
    )

    def __post_init__(self) -> None:
        for label, value in {
            "attempts per condition": self.attempts_per_condition,
            "negative segments per activity": self.negative_segments_per_activity,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label.capitalize()} must be a positive integer.")
        for label, value in {
            "positive capture duration": self.positive_capture_seconds,
            "pre-cue duration": self.pre_cue_seconds,
            "positive association duration": self.positive_association_seconds,
            "negative segment duration": self.negative_segment_seconds,
            "negative warm-up duration": self.negative_warmup_seconds,
            "negative post-activity tail duration": (
                self.negative_post_activity_tail_seconds
            ),
        }.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{label.capitalize()} must be positive and finite.")
        if self.pre_cue_seconds >= self.positive_capture_seconds:
            raise ValueError("Pre-cue duration must be shorter than positive capture.")
        if (
            self.pre_cue_seconds + self.positive_association_seconds
            >= self.positive_capture_seconds
        ):
            raise ValueError(
                "Positive association interval must leave post-association capture context."
            )


def positive_robustness_plan(
    attempts_per_condition: int = DEFAULT_ATTEMPTS_PER_CONDITION,
) -> tuple[PositivePlanEntry, ...]:
    """Return deterministic repetition-major round-robin positive ordering."""

    if (
        isinstance(attempts_per_condition, bool)
        or not isinstance(attempts_per_condition, int)
        or attempts_per_condition <= 0
    ):
        raise ValueError("Attempts per condition must be a positive integer.")
    conditions = tuple(
        (zone, strength)
        for strength in POSITIVE_STRENGTHS
        for zone in POSITIVE_ZONES
    )
    entries: list[PositivePlanEntry] = []
    for attempt_number in range(1, attempts_per_condition + 1):
        for zone, strength in conditions:
            entries.append(
                PositivePlanEntry(
                    collection_order_index=len(entries) + 1,
                    zone=zone,
                    strength=strength,
                    attempt_number_within_condition=attempt_number,
                )
            )
    return tuple(entries)


def negative_robustness_plan(
    segments_per_activity: int = DEFAULT_NEGATIVE_SEGMENTS_PER_ACTIVITY,
    *,
    start_order_index: int = 1,
) -> tuple[NegativePlanEntry, ...]:
    """Return deterministic repetition-major negative activity ordering."""

    if (
        isinstance(segments_per_activity, bool)
        or not isinstance(segments_per_activity, int)
        or segments_per_activity <= 0
    ):
        raise ValueError("Segments per activity must be a positive integer.")
    if isinstance(start_order_index, bool) or start_order_index <= 0:
        raise ValueError("Start order index must be positive.")
    entries: list[NegativePlanEntry] = []
    for repetition in range(1, segments_per_activity + 1):
        for activity in NEGATIVE_ACTIVITIES:
            entries.append(
                NegativePlanEntry(
                    collection_order_index=start_order_index + len(entries),
                    activity=activity,
                    repetition_index=repetition,
                )
            )
    return tuple(entries)


def record_robustness_segment(
    audio_backend: Any,
    *,
    device_index: int,
    sample_rate_hz: float,
    channels: int,
    activity_duration_seconds: float,
    warmup_duration_seconds: float,
    post_activity_tail_seconds: float,
    activity_cue_fn: Callable[[], None],
    activity_end_cue_fn: Callable[[], None],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> np.ndarray[Any, Any]:
    """Record warm-up, a cued activity, and a quiet completion tail."""

    total_duration_seconds = (
        float(warmup_duration_seconds)
        + float(activity_duration_seconds)
        + float(post_activity_tail_seconds)
    )
    try:
        captured = audio_backend.rec(
            max(1, round(float(sample_rate_hz) * total_duration_seconds)),
            samplerate=sample_rate_hz,
            channels=channels,
            dtype=ROBUSTNESS_DTYPE,
            device=device_index,
            blocking=False,
        )
    except Exception as error:
        details = describe_audio_error(error)
        raise RobustnessError(
            details["message"], category=details["category"]
        ) from error
    try:
        sleep_fn(warmup_duration_seconds)
        activity_cue_fn()
        sleep_fn(activity_duration_seconds)
        activity_end_cue_fn()
    except BaseException:
        _stop_robustness_capture(audio_backend)
        raise
    try:
        audio_backend.wait()
    except KeyboardInterrupt:
        _stop_robustness_capture(audio_backend)
        raise
    except Exception as error:
        _stop_robustness_capture(audio_backend)
        details = describe_audio_error(error)
        raise RobustnessError(
            details["message"], category=details["category"]
        ) from error
    try:
        audio = validate_robustness_capture(captured)
    except ValueError as error:
        raise RobustnessError(str(error), category="invalid_capture") from error
    frames = max(1, round(float(sample_rate_hz) * total_duration_seconds))
    if audio.shape != (frames, channels):
        raise RobustnessError(
            "Negative capture shape did not match the requested fixed domain.",
            category="invalid_capture",
        )
    return audio


def _stop_robustness_capture(audio_backend: Any) -> None:
    try:
        audio_backend.stop()
    except Exception:
        pass


def run_guided_robustness_collection(
    audio_backend: Any,
    *,
    device_index: int,
    dataset_root: Path = Path("datasets"),
    parameters: RobustnessCollectionParameters | None = None,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
    countdown_fn: Callable[[Callable[[str], None]], None] | None = None,
    positive_capture_fn: Callable[..., np.ndarray[Any, Any]] | None = None,
    negative_capture_fn: Callable[..., np.ndarray[Any, Any]] | None = None,
    pre_cue_sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], datetime] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Collect one explicit 30-positive/14-negative robustness session."""

    settings = parameters or RobustnessCollectionParameters()
    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn
    clock = (lambda: datetime.now(timezone.utc)) if now_fn is None else now_fn
    sleep_fn = time.sleep if pre_cue_sleep_fn is None else pre_cue_sleep_fn
    positive_record = (
        record_collection_attempt
        if positive_capture_fn is None
        else positive_capture_fn
    )
    negative_record = (
        record_robustness_segment
        if negative_capture_fn is None
        else negative_capture_fn
    )

    inventory, device = _selected_robustness_device(audio_backend, device_index)
    _validate_fixed_capture_domain(audio_backend, device)
    positive_plan = positive_robustness_plan(settings.attempts_per_condition)
    negative_plan = negative_robustness_plan(
        settings.negative_segments_per_activity,
        start_order_index=len(positive_plan) + 1,
    )
    created_at = _utc_timestamp(clock)
    metadata = _robustness_session_metadata(
        inventory=inventory,
        device=device,
        settings=settings,
        positive_plan=positive_plan,
        negative_plan=negative_plan,
    )
    try:
        session = create_robustness_dataset_session(
            Path(dataset_root),
            metadata,
            session_id=session_id,
            created_at_utc=created_at,
        )
    except (OSError, TypeError, ValueError) as error:
        raise RobustnessError(
            f"Could not create robustness dataset: {error}",
            category="persistence_error",
        ) from error

    write_output(f"Phase 3 robustness session: {session.session_id}")
    write_output(
        f"Endpoint {device['index']}: {device['name']} "
        f"({device['host_api']['name']})"
    )
    write_output("Fixed domain: 48000 Hz, 2 channels, float32")
    write_output(
        "Every structurally valid intended tap is retained even if the current "
        "detector would miss it."
    )

    for plan in positive_plan:
        write_output("")
        write_output(
            f"Positive {plan.collection_order_index}/{len(positive_plan)}: "
            f"{plan.zone} {plan.strength}, attempt "
            f"{plan.attempt_number_within_condition}/{settings.attempts_per_condition}"
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
                duration_seconds=settings.positive_capture_seconds,
                pre_cue_duration_seconds=settings.pre_cue_seconds,
                tap_cue_fn=lambda: write_output("TAP NOW"),
                sleep_fn=sleep_fn,
            )
            capture = _validated_exact_capture(
                capture, settings.positive_capture_seconds
            )
        except DatasetCollectionError as error:
            raise RobustnessError(str(error), category=error.category) from error
        filename = (
            f"{plan.zone.casefold()}_{plan.strength}_"
            f"{plan.attempt_number_within_condition:03d}"
        )
        record_id = f"{session.session_id}-{filename}"
        record_metadata = {
            "session_id": session.session_id,
            "collection_order_index": plan.collection_order_index,
            "intended_zone": plan.zone,
            "intended_strength": plan.strength,
            "attempt_number_within_condition": (
                plan.attempt_number_within_condition
            ),
            "captured_at_utc": captured_at,
            "semantic_label": "intended_desk_tap",
            "detector_acceptance_required_for_storage": False,
            **_capture_metadata(capture, device),
            "guided_cue": _guided_cue_metadata(settings),
        }
        _save_robustness_record(
            session,
            record_id=record_id,
            filename_stem=filename,
            record_type=POSITIVE_RECORD_TYPE,
            capture=capture,
            metadata=record_metadata,
        )
        write_output(f"Saved intended attempt: {filename}.npz")

    write_output("")
    write_output(
        "Negative segments contain no intended desk tap; the activity may still "
        "be loud or impulsive."
    )
    for plan in negative_plan:
        write_output("")
        write_output(
            f"Negative activity {plan.activity}, repetition "
            f"{plan.repetition_index}/{settings.negative_segments_per_activity}: "
            f"remain quiet for {settings.negative_warmup_seconds:g} s, then "
            f"perform only this activity for {settings.negative_segment_seconds:g} s."
        )
        read_input("Press Enter when ready to begin this segment: ")
        started_at = _utc_timestamp(clock)
        capture = negative_record(
            audio_backend,
            device_index=int(device["index"]),
            sample_rate_hz=ROBUSTNESS_SAMPLE_RATE_HZ,
            channels=ROBUSTNESS_CHANNEL_COUNT,
            activity_duration_seconds=settings.negative_segment_seconds,
            warmup_duration_seconds=settings.negative_warmup_seconds,
            post_activity_tail_seconds=(
                settings.negative_post_activity_tail_seconds
            ),
            activity_cue_fn=lambda activity=plan.activity: write_output(
                f"BEGIN {activity.upper()}"
            ),
            activity_end_cue_fn=lambda activity=plan.activity: write_output(
                f"END {activity.upper()} — REMAIN QUIET"
            ),
            sleep_fn=sleep_fn,
        )
        ended_at = _utc_timestamp(clock)
        total_negative_seconds = (
            settings.negative_warmup_seconds + settings.negative_segment_seconds
            + settings.negative_post_activity_tail_seconds
        )
        capture = _validated_exact_capture(
            capture, total_negative_seconds
        )
        warmup_end_frame = round(
            ROBUSTNESS_SAMPLE_RATE_HZ * settings.negative_warmup_seconds
        )
        activity_end_frame = warmup_end_frame + round(
            ROBUSTNESS_SAMPLE_RATE_HZ * settings.negative_segment_seconds
        )
        post_activity_end_frame = activity_end_frame + round(
            ROBUSTNESS_SAMPLE_RATE_HZ
            * settings.negative_post_activity_tail_seconds
        )
        filename = f"{plan.activity}_{plan.repetition_index:03d}"
        record_id = f"{session.session_id}-{filename}"
        record_metadata = {
            "session_id": session.session_id,
            "collection_order_index": plan.collection_order_index,
            "activity": plan.activity,
            "repetition_index": plan.repetition_index,
            "segment_started_at_utc": started_at,
            "segment_ended_at_utc": ended_at,
            "warmup_duration_seconds": settings.negative_warmup_seconds,
            "warmup_end_frame_index": warmup_end_frame,
            "activity_start_frame_index": warmup_end_frame,
            "activity_end_frame_index_exclusive": activity_end_frame,
            "activity_duration_seconds": settings.negative_segment_seconds,
            "post_activity_tail_duration_seconds": (
                settings.negative_post_activity_tail_seconds
            ),
            "post_activity_start_frame_index": activity_end_frame,
            "post_activity_end_frame_index_exclusive": post_activity_end_frame,
            "activity_cue": {
                "begin_cue_text": f"BEGIN {plan.activity.upper()}",
                "end_cue_text": f"END {plan.activity.upper()} — REMAIN QUIET",
                "cue_is_audible": False,
                "timing_caution": (
                    "The terminal cues define the guided protocol schedule; they "
                    "do not measure rendering or human reaction time."
                ),
            },
            "semantic_label": "no_intended_desk_tap",
            **_capture_metadata(capture, device),
        }
        _save_robustness_record(
            session,
            record_id=record_id,
            filename_stem=filename,
            record_type=NEGATIVE_RECORD_TYPE,
            capture=capture,
            metadata=record_metadata,
        )
        write_output(f"Saved negative activity segment: {filename}.npz")

    total_records = len(positive_plan) + len(negative_plan)
    write_output("")
    write_output(
        f"Robustness collection complete: {total_records} records in "
        f"{session.directory}"
    )
    return {
        "status": "completed",
        "session_id": session.session_id,
        "session_directory": str(session.directory),
        "manifest_path": str(session.manifest_path),
        "positive_attempts": len(positive_plan),
        "negative_segments": len(negative_plan),
        "sample_rate_hz": ROBUSTNESS_SAMPLE_RATE_HZ,
        "channel_count": ROBUSTNESS_CHANNEL_COUNT,
    }


def extract_descriptive_tapness_metrics(
    candidate_window: Any,
    *,
    sample_rate_hz: float,
    onset_offset_frames: int,
    learned_noise_floor_rms: float | None = None,
) -> dict[str, Any]:
    """Compute fixed descriptive impact-shape metrics without gating.

    A float64 processing copy is DC-centered per channel using the immediate
    pre-onset region. Pooled squared power prevents channel cancellation. The
    returned values are research descriptors only and contain no accept/reject
    threshold.
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
        "metric_schema_version": 1,
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
        "decision_use": "descriptive_only_no_tap_validation",
    }


def replay_robustness_dataset(
    session_path: Path,
    *,
    baseline_path: Path | None = None,
    detector_factory: Callable[[], StreamingTapDetector] = StreamingTapDetector,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Replay validated local evidence through the unchanged current detector."""

    dataset = load_robustness_dataset(Path(session_path))
    baseline = load_frozen_baseline(Path(baseline_path)) if baseline_path else None
    if baseline is not None:
        _validate_baseline_for_robustness(baseline, dataset)
    clock = (lambda: datetime.now(timezone.utc)) if now_fn is None else now_fn

    positive_results = [
        _replay_positive_record(record, detector_factory, baseline)
        for record in dataset.positive_records
    ]
    negative_results = [
        _replay_negative_record(record, detector_factory)
        for record in dataset.negative_records
    ]
    positive_summary = _summarize_positive_replay(positive_results, baseline)
    negative_summary = _summarize_negative_replay(negative_results)
    completeness = _collection_completeness(dataset)
    report = {
        "report_schema_version": ROBUSTNESS_REPORT_SCHEMA_VERSION,
        "report_type": ROBUSTNESS_REPORT_TYPE,
        "project_phase": "3B.0",
        "generated_at_utc": _utc_timestamp(clock),
        "session_id": dataset.session["session_id"],
        "session_path": str(dataset.directory),
        "dataset_integrity": {
            "status": "passed",
            "record_count": len(dataset.records),
            "positive_record_count": len(dataset.positive_records),
            "negative_record_count": len(dataset.negative_records),
            "dataset_files_modified": False,
        },
        "collection_completeness": completeness,
        "detector_protocol": {
            "implementation": "current unchanged StreamingTapDetector",
            "new_stage1_logic_used": False,
            "stage2_validator_used": False,
            "each_saved_capture_is_a_separate_continuity_epoch": True,
            "positive_candidate_recall_definition": (
                "fraction of intended attempts with at least one candidate-start "
                "diagnostic whose onset is inside the predeclared cue-relative "
                "association interval"
            ),
            "positive_association_boundary_rule": "start inclusive, end exclusive",
            "negative_rate_denominator": "labeled activity interval only",
        },
        "tapness_feature_definitions": tapness_feature_definitions(),
        "positive": {
            "summary": positive_summary,
            "attempts": positive_results,
        },
        "negative": {
            "summary": negative_summary,
            "segments": negative_results,
        },
        "frozen_spatial_baseline": (
            _baseline_report_metadata(baseline, baseline_path)
            if baseline is not None
            else {"applied": False}
        ),
        "limitations": [
            "This is offline development replay, not an untouched Phase 3B validation.",
            "Descriptive tapness metrics do not accept or reject candidates.",
            "Guided positive captures do not contain an independently measured physical impact timestamp.",
            "The cue-relative positive association interval is an engineering protocol, not a measured impact timestamp.",
            "Events outside a positive association interval do not satisfy intended-tap recall or spatial correctness.",
            "Negative activity means no intended desk tap; it does not mean silence.",
            "Negative warm-up and post-activity-tail events are reported separately and excluded from activity rates.",
            "No production detector threshold, geometry, floor adaptation, refractory behavior, or frozen spatial rule was changed.",
        ],
        "privacy": {
            "processing": "local_offline",
            "network_upload_performed": False,
            "waveforms_embedded_in_report": False,
        },
    }
    json.dumps(report, allow_nan=False)
    return report


def write_robustness_report(report: Mapping[str, Any], path: Path) -> Path:
    """Exclusively write a waveform-free robustness replay report."""

    target = Path(path)
    if target.suffix.casefold() != ".json":
        raise ValueError("Robustness report path must end in .json.")
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as output:
        output.write(serialized)
        output.write("\n")
    return target.resolve()


def format_robustness_summary(report: Mapping[str, Any]) -> str:
    """Render a concise evidence-scoped terminal replay summary."""

    positive = report["positive"]["summary"]
    negative = report["negative"]["summary"]
    spatial = positive["conditional_frozen_spatial"]
    lines = [
        "DeskSense Phase 3B.0 offline robustness replay",
        f"Session: {report['session_id']}",
        (
            "Collection complete: "
            f"{str(report['collection_completeness']['collection_complete']).lower()}"
        ),
        (
            "Positive Stage 1 candidate-start recall: "
            f"{positive['attempts_with_associated_candidate_start']}/"
            f"{positive['total_intended_attempts']} "
            f"({positive['stage1_candidate_start_recall'] * 100.0:.2f}%)"
        ),
        (
            "Positive associated completions: "
            f"attempts={positive['attempts_with_associated_completed_result']}, "
            f"detected={positive['associated_completed_detector_detections']}, "
            f"rejected={positive['associated_completed_detector_rejections']}"
        ),
        (
            "Positive extraneous clip events: "
            f"starts={positive['extraneous_candidate_starts']}, "
            f"completed={positive['extraneous_completed_events']}"
        ),
    ]
    if spatial["applied"]:
        lines.append(
            "Conditional frozen spatial result: "
            f"{spatial['correct_count']}/{spatial['classified_count']} correct "
            "among classified detected positive events"
        )
    else:
        lines.append("Conditional frozen spatial result: not requested")
    lines.extend(
        [
            (
                "Negative evidence: "
                f"{negative['labeled_activity_duration_seconds']:.2f} labeled s, "
                f"candidate starts={negative['activity_associated_candidate_starts']}, "
                "completed false events="
                f"{negative['activity_associated_completed_detections']}"
            ),
            (
                "Negative rates: "
                "candidate starts="
                f"{negative['candidate_start_rate_per_labeled_minute']:.3f}/min, "
                "completed false events="
                f"{negative['completed_false_event_rate_per_labeled_minute']:.3f}/min"
            ),
            (
                "Negative warm-up events (excluded from rates): "
                f"starts={negative['warmup_candidate_starts']}, "
                f"detected={negative['warmup_completed_detections']}, "
                f"rejected={negative['warmup_completed_rejections']}"
            ),
            (
                "Negative completion-tail events (excluded from rates): "
                f"starts={negative['post_activity_candidate_starts']}, "
                f"detected={negative['post_activity_completed_detections']}, "
                f"rejected={negative['post_activity_completed_rejections']}"
            ),
            "Tapness metrics are descriptive only; no Stage 2 gate was applied.",
        ]
    )
    return "\n".join(lines)


def tapness_feature_definitions() -> dict[str, str]:
    return {
        "pre_onset_rms": "Pooled RMS over up to 12 ms immediately before detector onset.",
        "impact_window_rms": "Maximum pooled RMS among 5 ms frames in the first 25 ms from onset.",
        "impact_peak_absolute": "Maximum absolute sample across both channels in the first 25 ms from onset.",
        "onset_contrast_rms_ratio": "Impact-window RMS divided by the larger of pre-onset RMS, learned floor, and a numerical floor.",
        "peak_dominant_contrast_ratio": "Impact peak divided by the same reference floor; descriptive, with no Holo multiplier or decision threshold.",
        "effective_energy_duration_seconds": "Post-onset pooled energy divided by maximum post-onset pooled instantaneous power and sample rate; threshold-free energy-equivalent duration.",
        "early_energy_fraction": "Fraction of post-onset multichannel energy occurring in the first 25 ms.",
        "late_to_impact_rms_ratio": "Pooled RMS from 40 ms after onset to window end divided by impact-window RMS.",
        "strong_sample_fraction": "Fraction of impact-region frames whose maximum channel magnitude is at least 50% of impact peak.",
    }


def _replay_positive_record(
    record: RobustnessRecord,
    detector_factory: Callable[[], StreamingTapDetector],
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    detector = detector_factory()
    results = tuple(detector.process_chunk(record.capture))
    starts = tuple(detector.drain_candidate_start_diagnostics())
    snapshot = detector.diagnostic_snapshot()
    association_start, association_end = _positive_association_bounds(
        record.metadata
    )
    start_records = [
        _candidate_start_report(
            start,
            _interval_relation(
                start.onset_frame_index,
                association_start,
                association_end,
                inside="associated",
                before="pre_association",
                after="post_association",
            ),
        )
        for start in starts
    ]
    events = []
    for result in results:
        relation = _interval_relation(
            result.onset_frame_index,
            association_start,
            association_end,
            inside="associated",
            before="pre_association",
            after="post_association",
        )
        associated = relation == "associated"
        events.append(
            _completed_event_report(
                result,
                baseline if associated else None,
                record.metadata["intended_zone"] if associated else None,
                interval_relation=relation,
            )
        )
    associated_starts = [
        item for item in start_records if item["interval_relation"] == "associated"
    ]
    associated_events = [
        item for item in events if item["interval_relation"] == "associated"
    ]
    extraneous_starts = len(start_records) - len(associated_starts)
    extraneous_events = len(events) - len(associated_events)
    return {
        "record_id": record.metadata["record_id"],
        "collection_order_index": record.metadata["collection_order_index"],
        "intended_zone": record.metadata["intended_zone"],
        "intended_strength": record.metadata["intended_strength"],
        "attempt_number_within_condition": record.metadata[
            "attempt_number_within_condition"
        ],
        "association_start_frame_index_inclusive": association_start,
        "association_end_frame_index_exclusive": association_end,
        "stage1_candidate_started": bool(associated_starts),
        "associated_candidate_start_count": len(associated_starts),
        "associated_completed_result": bool(associated_events),
        "associated_completed_result_count": len(associated_events),
        "associated_completed_detection_count": sum(
            item["status"] == "detected" for item in associated_events
        ),
        "associated_completed_rejection_count": sum(
            item["status"] == "rejected" for item in associated_events
        ),
        "all_candidate_start_count": len(start_records),
        "all_completed_result_count": len(events),
        "extraneous_candidate_start_count": extraneous_starts,
        "extraneous_completed_event_count": extraneous_events,
        "candidate_starts": start_records,
        "completed_events": events,
        "closest_armed_block": (
            asdict(snapshot.closest_armed_block)
            if snapshot.closest_armed_block is not None
            else None
        ),
        "detector_counters": asdict(snapshot.cumulative_counters),
    }


def _replay_negative_record(
    record: RobustnessRecord,
    detector_factory: Callable[[], StreamingTapDetector],
) -> dict[str, Any]:
    detector = detector_factory()
    results = tuple(detector.process_chunk(record.capture))
    starts = tuple(detector.drain_candidate_start_diagnostics())
    activity_start = int(record.metadata["activity_start_frame_index"])
    activity_end = int(record.metadata["activity_end_frame_index_exclusive"])
    start_records = [
        _candidate_start_report(
            start,
            _interval_relation(
                start.onset_frame_index,
                activity_start,
                activity_end,
                inside="activity",
                before="warmup",
                after="post_activity",
            ),
        )
        for start in starts
    ]
    events = [
        _completed_event_report(
            result,
            None,
            None,
            interval_relation=_interval_relation(
                result.onset_frame_index,
                activity_start,
                activity_end,
                inside="activity",
                before="warmup",
                after="post_activity",
            ),
        )
        for result in results
    ]
    activity_starts = [
        item for item in start_records if item["interval_relation"] == "activity"
    ]
    warmup_starts = [
        item for item in start_records if item["interval_relation"] == "warmup"
    ]
    activity_events = [
        item for item in events if item["interval_relation"] == "activity"
    ]
    warmup_events = [
        item for item in events if item["interval_relation"] == "warmup"
    ]
    return {
        "record_id": record.metadata["record_id"],
        "collection_order_index": record.metadata["collection_order_index"],
        "activity": record.metadata["activity"],
        "repetition_index": record.metadata["repetition_index"],
        "stored_capture_duration_seconds": float(
            record.capture.shape[0] / ROBUSTNESS_SAMPLE_RATE_HZ
        ),
        "warmup_duration_seconds": float(record.metadata["warmup_duration_seconds"]),
        "labeled_activity_duration_seconds": float(
            record.metadata["activity_duration_seconds"]
        ),
        "post_activity_tail_duration_seconds": float(
            record.metadata["post_activity_tail_duration_seconds"]
        ),
        "activity_start_frame_index": activity_start,
        "activity_end_frame_index_exclusive": activity_end,
        "post_activity_start_frame_index": int(
            record.metadata["post_activity_start_frame_index"]
        ),
        "post_activity_end_frame_index_exclusive": int(
            record.metadata["post_activity_end_frame_index_exclusive"]
        ),
        "activity_candidate_start_count": len(activity_starts),
        "activity_completed_detection_count": sum(
            item["status"] == "detected" for item in activity_events
        ),
        "activity_completed_rejection_count": sum(
            item["status"] == "rejected" for item in activity_events
        ),
        "warmup_candidate_start_count": len(warmup_starts),
        "warmup_completed_detection_count": sum(
            item["status"] == "detected" for item in warmup_events
        ),
        "warmup_completed_rejection_count": sum(
            item["status"] == "rejected" for item in warmup_events
        ),
        "post_activity_candidate_start_count": sum(
            item["interval_relation"] == "post_activity" for item in start_records
        ),
        "post_activity_completed_event_count": sum(
            item["interval_relation"] == "post_activity" for item in events
        ),
        "post_activity_completed_detection_count": sum(
            item["interval_relation"] == "post_activity"
            and item["status"] == "detected"
            for item in events
        ),
        "post_activity_completed_rejection_count": sum(
            item["interval_relation"] == "post_activity"
            and item["status"] == "rejected"
            for item in events
        ),
        "candidate_starts": start_records,
        "completed_events": events,
        "semantic_label": "no_intended_desk_tap",
    }


def _candidate_start_report(start: Any, interval_relation: str) -> dict[str, Any]:
    return {
        "stream_epoch": int(start.stream_epoch),
        "onset_frame_index": int(start.onset_frame_index),
        "interval_relation": interval_relation,
        "gate_evidence": asdict(start.block),
    }


def _completed_event_report(
    result: DetectionResult,
    baseline: Mapping[str, Any] | None,
    intended_zone: str | None,
    *,
    interval_relation: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": result.status,
        "interval_relation": interval_relation,
        "rejection_reasons": list(result.rejection_reasons),
        "onset_frame_index": result.onset_frame_index,
        "center_frame_index": result.center_frame_index,
        "window_start_frame_index": result.window_start_frame_index,
        "window_end_frame_index_exclusive": result.window_end_frame_index_exclusive,
        "finalized_after_frame_index_exclusive": (
            result.finalized_after_frame_index_exclusive
        ),
        "candidate_window_retained_in_report": False,
        "detector_metrics": dict(result.metrics),
        "descriptive_tapness_metrics": None,
        "frozen_spatial_prediction": None,
    }
    candidate = result.candidate_window
    if (
        candidate is not None
        and result.window_start_frame_index is not None
        and 0
        <= result.onset_frame_index - result.window_start_frame_index
        < candidate.shape[0]
    ):
        onset_offset = result.onset_frame_index - result.window_start_frame_index
        report["descriptive_tapness_metrics"] = extract_descriptive_tapness_metrics(
            candidate,
            sample_rate_hz=ROBUSTNESS_SAMPLE_RATE_HZ,
            onset_offset_frames=onset_offset,
            learned_noise_floor_rms=result.metrics.get("learned_noise_floor_rms"),
        )
    if baseline is not None and intended_zone is not None and result.status == "detected":
        features = extract_two_channel_features(candidate)
        feature_value = features.get(PRIMARY_FEATURE_NAME)
        if feature_value is not None:
            classifier = baseline["classifier"]
            decision = classify_peak_ratio_value(
                feature_value,
                classifier["threshold_db"],
                classifier["lower_feature_zone"],
                classifier["higher_feature_zone"],
            )
            predicted = str(decision["predicted_label"])
            report["frozen_spatial_prediction"] = {
                "actual_zone": intended_zone,
                "predicted_zone": predicted,
                "correct": predicted == intended_zone,
                "feature_value_db": float(feature_value),
                "threshold_db": float(classifier["threshold_db"]),
                "absolute_margin_db": float(decision["absolute_margin_db"]),
                "baseline_refit": False,
            }
    return report


def _summarize_positive_replay(
    attempts: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    total = len(attempts)
    with_start = sum(bool(item["stage1_candidate_started"]) for item in attempts)
    with_completion = sum(
        bool(item["associated_completed_result"]) for item in attempts
    )
    by_zone = {
        zone: _positive_group_summary(
            [item for item in attempts if item["intended_zone"] == zone]
        )
        for zone in POSITIVE_ZONES
    }
    by_strength = {
        strength: _positive_group_summary(
            [item for item in attempts if item["intended_strength"] == strength]
        )
        for strength in POSITIVE_STRENGTHS
    }
    predictions = [
        event["frozen_spatial_prediction"]
        for attempt in attempts
        for event in attempt["completed_events"]
        if event["interval_relation"] == "associated"
        and event["frozen_spatial_prediction"] is not None
    ]
    return {
        "total_intended_attempts": total,
        "attempts_with_associated_candidate_start": with_start,
        "stage1_candidate_start_recall": (
            float(with_start / total) if total else 0.0
        ),
        "attempts_with_associated_completed_result": with_completion,
        "associated_completion_rate": (
            float(with_completion / total) if total else 0.0
        ),
        "associated_completed_detector_detections": sum(
            int(item["associated_completed_detection_count"]) for item in attempts
        ),
        "associated_completed_detector_rejections": sum(
            int(item["associated_completed_rejection_count"]) for item in attempts
        ),
        "extraneous_candidate_starts": sum(
            int(item["extraneous_candidate_start_count"]) for item in attempts
        ),
        "extraneous_completed_events": sum(
            int(item["extraneous_completed_event_count"]) for item in attempts
        ),
        "by_zone": by_zone,
        "by_strength": by_strength,
        "conditional_frozen_spatial": {
            "applied": baseline is not None,
            "classified_count": len(predictions),
            "correct_count": sum(bool(item["correct"]) for item in predictions),
            "accuracy": (
                float(sum(bool(item["correct"]) for item in predictions) / len(predictions))
                if predictions
                else None
            ),
        },
    }


def _positive_group_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(items)
    with_start = sum(bool(item["stage1_candidate_started"]) for item in items)
    with_completion = sum(bool(item["associated_completed_result"]) for item in items)
    return {
        "intended_attempts": total,
        "attempts_with_associated_candidate_start": with_start,
        "stage1_candidate_start_recall": (
            float(with_start / total) if total else 0.0
        ),
        "attempts_with_associated_completed_result": with_completion,
        "associated_completion_rate": (
            float(with_completion / total) if total else 0.0
        ),
    }


def _summarize_negative_replay(
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        **_negative_group_summary(segments),
        "segment_count": len(segments),
        "per_activity": {
            activity: _negative_group_summary(
                [item for item in segments if item["activity"] == activity]
            )
            for activity in NEGATIVE_ACTIVITIES
        },
    }


def _negative_group_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stored_duration = float(
        sum(float(item["stored_capture_duration_seconds"]) for item in items)
    )
    activity_duration = float(
        sum(float(item["labeled_activity_duration_seconds"]) for item in items)
    )
    warmup_duration = float(
        sum(float(item["warmup_duration_seconds"]) for item in items)
    )
    tail_duration = float(
        sum(float(item["post_activity_tail_duration_seconds"]) for item in items)
    )
    starts = sum(int(item["activity_candidate_start_count"]) for item in items)
    detections = sum(
        int(item["activity_completed_detection_count"]) for item in items
    )
    rejections = sum(
        int(item["activity_completed_rejection_count"]) for item in items
    )
    minutes = activity_duration / 60.0
    return {
        "total_stored_duration_seconds": stored_duration,
        "labeled_activity_duration_seconds": activity_duration,
        "warmup_duration_seconds": warmup_duration,
        "post_activity_tail_duration_seconds": tail_duration,
        "activity_associated_candidate_starts": starts,
        "activity_associated_completed_detections": detections,
        "activity_associated_completed_rejections": rejections,
        "warmup_candidate_starts": sum(
            int(item["warmup_candidate_start_count"]) for item in items
        ),
        "warmup_completed_detections": sum(
            int(item["warmup_completed_detection_count"]) for item in items
        ),
        "warmup_completed_rejections": sum(
            int(item["warmup_completed_rejection_count"]) for item in items
        ),
        "post_activity_candidate_starts": sum(
            int(item["post_activity_candidate_start_count"]) for item in items
        ),
        "post_activity_completed_detections": sum(
            int(item["post_activity_completed_detection_count"]) for item in items
        ),
        "post_activity_completed_rejections": sum(
            int(item["post_activity_completed_rejection_count"]) for item in items
        ),
        "candidate_start_rate_per_labeled_minute": (
            float(starts / minutes) if minutes else 0.0
        ),
        "completed_false_event_rate_per_labeled_minute": (
            float(detections / minutes) if minutes else 0.0
        ),
    }


def _positive_association_bounds(metadata: Mapping[str, Any]) -> tuple[int, int]:
    association = metadata["guided_cue"]["intended_event_association"]
    return (
        int(association["start_frame_index_inclusive"]),
        int(association["end_frame_index_exclusive"]),
    )


def _interval_relation(
    onset_frame_index: int,
    start_frame_index: int,
    end_frame_index_exclusive: int,
    *,
    inside: str,
    before: str,
    after: str,
) -> str:
    if onset_frame_index < start_frame_index:
        return before
    if onset_frame_index < end_frame_index_exclusive:
        return inside
    return after


def _collection_completeness(
    dataset: LoadedRobustnessDataset,
) -> dict[str, Any]:
    positive_design = dataset.session["positive_design"]
    negative_design = dataset.session["negative_design"]
    expected_positive = int(positive_design["requested_attempt_count"])
    expected_negative = int(negative_design["requested_segment_count"])
    observed_positive = len(dataset.positive_records)
    observed_negative = len(dataset.negative_records)
    expected_positive_plan = [
        {
            "collection_order_index": int(item["collection_order_index"]),
            "intended_zone": str(item["zone"]),
            "intended_strength": str(item["strength"]),
            "attempt_number_within_condition": int(
                item["attempt_number_within_condition"]
            ),
        }
        for item in positive_design["plan"]
    ]
    observed_positive_plan = [
        {
            "collection_order_index": int(
                record.metadata["collection_order_index"]
            ),
            "intended_zone": str(record.metadata["intended_zone"]),
            "intended_strength": str(record.metadata["intended_strength"]),
            "attempt_number_within_condition": int(
                record.metadata["attempt_number_within_condition"]
            ),
        }
        for record in dataset.positive_records
    ]
    expected_negative_plan = [
        {
            "collection_order_index": int(item["collection_order_index"]),
            "activity": str(item["activity"]),
            "repetition_index": int(item["repetition_index"]),
        }
        for item in negative_design["plan"]
    ]
    observed_negative_plan = [
        {
            "collection_order_index": int(
                record.metadata["collection_order_index"]
            ),
            "activity": str(record.metadata["activity"]),
            "repetition_index": int(record.metadata["repetition_index"]),
        }
        for record in dataset.negative_records
    ]
    positive_plan_complete = expected_positive_plan == observed_positive_plan
    negative_plan_complete = expected_negative_plan == observed_negative_plan
    return {
        "expected_positive_attempts": expected_positive,
        "observed_positive_attempts": observed_positive,
        "expected_negative_segments": expected_negative,
        "observed_negative_segments": observed_negative,
        "positive_plan_complete": positive_plan_complete,
        "negative_plan_complete": negative_plan_complete,
        "collection_complete": (
            observed_positive == expected_positive
            and observed_negative == expected_negative
            and positive_plan_complete
            and negative_plan_complete
        ),
    }


def _selected_robustness_device(
    audio_backend: Any, device_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = collect_audio_inventory(
        audio_backend, selected_device_index=device_index
    )
    selected = next(
        (
            device
            for device in inventory["input_devices"]
            if device["index"] == inventory["selected_input_device_index"]
        ),
        None,
    )
    if selected is None:
        message = (
            inventory["errors"][0]["message"]
            if inventory["errors"]
            else "No usable input device was selected."
        )
        raise RobustnessError(message, category="device_unavailable")
    return inventory, selected


def _validate_fixed_capture_domain(audio_backend: Any, device: Mapping[str, Any]) -> None:
    if int(device.get("max_input_channels", 0)) < ROBUSTNESS_CHANNEL_COUNT:
        raise RobustnessError(
            "Phase 3 robustness capture requires two input channels.",
            category="unsupported_configuration",
        )
    try:
        audio_backend.check_input_settings(
            device=int(device["index"]),
            channels=ROBUSTNESS_CHANNEL_COUNT,
            dtype=ROBUSTNESS_DTYPE,
            samplerate=ROBUSTNESS_SAMPLE_RATE_HZ,
        )
    except Exception as error:
        details = describe_audio_error(error)
        raise RobustnessError(
            "The selected endpoint does not support the fixed 48000 Hz, "
            f"two-channel float32 robustness domain: {details['message']}",
            category=details["category"],
        ) from error


def _validated_exact_capture(samples: Any, duration_seconds: float) -> np.ndarray[Any, Any]:
    try:
        capture = validate_robustness_capture(samples)
    except ValueError as error:
        raise RobustnessError(str(error), category="invalid_capture") from error
    expected_frames = round(ROBUSTNESS_SAMPLE_RATE_HZ * float(duration_seconds))
    if capture.shape != (expected_frames, ROBUSTNESS_CHANNEL_COUNT):
        raise RobustnessError(
            "Capture did not match the fixed requested frame/channel layout.",
            category="invalid_capture",
        )
    return capture


def _save_robustness_record(session: Any, **kwargs: Any) -> None:
    try:
        session.save_record(**kwargs)
    except (OSError, TypeError, ValueError) as error:
        raise RobustnessError(
            f"Could not persist robustness evidence: {error}",
            category="persistence_error",
        ) from error


def _robustness_session_metadata(
    *,
    inventory: Mapping[str, Any],
    device: Mapping[str, Any],
    settings: RobustnessCollectionParameters,
    positive_plan: Sequence[PositivePlanEntry],
    negative_plan: Sequence[NegativePlanEntry],
) -> dict[str, Any]:
    return {
        "source_mode": "guided_phase3_robustness_evidence_collection",
        "system": system_information(),
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
            "attempts_per_condition": settings.attempts_per_condition,
            "requested_attempt_count": len(positive_plan),
            "capture_duration_seconds": settings.positive_capture_seconds,
            "pre_cue_seconds": settings.pre_cue_seconds,
            "association_duration_seconds": settings.positive_association_seconds,
            "ordering": "repetition-major round robin over strength then zone",
            "plan": [asdict(item) for item in positive_plan],
            "detector_acceptance_required_for_storage": False,
        },
        "negative_design": {
            "activities": list(NEGATIVE_ACTIVITIES),
            "segments_per_activity": settings.negative_segments_per_activity,
            "requested_segment_count": len(negative_plan),
            "warmup_duration_seconds": settings.negative_warmup_seconds,
            "activity_duration_seconds": settings.negative_segment_seconds,
            "post_activity_tail_duration_seconds": (
                settings.negative_post_activity_tail_seconds
            ),
            "stored_capture_duration_seconds": (
                settings.negative_warmup_seconds
                + settings.negative_segment_seconds
                + settings.negative_post_activity_tail_seconds
            ),
            "ordering": "repetition-major round robin over activities",
            "semantic_label": "no_intended_desk_tap",
            "plan": [asdict(item) for item in negative_plan],
        },
        "privacy": {
            "waveforms_retained_locally": True,
            "network_upload_performed": False,
            "ordinary_sense_audio_saved": False,
            "note": "Robustness waveforms remain local under the ignored datasets root.",
        },
    }


def _capture_metadata(
    capture: np.ndarray[Any, Any], device: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "device": _endpoint_metadata(device),
        "sample_rate_hz": ROBUSTNESS_SAMPLE_RATE_HZ,
        "channel_count": ROBUSTNESS_CHANNEL_COUNT,
        "capture_dtype": ROBUSTNESS_DTYPE,
        "capture_frames": int(capture.shape[0]),
        "capture_duration_seconds": float(
            capture.shape[0] / ROBUSTNESS_SAMPLE_RATE_HZ
        ),
    }


def _guided_cue_metadata(settings: RobustnessCollectionParameters) -> dict[str, Any]:
    cue_frame = round(ROBUSTNESS_SAMPLE_RATE_HZ * settings.pre_cue_seconds)
    association_end = cue_frame + round(
        ROBUSTNESS_SAMPLE_RATE_HZ * settings.positive_association_seconds
    )
    return {
        "cue_text": "TAP NOW",
        "cue_is_audible": False,
        "capture_started_before_cue": True,
        "intended_pre_cue_seconds": settings.pre_cue_seconds,
        "intended_cue_offset_frames": cue_frame,
        "intended_event_association": {
            "start_frame_index_inclusive": cue_frame,
            "end_frame_index_exclusive": association_end,
            "duration_seconds": settings.positive_association_seconds,
            "semantics": (
                "Fixed guided-development association interval after the visual "
                "cue; not a measured physical impact timestamp."
            ),
        },
        "timing_caution": (
            "The visual cue schedule does not measure terminal rendering or human reaction time."
        ),
    }


def _endpoint_metadata(device: Mapping[str, Any]) -> dict[str, Any]:
    host_api = device.get("host_api", {})
    return {
        "index_at_collection": int(device["index"]),
        "name": str(device["name"]),
        "host_api": {
            "index": host_api.get("index"),
            "name": str(host_api.get("name", "Unknown host API")),
        },
    }


def _validate_baseline_for_robustness(
    baseline: Mapping[str, Any], dataset: LoadedRobustnessDataset
) -> None:
    compatibility = baseline["source_development_dataset"]["compatibility"]
    domain = dataset.session["capture_domain"]
    if (
        float(compatibility["sample_rate_hz"]) != float(domain["sample_rate_hz"])
        or int(compatibility["channel_count"]) != int(domain["channel_count"])
        or int(compatibility["tap_window_frames"]) != 9_600
    ):
        raise RobustnessError(
            "Frozen baseline is incompatible with the robustness replay domain."
        )
    endpoint = dataset.session["selected_endpoint"]
    if (
        compatibility["endpoint_name"] != endpoint["name"]
        or compatibility["host_api_name"] != endpoint["host_api"]["name"]
    ):
        raise RobustnessError(
            "Frozen baseline endpoint identity does not match the robustness session."
        )


def _baseline_report_metadata(
    baseline: Mapping[str, Any], baseline_path: Path | None
) -> dict[str, Any]:
    classifier = baseline["classifier"]
    return {
        "applied": True,
        "baseline_path": str(Path(baseline_path).resolve()) if baseline_path else None,
        "feature_name": baseline["feature"]["name"],
        "threshold_db": classifier["threshold_db"],
        "lower_feature_zone": classifier["lower_feature_zone"],
        "higher_feature_zone": classifier["higher_feature_zone"],
        "refit_performed": False,
        "normalization_fitted": False,
    }


def _pooled_rms(samples: np.ndarray[Any, Any]) -> float:
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


def _utc_timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Robustness timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat()
