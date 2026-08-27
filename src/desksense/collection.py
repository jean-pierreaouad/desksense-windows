"""Guided, labeled microphone capture for DeskSense Phase 2A."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desksense.characterization import (
    TRANSIENT_ENERGY_WINDOW_SECONDS,
    analyze_channel_activity,
    find_strongest_transient_window,
)
from desksense.dataset import DatasetSession, create_dataset_session
from desksense.diagnostics import (
    RecordingConfigurationError,
    collect_audio_inventory,
    describe_audio_error,
    find_characterization_configuration,
    system_information,
)


DEFAULT_ZONES = ("LEFT", "RIGHT")
DEFAULT_SAMPLES_PER_ZONE = 20
DEFAULT_CAPTURE_DURATION_SECONDS = 1.5
DEFAULT_PRE_CUE_DURATION_SECONDS = 0.200
DEFAULT_TAP_WINDOW_SECONDS = 0.200
DEFAULT_CLIPPING_THRESHOLD = 0.98
DEFAULT_MINIMUM_TRANSIENT_RMS = 4.0 / 32_768.0
DEFAULT_MINIMUM_TRANSIENT_TO_BACKGROUND_RMS_RATIO = 1.5
_ZONE_LABEL = re.compile(r"^[A-Z0-9][A-Z0-9_-]*$")


class DatasetCollectionError(RuntimeError):
    """A setup, hardware, or persistence failure that should stop collection."""

    def __init__(self, message: str, *, category: str = "collection_error") -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class CollectionParameters:
    """Reviewable Phase 2A capture and quality settings."""

    capture_duration_seconds: float = DEFAULT_CAPTURE_DURATION_SECONDS
    pre_cue_duration_seconds: float = DEFAULT_PRE_CUE_DURATION_SECONDS
    tap_window_seconds: float = DEFAULT_TAP_WINDOW_SECONDS
    transient_energy_window_seconds: float = TRANSIENT_ENERGY_WINDOW_SECONDS
    clipping_threshold: float = DEFAULT_CLIPPING_THRESHOLD
    minimum_transient_rms: float = DEFAULT_MINIMUM_TRANSIENT_RMS
    minimum_transient_to_background_rms_ratio: float = (
        DEFAULT_MINIMUM_TRANSIENT_TO_BACKGROUND_RMS_RATIO
    )

    def __post_init__(self) -> None:
        positive_values = {
            "capture duration": self.capture_duration_seconds,
            "pre-cue duration": self.pre_cue_duration_seconds,
            "tap window duration": self.tap_window_seconds,
            "transient energy window duration": self.transient_energy_window_seconds,
            "clipping threshold": self.clipping_threshold,
            "minimum transient RMS": self.minimum_transient_rms,
            "minimum transient/background RMS ratio": (
                self.minimum_transient_to_background_rms_ratio
            ),
        }
        for label, value in positive_values.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{label.capitalize()} must be positive and finite.")
        if self.tap_window_seconds >= self.capture_duration_seconds:
            raise ValueError("Tap window duration must be shorter than capture duration.")
        if self.pre_cue_duration_seconds >= self.capture_duration_seconds:
            raise ValueError("Pre-cue duration must be shorter than capture duration.")
        if self.clipping_threshold > 1.0:
            raise ValueError("Clipping threshold must not exceed normalized full scale.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "capture_duration_seconds": float(self.capture_duration_seconds),
            "pre_cue_duration_seconds": float(self.pre_cue_duration_seconds),
            "nominal_post_cue_duration_seconds": float(
                self.capture_duration_seconds - self.pre_cue_duration_seconds
            ),
            "cue_text": "TAP NOW",
            "capture_starts_before_cue": True,
            "cue_is_audible": False,
            "cue_timing_note": (
                "The cue is scheduled after the configured pre-cue interval "
                "while recording is active. OS scheduling, terminal rendering, "
                "and human reaction latency are not measured."
            ),
            "tap_window_seconds": float(self.tap_window_seconds),
            "tap_window_is_exploratory": True,
            "tap_window_note": (
                f"The configured {self.tap_window_seconds * 1000.0:g} ms window "
                "preserves short pre/post-event context; it is not claimed to "
                "be optimal or a final tap detector."
            ),
            "transient_energy_window_seconds": float(
                self.transient_energy_window_seconds
            ),
            "clipping_peak_absolute_threshold": float(
                self.clipping_threshold
            ),
            "minimum_transient_energy_window_rms": float(
                self.minimum_transient_rms
            ),
            "minimum_transient_rms_basis": (
                "Four normalized 16-bit LSBs, used only as a conservative "
                "absolute floor together with channel activity and contrast."
            ),
            "minimum_transient_to_background_rms_ratio": float(
                self.minimum_transient_to_background_rms_ratio
            ),
            "minimum_transient_to_background_db": float(
                20.0
                * math.log10(
                    self.minimum_transient_to_background_rms_ratio
                )
            ),
            "lag_used_for_quality_decision": False,
        }


@dataclass(frozen=True)
class CaptureQualityAssessment:
    """JSON-safe quality evidence plus the in-memory accepted tap window."""

    accepted: bool
    reasons: tuple[dict[str, str], ...]
    metrics: dict[str, Any]
    transient: dict[str, Any] | None
    tap_window: np.ndarray[Any, Any] | None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": "accepted" if self.accepted else "rejected",
            "reasons": [dict(reason) for reason in self.reasons],
            "metrics": self.metrics,
            "transient": self.transient,
        }


def alternating_collection_order(
    zones: Sequence[str], samples_per_zone: int
) -> tuple[str, ...]:
    """Return a round-robin target order with normalized zone labels."""

    normalized_zones = normalize_zones(zones)
    if (
        isinstance(samples_per_zone, bool)
        or not isinstance(samples_per_zone, int)
        or samples_per_zone <= 0
    ):
        raise ValueError("Samples per zone must be a positive integer.")
    return tuple(
        zone
        for _sample_number in range(samples_per_zone)
        for zone in normalized_zones
    )


def normalize_zones(zones: Sequence[str]) -> tuple[str, ...]:
    """Normalize labels while keeping filenames portable on Windows."""

    normalized: list[str] = []
    for raw_zone in zones:
        if not isinstance(raw_zone, str):
            raise ValueError("Zone labels must be strings.")
        zone = raw_zone.strip().upper()
        if not _ZONE_LABEL.fullmatch(zone):
            raise ValueError(
                "Zone labels must use only letters, numbers, '_' or '-', and "
                "must start with a letter or number."
            )
        if zone in normalized:
            raise ValueError("Zone labels must be unique.")
        normalized.append(zone)
    if not normalized:
        raise ValueError("At least one zone is required.")
    return tuple(normalized)


def assess_capture_quality(
    samples: Any,
    sample_rate_hz: float,
    *,
    expected_channels: int | None = None,
    parameters: CollectionParameters | None = None,
) -> CaptureQualityAssessment:
    """Apply conservative, exploratory checks and extract an exact tap window."""

    settings = parameters or CollectionParameters()
    thresholds = settings.to_metadata()
    try:
        sample_rate = float(sample_rate_hz)
    except (TypeError, ValueError, OverflowError):
        sample_rate = math.nan
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        return _invalid_assessment(
            "invalid_sample_rate",
            "The capture sample rate is not positive and finite.",
            thresholds,
        )

    if expected_channels is not None and (
        isinstance(expected_channels, bool)
        or not isinstance(expected_channels, int)
        or expected_channels <= 0
    ):
        return _invalid_assessment(
            "invalid_expected_channel_count",
            "Expected channel count must be a positive integer.",
            thresholds,
        )

    try:
        audio = np.asarray(samples, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        return _invalid_assessment(
            "invalid_capture_data",
            "Captured audio could not be converted to float32 numeric data.",
            thresholds,
        )
    if audio.ndim != 2:
        return _invalid_assessment(
            "invalid_capture_shape",
            "Captured audio must be a two-dimensional frames-by-channels array.",
            thresholds,
            shape=list(audio.shape),
        )
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        return _invalid_assessment(
            "empty_capture",
            "Captured audio contains no usable frames or channels.",
            thresholds,
            shape=[int(value) for value in audio.shape],
        )
    if expected_channels is not None and audio.shape[1] != expected_channels:
        return _invalid_assessment(
            "unexpected_channel_count",
            (
                f"Capture returned {audio.shape[1]} channel(s), but "
                f"{expected_channels} were requested."
            ),
            thresholds,
            shape=[int(value) for value in audio.shape],
        )
    if not np.all(np.isfinite(audio)):
        nonfinite_count = int(np.size(audio) - np.count_nonzero(np.isfinite(audio)))
        return _invalid_assessment(
            "non_finite_capture",
            "Captured audio contains NaN or infinite sample values.",
            thresholds,
            shape=[int(value) for value in audio.shape],
            nonfinite_sample_count=nonfinite_count,
        )

    audio = np.ascontiguousarray(audio)
    frame_count, channel_count = audio.shape
    tap_window_frames = max(1, round(sample_rate * settings.tap_window_seconds))
    reasons: list[dict[str, str]] = []
    raw_peak = float(np.max(np.abs(audio)))
    clipping_mask = np.abs(audio) >= settings.clipping_threshold
    clipped_sample_count = int(np.count_nonzero(clipping_mask))
    if clipped_sample_count:
        reasons.append(
            _reason(
                "near_clipping",
                "At least one sample reached the conservative near-clipping threshold.",
            )
        )

    per_channel, activity = analyze_channel_activity(audio)
    active_indexes = [
        index
        for index, channel in enumerate(per_channel)
        if not channel["effectively_inactive"]
    ]
    if not active_indexes:
        reasons.append(
            _reason(
                "signal_too_weak",
                "No channel contained a meaningful varying signal under the "
                "exploratory activity heuristic.",
            )
        )
    inactive_required_channels = (
        [
            channel_number
            for channel_number in range(1, expected_channels + 1)
            if channel_number - 1 not in active_indexes
        ]
        if expected_channels is not None
        else []
    )
    if inactive_required_channels:
        channel_list = ", ".join(
            str(channel) for channel in inactive_required_channels
        )
        reasons.append(
            _reason(
                "inactive_required_channel",
                f"Required channel(s) {channel_list} were effectively inactive "
                "under the existing channel-activity heuristic.",
            )
        )

    metrics: dict[str, Any] = {
        "frame_count": int(frame_count),
        "channel_count": int(channel_count),
        "captured_duration_seconds": float(frame_count / sample_rate),
        "raw_peak_absolute": raw_peak,
        "clipped_sample_count": clipped_sample_count,
        "clipped_sample_fraction": float(clipped_sample_count / audio.size),
        "active_channels": [index + 1 for index in active_indexes],
        "expected_channel_count": expected_channels,
        "inactive_required_channels": inactive_required_channels,
        "all_expected_channels_active": (
            not inactive_required_channels
            if expected_channels is not None
            else None
        ),
        "per_channel": per_channel,
        "activity_assessment": activity,
        "thresholds": thresholds,
    }

    if frame_count < tap_window_frames:
        reasons.append(
            _reason(
                "insufficient_tap_window",
                "The capture is shorter than the requested tap-centered window.",
            )
        )
        metrics["requested_tap_window_frames"] = int(tap_window_frames)

    transient: dict[str, Any] | None = None
    tap_window: np.ndarray[Any, Any] | None = None
    if active_indexes:
        transient = find_strongest_transient_window(
            audio,
            sample_rate,
            active_channel_indexes=active_indexes,
            energy_window_seconds=settings.transient_energy_window_seconds,
            analysis_window_seconds=settings.tap_window_seconds,
        )
        center = int(transient["center_sample"])
        exact_start = center - tap_window_frames // 2
        exact_end = exact_start + tap_window_frames
        transient["exact_tap_window_start_sample"] = exact_start
        transient["exact_tap_window_end_sample_exclusive"] = exact_end
        transient["exact_tap_window_duration_samples"] = int(tap_window_frames)
        transient["exact_tap_window_duration_seconds"] = float(
            tap_window_frames / sample_rate
        )
        boundary_lost = exact_start < 0 or exact_end > frame_count
        transient["exact_tap_window_available"] = not boundary_lost
        if boundary_lost:
            reasons.append(
                _reason(
                    "transient_too_close_to_boundary",
                    "The strongest transient is too close to a capture boundary "
                    "for the full centered tap window.",
                )
            )
        else:
            tap_window = np.ascontiguousarray(audio[exact_start:exact_end])

        transient_rms = math.sqrt(
            max(0.0, float(transient["energy_window_mean_square"]))
        )
        metrics["transient_energy_window_rms"] = float(transient_rms)
        if transient_rms < settings.minimum_transient_rms:
            reasons.append(
                _reason(
                    "signal_too_weak",
                    "The strongest local-energy region is below the conservative "
                    "absolute signal floor.",
                )
            )

        background_rms: float | None = None
        contrast_ratio: float | None = None
        contrast_db: float | None = None
        if not boundary_lost:
            selected = np.asarray(audio[:, active_indexes], dtype=np.float64)
            selected -= np.mean(selected, axis=0, dtype=np.float64)
            background = np.concatenate(
                (selected[:exact_start], selected[exact_end:]), axis=0
            )
            if background.size:
                background_rms = float(np.sqrt(np.mean(np.square(background))))
                numerical_floor = (
                    64.0
                    * np.finfo(np.float64).eps
                    * max(1.0, transient_rms)
                )
                if background_rms > numerical_floor:
                    contrast_ratio = float(transient_rms / background_rms)
                    contrast_db = float(20.0 * math.log10(contrast_ratio))
                    if (
                        contrast_ratio
                        < settings.minimum_transient_to_background_rms_ratio
                    ):
                        reasons.append(
                            _reason(
                                "no_distinct_transient",
                                "The strongest local-energy region is not clearly "
                                "above the surrounding background.",
                            )
                        )
        metrics["background_rms_outside_tap_window"] = background_rms
        metrics["transient_to_background_rms_ratio"] = contrast_ratio
        metrics["transient_to_background_db"] = contrast_db

    return CaptureQualityAssessment(
        accepted=not reasons,
        reasons=_deduplicate_reasons(reasons),
        metrics=metrics,
        transient=transient,
        tap_window=tap_window,
    )


def record_collection_attempt(
    audio_backend: Any,
    *,
    device_index: int,
    sample_rate_hz: float,
    channels: int,
    duration_seconds: float,
    pre_cue_duration_seconds: float,
    tap_cue_fn: Callable[[], None],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> np.ndarray[Any, Any]:
    """Start capture, run a silent pre-cue interval, then emit the tap cue."""

    frames = max(1, round(sample_rate_hz * duration_seconds))
    try:
        captured = audio_backend.rec(
            frames,
            samplerate=sample_rate_hz,
            channels=channels,
            dtype="float32",
            device=device_index,
            blocking=False,
        )
    except Exception as error:
        details = describe_audio_error(error)
        raise DatasetCollectionError(
            details["message"], category=details["category"]
        ) from error

    try:
        sleep_fn(pre_cue_duration_seconds)
        tap_cue_fn()
    except BaseException:
        _stop_collection_capture(audio_backend)
        raise

    try:
        audio_backend.wait()
    except KeyboardInterrupt:
        _stop_collection_capture(audio_backend)
        raise
    except Exception as error:
        _stop_collection_capture(audio_backend)
        details = describe_audio_error(error)
        raise DatasetCollectionError(
            details["message"], category=details["category"]
        ) from error

    captured_array = np.asarray(captured)
    if captured_array.ndim == 1 and channels == 1:
        captured_array = captured_array.reshape(-1, 1)
    if captured_array.ndim != 2 or captured_array.shape[1] != channels:
        raise DatasetCollectionError(
            "Captured audio channel count does not match the validated input "
            "configuration.",
            category="invalid_capture",
        )
    if captured_array.shape[0] == 0:
        raise DatasetCollectionError(
            "The audio backend returned an empty capture.",
            category="invalid_capture",
        )
    if captured_array.shape[0] != frames:
        raise DatasetCollectionError(
            "Captured audio frame count does not match the requested guided "
            "capture duration.",
            category="invalid_capture",
        )
    try:
        return np.ascontiguousarray(captured_array, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise DatasetCollectionError(
            "Captured audio could not be converted to float32.",
            category="invalid_capture",
        ) from error


def run_guided_collection(
    audio_backend: Any,
    *,
    device_index: int,
    samples_per_zone: int = DEFAULT_SAMPLES_PER_ZONE,
    zones: Sequence[str] = DEFAULT_ZONES,
    dataset_root: Path = Path("datasets"),
    parameters: CollectionParameters | None = None,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
    countdown_fn: Callable[[Callable[[str], None]], None] | None = None,
    capture_fn: Callable[..., np.ndarray[Any, Any]] | None = None,
    pre_cue_sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], datetime] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run an alternating guided session and persist only accepted captures."""

    settings = parameters or CollectionParameters()
    order = alternating_collection_order(zones, samples_per_zone)
    normalized_zones = normalize_zones(zones)
    read_input = input if input_fn is None else input_fn
    write_output = _terminal_output if output_fn is None else output_fn
    record = record_collection_attempt if capture_fn is None else capture_fn
    pre_cue_sleep = time.sleep if pre_cue_sleep_fn is None else pre_cue_sleep_fn
    clock = (lambda: datetime.now(timezone.utc)) if now_fn is None else now_fn

    inventory = collect_audio_inventory(
        audio_backend, selected_device_index=device_index
    )
    selected_device = next(
        (
            device
            for device in inventory["input_devices"]
            if device["index"] == inventory["selected_input_device_index"]
        ),
        None,
    )
    if selected_device is None:
        message = (
            inventory["errors"][0]["message"]
            if inventory["errors"]
            else "No usable input device was selected."
        )
        raise DatasetCollectionError(message, category="device_unavailable")
    try:
        configuration = find_characterization_configuration(
            audio_backend, selected_device
        )
    except RecordingConfigurationError as error:
        raise DatasetCollectionError(
            str(error), category=error.category
        ) from error

    created_at = _utc_timestamp(clock)
    session_metadata = _session_metadata(
        inventory=inventory,
        device=selected_device,
        configuration=configuration,
        zones=normalized_zones,
        samples_per_zone=samples_per_zone,
        parameters=settings,
    )
    try:
        session = create_dataset_session(
            Path(dataset_root),
            session_metadata,
            session_id=session_id,
            created_at_utc=created_at,
        )
    except (OSError, TypeError, ValueError) as error:
        raise DatasetCollectionError(
            f"Could not create dataset session: {error}",
            category="persistence_error",
        ) from error

    _announce_session(
        write_output,
        session,
        selected_device,
        configuration,
        settings,
        samples_per_zone,
        normalized_zones,
    )

    accepted_by_zone = {zone: 0 for zone in normalized_zones}
    accepted_total = 0
    attempt_index = 0
    for zone in order:
        accepted_number = accepted_by_zone[zone] + 1
        while accepted_by_zone[zone] < accepted_number:
            attempt_index += 1
            write_output("")
            write_output(
                f"Target {zone}: sample {accepted_number}/{samples_per_zone} "
                f"(overall {accepted_total + 1}/{len(order)})"
            )
            read_input("Press Enter when positioned and ready for the countdown: ")
            if countdown_fn is None:
                terminal_countdown(write_output)
            else:
                countdown_fn(write_output)

            captured_at = _utc_timestamp(clock)
            capture = record(
                audio_backend,
                device_index=int(selected_device["index"]),
                sample_rate_hz=float(configuration["sample_rate_hz"]),
                channels=int(configuration["channels"]),
                duration_seconds=settings.capture_duration_seconds,
                pre_cue_duration_seconds=settings.pre_cue_duration_seconds,
                tap_cue_fn=lambda: write_output("TAP NOW"),
                sleep_fn=pre_cue_sleep,
            )
            assessment = assess_capture_quality(
                capture,
                float(configuration["sample_rate_hz"]),
                expected_channels=int(configuration["channels"]),
                parameters=settings,
            )
            if assessment.accepted:
                confirmation = _request_confirmation(read_input)
                if not confirmation:
                    assessment = CaptureQualityAssessment(
                        accepted=False,
                        reasons=(
                            _reason(
                                "user_requested_retry",
                                "The user chose to discard this attempt and retry.",
                            ),
                        ),
                        metrics=assessment.metrics,
                        transient=assessment.transient,
                        tap_window=assessment.tap_window,
                    )

            if not assessment.accepted:
                rejection_metadata = _rejected_attempt_metadata(
                    zone=zone,
                    accepted_number=accepted_number,
                    attempt_index=attempt_index,
                    captured_at_utc=captured_at,
                    device=selected_device,
                    configuration=configuration,
                    assessment=assessment,
                    parameters=settings,
                )
                try:
                    session.log_rejected_attempt(rejection_metadata)
                except (OSError, TypeError, ValueError) as error:
                    raise DatasetCollectionError(
                        f"Could not log rejected attempt: {error}",
                        category="persistence_error",
                    ) from error
                write_output("Rejected; retrying the same zone:")
                for reason in assessment.reasons:
                    write_output(f"  - {reason['message']}")
                continue

            if assessment.tap_window is None or assessment.transient is None:
                raise DatasetCollectionError(
                    "An accepted assessment did not contain a tap window.",
                    category="analysis_error",
                )
            collection_order_index = accepted_total + 1
            filename_stem = f"{zone.casefold()}_{accepted_number:03d}"
            sample_id = f"{session.session_id}-{filename_stem}"
            sample_metadata = _accepted_sample_metadata(
                zone=zone,
                accepted_number=accepted_number,
                collection_order_index=collection_order_index,
                attempt_index=attempt_index,
                captured_at_utc=captured_at,
                device=selected_device,
                configuration=configuration,
                capture=capture,
                assessment=assessment,
                parameters=settings,
            )
            try:
                saved_path, _manifest_record = session.save_sample(
                    sample_id=sample_id,
                    filename_stem=filename_stem,
                    capture=capture,
                    tap_window=assessment.tap_window,
                    metadata=sample_metadata,
                )
            except (OSError, TypeError, ValueError) as error:
                raise DatasetCollectionError(
                    f"Could not persist accepted sample: {error}",
                    category="persistence_error",
                ) from error

            accepted_by_zone[zone] += 1
            accepted_total += 1
            write_output(
                f"Accepted {zone} #{accepted_number:03d}: {saved_path}"
            )

    write_output("")
    write_output(
        f"Collection complete: {accepted_total} accepted sample(s) in "
        f"{session.directory}"
    )
    return {
        "status": "completed",
        "session_id": session.session_id,
        "session_directory": str(session.directory),
        "manifest_path": str(session.manifest_path),
        "accepted_samples": accepted_total,
        "accepted_by_zone": accepted_by_zone,
        "attempts": attempt_index,
        "sample_rate_hz": float(configuration["sample_rate_hz"]),
        "channel_count": int(configuration["channels"]),
        "pre_cue_duration_seconds": float(settings.pre_cue_duration_seconds),
    }


def terminal_countdown(
    output_fn: Callable[[str], None],
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Print the silent preparation countdown before capture starts."""

    for count in ("3", "2", "1"):
        output_fn(count)
        sleep_fn(1.0)


def _request_confirmation(input_fn: Callable[[str], str]) -> bool:
    while True:
        response = input_fn(
            "Quality checks passed. Press Enter to accept, or type R to retry: "
        ).strip().casefold()
        if not response:
            return True
        if response in {"r", "retry"}:
            return False


def _session_metadata(
    *,
    inventory: Mapping[str, Any],
    device: Mapping[str, Any],
    configuration: Mapping[str, Any],
    zones: Sequence[str],
    samples_per_zone: int,
    parameters: CollectionParameters,
) -> dict[str, Any]:
    return {
        "dataset_kind": "guided_labeled_multichannel_tap_dataset",
        "project_phase": "2A",
        "system": system_information(),
        "audio_backend": dict(inventory["backend"]),
        "selected_endpoint": _endpoint_metadata(device),
        "capture_configuration": {
            "sample_rate_hz": float(configuration["sample_rate_hz"]),
            "sample_rate_source": str(configuration["sample_rate_source"]),
            "channel_count": int(configuration["channels"]),
            "dtype": "float32",
            "advertised_max_input_channels": int(
                configuration["advertised_max_input_channels"]
            ),
            "channel_search_limit": int(configuration["channel_search_limit"]),
            "channel_limit_applied": bool(configuration["channel_limit_applied"]),
        },
        "guided_tap_cue": _guided_cue_metadata(
            parameters, float(configuration["sample_rate_hz"])
        ),
        "zones": list(zones),
        "requested_samples_per_zone": int(samples_per_zone),
        "requested_total_samples": int(len(zones) * samples_per_zone),
        "collection_order": "round_robin_alternating_zones",
        "collection_parameters": parameters.to_metadata(),
        "inventory_warnings": [
            *[str(warning) for warning in inventory.get("warnings", [])],
            *[
                "Capability check warning: " + str(error.get("message", error))
                for error in inventory.get("errors", [])
                if inventory.get("status") == "capability_check_error"
            ],
        ],
        "privacy": {
            "waveforms_retained_locally": True,
            "network_upload_performed": False,
            "note": (
                "Microphone waveforms remain local unless the user explicitly "
                "moves or uploads the dataset directory."
            ),
        },
    }


def _accepted_sample_metadata(
    *,
    zone: str,
    accepted_number: int,
    collection_order_index: int,
    attempt_index: int,
    captured_at_utc: str,
    device: Mapping[str, Any],
    configuration: Mapping[str, Any],
    capture: np.ndarray[Any, Any],
    assessment: CaptureQualityAssessment,
    parameters: CollectionParameters,
) -> dict[str, Any]:
    return {
        "accepted": True,
        "zone": zone,
        "accepted_sample_number": int(accepted_number),
        "collection_order_index": int(collection_order_index),
        "attempt_index": int(attempt_index),
        "captured_at_utc": captured_at_utc,
        "device": _endpoint_metadata(device),
        "sample_rate_hz": float(configuration["sample_rate_hz"]),
        "channel_count": int(configuration["channels"]),
        "capture_dtype": "float32",
        "capture_frames": int(capture.shape[0]),
        "capture_duration_seconds": float(
            capture.shape[0] / float(configuration["sample_rate_hz"])
        ),
        "requested_capture_duration_seconds": float(
            parameters.capture_duration_seconds
        ),
        "guided_tap_cue": _guided_cue_metadata(
            parameters, float(configuration["sample_rate_hz"])
        ),
        "collection_parameters": parameters.to_metadata(),
        "capture_quality": assessment.to_metadata(),
    }


def _rejected_attempt_metadata(
    *,
    zone: str,
    accepted_number: int,
    attempt_index: int,
    captured_at_utc: str,
    device: Mapping[str, Any],
    configuration: Mapping[str, Any],
    assessment: CaptureQualityAssessment,
    parameters: CollectionParameters,
) -> dict[str, Any]:
    return {
        "attempt_index": int(attempt_index),
        "target_zone": zone,
        "prospective_accepted_sample_number": int(accepted_number),
        "captured_at_utc": captured_at_utc,
        "device": _endpoint_metadata(device),
        "sample_rate_hz": float(configuration["sample_rate_hz"]),
        "channel_count": int(configuration["channels"]),
        "guided_tap_cue": _guided_cue_metadata(
            parameters, float(configuration["sample_rate_hz"])
        ),
        "capture_quality": assessment.to_metadata(),
    }


def _endpoint_metadata(device: Mapping[str, Any]) -> dict[str, Any]:
    host_api = device.get("host_api", {})
    return {
        "index": int(device["index"]),
        "name": str(device["name"]),
        "host_api": {
            "index": (
                int(host_api["index"])
                if host_api.get("index") is not None
                else None
            ),
            "name": str(host_api.get("name", "Unknown host API")),
        },
    }


def _announce_session(
    output_fn: Callable[[str], None],
    session: DatasetSession,
    device: Mapping[str, Any],
    configuration: Mapping[str, Any],
    parameters: CollectionParameters,
    samples_per_zone: int,
    zones: Sequence[str],
) -> None:
    output_fn("DeskSense Phase 2A guided dataset collection")
    output_fn(f"Session: {session.session_id}")
    output_fn(f"Dataset directory: {session.directory}")
    output_fn(f"Device: {device['index']}: {device['name']}")
    output_fn(f"Host API: {device['host_api']['name']}")
    output_fn(
        f"Capture: {configuration['sample_rate_hz']:g} Hz, "
        f"{configuration['channels']} channel(s), "
        f"{parameters.capture_duration_seconds:g} s per attempt"
    )
    output_fn(
        "Cue timing: capture starts first, then TAP NOW is shown after a "
        f"{parameters.pre_cue_duration_seconds:g} s silent pre-cue interval."
    )
    output_fn(
        f"Targets: {samples_per_zone} per zone; order alternates "
        + " / ".join(zones)
    )
    output_fn(
        "Accepted microphone waveforms stay in this local dataset directory; "
        "no network upload is performed."
    )
    output_fn(
        "After each usable capture, press Enter to accept it or R to retry."
    )


def _invalid_assessment(
    code: str,
    message: str,
    thresholds: Mapping[str, Any],
    **measurements: Any,
) -> CaptureQualityAssessment:
    return CaptureQualityAssessment(
        accepted=False,
        reasons=(_reason(code, message),),
        metrics={"thresholds": dict(thresholds), **measurements},
        transient=None,
        tap_window=None,
    )


def _guided_cue_metadata(
    parameters: CollectionParameters, sample_rate_hz: float
) -> dict[str, Any]:
    total_frames = max(
        1, round(sample_rate_hz * parameters.capture_duration_seconds)
    )
    intended_cue_sample = min(
        total_frames,
        max(0, round(sample_rate_hz * parameters.pre_cue_duration_seconds)),
    )
    return {
        "cue_text": "TAP NOW",
        "cue_is_audible": False,
        "capture_started_before_cue": True,
        "intended_pre_cue_duration_seconds": float(
            parameters.pre_cue_duration_seconds
        ),
        "intended_cue_offset_samples": int(intended_cue_sample),
        "nominal_post_cue_duration_seconds": float(
            parameters.capture_duration_seconds
            - parameters.pre_cue_duration_seconds
        ),
        "nominal_post_cue_samples": int(total_frames - intended_cue_sample),
        "timing_kind": "intended_guided_visual_cue_schedule",
        "timing_caution": (
            "The stream is started before the cue, but OS scheduling, terminal "
            "rendering latency, and human reaction time are not measured."
        ),
    }


def _stop_collection_capture(audio_backend: Any) -> None:
    """Best-effort cleanup for a non-blocking convenience recording."""

    try:
        audio_backend.stop()
    except Exception:
        pass


def _reason(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _deduplicate_reasons(
    reasons: Sequence[dict[str, str]],
) -> tuple[dict[str, str], ...]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for reason in reasons:
        if reason["code"] in seen:
            continue
        seen.add(reason["code"])
        unique.append(reason)
    return tuple(unique)


def _utc_timestamp(clock: Callable[[], datetime]) -> str:
    timestamp = clock()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Collection timestamps must be timezone-aware.")
    return timestamp.astimezone(timezone.utc).isoformat()


def _terminal_output(message: str) -> None:
    print(message, flush=True)
