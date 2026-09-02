"""Injected live-audio adapter for DeskSense Phase 3A.2a.

This module deliberately does not import :mod:`sounddevice`.  A compatible
audio backend is supplied by the CLI after its existing lazy import.  The
PortAudio callback copies and queues audio only; the calling/main thread owns
detector processing, frozen tapness validation, spatial feature extraction,
frozen spatial inference, and event output.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from desksense.diagnostics import collect_audio_inventory, describe_audio_error
from desksense.features import PRIMARY_FEATURE_NAME, extract_two_channel_features
from desksense.frozen_baseline import FrozenBaselineError, load_frozen_baseline
from desksense.inference import classify_peak_ratio_value
from desksense.streaming import (
    CandidateStartDiagnostic,
    DetectionResult,
    DetectorDiagnosticSnapshot,
    DetectorState,
    StreamingDetectorConfig,
    StreamingTapDetector,
)
from desksense.tapness import (
    TapnessBaselineError,
    classify_tapness_metrics,
    extract_descriptive_tapness_metrics,
    load_tapness_baseline,
    validate_tapness_stage1_config,
)


LIVE_SAMPLE_RATE_HZ = 48_000.0
LIVE_CHANNEL_COUNT = 2
LIVE_DTYPE = "float32"
LIVE_TAP_WINDOW_FRAMES = 9_600
DEFAULT_QUEUE_CAPACITY_PACKETS = 8
DEFAULT_QUEUE_POLL_TIMEOUT_SECONDS = 0.1
DEFAULT_DIAGNOSTIC_REPORTING_SECONDS = 1.0

_STATUS_FLAG_NAMES = (
    "input_underflow",
    "input_overflow",
    "output_underflow",
    "output_overflow",
    "priming_output",
)


class RealtimeSensingError(RuntimeError):
    """A live setup, callback, stream, or processing operation failed."""

    def __init__(self, message: str, *, category: str = "realtime_error") -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class RealtimeSensingConfig:
    """Reviewable Phase 3A.2a transport settings.

    Eight queued callback packets, a 100 ms consumer poll, and a one-second
    opt-in diagnostic reporting interval are initial engineering values. Their
    physical suitability has not yet been measured on the Lenovo WDM-KS
    endpoint.
    """

    queue_capacity_packets: int = DEFAULT_QUEUE_CAPACITY_PACKETS
    queue_poll_timeout_seconds: float = DEFAULT_QUEUE_POLL_TIMEOUT_SECONDS
    diagnostic_reporting_seconds: float = DEFAULT_DIAGNOSTIC_REPORTING_SECONDS

    def __post_init__(self) -> None:
        if (
            isinstance(self.queue_capacity_packets, bool)
            or not isinstance(self.queue_capacity_packets, int)
            or self.queue_capacity_packets <= 0
        ):
            raise ValueError("Realtime queue capacity must be a positive integer.")
        try:
            timeout = float(self.queue_poll_timeout_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Realtime queue poll timeout must be positive and finite."
            ) from error
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(
                "Realtime queue poll timeout must be positive and finite."
            )
        try:
            diagnostic_interval = float(self.diagnostic_reporting_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Realtime diagnostic reporting interval must be positive and finite."
            ) from error
        if not math.isfinite(diagnostic_interval) or diagnostic_interval <= 0.0:
            raise ValueError(
                "Realtime diagnostic reporting interval must be positive and finite."
            )


@dataclass(frozen=True)
class LiveBaselineSpec:
    """Strictly loaded frozen model fields needed by live sensing."""

    baseline_path: Path
    source_session_id: str
    source_interaction_context: str
    endpoint_name: str
    host_api_name: str
    historical_device_index: int
    sample_rate_hz: float
    channel_count: int
    tap_window_frames: int
    tap_window_duration_seconds: float
    threshold_db: float
    lower_feature_zone: str
    higher_feature_zone: str
    direction: str


@dataclass(frozen=True)
class AudioChunkPacket:
    """One owned callback buffer and primitive callback metadata."""

    samples: np.ndarray[Any, Any]
    callback_sequence: int
    frame_count: int
    input_buffer_adc_time_seconds: float | None
    callback_current_time_seconds: float | None
    callback_arrival_monotonic_ns: int
    enqueue_attempt_monotonic_ns: int
    portaudio_status_flags: tuple[str, ...]
    discontinuity_before: bool
    discontinuity_reasons: tuple[str, ...]
    dropped_callback_count_before: int
    dropped_frame_count_before: int


@dataclass(frozen=True)
class LiveSensingEvent:
    """Structured live state, discontinuity, detection, or rejection event."""

    event_type: str
    status: str
    message: str
    stream_epoch: int | None = None
    callback_sequence: int | None = None
    rejection_reasons: tuple[str, ...] = ()
    predicted_zone: str | None = None
    feature_value_db: float | None = None
    threshold_db: float | None = None
    threshold_offset_db: float | None = None
    absolute_margin_db: float | None = None
    signed_margin_toward_right_db: float | None = None
    onset_frame_index: int | None = None
    center_frame_index: int | None = None
    window_start_frame_index: int | None = None
    window_end_frame_index_exclusive: int | None = None
    timing: Mapping[str, int | float | None] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveSensingSummary:
    """Bounded transport and processing totals returned on an injected stop."""

    packets_processed: int
    callback_count: int
    callback_frames: int
    dropped_callback_count: int
    dropped_frame_count: int
    queue_high_water_mark: int
    discontinuity_count: int
    detected_event_count: int
    rejected_event_count: int


@dataclass(frozen=True)
class _TimingSpan:
    stream_epoch: int
    start_frame_index: int
    end_frame_index_exclusive: int
    input_buffer_adc_time_seconds: float


class _InputCallbackBridge:
    """Copy callback buffers into a bounded queue without doing DSP work."""

    def __init__(
        self,
        *,
        capacity_packets: int,
        callback_abort: type[BaseException],
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.queue: queue.Queue[AudioChunkPacket] = queue.Queue(
            maxsize=capacity_packets
        )
        self._callback_abort = callback_abort
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._next_sequence = 0
        self._pending_reasons: list[str] = []
        self._pending_dropped_callbacks = 0
        self._pending_dropped_frames = 0
        self._fatal_error: BaseException | None = None
        self._fatal_error_event = threading.Event()
        self._callback_count = 0
        self._callback_frames = 0
        self._dropped_callback_count = 0
        self._dropped_frame_count = 0
        self._queue_high_water_mark = 0

    def callback(
        self,
        indata: Any,
        frames: Any,
        time_info: Any,
        status: Any,
    ) -> None:
        """Own one input buffer, snapshot primitives, and enqueue without wait."""

        try:
            arrival_ns = _clock_value(self._clock_ns())
            with self._lock:
                sequence = self._next_sequence
                self._next_sequence += 1
                self._callback_count += 1

            frame_count = int(frames)
            if frame_count < 0:
                raise ValueError("Callback frame count must not be negative.")
            with self._lock:
                self._callback_frames += frame_count

            # This is the one required ownership/conversion copy.  No detector
            # or feature calculation is performed in the callback.
            owned_samples = np.array(
                indata, dtype=np.float32, order="C", copy=True
            )
            if owned_samples.ndim != 2:
                raise ValueError("Callback audio must be a two-dimensional array.")
            if owned_samples.shape[0] != frame_count:
                raise ValueError(
                    "Callback frame count does not match the input buffer."
                )

            adc_time = _optional_time_field(time_info, "inputBufferAdcTime")
            callback_time = _optional_time_field(time_info, "currentTime")
            status_flags = _snapshot_status_flags(status)
            status_reasons = _status_discontinuity_reasons(status_flags)
            enqueue_ns = _clock_value(self._clock_ns())

            with self._lock:
                reasons = _ordered_unique(
                    (*self._pending_reasons, *status_reasons)
                )
                packet = AudioChunkPacket(
                    samples=owned_samples,
                    callback_sequence=sequence,
                    frame_count=frame_count,
                    input_buffer_adc_time_seconds=adc_time,
                    callback_current_time_seconds=callback_time,
                    callback_arrival_monotonic_ns=arrival_ns,
                    enqueue_attempt_monotonic_ns=enqueue_ns,
                    portaudio_status_flags=status_flags,
                    discontinuity_before=bool(reasons),
                    discontinuity_reasons=reasons,
                    dropped_callback_count_before=(
                        self._pending_dropped_callbacks
                    ),
                    dropped_frame_count_before=self._pending_dropped_frames,
                )
                try:
                    self.queue.put_nowait(packet)
                except queue.Full:
                    self._add_pending_reason("queue_overflow")
                    for reason in status_reasons:
                        self._add_pending_reason(reason)
                    self._pending_dropped_callbacks += 1
                    self._pending_dropped_frames += frame_count
                    self._dropped_callback_count += 1
                    self._dropped_frame_count += frame_count
                    return

                self._queue_high_water_mark = max(
                    self._queue_high_water_mark, self.queue.qsize()
                )
                self._pending_reasons.clear()
                self._pending_dropped_callbacks = 0
                self._pending_dropped_frames = 0
        except Exception as error:
            self._publish_fatal_error(error)
            raise self._callback_abort from error

    @property
    def fatal_error(self) -> BaseException | None:
        with self._lock:
            return self._fatal_error

    @property
    def fatal_error_event(self) -> threading.Event:
        return self._fatal_error_event

    def statistics(self) -> dict[str, int]:
        with self._lock:
            return {
                "callback_count": self._callback_count,
                "callback_frames": self._callback_frames,
                "dropped_callback_count": self._dropped_callback_count,
                "dropped_frame_count": self._dropped_frame_count,
                "queue_high_water_mark": self._queue_high_water_mark,
            }

    def _add_pending_reason(self, reason: str) -> None:
        if reason not in self._pending_reasons:
            self._pending_reasons.append(reason)

    def _publish_fatal_error(self, error: BaseException) -> None:
        with self._lock:
            if self._fatal_error is None:
                self._fatal_error = error
                self._fatal_error_event.set()


def load_live_baseline_spec(path: Path) -> LiveBaselineSpec:
    """Strictly load a frozen artifact and extract its immutable live domain."""

    baseline_path = Path(path)
    try:
        artifact = load_frozen_baseline(baseline_path)
    except (FrozenBaselineError, OSError, TypeError, ValueError) as error:
        raise RealtimeSensingError(
            f"Could not load the frozen baseline: {error}",
            category="baseline_error",
        ) from error

    source = artifact["source_development_dataset"]
    compatibility = source["compatibility"]
    classifier = artifact["classifier"]
    spec = LiveBaselineSpec(
        baseline_path=baseline_path.resolve(),
        source_session_id=str(source["session_id"]),
        source_interaction_context=str(source["interaction_context"]),
        endpoint_name=str(compatibility["endpoint_name"]),
        host_api_name=str(compatibility["host_api_name"]),
        historical_device_index=int(compatibility["device_index_at_collection"]),
        sample_rate_hz=float(compatibility["sample_rate_hz"]),
        channel_count=int(compatibility["channel_count"]),
        tap_window_frames=int(compatibility["tap_window_frames"]),
        tap_window_duration_seconds=float(
            compatibility["tap_window_duration_seconds"]
        ),
        threshold_db=float(classifier["threshold_db"]),
        lower_feature_zone=str(classifier["lower_feature_zone"]),
        higher_feature_zone=str(classifier["higher_feature_zone"]),
        direction=str(classifier["direction"]),
    )
    _validate_required_live_domain(spec)
    return spec


def validate_live_compatibility(
    baseline: LiveBaselineSpec,
    device: Mapping[str, Any],
) -> None:
    """Require the frozen endpoint/domain without comparing device indexes."""

    _validate_required_live_domain(baseline)
    mismatches: list[str] = []
    if str(device.get("name", "")) != baseline.endpoint_name:
        mismatches.append("endpoint name")
    host_api = device.get("host_api")
    host_api_name = (
        str(host_api.get("name", "")) if isinstance(host_api, Mapping) else ""
    )
    if host_api_name != baseline.host_api_name:
        mismatches.append("host API name")
    try:
        maximum_channels = int(device.get("max_input_channels", 0))
    except (TypeError, ValueError, OverflowError):
        maximum_channels = 0
    if maximum_channels < baseline.channel_count:
        mismatches.append("available input channel count")
    if mismatches:
        raise RealtimeSensingError(
            "Selected live endpoint is incompatible with the frozen feature "
            "domain: "
            + ", ".join(mismatches)
            + ". The historical device index is intentionally not used as "
            "stable identity.",
            category="incompatible_live_domain",
        )


def process_detection_result(
    result: DetectionResult,
    baseline: LiveBaselineSpec,
    *,
    tapness_baseline: Mapping[str, Any] | None = None,
    callback_sequence: int | None = None,
    timing: Mapping[str, int | float | None] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> LiveSensingEvent:
    """Apply optional frozen tapness then unchanged frozen spatial inference.

    Only ``result.status == 'detected'`` reaches Stage 2. A Stage 2 rejection
    never reaches Stage 3. A detector-rejected near-clipping result can contain
    a candidate array and is never classified.
    """

    common = {
        "stream_epoch": int(result.stream_epoch),
        "callback_sequence": callback_sequence,
        "onset_frame_index": int(result.onset_frame_index),
        "center_frame_index": (
            int(result.center_frame_index)
            if result.center_frame_index is not None
            else None
        ),
        "window_start_frame_index": (
            int(result.window_start_frame_index)
            if result.window_start_frame_index is not None
            else None
        ),
        "window_end_frame_index_exclusive": (
            int(result.window_end_frame_index_exclusive)
            if result.window_end_frame_index_exclusive is not None
            else None
        ),
        "timing": dict(timing or {}),
        "details": {"detector_metrics": dict(result.metrics)},
    }
    if result.status != "detected":
        reasons = result.rejection_reasons or (
            f"detector_status_{result.status}",
        )
        return LiveSensingEvent(
            event_type="rejection",
            status="rejected",
            message="Tap candidate rejected by the streaming detector.",
            rejection_reasons=tuple(str(reason) for reason in reasons),
            **common,
        )

    candidate = result.candidate_window
    expected_shape = (baseline.tap_window_frames, baseline.channel_count)
    if candidate is None or candidate.shape != expected_shape:
        actual_shape = None if candidate is None else tuple(candidate.shape)
        return LiveSensingEvent(
            event_type="rejection",
            status="rejected",
            message="Detected candidate did not match the frozen feature domain.",
            rejection_reasons=("invalid_candidate_shape",),
            details={
                **common["details"],
                "expected_candidate_shape": expected_shape,
                "actual_candidate_shape": actual_shape,
            },
            **{key: value for key, value in common.items() if key != "details"},
        )

    if tapness_baseline is not None:
        if result.window_start_frame_index is None:
            return LiveSensingEvent(
                event_type="rejection",
                status="rejected",
                message="Tapness analysis requires a valid candidate onset offset.",
                rejection_reasons=("invalid_tapness_onset_offset",),
                **common,
            )
        onset_offset = result.onset_frame_index - result.window_start_frame_index
        try:
            tapness_metrics = extract_descriptive_tapness_metrics(
                candidate,
                sample_rate_hz=baseline.sample_rate_hz,
                onset_offset_frames=onset_offset,
                learned_noise_floor_rms=result.metrics.get(
                    "learned_noise_floor_rms"
                ),
            )
            tapness_decision = classify_tapness_metrics(
                tapness_metrics, tapness_baseline
            )
        except (TapnessBaselineError, TypeError, ValueError, OverflowError) as error:
            return LiveSensingEvent(
                event_type="rejection",
                status="rejected",
                message="Tapness candidate analysis failed safely.",
                rejection_reasons=("tapness_inference_failed",),
                details={**common["details"], "tapness_error": str(error)},
                **{key: value for key, value in common.items() if key != "details"},
            )
        common["details"]["tapness_metrics"] = tapness_metrics
        common["details"]["tapness_inference"] = tapness_decision
        if not tapness_decision["tap_accepted"]:
            return LiveSensingEvent(
                event_type="rejection",
                status="rejected",
                message="Candidate rejected by the frozen tapness baseline.",
                rejection_reasons=("tapness_non_tap",),
                **common,
            )

    try:
        features = extract_two_channel_features(candidate)
    except (TypeError, ValueError, OverflowError) as error:
        return LiveSensingEvent(
            event_type="rejection",
            status="rejected",
            message="Detected candidate could not be analyzed safely.",
            rejection_reasons=("feature_extraction_failed",),
            details={**common["details"], "feature_error": str(error)},
            **{key: value for key, value in common.items() if key != "details"},
        )
    feature_value = features.get(PRIMARY_FEATURE_NAME)
    if feature_value is None:
        return LiveSensingEvent(
            event_type="rejection",
            status="rejected",
            message="Primary peak-ratio feature was undefined; no label emitted.",
            rejection_reasons=("undefined_primary_feature",),
            details={**common["details"], "features": features},
            **{key: value for key, value in common.items() if key != "details"},
        )

    decision = classify_peak_ratio_value(
        feature_value,
        baseline.threshold_db,
        baseline.lower_feature_zone,
        baseline.higher_feature_zone,
    )
    complete_ns = _clock_value(clock_ns())
    accepted_timing = dict(common["timing"])
    accepted_timing["feature_inference_complete_monotonic_ns"] = complete_ns
    return LiveSensingEvent(
        event_type="detection",
        status="detected",
        message="Tap classified with the unchanged frozen baseline.",
        predicted_zone=str(decision["predicted_label"]),
        feature_value_db=float(decision["feature_value_db"]),
        threshold_db=float(decision["threshold_db"]),
        threshold_offset_db=float(decision["threshold_offset_db"]),
        absolute_margin_db=float(decision["absolute_margin_db"]),
        signed_margin_toward_right_db=float(
            decision["signed_margin_toward_RIGHT_db"]
        ),
        timing=accepted_timing,
        **{key: value for key, value in common.items() if key != "timing"},
    )


def run_live_sensing(
    audio_backend: Any,
    *,
    device_index: int,
    baseline_path: Path,
    tapness_baseline_path: Path | None = None,
    event_handler: Callable[[LiveSensingEvent], None],
    config: RealtimeSensingConfig | None = None,
    detector: StreamingTapDetector | None = None,
    diagnostics_enabled: bool = False,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    stop_requested: Callable[[], bool] | None = None,
) -> LiveSensingSummary:
    """Run the injected stream while processing audio only on the caller thread."""

    settings = config or RealtimeSensingConfig()
    should_stop = (lambda: False) if stop_requested is None else stop_requested
    baseline = load_live_baseline_spec(Path(baseline_path))
    try:
        tapness_baseline = (
            load_tapness_baseline(Path(tapness_baseline_path))
            if tapness_baseline_path is not None
            else None
        )
    except (TapnessBaselineError, OSError, TypeError, ValueError) as error:
        raise RealtimeSensingError(
            f"Could not load the frozen tapness baseline: {error}",
            category="tapness_baseline_error",
        ) from error
    selected_device = _resolve_selected_device(audio_backend, device_index)
    validate_live_compatibility(baseline, selected_device)

    live_detector = detector or StreamingTapDetector(
        StreamingDetectorConfig(
            sample_rate_hz=baseline.sample_rate_hz,
            channel_count=baseline.channel_count,
            tap_window_seconds=(
                baseline.tap_window_frames / baseline.sample_rate_hz
            ),
        )
    )
    _validate_detector_domain(live_detector, baseline)
    if tapness_baseline is not None:
        try:
            validate_tapness_stage1_config(live_detector.config, tapness_baseline)
        except TapnessBaselineError as error:
            raise RealtimeSensingError(
                f"Tapness baseline is incompatible with the live detector: {error}",
                category="tapness_baseline_error",
            ) from error
    _check_input_settings(audio_backend, selected_device, baseline)

    callback_abort = getattr(audio_backend, "CallbackAbort", None)
    if not isinstance(callback_abort, type) or not issubclass(
        callback_abort, BaseException
    ):
        raise RealtimeSensingError(
            "Audio backend does not expose a usable CallbackAbort exception.",
            category="audio_backend_error",
        )
    bridge = _InputCallbackBridge(
        capacity_packets=settings.queue_capacity_packets,
        callback_abort=callback_abort,
        clock_ns=clock_ns,
    )
    finished_event = threading.Event()
    stream: Any | None = None
    started = False
    primary_error: BaseException | None = None
    packets_processed = 0
    discontinuity_count = 0
    detected_count = 0
    rejected_count = 0
    expected_sequence = 0
    epoch_submitted_frames = 0
    armed_announced_epoch: int | None = None
    timing_spans: deque[_TimingSpan] = deque()
    diagnostic_interval_frames = max(
        1,
        int(
            round(
                float(live_detector.config.sample_rate_hz)
                * float(settings.diagnostic_reporting_seconds)
            )
        ),
    )
    next_diagnostic_report_frame = diagnostic_interval_frames

    try:
        try:
            stream = audio_backend.InputStream(
                device=int(selected_device["index"]),
                samplerate=baseline.sample_rate_hz,
                channels=baseline.channel_count,
                dtype=LIVE_DTYPE,
                blocksize=0,
                callback=bridge.callback,
                finished_callback=finished_event.set,
            )
        except Exception as error:
            _raise_audio_failure("Could not construct the live input stream", error)

        _validate_opened_stream(stream, baseline)
        try:
            stream.start()
            started = True
        except Exception as error:
            _raise_audio_failure("Could not start the live input stream", error)

        event_handler(
            _startup_event(
                baseline,
                selected_device,
                stream,
                settings,
                live_detector,
                tapness_baseline_path=tapness_baseline_path,
            )
        )

        while not should_stop():
            _raise_callback_failure(bridge)
            try:
                packet = bridge.queue.get(
                    timeout=float(settings.queue_poll_timeout_seconds)
                )
            except queue.Empty:
                _raise_callback_failure(bridge)
                if finished_event.is_set() or _stream_is_inactive(stream):
                    raise RealtimeSensingError(
                        "Live input stream became inactive unexpectedly.",
                        category="stream_terminated",
                    )
                continue

            _raise_callback_failure(bridge)
            processing_start_ns = _clock_value(clock_ns())
            reasons = list(packet.discontinuity_reasons)
            if packet.callback_sequence > expected_sequence:
                reasons.append("callback_sequence_gap")
            elif packet.callback_sequence < expected_sequence:
                raise RealtimeSensingError(
                    "Live callback packets arrived out of sequence.",
                    category="transport_error",
                )
            expected_sequence = packet.callback_sequence + 1

            reasons_tuple = _ordered_unique(reasons)
            if packet.discontinuity_before or reasons_tuple:
                ended_epoch_diagnostics = live_detector.notify_discontinuity()
                discontinuity_count += 1
                epoch_submitted_frames = 0
                armed_announced_epoch = None
                timing_spans.clear()
                next_diagnostic_report_frame = diagnostic_interval_frames
                discontinuity_details: dict[str, Any] = {
                    "dropped_callback_count_before": (
                        packet.dropped_callback_count_before
                    ),
                    "dropped_frame_count_before": (
                        packet.dropped_frame_count_before
                    ),
                    "portaudio_status_flags": packet.portaudio_status_flags,
                }
                if diagnostics_enabled and isinstance(
                    ended_epoch_diagnostics, DetectorDiagnosticSnapshot
                ):
                    discontinuity_details["ended_detector_epoch"] = (
                        _diagnostic_snapshot_details(ended_epoch_diagnostics)
                    )
                event_handler(
                    LiveSensingEvent(
                        event_type="discontinuity",
                        status="relearning",
                        message=(
                            "Audio continuity was lost; detector history was "
                            "cleared and startup learning restarted."
                        ),
                        stream_epoch=int(live_detector.stream_epoch),
                        callback_sequence=packet.callback_sequence,
                        rejection_reasons=reasons_tuple,
                        timing={
                            "processing_start_monotonic_ns": processing_start_ns,
                        },
                        details=discontinuity_details,
                    )
                )

            packet_start = epoch_submitted_frames
            packet_end = packet_start + packet.frame_count
            if packet.input_buffer_adc_time_seconds is not None:
                timing_spans.append(
                    _TimingSpan(
                        stream_epoch=int(live_detector.stream_epoch),
                        start_frame_index=packet_start,
                        end_frame_index_exclusive=packet_end,
                        input_buffer_adc_time_seconds=(
                            packet.input_buffer_adc_time_seconds
                        ),
                    )
                )
            _trim_timing_spans(
                timing_spans,
                retain_from=max(
                    0,
                    packet_end - live_detector.config.history_capacity_frames,
                ),
            )

            state_before = live_detector.state
            try:
                results = live_detector.process_chunk(packet.samples)
            except (TypeError, ValueError, OverflowError) as error:
                raise RealtimeSensingError(
                    f"Live detector rejected a callback packet: {error}",
                    category="invalid_live_audio",
                ) from error
            epoch_submitted_frames = packet_end
            packets_processed += 1
            result_available_ns = _clock_value(clock_ns())

            if (
                armed_announced_epoch != live_detector.stream_epoch
                and state_before is DetectorState.LEARNING
                and live_detector.state is not DetectorState.LEARNING
            ):
                armed_announced_epoch = int(live_detector.stream_epoch)
                event_handler(
                    LiveSensingEvent(
                        event_type="state",
                        status="armed",
                        message="Startup noise learning complete; detector armed.",
                        stream_epoch=int(live_detector.stream_epoch),
                        callback_sequence=packet.callback_sequence,
                        timing={
                            "detector_state_available_monotonic_ns": (
                                result_available_ns
                            )
                        },
                        details={
                            "noise_floor_rms": float(
                                live_detector.noise_floor_rms
                            )
                        },
                    )
                )

            if diagnostics_enabled:
                for candidate_start in _drain_candidate_start_diagnostics(
                    live_detector
                ):
                    event_handler(
                        _candidate_start_event(
                            candidate_start,
                            callback_sequence=packet.callback_sequence,
                        )
                    )

                processed_frames = int(live_detector.processed_frame_count)
                if processed_frames >= next_diagnostic_report_frame:
                    snapshot = _consume_diagnostic_interval(live_detector)
                    if snapshot is not None:
                        event_handler(
                            _diagnostic_summary_event(
                                snapshot,
                                bridge.statistics(),
                                callback_sequence=packet.callback_sequence,
                                discontinuity_count=discontinuity_count,
                                portaudio_status_flags=(
                                    packet.portaudio_status_flags
                                ),
                            )
                        )
                    next_diagnostic_report_frame = (
                        processed_frames + diagnostic_interval_frames
                    )

            for result in results:
                event_timing = _result_timing(
                    packet,
                    result,
                    timing_spans,
                    stream,
                    processing_start_ns=processing_start_ns,
                    result_available_ns=result_available_ns,
                )
                event = process_detection_result(
                    result,
                    baseline,
                    tapness_baseline=tapness_baseline,
                    callback_sequence=packet.callback_sequence,
                    timing=event_timing,
                    clock_ns=clock_ns,
                )
                event_handler(event)
                if event.status == "detected":
                    detected_count += 1
                else:
                    rejected_count += 1
    except KeyboardInterrupt as error:
        primary_error = error
        raise
    except RealtimeSensingError as error:
        primary_error = error
        raise
    except Exception as error:
        primary_error = error
        raise RealtimeSensingError(
            f"Live sensing failed: {error}", category="realtime_error"
        ) from error
    finally:
        cleanup_error = _stop_and_close_stream(stream, started=started)
        if cleanup_error is not None and primary_error is None:
            raise RealtimeSensingError(
                f"Could not cleanly close the live input stream: {cleanup_error}",
                category="stream_cleanup_error",
            ) from cleanup_error

    statistics = bridge.statistics()
    return LiveSensingSummary(
        packets_processed=packets_processed,
        callback_count=statistics["callback_count"],
        callback_frames=statistics["callback_frames"],
        dropped_callback_count=statistics["dropped_callback_count"],
        dropped_frame_count=statistics["dropped_frame_count"],
        queue_high_water_mark=statistics["queue_high_water_mark"],
        discontinuity_count=discontinuity_count,
        detected_event_count=detected_count,
        rejected_event_count=rejected_count,
    )


def _validate_required_live_domain(baseline: LiveBaselineSpec) -> None:
    mismatches: list[str] = []
    if not math.isclose(
        baseline.sample_rate_hz,
        LIVE_SAMPLE_RATE_HZ,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        mismatches.append("48,000 Hz sample rate")
    if baseline.channel_count != LIVE_CHANNEL_COUNT:
        mismatches.append("two-channel input")
    if baseline.tap_window_frames != LIVE_TAP_WINDOW_FRAMES:
        mismatches.append("9,600-frame candidate window")
    if mismatches:
        raise RealtimeSensingError(
            "Frozen baseline does not match the Phase 3A.2a live domain: "
            + ", ".join(mismatches),
            category="incompatible_live_domain",
        )


def _resolve_selected_device(
    audio_backend: Any, device_index: int
) -> Mapping[str, Any]:
    inventory = collect_audio_inventory(
        audio_backend, selected_device_index=int(device_index)
    )
    selected_index = inventory.get("selected_input_device_index")
    selected = next(
        (
            device
            for device in inventory.get("input_devices", [])
            if device.get("index") == selected_index
        ),
        None,
    )
    if selected is None:
        errors = inventory.get("errors", [])
        message = (
            str(errors[0].get("message", errors[0]))
            if errors
            else "No usable explicit input device was selected."
        )
        raise RealtimeSensingError(message, category="device_unavailable")
    return selected


def _validate_detector_domain(
    detector: StreamingTapDetector, baseline: LiveBaselineSpec
) -> None:
    config = detector.config
    mismatches: list[str] = []
    if not math.isclose(
        float(config.sample_rate_hz),
        baseline.sample_rate_hz,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        mismatches.append("sample rate")
    if int(config.channel_count) != baseline.channel_count:
        mismatches.append("channel count")
    if int(config.tap_window_frames) != baseline.tap_window_frames:
        mismatches.append("candidate-window frame count")
    if mismatches:
        raise RealtimeSensingError(
            "Streaming detector is incompatible with the frozen feature domain: "
            + ", ".join(mismatches),
            category="incompatible_live_domain",
        )


def _check_input_settings(
    audio_backend: Any,
    device: Mapping[str, Any],
    baseline: LiveBaselineSpec,
) -> None:
    try:
        audio_backend.check_input_settings(
            device=int(device["index"]),
            channels=baseline.channel_count,
            dtype=LIVE_DTYPE,
            samplerate=baseline.sample_rate_hz,
        )
    except Exception as error:
        _raise_audio_failure(
            "Required live input configuration is unavailable", error
        )


def _validate_opened_stream(stream: Any, baseline: LiveBaselineSpec) -> None:
    mismatches: list[str] = []
    samplerate = getattr(stream, "samplerate", None)
    if samplerate is not None:
        try:
            matches_rate = math.isclose(
                float(samplerate),
                baseline.sample_rate_hz,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        except (TypeError, ValueError, OverflowError):
            matches_rate = False
        if not matches_rate:
            mismatches.append("opened sample rate")
    channels = getattr(stream, "channels", None)
    if channels is not None and channels != baseline.channel_count:
        mismatches.append("opened channel count")
    dtype = getattr(stream, "dtype", None)
    if dtype is not None:
        try:
            matches_dtype = np.dtype(dtype) == np.dtype(LIVE_DTYPE)
        except (TypeError, ValueError):
            matches_dtype = False
        if not matches_dtype:
            mismatches.append("opened dtype")
    if mismatches:
        raise RealtimeSensingError(
            "Opened stream is incompatible with the frozen feature domain: "
            + ", ".join(mismatches),
            category="incompatible_live_domain",
        )


def _startup_event(
    baseline: LiveBaselineSpec,
    device: Mapping[str, Any],
    stream: Any,
    settings: RealtimeSensingConfig,
    detector: StreamingTapDetector,
    *,
    tapness_baseline_path: Path | None,
) -> LiveSensingEvent:
    host_api = device.get("host_api", {})
    return LiveSensingEvent(
        event_type="startup",
        status="learning",
        message="Live input started; learning room noise before arming.",
        stream_epoch=int(detector.stream_epoch),
        timing={"stream_reported_latency_seconds": _stream_latency(stream)},
        details={
            "device_index": int(device["index"]),
            "endpoint_name": str(device["name"]),
            "host_api_name": str(host_api.get("name", "Unknown host API")),
            "sample_rate_hz": baseline.sample_rate_hz,
            "channel_count": baseline.channel_count,
            "dtype": LIVE_DTYPE,
            "tap_window_frames": baseline.tap_window_frames,
            "baseline_path": str(baseline.baseline_path),
            "baseline_source_session_id": baseline.source_session_id,
            "tapness_baseline_path": (
                str(tapness_baseline_path)
                if tapness_baseline_path is not None
                else None
            ),
            "threshold_db": baseline.threshold_db,
            "direction": baseline.direction,
            "queue_capacity_packets": settings.queue_capacity_packets,
            "startup_learning_seconds": float(
                detector.config.startup_learning_frames
                / detector.config.sample_rate_hz
            ),
        },
    )


def _drain_candidate_start_diagnostics(
    detector: Any,
) -> tuple[CandidateStartDiagnostic, ...]:
    drain = getattr(detector, "drain_candidate_start_diagnostics", None)
    if not callable(drain):
        return ()
    records = tuple(drain())
    return tuple(
        record for record in records if isinstance(record, CandidateStartDiagnostic)
    )


def _consume_diagnostic_interval(
    detector: Any,
) -> DetectorDiagnosticSnapshot | None:
    consume = getattr(detector, "consume_diagnostic_interval", None)
    if not callable(consume):
        return None
    snapshot = consume()
    return snapshot if isinstance(snapshot, DetectorDiagnosticSnapshot) else None


def _candidate_start_event(
    record: CandidateStartDiagnostic,
    *,
    callback_sequence: int,
) -> LiveSensingEvent:
    block = record.block
    return LiveSensingEvent(
        event_type="diagnostic",
        status="candidate_started",
        message=f"Stage 1 {record.route} route started a candidate collection.",
        stream_epoch=record.stream_epoch,
        callback_sequence=callback_sequence,
        onset_frame_index=record.onset_frame_index,
        details={"candidate_start_route": record.route, "gate_block": asdict(block)},
    )


def _diagnostic_summary_event(
    snapshot: DetectorDiagnosticSnapshot,
    transport_statistics: Mapping[str, int],
    *,
    callback_sequence: int,
    discontinuity_count: int,
    portaudio_status_flags: tuple[str, ...],
) -> LiveSensingEvent:
    details = _diagnostic_snapshot_details(snapshot)
    details["transport"] = {
        "callback_count": int(transport_statistics["callback_count"]),
        "callback_frames": int(transport_statistics["callback_frames"]),
        "queue_high_water_mark": int(
            transport_statistics["queue_high_water_mark"]
        ),
        "dropped_callback_count": int(
            transport_statistics["dropped_callback_count"]
        ),
        "dropped_frame_count": int(transport_statistics["dropped_frame_count"]),
        "discontinuity_count": int(discontinuity_count),
        "current_packet_portaudio_status_flags": portaudio_status_flags,
    }
    return LiveSensingEvent(
        event_type="diagnostic",
        status="summary",
        message="Low-rate detector decision summary.",
        stream_epoch=snapshot.stream_epoch,
        callback_sequence=callback_sequence,
        details=details,
    )


def _diagnostic_snapshot_details(
    snapshot: DetectorDiagnosticSnapshot,
) -> dict[str, Any]:
    return {
        "detector_state": snapshot.state.value,
        "processed_frame_count": snapshot.processed_frame_count,
        "learned_noise_floor_rms": snapshot.learned_noise_floor_rms,
        "cumulative_counters": asdict(snapshot.cumulative_counters),
        "interval_counters": asdict(snapshot.interval_counters),
        "closest_armed_block": (
            asdict(snapshot.closest_armed_block)
            if snapshot.closest_armed_block is not None
            else None
        ),
        "buffered_candidate_start_record_count": (
            snapshot.buffered_candidate_start_record_count
        ),
    }


def _result_timing(
    packet: AudioChunkPacket,
    result: DetectionResult,
    spans: deque[_TimingSpan],
    stream: Any,
    *,
    processing_start_ns: int,
    result_available_ns: int,
) -> dict[str, int | float | None]:
    queue_dwell_ns = processing_start_ns - packet.enqueue_attempt_monotonic_ns
    if queue_dwell_ns < 0:
        queue_dwell_ns = 0
    onset_adc = _estimate_adc_time(
        spans, result.stream_epoch, result.onset_frame_index
    )
    center_adc = (
        _estimate_adc_time(spans, result.stream_epoch, result.center_frame_index)
        if result.center_frame_index is not None
        else None
    )
    stream_time = _stream_time(stream)
    callback_buffer_age = _nonnegative_difference(
        packet.callback_current_time_seconds,
        packet.input_buffer_adc_time_seconds,
    )
    onset_to_callback = _nonnegative_difference(
        packet.callback_current_time_seconds,
        onset_adc,
    )
    center_to_callback = _nonnegative_difference(
        packet.callback_current_time_seconds,
        center_adc,
    )
    callback_to_result = _nonnegative_nanoseconds_duration(
        result_available_ns,
        packet.callback_arrival_monotonic_ns,
    )
    return {
        "callback_arrival_monotonic_ns": (
            packet.callback_arrival_monotonic_ns
        ),
        "enqueue_attempt_monotonic_ns": (
            packet.enqueue_attempt_monotonic_ns
        ),
        "processing_start_monotonic_ns": processing_start_ns,
        "detector_result_available_monotonic_ns": result_available_ns,
        "queue_dwell_seconds": float(queue_dwell_ns / 1_000_000_000.0),
        "input_buffer_adc_time_seconds": (
            packet.input_buffer_adc_time_seconds
        ),
        "callback_current_time_seconds": packet.callback_current_time_seconds,
        "callback_buffer_age_seconds": callback_buffer_age,
        "estimated_onset_adc_time_seconds": onset_adc,
        "estimated_center_adc_time_seconds": center_adc,
        "portaudio_onset_to_callback_seconds": onset_to_callback,
        "portaudio_center_to_callback_seconds": center_to_callback,
        "python_callback_to_result_seconds": callback_to_result,
        # Raw backend evidence only. PortAudio stream.time is not assumed to
        # share a usable absolute origin with inputBufferAdcTime on every host.
        "stream_time_at_result_seconds": stream_time,
        "detector_latency_seconds": _optional_finite(
            result.metrics.get("detector_latency_seconds")
        ),
    }


def _estimate_adc_time(
    spans: deque[_TimingSpan],
    stream_epoch: int,
    frame_index: int,
) -> float | None:
    for span in reversed(spans):
        if (
            span.stream_epoch == stream_epoch
            and span.start_frame_index
            <= frame_index
            < span.end_frame_index_exclusive
        ):
            return float(
                span.input_buffer_adc_time_seconds
                + (frame_index - span.start_frame_index) / LIVE_SAMPLE_RATE_HZ
            )
    return None


def _trim_timing_spans(
    spans: deque[_TimingSpan], *, retain_from: int
) -> None:
    while spans and spans[0].end_frame_index_exclusive <= retain_from:
        spans.popleft()


def _raise_callback_failure(bridge: _InputCallbackBridge) -> None:
    error = bridge.fatal_error
    if error is not None:
        raise RealtimeSensingError(
            f"Live audio callback failed: {error}",
            category="callback_error",
        ) from error


def _stream_is_inactive(stream: Any) -> bool:
    active = getattr(stream, "active", None)
    if active is None:
        return False
    try:
        return not bool(active)
    except Exception:
        return False


def _stream_latency(stream: Any) -> float | None:
    return _nonnegative_optional_finite(getattr(stream, "latency", None))


def _stream_time(stream: Any) -> float | None:
    try:
        return _optional_finite(getattr(stream, "time", None))
    except Exception:
        return None


def _stop_and_close_stream(
    stream: Any | None, *, started: bool
) -> BaseException | None:
    if stream is None:
        return None
    first_error: BaseException | None = None
    if started:
        try:
            stream.stop()
        except Exception as error:
            first_error = error
    try:
        stream.close()
    except Exception as error:
        if first_error is None:
            first_error = error
    return first_error


def _raise_audio_failure(prefix: str, error: BaseException) -> None:
    details = describe_audio_error(error)
    raise RealtimeSensingError(
        f"{prefix}: {details['message']}", category=details["category"]
    ) from error


def _snapshot_status_flags(status: Any) -> tuple[str, ...]:
    if status is None:
        return ()
    flags: list[str] = []
    for name in _STATUS_FLAG_NAMES:
        try:
            enabled = bool(getattr(status, name, False))
        except Exception:
            enabled = False
        if enabled:
            flags.append(name)
    try:
        has_unrecognized_flag = bool(status) and not flags
    except Exception:
        has_unrecognized_flag = False
    if has_unrecognized_flag:
        flags.append("unrecognized_input_status")
    return tuple(flags)


def _status_discontinuity_reasons(
    status_flags: tuple[str, ...]
) -> tuple[str, ...]:
    reasons: list[str] = []
    for flag in status_flags:
        if flag == "input_overflow":
            reasons.append("portaudio_input_overflow")
        elif flag == "input_underflow":
            reasons.append("portaudio_input_underflow")
        else:
            # Output flags are not expected on InputStream.  Conservatively
            # prevent history from spanning any unexpected non-empty status.
            reasons.append(f"portaudio_{flag}")
    return tuple(reasons)


def _optional_time_field(value: Any, name: str) -> float | None:
    try:
        field_value = getattr(value, name)
    except Exception:
        return None
    return _optional_finite(field_value)


def _optional_finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_optional_finite(value: Any) -> float | None:
    result = _optional_finite(value)
    return result if result is not None and result >= 0.0 else None


def _nonnegative_difference(
    later: float | None, earlier: float | None
) -> float | None:
    if later is None or earlier is None:
        return None
    difference = later - earlier
    if not math.isfinite(difference) or difference < 0.0:
        return None
    return float(difference)


def _nonnegative_nanoseconds_duration(
    later_ns: int, earlier_ns: int
) -> float | None:
    if isinstance(later_ns, bool) or isinstance(earlier_ns, bool):
        return None
    try:
        difference_ns = int(later_ns) - int(earlier_ns)
    except (TypeError, ValueError, OverflowError):
        return None
    if difference_ns < 0:
        return None
    return float(difference_ns / 1_000_000_000.0)


def _ordered_unique(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = str(value)
        if text not in result:
            result.append(text)
    return tuple(result)


def _clock_value(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Monotonic clock must return an integer nanosecond value.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Monotonic clock must return an integer nanosecond value."
        ) from error
    if result < 0:
        raise ValueError("Monotonic clock must not return a negative value.")
    return result
