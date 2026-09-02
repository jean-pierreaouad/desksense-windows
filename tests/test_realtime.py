from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import desksense.realtime as realtime
import desksense.robustness as robustness
from desksense.streaming import DetectionResult, DetectorState, StreamingTapDetector


BASELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "lenovo-left-right-v1.json"
)
TAPNESS_BASELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "lenovo-tapness-v1.json"
)


class FakeCallbackAbort(Exception):
    """Stand-in for sounddevice.CallbackAbort used only by fake callbacks."""


class FakeStatus:
    def __init__(
        self,
        *,
        input_overflow: bool = False,
        input_underflow: bool = False,
        output_overflow: bool = False,
        output_underflow: bool = False,
        priming_output: bool = False,
    ) -> None:
        self.input_overflow = input_overflow
        self.input_underflow = input_underflow
        self.output_overflow = output_overflow
        self.output_underflow = output_underflow
        self.priming_output = priming_output

    def __bool__(self) -> bool:
        return any(
            (
                self.input_overflow,
                self.input_underflow,
                self.output_overflow,
                self.output_underflow,
                self.priming_output,
            )
        )

    def __str__(self) -> str:
        return "input overflow" if self.input_overflow else ""


def _time_info(
    *, adc: Any = 10.0, current: Any = 10.005, dac: Any = 0.0
) -> SimpleNamespace:
    return SimpleNamespace(
        inputBufferAdcTime=adc,
        currentTime=current,
        outputBufferDacTime=dac,
    )


class _CounterClock:
    def __init__(self, start: int = 1_000_000, step: int = 10_000) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> int:
        value = self.value
        self.value += self.step
        return value


class FakeInputStream:
    def __init__(self, backend: "FakeRealtimeBackend", kwargs: dict[str, Any]) -> None:
        self.backend = backend
        self.kwargs = kwargs
        self.samplerate = backend.opened_samplerate
        self.channels = backend.opened_channels
        self.dtype = backend.opened_dtype
        self.latency = backend.stream_latency
        self.time = backend.stream_time
        self.active = False
        self.stopped = True
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

    def start(self) -> "FakeInputStream":
        self.start_calls += 1
        self.backend.operations.append("stream_start")
        if self.backend.start_error is not None:
            raise self.backend.start_error
        self.active = True
        self.stopped = False
        for samples, time_info, status in self.backend.callback_packets:
            try:
                self.kwargs["callback"](
                    samples,
                    int(samples.shape[0]),
                    time_info,
                    status,
                )
            except FakeCallbackAbort:
                if not self.backend.swallow_callback_abort:
                    raise
                break
        if self.backend.become_inactive_after_start:
            self.active = False
            self.stopped = True
            finished = self.kwargs.get("finished_callback")
            if finished is not None:
                finished()
        return self

    def stop(self) -> None:
        self.stop_calls += 1
        if self.backend.stop_error is not None:
            raise self.backend.stop_error
        self.active = False
        self.stopped = True

    def close(self) -> None:
        self.close_calls += 1
        if self.backend.close_error is not None:
            raise self.backend.close_error
        self.active = False
        self.stopped = True


class FakeRealtimeBackend:
    CallbackAbort = FakeCallbackAbort

    def __init__(
        self,
        *,
        device_index: int = 7,
        endpoint_name: str | None = None,
        host_api_name: str = "Windows WDM-KS",
        max_input_channels: int = 2,
        default_samplerate: float = 48_000.0,
    ) -> None:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        expected = baseline["source_development_dataset"]["compatibility"]
        self.device_index = device_index
        self.endpoint_name = endpoint_name or str(expected["endpoint_name"])
        self.host_api_name = host_api_name
        self.max_input_channels = max_input_channels
        self.default_samplerate = default_samplerate
        self.opened_samplerate: Any = 48_000.0
        self.opened_channels: Any = 2
        self.opened_dtype: Any = "float32"
        self.stream_latency: Any = 0.0125
        self.stream_time: Any = 20.0
        self.callback_packets: list[tuple[np.ndarray, Any, Any]] = []
        self.become_inactive_after_start = False
        self.swallow_callback_abort = False
        self.start_error: Exception | None = None
        self.stop_error: Exception | None = None
        self.close_error: Exception | None = None
        self.check_input_settings_error: Exception | None = None
        self.input_stream_calls: list[dict[str, Any]] = []
        self.input_settings_calls: list[dict[str, Any]] = []
        self.streams: list[FakeInputStream] = []
        self.operations: list[str] = []

    def query_devices(self, device: int | None = None, kind: str | None = None) -> Any:
        record = {
            "index": self.device_index,
            "name": self.endpoint_name,
            "hostapi": 0,
            "max_input_channels": self.max_input_channels,
            "max_output_channels": 0,
            "default_samplerate": self.default_samplerate,
        }
        if device is None and kind is None:
            unavailable = {
                "name": "Unavailable placeholder",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 0,
                "default_samplerate": 48_000.0,
            }
            return [dict(unavailable) for _ in range(self.device_index)] + [record]
        if int(device) != self.device_index:
            raise RuntimeError("Invalid device")
        return dict(record)

    def query_hostapis(self) -> list[dict[str, Any]]:
        return [
            {
                "name": self.host_api_name,
                "default_input_device": self.device_index,
            }
        ]

    def check_input_settings(self, **kwargs: Any) -> None:
        self.operations.append("check_input_settings")
        self.input_settings_calls.append(dict(kwargs))
        if self.check_input_settings_error is not None:
            raise self.check_input_settings_error

    def InputStream(self, **kwargs: Any) -> FakeInputStream:
        self.operations.append("construct_stream")
        self.input_stream_calls.append(dict(kwargs))
        stream = FakeInputStream(self, dict(kwargs))
        self.streams.append(stream)
        return stream


class RecordingDetector:
    def __init__(
        self,
        results_by_call: Iterable[tuple[DetectionResult, ...]] = (),
    ) -> None:
        self.config = SimpleNamespace(
            sample_rate_hz=48_000.0,
            channel_count=2,
            tap_window_frames=9_600,
            history_capacity_frames=11_040,
            startup_learning_frames=36_000,
        )
        self.state = DetectorState.LEARNING
        self.stream_epoch = 0
        self.results_by_call = list(results_by_call)
        self.processed: list[np.ndarray] = []
        self.discontinuity_calls = 0
        self.operations: list[str] = []
        self.noise_floor_rms = 1.0e-5

    def process_chunk(self, samples: Any) -> tuple[DetectionResult, ...]:
        self.operations.append("process")
        self.processed.append(np.array(samples, copy=True))
        self.state = DetectorState.ARMED
        if self.results_by_call:
            return self.results_by_call.pop(0)
        return ()

    def notify_discontinuity(self) -> None:
        self.operations.append("discontinuity")
        self.discontinuity_calls += 1
        self.stream_epoch += 1
        self.state = DetectorState.LEARNING


def _candidate_result(
    *,
    status: str = "detected",
    candidate: np.ndarray | None = None,
    reasons: tuple[str, ...] = (),
    onset: int = 6_000,
    center: int = 6_010,
) -> DetectionResult:
    if candidate is None:
        candidate = np.zeros((9_600, 2), dtype=np.float32)
        candidate[4_800, 0] = 0.25
        candidate[4_800, 1] = 0.50
    return DetectionResult(
        status=status,
        rejection_reasons=reasons,
        stream_epoch=0,
        onset_frame_index=onset,
        center_frame_index=center,
        window_start_frame_index=center - 4_800,
        window_end_frame_index_exclusive=center + 4_800,
        finalized_after_frame_index_exclusive=center + 4_810,
        candidate_window=candidate,
        metrics={"detector_latency_seconds": 0.1},
        state_before=DetectorState.COLLECTING,
        state_after=DetectorState.REFRACTORY,
    )


def _run_until_processed(detector: RecordingDetector) -> Callable[[], bool]:
    return lambda: bool(detector.processed)


def _bridge(
    *, capacity: int = 8, clock: Callable[[], int] | None = None
) -> tuple[FakeRealtimeBackend, Any]:
    backend = FakeRealtimeBackend()
    bridge = realtime._InputCallbackBridge(
        capacity_packets=capacity,
        callback_abort=backend.CallbackAbort,
        clock_ns=clock or _CounterClock(),
    )
    return backend, bridge


def _invoke_callback(
    bridge: Any,
    samples: np.ndarray,
    *,
    status: Any | None = None,
    time_info: Any | None = None,
) -> None:
    bridge.callback(
        samples,
        int(samples.shape[0]),
        time_info if time_info is not None else _time_info(),
        status if status is not None else FakeStatus(),
    )


def _packet(
    *,
    sequence: int = 0,
    samples: np.ndarray | None = None,
    discontinuity_reasons: tuple[str, ...] = (),
) -> realtime.AudioChunkPacket:
    audio = (
        samples
        if samples is not None
        else np.zeros((240, 2), dtype=np.float32)
    )
    return realtime.AudioChunkPacket(
        samples=np.ascontiguousarray(audio, dtype=np.float32),
        callback_sequence=sequence,
        frame_count=int(audio.shape[0]),
        input_buffer_adc_time_seconds=10.0,
        callback_current_time_seconds=10.005,
        callback_arrival_monotonic_ns=1_000_000,
        enqueue_attempt_monotonic_ns=1_010_000,
        portaudio_status_flags=(),
        discontinuity_before=bool(discontinuity_reasons),
        discontinuity_reasons=discontinuity_reasons,
        dropped_callback_count_before=0,
        dropped_frame_count_before=0,
    )


def _timing_fixture(
    *,
    packet: realtime.AudioChunkPacket | None = None,
    result: DetectionResult | None = None,
    spans: deque[realtime._TimingSpan] | None = None,
    stream_time: Any = 82_341.0,
    processing_start_ns: int = 1_010_000_000,
    result_available_ns: int = 1_020_000_000,
) -> dict[str, int | float | None]:
    timing_packet = packet or replace(
        _packet(),
        input_buffer_adc_time_seconds=10.0,
        callback_current_time_seconds=10.1,
        callback_arrival_monotonic_ns=1_000_000_000,
        enqueue_attempt_monotonic_ns=1_005_000_000,
    )
    timing_result = result or _candidate_result(onset=0, center=1_200)
    timing_spans = spans
    if timing_spans is None:
        timing_spans = deque(
            [
                realtime._TimingSpan(
                    stream_epoch=0,
                    start_frame_index=0,
                    end_frame_index_exclusive=2_400,
                    input_buffer_adc_time_seconds=10.0,
                )
            ]
        )
    return realtime._result_timing(
        timing_packet,
        timing_result,
        timing_spans,
        SimpleNamespace(time=stream_time),
        processing_start_ns=processing_start_ns,
        result_available_ns=result_available_ns,
    )


def test_result_timing_keeps_raw_components_without_combined_latency() -> None:
    timing = _timing_fixture(stream_time=82_341.0)

    assert timing["estimated_onset_adc_time_seconds"] == pytest.approx(10.0)
    assert timing["estimated_center_adc_time_seconds"] == pytest.approx(10.025)
    assert timing["portaudio_onset_to_callback_seconds"] == pytest.approx(0.1)
    assert timing["portaudio_center_to_callback_seconds"] == pytest.approx(0.075)
    assert timing["python_callback_to_result_seconds"] == pytest.approx(0.02)
    assert "approximate_onset_to_result_seconds" not in timing
    assert "approximate_center_to_result_seconds" not in timing
    assert timing["stream_time_at_result_seconds"] == pytest.approx(82_341.0)


def test_huge_unrelated_stream_time_cannot_create_combined_timing() -> None:
    ordinary = _timing_fixture(stream_time=10.2)
    unrelated = _timing_fixture(stream_time=82_341_000_000.0)

    for timing in (ordinary, unrelated):
        assert "approximate_onset_to_result_seconds" not in timing
        assert "approximate_center_to_result_seconds" not in timing
        assert timing["portaudio_onset_to_callback_seconds"] == pytest.approx(0.1)
        assert timing["python_callback_to_result_seconds"] == pytest.approx(0.02)


def test_missing_callback_current_time_makes_portaudio_durations_unavailable() -> None:
    packet = replace(
        _packet(),
        callback_current_time_seconds=None,
        callback_arrival_monotonic_ns=1_000_000_000,
    )
    timing = _timing_fixture(packet=packet)

    assert timing["portaudio_onset_to_callback_seconds"] is None
    assert timing["portaudio_center_to_callback_seconds"] is None


def test_missing_onset_mapping_does_not_hide_valid_center_mapping() -> None:
    timing = _timing_fixture(result=_candidate_result(onset=3_000, center=1_200))

    assert timing["estimated_onset_adc_time_seconds"] is None
    assert timing["portaudio_onset_to_callback_seconds"] is None
    assert timing["portaudio_center_to_callback_seconds"] == pytest.approx(0.075)


def test_missing_center_mapping_makes_only_center_timing_unavailable() -> None:
    result = replace(
        _candidate_result(onset=0, center=1_200),
        center_frame_index=None,
    )
    timing = _timing_fixture(result=result)

    assert timing["portaudio_onset_to_callback_seconds"] == pytest.approx(0.1)
    assert timing["estimated_center_adc_time_seconds"] is None
    assert timing["portaudio_center_to_callback_seconds"] is None


@pytest.mark.parametrize("callback_time", [float("nan"), float("inf")])
def test_nonfinite_portaudio_callback_time_makes_portaudio_durations_unavailable(
    callback_time: float,
) -> None:
    packet = replace(
        _packet(),
        callback_current_time_seconds=callback_time,
        callback_arrival_monotonic_ns=1_000_000_000,
    )
    timing = _timing_fixture(packet=packet)

    assert timing["portaudio_onset_to_callback_seconds"] is None
    assert timing["portaudio_center_to_callback_seconds"] is None


def test_callback_time_earlier_than_adc_estimates_fails_safely() -> None:
    packet = replace(
        _packet(),
        callback_current_time_seconds=9.9,
        callback_arrival_monotonic_ns=1_000_000_000,
    )
    timing = _timing_fixture(packet=packet)

    assert timing["portaudio_onset_to_callback_seconds"] is None
    assert timing["portaudio_center_to_callback_seconds"] is None


def test_result_timestamp_before_callback_arrival_fails_safely() -> None:
    timing = _timing_fixture(result_available_ns=999_000_000)

    assert timing["python_callback_to_result_seconds"] is None
    assert "approximate_onset_to_result_seconds" not in timing
    assert "approximate_center_to_result_seconds" not in timing


def test_queue_dwell_and_detector_lookahead_timing_are_unchanged() -> None:
    timing = _timing_fixture()

    assert timing["queue_dwell_seconds"] == pytest.approx(0.005)
    assert timing["detector_latency_seconds"] == pytest.approx(0.1)


def test_cli_omits_combined_timing_even_if_bogus_value_is_supplied() -> None:
    from desksense.cli import format_live_sensing_event

    timing = _timing_fixture(
        packet=replace(
            _packet(),
            callback_current_time_seconds=None,
            callback_arrival_monotonic_ns=1_000_000_000,
            enqueue_attempt_monotonic_ns=1_005_000_000,
        ),
        stream_time=82_341.0,
    )
    timing["approximate_onset_to_result_seconds"] = 84_714.0
    timing["approximate_center_to_result_seconds"] = 84_713.9
    event = realtime.LiveSensingEvent(
        event_type="detection",
        status="detected",
        message="detected",
        stream_epoch=0,
        predicted_zone="RIGHT",
        feature_value_db=2.0,
        threshold_db=0.127932,
        absolute_margin_db=1.872068,
        onset_frame_index=100,
        center_frame_index=120,
        timing=timing,
    )

    rendered = format_live_sensing_event(event)

    assert "onset-to-result" not in rendered
    assert "84714000" not in rendered
    assert "queue dwell=5.00 ms" in rendered
    assert "detector lookahead=100.00 ms" in rendered


def test_realtime_transport_defaults_are_explicit_and_bounded() -> None:
    config = realtime.RealtimeSensingConfig()

    assert config.queue_capacity_packets == 8
    assert config.queue_poll_timeout_seconds == pytest.approx(0.1)
    _, bridge = _bridge()
    assert bridge.queue.maxsize == 8


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_queue_capacity_is_rejected(capacity: Any) -> None:
    with pytest.raises(ValueError, match="queue capacity"):
        realtime.RealtimeSensingConfig(queue_capacity_packets=capacity)


def test_real_frozen_baseline_loads_as_exact_live_spec() -> None:
    spec = realtime.load_live_baseline_spec(BASELINE_PATH)

    assert spec.sample_rate_hz == 48_000.0
    assert spec.channel_count == 2
    assert spec.tap_window_frames == 9_600
    assert spec.host_api_name == "Windows WDM-KS"
    assert spec.threshold_db == pytest.approx(0.12793235855251162)
    assert (spec.lower_feature_zone, spec.higher_feature_zone) == (
        "LEFT",
        "RIGHT",
    )


def test_callback_owns_exactly_one_contiguous_float32_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bridge = _bridge()
    original = np.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64
    )
    before = original.copy()
    real_array = np.array
    copy_calls: list[dict[str, Any]] = []

    def recording_array(value: Any, **kwargs: Any) -> np.ndarray:
        copy_calls.append(dict(kwargs))
        return real_array(value, **kwargs)

    monkeypatch.setattr(realtime.np, "array", recording_array)
    _invoke_callback(bridge, original)
    packet = bridge.queue.get_nowait()

    assert len(copy_calls) == 1
    assert copy_calls[0] == {
        "dtype": np.float32,
        "order": "C",
        "copy": True,
    }
    assert np.array_equal(original, before)
    assert packet.samples.dtype == np.float32
    assert packet.samples.flags.c_contiguous
    assert np.array_equal(packet.samples, before.astype(np.float32))
    assert np.array_equal(packet.samples[:, 0], (1.0, 2.0, 3.0))
    assert np.array_equal(packet.samples[:, 1], (10.0, 20.0, 30.0))
    assert not np.shares_memory(packet.samples, original)


def test_callback_copy_survives_portaudio_buffer_reuse() -> None:
    _, bridge = _bridge()
    original = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    expected = original.copy()

    _invoke_callback(bridge, original)
    original[:] = -99.0

    assert np.array_equal(bridge.queue.get_nowait().samples, expected)


def test_callback_does_no_detector_feature_inference_or_output_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("main-thread work ran inside the callback")

    monkeypatch.setattr(realtime.StreamingTapDetector, "process_chunk", forbidden)
    monkeypatch.setattr(realtime, "extract_descriptive_tapness_metrics", forbidden)
    monkeypatch.setattr(realtime, "classify_tapness_metrics", forbidden)
    monkeypatch.setattr(realtime, "extract_two_channel_features", forbidden)
    monkeypatch.setattr(realtime, "classify_peak_ratio_value", forbidden)
    _, bridge = _bridge()

    _invoke_callback(bridge, np.zeros((4, 2), dtype=np.float32))

    assert bridge.queue.qsize() == 1


def test_callback_snapshots_primitive_timing_status_and_sequence() -> None:
    clock = _CounterClock(start=100, step=7)
    _, bridge = _bridge(clock=clock)

    _invoke_callback(
        bridge,
        np.zeros((5, 2), dtype=np.float32),
        status=FakeStatus(output_underflow=True),
        time_info=_time_info(adc=1.25, current=1.30, dac=0.0),
    )
    packet = bridge.queue.get_nowait()

    assert packet.callback_sequence == 0
    assert packet.frame_count == 5
    assert packet.input_buffer_adc_time_seconds == 1.25
    assert packet.callback_current_time_seconds == 1.30
    assert packet.callback_arrival_monotonic_ns == 100
    assert packet.enqueue_attempt_monotonic_ns == 107
    assert packet.portaudio_status_flags == ("output_underflow",)


def test_malformed_or_unavailable_portaudio_times_do_not_reject_audio() -> None:
    _, bridge = _bridge()

    _invoke_callback(
        bridge,
        np.zeros((3, 2), dtype=np.float32),
        time_info=_time_info(adc="unknown", current=float("nan"), dac=-1.0),
    )
    packet = bridge.queue.get_nowait()

    assert packet.input_buffer_adc_time_seconds is None
    assert packet.callback_current_time_seconds is None
    assert packet.samples.shape == (3, 2)


def test_queue_full_drops_newest_without_removing_older_packet() -> None:
    _, bridge = _bridge(capacity=1)
    first = np.full((4, 2), 1.0, dtype=np.float32)
    dropped = np.full((6, 2), 2.0, dtype=np.float32)

    _invoke_callback(bridge, first)
    _invoke_callback(bridge, dropped)

    retained = bridge.queue.get_nowait()
    stats = bridge.statistics()
    assert retained.callback_sequence == 0
    assert np.array_equal(retained.samples, first)
    assert stats["queue_high_water_mark"] == 1
    assert stats["dropped_callback_count"] == 1
    assert stats["dropped_frame_count"] == 6


def test_next_retained_packet_explicitly_accumulates_queue_drop_boundary() -> None:
    _, bridge = _bridge(capacity=1)
    _invoke_callback(bridge, np.zeros((3, 2), dtype=np.float32))
    _invoke_callback(bridge, np.zeros((4, 2), dtype=np.float32))
    _invoke_callback(bridge, np.zeros((5, 2), dtype=np.float32))
    bridge.queue.get_nowait()

    _invoke_callback(bridge, np.ones((6, 2), dtype=np.float32))
    packet = bridge.queue.get_nowait()

    assert packet.callback_sequence == 3
    assert packet.discontinuity_before is True
    assert packet.discontinuity_reasons == ("queue_overflow",)
    assert packet.dropped_callback_count_before == 2
    assert packet.dropped_frame_count_before == 9


def test_portaudio_input_overflow_marks_affected_retained_packet() -> None:
    _, bridge = _bridge()

    _invoke_callback(
        bridge,
        np.zeros((8, 2), dtype=np.float32),
        status=FakeStatus(input_overflow=True),
    )
    packet = bridge.queue.get_nowait()

    assert packet.portaudio_status_flags == ("input_overflow",)
    assert packet.discontinuity_before is True
    assert packet.discontinuity_reasons == ("portaudio_input_overflow",)


def test_callback_internal_error_is_preserved_and_aborted() -> None:
    backend, bridge = _bridge()

    with pytest.raises(backend.CallbackAbort):
        bridge.callback(
            np.zeros((2, 2), dtype=np.float32),
            -1,
            _time_info(),
            FakeStatus(),
        )

    assert bridge.fatal_error is not None
    assert isinstance(bridge.fatal_error, ValueError)
    with pytest.raises(realtime.RealtimeSensingError, match="callback failed"):
        realtime._raise_callback_failure(bridge)


def _stop_after_processed(
    detector: RecordingDetector, count: int
) -> Callable[[], bool]:
    return lambda: len(detector.processed) >= count


class _PreloadedBridge:
    def __init__(self, packets: Iterable[realtime.AudioChunkPacket]) -> None:
        packet_list = list(packets)
        self.queue: queue.Queue[realtime.AudioChunkPacket] = queue.Queue()
        for packet in packet_list:
            self.queue.put_nowait(packet)
        self.fatal_error: BaseException | None = None
        self.callback = lambda *args, **kwargs: None
        self._packet_list = packet_list

    def statistics(self) -> dict[str, int]:
        return {
            "callback_count": len(self._packet_list),
            "callback_frames": sum(
                packet.frame_count for packet in self._packet_list
            ),
            "dropped_callback_count": 0,
            "dropped_frame_count": 0,
            "queue_high_water_mark": len(self._packet_list),
        }


def _install_preloaded_bridge(
    monkeypatch: pytest.MonkeyPatch,
    packets: Iterable[realtime.AudioChunkPacket],
) -> _PreloadedBridge:
    bridge = _PreloadedBridge(packets)
    monkeypatch.setattr(
        realtime,
        "_InputCallbackBridge",
        lambda **kwargs: bridge,
    )
    return bridge


def test_live_runner_opens_exact_frozen_domain_after_settings_check() -> None:
    backend = FakeRealtimeBackend(device_index=31)
    detector = RecordingDetector()
    events: list[realtime.LiveSensingEvent] = []

    summary = realtime.run_live_sensing(
        backend,
        device_index=31,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        stop_requested=lambda: True,
    )

    assert backend.input_settings_calls[-1] == (
        {
            "device": 31,
            "channels": 2,
            "dtype": "float32",
            "samplerate": 48_000.0,
        }
    )
    assert backend.operations[-3:] == [
        "check_input_settings", "construct_stream", "stream_start"
    ]
    stream_args = backend.input_stream_calls[0]
    assert stream_args.keys() == {
        "device",
        "samplerate",
        "channels",
        "dtype",
        "blocksize",
        "callback",
        "finished_callback",
    }
    assert stream_args["device"] == 31
    assert stream_args["samplerate"] == 48_000.0
    assert stream_args["channels"] == 2
    assert stream_args["dtype"] == "float32"
    assert stream_args["blocksize"] == 0
    assert callable(stream_args["callback"])
    assert callable(stream_args["finished_callback"])
    assert events[0].event_type == "startup"
    assert summary.packets_processed == 0
    assert backend.streams[0].stop_calls == 1
    assert backend.streams[0].close_calls == 1


def test_changed_device_index_is_allowed_when_endpoint_identity_matches() -> None:
    backend = FakeRealtimeBackend(device_index=73)

    realtime.run_live_sensing(
        backend,
        device_index=73,
        baseline_path=BASELINE_PATH,
        event_handler=lambda event: None,
        detector=RecordingDetector(),
        stop_requested=lambda: True,
    )

    assert backend.streams[0].start_calls == 1


@pytest.mark.parametrize(
    ("backend", "match"),
    [
        (FakeRealtimeBackend(endpoint_name="Different endpoint"), "endpoint name"),
        (FakeRealtimeBackend(host_api_name="Windows WASAPI"), "host API name"),
        (FakeRealtimeBackend(max_input_channels=1), "input channel count"),
    ],
)
def test_endpoint_incompatibility_fails_before_stream_construction(
    backend: FakeRealtimeBackend, match: str
) -> None:
    with pytest.raises(realtime.RealtimeSensingError, match=match):
        realtime.run_live_sensing(
            backend,
            device_index=backend.device_index,
            baseline_path=BASELINE_PATH,
            event_handler=lambda event: None,
            detector=RecordingDetector(),
            stop_requested=lambda: True,
        )

    assert backend.input_stream_calls == []


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        ("opened_samplerate", 44_100.0, "opened sample rate"),
        ("opened_channels", 1, "opened channel count"),
        ("opened_dtype", "float64", "opened dtype"),
    ],
)
def test_opened_stream_domain_mismatch_fails_before_start_and_closes(
    attribute: str, value: Any, match: str
) -> None:
    backend = FakeRealtimeBackend()
    setattr(backend, attribute, value)

    with pytest.raises(realtime.RealtimeSensingError, match=match):
        realtime.run_live_sensing(
            backend,
            device_index=backend.device_index,
            baseline_path=BASELINE_PATH,
            event_handler=lambda event: None,
            detector=RecordingDetector(),
            stop_requested=lambda: True,
        )

    assert backend.streams[0].start_calls == 0
    assert backend.streams[0].stop_calls == 0
    assert backend.streams[0].close_calls == 1


def test_callback_packets_reach_detector_in_sequence_with_shape_unchanged() -> None:
    backend = FakeRealtimeBackend()
    first = np.arange(14, dtype=np.float32).reshape(7, 2)
    second = np.arange(22, dtype=np.float32).reshape(11, 2) + 100
    backend.callback_packets = [
        (first, _time_info(), FakeStatus()),
        (second, _time_info(adc=10.1, current=10.105), FakeStatus()),
    ]
    detector = RecordingDetector()

    summary = realtime.run_live_sensing(
        backend,
        device_index=backend.device_index,
        baseline_path=BASELINE_PATH,
        event_handler=lambda event: None,
        detector=detector,
        stop_requested=_stop_after_processed(detector, 2),
    )

    assert len(detector.processed) == 2
    assert np.array_equal(detector.processed[0], first)
    assert np.array_equal(detector.processed[1], second)
    assert summary.packets_processed == 2


def test_explicit_discontinuity_resets_before_post_gap_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = RecordingDetector()
    packet = _packet(
        sequence=0,
        samples=np.ones((9, 2), dtype=np.float32),
        discontinuity_reasons=("portaudio_input_overflow",),
    )
    _install_preloaded_bridge(monkeypatch, [packet])

    realtime.run_live_sensing(
        FakeRealtimeBackend(),
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=lambda event: None,
        detector=detector,
        stop_requested=_stop_after_processed(detector, 1),
    )

    assert detector.operations == ["discontinuity", "process"]
    assert detector.discontinuity_calls == 1


def test_sequence_gap_is_secondary_continuity_check_and_resets_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = RecordingDetector()
    packet = replace(
        _packet(
            sequence=2,
            discontinuity_reasons=("portaudio_input_overflow",),
        ),
        discontinuity_before=True,
    )
    events: list[realtime.LiveSensingEvent] = []
    _install_preloaded_bridge(monkeypatch, [packet])

    summary = realtime.run_live_sensing(
        FakeRealtimeBackend(),
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        stop_requested=_stop_after_processed(detector, 1),
    )

    assert detector.discontinuity_calls == 1
    discontinuity = next(e for e in events if e.event_type == "discontinuity")
    assert discontinuity.rejection_reasons == (
        "portaudio_input_overflow",
        "callback_sequence_gap",
    )
    assert summary.discontinuity_count == 1


class _GapSensitiveDetector(RecordingDetector):
    def __init__(self) -> None:
        super().__init__()
        self.accumulated: list[np.ndarray] = []

    def process_chunk(self, samples: Any) -> tuple[DetectionResult, ...]:
        result = super().process_chunk(samples)
        self.accumulated.append(np.array(samples, copy=True))
        return result

    def notify_discontinuity(self) -> None:
        super().notify_discontinuity()
        self.accumulated.clear()


def test_detector_audio_history_cannot_bridge_a_transport_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _GapSensitiveDetector()
    before = _packet(sequence=0, samples=np.full((4, 2), 1.0, np.float32))
    after = _packet(
        sequence=1,
        samples=np.full((5, 2), 2.0, np.float32),
        discontinuity_reasons=("queue_overflow",),
    )
    _install_preloaded_bridge(monkeypatch, [before, after])

    realtime.run_live_sensing(
        FakeRealtimeBackend(),
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=lambda event: None,
        detector=detector,
        stop_requested=_stop_after_processed(detector, 2),
    )

    assert len(detector.accumulated) == 1
    assert np.array_equal(detector.accumulated[0], after.samples)


def test_rejected_candidate_with_window_is_never_classified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = np.ones((9_600, 2), dtype=np.float32)
    detector = RecordingDetector(
        [(_candidate_result(status="rejected", candidate=candidate,
                            reasons=("near_clipping",)),)]
    )
    backend = FakeRealtimeBackend()
    backend.callback_packets = [
        (np.zeros((240, 2), np.float32), _time_info(), FakeStatus())
    ]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("rejected audio reached feature extraction")

    monkeypatch.setattr(realtime, "extract_two_channel_features", forbidden)
    events: list[realtime.LiveSensingEvent] = []
    realtime.run_live_sensing(
        backend,
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        stop_requested=_stop_after_processed(detector, 1),
    )

    rejection = next(e for e in events if e.event_type == "rejection")
    assert rejection.rejection_reasons == ("near_clipping",)
    assert not any(e.event_type == "detection" for e in events)


def test_valid_candidate_uses_existing_features_and_frozen_baseline() -> None:
    candidate = np.zeros((9_600, 2), dtype=np.float32)
    candidate[4_800] = (0.25, 0.50)
    detector = RecordingDetector([(_candidate_result(candidate=candidate),)])
    backend = FakeRealtimeBackend()
    backend.callback_packets = [
        (np.zeros((240, 2), np.float32), _time_info(), FakeStatus())
    ]
    events: list[realtime.LiveSensingEvent] = []

    summary = realtime.run_live_sensing(
        backend,
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        stop_requested=_stop_after_processed(detector, 1),
    )

    detection = next(e for e in events if e.event_type == "detection")
    assert detection.predicted_zone == "RIGHT"
    assert detection.feature_value_db == pytest.approx(20.0 * np.log10(2.0))
    assert detection.threshold_db == pytest.approx(0.12793235855251162)
    assert detection.absolute_margin_db == pytest.approx(
        detection.feature_value_db - detection.threshold_db
    )
    assert summary.detected_event_count == 1


def test_stage2_rejection_blocks_spatial_feature_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _candidate_result()
    spatial = realtime.load_live_baseline_spec(BASELINE_PATH)
    tapness = realtime.load_tapness_baseline(TAPNESS_BASELINE_PATH)

    monkeypatch.setattr(
        realtime,
        "classify_tapness_metrics",
        lambda *args, **kwargs: {
            "predicted_label": "NON_TAP",
            "tap_accepted": False,
            "uncalibrated_model_output": 0.1,
            "decision_score": -2.0,
            "decision_threshold": 0.4,
            "model_margin": -0.3,
            "tie_rule": "score >= threshold predicts TAP",
            "score_is_calibrated_probability": False,
        },
    )
    monkeypatch.setattr(
        realtime,
        "extract_two_channel_features",
        lambda *args, **kwargs: pytest.fail("Stage 3 must not run"),
    )

    event = realtime.process_detection_result(
        result, spatial, tapness_baseline=tapness
    )

    assert event.status == "rejected"
    assert event.rejection_reasons == ("tapness_non_tap",)
    assert event.predicted_zone is None


def test_stage2_acceptance_invokes_unchanged_spatial_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _candidate_result()
    spatial = realtime.load_live_baseline_spec(BASELINE_PATH)
    tapness = realtime.load_tapness_baseline(TAPNESS_BASELINE_PATH)
    monkeypatch.setattr(
        realtime,
        "classify_tapness_metrics",
        lambda *args, **kwargs: {
            "predicted_label": "TAP",
            "tap_accepted": True,
            "uncalibrated_model_output": 0.9,
            "decision_score": 2.0,
            "decision_threshold": 0.4,
            "model_margin": 0.5,
            "tie_rule": "score >= threshold predicts TAP",
            "score_is_calibrated_probability": False,
        },
    )

    event = realtime.process_detection_result(
        result, spatial, tapness_baseline=tapness
    )

    assert event.status == "detected"
    assert event.predicted_zone == "RIGHT"
    assert event.details["tapness_inference"]["tap_accepted"] is True


def test_live_and_offline_stage2_predictions_match_for_same_candidate() -> None:
    result = _candidate_result()
    spatial = realtime.load_live_baseline_spec(BASELINE_PATH)
    tapness = realtime.load_tapness_baseline(TAPNESS_BASELINE_PATH)

    live = realtime.process_detection_result(
        result, spatial, tapness_baseline=tapness
    )
    offline = robustness._completed_event_report(
        result,
        None,
        None,
        tapness_baseline=tapness,
        interval_relation="associated",
    )

    assert live.details["tapness_metrics"] == offline["descriptive_tapness_metrics"]
    assert live.details["tapness_inference"] == offline["frozen_tapness_prediction"]


def test_undefined_primary_feature_is_structured_rejection() -> None:
    detector = RecordingDetector(
        [(_candidate_result(candidate=np.zeros((9_600, 2), np.float32)),)]
    )
    backend = FakeRealtimeBackend()
    backend.callback_packets = [
        (np.zeros((240, 2), np.float32), _time_info(), FakeStatus())
    ]
    events: list[realtime.LiveSensingEvent] = []

    realtime.run_live_sensing(
        backend,
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        stop_requested=_stop_after_processed(detector, 1),
    )

    rejection = next(e for e in events if e.event_type == "rejection")
    assert rejection.rejection_reasons == ("undefined_primary_feature",)
    assert rejection.predicted_zone is None


def test_multiple_detection_results_are_emitted_in_detector_order() -> None:
    right = np.zeros((9_600, 2), np.float32)
    right[4_800] = (0.2, 0.4)
    left = np.zeros((9_600, 2), np.float32)
    left[4_800] = (0.4, 0.2)
    detector = RecordingDetector(
        [
            (
                _candidate_result(candidate=right, onset=100, center=110),
                _candidate_result(candidate=left, onset=200, center=210),
            )
        ]
    )
    backend = FakeRealtimeBackend()
    backend.callback_packets = [
        (np.zeros((240, 2), np.float32), _time_info(), FakeStatus())
    ]
    events: list[realtime.LiveSensingEvent] = []

    realtime.run_live_sensing(
        backend,
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        stop_requested=_stop_after_processed(detector, 1),
    )

    detections = [e for e in events if e.event_type == "detection"]
    assert [(e.onset_frame_index, e.predicted_zone) for e in detections] == [
        (100, "RIGHT"),
        (200, "LEFT"),
    ]


def test_startup_and_armed_events_repeat_only_after_discontinuity() -> None:
    backend = FakeRealtimeBackend()
    backend.callback_packets = [
        (np.zeros((4, 2), np.float32), _time_info(), FakeStatus()),
        (
            np.zeros((5, 2), np.float32),
            _time_info(adc=10.1, current=10.105),
            FakeStatus(input_overflow=True),
        ),
    ]
    detector = RecordingDetector()
    events: list[realtime.LiveSensingEvent] = []

    realtime.run_live_sensing(
        backend,
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        stop_requested=_stop_after_processed(detector, 2),
    )

    assert [e.event_type for e in events].count("startup") == 1
    assert [e.status for e in events].count("armed") == 2
    assert [e.event_type for e in events].count("discontinuity") == 1


def test_queue_drop_statistics_are_returned() -> None:
    backend = FakeRealtimeBackend()
    backend.callback_packets = [
        (np.zeros((3, 2), np.float32), _time_info(), FakeStatus()),
        (np.zeros((5, 2), np.float32), _time_info(), FakeStatus()),
    ]
    detector = RecordingDetector()

    summary = realtime.run_live_sensing(
        backend,
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=lambda event: None,
        detector=detector,
        config=realtime.RealtimeSensingConfig(queue_capacity_packets=1),
        stop_requested=_stop_after_processed(detector, 1),
    )

    assert summary.callback_count == 2
    assert summary.callback_frames == 8
    assert summary.dropped_callback_count == 1
    assert summary.dropped_frame_count == 5
    assert summary.queue_high_water_mark == 1


def test_callback_fatal_error_reaches_main_loop_and_stream_closes() -> None:
    backend = FakeRealtimeBackend()
    backend.swallow_callback_abort = True
    backend.callback_packets = [
        (np.zeros((4,), np.float32), _time_info(), FakeStatus())
    ]

    with pytest.raises(realtime.RealtimeSensingError, match="callback failed"):
        realtime.run_live_sensing(
            backend,
            device_index=7,
            baseline_path=BASELINE_PATH,
            event_handler=lambda event: None,
            detector=RecordingDetector(),
            stop_requested=lambda: False,
        )

    assert backend.streams[0].stop_calls == 1
    assert backend.streams[0].close_calls == 1


def test_unexpected_finished_stream_is_reported_and_closed() -> None:
    backend = FakeRealtimeBackend()
    backend.become_inactive_after_start = True

    with pytest.raises(realtime.RealtimeSensingError, match="inactive"):
        realtime.run_live_sensing(
            backend,
            device_index=7,
            baseline_path=BASELINE_PATH,
            event_handler=lambda event: None,
            detector=RecordingDetector(),
            config=realtime.RealtimeSensingConfig(
                queue_poll_timeout_seconds=0.001
            ),
            stop_requested=lambda: False,
        )

    assert backend.streams[0].close_calls == 1


def test_keyboard_interrupt_stops_and_closes_stream() -> None:
    backend = FakeRealtimeBackend()

    def interrupt() -> bool:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        realtime.run_live_sensing(
            backend,
            device_index=7,
            baseline_path=BASELINE_PATH,
            event_handler=lambda event: None,
            detector=RecordingDetector(),
            stop_requested=interrupt,
        )

    assert backend.streams[0].stop_calls == 1
    assert backend.streams[0].close_calls == 1


def test_start_failure_closes_constructed_stream_without_stopping() -> None:
    backend = FakeRealtimeBackend()
    backend.start_error = RuntimeError("start denied")

    with pytest.raises(realtime.RealtimeSensingError, match="start denied"):
        realtime.run_live_sensing(
            backend,
            device_index=7,
            baseline_path=BASELINE_PATH,
            event_handler=lambda event: None,
            detector=RecordingDetector(),
            stop_requested=lambda: True,
        )

    assert backend.streams[0].stop_calls == 0
    assert backend.streams[0].close_calls == 1


def test_cleanup_failure_does_not_mask_primary_sensing_error() -> None:
    backend = FakeRealtimeBackend()
    backend.stop_error = RuntimeError("secondary cleanup failure")

    def primary_error() -> bool:
        raise realtime.RealtimeSensingError(
            "primary sensing failure", category="primary"
        )

    with pytest.raises(realtime.RealtimeSensingError, match="primary sensing"):
        realtime.run_live_sensing(
            backend,
            device_index=7,
            baseline_path=BASELINE_PATH,
            event_handler=lambda event: None,
            detector=RecordingDetector(),
            stop_requested=primary_error,
        )

    assert backend.streams[0].close_calls == 1


def test_live_runner_calls_no_audio_or_report_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("live sensing attempted filesystem persistence")

    monkeypatch.setattr(np, "save", forbidden)
    monkeypatch.setattr(np, "savez", forbidden)
    monkeypatch.setattr(np, "savez_compressed", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    backend = FakeRealtimeBackend()

    realtime.run_live_sensing(
        backend,
        device_index=7,
        baseline_path=BASELINE_PATH,
        event_handler=lambda event: None,
        detector=RecordingDetector(),
        stop_requested=lambda: True,
    )


def test_live_diagnostics_emit_candidate_start_summary_then_detection() -> None:
    backend = FakeRealtimeBackend()
    audio = np.zeros((72_000, 2), dtype=np.float32)
    audio[40_003] = (0.25, 0.05)
    backend.callback_packets = [
        (audio, _time_info(adc=10.0, current=11.5), FakeStatus())
    ]
    detector = StreamingTapDetector()
    events: list[realtime.LiveSensingEvent] = []

    realtime.run_live_sensing(
        backend,
        device_index=backend.device_index,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        diagnostics_enabled=True,
        stop_requested=lambda: detector.processed_frame_count > 0,
    )

    statuses = [(event.event_type, event.status) for event in events]
    start_index = statuses.index(("diagnostic", "candidate_started"))
    summary_index = statuses.index(("diagnostic", "summary"))
    detection_index = statuses.index(("detection", "detected"))
    assert start_index < detection_index
    assert summary_index < detection_index

    candidate_start = events[start_index]
    assert candidate_start.onset_frame_index == 40_003
    gate = candidate_start.details["gate_block"]
    assert gate["block_start_frame_index"] == 39_840
    assert gate["block_end_frame_index_exclusive"] == 40_080
    assert gate["all_gates_score"] >= 1.0

    diagnostic_summary = events[summary_index]
    counts = diagnostic_summary.details["interval_counters"]
    assert counts["onset_candidates_started"] == 1
    assert counts["completed_detections"] == 1
    assert counts["all_gates_pass_count"] == 1
    transport = diagnostic_summary.details["transport"]
    assert transport["callback_count"] == 1
    assert transport["callback_frames"] == 72_000
    assert transport["queue_high_water_mark"] == 1
    assert transport["dropped_callback_count"] == 0


def test_live_diagnostics_are_opt_in() -> None:
    backend = FakeRealtimeBackend()
    audio = np.zeros((72_000, 2), dtype=np.float32)
    audio[40_003] = (0.25, 0.05)
    backend.callback_packets = [(audio, _time_info(), FakeStatus())]
    detector = StreamingTapDetector()
    events: list[realtime.LiveSensingEvent] = []

    realtime.run_live_sensing(
        backend,
        device_index=backend.device_index,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        stop_requested=lambda: detector.processed_frame_count > 0,
    )

    assert not any(event.event_type == "diagnostic" for event in events)
    assert any(event.event_type == "detection" for event in events)


def test_diagnostic_discontinuity_reports_pending_candidate_discard() -> None:
    backend = FakeRealtimeBackend()
    first = np.zeros((40_240, 2), dtype=np.float32)
    first[40_003] = (0.25, 0.05)
    second = np.zeros((240, 2), dtype=np.float32)
    backend.callback_packets = [
        (first, _time_info(), FakeStatus()),
        (
            second,
            _time_info(adc=11.0, current=11.005),
            FakeStatus(input_overflow=True),
        ),
    ]
    detector = StreamingTapDetector()
    events: list[realtime.LiveSensingEvent] = []

    realtime.run_live_sensing(
        backend,
        device_index=backend.device_index,
        baseline_path=BASELINE_PATH,
        event_handler=events.append,
        detector=detector,
        diagnostics_enabled=True,
        stop_requested=lambda: any(
            event.event_type == "discontinuity" for event in events
        ),
    )

    discontinuity = next(
        event for event in events if event.event_type == "discontinuity"
    )
    ended = discontinuity.details["ended_detector_epoch"]
    assert ended["detector_state"] == "collecting"
    assert ended["cumulative_counters"]["onset_candidates_started"] == 1
    assert (
        ended["cumulative_counters"][
            "candidates_discarded_by_discontinuity"
        ]
        == 1
    )
    assert detector.stream_epoch == 1
    assert (
        detector.diagnostic_snapshot().cumulative_counters.fixed_blocks_processed_total
        == 1
    )
