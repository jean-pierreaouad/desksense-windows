from __future__ import annotations

import json

import numpy as np
import pytest

from desksense.collection import (
    CollectionParameters,
    alternating_collection_order,
    assess_capture_quality,
    normalize_zones,
    record_collection_attempt,
    terminal_countdown,
)


def _parameters() -> CollectionParameters:
    return CollectionParameters(
        capture_duration_seconds=1.5,
        tap_window_seconds=0.200,
        transient_energy_window_seconds=0.005,
    )


def _clear_tap(
    *,
    frames: int = 1_500,
    channels: int = 2,
    center: int = 750,
    amplitude: float = 0.1,
) -> np.ndarray:
    samples = np.zeros((frames, channels), dtype=np.float32)
    samples[center, 0] = amplitude
    if channels > 1:
        samples[center, 1] = amplitude * 0.6
    return samples


def _reason_codes(assessment) -> set[str]:
    return {reason["code"] for reason in assessment.reasons}


def test_collection_order_alternates_zones() -> None:
    assert alternating_collection_order(("left", "right"), 3) == (
        "LEFT",
        "RIGHT",
        "LEFT",
        "RIGHT",
        "LEFT",
        "RIGHT",
    )


def test_zone_validation_rejects_duplicates_and_path_like_labels() -> None:
    with pytest.raises(ValueError, match="unique"):
        normalize_zones(("left", "LEFT"))
    with pytest.raises(ValueError, match="Zone labels"):
        normalize_zones(("../left", "right"))


def test_clear_transient_is_accepted_with_exact_centered_window() -> None:
    samples = _clear_tap()

    assessment = assess_capture_quality(
        samples,
        1_000,
        expected_channels=2,
        parameters=_parameters(),
    )

    assert assessment.accepted is True
    assert assessment.reasons == ()
    assert assessment.metrics["inactive_required_channels"] == []
    assert assessment.metrics["all_expected_channels_active"] is True
    assert assessment.transient is not None
    assert assessment.transient["center_sample"] == 750
    assert assessment.transient["exact_tap_window_start_sample"] == 650
    assert assessment.transient["exact_tap_window_end_sample_exclusive"] == 850
    assert assessment.tap_window is not None
    assert assessment.tap_window.shape == (200, 2)
    assert assessment.tap_window.dtype == np.float32
    np.testing.assert_array_equal(assessment.tap_window, samples[650:850])


def test_obviously_quiet_capture_is_rejected() -> None:
    assessment = assess_capture_quality(
        np.zeros((1_500, 2), dtype=np.float32),
        1_000,
        expected_channels=2,
        parameters=_parameters(),
    )

    assert assessment.accepted is False
    assert "signal_too_weak" in _reason_codes(assessment)
    assert "inactive_required_channel" in _reason_codes(assessment)
    assert assessment.tap_window is None
    json.dumps(assessment.to_metadata(), allow_nan=False)


def test_inactive_required_channel_rejects_spatial_capture() -> None:
    samples = _clear_tap()
    samples[:, 1] = 0.0

    assessment = assess_capture_quality(
        samples,
        1_000,
        expected_channels=2,
        parameters=_parameters(),
    )

    assert assessment.accepted is False
    assert "inactive_required_channel" in _reason_codes(assessment)
    assert assessment.metrics["active_channels"] == [1]
    assert assessment.metrics["inactive_required_channels"] == [2]
    assert assessment.metrics["all_expected_channels_active"] is False
    json.dumps(assessment.to_metadata(), allow_nan=False)


def test_nonzero_constant_capture_is_stable_and_rejected() -> None:
    assessment = assess_capture_quality(
        np.full((1_500, 2), 0.25, dtype=np.float32),
        1_000,
        expected_channels=2,
        parameters=_parameters(),
    )

    assert assessment.accepted is False
    assert "signal_too_weak" in _reason_codes(assessment)
    json.dumps(assessment.to_metadata(), allow_nan=False)


def test_near_clipping_capture_is_rejected() -> None:
    samples = _clear_tap(amplitude=0.99)

    assessment = assess_capture_quality(
        samples,
        1_000,
        expected_channels=2,
        parameters=_parameters(),
    )

    assert assessment.accepted is False
    assert "near_clipping" in _reason_codes(assessment)
    assert assessment.metrics["clipped_sample_count"] == 1


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_capture_is_rejected_without_nonfinite_metadata(
    value: float,
) -> None:
    samples = _clear_tap()
    samples[10, 0] = value

    assessment = assess_capture_quality(
        samples,
        1_000,
        expected_channels=2,
        parameters=_parameters(),
    )

    assert assessment.accepted is False
    assert "non_finite_capture" in _reason_codes(assessment)
    json.dumps(assessment.to_metadata(), allow_nan=False)


def test_boundary_transient_is_rejected() -> None:
    samples = _clear_tap(center=20)

    assessment = assess_capture_quality(
        samples,
        1_000,
        expected_channels=2,
        parameters=_parameters(),
    )

    assert assessment.accepted is False
    assert "transient_too_close_to_boundary" in _reason_codes(assessment)
    assert assessment.tap_window is None


def test_capture_shorter_than_tap_window_is_rejected() -> None:
    samples = _clear_tap(frames=100, center=50)

    assessment = assess_capture_quality(
        samples,
        1_000,
        expected_channels=2,
        parameters=_parameters(),
    )

    assert assessment.accepted is False
    assert "insufficient_tap_window" in _reason_codes(assessment)


def test_stationary_signal_without_distinct_transient_is_rejected() -> None:
    timeline = np.arange(1_500, dtype=np.float64) / 1_000
    tone = (0.01 * np.sin(2 * np.pi * 25 * timeline)).astype(np.float32)
    samples = np.column_stack((tone, tone))

    assessment = assess_capture_quality(
        samples,
        1_000,
        expected_channels=2,
        parameters=_parameters(),
    )

    assert assessment.accepted is False
    assert "no_distinct_transient" in _reason_codes(assessment)


def test_recording_starts_before_cue_and_waits_for_complete_layout(
    audio_backend_factory, monkeypatch
) -> None:
    recording = _clear_tap()
    backend = audio_backend_factory(recording=recording)
    events: list[object] = []
    original_rec = backend.rec
    original_wait = backend.wait

    def tracked_rec(*args, **kwargs):
        result = original_rec(*args, **kwargs)
        assert backend.recording_active is True
        events.append("recording_started")
        return result

    def pre_cue_sleep(duration: float) -> None:
        assert backend.recording_active is True
        events.append(("pre_cue_sleep", duration))

    def show_tap_cue() -> None:
        assert backend.recording_active is True
        events.append("TAP NOW")

    def tracked_wait() -> None:
        assert backend.recording_active is True
        events.append("capture_wait")
        original_wait()

    monkeypatch.setattr(backend, "rec", tracked_rec)
    monkeypatch.setattr(backend, "wait", tracked_wait)

    captured = record_collection_attempt(
        backend,
        device_index=1,
        sample_rate_hz=1_000,
        channels=2,
        duration_seconds=1.5,
        pre_cue_duration_seconds=0.2,
        tap_cue_fn=show_tap_cue,
        sleep_fn=pre_cue_sleep,
    )

    np.testing.assert_array_equal(captured, recording)
    assert captured.shape == (1_500, 2)
    assert captured.dtype == np.float32
    assert events == [
        "recording_started",
        ("pre_cue_sleep", 0.2),
        "TAP NOW",
        "capture_wait",
    ]
    assert backend.wait_calls == 1
    assert backend.recording_active is False
    assert backend.record_calls == [
        {
            "frames": 1_500,
            "samplerate": 1_000,
            "channels": 2,
            "dtype": "float32",
            "device": 1,
            "blocking": False,
        }
    ]


def test_countdown_is_silent_and_injected_sleep_avoids_waiting() -> None:
    output: list[str] = []
    sleeps: list[float] = []

    terminal_countdown(output.append, sleep_fn=sleeps.append)

    assert output == ["3", "2", "1"]
    assert sleeps == [1.0, 1.0, 1.0]
