from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Iterable, Sequence

import numpy as np
import pytest

from desksense.features import PRIMARY_FEATURE_NAME, extract_two_channel_features
from desksense.streaming import (
    DetectionResult,
    DetectorState,
    StreamingDetectorConfig,
    StreamingTapDetector,
)


def _small_config(**overrides: object) -> StreamingDetectorConfig:
    values: dict[str, object] = {
        "sample_rate_hz": 1_000.0,
        "startup_learning_seconds": 0.050,
        "minimum_crest_factor": 2.0,
    }
    values.update(overrides)
    return StreamingDetectorConfig(**values)


def _impulse_stream(
    *,
    frame_count: int = 800,
    centers: Sequence[int] = (200,),
    amplitudes: tuple[float, float] = (0.25, 0.05),
) -> np.ndarray:
    audio = np.zeros((frame_count, 2), dtype=np.float32)
    for center in centers:
        audio[center, 0] = amplitudes[0]
        audio[center, 1] = amplitudes[1]
    return audio


def _partition(audio: np.ndarray, sizes: Iterable[int]) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    start = 0
    sizes_list = list(sizes)
    if not sizes_list or any(size <= 0 for size in sizes_list):
        raise ValueError("Partition sizes must be positive.")
    index = 0
    while start < audio.shape[0]:
        size = sizes_list[index % len(sizes_list)]
        chunks.append(audio[start : start + size].copy())
        start += size
        index += 1
    return chunks


def _feed(
    detector: StreamingTapDetector,
    chunks: Iterable[np.ndarray],
) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    for chunk in chunks:
        results.extend(detector.process_chunk(chunk))
    return results


def _event_projection(result: DetectionResult) -> tuple[object, ...]:
    assert result.candidate_window is not None
    return (
        result.status,
        result.rejection_reasons,
        result.stream_epoch,
        result.onset_frame_index,
        result.center_frame_index,
        result.window_start_frame_index,
        result.window_end_frame_index_exclusive,
        result.candidate_window.tobytes(),
    )


def test_default_geometry_preserves_validated_window_domain() -> None:
    config = StreamingDetectorConfig()

    assert config.internal_block_frames == 240
    assert config.startup_learning_frames == 36_000
    assert config.center_search_pre_onset_frames == 576
    assert config.center_search_post_onset_frames == 1_200
    assert config.tap_window_frames == 9_600
    assert config.tap_window_pre_center_frames == 4_800
    assert config.transient_energy_window_frames == 240
    assert config.refractory_frames == 12_000
    assert config.history_capacity_frames == 11_040
    assert "not physically validated" in config.to_metadata()["threshold_status"]


def test_no_candidate_is_emitted_during_startup_learning() -> None:
    detector = StreamingTapDetector(_small_config())
    audio = _impulse_stream(frame_count=50, centers=(25,))

    results = detector.process_chunk(audio)

    assert results == ()
    assert detector.state is DetectorState.ARMED
    assert detector.processed_frame_count == 50


def test_deterministic_quiet_noise_does_not_trigger_after_learning() -> None:
    generator = np.random.default_rng(20260829)
    audio = generator.normal(0.0, 1.0e-5, (2_000, 2)).astype(np.float32)
    detector = StreamingTapDetector(_small_config())

    assert detector.process_chunk(audio) == ()
    assert detector.state is DetectorState.ARMED


def test_sustained_non_impulsive_level_is_handled_without_trigger() -> None:
    audio = np.zeros((1_000, 2), dtype=np.float32)
    audio[100:] = (0.02, -0.02)
    detector = StreamingTapDetector(_small_config())

    assert detector.process_chunk(audio) == ()


def test_isolated_impulse_emits_one_exact_centered_candidate() -> None:
    audio = _impulse_stream()
    detector = StreamingTapDetector(_small_config())

    results = detector.process_chunk(audio)

    assert len(results) == 1
    result = results[0]
    assert result.status == "detected"
    assert result.rejection_reasons == ()
    assert result.onset_frame_index == 200
    assert result.center_frame_index == 200
    assert result.window_start_frame_index == 100
    assert result.window_end_frame_index_exclusive == 300
    assert result.candidate_window is not None
    assert result.candidate_window.shape == (200, 2)
    assert np.array_equal(result.candidate_window, audio[100:300])
    assert result.candidate_window[100] == pytest.approx((0.25, 0.05))


def test_impulse_split_across_caller_boundary_triggers_once() -> None:
    audio = _impulse_stream()
    audio[199] = (0.08, 0.02)
    audio[201] = (-0.12, -0.03)
    detector = StreamingTapDetector(_small_config())

    results = _feed(detector, [audio[:200], audio[200:201], audio[201:]])

    assert len(results) == 1
    assert results[0].center_frame_index == 200


@pytest.mark.parametrize(
    "sizes",
    [
        (5,),
        (7, 113, 2, 59, 1, 31),
        (1,),
        (257, 3, 89),
    ],
)
def test_chunk_partitioning_does_not_change_logical_result(
    sizes: tuple[int, ...],
) -> None:
    audio = _impulse_stream()
    reference_detector = StreamingTapDetector(_small_config())
    reference = reference_detector.process_chunk(audio)
    detector = StreamingTapDetector(_small_config())

    partitioned = _feed(detector, _partition(audio, sizes))

    assert len(reference) == len(partitioned) == 1
    assert _event_projection(partitioned[0]) == _event_projection(reference[0])
    assert detector.noise_floor_rms == pytest.approx(
        reference_detector.noise_floor_rms, rel=0.0, abs=0.0
    )


def test_non_block_aligned_impulse_is_partition_invariant() -> None:
    audio = _impulse_stream(centers=(203,))
    partitions = (
        [audio.copy()],
        _partition(audio, (17, 2, 111, 4, 39, 1)),
        _partition(audio, (1,)),
    )
    projections: list[tuple[object, ...]] = []

    for chunks in partitions:
        results = _feed(StreamingTapDetector(_small_config()), chunks)
        assert len(results) == 1
        projections.append(_event_projection(results[0]))

    assert projections[1:] == [projections[0], projections[0]]
    assert projections[0][3:7] == (203, 203, 103, 303)


def test_onset_precedes_stronger_refined_transient_center() -> None:
    audio = _impulse_stream(frame_count=800, centers=())
    audio[203] = (0.15, 0.03)
    audio[210] = (0.30, 0.06)
    detector = StreamingTapDetector(_small_config())

    result = detector.process_chunk(audio)[0]

    assert result.onset_frame_index == 203
    assert result.center_frame_index == 210
    assert result.window_start_frame_index == 110
    assert result.window_end_frame_index_exclusive == 310
    assert result.candidate_window is not None
    assert np.array_equal(result.candidate_window, audio[110:310])
    assert result.candidate_window[100] == pytest.approx((0.30, 0.06))


def test_earliest_equal_transient_wins_deterministically() -> None:
    audio = _impulse_stream(centers=(200, 201))
    detector = StreamingTapDetector(_small_config())

    result = detector.process_chunk(audio)[0]

    assert result.center_frame_index == 200
    assert result.metrics["transient_ties_choose_earliest"] is True


def test_production_window_is_9600_by_2_and_center_is_index_4800() -> None:
    config = StreamingDetectorConfig(startup_learning_seconds=0.020)
    center = 6_000
    audio = _impulse_stream(frame_count=12_000, centers=(center,))
    detector = StreamingTapDetector(config)

    result = detector.process_chunk(audio)[0]

    assert result.center_frame_index == center
    assert result.candidate_window is not None
    assert result.candidate_window.shape == (9_600, 2)
    assert result.candidate_window[4_800] == pytest.approx((0.25, 0.05))
    assert np.array_equal(
        result.candidate_window,
        audio[center - 4_800 : center + 4_800],
    )


def test_candidate_emits_only_after_exact_post_center_context_arrives() -> None:
    audio = _impulse_stream()
    detector = StreamingTapDetector(_small_config())

    assert detector.process_chunk(audio[:299]) == ()
    results = detector.process_chunk(audio[299:300])

    assert len(results) == 1
    assert results[0].center_frame_index == 200
    assert results[0].window_end_frame_index_exclusive == 300
    assert results[0].finalized_after_frame_index_exclusive == 300


def test_input_is_immutable_and_result_window_owns_its_memory() -> None:
    audio = _impulse_stream()
    original = audio.copy()
    detector = StreamingTapDetector(_small_config())

    result = detector.process_chunk(audio)[0]
    assert result.candidate_window is not None
    emitted = result.candidate_window.copy()
    detector.process_chunk(np.zeros((2_000, 2), dtype=np.float32))

    assert np.array_equal(audio, original)
    assert np.array_equal(result.candidate_window, emitted)
    assert not np.shares_memory(result.candidate_window, audio)


def test_refractory_suppresses_nearby_event_and_allows_later_event() -> None:
    audio = _impulse_stream(
        frame_count=900,
        centers=(200, 350, 550),
    )
    detector = StreamingTapDetector(_small_config())

    results = detector.process_chunk(audio)

    assert [result.center_frame_index for result in results] == [200, 550]
    assert all(result.status == "detected" for result in results)
    assert results[0].metrics["refractory_anchor_frame_index"] == 200
    assert results[0].metrics["refractory_until_frame_index_exclusive"] == 450


def test_near_clipping_candidate_is_structurally_rejected() -> None:
    audio = _impulse_stream(amplitudes=(0.99, 0.20))
    detector = StreamingTapDetector(_small_config())

    result = detector.process_chunk(audio)[0]

    assert result.status == "rejected"
    assert result.rejection_reasons == ("near_clipping",)
    assert result.metrics["clipped_sample_count"] == 1
    assert result.candidate_window is not None


def test_missing_required_pre_history_is_rejected_without_padding() -> None:
    config = _small_config(startup_learning_seconds=0.010)
    audio = _impulse_stream(frame_count=150, centers=(20,))
    detector = StreamingTapDetector(config)

    result = detector.process_chunk(audio)[0]

    assert result.status == "rejected"
    assert result.rejection_reasons == ("required_history_unavailable",)
    assert result.center_frame_index == 20
    assert result.window_start_frame_index == -80
    assert result.candidate_window is None


def test_reset_matches_a_fresh_detector_and_replay() -> None:
    audio = _impulse_stream()
    detector = StreamingTapDetector(_small_config())
    first = detector.process_chunk(audio)[0]

    detector.reset()
    replay = detector.process_chunk(audio)[0]

    assert detector.stream_epoch == 0
    assert _event_projection(replay) == _event_projection(first)


def test_discontinuity_discards_partial_candidate_and_restarts_learning() -> None:
    audio = _impulse_stream()
    detector = StreamingTapDetector(_small_config())
    assert detector.process_chunk(audio[:230]) == ()
    assert detector.state is DetectorState.COLLECTING

    detector.notify_discontinuity()
    tail_results = detector.process_chunk(audio[230:])

    assert tail_results == ()
    assert detector.stream_epoch == 1
    assert detector.state is DetectorState.ARMED
    assert detector.processed_frame_count == 570


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(8, dtype=np.float32),
        np.zeros((8, 1), dtype=np.float32),
        np.full((8, 2), np.nan, dtype=np.float32),
        np.full((8, 2), np.inf, dtype=np.float32),
        np.full((8, 2), 1.0 + 2.0j, dtype=np.complex64),
        np.full((8, 2), "not-a-number", dtype=object),
    ],
)
def test_invalid_chunk_is_atomic_and_future_detection_still_works(
    invalid: np.ndarray,
) -> None:
    prefix = np.zeros((53, 2), dtype=np.float32)
    suffix = _impulse_stream(frame_count=747, centers=(147,))
    detector = StreamingTapDetector(_small_config())
    reference = StreamingTapDetector(_small_config())
    detector.process_chunk(prefix)
    reference.process_chunk(prefix)
    snapshot = (
        detector.state,
        detector.processed_frame_count,
        detector.partial_block_frame_count,
        detector.retained_history_frame_count,
        detector.noise_floor_rms,
    )

    with pytest.raises(ValueError):
        detector.process_chunk(invalid)

    assert (
        detector.state,
        detector.processed_frame_count,
        detector.partial_block_frame_count,
        detector.retained_history_frame_count,
        detector.noise_floor_rms,
    ) == snapshot
    actual_results = detector.process_chunk(suffix)
    expected_results = reference.process_chunk(suffix)
    assert len(actual_results) == len(expected_results) == 1
    assert _event_projection(actual_results[0]) == _event_projection(
        expected_results[0]
    )


def test_empty_correctly_shaped_chunk_is_a_no_op() -> None:
    detector = StreamingTapDetector(_small_config())

    assert detector.process_chunk(np.empty((0, 2), dtype=np.float32)) == ()
    assert detector.processed_frame_count == 0
    assert detector.state is DetectorState.LEARNING


def test_history_memory_remains_structurally_bounded() -> None:
    detector = StreamingTapDetector(_small_config())
    quiet = np.zeros((detector.maximum_buffered_frames * 100 + 3, 2), dtype=np.float32)

    detector.process_chunk(quiet)

    assert (
        detector.retained_history_frame_count
        <= detector.config.history_capacity_frames
    )
    assert detector.buffered_frame_count <= detector.maximum_buffered_frames
    assert detector.history_storage_bytes == (
        detector.config.history_capacity_frames * 2 * np.dtype(np.float32).itemsize
    )
    assert detector.maximum_buffered_audio_bytes == (
        detector.maximum_buffered_frames * 2 * np.dtype(np.float32).itemsize
    )


def test_detector_candidate_is_compatible_with_existing_feature_extractor() -> None:
    audio = _impulse_stream()
    result = StreamingTapDetector(_small_config()).process_chunk(audio)[0]
    assert result.candidate_window is not None

    features = extract_two_channel_features(result.candidate_window)

    assert features[PRIMARY_FEATURE_NAME] == pytest.approx(
        20.0 * math.log10(0.05 / 0.25)
    )


def test_result_metrics_are_finite_and_window_is_raw_float32() -> None:
    result = StreamingTapDetector(_small_config()).process_chunk(
        _impulse_stream()
    )[0]
    assert result.candidate_window is not None

    assert result.candidate_window.dtype == np.float32
    assert all(
        not isinstance(value, float) or math.isfinite(value)
        for value in result.metrics.values()
    )
    assert result.metrics["candidate_selection_is_phase2a_identical"] is False


def test_odd_frame_tap_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="even number of frames"):
        _small_config(tap_window_seconds=0.201)


def test_pure_streaming_and_inference_imports_do_not_load_sounddevice() -> None:
    command = (
        "import sys; import desksense.streaming, desksense.inference; "
        "assert 'sounddevice' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
