from __future__ import annotations

import json

import numpy as np
import pytest

from desksense.characterization import (
    analyze_channel_activity,
    analyze_multichannel_audio,
    analyze_signal_pair,
    bounded_normalized_cross_correlation,
    find_strongest_transient_window,
    normalized_difference_energy,
)


def test_identical_channels_have_matching_pair_metrics() -> None:
    signal = np.random.default_rng(10).normal(0.0, 0.01, 8_192)

    metrics = analyze_signal_pair(
        signal,
        signal.copy(),
        sample_rate_hz=48_000,
        max_lag_samples=48,
    )

    assert metrics["status"] == "ok"
    assert metrics["pearson_correlation"] == pytest.approx(1.0)
    assert metrics["normalized_difference_energy"] == pytest.approx(0.0)
    assert metrics["ac_rms_ratio_b_to_a"] == pytest.approx(1.0)
    assert metrics["relative_ac_level_db_b_minus_a"] == pytest.approx(0.0)
    assert metrics["max_normalized_cross_correlation"] == pytest.approx(1.0)
    assert metrics["lag_samples"] == 0
    assert metrics["lag_microseconds"] == pytest.approx(0.0)
    assert metrics["lag_aligned_normalized_difference_energy"] == pytest.approx(
        0.0
    )


def test_seeded_independent_channels_are_dissimilar() -> None:
    generator = np.random.default_rng(2026)
    channel_a = generator.normal(0.0, 0.01, 20_000)
    channel_b = generator.normal(0.0, 0.01, 20_000)

    metrics = analyze_signal_pair(
        channel_a,
        channel_b,
        sample_rate_hz=48_000,
        max_lag_samples=48,
    )

    assert metrics["status"] == "ok"
    assert abs(metrics["pearson_correlation"]) < 0.05
    assert metrics["normalized_difference_energy"] == pytest.approx(
        1.0, abs=0.05
    )
    assert metrics["max_normalized_cross_correlation_absolute"] < 0.05


def test_delayed_copy_reports_positive_lag_and_reverse_reports_negative_lag() -> None:
    source = np.random.default_rng(11).normal(0.0, 0.01, 8_192)
    delay_samples = 13
    delayed = np.zeros_like(source)
    delayed[delay_samples:] = source[:-delay_samples]

    forward = bounded_normalized_cross_correlation(
        source,
        delayed,
        sample_rate_hz=48_000,
        max_lag_samples=24,
    )
    reverse = bounded_normalized_cross_correlation(
        delayed,
        source,
        sample_rate_hz=48_000,
        max_lag_samples=24,
    )

    assert forward["status"] == "ok"
    assert forward["absolute_coefficient"] == pytest.approx(1.0)
    assert forward["lag_samples"] == delay_samples
    assert forward["lag_microseconds"] == pytest.approx(
        delay_samples * 1_000_000 / 48_000
    )
    assert reverse["status"] == "ok"
    assert reverse["absolute_coefficient"] == pytest.approx(1.0)
    assert reverse["lag_samples"] == -delay_samples


def test_silent_channels_are_inactive_and_pair_analysis_is_skipped() -> None:
    samples = np.zeros((1_000, 2), dtype=np.float32)

    per_channel, assessment = analyze_channel_activity(samples)
    pair = analyze_signal_pair(
        samples[:, 0],
        samples[:, 1],
        sample_rate_hz=48_000,
        max_lag_samples=48,
    )
    cross_correlation = bounded_normalized_cross_correlation(
        samples[:, 0],
        samples[:, 1],
        sample_rate_hz=48_000,
        max_lag_samples=48,
    )

    assert assessment["all_channels_below_absolute_floors"] is True
    assert all(metric["effectively_inactive"] for metric in per_channel)
    assert pair["status"] == "not_analyzed"
    assert cross_correlation["status"] == "not_analyzed"


def test_nonzero_constant_arrays_are_numerically_stable() -> None:
    channel_a = np.full(1_024, 0.25)
    channel_b = np.full(1_024, -0.125)

    per_channel, _ = analyze_channel_activity(
        np.column_stack((channel_a, channel_b))
    )
    pair = analyze_signal_pair(
        channel_a,
        channel_b,
        sample_rate_hz=44_100,
        max_lag_samples=44,
    )

    assert per_channel[0]["dc_offset"] == pytest.approx(0.25)
    assert per_channel[1]["dc_offset"] == pytest.approx(-0.125)
    assert per_channel[0]["standard_deviation"] == pytest.approx(0.0)
    assert per_channel[1]["standard_deviation"] == pytest.approx(0.0)
    assert all(metric["effectively_inactive"] for metric in per_channel)
    assert pair["status"] == "not_analyzed"
    assert normalized_difference_energy(channel_a, channel_b) is None
    json.dumps(
        {"per_channel": per_channel, "pair": pair},
        allow_nan=False,
    )


def test_inactivity_uses_absolute_and_relative_signal_levels() -> None:
    generator = np.random.default_rng(12)
    strong = generator.normal(0.0, 0.01, 10_000)
    weak = generator.normal(0.0, 1e-7, 10_000)

    per_channel, assessment = analyze_channel_activity(
        np.column_stack((strong, weak))
    )

    assert assessment["all_channels_below_absolute_floors"] is False
    assert per_channel[0]["effectively_inactive"] is False
    assert per_channel[1]["effectively_inactive"] is True
    assert (
        per_channel[1]["activity_assessment_reason"]
        == "absolutely_quiet_and_at_least_30_db_below_strongest"
    )
    assert per_channel[1]["relative_standard_deviation_db_to_strongest"] < -30


def test_low_rms_channel_with_a_clear_transient_is_not_marked_inactive() -> None:
    generator = np.random.default_rng(13)
    strong = generator.normal(0.0, 0.01, 10_000)
    sparse_transient = np.zeros(10_000)
    sparse_transient[5_000] = 0.001

    per_channel, _ = analyze_channel_activity(
        np.column_stack((strong, sparse_transient))
    )

    assert per_channel[1]["standard_deviation"] < 3.0518e-5
    assert per_channel[1]["centered_peak_absolute"] > 0.0009
    assert per_channel[1]["effectively_inactive"] is False


@pytest.mark.parametrize(
    ("channel_b", "expected"),
    [
        (np.array([-1.0, 1.0, -1.0, 1.0]), 0.0),
        (np.array([1.0, -1.0, 1.0, -1.0]), 2.0),
        (np.array([-2.0, 2.0, -2.0, 2.0]), 0.2),
        (np.array([-1.0, -1.0, 1.0, 1.0]), 1.0),
    ],
)
def test_normalized_difference_energy_known_values(
    channel_b: np.ndarray, expected: float
) -> None:
    channel_a = np.array([-1.0, 1.0, -1.0, 1.0])

    result = normalized_difference_energy(channel_a, channel_b)

    assert result == pytest.approx(expected)


def test_transient_window_is_centered_on_strongest_impulse() -> None:
    samples = np.zeros((500, 2))
    samples[250, 0] = 1.0
    samples[250, 1] = 0.5

    transient = find_strongest_transient_window(
        samples,
        1_000,
        active_channel_indexes=[0, 1],
        energy_window_seconds=0.005,
        analysis_window_seconds=0.100,
    )

    assert transient["status"] == "ok"
    assert transient["center_sample"] == 250
    assert transient["start_sample"] == 200
    assert transient["end_sample_exclusive"] == 300
    assert transient["duration_samples"] == 100
    assert transient["active_channels_used"] == [1, 2]


@pytest.mark.parametrize(
    ("impulse_sample", "expected_start", "expected_end"),
    [(2, 0, 100), (497, 400, 500)],
)
def test_transient_window_is_clamped_at_recording_boundaries(
    impulse_sample: int, expected_start: int, expected_end: int
) -> None:
    samples = np.zeros((500, 1))
    samples[impulse_sample, 0] = 1.0

    transient = find_strongest_transient_window(
        samples,
        1_000,
        active_channel_indexes=[0],
        energy_window_seconds=0.005,
        analysis_window_seconds=0.100,
    )

    assert transient["center_sample"] == impulse_sample
    assert transient["start_sample"] == expected_start
    assert transient["end_sample_exclusive"] == expected_end


def test_multichannel_analysis_includes_active_pairs_and_skips_inactive_pairs() -> None:
    generator = np.random.default_rng(14)
    first = generator.normal(0.0, 0.01, 4_000)
    duplicate = first.copy()
    independent = generator.normal(0.0, 0.01, 4_000)
    inactive = np.zeros(4_000)
    samples = np.column_stack((first, duplicate, independent, inactive))

    analysis = analyze_multichannel_audio(samples, sample_rate_hz=4_000)

    assert analysis["analysis_kind"] == (
        "exploratory_microphone_channel_characterization"
    )
    assert analysis["frame_count"] == 4_000
    assert analysis["channel_count"] == 4
    assert [
        channel["effectively_inactive"] for channel in analysis["per_channel"]
    ] == [False, False, False, True]

    whole_pairs = analysis["whole_recording"]["pairwise"]
    assert len(whole_pairs) == 6
    assert sum(pair["status"] == "ok" for pair in whole_pairs) == 3
    assert sum(pair["status"] == "not_analyzed" for pair in whole_pairs) == 3
    duplicate_pair = next(
        pair
        for pair in whole_pairs
        if (pair["channel_a"], pair["channel_b"]) == (1, 2)
    )
    assert duplicate_pair["heuristic_label"] == "duplicate_like"
    assert analysis["transient_window"]["duration_samples"] == 400


def test_characterization_result_is_strictly_json_serializable_without_audio() -> None:
    generator = np.random.default_rng(15)
    samples = generator.normal(0.0, 0.01, (2_000, 3))

    analysis = analyze_multichannel_audio(samples, sample_rate_hz=8_000)

    encoded = json.dumps(analysis, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["channel_count"] == 3
    assert "raw_audio" not in encoded
    assert "samples" not in decoded
