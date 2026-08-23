"""Pure NumPy microphone-channel characterization helpers.

The routines in this module analyze finite in-memory samples. They do not open
audio devices, retain recordings, or implement DeskSense's future tap detector.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np

DEFAULT_CHARACTERIZATION_SECONDS = 5.0
MAX_CHARACTERIZATION_CHANNELS = 8
MAX_LAG_SECONDS = 0.001
TRANSIENT_ENERGY_WINDOW_SECONDS = 0.005
TRANSIENT_ANALYSIS_WINDOW_SECONDS = 0.100

SIXTEEN_BIT_LSB = 2.0**-15
ACTIVITY_STANDARD_DEVIATION_FLOOR = SIXTEEN_BIT_LSB
ACTIVITY_CENTERED_PEAK_FLOOR = 4.0 * SIXTEEN_BIT_LSB
ACTIVITY_RELATIVE_FLOOR_DB = -30.0
ACTIVITY_RELATIVE_FLOOR_RATIO = 10.0 ** (ACTIVITY_RELATIVE_FLOOR_DB / 20.0)

_NUMERICAL_EPSILON_FACTOR = 64.0


def analyze_multichannel_audio(
    samples: Any,
    sample_rate_hz: float,
    *,
    max_lag_seconds: float = MAX_LAG_SECONDS,
    transient_energy_window_seconds: float = TRANSIENT_ENERGY_WINDOW_SECONDS,
    transient_analysis_window_seconds: float = TRANSIENT_ANALYSIS_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Characterize channel activity, independence, and a transient window."""

    audio = _as_audio_matrix(samples)
    sample_rate = _positive_finite_float(sample_rate_hz, "sample rate")
    max_lag = max(1, round(sample_rate * max_lag_seconds))

    per_channel, activity = analyze_channel_activity(audio)
    whole_pairs = analyze_channel_pairs(
        audio,
        per_channel,
        sample_rate_hz=sample_rate,
        max_lag_samples=max_lag,
    )

    active_indexes = [
        metric["channel"] - 1
        for metric in per_channel
        if not metric["effectively_inactive"]
    ]
    transient_metadata = find_strongest_transient_window(
        audio,
        sample_rate,
        active_channel_indexes=active_indexes,
        energy_window_seconds=transient_energy_window_seconds,
        analysis_window_seconds=transient_analysis_window_seconds,
    )

    if transient_metadata["status"] == "ok":
        start = transient_metadata["start_sample"]
        end = transient_metadata["end_sample_exclusive"]
        transient_audio = audio[start:end]
        transient_channels, transient_activity = analyze_channel_activity(
            transient_audio
        )
        transient_pairs = analyze_channel_pairs(
            transient_audio,
            transient_channels,
            sample_rate_hz=sample_rate,
            max_lag_samples=max_lag,
        )
    else:
        transient_channels = []
        transient_activity = activity_thresholds()
        transient_pairs = []

    transient_by_pair = {
        (pair["channel_a"], pair["channel_b"]): pair
        for pair in transient_pairs
    }
    for pair in whole_pairs:
        matching_transient = transient_by_pair.get(
            (pair["channel_a"], pair["channel_b"])
        )
        pair["heuristic_label"] = _combined_heuristic_label(
            pair, matching_transient
        )

    return {
        "analysis_kind": "exploratory_microphone_channel_characterization",
        "exploratory": True,
        "sample_rate_hz": sample_rate,
        "frame_count": int(audio.shape[0]),
        "channel_count": int(audio.shape[1]),
        "duration_seconds": float(audio.shape[0] / sample_rate),
        "analysis_parameters": {
            "activity_assessment": activity,
            "maximum_lag_samples": max_lag,
            "maximum_lag_microseconds": float(max_lag * 1_000_000.0 / sample_rate),
            "lag_sign_convention": (
                "Positive lag means channel_b arrives later than channel_a."
            ),
            "cross_correlation_selection": (
                "Largest absolute normalized coefficient; signed value retained; "
                "ties prefer the smallest absolute lag."
            ),
            "transient_energy_window_seconds": float(
                transient_energy_window_seconds
            ),
            "transient_analysis_window_seconds": float(
                transient_analysis_window_seconds
            ),
        },
        "per_channel": per_channel,
        "whole_recording": {
            "pairwise": whole_pairs,
        },
        "transient_window": {
            **transient_metadata,
            "exploratory": True,
            "method_note": (
                "Strongest local mean-square-energy window; this is not the "
                "final DeskSense tap detector."
            ),
            "activity_assessment": transient_activity,
            "per_channel": transient_channels,
            "pairwise": transient_pairs,
        },
        "heuristic_interpretation": heuristic_interpretation_guide(),
    }


def analyze_channel_activity(
    samples: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Calculate per-channel levels and a conservative inactivity assessment."""

    audio = _as_audio_matrix(samples)
    means = np.mean(audio, axis=0, dtype=np.float64)
    centered = audio - means
    peaks = np.max(np.abs(audio), axis=0)
    root_mean_squares = np.sqrt(np.mean(np.square(audio), axis=0))
    standard_deviations = np.sqrt(np.mean(np.square(centered), axis=0))
    centered_peaks = np.max(np.abs(centered), axis=0)

    strongest_std = float(np.max(standard_deviations))
    strongest_centered_peak = float(np.max(centered_peaks))
    all_below_absolute_floors = (
        strongest_std <= ACTIVITY_STANDARD_DEVIATION_FLOOR
        and strongest_centered_peak <= ACTIVITY_CENTERED_PEAK_FLOOR
    )

    metrics: list[dict[str, Any]] = []
    for index in range(audio.shape[1]):
        standard_deviation = float(standard_deviations[index])
        centered_peak = float(centered_peaks[index])
        relative_std = _safe_ratio(standard_deviation, strongest_std)
        relative_peak = _safe_ratio(centered_peak, strongest_centered_peak)

        below_absolute_floors = (
            standard_deviation <= ACTIVITY_STANDARD_DEVIATION_FLOOR
            and centered_peak <= ACTIVITY_CENTERED_PEAK_FLOOR
        )
        below_relative_floors = (
            relative_std is not None
            and relative_peak is not None
            and relative_std <= ACTIVITY_RELATIVE_FLOOR_RATIO
            and relative_peak <= ACTIVITY_RELATIVE_FLOOR_RATIO
        )

        if all_below_absolute_floors:
            effectively_inactive = True
            reason = "all_channels_below_absolute_activity_floors"
        elif below_absolute_floors and below_relative_floors:
            effectively_inactive = True
            reason = "absolutely_quiet_and_at_least_30_db_below_strongest"
        else:
            effectively_inactive = False
            reason = "sufficient_absolute_or_relative_ac_signal"

        metrics.append(
            {
                "channel": index + 1,
                "peak_absolute": float(peaks[index]),
                "rms": float(root_mean_squares[index]),
                "dc_offset": float(means[index]),
                "standard_deviation": standard_deviation,
                "centered_peak_absolute": centered_peak,
                "relative_standard_deviation_to_strongest": relative_std,
                "relative_standard_deviation_db_to_strongest": _ratio_to_db(
                    relative_std
                ),
                "relative_centered_peak_to_strongest": relative_peak,
                "relative_centered_peak_db_to_strongest": _ratio_to_db(
                    relative_peak
                ),
                "effectively_inactive": effectively_inactive,
                "activity_assessment_reason": reason,
            }
        )

    assessment = activity_thresholds()
    assessment.update(
        {
            "strongest_channel_standard_deviation": strongest_std,
            "strongest_channel_centered_peak_absolute": strongest_centered_peak,
            "all_channels_below_absolute_floors": all_below_absolute_floors,
        }
    )
    return metrics, assessment


def analyze_channel_pairs(
    samples: Any,
    per_channel: list[dict[str, Any]],
    *,
    sample_rate_hz: float,
    max_lag_samples: int,
) -> list[dict[str, Any]]:
    """Analyze every active pair and explicitly skip inactive pairs."""

    audio = _as_audio_matrix(samples)
    if len(per_channel) != audio.shape[1]:
        raise ValueError("Per-channel metadata does not match captured audio.")

    pairs: list[dict[str, Any]] = []
    for channel_a, channel_b in combinations(range(audio.shape[1]), 2):
        pair_identity = {
            "channel_a": channel_a + 1,
            "channel_b": channel_b + 1,
        }
        inactive = [
            metric["channel"]
            for metric in (per_channel[channel_a], per_channel[channel_b])
            if metric["effectively_inactive"]
        ]
        if inactive:
            pairs.append(
                {
                    **pair_identity,
                    "status": "not_analyzed",
                    "reason": (
                        "Effectively inactive channel(s): "
                        + ", ".join(str(channel) for channel in inactive)
                    ),
                }
            )
            continue

        pair_metrics = analyze_signal_pair(
            audio[:, channel_a],
            audio[:, channel_b],
            sample_rate_hz=sample_rate_hz,
            max_lag_samples=max_lag_samples,
        )
        pairs.append({**pair_identity, **pair_metrics})
    return pairs


def analyze_signal_pair(
    channel_a: Any,
    channel_b: Any,
    *,
    sample_rate_hz: float,
    max_lag_samples: int,
) -> dict[str, Any]:
    """Calculate robust whole-signal metrics for a channel pair."""

    signal_a, signal_b = _as_signal_pair(channel_a, channel_b)
    sample_rate = _positive_finite_float(sample_rate_hz, "sample rate")
    centered_a = signal_a - float(np.mean(signal_a))
    centered_b = signal_b - float(np.mean(signal_b))
    energy_a = float(np.dot(centered_a, centered_a))
    energy_b = float(np.dot(centered_b, centered_b))

    if _energy_is_numerically_zero(signal_a, energy_a) or _energy_is_numerically_zero(
        signal_b, energy_b
    ):
        return {
            "status": "not_analyzed",
            "reason": "One or both channels are silent or near-constant.",
        }

    pearson = float(
        np.clip(
            np.dot(centered_a, centered_b) / math.sqrt(energy_a * energy_b),
            -1.0,
            1.0,
        )
    )
    difference_energy = _normalized_difference_energy_centered(
        centered_a, centered_b
    )
    std_a = math.sqrt(energy_a / signal_a.size)
    std_b = math.sqrt(energy_b / signal_b.size)
    rms_ratio = _safe_ratio(std_b, std_a)
    cross_correlation = bounded_normalized_cross_correlation(
        centered_a,
        centered_b,
        sample_rate_hz=sample_rate,
        max_lag_samples=max_lag_samples,
    )
    if cross_correlation["status"] != "ok":
        return {
            "status": "not_analyzed",
            "reason": cross_correlation["reason"],
        }

    return {
        "status": "ok",
        "pearson_correlation": pearson,
        "normalized_difference_energy": difference_energy,
        "ac_rms_ratio_b_to_a": rms_ratio,
        "relative_ac_level_db_b_minus_a": _ratio_to_db(rms_ratio),
        "max_normalized_cross_correlation": cross_correlation["coefficient"],
        "max_normalized_cross_correlation_absolute": cross_correlation[
            "absolute_coefficient"
        ],
        "lag_samples": cross_correlation["lag_samples"],
        "lag_microseconds": cross_correlation["lag_microseconds"],
        "lag_aligned_normalized_difference_energy": cross_correlation[
            "lag_aligned_normalized_difference_energy"
        ],
    }


def normalized_difference_energy(channel_a: Any, channel_b: Any) -> float | None:
    """Return DC-removed difference energy normalized by both signal energies."""

    signal_a, signal_b = _as_signal_pair(channel_a, channel_b)
    centered_a = signal_a - float(np.mean(signal_a))
    centered_b = signal_b - float(np.mean(signal_b))
    return _normalized_difference_energy_centered(centered_a, centered_b)


def bounded_normalized_cross_correlation(
    channel_a: Any,
    channel_b: Any,
    *,
    sample_rate_hz: float,
    max_lag_samples: int,
) -> dict[str, Any]:
    """Find maximum-absolute normalized correlation within a bounded lag."""

    signal_a, signal_b = _as_signal_pair(channel_a, channel_b)
    sample_rate = _positive_finite_float(sample_rate_hz, "sample rate")
    try:
        requested_lag = int(max_lag_samples)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Maximum lag must be a non-negative integer.") from error
    if requested_lag < 0:
        raise ValueError("Maximum lag must be a non-negative integer.")
    if signal_a.size < 2:
        return {
            "status": "not_analyzed",
            "reason": "At least two samples are required for correlation.",
        }

    centered_a = signal_a - float(np.mean(signal_a))
    centered_b = signal_b - float(np.mean(signal_b))
    maximum_lag = min(requested_lag, signal_a.size - 2)

    prefix_a = _prefix_sum(centered_a)
    prefix_b = _prefix_sum(centered_b)
    prefix_a_squared = _prefix_sum(np.square(centered_a))
    prefix_b_squared = _prefix_sum(np.square(centered_b))

    best_coefficient: float | None = None
    best_lag = 0
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag >= 0:
            start_a, start_b, length = 0, lag, signal_a.size - lag
        else:
            start_a, start_b, length = -lag, 0, signal_a.size + lag
        if length < 2:
            continue

        end_a = start_a + length
        end_b = start_b + length
        sum_a = _interval_sum(prefix_a, start_a, end_a)
        sum_b = _interval_sum(prefix_b, start_b, end_b)
        square_sum_a = _interval_sum(prefix_a_squared, start_a, end_a)
        square_sum_b = _interval_sum(prefix_b_squared, start_b, end_b)
        energy_a = square_sum_a - (sum_a * sum_a / length)
        energy_b = square_sum_b - (sum_b * sum_b / length)
        if _overlap_energy_is_zero(energy_a, square_sum_a, sum_a, length):
            continue
        if _overlap_energy_is_zero(energy_b, square_sum_b, sum_b, length):
            continue

        product_sum = float(
            np.dot(centered_a[start_a:end_a], centered_b[start_b:end_b])
        )
        covariance = product_sum - (sum_a * sum_b / length)
        coefficient = float(
            np.clip(covariance / math.sqrt(energy_a * energy_b), -1.0, 1.0)
        )

        if best_coefficient is None or _correlation_candidate_is_better(
            coefficient, lag, best_coefficient, best_lag
        ):
            best_coefficient = coefficient
            best_lag = lag

    if best_coefficient is None:
        return {
            "status": "not_analyzed",
            "reason": "Normalized correlation is undefined for near-constant overlap.",
        }

    aligned_a, aligned_b = _aligned_views(centered_a, centered_b, best_lag)
    aligned_difference = _normalized_difference_energy_centered(
        aligned_a - float(np.mean(aligned_a)),
        aligned_b - float(np.mean(aligned_b)),
    )
    return {
        "status": "ok",
        "coefficient": best_coefficient,
        "absolute_coefficient": abs(best_coefficient),
        "lag_samples": best_lag,
        "lag_microseconds": float(best_lag * 1_000_000.0 / sample_rate),
        "lag_aligned_normalized_difference_energy": aligned_difference,
    }


def find_strongest_transient_window(
    samples: Any,
    sample_rate_hz: float,
    *,
    active_channel_indexes: list[int],
    energy_window_seconds: float = TRANSIENT_ENERGY_WINDOW_SECONDS,
    analysis_window_seconds: float = TRANSIENT_ANALYSIS_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Locate a strongest local-energy region without claiming tap detection."""

    audio = _as_audio_matrix(samples)
    sample_rate = _positive_finite_float(sample_rate_hz, "sample rate")
    energy_duration = _positive_finite_float(
        energy_window_seconds, "transient energy window"
    )
    analysis_duration = _positive_finite_float(
        analysis_window_seconds, "transient analysis window"
    )
    if not active_channel_indexes:
        return {
            "status": "not_available",
            "reason": "No active channels were available for transient exploration.",
        }
    if any(
        index < 0 or index >= audio.shape[1] for index in active_channel_indexes
    ):
        raise ValueError("Active channel index is outside the captured channel range.")

    selected = audio[:, active_channel_indexes]
    centered = selected - np.mean(selected, axis=0, dtype=np.float64)
    pooled_power = np.mean(np.square(centered), axis=1)
    energy_samples = min(
        audio.shape[0], max(1, round(sample_rate * energy_duration))
    )
    power_prefix = _prefix_sum(pooled_power)
    moving_energy = power_prefix[energy_samples:] - power_prefix[:-energy_samples]
    energy_start = int(np.argmax(moving_energy))
    energy_end = energy_start + energy_samples
    local_peak_offset = int(np.argmax(pooled_power[energy_start:energy_end]))
    center_sample = energy_start + local_peak_offset

    requested_analysis_samples = max(1, round(sample_rate * analysis_duration))
    analysis_samples = min(audio.shape[0], requested_analysis_samples)
    start_sample = max(
        0,
        min(
            center_sample - analysis_samples // 2,
            audio.shape[0] - analysis_samples,
        ),
    )
    end_sample = start_sample + analysis_samples

    return {
        "status": "ok",
        "active_channels_used": [index + 1 for index in active_channel_indexes],
        "energy_window_samples": energy_samples,
        "energy_window_start_sample": energy_start,
        "energy_window_end_sample_exclusive": energy_end,
        "energy_window_mean_square": float(
            moving_energy[energy_start] / energy_samples
        ),
        "center_sample": center_sample,
        "center_time_seconds": float(center_sample / sample_rate),
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "duration_samples": analysis_samples,
        "duration_seconds": float(analysis_samples / sample_rate),
    }


def activity_thresholds() -> dict[str, Any]:
    """Return the reported, reviewable channel-activity heuristic."""

    return {
        "is_heuristic": True,
        "method": (
            "A channel is inactive when all channels are below the absolute "
            "floors, or when it is below both absolute floors and at least "
            "30 dB below the strongest channel in standard deviation and "
            "centered peak."
        ),
        "absolute_standard_deviation_floor": ACTIVITY_STANDARD_DEVIATION_FLOOR,
        "absolute_standard_deviation_floor_dbfs": _ratio_to_db(
            ACTIVITY_STANDARD_DEVIATION_FLOOR
        ),
        "absolute_centered_peak_floor": ACTIVITY_CENTERED_PEAK_FLOOR,
        "absolute_centered_peak_floor_dbfs": _ratio_to_db(
            ACTIVITY_CENTERED_PEAK_FLOOR
        ),
        "relative_floor_ratio": ACTIVITY_RELATIVE_FLOOR_RATIO,
        "relative_floor_db": ACTIVITY_RELATIVE_FLOOR_DB,
        "absolute_floor_basis": (
            "One normalized 16-bit LSB for standard deviation and four LSBs "
            "for centered peak; combined with the relative test rather than "
            "treated as a universal microphone threshold."
        ),
    }


def heuristic_interpretation_guide() -> dict[str, Any]:
    """Describe conservative labels applied only after two analysis scopes agree."""

    return {
        "is_heuristic": True,
        "basis": "Whole-recording and transient-window metrics must corroborate.",
        "labels": {
            "duplicate_like": (
                "Both scopes show positive correlation >= 0.999, normalized "
                "difference energy <= 0.002, level difference <= 0.25 dB, "
                "and lag <= 1 sample."
            ),
            "highly_similar": (
                "Both scopes show positive maximum correlation >= 0.95, "
                "lag-aligned difference energy <= 0.10, and level difference "
                "<= 3 dB."
            ),
            "inverted_similarity": (
                "Both scopes show strong negative-polarity correlation; this "
                "is not labeled as a duplicate."
            ),
            "meaningfully_different": (
                "Both scopes show maximum absolute correlation < 0.8 and "
                "normalized difference energy > 0.25."
            ),
            "mixed_or_inconclusive": "The metrics or scopes do not agree.",
            "insufficient_signal": "One or both scopes could not be analyzed.",
        },
        "caution": (
            "The transient window is a subset of the whole recording, so their "
            "agreement is not independent confirmation. A single recording and "
            "shared Windows audio processing cannot prove physical microphone "
            "independence or duplication."
        ),
    }


def _combined_heuristic_label(
    whole: dict[str, Any], transient: dict[str, Any] | None
) -> str:
    if whole.get("status") != "ok" or not transient or transient.get("status") != "ok":
        return "insufficient_signal"
    if _duplicate_like(whole) and _duplicate_like(transient):
        return "duplicate_like"
    if _inverted_similarity(whole) and _inverted_similarity(transient):
        return "inverted_similarity"
    if _highly_similar(whole) and _highly_similar(transient):
        return "highly_similar"
    if _meaningfully_different(whole) and _meaningfully_different(transient):
        return "meaningfully_different"
    return "mixed_or_inconclusive"


def _duplicate_like(metrics: dict[str, Any]) -> bool:
    return (
        metrics["pearson_correlation"] >= 0.999
        and metrics["max_normalized_cross_correlation"] >= 0.999
        and metrics["normalized_difference_energy"] <= 0.002
        and abs(metrics["relative_ac_level_db_b_minus_a"]) <= 0.25
        and abs(metrics["lag_samples"]) <= 1
    )


def _highly_similar(metrics: dict[str, Any]) -> bool:
    aligned_difference = metrics["lag_aligned_normalized_difference_energy"]
    return (
        metrics["max_normalized_cross_correlation"] >= 0.95
        and aligned_difference is not None
        and aligned_difference <= 0.10
        and abs(metrics["relative_ac_level_db_b_minus_a"]) <= 3.0
    )


def _inverted_similarity(metrics: dict[str, Any]) -> bool:
    return (
        metrics["max_normalized_cross_correlation"] <= -0.95
        and abs(metrics["relative_ac_level_db_b_minus_a"]) <= 3.0
    )


def _meaningfully_different(metrics: dict[str, Any]) -> bool:
    return (
        metrics["max_normalized_cross_correlation_absolute"] < 0.8
        and metrics["normalized_difference_energy"] > 0.25
    )


def _normalized_difference_energy_centered(
    centered_a: np.ndarray[Any, Any], centered_b: np.ndarray[Any, Any]
) -> float | None:
    energy_a = float(np.dot(centered_a, centered_a))
    energy_b = float(np.dot(centered_b, centered_b))
    denominator = energy_a + energy_b
    if denominator <= _energy_tolerance(energy_a, energy_b):
        return None
    difference = centered_a - centered_b
    value = float(np.dot(difference, difference) / denominator)
    return float(np.clip(value, 0.0, 2.0))


def _energy_is_numerically_zero(signal: np.ndarray[Any, Any], energy: float) -> bool:
    raw_energy = float(np.dot(signal, signal))
    mean_energy = float(signal.size * np.mean(signal) ** 2)
    tolerance = _NUMERICAL_EPSILON_FACTOR * np.finfo(np.float64).eps * max(
        raw_energy, mean_energy, np.finfo(np.float64).tiny
    )
    return energy <= max(tolerance, np.finfo(np.float64).tiny)


def _overlap_energy_is_zero(
    energy: float, square_sum: float, sample_sum: float, length: int
) -> bool:
    mean_energy = sample_sum * sample_sum / length
    tolerance = _NUMERICAL_EPSILON_FACTOR * np.finfo(np.float64).eps * max(
        abs(square_sum), abs(mean_energy), np.finfo(np.float64).tiny
    )
    return energy <= max(tolerance, np.finfo(np.float64).tiny)


def _energy_tolerance(*energies: float) -> float:
    return _NUMERICAL_EPSILON_FACTOR * np.finfo(np.float64).eps * max(
        *(abs(energy) for energy in energies), np.finfo(np.float64).tiny
    )


def _correlation_candidate_is_better(
    coefficient: float,
    lag: int,
    best_coefficient: float,
    best_lag: int,
) -> bool:
    score = abs(coefficient)
    best_score = abs(best_coefficient)
    tolerance = 1e-12
    if score > best_score + tolerance:
        return True
    if abs(score - best_score) <= tolerance:
        return (abs(lag), lag) < (abs(best_lag), best_lag)
    return False


def _aligned_views(
    signal_a: np.ndarray[Any, Any], signal_b: np.ndarray[Any, Any], lag: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if lag > 0:
        return signal_a[:-lag], signal_b[lag:]
    if lag < 0:
        return signal_a[-lag:], signal_b[:lag]
    return signal_a, signal_b


def _as_audio_matrix(samples: Any) -> np.ndarray[Any, Any]:
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if audio.ndim != 2:
        raise ValueError("Audio must be a one- or two-dimensional array.")
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError("Audio is empty.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite samples.")
    return audio


def _as_signal_pair(
    channel_a: Any, channel_b: Any
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    signal_a = np.asarray(channel_a, dtype=np.float64)
    signal_b = np.asarray(channel_b, dtype=np.float64)
    if signal_a.ndim != 1 or signal_b.ndim != 1:
        raise ValueError("Channel-pair inputs must be one-dimensional.")
    if signal_a.size == 0 or signal_a.size != signal_b.size:
        raise ValueError("Channel-pair inputs must have equal non-zero length.")
    if not np.all(np.isfinite(signal_a)) or not np.all(np.isfinite(signal_b)):
        raise ValueError("Channel-pair inputs contain non-finite samples.")
    return signal_a, signal_b


def _prefix_sum(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(values)))


def _interval_sum(
    prefix: np.ndarray[Any, Any], start: int, end: int
) -> float:
    return float(prefix[end] - prefix[start])


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _ratio_to_db(ratio: float | None) -> float | None:
    if ratio is None or ratio <= 0:
        return None
    value = 20.0 * math.log10(ratio)
    return float(value) if math.isfinite(value) else None


def _positive_finite_float(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label.capitalize()} must be a positive finite number.") from error
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{label.capitalize()} must be a positive finite number.")
    return converted
