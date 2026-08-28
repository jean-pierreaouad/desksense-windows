"""Deterministic time-domain features for retained DeskSense tap windows."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


PRIMARY_FEATURE_NAME = "peak_ratio_db_ch2_minus_ch1"
DESCRIPTIVE_FEATURE_NAMES = (
    "rms_ratio_db_ch2_minus_ch1",
    PRIMARY_FEATURE_NAME,
    "zero_lag_pearson_correlation",
    "normalized_channel_difference_energy",
)
_NUMERICAL_EPSILON_FACTOR = 64.0


def extract_two_channel_features(samples: Any) -> dict[str, float | None]:
    """Extract the fixed Phase 2B feature set from one tap window.

    Stored float32 samples are converted to float64 for calculation only. The
    input array is never modified. Amplitude metrics use the complete retained
    window without alignment or filtering; relationship metrics remove each
    channel's mean first.
    """

    audio = _as_audio_matrix(samples)
    if audio.shape[1] < 2:
        raise ValueError("At least two channels are required for Phase 2B features.")

    channel_1 = audio[:, 0]
    channel_2 = audio[:, 1]
    channel_1_rms = float(np.sqrt(np.mean(np.square(channel_1))))
    channel_2_rms = float(np.sqrt(np.mean(np.square(channel_2))))
    channel_1_peak = float(np.max(np.abs(channel_1)))
    channel_2_peak = float(np.max(np.abs(channel_2)))

    rms_ratio = safe_ratio(channel_2_rms, channel_1_rms)
    peak_ratio = safe_ratio(channel_2_peak, channel_1_peak)
    return {
        "channel_1_rms": channel_1_rms,
        "channel_2_rms": channel_2_rms,
        "rms_ratio_ch2_to_ch1": rms_ratio,
        "rms_ratio_db_ch2_minus_ch1": ratio_to_db(rms_ratio),
        "channel_1_peak_absolute": channel_1_peak,
        "channel_2_peak_absolute": channel_2_peak,
        "peak_ratio_ch2_to_ch1": peak_ratio,
        PRIMARY_FEATURE_NAME: ratio_to_db(peak_ratio),
        "zero_lag_pearson_correlation": pearson_correlation(
            channel_1, channel_2
        ),
        "normalized_channel_difference_energy": (
            normalized_difference_energy(channel_1, channel_2)
        ),
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite non-negative ratio, or ``None`` when undefined."""

    numerator_value = _finite_nonnegative_float(numerator, "ratio numerator")
    denominator_value = _finite_nonnegative_float(
        denominator, "ratio denominator"
    )
    tolerance = (
        _NUMERICAL_EPSILON_FACTOR
        * np.finfo(np.float64).eps
        * max(1.0, numerator_value, denominator_value)
    )
    if denominator_value <= tolerance:
        return None
    value = numerator_value / denominator_value
    return float(value) if math.isfinite(value) else None


def ratio_to_db(ratio: float | None) -> float | None:
    """Convert an amplitude ratio to dB without emitting infinities."""

    if ratio is None:
        return None
    value = _finite_nonnegative_float(ratio, "amplitude ratio")
    if value <= 0.0:
        return None
    converted = 20.0 * math.log10(value)
    return float(converted) if math.isfinite(converted) else None


def pearson_correlation(channel_1: Any, channel_2: Any) -> float | None:
    """Return zero-lag Pearson correlation after DC removal."""

    signal_1, signal_2 = _as_signal_pair(channel_1, channel_2)
    centered_1 = signal_1 - float(np.mean(signal_1))
    centered_2 = signal_2 - float(np.mean(signal_2))
    energy_1 = float(np.dot(centered_1, centered_1))
    energy_2 = float(np.dot(centered_2, centered_2))
    if _centered_energy_is_zero(signal_1, energy_1):
        return None
    if _centered_energy_is_zero(signal_2, energy_2):
        return None
    value = float(
        np.dot(centered_1, centered_2) / math.sqrt(energy_1 * energy_2)
    )
    return float(np.clip(value, -1.0, 1.0))


def normalized_difference_energy(
    channel_1: Any, channel_2: Any
) -> float | None:
    """Return DC-removed difference energy normalized by both energies."""

    signal_1, signal_2 = _as_signal_pair(channel_1, channel_2)
    centered_1 = signal_1 - float(np.mean(signal_1))
    centered_2 = signal_2 - float(np.mean(signal_2))
    energy_1 = float(np.dot(centered_1, centered_1))
    energy_2 = float(np.dot(centered_2, centered_2))
    denominator = energy_1 + energy_2
    tolerance = (
        _NUMERICAL_EPSILON_FACTOR
        * np.finfo(np.float64).eps
        * max(abs(energy_1), abs(energy_2), np.finfo(np.float64).tiny)
    )
    if denominator <= max(tolerance, np.finfo(np.float64).tiny):
        return None
    difference = centered_1 - centered_2
    value = float(np.dot(difference, difference) / denominator)
    return float(np.clip(value, 0.0, 2.0))


def feature_definitions() -> dict[str, dict[str, Any]]:
    """Return JSON-safe definitions for every emitted numerical feature."""

    return {
        "channel_1_rms": {
            "definition": "sqrt(mean(channel_1 ** 2)) over the full tap window",
            "units": "normalized amplitude",
        },
        "channel_2_rms": {
            "definition": "sqrt(mean(channel_2 ** 2)) over the full tap window",
            "units": "normalized amplitude",
        },
        "rms_ratio_ch2_to_ch1": {
            "definition": "channel_2_rms / channel_1_rms",
            "units": "ratio",
        },
        "rms_ratio_db_ch2_minus_ch1": {
            "definition": "20 * log10(channel_2_rms / channel_1_rms)",
            "units": "dB",
        },
        "channel_1_peak_absolute": {
            "definition": "max(abs(channel_1)) over the full tap window",
            "units": "normalized amplitude",
        },
        "channel_2_peak_absolute": {
            "definition": "max(abs(channel_2)) over the full tap window",
            "units": "normalized amplitude",
        },
        "peak_ratio_ch2_to_ch1": {
            "definition": "channel_2_peak_absolute / channel_1_peak_absolute",
            "units": "ratio",
        },
        PRIMARY_FEATURE_NAME: {
            "definition": (
                "20 * log10(channel_2_peak_absolute / "
                "channel_1_peak_absolute)"
            ),
            "units": "dB",
            "role": "predeclared primary simple LEFT/RIGHT baseline feature",
        },
        "zero_lag_pearson_correlation": {
            "definition": (
                "Pearson correlation after subtracting each channel mean; "
                "no temporal alignment"
            ),
            "units": "coefficient",
        },
        "normalized_channel_difference_energy": {
            "definition": (
                "sum((ch1_centered - ch2_centered) ** 2) / "
                "(sum(ch1_centered ** 2) + sum(ch2_centered ** 2))"
            ),
            "units": "normalized energy",
            "interpretation": (
                "0 means identical DC-removed signals, about 1 is typical for "
                "equal-energy uncorrelated signals, and 2 means equal-energy "
                "polarity inversion."
            ),
        },
    }


def _as_audio_matrix(samples: Any) -> np.ndarray[Any, Any]:
    try:
        audio = np.asarray(samples, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Tap-window samples must be numeric.") from error
    if audio.ndim != 2:
        raise ValueError("Tap-window samples must be a two-dimensional array.")
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError("Tap-window samples must not be empty.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Tap-window samples contain non-finite values.")
    return audio


def _as_signal_pair(
    channel_1: Any, channel_2: Any
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    signal_1 = np.asarray(channel_1, dtype=np.float64)
    signal_2 = np.asarray(channel_2, dtype=np.float64)
    if signal_1.ndim != 1 or signal_2.ndim != 1:
        raise ValueError("Channel inputs must be one-dimensional.")
    if signal_1.size == 0 or signal_1.size != signal_2.size:
        raise ValueError("Channel inputs must have equal non-zero length.")
    if not np.all(np.isfinite(signal_1)) or not np.all(np.isfinite(signal_2)):
        raise ValueError("Channel inputs contain non-finite values.")
    return signal_1, signal_2


def _centered_energy_is_zero(
    signal: np.ndarray[Any, Any], centered_energy: float
) -> bool:
    raw_energy = float(np.dot(signal, signal))
    mean_energy = float(signal.size * float(np.mean(signal)) ** 2)
    tolerance = (
        _NUMERICAL_EPSILON_FACTOR
        * np.finfo(np.float64).eps
        * max(raw_energy, mean_energy, np.finfo(np.float64).tiny)
    )
    return centered_energy <= max(tolerance, np.finfo(np.float64).tiny)


def _finite_nonnegative_float(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label.capitalize()} must be finite and non-negative.") from error
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label.capitalize()} must be finite and non-negative.")
    return converted
