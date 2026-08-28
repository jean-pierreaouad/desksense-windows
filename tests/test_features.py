from __future__ import annotations

import json
import math

import numpy as np
import pytest

from desksense.features import (
    PRIMARY_FEATURE_NAME,
    extract_two_channel_features,
    feature_definitions,
    normalized_difference_energy,
    pearson_correlation,
    ratio_to_db,
    safe_ratio,
)


def test_extract_features_matches_known_amplitude_and_energy_values() -> None:
    channel_1 = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    channel_2 = 2.0 * channel_1

    features = extract_two_channel_features(
        np.column_stack((channel_1, channel_2))
    )

    expected_ratio_db = 20.0 * math.log10(2.0)
    assert features["channel_1_rms"] == pytest.approx(1.0)
    assert features["channel_2_rms"] == pytest.approx(2.0)
    assert features["rms_ratio_ch2_to_ch1"] == pytest.approx(2.0)
    assert features["rms_ratio_db_ch2_minus_ch1"] == pytest.approx(
        expected_ratio_db
    )
    assert features["channel_1_peak_absolute"] == pytest.approx(1.0)
    assert features["channel_2_peak_absolute"] == pytest.approx(2.0)
    assert features["peak_ratio_ch2_to_ch1"] == pytest.approx(2.0)
    assert features[PRIMARY_FEATURE_NAME] == pytest.approx(expected_ratio_db)
    assert features["zero_lag_pearson_correlation"] == pytest.approx(1.0)
    assert features["normalized_channel_difference_energy"] == pytest.approx(
        0.2
    )


@pytest.mark.parametrize(
    ("ratio", "expected_db"),
    [
        (1.0, 0.0),
        (2.0, 20.0 * math.log10(2.0)),
        (0.5, 20.0 * math.log10(0.5)),
        (10.0, 20.0),
    ],
)
def test_ratio_to_db_uses_amplitude_decibels(
    ratio: float, expected_db: float
) -> None:
    assert ratio_to_db(ratio) == pytest.approx(expected_db)


def test_known_pearson_correlations_remove_dc_offset() -> None:
    channel_1 = np.array([-2.0, -1.0, 1.0, 2.0])
    positively_related = 3.0 * channel_1 + 10.0
    negatively_related = -2.0 * channel_1 - 4.0

    assert pearson_correlation(channel_1, positively_related) == pytest.approx(
        1.0
    )
    assert pearson_correlation(channel_1, negatively_related) == pytest.approx(
        -1.0
    )


@pytest.mark.parametrize(
    ("channel_2", "expected"),
    [
        (np.array([-1.0, 1.0, -1.0, 1.0]), 0.0),
        (np.array([1.0, -1.0, 1.0, -1.0]), 2.0),
        (np.array([-2.0, 2.0, -2.0, 2.0]), 0.2),
        (np.array([-1.0, -1.0, 1.0, 1.0]), 1.0),
    ],
)
def test_normalized_difference_energy_has_known_values(
    channel_2: np.ndarray, expected: float
) -> None:
    channel_1 = np.array([-1.0, 1.0, -1.0, 1.0])

    assert normalized_difference_energy(channel_1, channel_2) == pytest.approx(
        expected
    )


def test_silent_and_near_zero_inputs_remain_json_safe() -> None:
    silent_features = extract_two_channel_features(
        np.zeros((32, 2), dtype=np.float32)
    )
    near_zero_ratio = safe_ratio(2.0e-20, 1.0e-20)

    assert silent_features["rms_ratio_ch2_to_ch1"] is None
    assert silent_features["rms_ratio_db_ch2_minus_ch1"] is None
    assert silent_features["peak_ratio_ch2_to_ch1"] is None
    assert silent_features[PRIMARY_FEATURE_NAME] is None
    assert silent_features["zero_lag_pearson_correlation"] is None
    assert silent_features["normalized_channel_difference_energy"] is None
    assert near_zero_ratio is None
    assert ratio_to_db(near_zero_ratio) is None
    json.dumps(silent_features, allow_nan=False)


def test_feature_output_contains_only_finite_numbers_or_null() -> None:
    generator = np.random.default_rng(20260827)
    samples = generator.normal(0.0, 0.02, (1_000, 2)).astype(np.float32)

    features = extract_two_channel_features(samples)

    assert all(
        value is None or (isinstance(value, float) and math.isfinite(value))
        for value in features.values()
    )
    json.dumps(features, allow_nan=False)


def test_non_finite_samples_are_rejected() -> None:
    samples = np.zeros((8, 2), dtype=np.float32)
    samples[3, 1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        extract_two_channel_features(samples)


def test_primary_feature_definition_records_predeclared_baseline_role() -> None:
    definitions = feature_definitions()

    assert definitions[PRIMARY_FEATURE_NAME]["units"] == "dB"
    assert "predeclared primary" in definitions[PRIMARY_FEATURE_NAME]["role"]
    json.dumps(definitions, allow_nan=False)
