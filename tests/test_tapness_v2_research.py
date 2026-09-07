from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import desksense.tapness_v2_research as research
from desksense.robustness_dataset import (
    POSITIVE_RECORD_TYPE,
    LoadedRobustnessDataset,
    RobustnessRecord,
)
from desksense.streaming import (
    ArmedBlockDiagnostic,
    CandidateStartDiagnostic,
    DetectionResult,
    DetectorDiagnosticCounters,
    DetectorDiagnosticSnapshot,
    DetectorState,
)
from desksense.tapness_v2_research import (
    TAPNESS_V2_RESEARCH_FEATURE_NAMES,
    TapnessV2ResearchError,
    evaluate_tapness_v2_research_pair,
    extract_tapness_v2_research_features,
    fit_research_logistic_model,
    grouped_v2_research_folds,
    tapness_v2_research_feature_schema,
    transform_tapness_v2_research_features,
    write_tapness_v2_research_report,
)
from desksense.tapness import robustness_dataset_fingerprint


SAMPLE_RATE = 48_000.0
FRAME_COUNT = 9_600
CENTER = 4_800


def _extract(candidate: np.ndarray, *, onset: int = 4_400) -> dict[str, float]:
    return extract_tapness_v2_research_features(
        candidate,
        sample_rate_hz=SAMPLE_RATE,
        onset_frame_index=onset,
        center_frame_index=CENTER,
        window_start_frame_index=0,
        window_end_frame_index_exclusive=FRAME_COUNT,
    )


def _candidate() -> np.ndarray:
    return np.zeros((FRAME_COUNT, 2), dtype=np.float32)


def _tone(frequency_hz: float) -> np.ndarray:
    time = np.arange(2_400, dtype=np.float64) / SAMPLE_RATE
    return np.sin(2.0 * np.pi * frequency_hz * time).astype(np.float32)


def test_feature_schema_freezes_exact_order_and_geometry() -> None:
    schema = tapness_v2_research_feature_schema()

    assert TAPNESS_V2_RESEARCH_FEATURE_NAMES == (
        "spectral_bandwidth_hz",
        "post_0_100_zero_crossing_rate",
    )
    assert schema["ordered_feature_names"] == list(TAPNESS_V2_RESEARCH_FEATURE_NAMES)
    assert schema["candidate_domain"] == {
        "sample_rate_hz": 48_000.0,
        "channel_count": 2,
        "frame_count": 9_600,
        "center_offset_frames": 4_800,
        "raw_candidate_modified": False,
    }
    interval = schema["features"][0]["candidate_interval"]
    assert interval["start_offset_frames"] == 4_800
    assert interval["end_offset_frames_exclusive"] == 7_200
    assert interval["frame_count"] == 2_400
    assert "np.hanning(2400)" in schema["features"][0]["window"]
    assert schema["features"][1]["candidate_interval"]["maximum_frame_count"] == 4_800


def test_spectral_interval_ignores_audio_after_frame_7200() -> None:
    quiet = _candidate()
    outside = quiet.copy()
    outside[7_200:, 0] = _tone(1_000.0)

    assert _extract(quiet)["spectral_bandwidth_hz"] == 0.0
    assert _extract(outside)["spectral_bandwidth_hz"] == 0.0


def test_hann_fft_power_bandwidth_for_known_single_tone() -> None:
    candidate = _candidate()
    candidate[CENTER:7_200, 0] = _tone(1_000.0)

    features = _extract(candidate)

    assert features["spectral_bandwidth_hz"] == pytest.approx(
        11.551819495319968, abs=1.0e-10
    )


def test_hann_fft_power_bandwidth_for_known_two_frequency_signal() -> None:
    candidate = _candidate()
    candidate[CENTER:7_200, 0] = _tone(1_000.0) + _tone(3_000.0)

    features = _extract(candidate)

    assert features["spectral_bandwidth_hz"] == pytest.approx(
        1_000.0667200598697, abs=1.0e-9
    )


def test_zero_energy_returns_zero_features() -> None:
    assert _extract(_candidate()) == {
        "spectral_bandwidth_hz": 0.0,
        "post_0_100_zero_crossing_rate": 0.0,
    }


def test_zcr_known_alternating_signal() -> None:
    candidate = _candidate()
    alternating = np.where(np.arange(4_800) % 2 == 0, 1.0, -1.0)
    candidate[CENTER:, 0] = alternating
    candidate[CENTER:, 1] = alternating

    assert _extract(candidate, onset=CENTER)["post_0_100_zero_crossing_rate"] == 1.0


def test_zcr_no_crossing_signal() -> None:
    candidate = _candidate()
    candidate[CENTER:] = 1.0

    assert _extract(candidate, onset=CENTER)["post_0_100_zero_crossing_rate"] == 0.0


def test_zcr_averages_channels_separately() -> None:
    candidate = _candidate()
    candidate[CENTER:, 0] = np.where(np.arange(4_800) % 2 == 0, 1.0, -1.0)
    candidate[CENTER:, 1] = 1.0

    assert _extract(candidate, onset=CENTER)["post_0_100_zero_crossing_rate"] == 0.5


def test_opposite_polarity_channels_are_not_averaged_away() -> None:
    candidate = _candidate()
    tone = _tone(1_000.0)
    candidate[CENTER:7_200, 0] = tone
    candidate[CENTER:7_200, 1] = -tone

    opposite = _extract(candidate)
    one_channel = _candidate()
    one_channel[CENTER:7_200, 0] = tone
    reference = _extract(one_channel)

    assert opposite["spectral_bandwidth_hz"] == pytest.approx(
        reference["spectral_bandwidth_hz"]
    )
    assert opposite["spectral_bandwidth_hz"] > 0.0


def test_extraction_does_not_modify_input_and_is_deterministic() -> None:
    random = np.random.default_rng(19)
    candidate = random.normal(0.0, 0.1, (FRAME_COUNT, 2)).astype(np.float32)
    before = candidate.tobytes()

    first = _extract(candidate)
    second = _extract(candidate)

    assert candidate.tobytes() == before
    assert first == second


@pytest.mark.parametrize(
    ("candidate", "overrides"),
    [
        (np.zeros((9_599, 2), np.float32), {}),
        (np.zeros((9_600, 1), np.float32), {}),
        (np.full((9_600, 2), np.nan, np.float32), {}),
        (_candidate(), {"sample_rate_hz": 44_100.0}),
        (_candidate(), {"onset_frame_index": -1}),
        (_candidate(), {"center_frame_index": 4_799}),
        (_candidate(), {"window_end_frame_index_exclusive": 9_599}),
    ],
)
def test_invalid_candidate_or_timing_fails_safely(
    candidate: np.ndarray, overrides: dict[str, object]
) -> None:
    arguments: dict[str, object] = {
        "sample_rate_hz": SAMPLE_RATE,
        "onset_frame_index": 4_400,
        "center_frame_index": CENTER,
        "window_start_frame_index": 0,
        "window_end_frame_index_exclusive": FRAME_COUNT,
    }
    arguments.update(overrides)

    with pytest.raises((ValueError, TapnessV2ResearchError)):
        extract_tapness_v2_research_features(candidate, **arguments)


def test_transform_uses_exact_order_and_fixed_safe_logs() -> None:
    transformed = transform_tapness_v2_research_features(
        {
            "post_0_100_zero_crossing_rate": 0.25,
            "spectral_bandwidth_hz": 100.0,
        }
    )

    np.testing.assert_array_equal(transformed, [math.log(100.0), math.log(0.25)])
    zeros = transform_tapness_v2_research_features(
        {"spectral_bandwidth_hz": 0.0, "post_0_100_zero_crossing_rate": 0.0}
    )
    np.testing.assert_array_equal(zeros, [math.log(1.0e-12), math.log(1.0e-12)])


@pytest.mark.parametrize(
    "features",
    [
        {"spectral_bandwidth_hz": 1.0},
        {
            "spectral_bandwidth_hz": -1.0,
            "post_0_100_zero_crossing_rate": 0.1,
        },
        {
            "spectral_bandwidth_hz": 1.0,
            "post_0_100_zero_crossing_rate": np.inf,
        },
    ],
)
def test_transform_rejects_missing_negative_or_nonfinite_values(
    features: dict[str, float]
) -> None:
    with pytest.raises(TapnessV2ResearchError):
        transform_tapness_v2_research_features(features)


def test_schema_and_module_have_no_spatial_inputs_or_dependencies() -> None:
    schema = tapness_v2_research_feature_schema()

    assert schema["spatial_inputs_used"] is False
    assert all(
        forbidden not in " ".join(TAPNESS_V2_RESEARCH_FEATURE_NAMES).lower()
        for forbidden in ("left", "right", "zone", "margin", "ch2_minus_ch1")
    )
    assert "extract_two_channel_features" not in vars(research)
    assert "classify_peak_ratio_value" not in vars(research)


def test_research_model_standardization_uses_only_supplied_training_rows() -> None:
    matrix = np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])
    labels = np.asarray([0, 0, 1, 1])

    model = fit_research_logistic_model(matrix, labels)

    np.testing.assert_allclose(model.means, [3.0, 5.0])
    np.testing.assert_allclose(model.scales, np.std(matrix, axis=0))


def _group_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target, prefix in ((1, "p"), (0, "n")):
        for index in range(5):
            repeat = 2 if target == 0 and index == 0 else 1
            for event in range(repeat):
                rows.append(
                    {
                        "session_label": "A",
                        "target": target,
                        "source_group_id": f"A-{prefix}{index}",
                        "event": event,
                    }
                )
    return rows


def test_grouped_folds_keep_every_record_wholly_in_one_fold() -> None:
    rows = _group_rows()
    folds = grouped_v2_research_folds(rows)
    fold_by_index = {
        row_index: fold_index
        for fold_index, fold in enumerate(folds)
        for row_index in fold
    }
    group_folds: dict[str, set[int]] = {}
    for index, row in enumerate(rows):
        group_folds.setdefault(str(row["source_group_id"]), set()).add(
            fold_by_index[index]
        )

    assert all(len(value) == 1 for value in group_folds.values())
    assert len(folds) == 5


def test_inner_oof_reports_disjoint_groups_and_training_only_means() -> None:
    rows = _group_rows()
    matrix = np.asarray(
        [
            [float(index), float(index * index + int(row["target"]))]
            for index, row in enumerate(rows)
        ]
    )

    _, _, _, reports = research._fit_with_grouped_oof_threshold(rows, matrix)

    for report in reports:
        held_out = set(report["held_out_group_ids"])
        training = set(report["training_group_ids"])
        assert held_out.isdisjoint(training)
        training_indexes = [
            index
            for index, row in enumerate(rows)
            if row["source_group_id"] in training
        ]
        np.testing.assert_allclose(
            report["training_only_standardization"]["means"],
            np.mean(matrix[training_indexes], axis=0),
        )


def test_import_does_not_load_sounddevice() -> None:
    command = (
        "import sys; import desksense.tapness_v2_research; "
        "print('sounddevice' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "False"


def test_report_writer_is_json_safe_and_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    report = {
        "report_type": "desksense_phase3b3_stage2_v2_research_reproduction",
        "value": 1.0,
    }

    write_tapness_v2_research_report(report, destination)

    assert json.loads(destination.read_text(encoding="utf-8"))["value"] == 1.0
    with pytest.raises(TapnessV2ResearchError, match="Refusing to overwrite"):
        write_tapness_v2_research_report(report, destination)


def test_detector_rejection_with_candidate_bytes_is_not_stage2_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = ArmedBlockDiagnostic(
        stream_epoch=0,
        block_start_frame_index=49_920,
        block_end_frame_index_exclusive=50_160,
        block_rms=0.1,
        block_peak_absolute=0.2,
        block_crest_factor=2.0,
        learned_noise_floor_rms=0.001,
        required_rms_threshold=0.01,
        required_peak_threshold=0.02,
        required_crest_threshold=3.0,
        rms_ratio=10.0,
        peak_ratio=10.0,
        crest_ratio=2.0 / 3.0,
        all_gates_score=2.0 / 3.0,
    )
    start = CandidateStartDiagnostic(
        stream_epoch=0,
        onset_frame_index=50_000,
        block=block,
        route="strong_impact_recovery",
    )
    rejected = DetectionResult(
        status="rejected",
        rejection_reasons=("near_clipping",),
        stream_epoch=0,
        onset_frame_index=50_000,
        center_frame_index=50_100,
        window_start_frame_index=45_300,
        window_end_frame_index_exclusive=54_900,
        finalized_after_frame_index_exclusive=54_900,
        candidate_window=np.zeros((9_600, 2), dtype=np.float32),
        metrics={},
        state_before=DetectorState.COLLECTING,
        state_after=DetectorState.REFRACTORY,
    )

    class RejectedDetector:
        def __init__(self) -> None:
            self.emitted = False

        def process_chunk(self, chunk):
            if self.emitted:
                return ()
            self.emitted = True
            return (rejected,)

        def drain_candidate_start_diagnostics(self):
            return (start,) if self.emitted and not hasattr(self, "drained") else ()

        def diagnostic_snapshot(self):
            self.drained = True
            counters = DetectorDiagnosticCounters(
                onset_candidates_started=1,
                completed_rejections=1,
            )
            return DetectorDiagnosticSnapshot(
                stream_epoch=0,
                state=DetectorState.REFRACTORY,
                processed_frame_count=96_000,
                learned_noise_floor_rms=0.001,
                cumulative_counters=counters,
                interval_counters=counters,
                closest_armed_block=None,
                buffered_candidate_start_record_count=0,
            )

    record = RobustnessRecord(
        metadata={
            "record_id": "rejected-positive",
            "record_type": POSITIVE_RECORD_TYPE,
            "intended_zone": "LEFT",
            "intended_strength": "normal",
            "guided_cue": {
                "intended_event_association": {
                    "start_frame_index_inclusive": 43_200,
                    "end_frame_index_exclusive": 79_200,
                }
            },
        },
        capture=np.zeros((96_000, 2), dtype=np.float32),
        path=Path("unused.npz"),
    )
    dataset = LoadedRobustnessDataset(
        directory=Path("unused"),
        session={"session_id": "rejected-test"},
        records=(record,),
    )
    monkeypatch.setattr(
        research,
        "extract_tapness_v2_research_features",
        lambda *args, **kwargs: pytest.fail("rejected result reached Stage 2"),
    )

    rows, population, _ = research._collect_research_candidates(
        "test", dataset, RejectedDetector
    )

    assert rows == []
    assert population["associated_candidate_start_count"] == 1
    assert population["positive_associated_candidate_start_identities"] == [
        {
            "source_record_id": "rejected-positive",
            "onset_frame_index": 50_000,
            "candidate_start_route": "strong_impact_recovery",
        }
    ]
    assert population["negative_activity_candidate_start_identities"] == []
    assert population["associated_completed_detections"] == 0
    assert population["associated_completed_rejections"] == 1


def test_experiment_19_exact_local_reproduction_when_datasets_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    session_a = root / "datasets" / "20260901T131308.362205Z-f9e2b1ec"
    session_b = root / "datasets" / "20260904T171453.458221Z-4ecb16ad"
    if not session_a.is_dir() or not session_b.is_dir():
        pytest.skip("Ignored local Experiment 19 datasets are unavailable.")
    before = {
        "A": robustness_dataset_fingerprint(session_a)["sha256"],
        "B": robustness_dataset_fingerprint(session_b)["sha256"],
    }

    report = evaluate_tapness_v2_research_pair(session_a, session_b)

    after = {
        "A": robustness_dataset_fingerprint(session_a)["sha256"],
        "B": robustness_dataset_fingerprint(session_b)["sha256"],
    }
    assert before == after == {
        "A": "88a003966141d15858a2afec390c42e567439eb5f538b958e7aa60ee7638ab83",
        "B": "6b6b7cff6ff4f597d0c4c9bc8ccea3544cda99e4d25bf9be1cc732ee6b9ee5ac",
    }
    assert report["experiment_19_reproduction_gate"]["passed"] is True
    assert report["candidate_populations"]["A"]["positive_candidate_count"] == 29
    assert report["candidate_populations"]["A"]["negative_candidate_count"] == 50
    assert report["candidate_populations"]["B"]["positive_candidate_count"] == 30
    assert report["candidate_populations"]["B"]["negative_candidate_count"] == 74
    a_to_b = report["development_results"]["train_A_test_B"]["metrics"]
    b_to_a = report["development_results"]["train_B_test_A"]["metrics"]
    pooled = report["development_results"]["pooled_nested_grouped_oof"]["metrics"]
    assert (a_to_b["positive_attempts_accepted"], a_to_b["negative_false_accept_count"]) == (30, 2)
    assert (b_to_a["positive_attempts_accepted"], b_to_a["negative_false_accept_count"]) == (29, 1)
    assert (pooled["positive_attempts_accepted"], pooled["negative_false_accept_count"]) == (59, 3)
    assert a_to_b["false_accepts_by_activity"] == {"desk_object_interaction": 2}
    assert b_to_a["false_accepts_by_activity"] == {"desk_object_interaction": 1}
    assert pooled["false_accepts_by_activity"] == {"desk_object_interaction": 3}
    assert "candidate_window" not in json.dumps(report, allow_nan=False)
