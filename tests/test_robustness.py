from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import desksense.robustness as robustness

from desksense.robustness import (
    NEGATIVE_ACTIVITIES,
    POSITIVE_STRENGTHS,
    POSITIVE_ZONES,
    RobustnessCollectionParameters,
    extract_descriptive_tapness_metrics,
    negative_robustness_plan,
    positive_robustness_plan,
    record_robustness_segment,
    replay_robustness_dataset,
    run_guided_robustness_collection,
    write_robustness_report,
)
from desksense.robustness_dataset import (
    NEGATIVE_RECORD_TYPE,
    POSITIVE_RECORD_TYPE,
    create_robustness_dataset_session,
    load_robustness_dataset,
)
from desksense.streaming import (
    ArmedBlockDiagnostic,
    CandidateStartDiagnostic,
    DetectionResult,
    DetectorDiagnosticCounters,
    DetectorDiagnosticSnapshot,
    DetectorState,
    StreamingTapDetector,
)


def _endpoint_from_baseline() -> dict:
    artifact = json.loads(
        Path("baselines/lenovo-left-right-v1.json").read_text(encoding="utf-8")
    )
    compatibility = artifact["source_development_dataset"]["compatibility"]
    return {
        "index_at_collection": 18,
        "name": compatibility["endpoint_name"],
        "host_api": {"index": 3, "name": compatibility["host_api_name"]},
    }


def _session_metadata(endpoint: dict | None = None) -> dict:
    return {
        "source_mode": "synthetic_test",
        "selected_endpoint": endpoint or {
            "index_at_collection": 1,
            "name": "Microphone Array",
            "host_api": {"index": 0, "name": "Windows WASAPI"},
        },
        "capture_domain": {
            "sample_rate_hz": 48_000.0,
            "channel_count": 2,
            "dtype": "float32",
        },
        "positive_design": {
            "requested_attempt_count": 2,
            "plan": [
                {
                    "collection_order_index": 1,
                    "zone": "RIGHT",
                    "strength": "normal",
                    "attempt_number_within_condition": 1,
                },
                {
                    "collection_order_index": 2,
                    "zone": "LEFT",
                    "strength": "light",
                    "attempt_number_within_condition": 1,
                },
            ],
        },
        "negative_design": {
            "requested_segment_count": 1,
            "plan": [
                {
                    "collection_order_index": 3,
                    "activity": "typing",
                    "repetition_index": 1,
                }
            ],
        },
    }


def _record_metadata(
    session_id: str, order: int, capture: np.ndarray, endpoint: dict
) -> dict:
    return {
        "session_id": session_id,
        "collection_order_index": order,
        "device": endpoint,
        "sample_rate_hz": 48_000.0,
        "channel_count": 2,
        "capture_dtype": "float32",
        "capture_frames": capture.shape[0],
        "capture_duration_seconds": capture.shape[0] / 48_000.0,
    }


def _impulse_capture(
    frames: int,
    *,
    center: int = 50_000,
    right_dominant: bool = True,
) -> np.ndarray:
    capture = np.zeros((frames, 2), dtype=np.float32)
    capture[center, 0] = 0.04 if right_dominant else 0.10
    capture[center, 1] = 0.10 if right_dominant else 0.04
    return capture


def _candidate_start(onset: int) -> CandidateStartDiagnostic:
    block_start = onset - (onset % 240)
    block = ArmedBlockDiagnostic(
        stream_epoch=0,
        block_start_frame_index=block_start,
        block_end_frame_index_exclusive=block_start + 240,
        block_rms=0.01,
        block_peak_absolute=0.10,
        block_crest_factor=10.0,
        learned_noise_floor_rms=0.001,
        required_rms_threshold=0.003,
        required_peak_threshold=0.008,
        required_crest_threshold=3.0,
        rms_ratio=10.0 / 3.0,
        peak_ratio=12.5,
        crest_ratio=10.0 / 3.0,
        all_gates_score=10.0 / 3.0,
    )
    return CandidateStartDiagnostic(
        stream_epoch=0,
        onset_frame_index=onset,
        block=block,
    )


def _detection_result(onset: int, *, status: str = "detected") -> DetectionResult:
    window = np.zeros((9_600, 2), dtype=np.float32)
    window[4_800] = (0.04, 0.10)
    return DetectionResult(
        status=status,
        rejection_reasons=() if status == "detected" else ("test_rejection",),
        stream_epoch=0,
        onset_frame_index=onset,
        center_frame_index=onset,
        window_start_frame_index=onset - 4_800,
        window_end_frame_index_exclusive=onset + 4_800,
        finalized_after_frame_index_exclusive=onset + 4_800,
        candidate_window=window,
        metrics={"learned_noise_floor_rms": 0.001},
        state_before=DetectorState.COLLECTING,
        state_after=DetectorState.REFRACTORY,
    )


class _FakeReplayDetector:
    def __init__(
        self,
        *,
        starts: tuple[CandidateStartDiagnostic, ...] = (),
        results: tuple[DetectionResult, ...] = (),
    ) -> None:
        self.starts = starts
        self.results = results

    def process_chunk(self, _capture):
        return self.results

    def drain_candidate_start_diagnostics(self):
        return self.starts

    def diagnostic_snapshot(self) -> DetectorDiagnosticSnapshot:
        counters = DetectorDiagnosticCounters(
            onset_candidates_started=len(self.starts),
            completed_detections=sum(
                result.status == "detected" for result in self.results
            ),
            completed_rejections=sum(
                result.status == "rejected" for result in self.results
            ),
        )
        return DetectorDiagnosticSnapshot(
            stream_epoch=0,
            state=DetectorState.ARMED,
            processed_frame_count=96_000,
            learned_noise_floor_rms=0.001,
            cumulative_counters=counters,
            interval_counters=counters,
            closest_armed_block=None,
            buffered_candidate_start_record_count=0,
        )


def _replay_session(tmp_path):
    endpoint = _endpoint_from_baseline()
    session = create_robustness_dataset_session(
        tmp_path,
        _session_metadata(endpoint),
        session_id="replay-test",
        created_at_utc="2026-08-31T00:00:00+00:00",
    )
    detected = _impulse_capture(96_000)
    missed = np.zeros((96_000, 2), dtype=np.float32)
    negative = _impulse_capture(540_000, center=100_000)
    session.save_record(
        record_id="replay-test-right-normal-001",
        filename_stem="right_normal_001",
        record_type=POSITIVE_RECORD_TYPE,
        capture=detected,
        metadata={
            **_record_metadata(session.session_id, 1, detected, endpoint),
            "intended_zone": "RIGHT",
            "intended_strength": "normal",
            "attempt_number_within_condition": 1,
            "captured_at_utc": "2026-08-31T00:00:01+00:00",
            "semantic_label": "intended_desk_tap",
            "detector_acceptance_required_for_storage": False,
            "guided_cue": {
                "intended_cue_offset_frames": 43_200,
                "intended_event_association": {
                    "start_frame_index_inclusive": 43_200,
                    "end_frame_index_exclusive": 79_200,
                    "duration_seconds": 0.75,
                },
            },
        },
    )
    session.save_record(
        record_id="replay-test-left-light-001",
        filename_stem="left_light_001",
        record_type=POSITIVE_RECORD_TYPE,
        capture=missed,
        metadata={
            **_record_metadata(session.session_id, 2, missed, endpoint),
            "intended_zone": "LEFT",
            "intended_strength": "light",
            "attempt_number_within_condition": 1,
            "captured_at_utc": "2026-08-31T00:00:02+00:00",
            "semantic_label": "intended_desk_tap",
            "detector_acceptance_required_for_storage": False,
            "guided_cue": {
                "intended_cue_offset_frames": 43_200,
                "intended_event_association": {
                    "start_frame_index_inclusive": 43_200,
                    "end_frame_index_exclusive": 79_200,
                    "duration_seconds": 0.75,
                },
            },
        },
    )
    session.save_record(
        record_id="replay-test-typing-001",
        filename_stem="typing_001",
        record_type=NEGATIVE_RECORD_TYPE,
        capture=negative,
        metadata={
            **_record_metadata(session.session_id, 3, negative, endpoint),
            "activity": "typing",
            "repetition_index": 1,
            "semantic_label": "no_intended_desk_tap",
            "segment_started_at_utc": "2026-08-31T00:01:00+00:00",
            "segment_ended_at_utc": "2026-08-31T00:01:11.250000+00:00",
            "warmup_duration_seconds": 1.0,
            "warmup_end_frame_index": 48_000,
            "activity_start_frame_index": 48_000,
            "activity_end_frame_index_exclusive": 528_000,
            "activity_duration_seconds": 10.0,
            "post_activity_tail_duration_seconds": 0.25,
            "post_activity_start_frame_index": 528_000,
            "post_activity_end_frame_index_exclusive": 540_000,
        },
    )
    return session


def _single_positive_session(tmp_path, *, expected_positive_count: int = 1):
    endpoint = _endpoint_from_baseline()
    metadata = _session_metadata(endpoint)
    metadata["positive_design"] = {
        "requested_attempt_count": expected_positive_count,
        "plan": [
            {
                "collection_order_index": index,
                "zone": "RIGHT" if index == 1 else "LEFT",
                "strength": "normal" if index == 1 else "light",
                "attempt_number_within_condition": 1,
            }
            for index in range(1, expected_positive_count + 1)
        ],
    }
    metadata["negative_design"] = {
        "requested_segment_count": 0,
        "plan": [],
    }
    session = create_robustness_dataset_session(
        tmp_path,
        metadata,
        session_id="single-positive",
        created_at_utc="2026-08-31T00:00:00+00:00",
    )
    capture = np.zeros((96_000, 2), dtype=np.float32)
    session.save_record(
        record_id="single-positive-right-normal-001",
        filename_stem="right_normal_001",
        record_type=POSITIVE_RECORD_TYPE,
        capture=capture,
        metadata={
            **_record_metadata(session.session_id, 1, capture, endpoint),
            "intended_zone": "RIGHT",
            "intended_strength": "normal",
            "attempt_number_within_condition": 1,
            "captured_at_utc": "2026-08-31T00:00:01+00:00",
            "semantic_label": "intended_desk_tap",
            "detector_acceptance_required_for_storage": False,
            "guided_cue": {
                "intended_cue_offset_frames": 43_200,
                "intended_event_association": {
                    "start_frame_index_inclusive": 43_200,
                    "end_frame_index_exclusive": 79_200,
                    "duration_seconds": 0.75,
                },
            },
        },
    )
    return session


def _single_negative_session(tmp_path, *, impulse_center: int | None = None):
    endpoint = _endpoint_from_baseline()
    metadata = _session_metadata(endpoint)
    metadata["positive_design"] = {
        "requested_attempt_count": 0,
        "plan": [],
    }
    metadata["negative_design"] = {
        "requested_segment_count": 1,
        "plan": [
            {
                "collection_order_index": 1,
                "activity": "typing",
                "repetition_index": 1,
            }
        ],
    }
    session = create_robustness_dataset_session(
        tmp_path,
        metadata,
        session_id="single-negative",
        created_at_utc="2026-08-31T00:00:00+00:00",
    )
    capture = (
        np.zeros((540_000, 2), dtype=np.float32)
        if impulse_center is None
        else _impulse_capture(540_000, center=impulse_center)
    )
    session.save_record(
        record_id="single-negative-typing-001",
        filename_stem="typing_001",
        record_type=NEGATIVE_RECORD_TYPE,
        capture=capture,
        metadata={
            **_record_metadata(session.session_id, 1, capture, endpoint),
            "activity": "typing",
            "repetition_index": 1,
            "semantic_label": "no_intended_desk_tap",
            "segment_started_at_utc": "2026-08-31T00:00:01+00:00",
            "segment_ended_at_utc": "2026-08-31T00:00:12.250000+00:00",
            "warmup_duration_seconds": 1.0,
            "warmup_end_frame_index": 48_000,
            "activity_start_frame_index": 48_000,
            "activity_end_frame_index_exclusive": 528_000,
            "activity_duration_seconds": 10.0,
            "post_activity_tail_duration_seconds": 0.25,
            "post_activity_start_frame_index": 528_000,
            "post_activity_end_frame_index_exclusive": 540_000,
        },
    )
    return session


def test_default_robustness_plans_are_complete_and_round_robin() -> None:
    positives = positive_robustness_plan()
    negatives = negative_robustness_plan(start_order_index=len(positives) + 1)

    assert len(positives) == 30
    assert {
        (entry.zone, entry.strength) for entry in positives
    } == set((zone, strength) for zone in POSITIVE_ZONES for strength in POSITIVE_STRENGTHS)
    assert all(
        sum(
            item.zone == zone and item.strength == strength for item in positives
        )
        == 5
        for zone in POSITIVE_ZONES
        for strength in POSITIVE_STRENGTHS
    )
    assert [entry.collection_order_index for entry in positives] == list(range(1, 31))
    assert len(negatives) == 14
    assert {entry.activity for entry in negatives} == set(NEGATIVE_ACTIVITIES)
    assert all(
        sum(item.activity == activity for item in negatives) == 2
        for activity in NEGATIVE_ACTIVITIES
    )
    assert negatives[0].collection_order_index == 31


def test_revised_default_timing_leaves_detector_learning_before_positive_cue() -> None:
    parameters = RobustnessCollectionParameters()

    assert parameters.positive_capture_seconds == 2.0
    assert parameters.pre_cue_seconds == 0.9
    assert parameters.positive_association_seconds == 0.75
    assert parameters.negative_warmup_seconds == 1.0
    assert parameters.negative_segment_seconds == 10.0
    assert parameters.negative_post_activity_tail_seconds == 0.25

    associated = _impulse_capture(96_000, center=50_000)
    detector = StreamingTapDetector()
    results = detector.process_chunk(associated)
    starts = detector.drain_candidate_start_diagnostics()
    assert any(43_200 <= start.onset_frame_index < 79_200 for start in starts)
    assert any(43_200 <= result.onset_frame_index < 79_200 for result in results)

    early = _impulse_capture(96_000, center=9_600)
    early_detector = StreamingTapDetector()
    early_detector.process_chunk(early)
    assert early_detector.drain_candidate_start_diagnostics() == ()


def test_associated_candidate_start_defines_stage1_recall_without_completion(
    tmp_path,
) -> None:
    session = _single_positive_session(tmp_path)
    fake = _FakeReplayDetector(starts=(_candidate_start(50_000),), results=())

    report = replay_robustness_dataset(
        session.directory,
        detector_factory=lambda: fake,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    attempt = report["positive"]["attempts"][0]
    summary = report["positive"]["summary"]
    assert attempt["stage1_candidate_started"] is True
    assert attempt["associated_completed_result"] is False
    assert summary["stage1_candidate_start_recall"] == 1.0
    assert summary["associated_completion_rate"] == 0.0


def test_preassociation_completion_is_extraneous_and_not_spatially_classified(
    tmp_path,
) -> None:
    session = _single_positive_session(tmp_path)
    fake = _FakeReplayDetector(
        starts=(_candidate_start(40_000),),
        results=(_detection_result(40_000),),
    )

    report = replay_robustness_dataset(
        session.directory,
        baseline_path=Path("baselines/lenovo-left-right-v1.json"),
        detector_factory=lambda: fake,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    attempt = report["positive"]["attempts"][0]
    summary = report["positive"]["summary"]
    assert attempt["stage1_candidate_started"] is False
    assert attempt["associated_completed_result"] is False
    assert attempt["extraneous_candidate_start_count"] == 1
    assert attempt["extraneous_completed_event_count"] == 1
    assert attempt["completed_events"][0]["interval_relation"] == "pre_association"
    assert attempt["completed_events"][0]["frozen_spatial_prediction"] is None
    assert summary["stage1_candidate_start_recall"] == 0.0
    assert summary["conditional_frozen_spatial"]["classified_count"] == 0


def test_frozen_spatial_classification_uses_only_associated_completions(
    tmp_path,
) -> None:
    session = _single_positive_session(tmp_path)
    fake = _FakeReplayDetector(
        starts=(_candidate_start(40_000), _candidate_start(50_000)),
        results=(_detection_result(40_000), _detection_result(50_000)),
    )

    report = replay_robustness_dataset(
        session.directory,
        baseline_path=Path("baselines/lenovo-left-right-v1.json"),
        detector_factory=lambda: fake,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    events = report["positive"]["attempts"][0]["completed_events"]
    assert events[0]["frozen_spatial_prediction"] is None
    assert events[1]["frozen_spatial_prediction"]["predicted_zone"] == "RIGHT"
    spatial = report["positive"]["summary"]["conditional_frozen_spatial"]
    assert spatial["classified_count"] == 1
    assert spatial["correct_count"] == 1


def test_stage2_rejection_blocks_stage3_and_uses_correct_denominators(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _single_positive_session(tmp_path)
    fake = _FakeReplayDetector(
        starts=(_candidate_start(50_000),),
        results=(_detection_result(50_000),),
    )
    fake.config = StreamingTapDetector().config
    monkeypatch.setattr(robustness, "validate_tapness_stage1_config", lambda *args: None)
    monkeypatch.setattr(
        robustness,
        "classify_tapness_metrics",
        lambda *args: {
            "predicted_label": "NON_TAP",
            "tap_accepted": False,
            "uncalibrated_model_output": 0.1,
            "decision_score": -1.0,
            "decision_threshold": 0.4,
            "model_margin": -0.3,
            "tie_rule": "score >= threshold predicts TAP",
            "score_is_calibrated_probability": False,
        },
    )

    report = replay_robustness_dataset(
        session.directory,
        baseline_path=Path("baselines/lenovo-left-right-v1.json"),
        tapness_baseline_path=Path("baselines/lenovo-tapness-v1.json"),
        detector_factory=lambda: fake,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    event = report["positive"]["attempts"][0]["completed_events"][0]
    summary = report["positive"]["summary"]
    assert event["frozen_tapness_prediction"]["tap_accepted"] is False
    assert event["frozen_spatial_prediction"] is None
    assert summary["stage2_tapness"]["attempts_with_accepted_tap"] == 0
    assert summary["end_to_end"]["total_intended_taps"] == 1
    assert summary["end_to_end"]["intended_taps_accepted_and_spatially_correct"] == 0


def test_stage1_policy_mismatch_fails_before_offline_stage2(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _single_positive_session(tmp_path)
    fake = _FakeReplayDetector(
        starts=(_candidate_start(50_000),),
        results=(_detection_result(50_000),),
    )
    fake.config = replace(
        StreamingTapDetector().config,
        minimum_crest_factor=2.5,
    )
    monkeypatch.setattr(
        robustness,
        "classify_tapness_metrics",
        lambda *args: pytest.fail("Stage 2 must not run after a policy mismatch"),
    )

    with pytest.raises(robustness.RobustnessError, match="incompatible") as error:
        replay_robustness_dataset(
            session.directory,
            tapness_baseline_path=Path("baselines/lenovo-tapness-v1.json"),
            detector_factory=lambda: fake,
            now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
        )

    assert error.value.category == "tapness_baseline_error"


def test_negative_replay_excludes_warmup_events_and_uses_labeled_denominator(
    tmp_path,
) -> None:
    session = _single_negative_session(tmp_path)
    fake = _FakeReplayDetector(
        starts=(_candidate_start(40_000), _candidate_start(100_000)),
        results=(_detection_result(40_000), _detection_result(100_000)),
    )

    report = replay_robustness_dataset(
        session.directory,
        detector_factory=lambda: fake,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    segment = report["negative"]["segments"][0]
    summary = report["negative"]["summary"]
    assert segment["stored_capture_duration_seconds"] == 11.25
    assert segment["warmup_duration_seconds"] == 1.0
    assert segment["labeled_activity_duration_seconds"] == 10.0
    assert segment["warmup_candidate_start_count"] == 1
    assert segment["activity_candidate_start_count"] == 1
    assert summary["warmup_candidate_starts"] == 1
    assert summary["warmup_completed_detections"] == 1
    assert summary["activity_associated_candidate_starts"] == 1
    assert summary["activity_associated_completed_detections"] == 1
    assert summary["candidate_start_rate_per_labeled_minute"] == pytest.approx(6.0)
    assert summary[
        "completed_false_event_rate_per_labeled_minute"
    ] == pytest.approx(6.0)


def test_near_activity_end_candidate_completes_in_tail_and_counts_by_onset(
    tmp_path,
) -> None:
    session = _single_negative_session(tmp_path, impulse_center=527_000)

    report = replay_robustness_dataset(
        session.directory,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    segment = report["negative"]["segments"][0]
    assert segment["stored_capture_duration_seconds"] == 11.25
    assert segment["activity_start_frame_index"] == 48_000
    assert segment["activity_end_frame_index_exclusive"] == 528_000
    assert segment["post_activity_start_frame_index"] == 528_000
    assert segment["post_activity_end_frame_index_exclusive"] == 540_000
    assert segment["activity_candidate_start_count"] == 1
    assert segment["activity_completed_detection_count"] == 1
    event = segment["completed_events"][0]
    assert event["interval_relation"] == "activity"
    assert event["onset_frame_index"] < 528_000
    assert event["finalized_after_frame_index_exclusive"] > 528_000


def test_tail_onset_is_reported_but_excluded_from_activity_rates(tmp_path) -> None:
    session = _single_negative_session(tmp_path)
    fake = _FakeReplayDetector(
        starts=(_candidate_start(530_000),),
        results=(_detection_result(530_000),),
    )

    report = replay_robustness_dataset(
        session.directory,
        detector_factory=lambda: fake,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    segment = report["negative"]["segments"][0]
    summary = report["negative"]["summary"]
    assert segment["candidate_starts"][0]["interval_relation"] == "post_activity"
    assert segment["completed_events"][0]["interval_relation"] == "post_activity"
    assert segment["activity_candidate_start_count"] == 0
    assert segment["activity_completed_detection_count"] == 0
    assert segment["post_activity_candidate_start_count"] == 1
    assert segment["post_activity_completed_detection_count"] == 1
    assert summary["labeled_activity_duration_seconds"] == 10.0
    assert summary["post_activity_tail_duration_seconds"] == 0.25
    assert summary["candidate_start_rate_per_labeled_minute"] == 0.0
    assert summary["completed_false_event_rate_per_labeled_minute"] == 0.0
    assert summary["post_activity_candidate_starts"] == 1
    assert summary["post_activity_completed_detections"] == 1


def test_partial_and_full_plans_report_collection_completeness(tmp_path) -> None:
    partial = _single_positive_session(
        tmp_path / "partial", expected_positive_count=2
    )
    partial_report = replay_robustness_dataset(
        partial.directory,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )
    assert partial_report["dataset_integrity"]["status"] == "passed"
    assert partial_report["collection_completeness"] == {
        "expected_positive_attempts": 2,
        "observed_positive_attempts": 1,
        "expected_negative_segments": 0,
        "observed_negative_segments": 0,
        "positive_plan_complete": False,
        "negative_plan_complete": True,
        "collection_complete": False,
    }

    full = _single_positive_session(tmp_path / "full")
    full_report = replay_robustness_dataset(
        full.directory,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )
    assert full_report["collection_completeness"]["collection_complete"] is True


def test_default_partial_session_reports_expected_30_positive_and_14_negative(
    tmp_path,
) -> None:
    endpoint = _endpoint_from_baseline()
    metadata = _session_metadata(endpoint)
    positives = positive_robustness_plan()
    negatives = negative_robustness_plan(
        start_order_index=len(positives) + 1
    )
    metadata["positive_design"] = {
        "requested_attempt_count": len(positives),
        "plan": [asdict(item) for item in positives],
    }
    metadata["negative_design"] = {
        "requested_segment_count": len(negatives),
        "plan": [asdict(item) for item in negatives],
    }
    session = create_robustness_dataset_session(
        tmp_path,
        metadata,
        session_id="empty-default-plan",
        created_at_utc="2026-08-31T00:00:00+00:00",
    )

    report = replay_robustness_dataset(
        session.directory,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    completeness = report["collection_completeness"]
    assert completeness["expected_positive_attempts"] == 30
    assert completeness["observed_positive_attempts"] == 0
    assert completeness["expected_negative_segments"] == 14
    assert completeness["observed_negative_segments"] == 0
    assert completeness["collection_complete"] is False


def test_guided_collection_retains_quiet_intended_attempts_and_negative_labels(
    tmp_path, audio_backend_factory
) -> None:
    backend = audio_backend_factory()
    parameters = RobustnessCollectionParameters(
        attempts_per_condition=1,
        positive_capture_seconds=0.010,
        pre_cue_seconds=0.001,
        positive_association_seconds=0.004,
        negative_segments_per_activity=1,
        negative_segment_seconds=0.010,
        negative_warmup_seconds=0.002,
        negative_post_activity_tail_seconds=0.002,
    )
    positive_calls = 0
    negative_calls = 0

    def positive_capture(*_args, **_kwargs):
        nonlocal positive_calls
        positive_calls += 1
        return np.zeros((480, 2), dtype=np.float32)

    def negative_capture(*_args, **_kwargs):
        nonlocal negative_calls
        negative_calls += 1
        result = np.zeros((672, 2), dtype=np.float32)
        result[:, 0] = negative_calls / 100.0
        return result

    output: list[str] = []
    result = run_guided_robustness_collection(
        backend,
        device_index=1,
        dataset_root=tmp_path,
        parameters=parameters,
        input_fn=lambda _prompt: "",
        output_fn=output.append,
        countdown_fn=lambda _output: None,
        positive_capture_fn=positive_capture,
        negative_capture_fn=negative_capture,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
        session_id="guided-robustness-test",
    )

    loaded = load_robustness_dataset(Path(result["session_directory"]))
    assert positive_calls == 6
    assert negative_calls == 7
    assert result["positive_attempts"] == 6
    assert result["negative_segments"] == 7
    assert len(loaded.positive_records) == 6
    assert len(loaded.negative_records) == 7
    assert all(
        record.metadata["detector_acceptance_required_for_storage"] is False
        for record in loaded.positive_records
    )
    assert all(record.capture.dtype == np.float32 for record in loaded.records)
    positive_cue = loaded.positive_records[0].metadata["guided_cue"]
    assert positive_cue["intended_cue_offset_frames"] == 48
    assert positive_cue["intended_event_association"] == {
        "start_frame_index_inclusive": 48,
        "end_frame_index_exclusive": 240,
        "duration_seconds": 0.004,
        "semantics": (
            "Fixed guided-development association interval after the visual "
            "cue; not a measured physical impact timestamp."
        ),
    }
    negative_metadata = loaded.negative_records[0].metadata
    assert negative_metadata["warmup_end_frame_index"] == 96
    assert negative_metadata["activity_start_frame_index"] == 96
    assert negative_metadata["activity_end_frame_index_exclusive"] == 576
    assert negative_metadata["activity_duration_seconds"] == 0.010
    assert negative_metadata["post_activity_start_frame_index"] == 576
    assert negative_metadata["post_activity_end_frame_index_exclusive"] == 672
    assert negative_metadata["post_activity_tail_duration_seconds"] == 0.002
    assert any("Every structurally valid" in line for line in output)


def test_negative_recorder_captures_default_warmup_plus_activity(
    audio_backend_factory,
) -> None:
    recording = np.zeros((540_000, 2), dtype=np.float32)
    backend = audio_backend_factory(recording=recording)
    timeline: list[tuple[str, float | str, bool]] = []

    captured = record_robustness_segment(
        backend,
        device_index=18,
        sample_rate_hz=48_000.0,
        channels=2,
        activity_duration_seconds=10.0,
        warmup_duration_seconds=1.0,
        post_activity_tail_seconds=0.25,
        activity_cue_fn=lambda: timeline.append(
            ("cue", "BEGIN TYPING", backend.recording_active)
        ),
        activity_end_cue_fn=lambda: timeline.append(
            ("cue", "END TYPING — REMAIN QUIET", backend.recording_active)
        ),
        sleep_fn=lambda seconds: timeline.append(
            ("sleep", seconds, backend.recording_active)
        ),
    )

    assert captured.shape == (540_000, 2)
    assert backend.record_calls == [
        {
            "frames": 540_000,
            "samplerate": 48_000.0,
            "channels": 2,
            "dtype": "float32",
            "device": 18,
            "blocking": False,
        }
    ]
    assert backend.wait_calls == 1
    assert timeline == [
        ("sleep", 1.0, True),
        ("cue", "BEGIN TYPING", True),
        ("sleep", 10.0, True),
        ("cue", "END TYPING — REMAIN QUIET", True),
    ]


def test_descriptive_tapness_metrics_are_finite_and_do_not_modify_raw_window() -> None:
    window = np.zeros((9_600, 2), dtype=np.float32)
    window[4_800:5_040] = (0.10, -0.05)
    original = window.tobytes()

    metrics = extract_descriptive_tapness_metrics(
        window,
        sample_rate_hz=48_000.0,
        onset_offset_frames=4_800,
        learned_noise_floor_rms=0.001,
    )

    assert window.tobytes() == original
    assert metrics["impact_window_rms"] > 0.0
    assert metrics["impact_peak_absolute"] == pytest.approx(0.10)
    assert metrics["onset_contrast_rms_ratio"] > 1.0
    assert 0.0 <= metrics["early_energy_fraction"] <= 1.0
    assert metrics["decision_use"] == "descriptive_only_or_frozen_tapness_v1_input"
    json.dumps(metrics, allow_nan=False)


def test_offline_replay_is_deterministic_retains_miss_and_reports_negative_rate(
    tmp_path,
) -> None:
    session = _replay_session(tmp_path)
    files = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in session.directory.rglob("*")
        if path.is_file()
    }

    first = replay_robustness_dataset(
        session.directory,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )
    second = replay_robustness_dataset(
        session.directory,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    assert first == second
    attempts = first["positive"]["attempts"]
    assert attempts[0]["stage1_candidate_started"] is True
    assert attempts[1]["stage1_candidate_started"] is False
    assert attempts[1]["intended_zone"] == "LEFT"
    assert first["positive"]["summary"][
        "stage1_candidate_start_recall"
    ] == pytest.approx(0.5)
    negative = first["negative"]["summary"]
    assert negative["activity_associated_candidate_starts"] == 1
    assert negative["activity_associated_completed_detections"] == 1
    assert negative["candidate_start_rate_per_labeled_minute"] == pytest.approx(6.0)
    assert negative[
        "completed_false_event_rate_per_labeled_minute"
    ] == pytest.approx(6.0)
    assert negative["per_activity"]["typing"][
        "completed_false_event_rate_per_labeled_minute"
    ] == pytest.approx(6.0)
    assert first["collection_completeness"]["collection_complete"] is True
    assert all(
        event["frozen_spatial_prediction"] is None
        for event in first["negative"]["segments"][0]["completed_events"]
    )
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in files.items()
    )
    assert "sounddevice" not in sys.modules


def test_optional_frozen_classifier_is_unchanged_and_only_applies_to_positives(
    tmp_path,
) -> None:
    session = _replay_session(tmp_path)
    baseline_path = Path("baselines/lenovo-left-right-v1.json")
    baseline_before = baseline_path.read_bytes()

    report = replay_robustness_dataset(
        session.directory,
        baseline_path=baseline_path,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    spatial = report["positive"]["summary"]["conditional_frozen_spatial"]
    assert spatial == {
        "applied": True,
        "classified_count": 1,
        "correct_count": 1,
        "accuracy": 1.0,
    }
    prediction = report["positive"]["attempts"][0]["completed_events"][0][
        "frozen_spatial_prediction"
    ]
    assert prediction["predicted_zone"] == "RIGHT"
    assert prediction["threshold_db"] == pytest.approx(0.12793235855251162)
    assert prediction["baseline_refit"] is False
    assert report["frozen_spatial_baseline"]["refit_performed"] is False
    assert baseline_path.read_bytes() == baseline_before


def test_report_is_waveform_free_and_refuses_overwrite(tmp_path) -> None:
    session = _replay_session(tmp_path / "data")
    report = replay_robustness_dataset(
        session.directory,
        now_fn=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc),
    )
    path = tmp_path / "report.json"

    write_robustness_report(report, path)

    text = path.read_text(encoding="utf-8")
    assert '"capture"' not in text
    assert '"candidate_window":' not in text
    with pytest.raises(FileExistsError):
        write_robustness_report(report, path)
