from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import desksense.robustness_external as external
from desksense.robustness import _summarize_negative_replay, _summarize_positive_replay
from desksense.robustness_dataset import (
    NEGATIVE_RECORD_TYPE,
    POSITIVE_RECORD_TYPE,
    ROBUSTNESS_DATASET_KIND,
    ROBUSTNESS_SCHEMA_VERSION,
    LoadedRobustnessDataset,
    RobustnessRecord,
    create_robustness_dataset_session,
)
from desksense.tapness import TapnessBaselineError, create_tapness_baseline


TAPNESS_PATH = Path("baselines/lenovo-tapness-v1.json")
SPATIAL_PATH = Path("baselines/lenovo-left-right-v1.json")


def _artifact_context():
    tapness, spatial, identities = external._load_and_validate_frozen_artifacts(
        TAPNESS_PATH, SPATIAL_PATH
    )
    compatibility = spatial["source_development_dataset"]["compatibility"]
    device = {
        "index": 99,
        "name": compatibility["endpoint_name"],
        "host_api": {"index": 4, "name": compatibility["host_api_name"]},
    }
    return tapness, spatial, identities, device


def _fake_loaded_external(*, positive_count=30, negative_count=7):
    tapness, _spatial, identities, device = _artifact_context()
    positives = external.external_positive_plan()
    negatives = external.external_negative_plan()
    metadata = external._external_session_metadata(
        inventory={"backend": {"name": "fake"}},
        device=device,
        positive_plan=positives,
        negative_plan=negatives,
        artifacts=identities,
        tapness=tapness,
    )
    session = {
        **metadata,
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "dataset_kind": ROBUSTNESS_DATASET_KIND,
        "project_phase": external.EXTERNAL_PROJECT_PHASE,
        "session_id": "session-b-test",
        "created_at_utc": "2026-09-02T00:00:00+00:00",
    }
    endpoint = metadata["selected_endpoint"]
    records = []
    positive_audio = np.broadcast_to(
        np.zeros((1, 2), dtype=np.float32), (96_000, 2)
    )
    for plan in positives[:positive_count]:
        records.append(
            RobustnessRecord(
                metadata={
                    "record_id": f"positive-{plan.collection_order_index}",
                    "record_type": POSITIVE_RECORD_TYPE,
                    "collection_order_index": plan.collection_order_index,
                    "intended_zone": plan.zone,
                    "intended_strength": plan.strength,
                    "attempt_number_within_condition": plan.attempt_number_within_condition,
                    "detector_acceptance_required_for_storage": False,
                    "model_prediction_used_for_retry_or_storage": False,
                    "guided_cue": external._external_guided_cue_metadata(),
                },
                capture=positive_audio,
                path=Path(f"positive-{plan.collection_order_index}.npz"),
            )
        )
    if positive_count == 30:
        for plan in negatives[:negative_count]:
            activity_frames = round(48_000 * plan.activity_duration_seconds)
            tail_end = 48_000 + activity_frames + 12_000
            audio = np.broadcast_to(
                np.zeros((1, 2), dtype=np.float32), (tail_end, 2)
            )
            records.append(
                RobustnessRecord(
                    metadata={
                        "record_id": f"negative-{plan.activity}",
                        "record_type": NEGATIVE_RECORD_TYPE,
                        "collection_order_index": plan.collection_order_index,
                        "activity": plan.activity,
                        "repetition_index": 1,
                        "activity_duration_seconds": plan.activity_duration_seconds,
                        "model_prediction_used_for_retry_or_storage": False,
                        "warmup_end_frame_index": 48_000,
                        "activity_start_frame_index": 48_000,
                        "activity_end_frame_index_exclusive": 48_000 + activity_frames,
                        "post_activity_start_frame_index": 48_000 + activity_frames,
                        "post_activity_end_frame_index_exclusive": tail_end,
                    },
                    capture=audio,
                    path=Path(f"negative-{plan.activity}.npz"),
                )
            )
    return LoadedRobustnessDataset(
        directory=Path("session-b-test"),
        session=session,
        records=tuple(records),
    )


def _attempt(plan, *, stage1=True, stage2=True, correct=True):
    event = None
    if stage1:
        event = {
            "interval_relation": "associated",
            "frozen_tapness_prediction": (
                {"tap_accepted": stage2} if stage2 else {"tap_accepted": False}
            ),
            "frozen_spatial_prediction": (
                {
                    "actual_zone": plan.zone,
                    "predicted_zone": plan.zone if correct else ("RIGHT" if plan.zone == "LEFT" else "LEFT"),
                    "correct": correct,
                }
                if stage2
                else None
            ),
        }
    return {
        "record_id": f"p-{plan.collection_order_index}",
        "intended_zone": plan.zone,
        "intended_strength": plan.strength,
        "stage1_candidate_started": stage1,
        "associated_candidate_start_count": int(stage1),
        "associated_completed_result": stage1,
        "associated_completed_detection_count": int(stage1),
        "associated_completed_rejection_count": 0,
        "extraneous_candidate_start_count": 0,
        "extraneous_completed_event_count": 0,
        "completed_events": [] if event is None else [event],
    }


def _negative_segment(plan, *, candidate_starts=0, false_accepts=0):
    events = [
        {
            "interval_relation": "activity",
            "status": "detected",
            "frozen_tapness_prediction": {"tap_accepted": index < false_accepts},
        }
        for index in range(candidate_starts)
    ]
    return {
        "activity": plan.activity,
        "stored_capture_duration_seconds": 1.25 + plan.activity_duration_seconds,
        "warmup_duration_seconds": 1.0,
        "labeled_activity_duration_seconds": plan.activity_duration_seconds,
        "post_activity_tail_duration_seconds": 0.25,
        "activity_candidate_start_count": candidate_starts,
        "activity_completed_detection_count": candidate_starts,
        "activity_completed_rejection_count": 0,
        "warmup_candidate_start_count": 0,
        "warmup_completed_detection_count": 0,
        "warmup_completed_rejection_count": 0,
        "post_activity_candidate_start_count": 0,
        "post_activity_completed_detection_count": 0,
        "post_activity_completed_rejection_count": 0,
        "completed_events": events,
    }


def _fake_replay(
    *,
    stage1_count=27,
    stage2_count=27,
    end_to_end_count=None,
    false_accepts=1,
):
    if end_to_end_count is None:
        end_to_end_count = stage2_count
    plans = external.external_positive_plan()
    attempts = [
        _attempt(
            plan,
            stage1=index < stage1_count,
            stage2=index < stage2_count,
            correct=index < end_to_end_count,
        )
        for index, plan in enumerate(plans)
    ]
    negative_segments = [
        _negative_segment(
            plan,
            candidate_starts=max(1, false_accepts) if index == 0 else (1 if index < 3 else 0),
            false_accepts=false_accepts if index == 0 else 0,
        )
        for index, plan in enumerate(external.external_negative_plan())
    ]
    return {
        "generated_at_utc": "2026-09-02T12:00:00+00:00",
        "positive": {
            "attempts": attempts,
            "summary": _summarize_positive_replay(attempts, {}, {}),
        },
        "negative": {
            "segments": negative_segments,
            "summary": _summarize_negative_replay(negative_segments, {}),
        },
    }


def test_external_positive_plan_is_exactly_balanced() -> None:
    plan = external.external_positive_plan()
    assert len(plan) == 30
    counts = {
        (zone, strength): sum(
            item.zone == zone and item.strength == strength for item in plan
        )
        for zone in ("LEFT", "RIGHT")
        for strength in ("light", "normal", "firm")
    }
    assert set(counts.values()) == {5}


def test_external_negative_plan_is_exactly_300_seconds() -> None:
    plan = external.external_negative_plan()
    assert len(plan) == 7
    assert sum(item.activity_duration_seconds for item in plan) == 300.0
    assert plan[0].activity == "quiet"
    assert plan[0].activity_duration_seconds == 30.0
    assert all(item.activity_duration_seconds == 45.0 for item in plan[1:])


def test_external_negative_frame_boundaries_are_exact() -> None:
    dataset = _fake_loaded_external()
    completeness = external._validate_external_dataset(dataset)
    assert completeness["external_collection_complete"] is True
    for record, plan in zip(dataset.negative_records, external.external_negative_plan()):
        expected_end = 48_000 + round(48_000 * plan.activity_duration_seconds)
        assert record.metadata["warmup_end_frame_index"] == 48_000
        assert record.metadata["activity_end_frame_index_exclusive"] == expected_end
        assert record.metadata["post_activity_end_frame_index_exclusive"] == expected_end + 12_000


def test_external_metadata_binds_role_protocol_and_frozen_artifacts() -> None:
    dataset = _fake_loaded_external()
    assert dataset.session["evidence_role"] == "external_validation"
    assert dataset.session["no_fitting_allowed"] is True
    assert dataset.session["external_validation_protocol"]["labeled_negative_denominator_seconds"] == 300.0
    assert dataset.session["frozen_pipeline"]["tapness_baseline"]["file_sha256"] == external.EXPECTED_TAPNESS_FILE_SHA256
    assert dataset.session["frozen_pipeline"]["spatial_baseline"]["file_sha256"] == external.EXPECTED_SPATIAL_FILE_SHA256
    assert dataset.session["project_phase"] == external.EXTERNAL_PROJECT_PHASE
    assert dataset.session["source_mode"] == external.EXTERNAL_SOURCE_MODE
    assert dataset.session["physical_positive_interaction_protocol"] == (
        external.external_positive_interaction_protocol()
    )


def test_physical_protocol_freezes_hand_contact_geometry_and_abort_policy() -> None:
    protocol = external.external_positive_interaction_protocol()
    assert protocol["hand_and_finger"] == {
        "same_for_both_zones": True,
        "required_hand": "RIGHT",
        "required_finger": "RIGHT index finger",
    }
    assert protocol["contact_method"]["required"] == "fleshy fingertip pad"
    assert protocol["intended_taps_per_cue"] == 1
    assert protocol["zone_geometry"]["LEFT"]["distance_outside_corresponding_laptop_edge_centimeters"] == [7, 10]
    assert protocol["zone_geometry"]["RIGHT"]["orientation"] == "toward user/touchpad side, away from screen"
    policy = protocol["procedural_error_policy"]
    assert policy["partial_session_must_not_be_evaluated"] is True
    assert policy["model_output_must_not_influence_abort_retry_or_deletion"] is True
    assert policy["per_attempt_retry_or_deletion_supported"] is False


def test_guided_external_collection_uses_fixed_protocol_without_predictions(
    monkeypatch, tmp_path
) -> None:
    tapness, spatial, identities, device = _artifact_context()
    saved = []
    positive_calls = []
    negative_calls = []
    output = []

    class FakeSession:
        session_id = "session-b-collection"
        directory = tmp_path / "session-b-collection"
        manifest_path = directory / "manifest.jsonl"

        def save_record(self, **kwargs):
            saved.append(
                {
                    "record_type": kwargs["record_type"],
                    "shape": kwargs["capture"].shape,
                    "metadata": kwargs["metadata"],
                }
            )

    monkeypatch.setattr(
        external,
        "_load_and_validate_frozen_artifacts",
        lambda *_args: (tapness, spatial, identities),
    )
    monkeypatch.setattr(
        external,
        "_selected_robustness_device",
        lambda *_args: ({"backend": {"name": "fake"}}, device),
    )
    monkeypatch.setattr(external, "_validate_fixed_capture_domain", lambda *_args: None)
    monkeypatch.setattr(
        external,
        "create_robustness_dataset_session",
        lambda *_args, **_kwargs: FakeSession(),
    )

    def positive_capture(_backend, **kwargs):
        positive_calls.append(kwargs)
        return np.zeros((96_000, 2), dtype=np.float32)

    def negative_capture(_backend, **kwargs):
        negative_calls.append(kwargs)
        seconds = (
            kwargs["warmup_duration_seconds"]
            + kwargs["activity_duration_seconds"]
            + kwargs["post_activity_tail_seconds"]
        )
        kwargs["activity_cue_fn"]()
        kwargs["activity_end_cue_fn"]()
        return np.zeros((round(48_000 * seconds), 2), dtype=np.float32)

    result = external.run_guided_external_robustness_collection(
        object(),
        device_index=18,
        tapness_baseline_path=TAPNESS_PATH,
        spatial_baseline_path=SPATIAL_PATH,
        dataset_root=tmp_path,
        input_fn=lambda _prompt: "",
        output_fn=output.append,
        countdown_fn=lambda _write: None,
        positive_capture_fn=positive_capture,
        negative_capture_fn=negative_capture,
        sleep_fn=lambda _seconds: None,
        now_fn=lambda: datetime(2026, 9, 2, tzinfo=timezone.utc),
    )
    assert result["positive_attempts"] == 30
    assert result["negative_segments"] == 7
    assert len(positive_calls) == 30
    assert len(negative_calls) == 7
    assert len(saved) == 37
    assert all(item["shape"] == (96_000, 2) for item in saved[:30])
    assert [call["activity_duration_seconds"] for call in negative_calls] == [
        30.0,
        45.0,
        45.0,
        45.0,
        45.0,
        45.0,
        45.0,
    ]
    assert all(
        item["metadata"]["model_prediction_used_for_retry_or_storage"] is False
        for item in saved
    )
    assert not any("predicted" in line.casefold() for line in output)
    assert any("same RIGHT index finger" in line for line in output)
    assert any("exactly one intended tap" in line for line in output)
    assert any("fleshy fingertip pad" in line for line in output)
    assert any("press Ctrl+C immediately" in line for line in output)
    assert sum("LIGHT definition:" in line for line in output) == 10
    assert sum("NORMAL definition:" in line for line in output) == 10
    assert sum("FIRM definition:" in line for line in output) == 10
    assert sum(line.startswith("BEGIN ") for line in output) == 7
    assert sum(line.startswith("END ") for line in output) == 7


def test_partial_external_session_is_readable_but_incomplete(tmp_path) -> None:
    dataset = _fake_loaded_external(positive_count=0, negative_count=0)
    session = create_robustness_dataset_session(
        tmp_path,
        {
            key: value
            for key, value in dataset.session.items()
            if key not in {"schema_version", "dataset_kind", "session_id", "created_at_utc"}
        },
        session_id="partial-session-b",
        created_at_utc="2026-09-02T00:00:00+00:00",
    )
    _loaded, completeness = external.load_external_robustness_dataset(session.directory)
    assert completeness["external_collection_complete"] is False
    with pytest.raises(external.ExternalRobustnessError, match="incomplete"):
        external.load_external_robustness_dataset(
            session.directory, require_complete=True
        )


def test_external_loader_rejects_wrong_role() -> None:
    dataset = _fake_loaded_external()
    dataset.session["evidence_role"] = "development"
    with pytest.raises(external.ExternalRobustnessError, match="evidence_role"):
        external._validate_external_dataset(dataset)


def test_external_loader_rejects_wrong_denominator_and_design() -> None:
    dataset = _fake_loaded_external()
    dataset.session["external_validation_protocol"]["labeled_negative_denominator_seconds"] = 299.0
    with pytest.raises(external.ExternalRobustnessError, match="protocol metadata"):
        external._validate_external_dataset(dataset)


@pytest.mark.parametrize(
    ("field", "wrong_value", "match"),
    [
        ("project_phase", "3B.0", "project_phase"),
        ("source_mode", "synthetic_test", "source_mode"),
    ],
)
def test_external_loader_rejects_wrong_phase_and_source_mode(
    field, wrong_value, match
) -> None:
    dataset = _fake_loaded_external()
    dataset.session[field] = wrong_value
    with pytest.raises(external.ExternalRobustnessError, match=match):
        external._validate_external_dataset(dataset)


def test_external_loader_requires_exact_physical_protocol() -> None:
    dataset = _fake_loaded_external()
    dataset.session["physical_positive_interaction_protocol"]["hand_and_finger"][
        "required_finger"
    ] = "LEFT index finger"
    with pytest.raises(external.ExternalRobustnessError, match="physical positive"):
        external._validate_external_dataset(dataset)


def test_external_loader_requires_record_prediction_independence() -> None:
    dataset = _fake_loaded_external()
    dataset.positive_records[0].metadata[
        "model_prediction_used_for_retry_or_storage"
    ] = True
    with pytest.raises(external.ExternalRobustnessError, match="positive record"):
        external._validate_external_dataset(dataset)
    dataset = _fake_loaded_external()
    dataset.negative_records[0].metadata.pop(
        "model_prediction_used_for_retry_or_storage"
    )
    with pytest.raises(external.ExternalRobustnessError, match="negative record"):
        external._validate_external_dataset(dataset)


def test_exact_observed_plans_drive_completeness() -> None:
    dataset = _fake_loaded_external()
    records = list(dataset.records)
    records[0], records[1] = records[1], records[0]
    reordered = LoadedRobustnessDataset(
        dataset.directory, dataset.session, tuple(records)
    )
    with pytest.raises(external.ExternalRobustnessError, match="planned prefix"):
        external._validate_external_dataset(reordered)
    dataset = _fake_loaded_external()
    dataset.session["positive_design"]["plan"][0]["zone"] = "RIGHT"
    with pytest.raises(external.ExternalRobustnessError, match="positive design"):
        external._validate_external_dataset(dataset)


def test_tapness_fitting_refuses_external_role_before_replay(tmp_path, monkeypatch) -> None:
    dataset = _fake_loaded_external(positive_count=0, negative_count=0)
    session = create_robustness_dataset_session(
        tmp_path,
        {
            key: value
            for key, value in dataset.session.items()
            if key not in {"schema_version", "dataset_kind", "session_id", "created_at_utc"}
        },
        session_id="external-no-fit",
        created_at_utc="2026-09-02T00:00:00+00:00",
    )
    monkeypatch.setattr(
        "desksense.robustness.replay_robustness_dataset",
        lambda *_args, **_kwargs: pytest.fail("external data reached replay/fitting"),
    )
    with pytest.raises(TapnessBaselineError, match="Refusing to fit"):
        create_tapness_baseline(session.directory)


def test_external_evaluation_never_fits_and_preserves_files(tmp_path, monkeypatch) -> None:
    dataset = _fake_loaded_external()
    session_dir = tmp_path / "session-b"
    session_dir.mkdir()
    marker = session_dir / "marker.bin"
    marker.write_bytes(b"external dataset bytes")
    dataset = LoadedRobustnessDataset(session_dir, dataset.session, dataset.records)
    before = hashlib.sha256(marker.read_bytes()).hexdigest()
    tap_before = TAPNESS_PATH.read_bytes()
    spatial_before = SPATIAL_PATH.read_bytes()
    monkeypatch.setattr(
        external,
        "load_external_robustness_dataset",
        lambda *_args, **_kwargs: (
            dataset,
            {
                "external_collection_complete": True,
                "expected_positive_attempts": 30,
                "observed_positive_attempts": 30,
                "expected_negative_segments": 7,
                "observed_negative_segments": 7,
                "expected_labeled_negative_seconds": 300.0,
                "positive_plan_complete": True,
                "negative_plan_complete": True,
            },
        ),
    )
    monkeypatch.setattr(external, "replay_robustness_dataset", lambda *_a, **_k: _fake_replay())
    monkeypatch.setattr(
        external,
        "robustness_dataset_fingerprint",
        lambda _path: {"algorithm": "SHA-256", "sha256": "a" * 64},
    )
    report = external.evaluate_robustness_external(
        session_dir,
        tapness_baseline_path=TAPNESS_PATH,
        spatial_baseline_path=SPATIAL_PATH,
    )
    assert report["no_refit_declaration"]["external_samples_used_for_fitting"] is False
    assert report["dataset_integrity"]["fingerprint_before_replay"] == report[
        "dataset_integrity"
    ]["fingerprint_after_replay"]
    assert report["dataset_integrity"]["dataset_files_modified"] is False
    assert report["artifact_integrity"]["verified_unchanged"] is True
    assert hashlib.sha256(marker.read_bytes()).hexdigest() == before
    assert TAPNESS_PATH.read_bytes() == tap_before
    assert SPATIAL_PATH.read_bytes() == spatial_before


def test_external_metrics_preserve_stage_denominators_and_gates(monkeypatch) -> None:
    dataset = _fake_loaded_external()
    monkeypatch.setattr(
        external,
        "load_external_robustness_dataset",
        lambda *_args, **_kwargs: (
            dataset,
            {"external_collection_complete": True},
        ),
    )
    monkeypatch.setattr(external, "replay_robustness_dataset", lambda *_a, **_k: _fake_replay())
    monkeypatch.setattr(
        external,
        "robustness_dataset_fingerprint",
        lambda _path: {"algorithm": "SHA-256", "sha256": "a" * 64},
    )
    report = external.evaluate_robustness_external(
        Path("unused"),
        tapness_baseline_path=TAPNESS_PATH,
        spatial_baseline_path=SPATIAL_PATH,
    )
    assert report["positive_metrics"]["stage1"]["attempts_with_candidate_start"] == 27
    assert report["positive_metrics"]["stage2"]["attempts_accepted_as_tap"] == 27
    assert report["positive_metrics"]["stage3"]["classified_count"] == 27
    assert report["positive_metrics"]["end_to_end"]["total_intended_attempts"] == 30
    assert report["negative_metrics"]["labeled_activity_seconds"] == 300.0
    assert report["negative_metrics"]["stage2_false_accepts"] == 1
    assert report["engineering_gates"]["engineering_gate_pass"] is True
    context = report["collection_context"]
    assert context["selected_endpoint"]["host_api"]["name"] == "Windows WDM-KS"
    assert context["capture_domain"] == {
        "sample_rate_hz": 48_000.0,
        "channel_count": 2,
        "dtype": "float32",
    }
    assert context["physical_positive_interaction_protocol"]["hand_and_finger"][
        "required_finger"
    ] == "RIGHT index finger"
    assert "seven separately recorded" in context["negative_denominator"][
        "semantics"
    ]
    assert "not one continuous" in context["negative_denominator"]["semantics"]


@pytest.mark.parametrize(
    ("stage1", "stage2", "false_accepts"),
    [(26, 27, 1), (27, 26, 1), (27, 27, 2)],
)
def test_each_predeclared_required_gate_can_fail(
    stage1, stage2, false_accepts
) -> None:
    replay = _fake_replay(
        stage1_count=stage1,
        stage2_count=stage2,
        false_accepts=false_accepts,
    )
    positive = external._external_positive_metrics(replay)
    negative = external._external_negative_metrics(replay)
    assert external._engineering_gates(positive, negative)["engineering_gate_pass"] is False


def test_cell_preference_is_reported_but_not_part_of_required_gate() -> None:
    replay = _fake_replay(stage1_count=27, stage2_count=27, false_accepts=1)
    positive = external._external_positive_metrics(replay)
    negative = external._external_negative_metrics(replay)
    gates = external._engineering_gates(positive, negative)
    assert gates["engineering_gate_pass"] is True
    assert gates["cell_preference"]["included_in_required_overall_gate"] is False


def test_end_to_end_26_of_30_fails_complete_pipeline_gate() -> None:
    replay = _fake_replay(
        stage1_count=27,
        stage2_count=27,
        end_to_end_count=26,
        false_accepts=1,
    )
    positive = external._external_positive_metrics(replay)
    negative = external._external_negative_metrics(replay)
    gates = external._engineering_gates(positive, negative)
    assert positive["stage3"]["correct_count"] == 26
    assert gates["predeclared_required_gates"][
        "end_to_end_correct_zone_outputs_at_least_27_of_30"
    ] is False
    assert gates["engineering_gate_pass"] is False


def test_end_to_end_27_of_30_passes_when_other_hard_gates_pass() -> None:
    replay = _fake_replay(
        stage1_count=27,
        stage2_count=27,
        end_to_end_count=27,
        false_accepts=1,
    )
    positive = external._external_positive_metrics(replay)
    negative = external._external_negative_metrics(replay)
    gates = external._engineering_gates(positive, negative)
    assert gates["predeclared_required_gates"][
        "end_to_end_correct_zone_outputs_at_least_27_of_30"
    ] is True
    assert gates["engineering_gate_pass"] is True


def test_stage2_rejection_blocks_spatial_and_stage1_miss_stays_failure() -> None:
    attempts = [
        _attempt(external.external_positive_plan()[0], stage1=False, stage2=False),
        _attempt(external.external_positive_plan()[1], stage1=True, stage2=False),
    ]
    summary = _summarize_positive_replay(attempts, {}, {})
    assert summary["stage2_tapness"]["attempts_with_accepted_tap"] == 0
    assert summary["conditional_frozen_spatial"]["classified_count"] == 0
    assert summary["end_to_end"]["total_intended_taps"] == 2
    assert summary["end_to_end"]["intended_taps_accepted_and_spatially_correct"] == 0


def test_negative_denominator_and_activity_rates_exclude_context() -> None:
    replay = _fake_replay(false_accepts=1)
    metrics = external._external_negative_metrics(replay)
    assert metrics["labeled_activity_seconds"] == 300.0
    assert metrics["stage1_candidates_per_minute"] == pytest.approx(0.6)
    assert metrics["stage2_false_accepts_per_minute"] == pytest.approx(0.2)
    assert metrics["excluded_context"]["warmup_seconds"] == 7.0
    assert metrics["excluded_context"]["post_activity_tail_seconds"] == 1.75
    assert "stage3_predictions_for_negatives" not in metrics
    assert metrics["stage3_negative_evaluation"]["status"] == "not_applicable"
    assert metrics["stage3_negative_evaluation"]["left_right_accuracy_reported"] is False


def test_wrong_artifact_and_wrong_stage1_policy_fail(monkeypatch) -> None:
    dataset = _fake_loaded_external()
    tapness, _spatial, identities, _device = _artifact_context()
    wrong = json.loads(json.dumps(tapness))
    wrong["stage1_policy"]["strong_impact_recovery"]["rms_ratio_minimum_inclusive"] = 7.0
    with pytest.raises(external.ExternalRobustnessError, match="Stage 1 policy"):
        external._validate_dataset_artifact_binding(dataset, identities, wrong)
    dataset.session["frozen_pipeline"]["tapness_baseline"]["file_sha256"] = "0" * 64
    with pytest.raises(external.ExternalRobustnessError, match="Tapness artifact differs"):
        external._validate_dataset_artifact_binding(dataset, identities, tapness)


def test_artifact_identity_is_path_independent_but_content_strict(tmp_path) -> None:
    tap_copy = tmp_path / "tapness.json"
    spatial_copy = tmp_path / "spatial.json"
    tap_copy.write_bytes(TAPNESS_PATH.read_bytes())
    spatial_copy.write_bytes(SPATIAL_PATH.read_bytes())
    _tapness, _spatial, identities = external._load_and_validate_frozen_artifacts(
        tap_copy, spatial_copy
    )
    assert identities["tapness_baseline"]["file_sha256"] == external.EXPECTED_TAPNESS_FILE_SHA256
    modified = json.loads(tap_copy.read_text(encoding="utf-8"))
    modified["model"]["decision_threshold"] += 0.01
    tap_copy.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(external.ExternalRobustnessError):
        external._load_and_validate_frozen_artifacts(tap_copy, spatial_copy)


def test_real_temporary_dataset_fingerprint_mutation_is_detected(tmp_path) -> None:
    dataset = _fake_loaded_external(positive_count=0, negative_count=0)
    session = create_robustness_dataset_session(
        tmp_path,
        {
            key: value
            for key, value in dataset.session.items()
            if key not in {"schema_version", "dataset_kind", "session_id", "created_at_utc"}
        },
        session_id="integrity-session-b",
        created_at_utc="2026-09-02T00:00:00+00:00",
    )
    tap_copy = tmp_path / "tapness.json"
    spatial_copy = tmp_path / "spatial.json"
    tap_copy.write_bytes(TAPNESS_PATH.read_bytes())
    spatial_copy.write_bytes(SPATIAL_PATH.read_bytes())
    before = external._external_integrity_snapshot(
        session.directory, tap_copy, spatial_copy
    )
    session_path = session.directory / "session.json"
    session_json = json.loads(session_path.read_text(encoding="utf-8"))
    session_json["audit_marker"] = "changed"
    session_path.write_text(json.dumps(session_json), encoding="utf-8")
    after = external._external_integrity_snapshot(
        session.directory, tap_copy, spatial_copy
    )
    with pytest.raises(external.ExternalRobustnessError, match="dataset bytes changed"):
        external._require_external_integrity_unchanged(before, after)


def test_real_temporary_artifact_hash_mutation_is_detected(tmp_path) -> None:
    dataset = _fake_loaded_external(positive_count=0, negative_count=0)
    session = create_robustness_dataset_session(
        tmp_path,
        {
            key: value
            for key, value in dataset.session.items()
            if key not in {"schema_version", "dataset_kind", "session_id", "created_at_utc"}
        },
        session_id="artifact-integrity-session-b",
        created_at_utc="2026-09-02T00:00:00+00:00",
    )
    tap_copy = tmp_path / "tapness-copy.json"
    spatial_copy = tmp_path / "spatial-copy.json"
    tap_copy.write_bytes(TAPNESS_PATH.read_bytes())
    spatial_copy.write_bytes(SPATIAL_PATH.read_bytes())
    before = external._external_integrity_snapshot(
        session.directory, tap_copy, spatial_copy
    )
    tap_copy.write_bytes(tap_copy.read_bytes() + b"\n")
    after = external._external_integrity_snapshot(
        session.directory, tap_copy, spatial_copy
    )
    with pytest.raises(external.ExternalRobustnessError, match="tapness artifact bytes changed"):
        external._require_external_integrity_unchanged(before, after)


def test_evaluation_raises_if_dataset_changes_during_replay(monkeypatch) -> None:
    dataset = _fake_loaded_external()
    fingerprints = iter(
        [
            {"algorithm": "SHA-256", "sha256": "a" * 64},
            {"algorithm": "SHA-256", "sha256": "b" * 64},
        ]
    )
    monkeypatch.setattr(
        external,
        "load_external_robustness_dataset",
        lambda *_args, **_kwargs: (
            dataset,
            {"external_collection_complete": True},
        ),
    )
    monkeypatch.setattr(
        external, "replay_robustness_dataset", lambda *_a, **_k: _fake_replay()
    )
    monkeypatch.setattr(
        external,
        "robustness_dataset_fingerprint",
        lambda _path: next(fingerprints),
    )
    with pytest.raises(external.ExternalRobustnessError, match="dataset bytes changed"):
        external.evaluate_robustness_external(
            Path("unused"),
            tapness_baseline_path=TAPNESS_PATH,
            spatial_baseline_path=SPATIAL_PATH,
        )


def test_external_report_is_waveform_free_and_exclusive(tmp_path) -> None:
    report = {
        "report_type": external.EXTERNAL_REPORT_TYPE,
        "value": 1.0,
        "waveforms_embedded_in_report": False,
    }
    path = tmp_path / "external.json"
    external.write_external_robustness_report(report, path)
    assert "capture" not in path.read_text(encoding="utf-8")
    with pytest.raises(external.ExternalRobustnessError, match="overwrite"):
        external.write_external_robustness_report(report, path)
    with pytest.raises(external.ExternalRobustnessError, match="waveform"):
        external.write_external_robustness_report(
            {
                "report_type": external.EXTERNAL_REPORT_TYPE,
                "capture": [[0.0, 0.0]],
            },
            tmp_path / "forbidden.json",
        )


def test_offline_external_module_does_not_import_sounddevice() -> None:
    assert "sounddevice" not in sys.modules
