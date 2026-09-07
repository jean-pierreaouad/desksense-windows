from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import desksense.robustness_replication as replication
from desksense.robustness_dataset import (
    NEGATIVE_RECORD_TYPE,
    POSITIVE_RECORD_TYPE,
    ROBUSTNESS_DATASET_KIND,
    ROBUSTNESS_SCHEMA_VERSION,
    LoadedRobustnessDataset,
    RobustnessRecord,
)
from desksense.tapness_v2_provisional import (
    classify_with_provisional_tapness_v2,
    load_provisional_tapness_v2_artifact,
    validate_provisional_tapness_v2_artifact,
)


PROVISIONAL_PATH = Path(
    "research_baselines/lenovo-tapness-v2-provisional-ab.json"
)
V1_PATH = Path("baselines/lenovo-tapness-v1.json")


class FakeBackend:
    __version__ = "0.test"

    def __init__(self, artifact: dict) -> None:
        compatibility = artifact["compatibility"]
        self.devices = [
            {
                "name": compatibility["endpoint_name"],
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000.0,
            }
        ]
        self.host_apis = [
            {"name": compatibility["host_api_name"], "default_input_device": 0}
        ]
        self.default = SimpleNamespace(device=(0, -1))
        self.settings_calls: list[dict] = []

    def get_portaudio_version(self):
        return 1, "fake"

    def query_hostapis(self):
        return [dict(value) for value in self.host_apis]

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            return [{**self.devices[0], "index": 0}]
        return {**self.devices[int(device or 0)], "index": int(device or 0)}

    def check_input_settings(self, **kwargs):
        self.settings_calls.append(dict(kwargs))


def _artifact() -> dict:
    return load_provisional_tapness_v2_artifact(PROVISIONAL_PATH)


def _fake_loaded_r1(*, positive_count: int = 30, negative_count: int = 7):
    artifact = _artifact()
    device = {
        "index": 18,
        "name": artifact["compatibility"]["endpoint_name"],
        "host_api": {
            "index": 7,
            "name": artifact["compatibility"]["host_api_name"],
        },
    }
    metadata = replication._r1_session_metadata(
        inventory={"backend": {"name": "fake"}},
        device=device,
        positive_plan=replication.r1_positive_plan(),
        negative_plan=replication.r1_negative_plan(),
        provisional=artifact,
        provisional_identity=replication._provisional_identity(
            PROVISIONAL_PATH, artifact
        ),
    )
    session = {
        **metadata,
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "dataset_kind": ROBUSTNESS_DATASET_KIND,
        "session_id": "r1-test",
        "created_at_utc": "2026-09-05T00:00:00Z",
    }
    records: list[RobustnessRecord] = []
    for plan in replication.r1_positive_plan()[:positive_count]:
        records.append(
            RobustnessRecord(
                metadata={
                    "record_id": f"r1-test-p-{plan.collection_order_index}",
                    "record_type": POSITIVE_RECORD_TYPE,
                    "collection_order_index": plan.collection_order_index,
                    "intended_zone": plan.zone,
                    "intended_strength": plan.strength,
                    "attempt_number_within_condition": plan.attempt_number_within_condition,
                    "detector_acceptance_required_for_storage": False,
                    "model_prediction_used_for_retry_or_storage": False,
                    "provisional_v2_prediction_performed_during_collection": False,
                    "guided_cue": replication._r1_guided_cue_metadata(),
                },
                capture=np.broadcast_to(
                    np.zeros((1, 2), dtype=np.float32), (96_000, 2)
                ),
                path=Path("unused.npz"),
            )
        )
    if positive_count == 30:
        for plan in replication.r1_negative_plan()[:negative_count]:
            activity_end = 48_000 + round(
                48_000 * plan.activity_duration_seconds
            )
            tail_end = activity_end + 12_000
            records.append(
                RobustnessRecord(
                    metadata={
                        "record_id": f"r1-test-n-{plan.activity}",
                        "record_type": NEGATIVE_RECORD_TYPE,
                        "collection_order_index": plan.collection_order_index,
                        "activity": plan.activity,
                        "repetition_index": 1,
                        "activity_duration_seconds": plan.activity_duration_seconds,
                        "model_prediction_used_for_retry_or_storage": False,
                        "provisional_v2_prediction_performed_during_collection": False,
                        "warmup_end_frame_index": 48_000,
                        "activity_start_frame_index": 48_000,
                        "activity_end_frame_index_exclusive": activity_end,
                        "post_activity_start_frame_index": activity_end,
                        "post_activity_end_frame_index_exclusive": tail_end,
                    },
                    capture=np.broadcast_to(
                        np.zeros((1, 2), dtype=np.float32), (tail_end, 2)
                    ),
                    path=Path("unused.npz"),
                )
            )
    return LoadedRobustnessDataset(
        directory=Path("r1-test"), session=session, records=tuple(records)
    )


def _zero_stage1_replay(dataset: LoadedRobustnessDataset) -> dict:
    return {
        "positive": {
            "attempts": [
                {
                    "record_id": record.metadata["record_id"],
                    "stage1_candidate_started": False,
                    "associated_candidate_start_count": 0,
                    "associated_completed_detection_count": 0,
                    "associated_completed_rejection_count": 0,
                    "candidate_starts": [],
                    "completed_events": [],
                }
                for record in dataset.positive_records
            ],
            "summary": {"stage2_tapness": {}},
        },
        "negative": {
            "segments": [
                {
                    "record_id": record.metadata["record_id"],
                    "activity": record.metadata["activity"],
                    "activity_candidate_start_count": 0,
                    "activity_completed_detection_count": 0,
                    "activity_completed_rejection_count": 0,
                    "candidate_starts": [],
                    "completed_events": [],
                }
                for record in dataset.negative_records
            ],
            "summary": {"stage2_tapness": {}, "per_activity": {}},
        },
    }


def _matching_identity_replays() -> tuple[dict, dict]:
    candidate_replay = {
        "candidate_memberships": [
            {
                "source_record_id": "positive-001",
                "onset_frame_index": 50_000,
                "target": 1,
            },
            {
                "source_record_id": "negative-typing-001",
                "onset_frame_index": 60_000,
                "target": 0,
            },
        ],
        "candidate_population": {
            "positive_candidate_count": 1,
            "negative_candidate_count": 1,
            "associated_candidate_start_count": 2,
            "positive_associated_candidate_start_identities": [
                {
                    "source_record_id": "positive-001",
                    "onset_frame_index": 50_000,
                    "candidate_start_route": "ordinary",
                }
            ],
            "negative_activity_candidate_start_identities": [
                {
                    "source_record_id": "negative-typing-001",
                    "onset_frame_index": 60_000,
                    "candidate_start_route": "strong_impact_recovery",
                }
            ],
        },
    }
    stage1_replay = {
        "positive": {
            "attempts": [
                {
                    "record_id": "positive-001",
                    "associated_completed_detection_count": 1,
                    "associated_candidate_start_count": 1,
                    "candidate_starts": [
                        {
                            "onset_frame_index": 50_000,
                            "candidate_start_route": "ordinary",
                            "interval_relation": "associated",
                        }
                    ],
                    "completed_events": [
                        {
                            "onset_frame_index": 50_000,
                            "interval_relation": "associated",
                            "status": "detected",
                        }
                    ],
                }
            ]
        },
        "negative": {
            "segments": [
                {
                    "record_id": "negative-typing-001",
                    "activity_completed_detection_count": 1,
                    "activity_candidate_start_count": 1,
                    "candidate_starts": [
                        {
                            "onset_frame_index": 60_000,
                            "candidate_start_route": "strong_impact_recovery",
                            "interval_relation": "activity",
                        }
                    ],
                    "completed_events": [
                        {
                            "onset_frame_index": 60_000,
                            "interval_relation": "activity",
                            "status": "detected",
                        }
                    ],
                }
            ]
        },
    }
    return candidate_replay, stage1_replay


def test_r1_positive_plan_is_exactly_30_and_balanced() -> None:
    plan = replication.r1_positive_plan()
    assert len(plan) == 30
    counts = Counter((item.zone, item.strength) for item in plan)
    assert set(counts) == {
        (zone, strength)
        for zone in ("LEFT", "RIGHT")
        for strength in ("light", "normal", "firm")
    }
    assert set(counts.values()) == {5}


def test_r1_negative_plan_is_exactly_session_b_300_seconds() -> None:
    plan = replication.r1_negative_plan()
    assert len(plan) == 7
    assert sum(item.activity_duration_seconds for item in plan) == 300.0
    assert plan[0].activity == "quiet"
    assert plan[0].activity_duration_seconds == 30.0
    assert all(item.activity_duration_seconds == 45.0 for item in plan[1:])


def test_r1_metadata_binds_role_artifact_protocol_and_prediction_independence() -> None:
    dataset = _fake_loaded_r1()
    completeness = replication._validate_r1_dataset(dataset)
    assert completeness["development_replication_complete"] is True
    assert dataset.session["evidence_role"] == "development_replication"
    assert dataset.session["not_external_validation"] is True
    assert dataset.session["not_session_c"] is True
    pipeline = dataset.session["provisional_v2_pipeline"]
    assert pipeline["artifact_loaded_for_identity_only_during_collection"] is True
    assert pipeline["predictions_performed_during_collection"] is False
    assert all(
        record.metadata["model_prediction_used_for_retry_or_storage"] is False
        for record in dataset.records
    )


def test_r1_physical_protocol_exactly_reuses_session_b_protocol() -> None:
    protocol = _fake_loaded_r1().session["physical_positive_interaction_protocol"]
    assert protocol["hand_and_finger"] == {
        "same_for_both_zones": True,
        "required_hand": "RIGHT",
        "required_finger": "RIGHT index finger",
    }
    assert protocol["contact_method"] == {
        "required": "fleshy fingertip pad",
        "forbidden": ["fingernail", "knuckle"],
    }
    assert protocol["intended_taps_per_cue"] == 1
    assert protocol["zone_geometry"]["LEFT"][
        "distance_outside_corresponding_laptop_edge_centimeters"
    ] == [7, 10]
    assert protocol["procedural_error_policy"]["abort_action"] == (
        "Press Ctrl+C immediately."
    )


def test_partial_r1_is_readable_but_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _fake_loaded_r1(positive_count=12, negative_count=0)
    monkeypatch.setattr(replication, "load_robustness_dataset", lambda path: dataset)
    _, completeness = replication.load_v2_development_replication_dataset(
        Path("ignored"), require_complete=False
    )
    assert completeness["development_replication_complete"] is False
    with pytest.raises(replication.DevelopmentReplicationError, match="incomplete"):
        replication.load_v2_development_replication_dataset(
            Path("ignored"), require_complete=True
        )


def test_wrong_record_plan_or_prediction_dependence_is_rejected() -> None:
    dataset = _fake_loaded_r1()
    dataset.records[0].metadata["model_prediction_used_for_retry_or_storage"] = True
    with pytest.raises(replication.DevelopmentReplicationError, match="prediction"):
        replication._validate_r1_dataset(dataset)

    dataset = _fake_loaded_r1()
    dataset.records[0].metadata["intended_strength"] = "firm"
    with pytest.raises(replication.DevelopmentReplicationError, match="planned"):
        replication._validate_r1_dataset(dataset)


def test_r1_collection_is_prediction_independent_and_uses_exact_capture_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact()
    backend = FakeBackend(artifact)
    calls = {"positive": 0, "negative": 0}

    def positive_capture(*args, **kwargs):
        calls["positive"] += 1
        return np.zeros((96_000, 2), dtype=np.float32)

    def negative_capture(*args, activity_duration_seconds, **kwargs):
        calls["negative"] += 1
        frames = round((1.25 + activity_duration_seconds) * 48_000)
        return np.zeros((frames, 2), dtype=np.float32)

    monkeypatch.setattr(
        replication,
        "classify_with_provisional_tapness_v2",
        lambda *args, **kwargs: pytest.fail("collection must not predict"),
    )
    result = replication.run_guided_v2_development_replication_collection(
        backend,
        device_index=0,
        provisional_v2_artifact_path=PROVISIONAL_PATH,
        dataset_root=tmp_path,
        input_fn=lambda prompt: "",
        output_fn=lambda text: None,
        countdown_fn=lambda output: None,
        positive_capture_fn=positive_capture,
        negative_capture_fn=negative_capture,
        now_fn=lambda: datetime(2026, 9, 5, tzinfo=timezone.utc),
        session_id="r1-collection-test",
    )
    assert calls == {"positive": 30, "negative": 7}
    assert result["predictions_performed_during_collection"] is False
    dataset, completeness = replication.load_v2_development_replication_dataset(
        Path(result["session_directory"]), require_complete=True
    )
    assert completeness["development_replication_complete"] is True
    assert {
        "device": 0,
        "channels": 2,
        "dtype": "float32",
        "samplerate": 48_000.0,
    } in backend.settings_calls

    dataset_path = Path(result["session_directory"])
    dataset_before = replication._dataset_fingerprint(dataset_path)
    artifact_before = hashlib.sha256(PROVISIONAL_PATH.read_bytes()).hexdigest()
    monkeypatch.setattr(
        replication,
        "replay_tapness_v2_research_candidates",
        lambda *args, **kwargs: {
            "candidate_memberships": [],
                "candidate_population": {
                    "positive_candidate_count": 0,
                    "negative_candidate_count": 0,
                    "associated_candidate_start_count": 0,
                    "positive_associated_candidate_start_identities": [],
                    "negative_activity_candidate_start_identities": [],
                },
        },
    )
    monkeypatch.setattr(
        replication,
        "replay_robustness_dataset",
        lambda *args, **kwargs: _zero_stage1_replay(dataset),
    )
    report = replication.evaluate_v2_development_replication(
        dataset_path,
        provisional_v2_artifact_path=PROVISIONAL_PATH,
        now_fn=lambda: datetime(2026, 9, 5, tzinfo=timezone.utc),
    )
    assert report["model_fitting_performed"] is False
    assert report["positive_metrics"]["intended_attempts"] == 30
    assert report["negative_metrics"]["labeled_negative_seconds"] == 300.0
    assert report["stage3_evaluation"]["status"] == "not_performed"
    assert report["integrity"]["verified_unchanged"] is True
    assert replication._dataset_fingerprint(dataset_path) == dataset_before
    assert hashlib.sha256(PROVISIONAL_PATH.read_bytes()).hexdigest() == artifact_before


def _criterion_inputs(*, stage1=27, accepted=29, false_accepts=2):
    positive = {
        "stage1_candidate_attempts": stage1,
        "provisional_v2_accepted_attempts": accepted,
        "by_zone_and_strength": {
            f"{zone}-{strength}": {"provisional_v2_accepted_attempts": 5}
            for zone in ("LEFT", "RIGHT")
            for strength in ("light", "normal", "firm")
        },
    }
    negative = {"provisional_v2_false_accepts": false_accepts}
    return positive, negative


def test_r1_predeclared_required_criteria_exact_boundaries() -> None:
    positive, negative = _criterion_inputs()
    assert replication._r1_criteria_result(positive, negative)[
        "development_replication_pass"
    ] is True

    positive, negative = _criterion_inputs(accepted=28)
    assert replication._r1_criteria_result(positive, negative)[
        "development_replication_pass"
    ] is False
    positive, negative = _criterion_inputs(false_accepts=3)
    assert replication._r1_criteria_result(positive, negative)[
        "development_replication_pass"
    ] is False
    positive, negative = _criterion_inputs(stage1=26)
    assert replication._r1_criteria_result(positive, negative)[
        "development_replication_pass"
    ] is False


def test_cell_preference_is_reported_but_not_required() -> None:
    positive, negative = _criterion_inputs()
    positive["by_zone_and_strength"]["RIGHT-light"][
        "provisional_v2_accepted_attempts"
    ] = 3
    result = replication._r1_criteria_result(positive, negative)
    assert result["cell_preference_pass"] is False
    assert result["cell_preference_affects_required_pass"] is False
    assert result["development_replication_pass"] is True


def test_optional_v1_comparator_cannot_change_v2_criteria() -> None:
    positive, negative = _criterion_inputs()
    before = replication._r1_criteria_result(positive, negative)
    comparator = {"positive_stage2": {"accepted": 0}, "negative_stage2": {"accepted": 99}}
    after = replication._r1_criteria_result(positive, negative)
    assert comparator
    assert after == before
    assert replication.r1_replication_criteria()[
        "v1_comparator_part_of_replication_pass"
    ] is False


def test_provisional_classification_schema_contains_no_stage3_dependency() -> None:
    result = classify_with_provisional_tapness_v2(
        {
            "spectral_bandwidth_hz": 1_000.0,
            "post_0_100_zero_crossing_rate": 0.1,
        },
        _artifact(),
    )
    assert "predicted_zone" not in result
    assert "spatial_margin" not in result


def test_importing_replication_module_does_not_load_sounddevice() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import desksense.robustness_replication; print('sounddevice' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_provisional_artifact_file_is_stable_during_identity_loading() -> None:
    before = hashlib.sha256(PROVISIONAL_PATH.read_bytes()).hexdigest()
    load_provisional_tapness_v2_artifact(PROVISIONAL_PATH)
    after = hashlib.sha256(PROVISIONAL_PATH.read_bytes()).hexdigest()
    assert after == before


def test_r1_requires_exact_reviewed_provisional_artifact_bytes(tmp_path: Path) -> None:
    artifact = replication._load_exact_r1_provisional_artifact(PROVISIONAL_PATH)
    assert artifact["model"]["decision_threshold"] == pytest.approx(
        replication.EXPECTED_R1_PROVISIONAL_V2_THRESHOLD
    )
    assert hashlib.sha256(PROVISIONAL_PATH.read_bytes()).hexdigest() == (
        replication.EXPECTED_R1_PROVISIONAL_V2_FILE_SHA256
    )

    variants = []
    timestamp = copy.deepcopy(artifact)
    timestamp["created_at_utc"] = "2026-09-06T00:00:00Z"
    variants.append(timestamp)
    coefficients = copy.deepcopy(artifact)
    coefficients["model"]["coefficients"][0] += 1.0e-9
    variants.append(coefficients)
    threshold = copy.deepcopy(artifact)
    threshold["model"]["decision_threshold"] -= 1.0e-9
    threshold["model"]["threshold_selection"]["selected_threshold"] -= 1.0e-9
    variants.append(threshold)
    for index, value in enumerate(variants):
        validate_provisional_tapness_v2_artifact(value)
        changed = tmp_path / f"changed-provisional-{index}.json"
        changed.write_text(__import__("json").dumps(value), encoding="utf-8")
        with pytest.raises(
            replication.DevelopmentReplicationError, match="exact reviewed"
        ):
            replication._load_exact_r1_provisional_artifact(changed)


def test_collect_command_rejects_wrong_artifact_before_audio_or_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed_artifact = copy.deepcopy(_artifact())
    changed_artifact["created_at_utc"] = "2026-09-06T00:00:00Z"
    validate_provisional_tapness_v2_artifact(changed_artifact)
    changed_path = tmp_path / "changed-provisional.json"
    changed_path.write_text(
        __import__("json").dumps(changed_artifact), encoding="utf-8"
    )
    monkeypatch.setattr(
        replication,
        "_load_collect_audio_backend",
        lambda: pytest.fail("wrong artifact reached audio backend loading"),
    )
    monkeypatch.setattr(
        replication,
        "run_guided_v2_development_replication_collection",
        lambda *args, **kwargs: pytest.fail("wrong artifact reached session creation"),
    )

    assert replication.main(
        [
            "collect",
            "--device",
            "18",
            "--provisional-artifact",
            str(changed_path),
            "--dataset-root",
            str(tmp_path / "datasets"),
        ]
    ) == 1


def test_direct_collection_rejects_wrong_artifact_before_session_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed_artifact = copy.deepcopy(_artifact())
    changed_artifact["created_at_utc"] = "2026-09-06T00:00:00Z"
    changed_path = tmp_path / "changed-provisional.json"
    changed_path.write_text(
        __import__("json").dumps(changed_artifact), encoding="utf-8"
    )
    monkeypatch.setattr(
        replication,
        "create_robustness_dataset_session",
        lambda *args, **kwargs: pytest.fail("wrong artifact created an R1 session"),
    )

    with pytest.raises(
        replication.DevelopmentReplicationError, match="exact reviewed"
    ):
        replication.run_guided_v2_development_replication_collection(
            FakeBackend(_artifact()),
            device_index=0,
            provisional_v2_artifact_path=changed_path,
            dataset_root=tmp_path / "datasets",
        )


def test_r1_replay_cross_check_reports_canonical_identity_evidence() -> None:
    candidate_replay, stage1_replay = _matching_identity_replays()
    result = replication._cross_check_r1_replay_populations(
        candidate_replay, stage1_replay
    )
    assert result["status"] == "passed"
    assert result["count_match"] is True
    assert result["identity_match"] is True
    assert result["populations"]["positive_detected_candidates"]["count"] == 1
    assert len(
        result["populations"]["negative_candidate_starts"]["identity_sha256"]
    ) == 64


def test_r1_replay_cross_check_rejects_same_counts_wrong_detected_identity() -> None:
    candidate_replay, stage1_replay = _matching_identity_replays()
    candidate_replay["candidate_memberships"][0]["onset_frame_index"] = 50_001

    with pytest.raises(
        replication.DevelopmentReplicationError, match="canonical identities"
    ):
        replication._cross_check_r1_replay_populations(
            candidate_replay, stage1_replay
        )


def test_r1_replay_cross_check_rejects_same_counts_wrong_start_identity() -> None:
    candidate_replay, stage1_replay = _matching_identity_replays()
    candidate_replay["candidate_population"][
        "negative_activity_candidate_start_identities"
    ][0]["candidate_start_route"] = "ordinary"

    with pytest.raises(
        replication.DevelopmentReplicationError, match="canonical identities"
    ):
        replication._cross_check_r1_replay_populations(
            candidate_replay, stage1_replay
        )


def test_optional_v1_comparator_requires_exact_frozen_bytes(tmp_path: Path) -> None:
    replication._load_exact_r1_v1_comparator(V1_PATH)
    changed = tmp_path / "changed-v1.json"
    changed.write_bytes(V1_PATH.read_bytes() + b"\n")
    with pytest.raises(replication.DevelopmentReplicationError, match="exact frozen"):
        replication._load_exact_r1_v1_comparator(changed)


@pytest.mark.parametrize(
    "argv",
    [
        ["collect", "--provisional-artifact", str(PROVISIONAL_PATH)],
        ["collect", "--device", "18"],
        ["evaluate", "dataset", "--provisional-artifact", str(PROVISIONAL_PATH)],
        ["evaluate", "dataset", "--save-report", "report.json"],
    ],
)
def test_research_module_commands_require_their_explicit_arguments(argv) -> None:
    with pytest.raises(SystemExit) as error:
        replication.main(argv)
    assert error.value.code == 2


def test_evaluate_command_is_offline_and_prints_predeclared_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    report = {
        "positive_metrics": {
            "stage1_candidate_attempts": 27,
            "provisional_v2_accepted_attempts": 29,
        },
        "negative_metrics": {"provisional_v2_false_accepts": 2},
        "development_replication_criteria": {
            "required_results": {
                "stage1_pass": True,
                "provisional_v2_survival_pass": True,
                "provisional_v2_false_accept_pass": True,
            },
            "development_replication_pass": True,
        },
    }
    monkeypatch.setattr(
        replication,
        "_load_collect_audio_backend",
        lambda: pytest.fail("offline evaluate loaded sounddevice"),
    )
    monkeypatch.setattr(
        replication, "evaluate_v2_development_replication", lambda *a, **k: report
    )
    monkeypatch.setattr(
        replication,
        "write_v2_development_replication_report",
        lambda value, path: Path(path),
    )

    result = replication.main(
        [
            "evaluate",
            "r1-session",
            "--provisional-artifact",
            str(PROVISIONAL_PATH),
            "--save-report",
            str(tmp_path / "r1.json"),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "DEVELOPMENT REPLICATION ONLY" in output
    assert "27/30" in output
    assert "29/30" in output
    assert "2/300 labeled seconds" in output
    assert "development_replication_pass=True" in output


def test_report_writer_refuses_overwrite_and_non_json_path(tmp_path: Path) -> None:
    report = {"report_type": replication.R1_REPORT_TYPE, "finite": 1.0}
    destination = tmp_path / "r1.json"
    replication.write_v2_development_replication_report(report, destination)
    with pytest.raises(replication.DevelopmentReplicationError, match="overwrite"):
        replication.write_v2_development_replication_report(report, destination)
    with pytest.raises(replication.DevelopmentReplicationError, match="end in .json"):
        replication.write_v2_development_replication_report(
            report, tmp_path / "r1.txt"
        )


def test_module_help_is_research_only_and_does_not_load_sounddevice() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "desksense.robustness_replication", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "research-only" in result.stdout.casefold()
    assert "collect" in result.stdout
    assert "evaluate" in result.stdout


def test_r1_detector_rejection_never_reaches_provisional_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _fake_loaded_r1()
    stage1 = _zero_stage1_replay(dataset)
    rejected_attempt = stage1["positive"]["attempts"][0]
    rejected_attempt.update(
        {
            "stage1_candidate_started": True,
            "associated_candidate_start_count": 1,
            "associated_completed_rejection_count": 1,
            "candidate_starts": [
                {
                    "candidate_start_route": "strong_impact_recovery",
                    "interval_relation": "associated",
                    "onset_frame_index": 50_000,
                }
            ],
        }
    )
    monkeypatch.setattr(
        replication,
        "load_v2_development_replication_dataset",
        lambda *args, **kwargs: (
            dataset,
            {
                "development_replication_complete": True,
                "positive_plan_complete": True,
                "negative_plan_complete": True,
            },
        ),
    )
    monkeypatch.setattr(replication, "_dataset_fingerprint", lambda path: "a" * 64)
    monkeypatch.setattr(
        replication,
        "replay_tapness_v2_research_candidates",
        lambda *args, **kwargs: {
            "candidate_memberships": [],
            "candidate_population": {
                "positive_candidate_count": 0,
                "negative_candidate_count": 0,
                "associated_candidate_start_count": 1,
                "positive_associated_candidate_start_identities": [
                    {
                        "source_record_id": rejected_attempt["record_id"],
                        "onset_frame_index": 50_000,
                        "candidate_start_route": "strong_impact_recovery",
                    }
                ],
                "negative_activity_candidate_start_identities": [],
            },
        },
    )
    monkeypatch.setattr(
        replication, "replay_robustness_dataset", lambda *args, **kwargs: stage1
    )
    monkeypatch.setattr(
        replication,
        "classify_with_provisional_tapness_v2",
        lambda *args, **kwargs: pytest.fail(
            "detector rejection reached provisional Stage 2"
        ),
    )

    report = replication.evaluate_v2_development_replication(
        Path("ignored-r1"),
        provisional_v2_artifact_path=PROVISIONAL_PATH,
        now_fn=lambda: datetime(2026, 9, 5, tzinfo=timezone.utc),
    )

    assert report["positive_metrics"]["stage1_candidate_attempts"] == 1
    assert report["positive_metrics"][
        "associated_completed_detector_rejections"
    ] == 1
    assert report["positive_metrics"]["provisional_v2_accepted_attempts"] == 0
