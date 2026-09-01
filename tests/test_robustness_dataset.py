from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from desksense.robustness_dataset import (
    NEGATIVE_RECORD_TYPE,
    POSITIVE_RECORD_TYPE,
    ROBUSTNESS_DATASET_KIND,
    RobustnessDatasetError,
    create_robustness_dataset_session,
    load_robustness_dataset,
)


def _session_metadata() -> dict:
    return {
        "source_mode": "test",
        "selected_endpoint": {
            "index_at_collection": 18,
            "name": "Test Array",
            "host_api": {"index": 3, "name": "Windows WDM-KS"},
        },
        "capture_domain": {
            "sample_rate_hz": 48_000.0,
            "channel_count": 2,
            "dtype": "float32",
        },
        "positive_design": {
            "requested_attempt_count": 1,
            "plan": [
                {
                    "collection_order_index": 1,
                    "zone": "LEFT",
                    "strength": "light",
                    "attempt_number_within_condition": 1,
                }
            ],
        },
        "negative_design": {
            "requested_segment_count": 1,
            "plan": [
                {
                    "collection_order_index": 2,
                    "activity": "typing",
                    "repetition_index": 1,
                }
            ],
        },
    }


def _common(session_id: str, order: int, capture: np.ndarray) -> dict:
    return {
        "session_id": session_id,
        "collection_order_index": order,
        "device": _session_metadata()["selected_endpoint"],
        "sample_rate_hz": 48_000.0,
        "channel_count": 2,
        "capture_dtype": "float32",
        "capture_frames": capture.shape[0],
        "capture_duration_seconds": capture.shape[0] / 48_000.0,
    }


def _create_two_record_session(tmp_path):
    session = create_robustness_dataset_session(
        tmp_path,
        _session_metadata(),
        session_id="robustness-test",
        created_at_utc="2026-08-31T00:00:00+00:00",
    )
    positive = np.arange(40, dtype=np.float32).reshape(20, 2) / 100.0
    negative = -np.arange(24, dtype=np.float32).reshape(12, 2) / 100.0
    session.save_record(
        record_id="robustness-test-left-light-001",
        filename_stem="left_light_001",
        record_type=POSITIVE_RECORD_TYPE,
        capture=positive,
        metadata={
            **_common(session.session_id, 1, positive),
            "intended_zone": "LEFT",
            "intended_strength": "light",
            "attempt_number_within_condition": 1,
            "captured_at_utc": "2026-08-31T00:00:01+00:00",
            "semantic_label": "intended_desk_tap",
            "detector_acceptance_required_for_storage": False,
            "guided_cue": {
                "intended_cue_offset_frames": 2,
                "intended_event_association": {
                    "start_frame_index_inclusive": 2,
                    "end_frame_index_exclusive": 10,
                    "duration_seconds": 8 / 48_000.0,
                },
            },
        },
    )
    session.save_record(
        record_id="robustness-test-typing-001",
        filename_stem="typing_001",
        record_type=NEGATIVE_RECORD_TYPE,
        capture=negative,
        metadata={
            **_common(session.session_id, 2, negative),
            "activity": "typing",
            "repetition_index": 1,
            "semantic_label": "no_intended_desk_tap",
            "segment_started_at_utc": "2026-08-31T00:01:00+00:00",
            "segment_ended_at_utc": "2026-08-31T00:01:10+00:00",
            "warmup_duration_seconds": 4 / 48_000.0,
            "warmup_end_frame_index": 4,
            "activity_start_frame_index": 4,
            "activity_end_frame_index_exclusive": 10,
            "activity_duration_seconds": 6 / 48_000.0,
            "post_activity_tail_duration_seconds": 2 / 48_000.0,
            "post_activity_start_frame_index": 10,
            "post_activity_end_frame_index_exclusive": 12,
        },
    )
    return session, positive, negative


def test_robustness_round_trip_preserves_exact_float32_bytes_and_labels(
    tmp_path,
) -> None:
    session, positive, negative = _create_two_record_session(tmp_path)

    loaded = load_robustness_dataset(session.directory)

    assert loaded.session["dataset_kind"] == ROBUSTNESS_DATASET_KIND
    assert len(loaded.positive_records) == 1
    assert len(loaded.negative_records) == 1
    assert loaded.positive_records[0].capture.tobytes() == positive.tobytes()
    assert loaded.negative_records[0].capture.tobytes() == negative.tobytes()
    assert loaded.positive_records[0].metadata["intended_strength"] == "light"
    assert loaded.negative_records[0].metadata["activity"] == "typing"
    assert loaded.negative_records[0].metadata["segment_started_at_utc"].endswith(
        "+00:00"
    )
    assert not loaded.positive_records[0].capture.flags.writeable


def test_existing_robustness_session_and_capture_refuse_overwrite(tmp_path) -> None:
    session, positive, _negative = _create_two_record_session(tmp_path)
    manifest_before = session.manifest_path.read_bytes()

    with pytest.raises(FileExistsError):
        create_robustness_dataset_session(
            tmp_path, _session_metadata(), session_id="robustness-test"
        )
    with pytest.raises(FileExistsError):
        session.save_record(
            record_id="another-id",
            filename_stem="left_light_001",
            record_type=POSITIVE_RECORD_TYPE,
            capture=positive,
            metadata={
                **_common(session.session_id, 3, positive),
                "intended_zone": "LEFT",
                "intended_strength": "light",
                "attempt_number_within_condition": 2,
                "captured_at_utc": "2026-08-31T00:00:02+00:00",
                "semantic_label": "intended_desk_tap",
                "detector_acceptance_required_for_storage": False,
                "guided_cue": {
                    "intended_cue_offset_frames": 2,
                    "intended_event_association": {
                        "start_frame_index_inclusive": 2,
                        "end_frame_index_exclusive": 10,
                        "duration_seconds": 8 / 48_000.0,
                    },
                },
            },
        )

    assert session.manifest_path.read_bytes() == manifest_before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version=99), "schema version"),
        (lambda value: value.update(dataset_kind="phase2"), "not a Phase 3"),
        (
            lambda value: value["capture_domain"].update(channel_count=1),
            "channel count",
        ),
    ],
)
def test_robustness_session_schema_validation_fails_clearly(
    tmp_path, mutation, message
) -> None:
    session, _positive, _negative = _create_two_record_session(tmp_path)
    path = session.directory / "session.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RobustnessDatasetError, match=message):
        load_robustness_dataset(session.directory)


def test_manifest_metadata_mismatch_and_orphan_artifact_fail(tmp_path) -> None:
    session, _positive, _negative = _create_two_record_session(tmp_path)
    lines = session.manifest_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["intended_strength"] = "firm"
    lines[0] = json.dumps(first)
    session.manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RobustnessDatasetError, match="Embedded metadata"):
        load_robustness_dataset(session.directory)


def test_unreferenced_capture_and_undeclared_activity_fail_validation(
    tmp_path,
) -> None:
    session, _positive, _negative = _create_two_record_session(tmp_path / "orphan")
    source = next(session.positives_directory.glob("*.npz"))
    shutil.copyfile(source, session.positives_directory / "orphan.npz")
    with pytest.raises(RobustnessDatasetError, match="unreferenced NPZ"):
        load_robustness_dataset(session.directory)

    invalid_session = create_robustness_dataset_session(
        tmp_path / "label",
        _session_metadata(),
        session_id="invalid-label",
        created_at_utc="2026-08-31T00:00:00+00:00",
    )
    capture = np.zeros((12, 2), dtype=np.float32)
    invalid_session.save_record(
        record_id="invalid-label-negative-001",
        filename_stem="negative_001",
        record_type=NEGATIVE_RECORD_TYPE,
        capture=capture,
        metadata={
            **_common(invalid_session.session_id, 1, capture),
            "activity": "unlisted_activity",
            "repetition_index": 1,
            "semantic_label": "no_intended_desk_tap",
            "segment_started_at_utc": "2026-08-31T00:01:00+00:00",
            "segment_ended_at_utc": "2026-08-31T00:01:01+00:00",
            "warmup_duration_seconds": 4 / 48_000.0,
            "warmup_end_frame_index": 4,
            "activity_start_frame_index": 4,
            "activity_end_frame_index_exclusive": 10,
            "activity_duration_seconds": 6 / 48_000.0,
            "post_activity_tail_duration_seconds": 2 / 48_000.0,
            "post_activity_start_frame_index": 10,
            "post_activity_end_frame_index_exclusive": 12,
        },
    )
    with pytest.raises(RobustnessDatasetError, match="invalid activity"):
        load_robustness_dataset(invalid_session.directory)


def test_nonfinite_or_wrong_channel_capture_is_never_persisted(tmp_path) -> None:
    session = create_robustness_dataset_session(
        tmp_path, _session_metadata(), session_id="invalid-capture"
    )
    with pytest.raises(ValueError, match="two-channel"):
        session.save_record(
            record_id="bad-one",
            filename_stem="bad_one",
            record_type=POSITIVE_RECORD_TYPE,
            capture=np.zeros((10, 1), dtype=np.float32),
            metadata={},
        )
    capture = np.zeros((10, 2), dtype=np.float32)
    capture[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        session.save_record(
            record_id="bad-two",
            filename_stem="bad_two",
            record_type=POSITIVE_RECORD_TYPE,
            capture=capture,
            metadata={},
        )
    assert session.manifest_path.read_text(encoding="utf-8") == ""
