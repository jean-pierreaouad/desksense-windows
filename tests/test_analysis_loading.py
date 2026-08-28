from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from desksense import cli
from desksense.analysis import (
    DatasetIntegrityError,
    INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED,
    analyze_dataset,
    load_dataset_session,
    write_analysis_report,
)
from desksense.dataset import DatasetSession, create_dataset_session


_SAMPLE_RATE_HZ = 1_000.0
_CHANNEL_COUNT = 2
_CAPTURE_FRAMES = 64
_TAP_WINDOW_START = 16
_TAP_WINDOW_END = 48
_ENDPOINT = {
    "index": 18,
    "name": "Synthetic two-channel array",
    "host_api": {"index": 3, "name": "Windows WDM-KS"},
}


def _create_session(
    tmp_path: Path,
    *,
    samples_per_zone: int = 1,
    session_id: str = "analysis-session",
) -> DatasetSession:
    session = create_dataset_session(
        tmp_path / "datasets",
        {
            "dataset_kind": "guided_labeled_multichannel_tap_dataset",
            "project_phase": "2A",
            "selected_endpoint": _ENDPOINT,
            "capture_configuration": {
                "sample_rate_hz": _SAMPLE_RATE_HZ,
                "channel_count": _CHANNEL_COUNT,
                "dtype": "float32",
            },
            "zones": ["LEFT", "RIGHT"],
            "requested_samples_per_zone": samples_per_zone,
            "requested_total_samples": 2 * samples_per_zone,
            "collection_order": "round_robin_alternating_zones",
        },
        session_id=session_id,
        created_at_utc="2026-08-26T12:00:00+00:00",
    )
    for accepted_number in range(1, samples_per_zone + 1):
        _save_valid_sample(session, "LEFT", accepted_number)
        _save_valid_sample(session, "RIGHT", accepted_number)
    return session


def _save_valid_sample(
    session: DatasetSession,
    zone: str,
    accepted_number: int,
) -> tuple[Path, dict[str, Any]]:
    order_index = (accepted_number - 1) * 2 + (1 if zone == "LEFT" else 2)
    base = (0.35 * np.sin(np.linspace(0.0, 4.0 * np.pi, 32))).astype(
        np.float32
    )
    channel_2_scale = 0.5 if zone == "LEFT" else 2.0
    tap_window = np.column_stack((base, base * channel_2_scale)).astype(
        np.float32
    )
    capture = np.zeros((_CAPTURE_FRAMES, _CHANNEL_COUNT), dtype=np.float32)
    capture[_TAP_WINDOW_START:_TAP_WINDOW_END] = tap_window
    stem = f"{zone.lower()}_{accepted_number:03d}"
    return session.save_sample(
        sample_id=f"{session.session_id}-{stem}",
        filename_stem=stem,
        capture=capture,
        tap_window=tap_window,
        metadata={
            "accepted": True,
            "zone": zone,
            "accepted_sample_number": accepted_number,
            "collection_order_index": order_index,
            "attempt_index": order_index,
            "captured_at_utc": (
                f"2026-08-26T12:00:{order_index:02d}+00:00"
            ),
            "device": _ENDPOINT,
            "sample_rate_hz": _SAMPLE_RATE_HZ,
            "channel_count": _CHANNEL_COUNT,
            "capture_dtype": "float32",
            "capture_frames": _CAPTURE_FRAMES,
            "capture_duration_seconds": _CAPTURE_FRAMES / _SAMPLE_RATE_HZ,
            "capture_quality": {
                "accepted": True,
                "status": "accepted",
                "transient": {
                    "exact_tap_window_available": True,
                    "exact_tap_window_start_sample": _TAP_WINDOW_START,
                    "exact_tap_window_end_sample_exclusive": _TAP_WINDOW_END,
                },
            },
        },
    )


def _manifest_records(session: DatasetSession) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in session.manifest_path.read_text(encoding="utf-8").splitlines()
    ]


def _write_manifest(
    session: DatasetSession, records: list[dict[str, Any]]
) -> None:
    content = "".join(
        json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
        for record in records
    )
    session.manifest_path.write_text(content, encoding="utf-8", newline="\n")


def _artifact_members(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as artifact:
        return {
            "capture": np.asarray(artifact["capture"]).copy(),
            "tap_window": np.asarray(artifact["tap_window"]).copy(),
            "metadata": json.loads(str(artifact["metadata_json"])),
        }


def _rewrite_artifact(
    path: Path,
    *,
    capture: np.ndarray[Any, Any] | None = None,
    tap_window: np.ndarray[Any, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    members = _artifact_members(path)
    np.savez_compressed(
        path,
        capture=members["capture"] if capture is None else capture,
        tap_window=(
            members["tap_window"] if tap_window is None else tap_window
        ),
        metadata_json=np.asarray(
            json.dumps(
                members["metadata"] if metadata is None else metadata,
                allow_nan=False,
                sort_keys=True,
            )
        ),
    )


def _all_file_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _assert_no_waveform_keys(value: Any) -> None:
    forbidden = {"capture", "tap_window", "metadata_json", "raw_audio"}
    if isinstance(value, Mapping):
        assert forbidden.isdisjoint(str(key) for key in value)
        for child in value.values():
            _assert_no_waveform_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_waveform_keys(child)


def test_valid_dataset_loads_without_modifying_source_files(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    session.log_rejected_attempt({"attempt_index": 3, "target_zone": "RIGHT"})
    before = _all_file_bytes(session.directory)

    loaded = load_dataset_session(session.directory)

    assert len(loaded.samples) == 2
    assert loaded.rejected_attempt_count == 1
    assert loaded.integrity_validation["status"] == "passed"
    assert loaded.integrity_validation["counts_by_zone"] == {
        "LEFT": 1,
        "RIGHT": 1,
    }
    assert loaded.samples[0].tap_window.dtype == np.float32
    assert not loaded.samples[0].tap_window.flags.writeable
    assert _all_file_bytes(session.directory) == before


def test_missing_session_json_fails_clearly(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    (session.directory / "session.json").unlink()

    with pytest.raises(DatasetIntegrityError, match="Missing session.json"):
        load_dataset_session(session.directory)


def test_missing_manifest_fails_clearly(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    session.manifest_path.unlink()

    with pytest.raises(DatasetIntegrityError, match="Missing manifest.jsonl"):
        load_dataset_session(session.directory)


@pytest.mark.parametrize("filename", ["session.json", "manifest.jsonl"])
def test_invalid_dataset_json_fails_clearly(
    tmp_path: Path, filename: str
) -> None:
    session = _create_session(tmp_path)
    (session.directory / filename).write_text("{invalid\n", encoding="utf-8")

    with pytest.raises(DatasetIntegrityError, match=f"Invalid {filename}"):
        load_dataset_session(session.directory)


def test_missing_accepted_sample_npz_fails_clearly(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    (session.samples_directory / "left_001.npz").unlink()

    with pytest.raises(DatasetIntegrityError, match="Missing accepted sample NPZ"):
        load_dataset_session(session.directory)


def test_corrupt_npz_fails_clearly(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    (session.samples_directory / "left_001.npz").write_bytes(b"not an NPZ")

    with pytest.raises(DatasetIntegrityError, match="Could not read sample"):
        load_dataset_session(session.directory)


def test_truncated_npz_fails_clearly(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    sample_path = session.samples_directory / "left_001.npz"
    original = sample_path.read_bytes()
    sample_path.write_bytes(original[: len(original) // 2])

    with pytest.raises(DatasetIntegrityError, match="Could not read sample"):
        load_dataset_session(session.directory)


@pytest.mark.parametrize("missing_member", ["capture", "tap_window", "metadata_json"])
def test_missing_required_npz_member_is_rejected(
    tmp_path: Path, missing_member: str
) -> None:
    session = _create_session(tmp_path)
    sample_path = session.samples_directory / "left_001.npz"
    members = _artifact_members(sample_path)
    archive_members = {
        "capture": members["capture"],
        "tap_window": members["tap_window"],
        "metadata_json": np.asarray(json.dumps(members["metadata"])),
    }
    del archive_members[missing_member]
    np.savez_compressed(sample_path, **archive_members)

    with pytest.raises(DatasetIntegrityError, match=missing_member):
        load_dataset_session(session.directory)


def test_duplicate_sample_id_is_rejected_case_insensitively(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    records = _manifest_records(session)
    records[1]["sample_id"] = records[0]["sample_id"].upper()
    _write_manifest(session, records)

    with pytest.raises(DatasetIntegrityError, match="Duplicate sample_id"):
        load_dataset_session(session.directory)


def test_duplicate_accepted_number_within_zone_is_rejected(tmp_path: Path) -> None:
    session = _create_session(tmp_path, samples_per_zone=2)
    records = _manifest_records(session)
    records[2]["accepted_sample_number"] = 1
    _write_manifest(session, records)

    with pytest.raises(
        DatasetIntegrityError,
        match="Duplicate accepted sample number for LEFT: 1",
    ):
        load_dataset_session(session.directory)


def test_embedded_metadata_manifest_mismatch_is_rejected(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    sample_path = session.samples_directory / "left_001.npz"
    embedded = _artifact_members(sample_path)["metadata"]
    embedded["attempt_index"] = 99
    _rewrite_artifact(sample_path, metadata=embedded)

    with pytest.raises(DatasetIntegrityError, match="embedded metadata"):
        load_dataset_session(session.directory)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_waveform_data_is_rejected(
    tmp_path: Path, bad_value: float
) -> None:
    session = _create_session(tmp_path)
    sample_path = session.samples_directory / "left_001.npz"
    capture = _artifact_members(sample_path)["capture"]
    capture[0, 0] = bad_value
    _rewrite_artifact(sample_path, capture=capture)

    with pytest.raises(DatasetIntegrityError, match="non-finite values"):
        load_dataset_session(session.directory)


@pytest.mark.parametrize(
    ("array_name", "replacement", "message"),
    [
        ("capture", np.zeros(64, dtype=np.float32), "two-dimensional"),
        (
            "tap_window",
            np.zeros((32, 1), dtype=np.float32),
            "tap-window channel count disagrees",
        ),
        ("capture", np.zeros((64, 2), dtype=np.float64), "dtype must be float32"),
    ],
)
def test_invalid_array_shape_channel_count_or_dtype_is_rejected(
    tmp_path: Path,
    array_name: str,
    replacement: np.ndarray[Any, Any],
    message: str,
) -> None:
    session = _create_session(tmp_path)
    sample_path = session.samples_directory / "left_001.npz"
    kwargs = {array_name: replacement}
    _rewrite_artifact(sample_path, **kwargs)

    with pytest.raises(DatasetIntegrityError, match=message):
        load_dataset_session(session.directory)


def test_sample_rate_metadata_must_agree_with_session(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    records = _manifest_records(session)
    records[0]["sample_rate_hz"] = 2_000.0
    _write_manifest(session, records)

    with pytest.raises(DatasetIntegrityError, match="sample rate disagrees"):
        load_dataset_session(session.directory)


def test_collection_order_index_must_match_manifest_sequence(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    records = _manifest_records(session)
    records[0]["collection_order_index"] = 2
    _write_manifest(session, records)

    with pytest.raises(DatasetIntegrityError, match="must be 1, not 2"):
        load_dataset_session(session.directory)


def test_report_is_finite_waveform_free_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    session = _create_session(tmp_path, samples_per_zone=20)
    report = analyze_dataset(
        session.directory,
        interaction_context=INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED,
        now_fn=lambda: datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc),
    )

    serialized = json.dumps(report, allow_nan=False, sort_keys=True)
    assert json.loads(serialized)["status"] == "ok"
    _assert_no_waveform_keys(report)
    evidence = report["evidence_boundary"]
    assert evidence["scope_label"] == (
        "within-session LEFT/RIGHT interaction-condition separation"
    )
    assert evidence["hand_location_confound"] == {
        "present": True,
        "LEFT": "left hand + left location",
        "RIGHT": "right hand + right location",
        "consequence": "This dataset cannot isolate spatial location alone.",
        "provenance": (
            "Explicit analyst-supplied interaction context; tapping hand "
            "was not "
            "inferred from waveform or Phase 2A sample metadata."
        ),
    }
    assert report["report_privacy"]["raw_waveforms_included"] is False

    report_path = tmp_path / "reports" / "analysis.json"
    written = write_analysis_report(report, report_path)
    original = written.read_bytes()
    with pytest.raises(FileExistsError):
        write_analysis_report(report, report_path)
    assert written.read_bytes() == original

    in_dataset_path = session.directory / "analysis.json"
    with pytest.raises(ValueError, match="outside the source dataset"):
        write_analysis_report(report, in_dataset_path)
    assert not in_dataset_path.exists()


def test_same_hand_context_is_explicit_and_not_reported_as_confounded(
    tmp_path: Path,
) -> None:
    session = _create_session(tmp_path, samples_per_zone=20)

    report = analyze_dataset(
        session.directory,
        interaction_context="same-hand",
        now_fn=lambda: datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc),
    )

    evidence = report["evidence_boundary"]
    assert evidence["interaction_context"] == "same-hand"
    assert evidence["interaction_context_inferred_from_dataset"] is False
    assert evidence["hand_location_confound"]["present"] is False
    assert evidence["scope_label"] == (
        "within-session same-hand LEFT/RIGHT separation"
    )


def test_real_offline_cli_path_writes_report_without_loading_audio_backend(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    session = _create_session(tmp_path, samples_per_zone=20)
    destination = tmp_path / "reports" / "synthetic-analysis.json"
    before = _all_file_bytes(session.directory)

    monkeypatch.setattr(
        cli,
        "_load_audio_backend",
        lambda: pytest.fail("offline analysis must not initialize sounddevice"),
    )

    exit_code = cli.main(
        [
            "--analyze-dataset",
            str(session.directory),
            "--interaction-context",
            "hand-location-confounded",
            "--save-report",
            str(destination),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "within-session chronological holdout" in captured.out
    assert "within-session leave-one-pair-out cross-validation" in captured.out
    assert "left hand + left location" in captured.out
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["report_privacy"]["microphone_accessed"] is False
    assert report["report_privacy"]["raw_waveforms_included"] is False
    assert _all_file_bytes(session.directory) == before
