from __future__ import annotations

import json

import numpy as np
import pytest

import desksense.dataset as dataset_module
from desksense.dataset import create_dataset_session


def _new_session(tmp_path):
    return create_dataset_session(
        tmp_path / "datasets",
        {"dataset_kind": "test", "zones": ["LEFT", "RIGHT"]},
        session_id="test-session",
        created_at_utc="2026-08-25T12:00:00+00:00",
    )


def test_session_and_accepted_sample_are_persisted_losslessly(tmp_path) -> None:
    session = _new_session(tmp_path)
    capture = np.arange(80, dtype=np.float32).reshape(40, 2) / 100
    tap_window = capture[10:30].copy()

    sample_path, record = session.save_sample(
        sample_id="test-session-left_001",
        filename_stem="left_001",
        capture=capture,
        tap_window=tap_window,
        metadata={
            "zone": "LEFT",
            "accepted_sample_number": 1,
            "collection_order_index": 1,
            "sample_rate_hz": 1_000.0,
            "channel_count": 2,
        },
    )

    assert sample_path == (session.samples_directory / "left_001.npz").resolve()
    with np.load(sample_path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {"capture", "tap_window", "metadata_json"}
        assert artifact["capture"].dtype == np.float32
        np.testing.assert_array_equal(artifact["capture"], capture)
        np.testing.assert_array_equal(artifact["tap_window"], tap_window)
        embedded = json.loads(str(artifact["metadata_json"]))
    assert embedded == record
    assert record["saved_sample_path"] == "samples/left_001.npz"

    manifest_lines = session.manifest_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in manifest_lines] == [record]
    session_metadata = json.loads(
        (session.directory / "session.json").read_text(encoding="utf-8")
    )
    assert session_metadata["schema_version"] == 1
    assert session_metadata["session_id"] == "test-session"


def test_existing_session_is_not_overwritten(tmp_path) -> None:
    _new_session(tmp_path)
    original = (tmp_path / "datasets" / "test-session" / "session.json").read_bytes()

    with pytest.raises(FileExistsError):
        _new_session(tmp_path)

    assert (tmp_path / "datasets" / "test-session" / "session.json").read_bytes() == original


def test_existing_sample_is_not_overwritten_or_duplicated_in_manifest(
    tmp_path,
) -> None:
    session = _new_session(tmp_path)
    capture = np.zeros((20, 2), dtype=np.float32)
    kwargs = {
        "sample_id": "test-session-left_001",
        "filename_stem": "left_001",
        "capture": capture,
        "tap_window": capture[:10],
        "metadata": {"zone": "LEFT"},
    }
    session.save_sample(**kwargs)
    original_sample = (session.samples_directory / "left_001.npz").read_bytes()
    original_manifest = session.manifest_path.read_bytes()

    with pytest.raises(FileExistsError):
        session.save_sample(**kwargs)

    assert (session.samples_directory / "left_001.npz").read_bytes() == original_sample
    assert session.manifest_path.read_bytes() == original_manifest


def test_rejected_attempt_log_contains_metadata_but_no_waveform(tmp_path) -> None:
    session = _new_session(tmp_path)

    session.log_rejected_attempt(
        {
            "attempt_index": 1,
            "target_zone": "LEFT",
            "capture_quality": {
                "reasons": [{"code": "signal_too_weak"}]
            },
        }
    )

    record = json.loads(
        session.rejected_attempts_path.read_text(encoding="utf-8")
    )
    assert record["accepted"] is False
    assert record["waveform_saved"] is False
    assert "capture" not in record
    assert list(session.samples_directory.iterdir()) == []


def test_manifest_failure_removes_only_the_new_sample(
    tmp_path, monkeypatch
) -> None:
    session = _new_session(tmp_path)
    capture = np.zeros((20, 2), dtype=np.float32)

    def fail_manifest_append(_path, _value) -> None:
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(dataset_module, "_append_json_line", fail_manifest_append)

    with pytest.raises(OSError, match="simulated manifest failure"):
        session.save_sample(
            sample_id="test-session-left_001",
            filename_stem="left_001",
            capture=capture,
            tap_window=capture[:10],
            metadata={"zone": "LEFT"},
        )

    assert not (session.samples_directory / "left_001.npz").exists()
    assert session.manifest_path.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("session_id", ["CON", "nul.json", "unsafe."])
def test_windows_unsafe_session_names_are_rejected(
    tmp_path, session_id: str
) -> None:
    with pytest.raises(ValueError, match="Windows filename"):
        create_dataset_session(
            tmp_path / "datasets",
            {"dataset_kind": "test"},
            session_id=session_id,
        )
