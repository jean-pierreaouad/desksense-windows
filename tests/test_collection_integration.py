from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pytest

from desksense.collection import (
    CollectionParameters,
    DatasetCollectionError,
    run_guided_collection,
)


def _input_device() -> dict[str, object]:
    return {
        "name": "Synthetic two-channel array",
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 1_000.0,
    }


def _tap(center: int = 250) -> np.ndarray:
    samples = np.zeros((500, 2), dtype=np.float32)
    samples[center, 0] = 0.1
    samples[center, 1] = 0.05
    return samples


def _parameters() -> CollectionParameters:
    return CollectionParameters(
        capture_duration_seconds=0.5,
        tap_window_seconds=0.1,
        transient_energy_window_seconds=0.005,
    )


def _guided_capture(provider):
    def capture(
        _backend,
        *,
        pre_cue_duration_seconds,
        tap_cue_fn,
        sleep_fn,
        **_kwargs,
    ):
        sleep_fn(pre_cue_duration_seconds)
        tap_cue_fn()
        return provider()

    return capture


def test_rejection_retries_same_zone_without_advancing_number(
    tmp_path, audio_backend_factory
) -> None:
    backend = audio_backend_factory(
        devices=[_input_device()],
        default_input_index=0,
    )
    recordings = deque(
        [np.zeros((500, 2), dtype=np.float32), _tap(), _tap()]
    )
    prompts = deque(["", "", "", "", ""])
    output: list[str] = []

    result = run_guided_collection(
        backend,
        device_index=0,
        samples_per_zone=1,
        dataset_root=tmp_path / "datasets",
        parameters=_parameters(),
        input_fn=lambda _prompt: prompts.popleft(),
        output_fn=output.append,
        countdown_fn=lambda _write: None,
        capture_fn=_guided_capture(recordings.popleft),
        pre_cue_sleep_fn=lambda _duration: None,
        now_fn=lambda: datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc),
        session_id="retry-session",
    )

    assert result["accepted_samples"] == 2
    assert result["accepted_by_zone"] == {"LEFT": 1, "RIGHT": 1}
    assert result["attempts"] == 3
    session_directory = tmp_path / "datasets" / "retry-session"
    manifest = [
        json.loads(line)
        for line in (session_directory / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["zone"] for item in manifest] == ["LEFT", "RIGHT"]
    assert [item["accepted_sample_number"] for item in manifest] == [1, 1]
    assert [item["collection_order_index"] for item in manifest] == [1, 2]
    assert [item["attempt_index"] for item in manifest] == [2, 3]
    rejected = [
        json.loads(line)
        for line in (session_directory / "rejected_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rejected) == 1
    assert rejected[0]["target_zone"] == "LEFT"
    assert "signal_too_weak" in {
        reason["code"]
        for reason in rejected[0]["capture_quality"]["reasons"]
    }
    assert any("Rejected; retrying the same zone" in line for line in output)
    assert output.count("TAP NOW") == 3


def test_manual_retry_does_not_increment_or_persist_sample(
    tmp_path, audio_backend_factory
) -> None:
    backend = audio_backend_factory(
        devices=[_input_device()],
        default_input_index=0,
    )
    recordings = deque([_tap(), _tap(), _tap()])
    prompts = deque(["", "r", "", "", "", ""])

    result = run_guided_collection(
        backend,
        device_index=0,
        samples_per_zone=1,
        dataset_root=tmp_path / "datasets",
        parameters=_parameters(),
        input_fn=lambda _prompt: prompts.popleft(),
        output_fn=lambda _message: None,
        countdown_fn=lambda _write: None,
        capture_fn=_guided_capture(recordings.popleft),
        pre_cue_sleep_fn=lambda _duration: None,
        now_fn=lambda: datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc),
        session_id="manual-retry",
    )

    assert result["attempts"] == 3
    samples = sorted(
        path.name
        for path in (tmp_path / "datasets" / "manual-retry" / "samples").iterdir()
    )
    assert samples == ["left_001.npz", "right_001.npz"]
    rejected_record = json.loads(
        (tmp_path / "datasets" / "manual-retry" / "rejected_attempts.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert rejected_record["capture_quality"]["reasons"] == [
        {
            "code": "user_requested_retry",
            "message": "The user chose to discard this attempt and retry.",
        }
    ]


def test_session_metadata_preserves_endpoint_backend_and_parameters(
    tmp_path, audio_backend_factory
) -> None:
    backend = audio_backend_factory(
        devices=[_input_device()],
        default_input_index=0,
    )
    prompts = deque(["", "", "", ""])

    run_guided_collection(
        backend,
        device_index=0,
        samples_per_zone=1,
        dataset_root=tmp_path / "datasets",
        parameters=_parameters(),
        input_fn=lambda _prompt: prompts.popleft(),
        output_fn=lambda _message: None,
        countdown_fn=lambda _write: None,
        capture_fn=_guided_capture(_tap),
        pre_cue_sleep_fn=lambda _duration: None,
        now_fn=lambda: datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc),
        session_id="metadata-session",
    )

    session = json.loads(
        (tmp_path / "datasets" / "metadata-session" / "session.json")
        .read_text(encoding="utf-8")
    )
    assert session["selected_endpoint"] == {
        "index": 0,
        "name": "Synthetic two-channel array",
        "host_api": {"index": 0, "name": "Windows WASAPI"},
    }
    assert session["audio_backend"]["version"] == "0.test"
    assert session["audio_backend"]["portaudio_version_text"] == (
        "PortAudio test backend"
    )
    assert session["capture_configuration"]["sample_rate_hz"] == 1_000.0
    assert session["capture_configuration"]["channel_count"] == 2
    assert session["guided_tap_cue"] == {
        "cue_text": "TAP NOW",
        "cue_is_audible": False,
        "capture_started_before_cue": True,
        "intended_pre_cue_duration_seconds": 0.2,
        "intended_cue_offset_samples": 200,
        "nominal_post_cue_duration_seconds": 0.3,
        "nominal_post_cue_samples": 300,
        "timing_kind": "intended_guided_visual_cue_schedule",
        "timing_caution": (
            "The stream is started before the cue, but OS scheduling, terminal "
            "rendering latency, and human reaction time are not measured."
        ),
    }
    assert session["zones"] == ["LEFT", "RIGHT"]
    assert session["requested_samples_per_zone"] == 1
    assert session["collection_parameters"]["lag_used_for_quality_decision"] is False
    assert session["privacy"]["network_upload_performed"] is False
    manifest = json.loads(
        (tmp_path / "datasets" / "metadata-session" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert manifest["guided_tap_cue"]["intended_cue_offset_samples"] == 200
    assert manifest["guided_tap_cue"]["capture_started_before_cue"] is True


def test_invalid_device_does_not_create_a_dataset_session(
    tmp_path, audio_backend_factory
) -> None:
    dataset_root = tmp_path / "datasets"

    with pytest.raises(DatasetCollectionError, match="not an available"):
        run_guided_collection(
            audio_backend_factory(),
            device_index=99,
            samples_per_zone=1,
            dataset_root=dataset_root,
            parameters=_parameters(),
            input_fn=lambda _prompt: "",
            output_fn=lambda _message: None,
            countdown_fn=lambda _write: None,
        )

    assert not dataset_root.exists()


def test_configuration_permission_failure_does_not_create_session(
    tmp_path, audio_backend_factory
) -> None:
    dataset_root = tmp_path / "datasets"
    backend = audio_backend_factory(
        settings_error=RuntimeError("Access is denied by Windows")
    )

    with pytest.raises(DatasetCollectionError) as raised:
        run_guided_collection(
            backend,
            device_index=1,
            samples_per_zone=1,
            dataset_root=dataset_root,
            parameters=_parameters(),
            input_fn=lambda _prompt: "",
            output_fn=lambda _message: None,
            countdown_fn=lambda _write: None,
        )

    assert raised.value.category == "permission_denied"
    assert not dataset_root.exists()
    assert backend.record_calls == []
