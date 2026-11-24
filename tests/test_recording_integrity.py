from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from desksense.collection import DatasetCollectionError, record_collection_attempt
from desksense.robustness import RobustnessError, record_robustness_segment
from desksense.robustness_replication import (
    DevelopmentReplicationError,
    load_v2_development_replication_dataset,
    run_guided_v2_development_replication_collection,
)
from desksense.tapness_v2_provisional import load_provisional_tapness_v2_artifact


@pytest.fixture(scope="module")
def callback_flags():
    was_loaded = "sounddevice" in sys.modules
    from sounddevice import CallbackFlags

    # Keep the real flags class without affecting offline import assertions.
    if not was_loaded:
        sys.modules.pop("sounddevice")
    return CallbackFlags


@pytest.fixture(params=["positive", "negative"])
def capture(request):
    common = dict(
        device_index=0, sample_rate_hz=48_000, channels=2, sleep_fn=lambda _: None
    )
    if request.param == "positive":
        record = partial(
            record_collection_attempt,
            **common,
            duration_seconds=2.0,
            pre_cue_duration_seconds=0.9,
            tap_cue_fn=lambda: None,
        )
        return record, DatasetCollectionError, 96_000
    record = partial(
        record_robustness_segment,
        **common,
        activity_duration_seconds=1.0,
        warmup_duration_seconds=1.0,
        post_activity_tail_seconds=0.25,
        activity_cue_fn=lambda: None,
        activity_end_cue_fn=lambda: None,
    )
    return record, RobustnessError, 108_000


@pytest.mark.parametrize("empty_flags", [False, True], ids=["none", "empty-flags"])
def test_capture_accepts_clean_wait_status(
    capture, audio_backend_factory, callback_flags, monkeypatch, empty_flags
):
    record, _, frames = capture
    expected = np.full((frames, 2), 0.125, dtype=np.float32)
    backend = audio_backend_factory(recording=expected)
    wait = backend.wait

    def clean_wait():
        wait()
        return callback_flags() if empty_flags else None

    monkeypatch.setattr(backend, "wait", clean_wait)
    np.testing.assert_array_equal(record(backend), expected)
    assert backend.wait_calls == 1


@pytest.mark.parametrize("flag", ["input_overflow", "input_underflow"])
def test_capture_rejects_wait_flags(
    capture, audio_backend_factory, callback_flags, monkeypatch, flag
):
    record, error_type, _ = capture
    backend = audio_backend_factory()
    flags = callback_flags()
    setattr(flags, flag, True)
    monkeypatch.setattr(backend, "wait", lambda: flags)

    with pytest.raises(error_type, match=flag.replace("_", " ")) as error:
        record(backend)

    assert error.value.category == "invalid_capture"


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_capture_preserves_wait_failure_cleanup(
    capture, audio_backend_factory, monkeypatch, failure
):
    record, error_type, _ = capture
    backend = audio_backend_factory()

    def fail_wait():
        raise failure("capture wait failed")

    monkeypatch.setattr(backend, "wait", fail_wait)
    expected_error = KeyboardInterrupt if failure is KeyboardInterrupt else error_type
    with pytest.raises(expected_error, match="capture wait failed"):
        record(backend)

    assert backend.stop_calls == 1
    assert not backend.recording_active


@pytest.mark.parametrize(
    ("valid_positives", "valid_negatives", "expected_error"),
    [(1, 0, DevelopmentReplicationError), (30, 1, RobustnessError), (1, 0, KeyboardInterrupt)],
    ids=["positive-overflow", "negative-overflow", "operator-cancel"],
)
def test_r1_capture_failure_preserves_only_valid_records(
    tmp_path,
    audio_backend_factory,
    callback_flags,
    monkeypatch,
    valid_positives,
    valid_negatives,
    expected_error,
):
    artifact_path = Path("research_baselines/lenovo-tapness-v2-provisional-ab.json")
    compatibility = load_provisional_tapness_v2_artifact(artifact_path)["compatibility"]
    backend = audio_backend_factory(
        devices=[
            {
                "name": compatibility["endpoint_name"],
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000.0,
            }
        ],
        host_apis=[
            {"name": compatibility["host_api_name"], "default_input_device": 0}
        ],
        default_input_index=0,
    )
    cancel = expected_error is KeyboardInterrupt
    valid_count = valid_positives + valid_negatives
    wait = backend.wait
    flags = callback_flags()
    flags.input_overflow = True

    def wait_until_failure():
        if len(backend.record_calls) == valid_count + 1:
            if cancel:
                raise KeyboardInterrupt
            wait()
            return flags
        return wait()

    monkeypatch.setattr(backend, "wait", wait_until_failure)
    with pytest.raises(expected_error, match=None if cancel else "input overflow"):
        run_guided_v2_development_replication_collection(
            backend,
            device_index=0,
            provisional_v2_artifact_path=artifact_path,
            dataset_root=tmp_path,
            session_id="r1-capture-test",
            input_fn=lambda _: "",
            output_fn=lambda _: None,
            countdown_fn=lambda _: None,
            sleep_fn=lambda _: None,
        )

    session = tmp_path / "r1-capture-test"
    dataset, completeness = load_v2_development_replication_dataset(session)
    assert len(dataset.positive_records) == valid_positives
    assert len(dataset.negative_records) == valid_negatives
    assert not completeness["development_replication_complete"]
    with pytest.raises(DevelopmentReplicationError, match="incomplete"):
        load_v2_development_replication_dataset(session, require_complete=True)
    assert len(list(session.rglob("*.npz"))) == valid_count
    assert len(backend.record_calls) == valid_count + 1
