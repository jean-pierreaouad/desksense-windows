from __future__ import annotations

import json

import numpy as np
import pytest

from desksense.diagnostics import collect_characterization_report


def _input_device(
    name: str, channels: int, sample_rate: float = 48_000.0
) -> dict[str, object]:
    return {
        "name": name,
        "hostapi": 0,
        "max_input_channels": channels,
        "max_output_channels": 0,
        "default_samplerate": sample_rate,
    }


def test_characterization_records_explicit_device_and_serializes(
    audio_backend_factory,
) -> None:
    samples = np.column_stack(
        [
            np.linspace(-0.1, 0.1, 256),
            np.linspace(0.1, -0.1, 256),
            np.sin(np.linspace(0.0, 20.0, 256)) * 0.05,
            np.cos(np.linspace(0.0, 20.0, 256)) * 0.05,
        ]
    ).astype(np.float32)
    backend = audio_backend_factory(
        devices=[
            _input_device("Default microphone", 1),
            _input_device("Selected four-channel array", 4),
        ],
        default_input_index=0,
        recording=samples,
    )

    report = collect_characterization_report(
        backend,
        selected_device_index=1,
        duration_seconds=0.01,
    )

    assert report["status"] == "ok"
    assert report["report_type"] == "microphone_characterization"
    result = report["audio"]["characterization"]
    assert result["status"] == "ok"
    assert result["device"] == {
        "index": 1,
        "name": "Selected four-channel array",
        "host_api": {"index": 0, "name": "Windows WASAPI"},
    }
    assert result["sample_rate_hz"] == 48_000.0
    assert result["channel_count"] == 4
    assert result["frames_requested"] == 480
    assert result["frames_captured"] == 256
    assert result["raw_audio_included"] is False
    assert len(result["analysis"]["per_channel"]) == 4
    assert backend.record_calls[0]["device"] == 1
    assert backend.record_calls[0]["channels"] == 4
    serialized = json.dumps(report, allow_nan=False)
    assert '"raw_audio"' not in serialized
    assert '"samples"' not in serialized


def test_characterization_prioritizes_channels_over_default_rate(
    audio_backend_factory,
) -> None:
    device = _input_device("Flexible array", 4, sample_rate=96_000.0)
    backend = audio_backend_factory(
        devices=[device],
        default_input_index=0,
        unsupported_rates={(0, 44_100)},
        unsupported_settings={
            (0, 96_000, 4),
            (0, 96_000, 3),
            (0, 96_000, 2),
        },
        recording=np.zeros((32, 4), dtype=np.float32),
    )

    report = collect_characterization_report(
        backend, selected_device_index=0, duration_seconds=0.001
    )

    result = report["audio"]["characterization"]
    assert result["status"] == "ok"
    assert result["channel_count"] == 4
    assert result["sample_rate_hz"] == 48_000.0
    assert result["sample_rate_source"] == "validated_common_rate_fallback"


def test_characterization_tries_lower_channels_after_generic_host_error(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory(
        devices=[_input_device("Driver-sensitive array", 4)],
        default_input_index=0,
        recording=np.zeros((32, 3), dtype=np.float32),
    )
    original_check = backend.check_input_settings

    def generic_error_at_four_channels(
        *, device: int, channels: int, dtype: str, samplerate: float
    ) -> None:
        if channels == 4:
            raise RuntimeError("Unanticipated host error")
        original_check(
            device=device,
            channels=channels,
            dtype=dtype,
            samplerate=samplerate,
        )

    backend.check_input_settings = generic_error_at_four_channels

    report = collect_characterization_report(
        backend, selected_device_index=0, duration_seconds=0.001
    )

    result = report["audio"]["characterization"]
    assert report["status"] == "ok"
    assert result["status"] == "ok"
    assert result["channel_count"] == 3


def test_successful_characterization_demotes_other_endpoint_failure(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory(
        devices=[
            _input_device("Unavailable secondary endpoint", 2),
            _input_device("Selected working endpoint", 2),
        ],
        default_input_index=1,
        recording=np.zeros((32, 2), dtype=np.float32),
    )
    original_check = backend.check_input_settings

    def fail_secondary_endpoint(
        *, device: int, channels: int, dtype: str, samplerate: float
    ) -> None:
        if device == 0:
            raise RuntimeError("Unanticipated host error")
        original_check(
            device=device,
            channels=channels,
            dtype=dtype,
            samplerate=samplerate,
        )

    backend.check_input_settings = fail_secondary_endpoint

    report = collect_characterization_report(
        backend, selected_device_index=1, duration_seconds=0.001
    )

    audio = report["audio"]
    assert report["status"] == "ok"
    assert audio["status"] == "ok"
    assert audio["errors"] == []
    assert any("Unavailable secondary endpoint" in item for item in audio["warnings"])
    assert audio["input_devices"][0]["sample_rate_support"][0][
        "supported"
    ] is None


def test_characterization_reports_channel_safety_limit(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory(
        devices=[_input_device("Virtual 16-channel input", 16)],
        default_input_index=0,
        recording=np.zeros((32, 8), dtype=np.float32),
    )

    report = collect_characterization_report(
        backend, selected_device_index=0, duration_seconds=0.001
    )

    result = report["audio"]["characterization"]
    assert result["status"] == "ok"
    assert result["advertised_max_input_channels"] == 16
    assert result["channel_search_limit"] == 8
    assert result["channel_limit_applied"] is True
    assert result["channel_count"] == 8


def test_characterization_permission_error_is_reported(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory(
        settings_error=RuntimeError("Access is denied by Windows")
    )

    report = collect_characterization_report(
        backend, selected_device_index=1
    )

    assert report["status"] == "error"
    result = report["audio"]["characterization"]
    assert result["status"] == "error"
    assert result["error"]["category"] == "permission_denied"
    assert backend.record_calls == []


def test_characterization_invalid_device_does_not_record(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory()

    report = collect_characterization_report(
        backend, selected_device_index=99
    )

    assert report["status"] == "error"
    assert report["audio"]["status"] == "invalid_selection"
    assert report["audio"]["characterization"]["status"] == "not_run"
    assert backend.record_calls == []


@pytest.mark.parametrize(
    "captured",
    [
        np.empty((0, 2), dtype=np.float32),
        np.zeros((16, 1), dtype=np.float32),
        np.full((16, 2), np.nan, dtype=np.float32),
    ],
    ids=["empty", "wrong-channel-count", "non-finite"],
)
def test_characterization_rejects_invalid_capture_arrays(
    audio_backend_factory, captured: np.ndarray
) -> None:
    backend = audio_backend_factory(recording=captured)

    report = collect_characterization_report(
        backend, selected_device_index=1, duration_seconds=0.001
    )

    result = report["audio"]["characterization"]
    assert report["status"] == "error"
    assert result["status"] == "error"
    assert result["raw_audio_included"] is False
    assert "analysis" not in result
