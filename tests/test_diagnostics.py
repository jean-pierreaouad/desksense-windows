from __future__ import annotations

import json

import numpy as np
import pytest

from desksense.diagnostics import (
    channel_layout,
    collect_diagnostic_report,
    describe_audio_error,
    measure_channel_levels,
    write_json_report,
)


@pytest.mark.parametrize(
    ("channels", "expected"),
    [(0, "unavailable"), (1, "mono"), (2, "stereo"), (4, "multichannel")],
)
def test_channel_layout(channels: int, expected: str) -> None:
    assert channel_layout(channels) == expected


def test_measure_channel_levels() -> None:
    samples = np.array(
        [
            [1.0, 0.5],
            [-0.5, -0.5],
            [0.0, 0.5],
            [0.5, -0.5],
        ],
        dtype=np.float32,
    )

    levels = measure_channel_levels(samples)

    assert levels[0]["peak_absolute"] == pytest.approx(1.0)
    assert levels[0]["rms"] == pytest.approx(np.sqrt(0.375))
    assert levels[0]["peak_dbfs"] == pytest.approx(0.0)
    assert levels[1]["peak_absolute"] == pytest.approx(0.5)
    assert levels[1]["rms"] == pytest.approx(0.5)


def test_permission_error_without_message_is_classified() -> None:
    assert describe_audio_error(PermissionError())["category"] == (
        "permission_denied"
    )


def test_inventory_filters_outputs_and_reports_capabilities(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory(unsupported_rates={(1, 44_100)})

    report = collect_diagnostic_report(backend)

    assert report["status"] == "ok"
    audio = report["audio"]
    assert audio["default_input_device_index"] == 1
    assert audio["selected_input_device_index"] == 1
    assert len(audio["input_devices"]) == 1
    microphone = audio["input_devices"][0]
    assert microphone["name"] == "Microphone Array"
    assert microphone["host_api"]["name"] == "Windows WASAPI"
    assert microphone["max_input_channels"] == 2
    assert microphone["channel_layout"] == "stereo"
    assert microphone["default_sample_rate_hz"] == 48_000.0
    assert microphone["sample_rate_support"] == [
        {
            "sample_rate_hz": 44_100,
            "max_input_channels_checked": 2,
            "supported": False,
            "error": {
                "category": "unsupported_configuration",
                "message": "Sample rate 44100 is unsupported",
            },
        },
        {
            "sample_rate_hz": 48_000,
            "max_input_channels_checked": 2,
            "supported": True,
            "supported_input_channels": 2,
        },
    ]


def test_sample_rate_support_finds_a_lower_supported_channel_count(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory(max_supported_channels={1: 1})

    report = collect_diagnostic_report(backend)

    support = report["audio"]["input_devices"][0]["sample_rate_support"]
    assert support[0]["supported"] is True
    assert support[0]["supported_input_channels"] == 1
    assert support[1]["supported"] is True
    assert support[1]["supported_input_channels"] == 1


def test_sounddevice_style_default_pair_is_resolved_exactly(
    audio_backend_factory,
) -> None:
    class InputOutputPair:
        def __getitem__(self, index: int) -> int:
            return (1, -1)[index]

    backend = audio_backend_factory(default_input_index=None)
    backend.default.device = InputOutputPair()

    report = collect_diagnostic_report(backend)

    assert report["status"] == "ok"
    assert report["audio"]["default_input_device_index"] == 1


def test_queried_default_index_wins_when_device_signatures_are_identical(
    audio_backend_factory,
) -> None:
    duplicate = {
        "name": "Identical Microphone",
        "hostapi": 0,
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 48_000.0,
    }
    backend = audio_backend_factory(
        devices=[dict(duplicate), dict(duplicate)], default_input_index=1
    )
    backend.default.device = (None, -1)

    report = collect_diagnostic_report(backend)

    assert report["status"] == "ok"
    assert report["audio"]["default_input_device_index"] == 1


def test_portaudio_version_failure_does_not_block_inventory(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory()

    def fail_version_query():
        raise RuntimeError("Version unavailable")

    backend.get_portaudio_version = fail_version_query

    report = collect_diagnostic_report(backend)

    assert report["status"] == "ok"
    assert report["audio"]["backend"] == {
        "library": "sounddevice",
        "version": "0.test",
    }


def test_recording_uses_default_rate_and_highest_supported_channel_count(
    audio_backend_factory,
) -> None:
    samples = np.array([[0.25], [-0.5], [0.0]], dtype=np.float32)
    backend = audio_backend_factory(
        max_supported_channels={1: 1}, recording=samples
    )

    report = collect_diagnostic_report(
        backend, record=True, duration_seconds=0.001
    )

    assert report["status"] == "ok"
    recording = report["audio"]["recording"]
    assert recording["status"] == "ok"
    assert recording["sample_rate_hz"] == 48_000.0
    assert recording["sample_rate_source"] == "device_default"
    assert recording["channels"] == 1
    assert recording["frames_requested"] == 48
    assert recording["frames_captured"] == 3
    assert recording["channel_levels"][0]["peak_absolute"] == pytest.approx(0.5)
    assert recording["channel_levels"][0]["rms"] == pytest.approx(
        np.sqrt((0.25**2 + 0.5**2) / 3)
    )
    assert backend.record_calls == [
        {
            "frames": 48,
            "samplerate": 48_000.0,
            "channels": 1,
            "dtype": "float32",
            "device": 1,
            "blocking": True,
        }
    ]
    assert "raw_audio" not in recording


def test_recording_permission_error_is_reported(audio_backend_factory) -> None:
    backend = audio_backend_factory(
        recording_error=RuntimeError("Access is denied by Windows")
    )

    report = collect_diagnostic_report(backend, record=True)

    assert report["status"] == "error"
    recording = report["audio"]["recording"]
    assert recording["status"] == "error"
    assert recording["error"]["category"] == "permission_denied"


def test_configuration_permission_error_is_not_mislabeled(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory(
        settings_error=RuntimeError("Access is denied by Windows")
    )

    report = collect_diagnostic_report(backend, record=True)

    assert report["status"] == "error"
    audio = report["audio"]
    assert audio["status"] == "capability_check_error"
    assert audio["input_devices"][0]["sample_rate_support"][0][
        "supported"
    ] is None
    assert audio["recording"]["error"]["category"] == "permission_denied"


def test_capability_check_does_not_mask_permission_error_with_fallback_error(
    audio_backend_factory,
) -> None:
    backend = audio_backend_factory()

    def mixed_settings_errors(
        *, device: int, channels: int, dtype: str, samplerate: float
    ) -> None:
        if channels == 2:
            raise RuntimeError("Access is denied by Windows")
        raise RuntimeError("Invalid number of channels")

    backend.check_input_settings = mixed_settings_errors

    report = collect_diagnostic_report(backend)

    support = report["audio"]["input_devices"][0]["sample_rate_support"]
    assert support[0]["supported"] is None
    assert support[0]["error"]["category"] == "permission_denied"
    assert support[1]["supported"] is None
    assert support[1]["error"]["category"] == "permission_denied"


def test_missing_microphone_is_reported(audio_backend_factory) -> None:
    backend = audio_backend_factory(
        devices=[
            {
                "name": "Speakers",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000.0,
            }
        ],
        default_input_index=None,
    )

    report = collect_diagnostic_report(backend)

    assert report["status"] == "error"
    assert report["audio"]["status"] == "no_input_devices"
    assert report["audio"]["input_devices"] == []


def test_empty_portaudio_device_list_is_reported(audio_backend_factory) -> None:
    backend = audio_backend_factory(devices=[], default_input_index=None)

    report = collect_diagnostic_report(backend)

    assert report["status"] == "error"
    assert report["audio"]["status"] == "no_input_devices"


def test_inputs_without_a_default_are_still_listed(audio_backend_factory) -> None:
    backend = audio_backend_factory(default_input_index=None)

    report = collect_diagnostic_report(backend)

    assert report["status"] == "error"
    assert report["audio"]["status"] == "no_default_input"
    assert report["audio"]["input_devices"][0]["name"] == "Microphone Array"
    assert report["audio"]["selected_input_device_index"] is None


def test_enumeration_error_is_reported(audio_backend_factory) -> None:
    backend = audio_backend_factory(
        enumeration_error=RuntimeError("PortAudio initialization failed")
    )

    report = collect_diagnostic_report(backend)

    assert report["status"] == "error"
    assert report["audio"]["status"] == "enumeration_error"
    assert report["audio"]["errors"][0]["message"] == (
        "PortAudio initialization failed"
    )


def test_invalid_selected_device_is_reported(audio_backend_factory) -> None:
    backend = audio_backend_factory()

    report = collect_diagnostic_report(backend, selected_device_index=99)

    assert report["status"] == "error"
    assert report["audio"]["status"] == "invalid_selection"
    assert report["audio"]["selected_input_device_index"] is None


def test_json_report_is_machine_readable_and_not_overwritten(
    audio_backend_factory, tmp_path
) -> None:
    report = collect_diagnostic_report(audio_backend_factory())
    destination = tmp_path / "diagnostic.json"

    saved_path = write_json_report(report, destination)

    assert saved_path == destination.resolve()
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
    assert loaded["audio"]["recording"]["status"] == "not_requested"
    with pytest.raises(FileExistsError):
        write_json_report(report, destination)


def test_json_serialization_failure_does_not_leave_partial_report(
    tmp_path,
) -> None:
    destination = tmp_path / "invalid.json"

    with pytest.raises(ValueError):
        write_json_report({"not_finite": float("nan")}, destination)

    assert not destination.exists()
