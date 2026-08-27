from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from desksense import cli


def test_cli_records_with_mock_backend_and_saves_json(
    audio_backend_factory, monkeypatch, tmp_path, capsys
) -> None:
    backend = audio_backend_factory(
        recording=np.array([[0.1, -0.2], [0.3, 0.0]], dtype=np.float32)
    )
    monkeypatch.setattr(cli, "_load_audio_backend", lambda: backend)
    destination = tmp_path / "report.json"

    exit_code = cli.main(["--record", "--save-report", str(destination)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Microphone Array" in captured.out
    assert "Channel 1" in captured.out
    assert "Raw audio was discarded" in captured.out
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["audio"]["recording"]["status"] == "ok"
    assert "raw_audio" not in loaded["audio"]["recording"]


def test_cli_handles_backend_load_failure(monkeypatch, capsys) -> None:
    def fail_to_load():
        raise OSError("PortAudio DLL could not be loaded")

    monkeypatch.setattr(cli, "_load_audio_backend", fail_to_load)

    exit_code = cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PortAudio DLL could not be loaded" in captured.out


def test_cli_explains_permission_error_during_configuration(
    audio_backend_factory, monkeypatch, capsys
) -> None:
    backend = audio_backend_factory(
        settings_error=RuntimeError("Access is denied by Windows")
    )
    monkeypatch.setattr(cli, "_load_audio_backend", lambda: backend)

    exit_code = cli.main(["--record"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "could not check" in captured.out
    assert "Privacy & security > Microphone" in captured.out


def test_cli_rejects_non_json_report_path(
    audio_backend_factory, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        cli, "_load_audio_backend", lambda: audio_backend_factory()
    )

    exit_code = cli.main(["--save-report", str(tmp_path / "report.txt")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "must end in .json" in captured.err


def test_cli_auto_report_path_is_under_reports(
    audio_backend_factory, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        cli, "_load_audio_backend", lambda: audio_backend_factory()
    )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["--save-report"])

    captured = capsys.readouterr()
    report_files = list((tmp_path / "reports").glob("*.json"))
    assert exit_code == 0
    assert len(report_files) == 1
    assert "JSON report saved to" in captured.out


def test_cli_characterizes_explicit_device_and_saves_json(
    audio_backend_factory, monkeypatch, tmp_path, capsys
) -> None:
    devices = [
        {
            "name": "Chosen microphone array",
            "hostapi": 0,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 48_000.0,
        },
        {
            "name": "Default microphone",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48_000.0,
        },
    ]
    base = np.sin(np.linspace(0.0, 12.0, 512)) * 0.1
    backend = audio_backend_factory(
        devices=devices,
        default_input_index=1,
        recording=np.column_stack((base, base)).astype(np.float32),
    )
    monkeypatch.setattr(cli, "_load_audio_backend", lambda: backend)
    destination = tmp_path / "chosen-device.json"

    exit_code = cli.main(
        [
            "--characterize",
            "--device",
            "0",
            "--save-report",
            str(destination),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Microphone characterization: completed" in captured.out
    assert "Device: 0: Chosen microphone array" in captured.out
    assert "Whole-recording channel pairs" in captured.out
    assert "Raw audio was discarded" in captured.out
    assert backend.record_calls[0] == {
        "frames": 240_000,
        "samplerate": 48_000.0,
        "channels": 2,
        "dtype": "float32",
        "device": 0,
        "blocking": True,
    }
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["report_type"] == "microphone_characterization"
    assert loaded["audio"]["characterization"]["device"]["index"] == 0


def test_cli_record_and_characterize_are_mutually_exclusive(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--record", "--characterize"])

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "not allowed with argument" in captured.err


def test_cli_auto_characterization_report_uses_distinct_name(
    audio_backend_factory, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        cli, "_load_audio_backend", lambda: audio_backend_factory()
    )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(
        ["--characterize", "--device", "1", "--save-report"]
    )

    captured = capsys.readouterr()
    report_files = list(
        (tmp_path / "reports").glob("desksense-characterization-*.json")
    )
    assert exit_code == 0
    assert len(report_files) == 1
    assert "JSON report saved to" in captured.out


def test_dataset_collection_arguments_parse() -> None:
    args = cli.build_parser().parse_args(
        [
            "--collect-dataset",
            "--device",
            "18",
            "--samples-per-zone",
            "20",
            "--dataset-root",
            "local-datasets",
        ]
    )

    assert args.collect_dataset is True
    assert args.device == 18
    assert args.samples_per_zone == 20
    assert args.dataset_root == Path("local-datasets")
    assert args.record is False
    assert args.characterize is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--collect-dataset", "--record", "--device", "18"],
        ["--collect-dataset", "--characterize", "--device", "18"],
        ["--collect-dataset", "--device", "18", "--samples-per-zone", "0"],
    ],
)
def test_invalid_dataset_argument_combinations_are_rejected(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)


def test_collection_only_options_are_rejected_without_collection_mode() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--samples-per-zone", "3"])


def test_dataset_collection_requires_explicit_device() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--collect-dataset"])


def test_dataset_collection_does_not_mix_with_diagnostic_report_saving() -> None:
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--collect-dataset",
                "--device",
                "18",
                "--save-report",
                "reports/not-a-dataset.json",
            ]
        )


def test_dataset_collection_cli_passes_explicit_options_without_real_audio(
    monkeypatch, tmp_path
) -> None:
    backend = object()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "_load_audio_backend", lambda: backend)

    def fake_collection(audio_backend, **kwargs):
        calls.append({"audio_backend": audio_backend, **kwargs})
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_guided_collection", fake_collection)

    exit_code = cli.main(
        [
            "--collect-dataset",
            "--device",
            "18",
            "--samples-per-zone",
            "3",
            "--dataset-root",
            str(tmp_path / "datasets"),
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "audio_backend": backend,
            "device_index": 18,
            "samples_per_zone": 3,
            "dataset_root": tmp_path / "datasets",
        }
    ]
