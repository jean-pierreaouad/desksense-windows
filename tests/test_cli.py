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


def test_dataset_analysis_arguments_parse() -> None:
    args = cli.build_parser().parse_args(
        [
            "--analyze-dataset",
            "datasets/session-001",
            "--interaction-context",
            "hand-location-confounded",
            "--save-report",
            "reports/session-001-analysis.json",
        ]
    )

    assert args.analyze_dataset == Path("datasets/session-001")
    assert args.interaction_context == "hand-location-confounded"
    assert args.save_report == Path("reports/session-001-analysis.json")
    assert args.record is False
    assert args.characterize is False
    assert args.collect_dataset is False


@pytest.mark.parametrize(
    "hardware_mode",
    ["--record", "--characterize", "--collect-dataset"],
)
def test_dataset_analysis_is_mutually_exclusive_with_hardware_modes(
    hardware_mode: str,
) -> None:
    arguments = ["--analyze-dataset", "datasets/session-001", hardware_mode]
    if hardware_mode == "--collect-dataset":
        arguments.extend(["--device", "18"])

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)


@pytest.mark.parametrize(
    "extra_arguments",
    [
        ["--device", "18"],
        ["--samples-per-zone", "20"],
        ["--dataset-root", "datasets"],
    ],
)
def test_dataset_analysis_rejects_hardware_or_collection_only_options(
    extra_arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--analyze-dataset",
                "datasets/session-001",
                "--interaction-context",
                "hand-location-confounded",
                *extra_arguments,
            ]
        )


def test_dataset_analysis_requires_explicit_interaction_context() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--analyze-dataset", "datasets/session-001"])


def test_interaction_context_is_analysis_only() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--interaction-context", "hand-location-confounded"])


def test_dataset_analysis_cli_never_loads_audio_backend(
    monkeypatch, tmp_path, capsys
) -> None:
    session_path = tmp_path / "datasets" / "synthetic-session"
    destination = tmp_path / "analysis.json"
    report = {
        "generated_at_utc": "2026-08-27T12:34:56+00:00",
        "dataset": {"session_path": str(session_path)},
    }
    calls: list[object] = []

    def fail_if_audio_is_loaded():
        pytest.fail("offline analysis must not load sounddevice")

    monkeypatch.setattr(cli, "_load_audio_backend", fail_if_audio_is_loaded)
    monkeypatch.setattr(
        cli,
        "analyze_dataset",
        lambda path, *, interaction_context: (
            calls.append((path, interaction_context)) or report
        ),
    )
    monkeypatch.setattr(
        cli, "format_analysis_summary", lambda _report: "offline summary"
    )

    def fake_write(received_report, path):
        calls.append((received_report, path))
        return path.resolve()

    monkeypatch.setattr(cli, "write_analysis_report", fake_write)

    exit_code = cli.main(
        [
            "--analyze-dataset",
            str(session_path),
            "--interaction-context",
            "hand-location-confounded",
            "--save-report",
            str(destination),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "offline summary" in captured.out
    assert "JSON analysis report saved to" in captured.out
    assert calls == [
        (session_path, "hand-location-confounded"),
        (report, destination),
    ]


def test_dataset_analysis_without_save_option_uses_auto_report_path(
    monkeypatch, tmp_path
) -> None:
    session_path = tmp_path / "synthetic-session"
    report = {
        "generated_at_utc": "2026-08-27T12:34:56.123456+00:00",
        "dataset": {"session_path": str(session_path)},
    }
    written_paths: list[Path] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "_load_audio_backend",
        lambda: pytest.fail("offline analysis must not load sounddevice"),
    )
    monkeypatch.setattr(
        cli,
        "analyze_dataset",
        lambda _path, *, interaction_context: report,
    )
    monkeypatch.setattr(cli, "format_analysis_summary", lambda _report: "summary")

    def fake_write(_report, path):
        written_paths.append(path)
        return path.resolve()

    monkeypatch.setattr(cli, "write_analysis_report", fake_write)

    exit_code = cli.main(
        [
            "--analyze-dataset",
            str(session_path),
            "--interaction-context",
            "hand-location-confounded",
        ]
    )

    assert exit_code == 0
    assert written_paths == [
        Path("reports/desksense-dataset-analysis-20260827T123456.123456Z.json")
    ]


def test_invalid_dataset_analysis_path_fails_without_audio_backend(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "_load_audio_backend",
        lambda: pytest.fail("offline analysis must not load sounddevice"),
    )

    exit_code = cli.main(
        [
            "--analyze-dataset",
            str(tmp_path / "missing-session"),
            "--interaction-context",
            "hand-location-confounded",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Dataset analysis failed" in captured.err
    assert "does not exist" in captured.err


def test_freeze_baseline_arguments_parse() -> None:
    args = cli.build_parser().parse_args(
        [
            "--freeze-baseline",
            "datasets/development-session",
            "--interaction-context",
            "hand-location-confounded",
            "--save-baseline",
            "baselines/lenovo-left-right-v1.json",
        ]
    )

    assert args.freeze_baseline == Path("datasets/development-session")
    assert args.interaction_context == "hand-location-confounded"
    assert args.save_baseline == Path("baselines/lenovo-left-right-v1.json")
    assert args.evaluate_frozen_baseline is None
    assert args.baseline is None


def test_external_evaluation_arguments_parse() -> None:
    args = cli.build_parser().parse_args(
        [
            "--evaluate-frozen-baseline",
            "datasets/external-session",
            "--baseline",
            "baselines/lenovo-left-right-v1.json",
            "--interaction-context",
            "same-hand",
            "--save-report",
            "reports/phase2c-external.json",
        ]
    )

    assert args.evaluate_frozen_baseline == Path("datasets/external-session")
    assert args.baseline == Path("baselines/lenovo-left-right-v1.json")
    assert args.interaction_context == "same-hand"
    assert args.save_report == Path("reports/phase2c-external.json")
    assert args.freeze_baseline is None


@pytest.mark.parametrize(
    "arguments",
    [
        ["--freeze-baseline", "datasets/dev"],
        [
            "--freeze-baseline",
            "datasets/dev",
            "--interaction-context",
            "hand-location-confounded",
        ],
        [
            "--freeze-baseline",
            "datasets/dev",
            "--interaction-context",
            "hand-location-confounded",
            "--save-baseline",
            "baseline.json",
            "--save-report",
            "report.json",
        ],
        ["--evaluate-frozen-baseline", "datasets/external"],
        [
            "--evaluate-frozen-baseline",
            "datasets/external",
            "--interaction-context",
            "same-hand",
        ],
        ["--save-baseline", "baseline.json"],
        ["--baseline", "baseline.json"],
    ],
)
def test_phase2c_required_and_mode_specific_arguments(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(arguments)


@pytest.mark.parametrize(
    "other_mode",
    [
        ["--record"],
        ["--characterize"],
        ["--collect-dataset", "--device", "18"],
        ["--analyze-dataset", "datasets/analysis"],
        ["--evaluate-frozen-baseline", "datasets/external"],
    ],
)
def test_freeze_mode_is_mutually_exclusive_with_every_other_mode(
    other_mode: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["--freeze-baseline", "datasets/dev", *other_mode]
        )


@pytest.mark.parametrize(
    "mode",
    ["--freeze-baseline", "--evaluate-frozen-baseline"],
)
@pytest.mark.parametrize(
    "extra_arguments",
    [
        ["--device", "18"],
        ["--samples-per-zone", "20"],
        ["--dataset-root", "datasets"],
    ],
)
def test_phase2c_offline_modes_reject_hardware_collection_options(
    mode: str, extra_arguments: list[str]
) -> None:
    arguments = [
        mode,
        "datasets/session",
        "--interaction-context",
        "same-hand",
        *extra_arguments,
    ]
    arguments.extend(
        ["--save-baseline", "baseline.json"]
        if mode == "--freeze-baseline"
        else ["--baseline", "baseline.json"]
    )
    with pytest.raises(SystemExit):
        cli.main(arguments)


def test_freeze_cli_never_loads_audio_backend(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "datasets" / "development"
    destination = tmp_path / "baselines" / "baseline.json"
    artifact = {"artifact_type": "synthetic"}
    calls: list[object] = []
    monkeypatch.setattr(
        cli,
        "_load_audio_backend",
        lambda: pytest.fail("baseline freezing must not load sounddevice"),
    )
    monkeypatch.setattr(
        cli,
        "create_frozen_baseline",
        lambda path, *, interaction_context: (
            calls.append((path, interaction_context)) or artifact
        ),
    )
    monkeypatch.setattr(
        cli, "format_frozen_baseline_summary", lambda _artifact: "frozen summary"
    )

    def fake_write(received, path, *, source_session_path):
        calls.append((received, path, source_session_path))
        return path.resolve()

    monkeypatch.setattr(cli, "write_frozen_baseline", fake_write)

    exit_code = cli.main(
        [
            "--freeze-baseline",
            str(source),
            "--interaction-context",
            "hand-location-confounded",
            "--save-baseline",
            str(destination),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "frozen summary" in captured.out
    assert calls == [
        (source, "hand-location-confounded"),
        (artifact, destination, source),
    ]


def test_external_evaluation_cli_delegates_without_loading_audio_backend(
    monkeypatch, tmp_path, capsys
) -> None:
    external = tmp_path / "datasets" / "external"
    baseline = tmp_path / "baselines" / "baseline.json"
    destination = tmp_path / "reports" / "external.json"
    report = {"generated_at_utc": "2026-08-29T12:34:56+00:00"}
    calls: list[object] = []
    monkeypatch.setattr(
        cli,
        "_load_audio_backend",
        lambda: pytest.fail("external evaluation must not load sounddevice"),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_frozen_baseline",
        lambda session, model, *, interaction_context: (
            calls.append((session, model, interaction_context)) or report
        ),
    )
    monkeypatch.setattr(
        cli,
        "format_external_evaluation_summary",
        lambda _report: "external summary",
    )

    def fake_write(received, path, *, external_session_path, baseline_path):
        calls.append(
            (received, path, external_session_path, baseline_path)
        )
        return path.resolve()

    monkeypatch.setattr(cli, "write_external_evaluation_report", fake_write)

    exit_code = cli.main(
        [
            "--evaluate-frozen-baseline",
            str(external),
            "--baseline",
            str(baseline),
            "--interaction-context",
            "same-hand",
            "--save-report",
            str(destination),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "external summary" in captured.out
    assert calls == [
        (external, baseline, "same-hand"),
        (report, destination, external, baseline),
    ]


def test_freeze_missing_session_fails_clearly_without_audio_backend(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "_load_audio_backend",
        lambda: pytest.fail("baseline freezing must not load sounddevice"),
    )

    exit_code = cli.main(
        [
            "--freeze-baseline",
            str(tmp_path / "missing-development"),
            "--interaction-context",
            "hand-location-confounded",
            "--save-baseline",
            str(tmp_path / "baseline.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Frozen baseline creation failed" in captured.err
    assert "does not exist" in captured.err


def test_external_missing_baseline_fails_clearly_without_audio_backend(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "_load_audio_backend",
        lambda: pytest.fail("external evaluation must not load sounddevice"),
    )

    exit_code = cli.main(
        [
            "--evaluate-frozen-baseline",
            str(tmp_path / "missing-external"),
            "--baseline",
            str(tmp_path / "missing-baseline.json"),
            "--interaction-context",
            "same-hand",
            "--save-report",
            str(tmp_path / "report.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "External evaluation failed" in captured.err
    assert "does not exist" in captured.err
