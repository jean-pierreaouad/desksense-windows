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


def test_sense_arguments_parse() -> None:
    args = cli.build_parser().parse_args(
        [
            "--sense",
            "--device",
            "18",
            "--baseline",
            "baselines/lenovo-left-right-v1.json",
            "--tapness-baseline",
            "baselines/lenovo-tapness-v1.json",
        ]
    )

    assert args.sense is True
    assert args.device == 18
    assert args.baseline == Path("baselines/lenovo-left-right-v1.json")
    assert args.tapness_baseline == Path("baselines/lenovo-tapness-v1.json")
    assert args.record is False
    assert args.characterize is False
    assert args.collect_dataset is False
    assert args.sense_diagnostics is False


def test_tapness_fit_arguments_parse() -> None:
    args = cli.build_parser().parse_args(
        [
            "--fit-tapness-baseline",
            "datasets/development-a",
            "--save-tapness-baseline",
            "baselines/tapness.json",
        ]
    )

    assert args.fit_tapness_baseline == Path("datasets/development-a")
    assert args.save_tapness_baseline == Path("baselines/tapness.json")


def test_sense_requires_tapness_baseline(capsys) -> None:
    with pytest.raises(SystemExit):
        cli.main(
            ["--sense", "--device", "18", "--baseline", "spatial.json"]
        )
    assert "--sense requires --tapness-baseline PATH" in capsys.readouterr().err


def test_sense_diagnostics_arguments_parse() -> None:
    args = cli.build_parser().parse_args(
        [
            "--sense",
            "--sense-diagnostics",
            "--device",
            "18",
            "--baseline",
            "baseline.json",
            "--tapness-baseline",
            "tapness.json",
        ]
    )

    assert args.sense is True
    assert args.sense_diagnostics is True


@pytest.mark.parametrize(
    "other_mode",
    [
        ["--record"],
        ["--characterize"],
        ["--collect-dataset"],
        ["--analyze-dataset", "datasets/session"],
        ["--freeze-baseline", "datasets/development"],
        ["--evaluate-frozen-baseline", "datasets/external"],
    ],
)
def test_sense_is_mutually_exclusive_with_existing_modes(
    other_mode: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "--sense",
                "--device",
                "18",
                "--baseline",
                "baseline.json",
                "--tapness-baseline",
                "tapness.json",
                *other_mode,
            ]
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--sense", "--baseline", "baseline.json"],
        ["--sense", "--device", "18"],
        [
            "--sense",
            "--device",
            "18",
            "--baseline",
            "baseline.json",
            "--samples-per-zone",
            "2",
        ],
        [
            "--sense",
            "--device",
            "18",
            "--baseline",
            "baseline.json",
            "--dataset-root",
            "datasets",
        ],
        [
            "--sense",
            "--device",
            "18",
            "--baseline",
            "baseline.json",
            "--interaction-context",
            "same-hand",
        ],
        [
            "--sense",
            "--device",
            "18",
            "--baseline",
            "baseline.json",
            "--save-report",
            "report.json",
        ],
        [
            "--sense",
            "--device",
            "18",
            "--baseline",
            "baseline.json",
            "--save-baseline",
            "other-baseline.json",
        ],
    ],
)
def test_sense_requires_explicit_inputs_and_rejects_irrelevant_options(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(arguments)


def test_sense_cli_lazily_loads_backend_and_delegates(
    monkeypatch, capsys
) -> None:
    backend = object()
    baseline_path = Path("baselines/synthetic.json")
    tapness_path = Path("baselines/tapness.json")
    calls: list[object] = []

    def fake_load_backend():
        calls.append("load_backend")
        return backend

    def fake_run(
        audio_backend,
        *,
        device_index,
        baseline_path,
        tapness_baseline_path,
        event_handler,
        diagnostics_enabled,
    ):
        calls.append(
            (
                audio_backend,
                device_index,
                baseline_path,
                tapness_baseline_path,
                event_handler,
                diagnostics_enabled,
            )
        )
        event_handler(
            cli.LiveSensingEvent(
                event_type="state",
                status="armed",
                message="armed",
                stream_epoch=0,
            )
        )
        return object()

    monkeypatch.setattr(cli, "_load_audio_backend", fake_load_backend)
    monkeypatch.setattr(cli, "run_live_sensing", fake_run)

    exit_code = cli.main(
        [
            "--sense",
            "--device",
            "18",
            "--baseline",
            str(baseline_path),
            "--tapness-baseline",
            str(tapness_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Armed (detector epoch 0)." in captured.out
    assert calls == [
        "load_backend",
        (
            backend,
            18,
            baseline_path,
            tapness_path,
            cli._print_live_sensing_event,
            False,
        ),
    ]


def test_tapness_fit_cli_is_offline_and_writes_exclusively(
    monkeypatch, capsys
) -> None:
    artifact = {"artifact_type": "synthetic"}
    calls: list[object] = []
    monkeypatch.setattr(
        cli,
        "_load_audio_backend",
        lambda: pytest.fail("offline tapness fitting loaded audio"),
    )
    monkeypatch.setattr(
        cli,
        "create_tapness_baseline",
        lambda path: calls.append(("fit", path)) or artifact,
    )
    monkeypatch.setattr(
        cli,
        "write_tapness_baseline",
        lambda value, path: calls.append(("write", value, path)) or path,
    )
    monkeypatch.setattr(
        cli, "format_tapness_baseline_summary", lambda value: "tapness summary"
    )

    assert cli.main(
        [
            "--fit-tapness-baseline",
            "datasets/development-a",
            "--save-tapness-baseline",
            "baselines/tapness.json",
        ]
    ) == 0
    assert calls == [
        ("fit", Path("datasets/development-a")),
        ("write", artifact, Path("baselines/tapness.json")),
    ]
    assert "tapness summary" in capsys.readouterr().out


def test_sense_diagnostics_requires_sense(capsys) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--sense-diagnostics"])

    assert "--sense-diagnostics requires --sense" in capsys.readouterr().err


def test_sense_cli_passes_opt_in_diagnostics(monkeypatch) -> None:
    received: list[bool] = []
    monkeypatch.setattr(cli, "_load_audio_backend", lambda: object())

    def fake_run(*args, **kwargs):
        received.append(kwargs["diagnostics_enabled"])
        return object()

    monkeypatch.setattr(cli, "run_live_sensing", fake_run)

    assert (
        cli.main(
            [
                "--sense",
                "--sense-diagnostics",
                "--device",
                "18",
                "--baseline",
                "baseline.json",
                "--tapness-baseline",
                "tapness.json",
            ]
        )
        == 0
    )
    assert received == [True]


def test_sense_cli_reports_backend_load_failure_without_running(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "_load_audio_backend",
        lambda: (_ for _ in ()).throw(OSError("PortAudio unavailable")),
    )
    monkeypatch.setattr(
        cli,
        "run_live_sensing",
        lambda *args, **kwargs: pytest.fail("runner must not be called"),
    )

    exit_code = cli.main(
        [
            "--sense", "--device", "18", "--baseline", "baseline.json",
            "--tapness-baseline", "tapness.json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Could not load the audio backend" in captured.err
    assert "PortAudio unavailable" in captured.err


def test_sense_cli_interrupt_returns_130(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_load_audio_backend", lambda: object())

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_live_sensing", interrupt)

    exit_code = cli.main(
        [
            "--sense", "--device", "18", "--baseline", "baseline.json",
            "--tapness-baseline", "tapness.json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 130
    assert "DeskSense live sensing interrupted" in captured.err


def test_sense_cli_reports_realtime_permission_error(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "_load_audio_backend", lambda: object())

    def fail(*args, **kwargs):
        raise cli.RealtimeSensingError(
            "Access is denied by Windows", category="permission_denied"
        )

    monkeypatch.setattr(cli, "run_live_sensing", fail)

    exit_code = cli.main(
        [
            "--sense", "--device", "18", "--baseline", "baseline.json",
            "--tapness-baseline", "tapness.json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Live sensing failed: Access is denied by Windows" in captured.err
    assert "Privacy & security > Microphone" in captured.err


def test_live_startup_and_state_event_formatting() -> None:
    startup = cli.LiveSensingEvent(
        event_type="startup",
        status="learning",
        message="learning",
        stream_epoch=0,
        timing={"stream_reported_latency_seconds": 0.0125},
        details={
            "device_index": 18,
            "endpoint_name": "Microphone Array 1",
            "host_api_name": "Windows WDM-KS",
            "sample_rate_hz": 48_000.0,
            "channel_count": 2,
            "dtype": "float32",
            "baseline_path": "baselines/lenovo-left-right-v1.json",
            "baseline_source_session_id": "development-session",
            "threshold_db": 0.12793235855251162,
            "direction": "LEFT below threshold; RIGHT at or above threshold",
            "startup_learning_seconds": 0.75,
        },
    )

    rendered = cli.format_live_sensing_event(startup)

    assert "Device: 18: Microphone Array 1" in rendered
    assert "Host API: Windows WDM-KS" in rendered
    assert "48000 Hz, 2 channel(s), float32" in rendered
    assert "reported latency 12.50 ms" in rendered
    assert "source session development-session" in rendered
    assert "Frozen threshold: +0.127932 dB" in rendered
    assert "LEFT below threshold; RIGHT at or above threshold" in rendered
    assert "Learning room noise for 0.75 s" in rendered


def test_live_detection_event_formats_db_margin_and_timing_not_probability() -> None:
    event = cli.LiveSensingEvent(
        event_type="detection",
        status="detected",
        message="detected",
        stream_epoch=2,
        predicted_zone="RIGHT",
        feature_value_db=1.25,
        threshold_db=0.125,
        absolute_margin_db=1.125,
        onset_frame_index=40_000,
        center_frame_index=40_123,
        timing={
            "approximate_onset_to_result_seconds": 84_714.0,
            "queue_dwell_seconds": 0.003,
            "detector_latency_seconds": 0.1,
        },
        details={
            "tapness_inference": {
                "uncalibrated_model_output": 0.8,
                "model_margin": 0.2,
            }
        },
    )

    rendered = cli.format_live_sensing_event(event)

    assert "Tap: RIGHT" in rendered
    assert "peak ratio=+1.250000 dB" in rendered
    assert "threshold=+0.125000 dB" in rendered
    assert "margin=1.125000 dB" in rendered
    assert "epoch=2; onset=40000; center=40123" in rendered
    assert "onset-to-result" not in rendered
    assert "84714000" not in rendered
    assert "queue dwell=3.00 ms" in rendered
    assert "detector lookahead=100.00 ms" in rendered
    assert "tapness output=0.800000" in rendered
    assert "tapness margin=+0.200000" in rendered
    assert "probability" not in rendered.lower()


def test_live_rejection_and_discontinuity_event_formatting() -> None:
    rejection = cli.LiveSensingEvent(
        event_type="rejection",
        status="rejected",
        message="rejected",
        stream_epoch=1,
        rejection_reasons=("near_clipping",),
        onset_frame_index=1_000,
        center_frame_index=1_020,
    )
    discontinuity = cli.LiveSensingEvent(
        event_type="discontinuity",
        status="relearning",
        message="gap",
        stream_epoch=2,
        rejection_reasons=("queue_overflow", "callback_sequence_gap"),
        details={
            "dropped_callback_count_before": 2,
            "dropped_frame_count_before": 960,
        },
    )

    rejected_text = cli.format_live_sensing_event(rejection)
    discontinuity_text = cli.format_live_sensing_event(discontinuity)

    assert "Tap rejected: near_clipping" in rejected_text
    assert "epoch=1; onset=1000; center=1020" in rejected_text
    assert "Audio discontinuity: queue_overflow, callback_sequence_gap" in (
        discontinuity_text
    )
    assert "dropped callbacks=2, dropped frames=960" in discontinuity_text
    assert "Detector reset to learning (epoch 2)" in discontinuity_text


def test_live_candidate_start_and_summary_diagnostic_formatting() -> None:
    gate = {
        "block_start_frame_index": 40_000,
        "block_end_frame_index_exclusive": 40_240,
        "block_rms": 0.01,
        "block_peak_absolute": 0.25,
        "block_crest_factor": 25.0,
        "learned_noise_floor_rms": 0.0005,
        "required_rms_threshold": 0.0015,
        "required_peak_threshold": 0.003,
        "required_crest_threshold": 3.0,
        "rms_ratio": 6.666666,
        "peak_ratio": 83.333333,
        "crest_ratio": 8.333333,
        "all_gates_score": 6.666666,
    }
    candidate = cli.LiveSensingEvent(
        event_type="diagnostic",
        status="candidate_started",
        message="candidate",
        stream_epoch=0,
        onset_frame_index=40_003,
        details={"gate_block": gate},
    )
    summary = cli.LiveSensingEvent(
        event_type="diagnostic",
        status="summary",
        message="summary",
        stream_epoch=0,
        details={
            "detector_state": "armed",
            "learned_noise_floor_rms": 0.0005,
            "interval_counters": {
                "fixed_blocks_processed_total": 200,
                "learning_suppressed_blocks": 0,
                "armed_evaluated_blocks": 150,
                "collecting_blocks": 20,
                "refractory_suppressed_blocks": 30,
                "rms_pass_count": 2,
                "rms_fail_count": 148,
                "peak_pass_count": 1,
                "peak_fail_count": 149,
                "crest_pass_count": 10,
                "crest_fail_count": 140,
                "all_gates_pass_count": 1,
                "onset_candidates_started": 1,
                "completed_detections": 1,
                "completed_rejections": 0,
            },
            "closest_armed_block": gate,
            "transport": {
                "callback_count": 50,
                "callback_frames": 48_000,
                "queue_high_water_mark": 2,
                "dropped_callback_count": 0,
                "dropped_frame_count": 0,
                "discontinuity_count": 0,
                "current_packet_portaudio_status_flags": (),
            },
        },
    )

    candidate_text = cli.format_live_sensing_event(candidate)
    summary_text = cli.format_live_sensing_event(summary)

    assert "Diagnostic candidate start" in candidate_text
    assert "onset=40003" in candidate_text
    assert "block=[40000:40240)" in candidate_text
    assert "ratio=" in candidate_text
    assert "Sense diagnostics: epoch=0; state=armed" in summary_text
    assert "RMS=2/148" in summary_text
    assert "all-pass=1, starts=1, detected=1, rejected=0" in summary_text
    assert "callbacks=50, frames=48000" in summary_text
    assert "queue high-water=2" in summary_text


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
