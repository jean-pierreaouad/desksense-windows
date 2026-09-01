from __future__ import annotations

from pathlib import Path

import pytest

from desksense import cli


def test_robustness_collection_arguments_parse() -> None:
    args = cli.build_parser().parse_args(
        [
            "--collect-robustness-dataset",
            "--device",
            "18",
            "--dataset-root",
            "local-evidence",
        ]
    )

    assert args.collect_robustness_dataset is True
    assert args.device == 18
    assert args.dataset_root == Path("local-evidence")
    assert args.collect_dataset is False
    assert args.sense is False


def test_robustness_replay_arguments_parse_with_optional_baseline() -> None:
    args = cli.build_parser().parse_args(
        [
            "--replay-robustness-dataset",
            "datasets/robustness-session",
            "--baseline",
            "baselines/lenovo-left-right-v1.json",
            "--save-report",
            "reports/robustness.json",
        ]
    )

    assert args.replay_robustness_dataset == Path("datasets/robustness-session")
    assert args.baseline == Path("baselines/lenovo-left-right-v1.json")
    assert args.save_report == Path("reports/robustness.json")
    assert args.device is None


def test_robustness_collection_requires_explicit_device() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--collect-robustness-dataset"])


@pytest.mark.parametrize(
    "other_mode",
    [
        ["--sense"],
        ["--collect-dataset"],
        ["--analyze-dataset", "datasets/phase2"],
        ["--replay-robustness-dataset", "datasets/robustness"],
    ],
)
def test_robustness_collection_is_mutually_exclusive_with_other_modes(
    other_mode: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["--collect-robustness-dataset", "--device", "18", *other_mode]
        )


@pytest.mark.parametrize(
    "extra",
    [
        ["--device", "18"],
        ["--dataset-root", "datasets"],
        ["--samples-per-zone", "3"],
        ["--interaction-context", "same-hand"],
    ],
)
def test_offline_robustness_replay_rejects_hardware_or_unrelated_options(
    extra: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--replay-robustness-dataset",
                "datasets/robustness-session",
                *extra,
            ]
        )


def test_robustness_collection_rejects_report_and_baseline_options() -> None:
    for extra in (
        ["--save-report", "reports/not-applicable.json"],
        ["--baseline", "baseline.json"],
        ["--samples-per-zone", "3"],
    ):
        with pytest.raises(SystemExit):
            cli.main(
                ["--collect-robustness-dataset", "--device", "18", *extra]
            )


def test_robustness_collection_lazily_loads_backend_and_delegates(
    monkeypatch, tmp_path
) -> None:
    backend = object()
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(cli, "_load_audio_backend", lambda: backend)

    def fake_collection(audio_backend, **kwargs):
        calls.append((audio_backend, kwargs))
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_guided_robustness_collection", fake_collection)
    root = tmp_path / "local-evidence"

    assert (
        cli.main(
            [
                "--collect-robustness-dataset",
                "--device",
                "18",
                "--dataset-root",
                str(root),
            ]
        )
        == 0
    )
    assert calls == [(backend, {"device_index": 18, "dataset_root": root})]


def test_robustness_replay_is_offline_and_writes_requested_report(
    monkeypatch, tmp_path, capsys
) -> None:
    session = tmp_path / "session"
    baseline = tmp_path / "baseline.json"
    report_path = tmp_path / "report.json"
    report = {"generated_at_utc": "2026-08-31T12:00:00+00:00"}
    calls: list[object] = []

    def fail_if_audio_is_loaded():
        pytest.fail("offline robustness replay must not load sounddevice")

    monkeypatch.setattr(cli, "_load_audio_backend", fail_if_audio_is_loaded)

    def fake_replay(path, *, baseline_path):
        calls.append((path, baseline_path))
        return report

    monkeypatch.setattr(cli, "replay_robustness_dataset", fake_replay)
    monkeypatch.setattr(cli, "format_robustness_summary", lambda value: "summary")

    def fake_write(value, path):
        calls.append((value, path))
        return path.resolve()

    monkeypatch.setattr(cli, "write_robustness_report", fake_write)

    exit_code = cli.main(
        [
            "--replay-robustness-dataset",
            str(session),
            "--baseline",
            str(baseline),
            "--save-report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "summary" in captured.out
    assert calls == [
        (session, baseline),
        (report, report_path),
    ]

