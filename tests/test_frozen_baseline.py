from __future__ import annotations

import copy
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest

from desksense import frozen_baseline
from desksense.analysis import (
    INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED,
    INTERACTION_CONTEXT_SAME_HAND,
    load_dataset_session,
)
from desksense.dataset import DatasetSession, create_dataset_session
from desksense.features import PRIMARY_FEATURE_NAME
from desksense.frozen_baseline import (
    create_frozen_baseline,
    evaluate_frozen_baseline,
    format_external_evaluation_summary,
    load_frozen_baseline,
    wilson_score_interval,
    write_external_evaluation_report,
    write_frozen_baseline,
)


_SAMPLE_RATE_HZ = 1_000.0
_CAPTURE_FRAMES = 64
_TAP_WINDOW_START = 16
_TAP_WINDOW_END = 48
_ENDPOINT = {
    "index": 18,
    "name": "Synthetic two-channel array",
    "host_api": {"index": 3, "name": "Windows WDM-KS"},
}
_FIXED_TIME = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def _create_session(
    root: Path,
    *,
    session_id: str,
    left_db: Sequence[float],
    right_db: Sequence[float],
    zones: tuple[str, str] = ("LEFT", "RIGHT"),
    created_at_utc: str = "2026-08-28T10:00:00+00:00",
) -> DatasetSession:
    assert len(left_db) == len(right_db) > 0
    values = {"LEFT": left_db, "RIGHT": right_db}
    session = create_dataset_session(
        root,
        {
            "dataset_kind": "guided_labeled_multichannel_tap_dataset",
            "project_phase": "2A",
            "selected_endpoint": _ENDPOINT,
            "capture_configuration": {
                "sample_rate_hz": _SAMPLE_RATE_HZ,
                "channel_count": 2,
                "dtype": "float32",
            },
            "zones": list(zones),
            "requested_samples_per_zone": len(left_db),
            "requested_total_samples": 2 * len(left_db),
            "collection_order": "round_robin_alternating_zones",
        },
        session_id=session_id,
        created_at_utc=created_at_utc,
    )
    order_index = 0
    for accepted_number in range(1, len(left_db) + 1):
        for zone in zones:
            order_index += 1
            _save_sample(
                session,
                zone=zone,
                accepted_number=accepted_number,
                order_index=order_index,
                peak_ratio_db=float(values[zone][accepted_number - 1]),
            )
    return session


def _save_sample(
    session: DatasetSession,
    *,
    zone: str,
    accepted_number: int,
    order_index: int,
    peak_ratio_db: float,
) -> None:
    base = (0.35 * np.sin(np.linspace(0.0, 4.0 * np.pi, 32))).astype(
        np.float32
    )
    channel_2_scale = 10.0 ** (peak_ratio_db / 20.0)
    tap_window = np.column_stack((base, base * channel_2_scale)).astype(
        np.float32
    )
    capture = np.zeros((_CAPTURE_FRAMES, 2), dtype=np.float32)
    capture[_TAP_WINDOW_START:_TAP_WINDOW_END] = tap_window
    stem = f"{zone.lower()}_{accepted_number:03d}"
    session.save_sample(
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
            "captured_at_utc": f"2026-08-28T10:00:{order_index:02d}+00:00",
            "device": _ENDPOINT,
            "sample_rate_hz": _SAMPLE_RATE_HZ,
            "channel_count": 2,
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


def _source_session(
    root: Path, *, session_id: str = "source-session"
) -> DatasetSession:
    return _create_session(
        root,
        session_id=session_id,
        left_db=[-6.0] * 20,
        right_db=[6.0] * 20,
    )


def _freeze(session: DatasetSession) -> dict[str, Any]:
    return create_frozen_baseline(
        session.directory,
        interaction_context=INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED,
        now_fn=lambda: _FIXED_TIME,
    )


def _write_baseline(
    artifact: Mapping[str, Any], path: Path, source: DatasetSession
) -> Path:
    return write_frozen_baseline(
        artifact,
        path,
        source_session_path=source.directory,
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


def _fingerprint(session: DatasetSession) -> dict[str, Any]:
    return load_dataset_session(session.directory).source_fingerprint


def test_source_fingerprint_is_deterministic_and_absolute_path_independent(
    tmp_path: Path,
) -> None:
    source = _source_session(tmp_path / "first")
    first = _fingerprint(source)
    second = _fingerprint(source)
    copied_directory = tmp_path / "second" / source.session_id
    copied_directory.parent.mkdir(parents=True)
    shutil.copytree(source.directory, copied_directory)
    copied = load_dataset_session(copied_directory).source_fingerprint

    assert first == second
    assert first == copied
    assert len(first["digest_hex"]) == 64
    assert [component["path"] for component in first["components"]] == [
        "session.json",
        "manifest.jsonl",
        *[
            f"samples/{zone.lower()}_{number:03d}.npz"
            for number in range(1, 21)
            for zone in ("LEFT", "RIGHT")
        ],
    ]


@pytest.mark.parametrize("changed_member", ["session", "manifest", "npz"])
def test_source_fingerprint_changes_for_valid_exact_byte_changes(
    tmp_path: Path, changed_member: str
) -> None:
    source = _source_session(tmp_path / changed_member)
    before = _fingerprint(source)["digest_hex"]

    if changed_member == "session":
        session_path = source.directory / "session.json"
        session_path.write_bytes(session_path.read_bytes() + b" \n")
    elif changed_member == "manifest":
        manifest = source.manifest_path
        manifest.write_bytes(manifest.read_bytes().replace(b"\n", b" \n"))
    else:
        sample_path = source.samples_directory / "left_001.npz"
        with ZipFile(sample_path, mode="a") as archive:
            archive.comment = b"valid fingerprint-only ZIP comment"

    after = _fingerprint(source)["digest_hex"]

    assert after != before


def test_source_fingerprint_records_logical_manifest_order(tmp_path: Path) -> None:
    left_first = _create_session(
        tmp_path / "left-first",
        session_id="logical-order-session",
        left_db=[-6.0, -5.0],
        right_db=[6.0, 5.0],
        zones=("LEFT", "RIGHT"),
    )
    right_first = _create_session(
        tmp_path / "right-first",
        session_id="logical-order-session",
        left_db=[-6.0, -5.0],
        right_db=[6.0, 5.0],
        zones=("RIGHT", "LEFT"),
    )

    left_fingerprint = _fingerprint(left_first)
    right_fingerprint = _fingerprint(right_first)

    assert [item["path"] for item in left_fingerprint["components"][2:4]] == [
        "samples/left_001.npz",
        "samples/right_001.npz",
    ]
    assert [item["path"] for item in right_fingerprint["components"][2:4]] == [
        "samples/right_001.npz",
        "samples/left_001.npz",
    ]
    assert left_fingerprint["digest_hex"] != right_fingerprint["digest_hex"]


def test_frozen_baseline_fits_every_accepted_sample_and_records_membership(
    tmp_path: Path,
) -> None:
    source = _create_session(
        tmp_path / "datasets",
        session_id="all-accepted-source",
        left_db=[-10.0] * 15 + [10.0] * 5,
        right_db=[10.0] * 15 + [30.0] * 5,
    )
    before = _all_file_bytes(source.directory)

    artifact = _freeze(source)

    classifier = artifact["classifier"]
    development = artifact["source_development_dataset"]
    assert classifier["training_class_means_db"] == pytest.approx(
        {"LEFT": -5.0, "RIGHT": 15.0}, abs=1e-5
    )
    assert classifier["threshold_db"] == pytest.approx(5.0, abs=1e-5)
    assert classifier["lower_feature_zone"] == "LEFT"
    assert classifier["higher_feature_zone"] == "RIGHT"
    assert classifier["tie_rule"] == (
        "feature_value >= threshold_db predicts higher_feature_zone"
    )
    assert development["accepted_sample_count"] == 40
    assert development["accepted_count_by_zone"] == {"LEFT": 20, "RIGHT": 20}
    assert development["all_source_accepted_samples_used"] is True
    assert development["external_samples_used"] is False
    membership = development["training_membership"]
    assert len(membership) == 40
    assert [item["collection_order_index"] for item in membership] == list(
        range(1, 41)
    )
    assert {item["sample_id"] for item in membership} == {
        f"all-accepted-source-{zone.lower()}_{number:03d}"
        for number in range(1, 21)
        for zone in ("LEFT", "RIGHT")
    }
    assert development["fingerprint"] == _fingerprint(source)
    assert _all_file_bytes(source.directory) == before


def test_frozen_baseline_learns_reversed_direction_from_all_source_samples(
    tmp_path: Path,
) -> None:
    source = _create_session(
        tmp_path / "datasets",
        session_id="reverse-direction-source",
        left_db=[8.0, 4.0],
        right_db=[0.0, -4.0],
    )

    classifier = _freeze(source)["classifier"]

    assert classifier["training_class_means_db"] == pytest.approx(
        {"LEFT": 6.0, "RIGHT": -2.0}, abs=1e-5
    )
    assert classifier["threshold_db"] == pytest.approx(2.0, abs=1e-5)
    assert classifier["lower_feature_zone"] == "RIGHT"
    assert classifier["higher_feature_zone"] == "LEFT"


def test_frozen_artifact_is_strict_json_waveform_free_and_has_evidence_boundary(
    tmp_path: Path,
) -> None:
    source = _source_session(tmp_path / "datasets")

    artifact = _freeze(source)

    assert artifact["baseline_schema_version"] == 1
    assert artifact["project_phase"] == "2C"
    assert artifact["created_at_utc"] == _FIXED_TIME.isoformat()
    assert artifact["feature"] == {
        "name": PRIMARY_FEATURE_NAME,
        "units": "dB",
        "definition": (
            "20 * log10(channel_2_peak_absolute / channel_1_peak_absolute)"
        ),
        "definition_version": 1,
        "source_window": "complete retained tap_window array",
    }
    assert artifact["source_development_dataset"]["interaction_context"] == (
        INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED
    )
    assert any(
        "single-session" in limitation
        for limitation in artifact["evidence_limitations"]
    )
    _assert_no_waveform_keys(artifact)
    encoded = json.dumps(artifact, allow_nan=False, sort_keys=True)
    assert '"predictions"' not in encoded
    assert '"confusion_matrix"' not in encoded
    assert '"accuracy"' not in encoded


def test_frozen_baseline_write_load_is_exclusive_and_preserves_source(
    tmp_path: Path,
) -> None:
    source = _source_session(tmp_path / "datasets")
    artifact = _freeze(source)
    before = _all_file_bytes(source.directory)
    destination = tmp_path / "models" / "baseline.json"

    written = _write_baseline(artifact, destination, source)
    original = written.read_bytes()

    assert load_frozen_baseline(written) == artifact
    with pytest.raises(FileExistsError):
        _write_baseline(artifact, destination, source)
    assert written.read_bytes() == original
    assert _all_file_bytes(source.directory) == before
    with pytest.raises(ValueError, match="outside the source dataset"):
        _write_baseline(
            artifact,
            source.directory / "baseline.json",
            source,
        )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "extra_top_level_key",
        "unsupported_schema",
        "wrong_feature",
        "non_midpoint_threshold",
        "wrong_direction",
        "wrong_direction_string",
        "invalid_tie_rule",
        "inconsistent_count",
        "invalid_fingerprint",
        "duplicate_membership",
        "nonfinite_threshold",
        "forbidden_waveform_key",
    ],
)
def test_frozen_baseline_loader_strictly_rejects_inconsistent_artifacts(
    tmp_path: Path, invalid_case: str
) -> None:
    source = _source_session(tmp_path / "datasets")
    artifact = copy.deepcopy(_freeze(source))
    if invalid_case == "extra_top_level_key":
        artifact["unexpected"] = True
    elif invalid_case == "unsupported_schema":
        artifact["baseline_schema_version"] = 999
    elif invalid_case == "wrong_feature":
        artifact["feature"]["name"] = "post_hoc_feature"
    elif invalid_case == "non_midpoint_threshold":
        artifact["classifier"]["threshold_db"] += 1.0
    elif invalid_case == "wrong_direction":
        artifact["classifier"]["lower_feature_zone"] = "RIGHT"
    elif invalid_case == "wrong_direction_string":
        artifact["classifier"]["direction"] = "LEFT and RIGHT are arbitrary"
    elif invalid_case == "invalid_tie_rule":
        artifact["classifier"]["tie_rule"] = "ties predict LEFT"
    elif invalid_case == "inconsistent_count":
        artifact["source_development_dataset"]["accepted_sample_count"] += 1
    elif invalid_case == "invalid_fingerprint":
        artifact["source_development_dataset"]["fingerprint"][
            "digest_hex"
        ] = "not-a-sha256"
    elif invalid_case == "duplicate_membership":
        membership = artifact["source_development_dataset"][
            "training_membership"
        ]
        membership[1]["sample_id"] = membership[0]["sample_id"]
    elif invalid_case == "nonfinite_threshold":
        artifact["classifier"]["threshold_db"] = math.nan
    else:
        artifact["classifier"]["waveform"] = [0.0, 1.0]
    path = tmp_path / f"{invalid_case}.json"
    path.write_text(json.dumps(artifact, allow_nan=True), encoding="utf-8")

    with pytest.raises(frozen_baseline.FrozenBaselineValidationError):
        load_frozen_baseline(path)


def test_frozen_baseline_loader_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(frozen_baseline.FrozenBaselineValidationError):
        load_frozen_baseline(path)


def test_external_evaluation_uses_frozen_model_without_refitting(
    tmp_path: Path, monkeypatch
) -> None:
    source = _create_session(
        tmp_path / "source",
        session_id="development-source",
        left_db=[-6.0, -6.0, -6.0],
        right_db=[6.0, 6.0, 6.0],
    )
    artifact = _freeze(source)
    baseline_path = _write_baseline(
        artifact, tmp_path / "models" / "frozen.json", source
    )
    external = _create_session(
        tmp_path / "external",
        session_id="untouched-same-hand",
        left_db=[-6.0, -2.0, 1.0],
        right_db=[9.0, -1.0, 8.0],
        created_at_utc="2026-08-29T10:00:00+00:00",
    )
    source_before = _all_file_bytes(source.directory)
    external_before = _all_file_bytes(external.directory)
    baseline_before = baseline_path.read_bytes()

    def fail_if_refit(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("external evaluation must not fit or refit a baseline")

    monkeypatch.setattr(
        frozen_baseline, "fit_peak_ratio_baseline", fail_if_refit
    )
    monkeypatch.setattr(
        frozen_baseline, "create_frozen_baseline", fail_if_refit
    )

    report = evaluate_frozen_baseline(
        external.directory,
        baseline_path,
        interaction_context=INTERACTION_CONTEXT_SAME_HAND,
        now_fn=lambda: datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc),
    )

    protocol = report["evaluation_protocol"]
    metrics = report["metrics"]
    assert protocol["frozen_threshold_db"] == pytest.approx(
        artifact["classifier"]["threshold_db"]
    )
    assert protocol["frozen_lower_feature_zone"] == "LEFT"
    assert protocol["frozen_higher_feature_zone"] == "RIGHT"
    assert protocol["external_samples_used_to_fit_threshold"] is False
    assert protocol["threshold_refit_performed"] is False
    assert protocol["direction_relearned"] is False
    assert protocol["feature_selection_performed"] is False
    assert metrics["confusion_matrix"] == {
        "actual_LEFT": {"predicted_LEFT": 2, "predicted_RIGHT": 1},
        "actual_RIGHT": {"predicted_LEFT": 1, "predicted_RIGHT": 2},
    }
    assert metrics["correct_count"] == 4
    assert metrics["total_count"] == 6
    assert metrics["accuracy"] == pytest.approx(4.0 / 6.0)
    assert metrics["per_class"] == {
        "LEFT": {"correct_count": 2, "total_count": 3},
        "RIGHT": {"correct_count": 2, "total_count": 3},
    }
    assert metrics["margin_is_calibrated_probability"] is False
    interval = metrics["accuracy_wilson_95_interval"]
    assert interval["lower_bound"] == pytest.approx(0.29998832134152864)
    assert interval["upper_bound"] == pytest.approx(0.9032306674407831)
    assert report["external_session"]["accepted_sample_count"] == 6
    assert len(report["predictions"]) == 6
    assert report["external_session"]["accepted_count_by_zone"] == {
        "LEFT": 3,
        "RIGHT": 3,
    }
    assert report["precommitment_timing"] == {
        "baseline_created_at_utc": _FIXED_TIME.isoformat(),
        "external_session_created_at_utc": "2026-08-29T10:00:00+00:00",
        "baseline_timestamp_precedes_external_session": True,
        "enforced_as_hard_failure": False,
        "interpretation": (
            "Timestamps are recorded for provenance but are not treated as "
            "trusted proof of experimental ordering. The baseline should be "
            "reviewed and checkpointed before external collection."
        ),
    }

    requested_values = {
        "left_001": -6.0,
        "right_001": 9.0,
        "left_002": -2.0,
        "right_002": -1.0,
        "left_003": 1.0,
        "right_003": 8.0,
    }
    threshold = float(protocol["frozen_threshold_db"])
    for prediction in report["predictions"]:
        sample_suffix = str(prediction["sample_id"]).removeprefix(
            "untouched-same-hand-"
        )
        expected_value = requested_values[sample_suffix]
        assert prediction["feature_name"] == PRIMARY_FEATURE_NAME
        assert prediction["feature_value_db"] == pytest.approx(
            expected_value, abs=1e-5
        )
        assert prediction["threshold_db"] == pytest.approx(threshold)
        assert prediction["absolute_margin_db"] == pytest.approx(
            abs(expected_value - threshold), abs=1e-5
        )
        expected_signed_toward_right = expected_value - threshold
        assert prediction["signed_margin_toward_RIGHT_db"] == pytest.approx(
            expected_signed_toward_right, abs=1e-5
        )
        expected_actual_margin = (
            expected_signed_toward_right
            if prediction["actual_label"] == "RIGHT"
            else -expected_signed_toward_right
        )
        assert prediction["actual_class_margin_db"] == pytest.approx(
            expected_actual_margin, abs=1e-5
        )
        assert (prediction["actual_class_margin_db"] >= 0.0) is bool(
            prediction["correct"]
        )

    evidence = report["evidence_boundary"]
    assert evidence["source_interaction_context"] == (
        INTERACTION_CONTEXT_HAND_LOCATION_CONFOUNDED
    )
    assert evidence["external_interaction_context"] == (
        INTERACTION_CONTEXT_SAME_HAND
    )
    assert evidence["same_hand_controls_previous_hand_variable_more_cleanly"] is True
    assert "No external sample changed" in evidence["no_refit_declaration"]
    assert "not cross-device generalization" in evidence["scope_statement"]
    assert report["report_privacy"] == {
        "analysis_is_local_and_offline": True,
        "microphone_accessed": False,
        "source_dataset_files_modified": False,
        "external_dataset_files_modified": False,
        "baseline_file_modified": False,
        "raw_waveforms_included": False,
        "network_upload_performed": False,
    }
    _assert_no_waveform_keys(report)
    json.dumps(report, allow_nan=False)
    assert _all_file_bytes(source.directory) == source_before
    assert _all_file_bytes(external.directory) == external_before
    assert baseline_path.read_bytes() == baseline_before


def test_external_evaluation_uses_frozen_reverse_direction_and_tie_rule(
    tmp_path: Path,
) -> None:
    double_amplitude_db = 20.0 * math.log10(2.0)
    source = _create_session(
        tmp_path / "source",
        session_id="reverse-source",
        left_db=[double_amplitude_db, double_amplitude_db],
        right_db=[-double_amplitude_db, -double_amplitude_db],
    )
    artifact = _freeze(source)
    baseline_path = _write_baseline(
        artifact, tmp_path / "models" / "reverse.json", source
    )
    external = _create_session(
        tmp_path / "external",
        session_id="reverse-external",
        left_db=[0.0, double_amplitude_db],
        right_db=[-1.0, -double_amplitude_db],
        created_at_utc="2026-08-29T10:00:00+00:00",
    )

    report = evaluate_frozen_baseline(
        external.directory,
        baseline_path,
        interaction_context=INTERACTION_CONTEXT_SAME_HAND,
    )

    protocol = report["evaluation_protocol"]
    assert protocol["frozen_threshold_db"] == pytest.approx(0.0, abs=1e-5)
    assert protocol["frozen_lower_feature_zone"] == "RIGHT"
    assert protocol["frozen_higher_feature_zone"] == "LEFT"
    assert report["metrics"]["correct_count"] == 4
    assert {
        prediction["predicted_label"] for prediction in report["predictions"]
    } == {"LEFT", "RIGHT"}
    tie_prediction = next(
        prediction
        for prediction in report["predictions"]
        if prediction["sample_id"].endswith("left_001")
    )
    assert tie_prediction["feature_value_db"] == pytest.approx(0.0, abs=1e-6)
    assert tie_prediction["predicted_label"] == "LEFT"
    for prediction in report["predictions"]:
        value = float(prediction["feature_value_db"])
        assert prediction["signed_margin_toward_RIGHT_db"] == pytest.approx(
            -value, abs=1e-5
        )


def test_external_summary_and_report_writer_preserve_inputs_and_refuse_overwrite(
    tmp_path: Path,
) -> None:
    source = _create_session(
        tmp_path / "source",
        session_id="summary-source",
        left_db=[-6.0, -5.0],
        right_db=[6.0, 5.0],
    )
    artifact = _freeze(source)
    baseline_path = _write_baseline(
        artifact, tmp_path / "models" / "summary-baseline.json", source
    )
    external = _create_session(
        tmp_path / "external",
        session_id="summary-external",
        left_db=[-4.0, -3.0],
        right_db=[4.0, 3.0],
        created_at_utc="2026-08-29T10:00:00+00:00",
    )
    report = evaluate_frozen_baseline(
        external.directory,
        baseline_path,
        interaction_context=INTERACTION_CONTEXT_SAME_HAND,
        now_fn=lambda: datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc),
    )
    source_before = _all_file_bytes(source.directory)
    external_before = _all_file_bytes(external.directory)
    baseline_before = baseline_path.read_bytes()
    destination = tmp_path / "reports" / "external.json"

    summary = format_external_evaluation_summary(report)
    written = write_external_evaluation_report(
        report,
        destination,
        external_session_path=external.directory,
        baseline_path=baseline_path,
    )
    original_report = written.read_bytes()

    assert "cross-session external evaluation" in summary
    assert "no external refit occurred" in summary
    assert "95% Wilson interval" in summary
    assert "same hand/finger" in summary
    with pytest.raises(FileExistsError):
        write_external_evaluation_report(
            report,
            destination,
            external_session_path=external.directory,
            baseline_path=baseline_path,
        )
    assert written.read_bytes() == original_report
    with pytest.raises(ValueError, match="outside the evaluated dataset"):
        write_external_evaluation_report(
            report,
            external.directory / "report.json",
            external_session_path=external.directory,
            baseline_path=baseline_path,
        )
    with pytest.raises(ValueError, match="must not replace"):
        write_external_evaluation_report(
            report,
            baseline_path,
            external_session_path=external.directory,
            baseline_path=baseline_path,
        )
    assert _all_file_bytes(source.directory) == source_before
    assert _all_file_bytes(external.directory) == external_before
    assert baseline_path.read_bytes() == baseline_before


def test_external_evaluation_rejects_same_source_session_id(tmp_path: Path) -> None:
    source = _create_session(
        tmp_path / "source",
        session_id="shared-session-id",
        left_db=[-6.0, -5.0],
        right_db=[6.0, 5.0],
    )
    baseline_path = _write_baseline(
        _freeze(source), tmp_path / "models" / "same-id.json", source
    )
    different_bytes_same_id = _create_session(
        tmp_path / "external",
        session_id="shared-session-id",
        left_db=[-4.0, -3.0],
        right_db=[4.0, 3.0],
        created_at_utc="2026-08-29T10:00:00+00:00",
    )

    with pytest.raises(frozen_baseline.FrozenBaselineError, match="session ID"):
        evaluate_frozen_baseline(
            different_bytes_same_id.directory,
            baseline_path,
            interaction_context=INTERACTION_CONTEXT_SAME_HAND,
        )


def test_external_evaluation_rejects_source_fingerprint_even_if_id_is_changed(
    tmp_path: Path,
) -> None:
    source = _create_session(
        tmp_path / "source",
        session_id="fingerprint-source",
        left_db=[-6.0, -5.0],
        right_db=[6.0, 5.0],
    )
    artifact = copy.deepcopy(_freeze(source))
    artifact["source_development_dataset"]["session_id"] = "different-id"
    baseline_path = _write_baseline(
        artifact, tmp_path / "models" / "same-fingerprint.json", source
    )

    with pytest.raises(frozen_baseline.FrozenBaselineError, match="fingerprint"):
        evaluate_frozen_baseline(
            source.directory,
            baseline_path,
            interaction_context=INTERACTION_CONTEXT_SAME_HAND,
        )


def test_external_evaluation_validates_baseline_before_external_dataset(
    tmp_path: Path,
) -> None:
    invalid_baseline = tmp_path / "invalid-baseline.json"
    invalid_baseline.write_text("{invalid", encoding="utf-8")

    with pytest.raises(frozen_baseline.FrozenBaselineValidationError):
        evaluate_frozen_baseline(
            tmp_path / "missing-external-session",
            invalid_baseline,
            interaction_context=INTERACTION_CONTEXT_SAME_HAND,
        )


def test_external_evaluation_reports_missing_session_with_valid_baseline(
    tmp_path: Path,
) -> None:
    source = _source_session(tmp_path / "source")
    baseline_path = _write_baseline(
        _freeze(source), tmp_path / "models" / "valid.json", source
    )

    with pytest.raises(frozen_baseline.FrozenBaselineError, match="does not exist"):
        evaluate_frozen_baseline(
            tmp_path / "missing-external-session",
            baseline_path,
            interaction_context=INTERACTION_CONTEXT_SAME_HAND,
        )


@pytest.mark.parametrize(
    "external_created_at",
    ["2026-08-28T11:00:00+00:00", "2026-08-28T12:00:00+00:00"],
)
def test_external_timestamp_order_is_reported_but_not_enforced(
    tmp_path: Path, external_created_at: str
) -> None:
    source = _create_session(
        tmp_path / "source",
        session_id="time-source",
        left_db=[-6.0, -5.0],
        right_db=[6.0, 5.0],
    )
    baseline_path = _write_baseline(
        _freeze(source), tmp_path / "models" / "time.json", source
    )
    premature_external = _create_session(
        tmp_path / "external",
        session_id="premature-external",
        left_db=[-4.0, -3.0],
        right_db=[4.0, 3.0],
        created_at_utc=external_created_at,
    )

    report = evaluate_frozen_baseline(
        premature_external.directory,
        baseline_path,
        interaction_context=INTERACTION_CONTEXT_SAME_HAND,
    )

    timing = report["precommitment_timing"]
    assert timing["baseline_timestamp_precedes_external_session"] is False
    assert timing["enforced_as_hard_failure"] is False


@pytest.mark.parametrize(
    ("correct", "total", "expected_lower", "expected_upper"),
    [
        (20, 20, 0.8388698745050668, 1.0),
        (10, 10, 0.7224598312333834, 1.0),
        (0, 10, 0.0, 0.2775401687666166),
        (18, 20, 0.6989617935882066, 0.9721341060158468),
        (5, 10, 0.2365895936154873, 0.7634104063845126),
    ],
)
def test_wilson_score_interval_matches_known_values(
    correct: int,
    total: int,
    expected_lower: float,
    expected_upper: float,
) -> None:
    interval = wilson_score_interval(correct, total)

    assert interval["method"] == "two-sided Wilson score interval"
    assert interval["confidence_level"] == pytest.approx(0.95)
    assert interval["z_value"] == pytest.approx(1.96)
    assert interval["correct_count"] == correct
    assert interval["total_count"] == total
    assert interval["lower_bound"] == pytest.approx(expected_lower)
    assert interval["upper_bound"] == pytest.approx(expected_upper)


@pytest.mark.parametrize(
    ("correct", "total"),
    [(-1, 10), (11, 10), (0, 0)],
)
def test_wilson_score_interval_rejects_invalid_inputs(
    correct: int, total: int
) -> None:
    with pytest.raises(ValueError):
        wilson_score_interval(correct, total)
