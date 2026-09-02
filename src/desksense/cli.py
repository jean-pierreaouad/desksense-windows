"""Command-line interface for the DeskSense microphone feasibility probe."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from desksense.analysis import (
    DatasetAnalysisError,
    INTERACTION_CONTEXT_CHOICES,
    analyze_dataset,
    format_analysis_summary,
    write_analysis_report,
)
from desksense.characterization import DEFAULT_CHARACTERIZATION_SECONDS
from desksense.collection import (
    DEFAULT_SAMPLES_PER_ZONE,
    DatasetCollectionError,
    run_guided_collection,
)
from desksense.diagnostics import (
    DEFAULT_RECORDING_SECONDS,
    build_backend_error_report,
    collect_characterization_report,
    collect_diagnostic_report,
    describe_audio_error,
    write_json_report,
)
from desksense.frozen_baseline import (
    FrozenBaselineError,
    create_frozen_baseline,
    evaluate_frozen_baseline,
    format_external_evaluation_summary,
    format_frozen_baseline_summary,
    write_external_evaluation_report,
    write_frozen_baseline,
)
from desksense.realtime import (
    LiveSensingEvent,
    RealtimeSensingError,
    run_live_sensing,
)
from desksense.robustness import (
    RobustnessError,
    format_robustness_summary,
    replay_robustness_dataset,
    run_guided_robustness_collection,
    write_robustness_report,
)
from desksense.robustness_dataset import RobustnessDatasetError
from desksense.tapness import (
    TapnessBaselineError,
    create_tapness_baseline,
    format_tapness_baseline_summary,
    write_tapness_baseline,
)

_AUTO_REPORT = object()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="desksense-diagnose",
        description=(
            "Inspect Windows audio inputs, characterize microphone channels, or "
            "collect, analyze, freeze, externally evaluate, or sense taps with "
            "a frozen local baseline."
        ),
    )
    parser.add_argument(
        "--device",
        type=_nonnegative_device_index,
        metavar="INDEX",
        help=(
            "input device index to select (required for collection modes and "
            "--sense; "
            "diagnostic/characterization modes otherwise use the system default; "
            "not used by offline workflows)"
        ),
    )
    capture_mode = parser.add_mutually_exclusive_group()
    capture_mode.add_argument(
        "--record",
        action="store_true",
        help=(
            f"record about {DEFAULT_RECORDING_SECONDS:g} seconds and report "
            "per-channel levels; raw audio is not saved"
        ),
    )
    capture_mode.add_argument(
        "--characterize",
        action="store_true",
        help=(
            f"record about {DEFAULT_CHARACTERIZATION_SECONDS:g} seconds and "
            "perform exploratory channel-independence analysis; raw audio is "
            "not saved"
        ),
    )
    capture_mode.add_argument(
        "--collect-dataset",
        action="store_true",
        help=(
            "run guided alternating LEFT/RIGHT tap collection and retain accepted "
            "float32 waveforms locally under datasets/"
        ),
    )
    capture_mode.add_argument(
        "--collect-robustness-dataset",
        action="store_true",
        help=(
            "run the explicit Phase 3B.0 guided positive/negative evidence "
            "session and retain every structurally valid float32 capture locally"
        ),
    )
    capture_mode.add_argument(
        "--replay-robustness-dataset",
        type=Path,
        metavar="SESSION",
        help=(
            "validate and replay one Phase 3 robustness session through the "
            "current versioned Stage 1 detector entirely offline"
        ),
    )
    capture_mode.add_argument(
        "--analyze-dataset",
        type=Path,
        metavar="SESSION",
        help=(
            "validate and analyze one Phase 2A session entirely offline; a JSON "
            "analysis report is always written"
        ),
    )
    capture_mode.add_argument(
        "--freeze-baseline",
        type=Path,
        metavar="DEVELOPMENT_SESSION",
        help=(
            "fit the fixed LEFT/RIGHT midpoint baseline from every accepted "
            "sample in one validated development session; entirely offline"
        ),
    )
    capture_mode.add_argument(
        "--fit-tapness-baseline",
        type=Path,
        metavar="DEVELOPMENT_SESSION",
        help=(
            "fit the fixed Phase 3B.2 TAP/NON_TAP logistic baseline from one "
            "validated robustness development session; entirely offline"
        ),
    )
    capture_mode.add_argument(
        "--evaluate-frozen-baseline",
        type=Path,
        metavar="EXTERNAL_SESSION",
        help=(
            "apply an existing frozen baseline unchanged to every accepted "
            "sample in a different validated session; entirely offline"
        ),
    )
    capture_mode.add_argument(
        "--sense",
        action="store_true",
        help=(
            "continuously generate candidates from an explicit input device, "
            "apply frozen tapness validation, then apply the unchanged frozen "
            "LEFT/RIGHT baseline; no audio is saved"
        ),
    )
    parser.add_argument(
        "--sense-diagnostics",
        action="store_true",
        help=(
            "with --sense, print candidate-start evidence and low-rate detector "
            "decision summaries; no audio is saved"
        ),
    )
    parser.add_argument(
        "--samples-per-zone",
        type=_positive_integer,
        default=None,
        metavar="COUNT",
        help=(
            "accepted samples to collect for each zone in dataset mode "
            f"(default: {DEFAULT_SAMPLES_PER_ZONE})"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "local dataset directory used by --collect-dataset or "
            "--collect-robustness-dataset (default: datasets)"
        ),
    )
    parser.add_argument(
        "--interaction-context",
        choices=INTERACTION_CONTEXT_CHOICES,
        default=None,
        metavar="CONTEXT",
        help=(
            "required with offline dataset analysis, baseline freezing, and "
            "external evaluation because tapping-hand context is not stored in "
            "Phase 2A artifacts"
        ),
    )
    parser.add_argument(
        "--save-baseline",
        type=Path,
        metavar="PATH",
        help=(
            "required with --freeze-baseline; exclusively write the compact "
            "tracked-eligible frozen JSON artifact"
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        metavar="PATH",
        help=(
            "required with --evaluate-frozen-baseline and --sense, and optional "
            "with --replay-robustness-dataset; baseline is never modified"
        ),
    )
    parser.add_argument(
        "--tapness-baseline",
        type=Path,
        metavar="PATH",
        help=(
            "frozen TAP/NON_TAP artifact required with --sense and optional "
            "with --replay-robustness-dataset for Stage 2 evaluation"
        ),
    )
    parser.add_argument(
        "--save-tapness-baseline",
        type=Path,
        metavar="PATH",
        help=(
            "required with --fit-tapness-baseline; exclusively write the "
            "compact versioned tapness artifact"
        ),
    )
    parser.add_argument(
        "--save-report",
        nargs="?",
        const=_AUTO_REPORT,
        type=Path,
        metavar="PATH",
        help=(
            "save a JSON report; omit PATH to create a timestamped file under "
            "reports/ (dataset analysis, robustness replay, and external "
            "evaluation do this even when the option is omitted)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sense_diagnostics and not args.sense:
        parser.error("--sense-diagnostics requires --sense")

    if args.save_baseline is not None and args.freeze_baseline is None:
        parser.error("--save-baseline requires --freeze-baseline SESSION")
    if (
        args.save_tapness_baseline is not None
        and args.fit_tapness_baseline is None
    ):
        parser.error(
            "--save-tapness-baseline requires --fit-tapness-baseline SESSION"
        )
    if (
        args.tapness_baseline is not None
        and not args.sense
        and args.replay_robustness_dataset is None
    ):
        parser.error(
            "--tapness-baseline requires --sense or "
            "--replay-robustness-dataset SESSION"
        )
    if (
        args.baseline is not None
        and args.evaluate_frozen_baseline is None
        and not args.sense
        and args.replay_robustness_dataset is None
    ):
        parser.error(
            "--baseline requires --evaluate-frozen-baseline SESSION, "
            "--replay-robustness-dataset SESSION, or --sense"
        )

    if args.freeze_baseline is not None:
        if args.device is not None:
            parser.error("--device cannot be combined with --freeze-baseline")
        if args.samples_per_zone is not None or args.dataset_root is not None:
            parser.error(
                "--samples-per-zone and --dataset-root cannot be combined with "
                "--freeze-baseline"
            )
        if args.interaction_context is None:
            parser.error(
                "--freeze-baseline requires --interaction-context "
                "{hand-location-confounded,same-hand}"
            )
        if args.save_baseline is None:
            parser.error("--freeze-baseline requires --save-baseline PATH")
        if args.save_report is not None:
            parser.error("--save-report cannot be combined with --freeze-baseline")
        return _run_freeze_baseline_cli(args)

    if args.fit_tapness_baseline is not None:
        if args.device is not None:
            parser.error("--device cannot be combined with --fit-tapness-baseline")
        if args.save_tapness_baseline is None:
            parser.error(
                "--fit-tapness-baseline requires --save-tapness-baseline PATH"
            )
        if any(
            value is not None
            for value in (
                args.baseline,
                args.tapness_baseline,
                args.samples_per_zone,
                args.dataset_root,
                args.interaction_context,
                args.save_report,
            )
        ):
            parser.error(
                "--fit-tapness-baseline only accepts its development session "
                "and --save-tapness-baseline"
            )
        return _run_fit_tapness_baseline_cli(args)

    if args.evaluate_frozen_baseline is not None:
        if args.device is not None:
            parser.error(
                "--device cannot be combined with --evaluate-frozen-baseline"
            )
        if args.samples_per_zone is not None or args.dataset_root is not None:
            parser.error(
                "--samples-per-zone and --dataset-root cannot be combined with "
                "--evaluate-frozen-baseline"
            )
        if args.interaction_context is None:
            parser.error(
                "--evaluate-frozen-baseline requires --interaction-context "
                "{hand-location-confounded,same-hand}"
            )
        if args.baseline is None:
            parser.error("--evaluate-frozen-baseline requires --baseline PATH")
        return _run_external_evaluation_cli(args)

    if args.analyze_dataset is not None:
        if args.device is not None:
            parser.error("--device cannot be combined with --analyze-dataset")
        if args.samples_per_zone is not None or args.dataset_root is not None:
            parser.error(
                "--samples-per-zone and --dataset-root cannot be combined with "
                "--analyze-dataset"
            )
        if args.interaction_context is None:
            parser.error(
                "--analyze-dataset requires --interaction-context "
                "{hand-location-confounded,same-hand}"
            )
        return _run_dataset_analysis_cli(args)

    if args.replay_robustness_dataset is not None:
        if args.device is not None:
            parser.error(
                "--device cannot be combined with --replay-robustness-dataset"
            )
        if args.samples_per_zone is not None or args.dataset_root is not None:
            parser.error(
                "--samples-per-zone and --dataset-root cannot be combined with "
                "--replay-robustness-dataset"
            )
        if args.interaction_context is not None:
            parser.error(
                "--interaction-context cannot be combined with "
                "--replay-robustness-dataset"
            )
        return _run_robustness_replay_cli(args)

    if args.sense:
        if args.device is None:
            parser.error("--sense requires --device INDEX")
        if args.baseline is None:
            parser.error("--sense requires --baseline PATH")
        if args.tapness_baseline is None:
            parser.error("--sense requires --tapness-baseline PATH")
        if args.samples_per_zone is not None or args.dataset_root is not None:
            parser.error(
                "--samples-per-zone and --dataset-root cannot be combined with "
                "--sense"
            )
        if args.interaction_context is not None:
            parser.error("--interaction-context cannot be combined with --sense")
        if args.save_report is not None:
            parser.error("--save-report cannot be combined with --sense")
        return _run_sense_cli(args)

    if args.interaction_context is not None:
        parser.error(
            "--interaction-context requires an offline dataset analysis, "
            "freeze, or external-evaluation mode"
        )

    if args.collect_dataset:
        if args.device is None:
            parser.error("--collect-dataset requires --device INDEX")
        if args.save_report is not None:
            parser.error(
                "--save-report cannot be combined with --collect-dataset; "
                "dataset metadata is stored inside the session directory"
            )
        if args.samples_per_zone is None:
            args.samples_per_zone = DEFAULT_SAMPLES_PER_ZONE
        if args.dataset_root is None:
            args.dataset_root = Path("datasets")
        return _run_dataset_collection_cli(args)
    if args.collect_robustness_dataset:
        if args.device is None:
            parser.error("--collect-robustness-dataset requires --device INDEX")
        if args.samples_per_zone is not None:
            parser.error(
                "--samples-per-zone cannot be combined with "
                "--collect-robustness-dataset"
            )
        if args.baseline is not None:
            parser.error(
                "--baseline cannot be combined with --collect-robustness-dataset"
            )
        if args.save_report is not None:
            parser.error(
                "--save-report cannot be combined with "
                "--collect-robustness-dataset; evidence metadata is stored in "
                "the session directory"
            )
        if args.dataset_root is None:
            args.dataset_root = Path("datasets")
        return _run_robustness_collection_cli(args)
    if args.samples_per_zone is not None or args.dataset_root is not None:
        parser.error(
            "--samples-per-zone requires --collect-dataset; --dataset-root "
            "requires a collection mode"
        )

    try:
        audio_backend = _load_audio_backend()
    except Exception as error:
        report = build_backend_error_report(
            error,
            record_requested=args.record,
            characterize_requested=args.characterize,
        )
    else:
        try:
            if args.characterize:
                report = collect_characterization_report(
                    audio_backend,
                    selected_device_index=args.device,
                )
            else:
                report = collect_diagnostic_report(
                    audio_backend,
                    selected_device_index=args.device,
                    record=args.record,
                )
        except KeyboardInterrupt:
            print("\nDeskSense diagnostic interrupted.", file=sys.stderr)
            return 130
        except Exception as error:
            report = build_backend_error_report(
                error,
                record_requested=args.record,
                characterize_requested=args.characterize,
            )

    print(format_report(report))
    exit_code = 0 if report["status"] == "ok" else 1

    if args.save_report is not None:
        try:
            report_path = _report_path(
                args.save_report,
                report["generated_at_utc"],
                report_type=report.get("report_type", "hardware_diagnostic"),
            )
            saved_path = write_json_report(report, report_path)
        except (OSError, TypeError, ValueError) as error:
            print(f"Could not save JSON report: {error}", file=sys.stderr)
            exit_code = 1
        else:
            print(f"\nJSON report saved to: {saved_path}")

    return exit_code


def _run_freeze_baseline_cli(args: argparse.Namespace) -> int:
    """Fit and persist a Phase 2C baseline without loading audio hardware."""

    try:
        artifact = create_frozen_baseline(
            args.freeze_baseline,
            interaction_context=args.interaction_context,
        )
        saved_path = write_frozen_baseline(
            artifact,
            args.save_baseline,
            source_session_path=args.freeze_baseline,
        )
    except KeyboardInterrupt:
        print("\nDeskSense baseline freezing interrupted.", file=sys.stderr)
        return 130
    except (FrozenBaselineError, OSError, TypeError, ValueError) as error:
        print(f"Frozen baseline creation failed: {error}", file=sys.stderr)
        return 1

    print(format_frozen_baseline_summary(artifact))
    print(f"\nFrozen baseline saved to: {saved_path}")
    return 0


def _run_fit_tapness_baseline_cli(args: argparse.Namespace) -> int:
    """Fit Session A-style tapness evidence without loading audio hardware."""

    try:
        artifact = create_tapness_baseline(args.fit_tapness_baseline)
        saved_path = write_tapness_baseline(
            artifact, args.save_tapness_baseline
        )
    except KeyboardInterrupt:
        print("\nDeskSense tapness fitting interrupted.", file=sys.stderr)
        return 130
    except (
        TapnessBaselineError,
        RobustnessDatasetError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Tapness baseline creation failed: {error}", file=sys.stderr)
        return 1
    print(format_tapness_baseline_summary(artifact))
    print(f"\nFrozen tapness baseline saved to: {saved_path}")
    return 0


def _run_external_evaluation_cli(args: argparse.Namespace) -> int:
    """Evaluate a frozen model offline without fitting or loading audio."""

    try:
        report = evaluate_frozen_baseline(
            args.evaluate_frozen_baseline,
            args.baseline,
            interaction_context=args.interaction_context,
        )
    except KeyboardInterrupt:
        print("\nDeskSense external evaluation interrupted.", file=sys.stderr)
        return 130
    except (FrozenBaselineError, OSError, TypeError, ValueError) as error:
        print(f"External evaluation failed: {error}", file=sys.stderr)
        return 1

    print(format_external_evaluation_summary(report))
    requested_path = _AUTO_REPORT if args.save_report is None else args.save_report
    try:
        report_path = _external_evaluation_report_path(
            requested_path, report["generated_at_utc"]
        )
        saved_path = write_external_evaluation_report(
            report,
            report_path,
            external_session_path=args.evaluate_frozen_baseline,
            baseline_path=args.baseline,
        )
    except (FrozenBaselineError, OSError, TypeError, ValueError) as error:
        print(f"Could not save JSON external report: {error}", file=sys.stderr)
        return 1
    print(f"\nJSON external evaluation report saved to: {saved_path}")
    return 0


def _run_dataset_analysis_cli(args: argparse.Namespace) -> int:
    """Run the offline Phase 2B path without loading an audio backend."""

    try:
        report = analyze_dataset(
            args.analyze_dataset,
            interaction_context=args.interaction_context,
        )
    except KeyboardInterrupt:
        print("\nDeskSense dataset analysis interrupted.", file=sys.stderr)
        return 130
    except (DatasetAnalysisError, OSError, TypeError, ValueError) as error:
        print(f"Dataset analysis failed: {error}", file=sys.stderr)
        return 1

    print(format_analysis_summary(report))
    requested_path = (
        _AUTO_REPORT if args.save_report is None else args.save_report
    )
    try:
        report_path = _analysis_report_path(
            requested_path, report["generated_at_utc"]
        )
        saved_path = write_analysis_report(report, report_path)
    except (OSError, TypeError, ValueError) as error:
        print(f"Could not save JSON analysis report: {error}", file=sys.stderr)
        return 1
    print(f"\nJSON analysis report saved to: {saved_path}")
    return 0


def _run_dataset_collection_cli(args: argparse.Namespace) -> int:
    """Run explicit waveform collection without changing diagnostic behavior."""

    try:
        audio_backend = _load_audio_backend()
    except Exception as error:
        details = describe_audio_error(error)
        print(
            f"Could not load the audio backend: {details['message']}",
            file=sys.stderr,
        )
        if details["category"] == "permission_denied":
            _print_microphone_permission_guidance()
        return 1

    try:
        run_guided_collection(
            audio_backend,
            device_index=args.device,
            samples_per_zone=args.samples_per_zone,
            dataset_root=args.dataset_root,
        )
    except KeyboardInterrupt:
        print(
            "\nDeskSense dataset collection interrupted; previously accepted "
            "samples remain in the session directory.",
            file=sys.stderr,
        )
        return 130
    except EOFError:
        print(
            "\nDataset collection input closed; previously accepted samples "
            "remain in the session directory.",
            file=sys.stderr,
        )
        return 1
    except DatasetCollectionError as error:
        print(f"Dataset collection failed: {error}", file=sys.stderr)
        if error.category == "permission_denied":
            _print_microphone_permission_guidance()
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(f"Dataset collection failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_robustness_collection_cli(args: argparse.Namespace) -> int:
    """Run explicit local robustness capture with lazy backend loading."""

    try:
        audio_backend = _load_audio_backend()
    except Exception as error:
        details = describe_audio_error(error)
        print(
            f"Could not load the audio backend: {details['message']}",
            file=sys.stderr,
        )
        if details["category"] == "permission_denied":
            _print_microphone_permission_guidance()
        return 1
    try:
        run_guided_robustness_collection(
            audio_backend,
            device_index=args.device,
            dataset_root=args.dataset_root,
        )
    except KeyboardInterrupt:
        print(
            "\nRobustness collection interrupted; already saved evidence remains "
            "in the session directory.",
            file=sys.stderr,
        )
        return 130
    except EOFError:
        print(
            "\nRobustness collection input closed; already saved evidence remains "
            "in the session directory.",
            file=sys.stderr,
        )
        return 1
    except RobustnessError as error:
        print(f"Robustness collection failed: {error}", file=sys.stderr)
        if error.category == "permission_denied":
            _print_microphone_permission_guidance()
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(f"Robustness collection failed: {error}", file=sys.stderr)
        return 1
    return 0


def _run_robustness_replay_cli(args: argparse.Namespace) -> int:
    """Replay robustness evidence without loading or initializing audio."""

    try:
        report = replay_robustness_dataset(
            args.replay_robustness_dataset,
            baseline_path=args.baseline,
            tapness_baseline_path=args.tapness_baseline,
        )
    except KeyboardInterrupt:
        print("\nRobustness replay interrupted.", file=sys.stderr)
        return 130
    except (
        RobustnessError,
        RobustnessDatasetError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Robustness replay failed: {error}", file=sys.stderr)
        return 1
    print(format_robustness_summary(report))
    requested_path = _AUTO_REPORT if args.save_report is None else args.save_report
    try:
        report_path = _robustness_report_path(
            requested_path, report["generated_at_utc"]
        )
        saved_path = write_robustness_report(report, report_path)
    except (OSError, TypeError, ValueError) as error:
        print(f"Could not save JSON robustness report: {error}", file=sys.stderr)
        return 1
    print(f"\nJSON robustness replay report saved to: {saved_path}")
    return 0


def _run_sense_cli(args: argparse.Namespace) -> int:
    """Run the injected live adapter while preserving lazy backend loading."""

    try:
        audio_backend = _load_audio_backend()
    except Exception as error:
        details = describe_audio_error(error)
        print(
            f"Could not load the audio backend: {details['message']}",
            file=sys.stderr,
        )
        if details["category"] == "permission_denied":
            _print_microphone_permission_guidance()
        return 1

    try:
        run_live_sensing(
            audio_backend,
            device_index=args.device,
            baseline_path=args.baseline,
            tapness_baseline_path=args.tapness_baseline,
            event_handler=_print_live_sensing_event,
            diagnostics_enabled=args.sense_diagnostics,
        )
    except KeyboardInterrupt:
        print("\nDeskSense live sensing interrupted.", file=sys.stderr)
        return 130
    except RealtimeSensingError as error:
        print(f"Live sensing failed: {error}", file=sys.stderr)
        if error.category == "permission_denied":
            _print_microphone_permission_guidance()
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(f"Live sensing failed: {error}", file=sys.stderr)
        return 1
    return 0


def format_live_sensing_event(event: LiveSensingEvent) -> str:
    """Render one structured realtime event without per-audio-block noise."""

    if event.event_type == "startup":
        details = event.details
        latency = _format_optional_milliseconds(
            event.timing.get("stream_reported_latency_seconds")
        )
        return "\n".join(
            [
                "DeskSense live LEFT/RIGHT sensing",
                (
                    f"Device: {details['device_index']}: "
                    f"{details['endpoint_name']}"
                ),
                f"Host API: {details['host_api_name']}",
                (
                    "Stream: "
                    f"{_format_rate(float(details['sample_rate_hz']))}, "
                    f"{details['channel_count']} channel(s), {details['dtype']}, "
                    f"reported latency {latency}"
                ),
                (
                    f"Baseline: {details['baseline_path']} "
                    f"(source session {details['baseline_source_session_id']})"
                ),
                (
                    "Tapness baseline: "
                    f"{details.get('tapness_baseline_path') or 'not supplied'}"
                ),
                (
                    f"Frozen threshold: {float(details['threshold_db']):+.6f} dB; "
                    f"{details['direction']}"
                ),
                (
                    "Learning room noise for "
                    f"{float(details['startup_learning_seconds']):g} s..."
                ),
            ]
        )

    if event.event_type == "state" and event.status == "armed":
        return f"Armed (detector epoch {_format_optional_index(event.stream_epoch)})."

    if event.event_type == "discontinuity":
        reasons = _format_reason_codes(event.rejection_reasons)
        dropped_callbacks = int(
            event.details.get("dropped_callback_count_before", 0)
        )
        dropped_frames = int(event.details.get("dropped_frame_count_before", 0))
        return (
            "Audio discontinuity: "
            f"{reasons}; dropped callbacks={dropped_callbacks}, "
            f"dropped frames={dropped_frames}. Detector reset to learning "
            f"(epoch {_format_optional_index(event.stream_epoch)})."
        )

    if event.event_type == "diagnostic" and event.status == "candidate_started":
        gate = event.details.get("gate_block", {})
        route = event.details.get("candidate_start_route", "ordinary")
        return (
            "Diagnostic candidate start: "
            f"route={route}; "
            f"epoch={_format_optional_index(event.stream_epoch)}; "
            f"onset={_format_optional_index(event.onset_frame_index)}; "
            f"block=[{gate.get('block_start_frame_index', '?')}:"
            f"{gate.get('block_end_frame_index_exclusive', '?')}); "
            f"RMS={_format_diagnostic_float(gate.get('block_rms'))} "
            f"(threshold={_format_diagnostic_float(gate.get('required_rms_threshold'))}, "
            f"ratio={_format_diagnostic_float(gate.get('rms_ratio'))}); "
            f"peak={_format_diagnostic_float(gate.get('block_peak_absolute'))} "
            f"(threshold={_format_diagnostic_float(gate.get('required_peak_threshold'))}, "
            f"ratio={_format_diagnostic_float(gate.get('peak_ratio'))}); "
            f"crest={_format_diagnostic_float(gate.get('block_crest_factor'))} "
            f"(threshold={_format_diagnostic_float(gate.get('required_crest_threshold'))}, "
            f"ratio={_format_diagnostic_float(gate.get('crest_ratio'))}); "
            f"floor={_format_diagnostic_float(gate.get('learned_noise_floor_rms'))}."
        )

    if event.event_type == "diagnostic" and event.status == "summary":
        details = event.details
        counts = details.get("interval_counters", {})
        closest = details.get("closest_armed_block")
        transport = details.get("transport", {})
        closest_text = "closest armed block=none"
        if isinstance(closest, dict):
            closest_text = (
                "closest armed block="
                f"[{closest.get('block_start_frame_index', '?')}:"
                f"{closest.get('block_end_frame_index_exclusive', '?')}); "
                f"score={_format_diagnostic_float(closest.get('all_gates_score'))}; "
                f"ratios RMS/peak/crest="
                f"{_format_diagnostic_float(closest.get('rms_ratio'))}/"
                f"{_format_diagnostic_float(closest.get('peak_ratio'))}/"
                f"{_format_diagnostic_float(closest.get('crest_ratio'))}"
            )
        return "\n".join(
            [
                (
                    "Sense diagnostics: "
                    f"epoch={_format_optional_index(event.stream_epoch)}; "
                    f"state={details.get('detector_state', 'unknown')}; "
                    f"floor={_format_diagnostic_float(details.get('learned_noise_floor_rms'))}"
                ),
                (
                    "  interval blocks: "
                    f"total={counts.get('fixed_blocks_processed_total', 0)}, "
                    f"learning={counts.get('learning_suppressed_blocks', 0)}, "
                    f"armed={counts.get('armed_evaluated_blocks', 0)}, "
                    f"collecting={counts.get('collecting_blocks', 0)}, "
                    f"refractory={counts.get('refractory_suppressed_blocks', 0)}"
                ),
                (
                    "  gate pass/fail: "
                    f"RMS={counts.get('rms_pass_count', 0)}/"
                    f"{counts.get('rms_fail_count', 0)}, "
                    f"peak={counts.get('peak_pass_count', 0)}/"
                    f"{counts.get('peak_fail_count', 0)}, "
                    f"crest={counts.get('crest_pass_count', 0)}/"
                    f"{counts.get('crest_fail_count', 0)}; "
                    f"all-pass={counts.get('all_gates_pass_count', 0)}, "
                    f"starts={counts.get('onset_candidates_started', 0)}, "
                    f"detected={counts.get('completed_detections', 0)}, "
                    f"rejected={counts.get('completed_rejections', 0)}"
                ),
                f"  {closest_text}",
                (
                    "  transport: "
                    f"callbacks={transport.get('callback_count', 0)}, "
                    f"frames={transport.get('callback_frames', 0)}, "
                    f"queue high-water={transport.get('queue_high_water_mark', 0)}, "
                    f"dropped callbacks={transport.get('dropped_callback_count', 0)}, "
                    f"dropped frames={transport.get('dropped_frame_count', 0)}, "
                    f"discontinuities={transport.get('discontinuity_count', 0)}, "
                    f"PortAudio status={_format_reason_codes(transport.get('current_packet_portaudio_status_flags', ())) if transport.get('current_packet_portaudio_status_flags') else 'none'}"
                ),
            ]
        )
    if event.event_type == "detection" and event.status == "detected":
        timing_parts = _format_live_timing(event.timing)
        timing_suffix = f"; {timing_parts}" if timing_parts else ""
        tapness = event.details.get("tapness_inference", {})
        tapness_suffix = ""
        if tapness:
            tapness_suffix = (
                "; tapness output="
                f"{float(tapness['uncalibrated_model_output']):.6f}; "
                f"tapness margin={float(tapness['model_margin']):+.6f}"
            )
        return (
            f"Tap: {event.predicted_zone}; "
            f"peak ratio={float(event.feature_value_db):+.6f} dB; "
            f"threshold={float(event.threshold_db):+.6f} dB; "
            f"margin={float(event.absolute_margin_db):.6f} dB; "
            f"epoch={_format_optional_index(event.stream_epoch)}; "
            f"onset={_format_optional_index(event.onset_frame_index)}; "
            f"center={_format_optional_index(event.center_frame_index)}"
            f"{tapness_suffix}"
            f"{timing_suffix}"
        )

    if event.event_type == "rejection" or event.status == "rejected":
        tapness = event.details.get("tapness_inference", {})
        tapness_suffix = ""
        if tapness:
            tapness_suffix = (
                "; tapness output="
                f"{float(tapness['uncalibrated_model_output']):.6f}; "
                f"tapness margin={float(tapness['model_margin']):+.6f}"
            )
        return (
            "Tap rejected: "
            f"{_format_reason_codes(event.rejection_reasons)}; "
            f"epoch={_format_optional_index(event.stream_epoch)}; "
            f"onset={_format_optional_index(event.onset_frame_index)}; "
            f"center={_format_optional_index(event.center_frame_index)}"
            f"{tapness_suffix}"
        )

    return event.message


def _print_live_sensing_event(event: LiveSensingEvent) -> None:
    print(format_live_sensing_event(event), flush=True)


def _format_live_timing(timing: dict[str, Any] | Any) -> str:
    parts: list[str] = []
    queue_dwell = _format_optional_milliseconds(
        timing.get("queue_dwell_seconds"), unavailable=None
    )
    if queue_dwell is not None:
        parts.append(f"queue dwell={queue_dwell}")
    detector_latency = _format_optional_milliseconds(
        timing.get("detector_latency_seconds"), unavailable=None
    )
    if detector_latency is not None:
        parts.append(f"detector lookahead={detector_latency}")
    return "; ".join(parts)


def _format_diagnostic_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return "unavailable"
    if not math.isfinite(numeric):
        return "unavailable"
    return f"{numeric:.6g}"


def _format_optional_milliseconds(
    seconds: Any,
    *,
    unavailable: str | None = "unavailable",
) -> str | None:
    if seconds is None:
        return unavailable
    try:
        value = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return unavailable
    return f"{value * 1_000.0:.2f} ms"


def _format_reason_codes(reasons: Sequence[str]) -> str:
    return ", ".join(reasons) if reasons else "unspecified reason"


def _format_optional_index(value: int | None) -> str:
    return "unavailable" if value is None else str(value)


def format_report(report: dict[str, Any]) -> str:
    """Render a compact, human-readable diagnostic summary."""

    system = report["system"]
    operating_system = system["operating_system"]
    python = system["python"]
    audio = report["audio"]

    heading = (
        "DeskSense exploratory microphone characterization"
        if report.get("report_type") == "microphone_characterization"
        else "DeskSense microphone feasibility probe"
    )
    lines = [
        heading,
        (
            "Operating system: "
            f"{operating_system['name']} {operating_system['release']} "
            f"({operating_system['machine']})"
        ),
        (
            f"Python: {python['implementation']} {python['version']} "
            f"({python['architecture_bits']}-bit)"
        ),
    ]
    if operating_system["name"] != "Windows":
        lines.append("Warning: this milestone targets Windows; results may differ.")

    backend = audio["backend"]
    backend_line = f"Audio backend: {backend['library']} {backend['version']}"
    if backend.get("portaudio_version_text"):
        backend_line += f" / {backend['portaudio_version_text']}"
    lines.extend(["", backend_line])

    if audio["host_apis"]:
        host_names = ", ".join(host_api["name"] for host_api in audio["host_apis"])
        lines.append(f"Available host APIs: {host_names}")

    input_devices = audio["input_devices"]
    lines.extend(["", f"Audio input devices: {len(input_devices)}"])
    for device in input_devices:
        markers = []
        if device["is_default"]:
            markers.append("default")
        if device["is_selected"]:
            markers.append("selected")
        marker_text = f" [{', '.join(markers)}]" if markers else ""
        lines.append(f"  {device['index']}: {device['name']}{marker_text}")
        lines.append(f"     Host API: {device['host_api']['name']}")
        lines.append(
            "     Max input channels: "
            f"{device['max_input_channels']} ({device['channel_layout']})"
        )
        lines.append(
            "     Default sample rate: "
            f"{_format_rate(device['default_sample_rate_hz'])}"
        )
        for support in device["sample_rate_support"]:
            if support["supported"]:
                lines.append(
                    f"     {support['sample_rate_hz']} Hz: supported at "
                    f"{support['supported_input_channels']} channel(s)"
                )
            else:
                error = support.get("error", {})
                if support["supported"] is None:
                    lines.append(
                        f"     {support['sample_rate_hz']} Hz: could not check "
                        f"({error.get('message', 'unknown audio error')})"
                    )
                    continue
                lines.append(
                    f"     {support['sample_rate_hz']} Hz: not supported "
                    f"(checked 1-{support['max_input_channels_checked']} "
                    "channel(s))"
                )

    default_device = _find_device(
        input_devices, audio["default_input_device_index"]
    )
    selected_device = _find_device(
        input_devices, audio["selected_input_device_index"]
    )
    lines.extend(
        [
            "",
            "Default input: " + _format_device(default_device),
            "Selected input: " + _format_device(selected_device),
        ]
    )

    recording = audio["recording"]
    characterization = audio.get("characterization")
    permission_guidance_shown = False
    lines.append("")
    if recording["status"] == "not_requested":
        if characterization is None:
            lines.append("Recording test: not requested")
    elif recording["status"] == "ok":
        lines.append(
            "Recording test: completed "
            f"({recording['captured_duration_seconds']:.2f} s, "
            f"{_format_rate(recording['sample_rate_hz'])}, "
            f"{recording['channels']} channel(s))"
        )
        for level in recording["channel_levels"]:
            peak_dbfs = _format_dbfs(level["peak_dbfs"])
            rms_dbfs = _format_dbfs(level["rms_dbfs"])
            lines.append(
                f"  Channel {level['channel']}: "
                f"peak absolute={level['peak_absolute']:.6f} ({peak_dbfs}), "
                f"RMS={level['rms']:.6f} ({rms_dbfs})"
            )
        lines.append("Raw audio was discarded and was not written to disk.")
    else:
        error = recording.get("error", {})
        lines.append(f"Recording test: {recording['status'].replace('_', ' ')}")
        if error.get("message"):
            lines.append(f"Reason: {error['message']}")
        if error.get("category") == "permission_denied":
            lines.append(
                "Enable microphone access in Windows Settings > Privacy & security "
                "> Microphone, then retry."
            )
            permission_guidance_shown = True

    if characterization is not None:
        lines.append("")
        permission_guidance_shown = _append_characterization(
            lines, characterization
        ) or permission_guidance_shown

    for error in audio["errors"]:
        lines.append(f"Error: {error['message']}")
    for warning in audio["warnings"]:
        lines.append(f"Warning: {warning}")
    if not permission_guidance_shown and any(
        error.get("category") == "permission_denied"
        for error in audio["errors"]
    ):
        lines.append(
            "Enable microphone access in Windows Settings > Privacy & security "
            "> Microphone, then retry."
        )
    if audio["status"] == "no_input_devices":
        lines.append(
            "Check that a microphone is enabled in Windows and visible to other apps."
        )

    return "\n".join(lines)


def _load_audio_backend() -> Any:
    import sounddevice

    return sounddevice


def _report_path(
    requested: object,
    generated_at_utc: str,
    *,
    report_type: str = "hardware_diagnostic",
) -> Path:
    if requested is _AUTO_REPORT:
        timestamp = datetime.fromisoformat(generated_at_utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        prefix = (
            "desksense-characterization"
            if report_type == "microphone_characterization"
            else "desksense-diagnostic"
        )
        return Path("reports") / f"{prefix}-{timestamp}.json"
    if not isinstance(requested, Path):
        raise TypeError("Report path must be a filesystem path.")
    return requested


def _analysis_report_path(requested: object, generated_at_utc: str) -> Path:
    if requested is _AUTO_REPORT:
        timestamp = datetime.fromisoformat(generated_at_utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        return Path("reports") / f"desksense-dataset-analysis-{timestamp}.json"
    if not isinstance(requested, Path):
        raise TypeError("Analysis report path must be a filesystem path.")
    return requested


def _robustness_report_path(requested: object, generated_at_utc: str) -> Path:
    if requested is _AUTO_REPORT:
        timestamp = datetime.fromisoformat(generated_at_utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        return Path("reports") / f"desksense-robustness-replay-{timestamp}.json"
    if not isinstance(requested, Path):
        raise TypeError("Robustness report path must be a filesystem path.")
    return requested


def _external_evaluation_report_path(
    requested: object, generated_at_utc: str
) -> Path:
    if requested is _AUTO_REPORT:
        timestamp = datetime.fromisoformat(generated_at_utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        return Path("reports") / f"desksense-external-evaluation-{timestamp}.json"
    if not isinstance(requested, Path):
        raise TypeError("External evaluation report path must be a filesystem path.")
    return requested


def _nonnegative_device_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("device index must be an integer") from error
    if index < 0:
        raise argparse.ArgumentTypeError("device index must be zero or greater")
    return index


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return result


def _print_microphone_permission_guidance() -> None:
    print(
        "Enable microphone access in Windows Settings > Privacy & security > "
        "Microphone, then retry.",
        file=sys.stderr,
    )


def _find_device(
    devices: Sequence[dict[str, Any]], device_index: int | None
) -> dict[str, Any] | None:
    return next(
        (device for device in devices if device["index"] == device_index), None
    )


def _format_device(device: dict[str, Any] | None) -> str:
    if device is None:
        return "not available"
    return f"{device['index']}: {device['name']}"


def _format_rate(sample_rate: float | None) -> str:
    if sample_rate is None:
        return "unknown"
    if float(sample_rate).is_integer():
        return f"{int(sample_rate)} Hz"
    return f"{sample_rate:g} Hz"


def _format_dbfs(value: float | None) -> str:
    return "-inf dBFS" if value is None else f"{value:.2f} dBFS"


def _append_characterization(
    lines: list[str], characterization: dict[str, Any]
) -> bool:
    """Append a readable exploratory characterization summary."""

    status = characterization["status"]
    if status != "ok":
        lines.append(
            "Microphone characterization: " + status.replace("_", " ")
        )
        error = characterization.get("error", {})
        if error.get("message"):
            lines.append(f"Reason: {error['message']}")
        if error.get("category") == "permission_denied":
            lines.append(
                "Enable microphone access in Windows Settings > Privacy & security "
                "> Microphone, then retry."
            )
            return True
        return False

    device = characterization["device"]
    host_api = device["host_api"]
    lines.append("Microphone characterization: completed (exploratory)")
    lines.append(f"  Device: {device['index']}: {device['name']}")
    lines.append(f"  Host API: {host_api['name']}")
    lines.append(
        "  Capture: "
        f"{characterization['captured_duration_seconds']:.2f} s, "
        f"{_format_rate(characterization['sample_rate_hz'])}, "
        f"{characterization['channel_count']} channel(s), "
        f"{characterization['frames_captured']} frame(s)"
    )
    lines.append(
        "  Sample-rate choice: "
        + characterization["sample_rate_source"].replace("_", " ")
    )
    if characterization["channel_limit_applied"]:
        lines.append(
            "  Channel safety limit: analyzed up to "
            f"{characterization['channel_search_limit']} of "
            f"{characterization['advertised_max_input_channels']} advertised "
            "input channels"
        )

    analysis = characterization["analysis"]
    activity_assessment = analysis["analysis_parameters"]["activity_assessment"]
    lines.append("  Per-channel measurements:")
    lines.append(
        "    Inactivity heuristic uses std <= "
        f"{activity_assessment['absolute_standard_deviation_floor']:.8f}, "
        "centered peak <= "
        f"{activity_assessment['absolute_centered_peak_floor']:.8f}, and "
        "relative level <= "
        f"{activity_assessment['relative_floor_db']:.1f} dB in combination."
    )
    for channel in analysis["per_channel"]:
        activity_label = (
            "effectively inactive (heuristic)"
            if channel["effectively_inactive"]
            else "active"
        )
        lines.append(
            f"    Channel {channel['channel']}: "
            f"peak={channel['peak_absolute']:.6f}, "
            f"RMS={channel['rms']:.6f}, "
            f"mean/DC={channel['dc_offset']:.6f}, "
            f"std={channel['standard_deviation']:.6f}, "
            f"centered peak={channel['centered_peak_absolute']:.6f}, "
            "relative std="
            f"{_format_relative_db(channel['relative_standard_deviation_db_to_strongest'])}, "
            "relative centered peak="
            f"{_format_relative_db(channel['relative_centered_peak_db_to_strongest'])}; "
            f"{activity_label}"
        )

    lines.append("  Whole-recording channel pairs:")
    _append_pair_metrics(lines, analysis["whole_recording"]["pairwise"])

    transient = analysis["transient_window"]
    if transient["status"] == "ok":
        lines.append(
            "  Strongest transient window (exploratory, not a tap detector): "
            f"{transient['start_sample']}:{transient['end_sample_exclusive']} "
            f"({transient['duration_seconds'] * 1000.0:.1f} ms), centered near "
            f"{transient['center_time_seconds']:.6f} s"
        )
        lines.append("  Transient-window channel pairs:")
        _append_pair_metrics(lines, transient["pairwise"])
    else:
        lines.append(
            "  Strongest transient window: not available "
            f"({transient.get('reason', 'insufficient signal')})"
        )

    lines.append(
        "  Heuristic caution: "
        + analysis["heuristic_interpretation"]["caution"]
    )
    lines.append("Raw audio was discarded and was not written to disk.")
    return False


def _append_pair_metrics(
    lines: list[str], pairs: Sequence[dict[str, Any]]
) -> None:
    if not pairs:
        lines.append("    No channel pairs are available.")
        return
    for pair in pairs:
        identity = f"Ch{pair['channel_a']}-Ch{pair['channel_b']}"
        if pair["status"] != "ok":
            lines.append(
                f"    {identity}: not analyzed "
                f"({pair.get('reason', 'insufficient signal')})"
            )
            continue
        label = pair.get("heuristic_label")
        label_text = f", heuristic={label}" if label else ""
        lines.append(
            f"    {identity}: Pearson={pair['pearson_correlation']:.6f}, "
            "normalized difference energy="
            f"{pair['normalized_difference_energy']:.6f}, "
            f"AC RMS B/A={pair['ac_rms_ratio_b_to_a']:.6f} "
            f"({pair['relative_ac_level_db_b_minus_a']:+.2f} dB), "
            "max normalized cross-correlation="
            f"{pair['max_normalized_cross_correlation']:.6f}, "
            f"lag={pair['lag_samples']:+d} sample(s) "
            f"({pair['lag_microseconds']:+.2f} us){label_text}"
        )


def _format_relative_db(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:+.2f} dB"
