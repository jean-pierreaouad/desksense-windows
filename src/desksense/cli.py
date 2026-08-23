"""Command-line interface for the DeskSense microphone feasibility probe."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from desksense.characterization import DEFAULT_CHARACTERIZATION_SECONDS
from desksense.diagnostics import (
    DEFAULT_RECORDING_SECONDS,
    build_backend_error_report,
    collect_characterization_report,
    collect_diagnostic_report,
    write_json_report,
)

_AUTO_REPORT = object()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="desksense-diagnose",
        description=(
            "Enumerate audio inputs and optionally record a short in-memory sample "
            "for DeskSense hardware feasibility testing."
        ),
    )
    parser.add_argument(
        "--device",
        type=_nonnegative_device_index,
        metavar="INDEX",
        help="input device index to select (defaults to the system default input)",
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
    parser.add_argument(
        "--save-report",
        nargs="?",
        const=_AUTO_REPORT,
        type=Path,
        metavar="PATH",
        help=(
            "save a JSON report; omit PATH to create a timestamped file under "
            "reports/"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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


def _nonnegative_device_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("device index must be an integer") from error
    if index < 0:
        raise argparse.ArgumentTypeError("device index must be zero or greater")
    return index


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
