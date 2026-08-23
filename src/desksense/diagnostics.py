"""Hardware-independent logic for the DeskSense microphone probe."""

from __future__ import annotations

import json
import math
import platform
import struct
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desksense.characterization import (
    DEFAULT_CHARACTERIZATION_SECONDS,
    MAX_CHARACTERIZATION_CHANNELS,
    analyze_multichannel_audio,
)

COMMON_SAMPLE_RATES_HZ = (44_100, 48_000)
DEFAULT_RECORDING_SECONDS = 3.0
REPORT_SCHEMA_VERSION = 1


class RecordingConfigurationError(RuntimeError):
    """Raised when no usable input stream configuration can be found."""

    def __init__(
        self, message: str, *, category: str = "unsupported_configuration"
    ) -> None:
        super().__init__(message)
        self.category = category


def system_information() -> dict[str, Any]:
    """Return JSON-safe operating-system and Python runtime information."""

    uname = platform.uname()
    return {
        "operating_system": {
            "name": uname.system or platform.system(),
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "architecture_bits": struct.calcsize("P") * 8,
        },
    }


def channel_layout(channel_count: int) -> str:
    """Return a human-readable layout label from an input channel count."""

    if channel_count <= 0:
        return "unavailable"
    if channel_count == 1:
        return "mono"
    if channel_count == 2:
        return "stereo"
    return "multichannel"


def describe_audio_error(error: BaseException) -> dict[str, str]:
    """Classify an audio error without depending on sounddevice exception types."""

    message = str(error).strip() or error.__class__.__name__
    normalized = message.casefold()

    if isinstance(error, PermissionError) or any(
        marker in normalized
        for marker in (
            "access is denied",
            "access denied",
            "permission denied",
            "not permitted",
            "paaccessdenied",
        )
    ):
        category = "permission_denied"
    elif any(
        marker in normalized
        for marker in (
            "no default input",
            "no input device",
            "device unavailable",
            "invalid device",
        )
    ):
        category = "device_unavailable"
    elif any(
        marker in normalized
        for marker in (
            "invalid sample rate",
            "invalid number of channels",
            "invalid channel count",
            "invalid sample format",
            "sample format not supported",
            "not supported",
            "unsupported",
        )
    ):
        category = "unsupported_configuration"
    else:
        category = "audio_error"

    return {"category": category, "message": message}


def measure_channel_levels(samples: np.ndarray[Any, Any]) -> list[dict[str, Any]]:
    """Calculate peak absolute amplitude and RMS for every captured channel."""

    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if audio.ndim != 2:
        raise ValueError("Captured audio must be a one- or two-dimensional array.")
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError("Captured audio is empty.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Captured audio contains non-finite samples.")

    levels: list[dict[str, Any]] = []
    for index in range(audio.shape[1]):
        channel = audio[:, index]
        peak = float(np.max(np.abs(channel)))
        rms = float(np.sqrt(np.mean(np.square(channel))))
        levels.append(
            {
                "channel": index + 1,
                "peak_absolute": peak,
                "rms": rms,
                "peak_dbfs": _amplitude_to_dbfs(peak),
                "rms_dbfs": _amplitude_to_dbfs(rms),
            }
        )
    return levels


def collect_diagnostic_report(
    audio_backend: Any,
    *,
    selected_device_index: int | None = None,
    record: bool = False,
    duration_seconds: float = DEFAULT_RECORDING_SECONDS,
) -> dict[str, Any]:
    """Collect an inventory and, when requested, a short in-memory recording."""

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "hardware_diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "system": system_information(),
        "audio": collect_audio_inventory(
            audio_backend, selected_device_index=selected_device_index
        ),
    }

    audio = report["audio"]
    selected_index = audio["selected_input_device_index"]
    if not record:
        audio["recording"] = {"status": "not_requested"}
    elif selected_index is None:
        audio["recording"] = {
            "status": "not_run",
            "error": {
                "category": "device_unavailable",
                "message": "No usable input device was selected.",
            },
        }
    else:
        selected_device = next(
            device
            for device in audio["input_devices"]
            if device["index"] == selected_index
        )
        audio["recording"] = run_recording_test(
            audio_backend,
            selected_device,
            duration_seconds=duration_seconds,
        )

    if audio["status"] != "ok" or audio["recording"]["status"] in {
        "error",
        "not_run",
    }:
        report["status"] = "error"
    return report


def collect_characterization_report(
    audio_backend: Any,
    *,
    selected_device_index: int | None = None,
    duration_seconds: float = DEFAULT_CHARACTERIZATION_SECONDS,
) -> dict[str, Any]:
    """Collect an inventory and characterize one selected input in memory."""

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "microphone_characterization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "system": system_information(),
        "audio": collect_audio_inventory(
            audio_backend, selected_device_index=selected_device_index
        ),
    }

    audio = report["audio"]
    audio["recording"] = {"status": "not_requested"}
    selected_index = audio["selected_input_device_index"]
    if selected_index is None:
        audio["characterization"] = {
            "status": "not_run",
            "exploratory": True,
            "error": {
                "category": "device_unavailable",
                "message": "No usable input device was selected.",
            },
        }
    else:
        selected_device = next(
            device
            for device in audio["input_devices"]
            if device["index"] == selected_index
        )
        audio["characterization"] = run_characterization_test(
            audio_backend,
            selected_device,
            duration_seconds=duration_seconds,
        )

    if (
        audio["characterization"]["status"] == "ok"
        and audio["status"] == "capability_check_error"
    ):
        _downgrade_capability_errors_to_warnings(audio)

    if audio["status"] != "ok" or audio["characterization"]["status"] in {
        "error",
        "not_run",
    }:
        report["status"] = "error"
    return report


def collect_audio_inventory(
    audio_backend: Any, *, selected_device_index: int | None = None
) -> dict[str, Any]:
    """Enumerate input devices and validate common sample rates."""

    audio: dict[str, Any] = {
        "status": "ok",
        "backend": _backend_information(audio_backend),
        "host_apis": [],
        "input_devices": [],
        "default_input_device_index": None,
        "selected_input_device_index": None,
        "warnings": [],
        "errors": [],
    }

    raw_host_apis: list[Mapping[str, Any]] = []
    try:
        raw_host_apis = list(audio_backend.query_hostapis())
    except Exception as error:
        audio["warnings"].append(
            f"Audio host APIs could not be enumerated: {describe_audio_error(error)['message']}"
        )

    audio["host_apis"] = [
        _normalize_host_api(index, host_api)
        for index, host_api in enumerate(raw_host_apis)
    ]

    try:
        raw_devices = list(audio_backend.query_devices())
    except Exception as error:
        audio["status"] = "enumeration_error"
        audio["errors"].append(describe_audio_error(error))
        return audio

    for index, raw_device in enumerate(raw_devices):
        max_input_channels = _safe_int(raw_device.get("max_input_channels")) or 0
        if max_input_channels <= 0:
            continue

        host_api_index = _safe_int(raw_device.get("hostapi"))
        host_api_name = _host_api_name(audio["host_apis"], host_api_index)
        default_sample_rate = _positive_float(raw_device.get("default_samplerate"))
        device_name = str(raw_device.get("name", f"Input device {index}"))
        sample_rate_support = _check_common_sample_rates(
            audio_backend,
            device_index=index,
            channel_count=max_input_channels,
        )
        audio["input_devices"].append(
            {
                "index": index,
                "name": device_name,
                "host_api": {
                    "index": host_api_index,
                    "name": host_api_name,
                },
                "max_input_channels": max_input_channels,
                "channel_layout": channel_layout(max_input_channels),
                "default_sample_rate_hz": default_sample_rate,
                "sample_rate_support": sample_rate_support,
                "is_default": False,
                "is_selected": False,
            }
        )
        for rate_result in sample_rate_support:
            if rate_result["supported"] is not None:
                continue
            details = rate_result["error"]
            audio["status"] = "capability_check_error"
            audio["errors"].append(
                {
                    "category": details["category"],
                    "message": (
                        f"Could not check {rate_result['sample_rate_hz']} Hz "
                        f"support for device {index} ({device_name}): "
                        f"{details['message']}"
                    ),
                }
            )

    if not audio["input_devices"]:
        audio["status"] = "no_input_devices"
        audio["errors"].append(
            {
                "category": "device_unavailable",
                "message": "PortAudio reported no audio input devices.",
            }
        )
        return audio

    default_index, default_warning = _resolve_default_input_index(
        audio_backend,
        raw_devices=raw_devices,
        raw_host_apis=raw_host_apis,
    )
    input_indexes = {device["index"] for device in audio["input_devices"]}
    if default_index not in input_indexes:
        default_index = None
    if default_warning:
        audio["warnings"].append(default_warning)
    audio["default_input_device_index"] = default_index

    if selected_device_index is not None:
        if selected_device_index not in input_indexes:
            audio["status"] = "invalid_selection"
            audio["errors"].append(
                {
                    "category": "device_unavailable",
                    "message": (
                        f"Device index {selected_device_index} is not an available "
                        "audio input device."
                    ),
                }
            )
        else:
            audio["selected_input_device_index"] = selected_device_index
    elif default_index is None:
        audio["status"] = "no_default_input"
        audio["errors"].append(
            {
                "category": "device_unavailable",
                "message": "No default audio input device could be identified.",
            }
        )
    else:
        audio["selected_input_device_index"] = default_index

    for device in audio["input_devices"]:
        device["is_default"] = device["index"] == default_index
        device["is_selected"] = (
            device["index"] == audio["selected_input_device_index"]
        )

    return audio


def run_recording_test(
    audio_backend: Any,
    device: Mapping[str, Any],
    *,
    duration_seconds: float = DEFAULT_RECORDING_SECONDS,
) -> dict[str, Any]:
    """Record into memory using a validated device configuration and measure it."""

    try:
        validated_duration = float(duration_seconds)
    except (TypeError, ValueError, OverflowError):
        validated_duration = math.nan
    result: dict[str, Any] = {"status": "error"}

    if not math.isfinite(validated_duration) or validated_duration <= 0:
        result["error"] = {
            "category": "invalid_duration",
            "message": "Recording duration must be a positive finite number.",
        }
        return result
    result["requested_duration_seconds"] = validated_duration

    try:
        configuration = _find_recording_configuration(audio_backend, device)
    except RecordingConfigurationError as error:
        result["error"] = {
            "category": error.category,
            "message": str(error),
        }
        return result

    frames_requested = max(
        1, round(validated_duration * configuration["sample_rate_hz"])
    )
    try:
        captured = audio_backend.rec(
            frames_requested,
            samplerate=configuration["sample_rate_hz"],
            channels=configuration["channels"],
            dtype="float32",
            device=device["index"],
            blocking=True,
        )
        captured_array = np.asarray(captured)
        captured_channels = 1 if captured_array.ndim == 1 else (
            captured_array.shape[1] if captured_array.ndim == 2 else None
        )
        if captured_channels != configuration["channels"]:
            raise ValueError(
                "Captured audio channel count does not match the validated "
                "recording configuration."
            )
        levels = measure_channel_levels(captured_array)
    except Exception as error:
        result["error"] = describe_audio_error(error)
        return result

    frame_count = int(captured_array.shape[0])
    result.update(
        {
            "status": "ok",
            "device_index": int(device["index"]),
            "device_name": str(device["name"]),
            "sample_rate_hz": configuration["sample_rate_hz"],
            "sample_rate_source": configuration["sample_rate_source"],
            "channels": configuration["channels"],
            "frames_requested": frames_requested,
            "frames_captured": frame_count,
            "captured_duration_seconds": (
                frame_count / configuration["sample_rate_hz"]
            ),
            "channel_levels": levels,
        }
    )
    return result


def run_characterization_test(
    audio_backend: Any,
    device: Mapping[str, Any],
    *,
    duration_seconds: float = DEFAULT_CHARACTERIZATION_SECONDS,
) -> dict[str, Any]:
    """Capture and characterize one device without retaining raw audio."""

    try:
        validated_duration = float(duration_seconds)
    except (TypeError, ValueError, OverflowError):
        validated_duration = math.nan
    result: dict[str, Any] = {
        "status": "error",
        "exploratory": True,
        "raw_audio_included": False,
    }

    if not math.isfinite(validated_duration) or validated_duration <= 0:
        result["error"] = {
            "category": "invalid_duration",
            "message": "Characterization duration must be a positive finite number.",
        }
        return result
    result["requested_duration_seconds"] = validated_duration

    try:
        configuration = _find_characterization_configuration(
            audio_backend, device
        )
    except RecordingConfigurationError as error:
        result["error"] = {
            "category": error.category,
            "message": str(error),
        }
        return result

    frames_requested = max(
        1, round(validated_duration * configuration["sample_rate_hz"])
    )
    try:
        captured = audio_backend.rec(
            frames_requested,
            samplerate=configuration["sample_rate_hz"],
            channels=configuration["channels"],
            dtype="float32",
            device=device["index"],
            blocking=True,
        )
        captured_array = np.asarray(captured)
        captured_channels = 1 if captured_array.ndim == 1 else (
            captured_array.shape[1] if captured_array.ndim == 2 else None
        )
        if captured_channels != configuration["channels"]:
            raise ValueError(
                "Captured audio channel count does not match the validated "
                "characterization configuration."
            )
        analysis = analyze_multichannel_audio(
            captured_array, configuration["sample_rate_hz"]
        )
    except Exception as error:
        result["error"] = describe_audio_error(error)
        return result

    frame_count = int(captured_array.shape[0])
    host_api = device.get("host_api", {})
    result.update(
        {
            "status": "ok",
            "device": {
                "index": int(device["index"]),
                "name": str(device["name"]),
                "host_api": {
                    "index": _safe_int(host_api.get("index")),
                    "name": str(host_api.get("name", "Unknown host API")),
                },
            },
            "sample_rate_hz": configuration["sample_rate_hz"],
            "sample_rate_source": configuration["sample_rate_source"],
            "advertised_max_input_channels": configuration[
                "advertised_max_input_channels"
            ],
            "channel_search_limit": configuration["channel_search_limit"],
            "channel_limit_applied": configuration["channel_limit_applied"],
            "channel_count": configuration["channels"],
            "frames_requested": frames_requested,
            "frames_captured": frame_count,
            "captured_duration_seconds": (
                frame_count / configuration["sample_rate_hz"]
            ),
            "analysis": analysis,
        }
    )
    return result


def build_backend_error_report(
    error: BaseException,
    *,
    record_requested: bool,
    characterize_requested: bool = False,
) -> dict[str, Any]:
    """Build a valid report when sounddevice or PortAudio cannot be loaded."""

    details = describe_audio_error(error)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": (
            "microphone_characterization"
            if characterize_requested
            else "hardware_diagnostic"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "system": system_information(),
        "audio": {
            "status": "backend_unavailable",
            "backend": {"library": "sounddevice", "version": None},
            "host_apis": [],
            "input_devices": [],
            "default_input_device_index": None,
            "selected_input_device_index": None,
            "warnings": [],
            "errors": [details],
            "recording": {
                "status": "not_run" if record_requested else "not_requested",
                **({"error": details} if record_requested else {}),
            },
        },
    }
    if characterize_requested:
        report["audio"]["characterization"] = {
            "status": "not_run",
            "exploratory": True,
            "error": details,
        }
    return report


def write_json_report(report: Mapping[str, Any], path: Path) -> Path:
    """Write a JSON report without overwriting an existing file."""

    if path.suffix.casefold() != ".json":
        raise ValueError("Diagnostic report paths must end in .json.")

    serialized = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as report_file:
        report_file.write(serialized)
        report_file.write("\n")
    return path.resolve()


def _backend_information(audio_backend: Any) -> dict[str, Any]:
    information: dict[str, Any] = {
        "library": "sounddevice",
        "version": str(getattr(audio_backend, "__version__", "unknown")),
    }
    try:
        version_number, version_text = audio_backend.get_portaudio_version()
    except Exception:
        return information
    information["portaudio_version_number"] = int(version_number)
    information["portaudio_version_text"] = str(version_text)
    return information


def _normalize_host_api(
    index: int, host_api: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "index": index,
        "name": str(host_api.get("name", f"Host API {index}")),
        "default_input_device_index": _nonnegative_int(
            host_api.get("default_input_device")
        ),
    }


def _host_api_name(
    host_apis: Sequence[Mapping[str, Any]], host_api_index: int | None
) -> str:
    if host_api_index is not None:
        for host_api in host_apis:
            if host_api["index"] == host_api_index:
                return str(host_api["name"])
        return f"Unknown host API ({host_api_index})"
    return "Unknown host API"


def _check_common_sample_rates(
    audio_backend: Any, *, device_index: int, channel_count: int
) -> list[dict[str, Any]]:
    support: list[dict[str, Any]] = []
    for sample_rate in COMMON_SAMPLE_RATES_HZ:
        result: dict[str, Any] = {
            "sample_rate_hz": sample_rate,
            "max_input_channels_checked": channel_count,
            "supported": False,
        }
        last_error_details: dict[str, str] | None = None
        most_significant_error: dict[str, str] | None = None
        for channels in range(channel_count, 0, -1):
            try:
                audio_backend.check_input_settings(
                    device=device_index,
                    channels=channels,
                    dtype="float32",
                    samplerate=sample_rate,
                )
            except Exception as error:
                details = describe_audio_error(error)
                last_error_details = details
                if (
                    most_significant_error is None
                    or _audio_error_priority(details["category"])
                    > _audio_error_priority(
                        most_significant_error["category"]
                    )
                ):
                    most_significant_error = details
            else:
                result["supported"] = True
                result["supported_input_channels"] = channels
                break
        if not result["supported"] and last_error_details is not None:
            details = most_significant_error or last_error_details
            result["error"] = details
            if details["category"] != "unsupported_configuration":
                result["supported"] = None
        support.append(result)
    return support


def _resolve_default_input_index(
    audio_backend: Any,
    *,
    raw_devices: Sequence[Mapping[str, Any]],
    raw_host_apis: Sequence[Mapping[str, Any]],
) -> tuple[int | None, str | None]:
    configured_pair = getattr(
        getattr(audio_backend, "default", None), "device", None
    )
    if isinstance(configured_pair, (str, bytes)):
        configured = configured_pair
    else:
        try:
            configured = configured_pair["input"]
        except (IndexError, KeyError, TypeError):
            try:
                configured = configured_pair[0]
            except (IndexError, KeyError, TypeError):
                configured = configured_pair
    configured_index = _nonnegative_int(configured)
    if configured_index is not None and configured_index < len(raw_devices):
        return configured_index, None

    try:
        default_device = audio_backend.query_devices(kind="input")
    except Exception as error:
        return (
            None,
            "The default input device could not be queried: "
            f"{describe_audio_error(error)['message']}",
        )

    queried_index = _nonnegative_int(default_device.get("index"))
    if queried_index is not None and queried_index < len(raw_devices):
        return queried_index, None

    default_signature = _device_signature(default_device)
    for index, raw_device in enumerate(raw_devices):
        if _device_signature(raw_device) == default_signature:
            return index, None

    host_api_index = _nonnegative_int(default_device.get("hostapi"))
    if host_api_index is not None and host_api_index < len(raw_host_apis):
        host_default = _nonnegative_int(
            raw_host_apis[host_api_index].get("default_input_device")
        )
        if host_default is not None and host_default < len(raw_devices):
            return host_default, None

    return None, "PortAudio returned a default input that was not in the device list."


def _device_signature(device: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(device.get("name", "")),
        _safe_int(device.get("hostapi")),
        _safe_int(device.get("max_input_channels")),
        _positive_float(device.get("default_samplerate")),
    )


def _find_recording_configuration(
    audio_backend: Any, device: Mapping[str, Any]
) -> dict[str, Any]:
    max_channels = _safe_int(device.get("max_input_channels")) or 0
    if max_channels <= 0:
        raise RecordingConfigurationError(
            "The selected device has no input channels.",
            category="device_unavailable",
        )

    candidate_rates: list[tuple[float, str]] = []
    default_rate = _positive_float(device.get("default_sample_rate_hz"))
    if default_rate is not None:
        candidate_rates.append((default_rate, "device_default"))
    for rate_result in device.get("sample_rate_support", []):
        rate = _positive_float(rate_result.get("sample_rate_hz"))
        known_rates = {candidate[0] for candidate in candidate_rates}
        if rate_result.get("supported") and rate not in known_rates:
            candidate_rates.append((rate, "validated_common_rate_fallback"))

    if not candidate_rates:
        raise RecordingConfigurationError(
            "The selected device did not report a usable sample rate."
        )

    last_error: BaseException | None = None
    for sample_rate, sample_rate_source in candidate_rates:
        for channels in range(max_channels, 0, -1):
            try:
                audio_backend.check_input_settings(
                    device=device["index"],
                    channels=channels,
                    dtype="float32",
                    samplerate=sample_rate,
                )
            except Exception as error:
                details = describe_audio_error(error)
                if details["category"] != "unsupported_configuration":
                    raise RecordingConfigurationError(
                        details["message"], category=details["category"]
                    ) from error
                last_error = error
            else:
                return {
                    "sample_rate_hz": sample_rate,
                    "sample_rate_source": sample_rate_source,
                    "channels": channels,
                }

    detail = (
        f" Last PortAudio error: {describe_audio_error(last_error)['message']}"
        if last_error is not None
        else ""
    )
    raise RecordingConfigurationError(
        "No supported recording configuration was found for the selected device."
        + detail
    )


def _find_characterization_configuration(
    audio_backend: Any, device: Mapping[str, Any]
) -> dict[str, Any]:
    """Find a supported configuration, prioritizing useful channel count."""

    advertised_max_channels = _safe_int(device.get("max_input_channels")) or 0
    if advertised_max_channels <= 0:
        raise RecordingConfigurationError(
            "The selected device has no input channels.",
            category="device_unavailable",
        )

    channel_search_limit = min(
        advertised_max_channels, MAX_CHARACTERIZATION_CHANNELS
    )
    candidate_rates: list[tuple[float, str]] = []
    default_rate = _positive_float(device.get("default_sample_rate_hz"))
    if default_rate is not None:
        candidate_rates.append((default_rate, "device_default"))
    for rate_result in device.get("sample_rate_support", []):
        rate = _positive_float(rate_result.get("sample_rate_hz"))
        known_rates = {candidate[0] for candidate in candidate_rates}
        if rate_result.get("supported") is True and rate not in known_rates:
            candidate_rates.append((rate, "validated_common_rate_fallback"))

    if not candidate_rates:
        raise RecordingConfigurationError(
            "The selected device did not report a usable sample rate."
        )

    most_significant_error: dict[str, str] | None = None
    for channels in range(channel_search_limit, 0, -1):
        for sample_rate, sample_rate_source in candidate_rates:
            try:
                audio_backend.check_input_settings(
                    device=device["index"],
                    channels=channels,
                    dtype="float32",
                    samplerate=sample_rate,
                )
            except Exception as error:
                details = describe_audio_error(error)
                if details["category"] in {
                    "permission_denied",
                    "device_unavailable",
                }:
                    raise RecordingConfigurationError(
                        details["message"], category=details["category"]
                    ) from error
                if (
                    most_significant_error is None
                    or _audio_error_priority(details["category"])
                    > _audio_error_priority(
                        most_significant_error["category"]
                    )
                ):
                    most_significant_error = details
            else:
                return {
                    "sample_rate_hz": sample_rate,
                    "sample_rate_source": sample_rate_source,
                    "channels": channels,
                    "advertised_max_input_channels": advertised_max_channels,
                    "channel_search_limit": channel_search_limit,
                    "channel_limit_applied": (
                        channel_search_limit < advertised_max_channels
                    ),
                }

    detail = (
        f" PortAudio error: {most_significant_error['message']}"
        if most_significant_error is not None
        else ""
    )
    raise RecordingConfigurationError(
        "No supported characterization configuration was found for the selected "
        "device." + detail,
        category=(
            most_significant_error["category"]
            if most_significant_error is not None
            else "unsupported_configuration"
        ),
    )


def _amplitude_to_dbfs(amplitude: float) -> float | None:
    if amplitude <= 0:
        return None
    return 20.0 * math.log10(amplitude)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _nonnegative_int(value: Any) -> int | None:
    converted = _safe_int(value)
    return converted if converted is not None and converted >= 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) and converted > 0 else None


def _audio_error_priority(category: str) -> int:
    """Rank capability-check errors so fallback failures do not mask access issues."""

    return {
        "unsupported_configuration": 0,
        "audio_error": 1,
        "device_unavailable": 2,
        "permission_denied": 3,
    }.get(category, 1)


def _downgrade_capability_errors_to_warnings(audio: dict[str, Any]) -> None:
    """Keep endpoint-check details without failing a successful capture."""

    audio["warnings"].extend(
        f"Capability check warning: {error['message']}"
        for error in audio["errors"]
    )
    audio["errors"].clear()
    audio["status"] = "ok"
