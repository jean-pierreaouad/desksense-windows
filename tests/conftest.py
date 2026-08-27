from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


class FakeSoundDevice:
    """Small sounddevice stand-in used by hardware-independent tests."""

    __version__ = "0.test"

    def __init__(
        self,
        *,
        devices: list[dict[str, Any]] | None = None,
        host_apis: list[dict[str, Any]] | None = None,
        default_input_index: int | None = 1,
        unsupported_rates: set[tuple[int, int]] | None = None,
        unsupported_settings: set[tuple[int, int, int]] | None = None,
        max_supported_channels: dict[int, int] | None = None,
        recording: np.ndarray[Any, Any] | None = None,
        recording_error: Exception | None = None,
        settings_error: Exception | None = None,
        enumeration_error: Exception | None = None,
    ) -> None:
        self.devices = (
            devices
            if devices is not None
            else [
                {
                    "name": "Speakers",
                    "hostapi": 0,
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                    "default_samplerate": 48_000.0,
                },
                {
                    "name": "Microphone Array",
                    "hostapi": 0,
                    "max_input_channels": 2,
                    "max_output_channels": 0,
                    "default_samplerate": 48_000.0,
                },
            ]
        )
        self.host_apis = (
            host_apis
            if host_apis is not None
            else [
                {
                    "name": "Windows WASAPI",
                    "default_input_device": default_input_index
                    if default_input_index is not None
                    else -1,
                }
            ]
        )
        self.default_input_index = default_input_index
        self.default = SimpleNamespace(device=(-1, -1))
        self.unsupported_rates = unsupported_rates or set()
        self.unsupported_settings = unsupported_settings or set()
        self.max_supported_channels = max_supported_channels or {}
        self.recording = recording
        self.recording_error = recording_error
        self.settings_error = settings_error
        self.enumeration_error = enumeration_error
        self.record_calls: list[dict[str, Any]] = []
        self.settings_calls: list[dict[str, Any]] = []
        self.recording_active = False
        self.wait_calls = 0
        self.stop_calls = 0

    def get_portaudio_version(self) -> tuple[int, str]:
        return 1, "PortAudio test backend"

    def query_hostapis(self) -> list[dict[str, Any]]:
        return [dict(host_api) for host_api in self.host_apis]

    def query_devices(
        self, device: int | None = None, kind: str | None = None
    ) -> Any:
        if self.enumeration_error is not None and device is None and kind is None:
            raise self.enumeration_error
        if device is None and kind is None:
            return [
                {**dict(item), "index": index}
                for index, item in enumerate(self.devices)
            ]

        index = self.default_input_index if device is None else int(device)
        if index is None or index < 0 or index >= len(self.devices):
            raise RuntimeError("No default input device")
        selected = self.devices[index]
        if kind == "input" and selected["max_input_channels"] <= 0:
            raise RuntimeError("Invalid device: no input channels")
        return {**dict(selected), "index": index}

    def check_input_settings(
        self,
        *,
        device: int,
        channels: int,
        dtype: str,
        samplerate: float,
    ) -> None:
        self.settings_calls.append(
            {
                "device": device,
                "channels": channels,
                "dtype": dtype,
                "samplerate": samplerate,
            }
        )
        if self.settings_error is not None:
            raise self.settings_error
        if (device, round(samplerate)) in self.unsupported_rates:
            raise RuntimeError(f"Sample rate {samplerate:g} is unsupported")
        if (device, round(samplerate), channels) in self.unsupported_settings:
            raise RuntimeError(
                f"{channels} channel(s) at {samplerate:g} Hz are unsupported"
            )
        maximum = self.max_supported_channels.get(
            device, int(self.devices[device]["max_input_channels"])
        )
        if channels > maximum:
            raise RuntimeError(f"{channels} input channels are unsupported")

    def rec(
        self,
        frames: int,
        *,
        samplerate: float,
        channels: int,
        dtype: str,
        device: int,
        blocking: bool,
    ) -> np.ndarray[Any, Any]:
        self.record_calls.append(
            {
                "frames": frames,
                "samplerate": samplerate,
                "channels": channels,
                "dtype": dtype,
                "device": device,
                "blocking": blocking,
            }
        )
        if self.recording_error is not None:
            raise self.recording_error
        if not blocking:
            self.recording_active = True
        if self.recording is not None:
            return self.recording.copy()
        return np.zeros((frames, channels), dtype=np.float32)

    def wait(self) -> None:
        self.wait_calls += 1
        self.recording_active = False

    def stop(self) -> None:
        self.stop_calls += 1
        self.recording_active = False


@pytest.fixture
def audio_backend_factory():
    return FakeSoundDevice
