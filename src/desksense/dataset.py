"""Local, lossless persistence for guided DeskSense dataset sessions."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np


DATASET_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


@dataclass(frozen=True)
class DatasetSession:
    """Paths and append-only writers for one local collection session."""

    session_id: str
    directory: Path
    samples_directory: Path
    manifest_path: Path
    rejected_attempts_path: Path

    def save_sample(
        self,
        *,
        sample_id: str,
        filename_stem: str,
        capture: Any,
        tap_window: Any,
        metadata: Mapping[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        """Save one accepted sample without replacing an existing artifact."""

        _validate_safe_name(sample_id, "sample identifier")
        _validate_safe_name(filename_stem, "sample filename")
        capture_array = _float32_audio_matrix(capture, "capture")
        tap_window_array = _float32_audio_matrix(tap_window, "tap window")
        if capture_array.shape[1] != tap_window_array.shape[1]:
            raise ValueError(
                "Capture and tap window channel counts must match."
            )

        relative_path = Path("samples") / f"{filename_stem}.npz"
        complete_metadata = {
            **dict(metadata),
            "sample_id": sample_id,
            "saved_sample_path": relative_path.as_posix(),
            "storage": {
                "format": "NumPy NPZ (lossless compressed)",
                "capture_array": "capture",
                "tap_window_array": "tap_window",
                "capture_shape": [int(value) for value in capture_array.shape],
                "tap_window_shape": [
                    int(value) for value in tap_window_array.shape
                ],
                "dtype": "float32",
                "metadata_array": "metadata_json",
            },
        }
        metadata_json = _serialize_json(complete_metadata, indent=None)
        sample_path = self.directory / relative_path

        # Build the archive completely before publishing its final path. Passing
        # an exclusive file handle then prevents NumPy from adding an extension
        # or replacing an existing sample.
        archive = BytesIO()
        np.savez_compressed(
            archive,
            capture=capture_array,
            tap_window=tap_window_array,
            metadata_json=np.asarray(metadata_json),
        )
        archive_bytes = archive.getvalue()
        sample_created = False
        try:
            with sample_path.open("xb") as sample_file:
                sample_created = True
                bytes_written = sample_file.write(archive_bytes)
                if bytes_written != len(archive_bytes):
                    raise OSError("The complete NPZ archive could not be written.")
            _append_json_line(self.manifest_path, complete_metadata)
        except BaseException:
            if sample_created:
                try:
                    sample_path.unlink()
                except OSError as cleanup_error:
                    raise OSError(
                        "Sample indexing failed and the orphan artifact could not "
                        f"be removed: {sample_path} ({cleanup_error})"
                    ) from cleanup_error
            raise
        return sample_path.resolve(), complete_metadata

    def log_rejected_attempt(self, metadata: Mapping[str, Any]) -> None:
        """Append metadata for a rejected attempt without retaining its audio."""

        record = {
            **dict(metadata),
            "accepted": False,
            "waveform_saved": False,
        }
        _append_json_line(self.rejected_attempts_path, record)


def create_dataset_session(
    dataset_root: Path,
    metadata: Mapping[str, Any],
    *,
    session_id: str | None = None,
    created_at_utc: str | None = None,
) -> DatasetSession:
    """Create a unique session directory and its append-only index files."""

    created_at = created_at_utc or _utc_now()
    resolved_session_id = session_id or _new_session_id()
    _validate_safe_name(resolved_session_id, "session identifier")

    session_metadata = {
        **dict(metadata),
        "schema_version": DATASET_SCHEMA_VERSION,
        "session_id": resolved_session_id,
        "created_at_utc": created_at,
    }
    serialized_metadata = _serialize_json(session_metadata, indent=2)

    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / resolved_session_id
    directory.mkdir(exist_ok=False)
    samples_directory = directory / "samples"
    samples_directory.mkdir()

    session_path = directory / "session.json"
    manifest_path = directory / "manifest.jsonl"
    rejected_attempts_path = directory / "rejected_attempts.jsonl"
    _write_text_exclusive(session_path, serialized_metadata + "\n")
    _write_text_exclusive(manifest_path, "")
    _write_text_exclusive(rejected_attempts_path, "")

    return DatasetSession(
        session_id=resolved_session_id,
        directory=directory.resolve(),
        samples_directory=samples_directory.resolve(),
        manifest_path=manifest_path.resolve(),
        rejected_attempts_path=rejected_attempts_path.resolve(),
    )


def _float32_audio_matrix(samples: Any, label: str) -> np.ndarray[Any, Any]:
    try:
        audio = np.asarray(samples, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label.capitalize()} must be numeric audio data.") from error
    if audio.ndim != 2:
        raise ValueError(f"{label.capitalize()} must be a two-dimensional array.")
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError(f"{label.capitalize()} must not be empty.")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{label.capitalize()} contains non-finite samples.")
    return np.ascontiguousarray(audio)


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    serialized = _serialize_json(value, indent=None)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(serialized)
        output.write("\n")


def _serialize_json(value: Mapping[str, Any], *, indent: int | None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
        sort_keys=True,
    )


def _write_text_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(content)


def _validate_safe_name(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or not _SAFE_NAME.fullmatch(value)
        or value.endswith(".")
        or value.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            f"{label.capitalize()} must use only letters, numbers, '.', '_', "
            "and '-', must start with a letter or number, and must be a valid "
            "Windows filename."
        )


def _new_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
