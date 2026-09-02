"""Versioned local persistence for Phase 3 robustness evidence.

This schema is deliberately separate from the accepted-tap Phase 2A schema.
Every structurally valid intended positive attempt is retained, regardless of
whether the current streaming detector later finds a candidate.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np


ROBUSTNESS_SCHEMA_VERSION = 1
ROBUSTNESS_DATASET_KIND = "phase3_robustness_evidence_dataset"
POSITIVE_RECORD_TYPE = "positive_intended_tap"
NEGATIVE_RECORD_TYPE = "negative_activity_segment"
ROBUSTNESS_SAMPLE_RATE_HZ = 48_000.0
ROBUSTNESS_CHANNEL_COUNT = 2
ROBUSTNESS_DTYPE = "float32"
ROBUSTNESS_POSITIVE_ZONES = ("LEFT", "RIGHT")
ROBUSTNESS_POSITIVE_STRENGTHS = ("light", "normal", "firm")
ROBUSTNESS_NEGATIVE_ACTIVITIES = (
    "quiet",
    "typing",
    "speech",
    "trackpad",
    "hand_movement",
    "laptop_movement",
    "desk_object_interaction",
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class RobustnessDatasetError(RuntimeError):
    """A malformed, inconsistent, or incomplete robustness dataset."""


@dataclass(frozen=True)
class RobustnessRecord:
    """One validated manifest record and its lossless capture."""

    metadata: dict[str, Any]
    capture: np.ndarray[Any, Any]
    path: Path


@dataclass(frozen=True)
class LoadedRobustnessDataset:
    """A validated Phase 3 robustness session loaded read-only."""

    directory: Path
    session: dict[str, Any]
    records: tuple[RobustnessRecord, ...]

    @property
    def positive_records(self) -> tuple[RobustnessRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.metadata["record_type"] == POSITIVE_RECORD_TYPE
        )

    @property
    def negative_records(self) -> tuple[RobustnessRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.metadata["record_type"] == NEGATIVE_RECORD_TYPE
        )


@dataclass(frozen=True)
class RobustnessDatasetSession:
    """Exclusive writers for one append-only Phase 3 robustness session."""

    session_id: str
    directory: Path
    positives_directory: Path
    negatives_directory: Path
    manifest_path: Path

    def save_record(
        self,
        *,
        record_id: str,
        filename_stem: str,
        record_type: str,
        capture: Any,
        metadata: Mapping[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        """Persist one capture without replacing any existing artifact."""

        _validate_safe_name(record_id, "record identifier")
        _validate_safe_name(filename_stem, "record filename")
        if record_type not in {POSITIVE_RECORD_TYPE, NEGATIVE_RECORD_TYPE}:
            raise ValueError("Unsupported robustness record type.")
        audio = validate_robustness_capture(capture)
        subdirectory = (
            "positives" if record_type == POSITIVE_RECORD_TYPE else "negatives"
        )
        relative_path = Path(subdirectory) / f"{filename_stem}.npz"
        complete_metadata = {
            **dict(metadata),
            "record_id": record_id,
            "record_type": record_type,
            "saved_capture_path": relative_path.as_posix(),
            "storage": {
                "format": "NumPy NPZ (lossless compressed)",
                "capture_array": "capture",
                "metadata_array": "metadata_json",
                "capture_shape": [int(value) for value in audio.shape],
                "dtype": ROBUSTNESS_DTYPE,
            },
        }
        _ensure_json_safe(complete_metadata)
        metadata_json = _serialize_json(complete_metadata, indent=None)
        archive = BytesIO()
        np.savez_compressed(
            archive,
            capture=audio,
            metadata_json=np.asarray(metadata_json),
        )
        archive_bytes = archive.getvalue()
        sample_path = self.directory / relative_path
        created = False
        try:
            with sample_path.open("xb") as output:
                created = True
                written = output.write(archive_bytes)
                if written != len(archive_bytes):
                    raise OSError("The complete robustness NPZ was not written.")
            _append_json_line(self.manifest_path, complete_metadata)
        except BaseException:
            if created:
                try:
                    sample_path.unlink()
                except OSError as cleanup_error:
                    raise OSError(
                        "Manifest indexing failed and the new robustness artifact "
                        f"could not be removed: {sample_path} ({cleanup_error})"
                    ) from cleanup_error
            raise
        return sample_path.resolve(), complete_metadata


def create_robustness_dataset_session(
    dataset_root: Path,
    metadata: Mapping[str, Any],
    *,
    session_id: str | None = None,
    created_at_utc: str | None = None,
) -> RobustnessDatasetSession:
    """Create an exclusive Phase 3 robustness session directory."""

    resolved_id = session_id or _new_session_id()
    _validate_safe_name(resolved_id, "session identifier")
    session_metadata = {
        **dict(metadata),
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "dataset_kind": ROBUSTNESS_DATASET_KIND,
        "project_phase": str(metadata.get("project_phase", "3B.0")),
        "session_id": resolved_id,
        "created_at_utc": created_at_utc or _utc_now(),
    }
    _ensure_json_safe(session_metadata)

    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / resolved_id
    directory.mkdir(exist_ok=False)
    positives = directory / "positives"
    negatives = directory / "negatives"
    positives.mkdir()
    negatives.mkdir()
    _write_text_exclusive(
        directory / "session.json",
        _serialize_json(session_metadata, indent=2) + "\n",
    )
    manifest = directory / "manifest.jsonl"
    _write_text_exclusive(manifest, "")
    return RobustnessDatasetSession(
        session_id=resolved_id,
        directory=directory.resolve(),
        positives_directory=positives.resolve(),
        negatives_directory=negatives.resolve(),
        manifest_path=manifest.resolve(),
    )


def validate_robustness_capture(samples: Any) -> np.ndarray[Any, Any]:
    """Return an owned contiguous float32 stereo capture or raise clearly."""

    try:
        audio = np.asarray(samples, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Robustness capture must be numeric float32 audio.") from error
    if audio.ndim != 2:
        raise ValueError("Robustness capture must be a frames-by-channels matrix.")
    if audio.shape[0] <= 0 or audio.shape[1] != ROBUSTNESS_CHANNEL_COUNT:
        raise ValueError("Robustness capture must be non-empty two-channel audio.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Robustness capture contains non-finite samples.")
    return np.ascontiguousarray(audio)


def load_robustness_dataset(session_path: Path) -> LoadedRobustnessDataset:
    """Strictly validate and load one Phase 3 robustness session read-only."""

    directory = Path(session_path).resolve()
    session_file = directory / "session.json"
    manifest_file = directory / "manifest.jsonl"
    if not session_file.is_file():
        raise RobustnessDatasetError(f"Missing robustness session.json: {session_file}")
    if not manifest_file.is_file():
        raise RobustnessDatasetError(f"Missing robustness manifest.jsonl: {manifest_file}")
    session = _read_json_object(session_file, "session metadata")
    _validate_session_metadata(session)

    records: list[RobustnessRecord] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    expected_order = 1
    try:
        manifest_lines = manifest_file.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RobustnessDatasetError(f"Could not read robustness manifest: {error}") from error
    for line_number, line in enumerate(manifest_lines, start=1):
        if not line.strip():
            raise RobustnessDatasetError(
                f"Robustness manifest line {line_number} is empty."
            )
        try:
            metadata = json.loads(line)
        except (json.JSONDecodeError, TypeError) as error:
            raise RobustnessDatasetError(
                f"Robustness manifest line {line_number} is invalid JSON: {error}"
            ) from error
        if not isinstance(metadata, dict):
            raise RobustnessDatasetError(
                f"Robustness manifest line {line_number} must be a JSON object."
            )
        _ensure_json_safe(metadata, error_type=RobustnessDatasetError)
        record_id = _required_string(metadata, "record_id")
        if record_id in seen_ids:
            raise RobustnessDatasetError(f"Duplicate robustness record ID: {record_id}")
        seen_ids.add(record_id)
        order_index = _required_positive_int(metadata, "collection_order_index")
        if order_index != expected_order:
            raise RobustnessDatasetError(
                "Robustness collection_order_index values must be contiguous in "
                f"manifest order; expected {expected_order}, found {order_index}."
            )
        expected_order += 1
        record_type = _required_string(metadata, "record_type")
        if record_type not in {POSITIVE_RECORD_TYPE, NEGATIVE_RECORD_TYPE}:
            raise RobustnessDatasetError(
                f"Unsupported robustness record type: {record_type}"
            )
        relative_text = _required_string(metadata, "saved_capture_path")
        if relative_text in seen_paths:
            raise RobustnessDatasetError(
                f"Duplicate robustness capture path: {relative_text}"
            )
        seen_paths.add(relative_text)
        relative_path = Path(relative_text)
        expected_parent = (
            "positives" if record_type == POSITIVE_RECORD_TYPE else "negatives"
        )
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.parent != Path(expected_parent)
            or relative_path.suffix.casefold() != ".npz"
        ):
            raise RobustnessDatasetError(
                f"Unsafe or inconsistent robustness capture path: {relative_text}"
            )
        capture_path = directory / relative_path
        if not capture_path.is_file():
            raise RobustnessDatasetError(
                f"Missing robustness capture for {record_id}: {capture_path}"
            )
        capture, embedded = _load_record_archive(capture_path)
        if embedded != metadata:
            raise RobustnessDatasetError(
                f"Embedded metadata does not match manifest for {record_id}."
            )
        _validate_record_metadata(metadata, session, capture)
        capture.setflags(write=False)
        records.append(
            RobustnessRecord(metadata=metadata, capture=capture, path=capture_path)
        )

    referenced = {str((directory / Path(path)).resolve()) for path in seen_paths}
    actual = {
        str(path.resolve())
        for subdirectory in (directory / "positives", directory / "negatives")
        for path in subdirectory.glob("*.npz")
    }
    if actual != referenced:
        extras = sorted(actual - referenced)
        raise RobustnessDatasetError(
            "Robustness dataset contains unreferenced NPZ artifacts: "
            + ", ".join(extras)
        )
    return LoadedRobustnessDataset(
        directory=directory,
        session=session,
        records=tuple(records),
    )


def _validate_session_metadata(session: Mapping[str, Any]) -> None:
    if session.get("schema_version") != ROBUSTNESS_SCHEMA_VERSION:
        raise RobustnessDatasetError("Unsupported robustness dataset schema version.")
    if session.get("dataset_kind") != ROBUSTNESS_DATASET_KIND:
        raise RobustnessDatasetError("Session is not a Phase 3 robustness dataset.")
    _required_string(session, "session_id")
    _required_string(session, "source_mode")
    _required_timestamp(session, "created_at_utc")
    _validate_endpoint_metadata(session.get("selected_endpoint"), "session")
    domain = session.get("capture_domain")
    if not isinstance(domain, Mapping):
        raise RobustnessDatasetError("Robustness session lacks capture_domain metadata.")
    if _required_finite_number(domain, "sample_rate_hz") != ROBUSTNESS_SAMPLE_RATE_HZ:
        raise RobustnessDatasetError("Robustness sample rate must be 48000 Hz.")
    if domain.get("channel_count") != ROBUSTNESS_CHANNEL_COUNT:
        raise RobustnessDatasetError("Robustness channel count must be two.")
    if domain.get("dtype") != ROBUSTNESS_DTYPE:
        raise RobustnessDatasetError("Robustness dtype must be float32.")
    _validate_design_metadata(
        session.get("positive_design"),
        count_key="requested_attempt_count",
        label="positive",
    )
    _validate_design_metadata(
        session.get("negative_design"),
        count_key="requested_segment_count",
        label="negative",
    )


def _validate_record_metadata(
    metadata: Mapping[str, Any],
    session: Mapping[str, Any],
    capture: np.ndarray[Any, Any],
) -> None:
    if metadata.get("session_id") != session["session_id"]:
        raise RobustnessDatasetError("Record session ID does not match session.json.")
    if _required_finite_number(metadata, "sample_rate_hz") != ROBUSTNESS_SAMPLE_RATE_HZ:
        raise RobustnessDatasetError("Record sample rate does not match robustness domain.")
    if metadata.get("channel_count") != ROBUSTNESS_CHANNEL_COUNT:
        raise RobustnessDatasetError("Record channel count does not match robustness domain.")
    if metadata.get("capture_dtype") != ROBUSTNESS_DTYPE:
        raise RobustnessDatasetError("Record dtype does not match robustness domain.")
    if metadata.get("capture_frames") != int(capture.shape[0]):
        raise RobustnessDatasetError("Record frame count does not match its NPZ capture.")
    duration = metadata.get("capture_duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or not math.isclose(
            float(duration),
            capture.shape[0] / ROBUSTNESS_SAMPLE_RATE_HZ,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RobustnessDatasetError("Record duration does not match its NPZ capture.")
    _validate_endpoint_metadata(metadata.get("device"), "record")
    if metadata.get("device") != session.get("selected_endpoint"):
        raise RobustnessDatasetError(
            "Record endpoint metadata does not match session.json."
        )
    storage = metadata.get("storage")
    if not isinstance(storage, Mapping):
        raise RobustnessDatasetError("Record storage metadata is missing.")
    if storage.get("capture_shape") != [int(value) for value in capture.shape]:
        raise RobustnessDatasetError("Stored capture shape metadata is inconsistent.")
    if metadata["record_type"] == POSITIVE_RECORD_TYPE:
        if metadata.get("intended_zone") not in ROBUSTNESS_POSITIVE_ZONES:
            raise RobustnessDatasetError("Positive record has an invalid intended zone.")
        if metadata.get("intended_strength") not in ROBUSTNESS_POSITIVE_STRENGTHS:
            raise RobustnessDatasetError("Positive record has an invalid strength.")
        _required_positive_int(metadata, "attempt_number_within_condition")
        _required_timestamp(metadata, "captured_at_utc")
        if metadata.get("semantic_label") != "intended_desk_tap":
            raise RobustnessDatasetError("Positive record semantic label is invalid.")
        if metadata.get("detector_acceptance_required_for_storage") is not False:
            raise RobustnessDatasetError(
                "Positive robustness evidence must not require detector acceptance."
            )
        cue = metadata.get("guided_cue")
        if not isinstance(cue, Mapping):
            raise RobustnessDatasetError("Positive guided cue metadata is missing.")
        cue_frame = _required_nonnegative_int(cue, "intended_cue_offset_frames")
        association = cue.get("intended_event_association")
        if not isinstance(association, Mapping):
            raise RobustnessDatasetError(
                "Positive intended-event association metadata is missing."
            )
        association_start = _required_nonnegative_int(
            association, "start_frame_index_inclusive"
        )
        association_end = _required_positive_int(
            association, "end_frame_index_exclusive"
        )
        association_seconds = _required_finite_number(
            association, "duration_seconds"
        )
        if (
            association_start != cue_frame
            or association_start >= association_end
            or association_end > capture.shape[0]
            or not math.isclose(
                association_seconds,
                (association_end - association_start) / ROBUSTNESS_SAMPLE_RATE_HZ,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RobustnessDatasetError(
                "Positive intended-event association boundaries are inconsistent."
            )
    else:
        if metadata.get("activity") not in ROBUSTNESS_NEGATIVE_ACTIVITIES:
            raise RobustnessDatasetError("Negative record has an invalid activity.")
        _required_positive_int(metadata, "repetition_index")
        started = _required_timestamp(metadata, "segment_started_at_utc")
        ended = _required_timestamp(metadata, "segment_ended_at_utc")
        if ended < started:
            raise RobustnessDatasetError(
                "Negative segment end timestamp precedes its start timestamp."
            )
        if metadata.get("semantic_label") != "no_intended_desk_tap":
            raise RobustnessDatasetError("Negative record semantic label is invalid.")
        warmup_seconds = _required_finite_number(
            metadata, "warmup_duration_seconds"
        )
        activity_seconds = _required_finite_number(
            metadata, "activity_duration_seconds"
        )
        warmup_end = _required_positive_int(metadata, "warmup_end_frame_index")
        activity_start = _required_nonnegative_int(
            metadata, "activity_start_frame_index"
        )
        activity_end = _required_positive_int(
            metadata, "activity_end_frame_index_exclusive"
        )
        tail_seconds = _required_finite_number(
            metadata, "post_activity_tail_duration_seconds"
        )
        tail_start = _required_nonnegative_int(
            metadata, "post_activity_start_frame_index"
        )
        tail_end = _required_positive_int(
            metadata, "post_activity_end_frame_index_exclusive"
        )
        if (
            warmup_seconds <= 0.0
            or activity_seconds <= 0.0
            or tail_seconds <= 0.0
            or warmup_end != activity_start
            or activity_start >= activity_end
            or activity_end != tail_start
            or tail_start >= tail_end
            or tail_end != capture.shape[0]
            or not math.isclose(
                warmup_seconds,
                warmup_end / ROBUSTNESS_SAMPLE_RATE_HZ,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                activity_seconds,
                (activity_end - activity_start) / ROBUSTNESS_SAMPLE_RATE_HZ,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                tail_seconds,
                (tail_end - tail_start) / ROBUSTNESS_SAMPLE_RATE_HZ,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RobustnessDatasetError(
                "Negative warm-up/activity/tail boundaries are inconsistent."
            )


def _load_record_archive(path: Path) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"capture", "metadata_json"}:
                raise RobustnessDatasetError(
                    f"Robustness NPZ has unexpected arrays: {path}"
                )
            capture = np.asarray(archive["capture"])
            metadata_value = archive["metadata_json"]
            metadata_json = str(metadata_value.item())
    except RobustnessDatasetError:
        raise
    except Exception as error:
        raise RobustnessDatasetError(
            f"Could not read robustness NPZ {path}: {error}"
        ) from error
    if capture.dtype != np.dtype(np.float32):
        raise RobustnessDatasetError(f"Robustness capture is not float32: {path}")
    try:
        capture = validate_robustness_capture(capture)
        metadata = json.loads(metadata_json)
    except (ValueError, json.JSONDecodeError, TypeError) as error:
        raise RobustnessDatasetError(
            f"Invalid robustness NPZ content in {path}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise RobustnessDatasetError(f"Embedded metadata is not an object: {path}")
    return capture, metadata


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RobustnessDatasetError(f"Could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise RobustnessDatasetError(f"{label.capitalize()} must be a JSON object.")
    _ensure_json_safe(value, error_type=RobustnessDatasetError)
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise RobustnessDatasetError(f"Robustness metadata field {key} is invalid.")
    return value


def _required_positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RobustnessDatasetError(f"Robustness metadata field {key} is invalid.")
    return value


def _required_nonnegative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RobustnessDatasetError(f"Robustness metadata field {key} is invalid.")
    return value


def _required_finite_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise RobustnessDatasetError(f"Robustness metadata field {key} is invalid.")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RobustnessDatasetError(
            f"Robustness metadata field {key} is invalid."
        ) from error
    if not math.isfinite(converted):
        raise RobustnessDatasetError(f"Robustness metadata field {key} is invalid.")
    return converted


def _required_timestamp(mapping: Mapping[str, Any], key: str) -> datetime:
    value = _required_string(mapping, key)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RobustnessDatasetError(
            f"Robustness metadata field {key} is not an ISO timestamp."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RobustnessDatasetError(
            f"Robustness metadata field {key} must include a UTC offset."
        )
    return parsed.astimezone(timezone.utc)


def _validate_endpoint_metadata(value: Any, owner: str) -> None:
    if not isinstance(value, Mapping):
        raise RobustnessDatasetError(
            f"Robustness {owner} endpoint metadata is missing."
        )
    _required_string(value, "name")
    index = value.get("index_at_collection")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise RobustnessDatasetError(
            f"Robustness {owner} endpoint index is invalid."
        )
    host_api = value.get("host_api")
    if not isinstance(host_api, Mapping):
        raise RobustnessDatasetError(
            f"Robustness {owner} host API metadata is missing."
        )
    _required_string(host_api, "name")


def _validate_design_metadata(
    value: Any,
    *,
    count_key: str,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise RobustnessDatasetError(
            f"Robustness {label} design metadata is missing."
        )
    expected_count = _required_nonnegative_int(value, count_key)
    plan = value.get("plan")
    if not isinstance(plan, list) or len(plan) != expected_count:
        raise RobustnessDatasetError(
            f"Robustness {label} design plan is inconsistent."
        )
    if any(not isinstance(item, Mapping) for item in plan):
        raise RobustnessDatasetError(
            f"Robustness {label} design plan entries must be objects."
        )
    for item in plan:
        _required_positive_int(item, "collection_order_index")
        if label == "positive":
            if item.get("zone") not in ROBUSTNESS_POSITIVE_ZONES:
                raise RobustnessDatasetError(
                    "Robustness positive design contains an invalid zone."
                )
            if item.get("strength") not in ROBUSTNESS_POSITIVE_STRENGTHS:
                raise RobustnessDatasetError(
                    "Robustness positive design contains an invalid strength."
                )
            _required_positive_int(item, "attempt_number_within_condition")
        else:
            if item.get("activity") not in ROBUSTNESS_NEGATIVE_ACTIVITIES:
                raise RobustnessDatasetError(
                    "Robustness negative design contains an invalid activity."
                )
            _required_positive_int(item, "repetition_index")


def _ensure_json_safe(
    value: Mapping[str, Any],
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise error_type(f"Metadata is not finite JSON-safe data: {error}") from error


def _serialize_json(value: Mapping[str, Any], *, indent: int | None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
        sort_keys=True,
    )


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    serialized = _serialize_json(value, indent=None)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(serialized)
        output.write("\n")


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
            f"{label.capitalize()} must be a portable Windows filename component."
        )


def _new_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
