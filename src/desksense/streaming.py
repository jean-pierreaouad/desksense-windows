"""Pure, chunk-invariant streaming tap candidate detection.

Phase 3A.1 deliberately keeps audio hardware outside this module.  The
detector consumes arbitrary sequential frame-by-channel chunks, but performs
all noise updates and onset decisions on fixed internal blocks.  Its defaults
are initial engineering values for later physical validation, not calibrated
Lenovo thresholds.

The causal candidate-selection rule is new: an onset opens a bounded local
search, that search selects a strongest transient center, and the detector
then copies the exact 200 ms window around that center.  This is not identical
to Phase 2A's global search over a complete 1.5 second guided capture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class DetectorState(str, Enum):
    """Observable states of the pure streaming detector."""

    LEARNING = "learning"
    ARMED = "armed"
    COLLECTING = "collecting"
    REFRACTORY = "refractory"


@dataclass(frozen=True)
class StreamingDetectorConfig:
    """Reviewable Phase 3A.1 detector geometry and initial thresholds.

    Durations are converted to frame counts once the detector is created.
    Detection thresholds below are intentionally conservative starting points;
    they have not been tuned against the consumed Phase 2C external labels and
    are not yet validated for continuous Lenovo microphone operation.
    """

    sample_rate_hz: float = 48_000.0
    channel_count: int = 2
    internal_block_seconds: float = 0.005
    startup_learning_seconds: float = 0.750
    center_search_pre_onset_seconds: float = 0.012
    center_search_post_onset_seconds: float = 0.025
    tap_window_seconds: float = 0.200
    transient_energy_window_seconds: float = 0.005
    refractory_seconds: float = 0.250
    history_safety_seconds: float = 0.025
    minimum_noise_floor_rms: float = 1.0e-7
    onset_rms_noise_multiplier: float = 3.0
    onset_peak_noise_multiplier: float = 6.0
    minimum_onset_rms: float = 4.0 / 32_768.0
    minimum_onset_peak: float = 16.0 / 32_768.0
    minimum_crest_factor: float = 3.0
    noise_floor_adaptation_alpha: float = 0.02
    clipping_threshold: float = 0.98
    reject_near_clipping: bool = True

    def __post_init__(self) -> None:
        _positive_finite(self.sample_rate_hz, "sample rate")
        if (
            isinstance(self.channel_count, bool)
            or not isinstance(self.channel_count, int)
            or self.channel_count <= 0
        ):
            raise ValueError("Channel count must be a positive integer.")

        positive_durations = {
            "internal block duration": self.internal_block_seconds,
            "startup learning duration": self.startup_learning_seconds,
            "post-onset center-search duration": (
                self.center_search_post_onset_seconds
            ),
            "tap-window duration": self.tap_window_seconds,
            "transient-energy-window duration": (
                self.transient_energy_window_seconds
            ),
            "refractory duration": self.refractory_seconds,
        }
        for label, value in positive_durations.items():
            _positive_finite(value, label)
        for label, value in {
            "pre-onset center-search duration": (
                self.center_search_pre_onset_seconds
            ),
            "history safety duration": self.history_safety_seconds,
        }.items():
            _nonnegative_finite(value, label)

        positive_thresholds = {
            "minimum noise floor RMS": self.minimum_noise_floor_rms,
            "onset RMS/noise multiplier": self.onset_rms_noise_multiplier,
            "onset peak/noise multiplier": self.onset_peak_noise_multiplier,
            "minimum onset RMS": self.minimum_onset_rms,
            "minimum onset peak": self.minimum_onset_peak,
            "minimum crest factor": self.minimum_crest_factor,
            "clipping threshold": self.clipping_threshold,
        }
        for label, value in positive_thresholds.items():
            _positive_finite(value, label)
        if self.onset_rms_noise_multiplier < 1.0:
            raise ValueError("Onset RMS/noise multiplier must be at least 1.")
        if self.onset_peak_noise_multiplier < 1.0:
            raise ValueError("Onset peak/noise multiplier must be at least 1.")
        if self.minimum_crest_factor < 1.0:
            raise ValueError("Minimum crest factor must be at least 1.")
        if not 0.0 < float(self.noise_floor_adaptation_alpha) <= 1.0:
            raise ValueError(
                "Noise-floor adaptation alpha must be finite and in (0, 1]."
            )
        if float(self.clipping_threshold) > 1.0:
            raise ValueError(
                "Clipping threshold must not exceed normalized full scale."
            )
        if not isinstance(self.reject_near_clipping, bool):
            raise ValueError("Near-clipping rejection flag must be boolean.")

        internal_block_frames = self._frames(self.internal_block_seconds)
        if internal_block_frames <= 0:
            raise ValueError("Internal block duration must round to a frame.")
        frame_counts = {
            "internal block": internal_block_frames,
            "startup learning": self.startup_learning_frames,
            "post-onset center search": self.center_search_post_onset_frames,
            "tap window": self.tap_window_frames,
            "transient energy window": self.transient_energy_window_frames,
            "refractory interval": self.refractory_frames,
        }
        if any(value <= 0 for value in frame_counts.values()):
            raise ValueError("Configured positive durations must round to a frame.")
        if self.tap_window_frames % 2 != 0:
            raise ValueError(
                "Tap window must contain an even number of frames for exact "
                "centered extraction."
            )
        if self.transient_energy_window_frames > self.center_search_frames:
            raise ValueError(
                "Transient energy window must fit within the center-search region."
            )

    def _frames(self, seconds: float) -> int:
        return int(round(float(self.sample_rate_hz) * float(seconds)))

    @property
    def internal_block_frames(self) -> int:
        return self._frames(self.internal_block_seconds)

    @property
    def startup_learning_frames(self) -> int:
        requested = self._frames(self.startup_learning_seconds)
        blocks = math.ceil(requested / self.internal_block_frames)
        return int(blocks * self.internal_block_frames)

    @property
    def center_search_pre_onset_frames(self) -> int:
        return self._frames(self.center_search_pre_onset_seconds)

    @property
    def center_search_post_onset_frames(self) -> int:
        return self._frames(self.center_search_post_onset_seconds)

    @property
    def center_search_frames(self) -> int:
        return (
            self.center_search_pre_onset_frames
            + self.center_search_post_onset_frames
        )

    @property
    def tap_window_frames(self) -> int:
        return self._frames(self.tap_window_seconds)

    @property
    def tap_window_pre_center_frames(self) -> int:
        return self.tap_window_frames // 2

    @property
    def transient_energy_window_frames(self) -> int:
        return self._frames(self.transient_energy_window_seconds)

    @property
    def refractory_frames(self) -> int:
        return self._frames(self.refractory_seconds)

    @property
    def history_safety_frames(self) -> int:
        return self._frames(self.history_safety_seconds)

    @property
    def history_capacity_frames(self) -> int:
        """Return the fixed circular-history capacity.

        The first term preserves a completed exact tap window despite fixed
        block finalization overshoot.  The second preserves the earliest
        possible window start until the bounded center search completes.  A
        separate reviewable safety interval is then added.
        """

        completion_requirement = (
            self.tap_window_frames + self.internal_block_frames
        )
        search_requirement = (
            self.tap_window_pre_center_frames
            + self.center_search_pre_onset_frames
            + self.center_search_post_onset_frames
            + self.internal_block_frames
        )
        return int(
            max(completion_requirement, search_requirement)
            + self.history_safety_frames
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe configuration and derived frame counts."""

        return {
            "sample_rate_hz": float(self.sample_rate_hz),
            "channel_count": int(self.channel_count),
            "internal_block_seconds": float(self.internal_block_seconds),
            "internal_block_frames": self.internal_block_frames,
            "startup_learning_seconds_requested": float(
                self.startup_learning_seconds
            ),
            "startup_learning_frames_effective": self.startup_learning_frames,
            "center_search_pre_onset_seconds": float(
                self.center_search_pre_onset_seconds
            ),
            "center_search_pre_onset_frames": (
                self.center_search_pre_onset_frames
            ),
            "center_search_post_onset_seconds": float(
                self.center_search_post_onset_seconds
            ),
            "center_search_post_onset_frames": (
                self.center_search_post_onset_frames
            ),
            "tap_window_seconds": float(self.tap_window_seconds),
            "tap_window_frames": self.tap_window_frames,
            "tap_window_pre_center_frames": (
                self.tap_window_pre_center_frames
            ),
            "transient_energy_window_seconds": float(
                self.transient_energy_window_seconds
            ),
            "transient_energy_window_frames": (
                self.transient_energy_window_frames
            ),
            "refractory_seconds": float(self.refractory_seconds),
            "refractory_frames": self.refractory_frames,
            "history_safety_seconds": float(self.history_safety_seconds),
            "history_capacity_frames": self.history_capacity_frames,
            "minimum_noise_floor_rms": float(self.minimum_noise_floor_rms),
            "onset_rms_noise_multiplier": float(
                self.onset_rms_noise_multiplier
            ),
            "onset_peak_noise_multiplier": float(
                self.onset_peak_noise_multiplier
            ),
            "minimum_onset_rms": float(self.minimum_onset_rms),
            "minimum_onset_peak": float(self.minimum_onset_peak),
            "minimum_crest_factor": float(self.minimum_crest_factor),
            "noise_floor_adaptation_alpha_per_fixed_block": float(
                self.noise_floor_adaptation_alpha
            ),
            "clipping_threshold": float(self.clipping_threshold),
            "reject_near_clipping": self.reject_near_clipping,
            "threshold_status": (
                "initial Phase 3A engineering defaults; not physically "
                "validated or tuned from Phase 2C external labels"
            ),
        }


@dataclass(frozen=True)
class DetectionResult:
    """One completed causal candidate or a structured candidate rejection."""

    status: str
    rejection_reasons: tuple[str, ...]
    stream_epoch: int
    onset_frame_index: int
    center_frame_index: int | None
    window_start_frame_index: int | None
    window_end_frame_index_exclusive: int | None
    finalized_after_frame_index_exclusive: int
    candidate_window: np.ndarray[Any, Any] | None
    metrics: dict[str, Any]
    state_before: DetectorState
    state_after: DetectorState


@dataclass
class _PendingCandidate:
    onset_frame_index: int
    search_start_frame_index: int
    search_end_frame_index_exclusive: int
    onset_metrics: dict[str, float]
    center_frame_index: int | None = None
    window_start_frame_index: int | None = None
    window_end_frame_index_exclusive: int | None = None
    transient_metrics: dict[str, Any] | None = None


class _CircularAudioHistory:
    """Fixed-capacity float32 frame history indexed within one stream epoch."""

    def __init__(self, capacity_frames: int, channel_count: int) -> None:
        self._storage = np.empty(
            (capacity_frames, channel_count), dtype=np.float32
        )
        self._capacity = capacity_frames
        self._channel_count = channel_count
        self.clear()

    @property
    def start_frame_index(self) -> int:
        return self._start_frame_index

    @property
    def end_frame_index_exclusive(self) -> int:
        return self._end_frame_index_exclusive

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def storage_bytes(self) -> int:
        return int(self._storage.nbytes)

    def clear(self) -> None:
        self._start_frame_index = 0
        self._end_frame_index_exclusive = 0
        self._frame_count = 0

    def append(self, frames: np.ndarray[Any, Any]) -> None:
        if frames.ndim != 2 or frames.shape[1] != self._channel_count:
            raise RuntimeError("Internal history received an invalid audio block.")
        frame_count = int(frames.shape[0])
        if frame_count == 0:
            return
        if frame_count > self._capacity:
            frames = frames[-self._capacity :]
            frame_count = self._capacity

        old_end = self._end_frame_index_exclusive
        new_end = old_end + frame_count
        write_position = old_end % self._capacity
        first_count = min(frame_count, self._capacity - write_position)
        self._storage[write_position : write_position + first_count] = frames[
            :first_count
        ]
        remaining = frame_count - first_count
        if remaining:
            self._storage[:remaining] = frames[first_count:]

        self._end_frame_index_exclusive = new_end
        self._frame_count = min(self._capacity, self._frame_count + frame_count)
        self._start_frame_index = new_end - self._frame_count

    def contains(self, start: int, end_exclusive: int) -> bool:
        return (
            start >= self._start_frame_index
            and end_exclusive <= self._end_frame_index_exclusive
            and start <= end_exclusive
        )

    def copy_range(self, start: int, end_exclusive: int) -> np.ndarray[Any, Any]:
        if not self.contains(start, end_exclusive):
            raise LookupError(
                "Requested audio range is outside retained streaming history."
            )
        frame_count = end_exclusive - start
        result = np.empty((frame_count, self._channel_count), dtype=np.float32)
        if frame_count == 0:
            return result
        read_position = start % self._capacity
        first_count = min(frame_count, self._capacity - read_position)
        result[:first_count] = self._storage[
            read_position : read_position + first_count
        ]
        remaining = frame_count - first_count
        if remaining:
            result[first_count:] = self._storage[:remaining]
        return np.ascontiguousarray(result)


class StreamingTapDetector:
    """Pure fixed-block streaming detector with exact centered extraction."""

    def __init__(self, config: StreamingDetectorConfig | None = None) -> None:
        self.config = config or StreamingDetectorConfig()
        self._history = _CircularAudioHistory(
            self.config.history_capacity_frames,
            self.config.channel_count,
        )
        self._stream_epoch = 0
        self._initialize_epoch()

    @property
    def state(self) -> DetectorState:
        return self._state

    @property
    def stream_epoch(self) -> int:
        return self._stream_epoch

    @property
    def processed_frame_count(self) -> int:
        return self._processed_frame_count

    @property
    def partial_block_frame_count(self) -> int:
        return int(self._partial_block.shape[0])

    @property
    def retained_history_frame_count(self) -> int:
        return self._history.frame_count

    @property
    def buffered_frame_count(self) -> int:
        return self.retained_history_frame_count + self.partial_block_frame_count

    @property
    def maximum_buffered_frames(self) -> int:
        return (
            self.config.history_capacity_frames
            + self.config.internal_block_frames
            - 1
        )

    @property
    def history_storage_bytes(self) -> int:
        return self._history.storage_bytes

    @property
    def maximum_buffered_audio_bytes(self) -> int:
        return int(
            self.maximum_buffered_frames
            * self.config.channel_count
            * np.dtype(np.float32).itemsize
        )

    @property
    def noise_floor_rms(self) -> float:
        return float(self._noise_floor_rms)

    def reset(self) -> None:
        """Return to the same deterministic state as a new detector."""

        self._stream_epoch = 0
        self._initialize_epoch()

    def notify_discontinuity(self) -> None:
        """Start a new continuous epoch after a known stream gap.

        Absolute frame indexes are relative to a continuous epoch because a
        future audio adapter may not know how many frames an overflow lost.
        Clearing all history and learned state prevents a candidate from
        joining samples across that unknown gap.
        """

        self._stream_epoch += 1
        self._initialize_epoch()

    def process_chunk(self, samples: Any) -> tuple[DetectionResult, ...]:
        """Process one sequential chunk without depending on its partitioning.

        Invalid chunks raise ``ValueError`` before any detector state changes.
        Empty two-dimensional chunks with the configured channel count are
        valid no-ops.
        """

        chunk = _validated_float32_chunk(samples, self.config.channel_count)
        if chunk.shape[0] == 0:
            return ()

        if self._partial_block.size:
            working = np.concatenate((self._partial_block, chunk), axis=0)
        else:
            working = chunk

        block_frames = self.config.internal_block_frames
        complete_frame_count = (working.shape[0] // block_frames) * block_frames
        results: list[DetectionResult] = []
        for start in range(0, complete_frame_count, block_frames):
            block = working[start : start + block_frames]
            results.extend(self._process_fixed_block(block))

        self._partial_block = np.ascontiguousarray(
            working[complete_frame_count:].copy()
        )
        return tuple(results)

    def _initialize_epoch(self) -> None:
        self._state = DetectorState.LEARNING
        self._history.clear()
        self._partial_block = np.empty(
            (0, self.config.channel_count), dtype=np.float32
        )
        self._processed_frame_count = 0
        self._learning_power_sum = 0.0
        self._learning_frame_count = 0
        self._noise_floor_rms = float(self.config.minimum_noise_floor_rms)
        self._candidate: _PendingCandidate | None = None
        self._refractory_until_frame_index_exclusive: int | None = None

    def _process_fixed_block(
        self, block: np.ndarray[Any, Any]
    ) -> list[DetectionResult]:
        block_start = self._processed_frame_count
        self._history.append(block)
        self._processed_frame_count += int(block.shape[0])
        block_end = self._processed_frame_count

        block_rms, block_peak, crest_factor, frame_peaks = _block_metrics(block)

        if self._state is DetectorState.LEARNING:
            self._learning_power_sum += block_rms * block_rms * block.shape[0]
            self._learning_frame_count += int(block.shape[0])
            if self._learning_frame_count >= self.config.startup_learning_frames:
                learned = math.sqrt(
                    self._learning_power_sum / self._learning_frame_count
                )
                self._noise_floor_rms = max(
                    float(self.config.minimum_noise_floor_rms), float(learned)
                )
                self._state = DetectorState.ARMED
            return []

        if self._state is DetectorState.REFRACTORY:
            assert self._refractory_until_frame_index_exclusive is not None
            if block_start < self._refractory_until_frame_index_exclusive:
                if block_end >= self._refractory_until_frame_index_exclusive:
                    self._state = DetectorState.ARMED
                    self._refractory_until_frame_index_exclusive = None
                return []
            self._state = DetectorState.ARMED
            self._refractory_until_frame_index_exclusive = None

        if self._state is DetectorState.COLLECTING:
            result = self._advance_candidate()
            return [] if result is None else [result]

        rms_threshold = max(
            float(self.config.minimum_onset_rms),
            self._noise_floor_rms
            * float(self.config.onset_rms_noise_multiplier),
        )
        peak_threshold = max(
            float(self.config.minimum_onset_peak),
            self._noise_floor_rms
            * float(self.config.onset_peak_noise_multiplier),
        )
        triggered = (
            block_rms >= rms_threshold
            and block_peak >= peak_threshold
            and crest_factor >= float(self.config.minimum_crest_factor)
        )
        if not triggered:
            alpha = float(self.config.noise_floor_adaptation_alpha)
            self._noise_floor_rms = max(
                float(self.config.minimum_noise_floor_rms),
                float((1.0 - alpha) * self._noise_floor_rms + alpha * block_rms),
            )
            return []

        crossing_offsets = np.flatnonzero(frame_peaks >= peak_threshold)
        if crossing_offsets.size == 0:
            # The block peak gate above makes this unreachable unless floating
            # point inputs violate their own computed maximum.
            return []
        onset = block_start + int(crossing_offsets[0])
        self._candidate = _PendingCandidate(
            onset_frame_index=onset,
            search_start_frame_index=(
                onset - self.config.center_search_pre_onset_frames
            ),
            search_end_frame_index_exclusive=(
                onset + self.config.center_search_post_onset_frames
            ),
            onset_metrics={
                "learned_noise_floor_rms": float(self._noise_floor_rms),
                "onset_block_rms": float(block_rms),
                "onset_peak_absolute": float(block_peak),
                "onset_crest_factor": float(crest_factor),
                "onset_rms_threshold": float(rms_threshold),
                "onset_peak_threshold": float(peak_threshold),
            },
        )
        self._state = DetectorState.COLLECTING
        result = self._advance_candidate()
        return [] if result is None else [result]

    def _advance_candidate(self) -> DetectionResult | None:
        candidate = self._candidate
        if candidate is None:
            raise RuntimeError("Collecting state has no pending candidate.")

        if candidate.center_frame_index is None:
            if (
                self._processed_frame_count
                < candidate.search_end_frame_index_exclusive
            ):
                return None
            if not self._history.contains(
                candidate.search_start_frame_index,
                candidate.search_end_frame_index_exclusive,
            ):
                return self._complete_history_rejection(
                    candidate,
                    "required_history_unavailable",
                )
            search_audio = self._history.copy_range(
                candidate.search_start_frame_index,
                candidate.search_end_frame_index_exclusive,
            )
            local_center, transient_metrics = _strongest_transient_center(
                search_audio,
                self.config.transient_energy_window_frames,
            )
            candidate.center_frame_index = (
                candidate.search_start_frame_index + local_center
            )
            candidate.window_start_frame_index = (
                candidate.center_frame_index
                - self.config.tap_window_pre_center_frames
            )
            candidate.window_end_frame_index_exclusive = (
                candidate.window_start_frame_index
                + self.config.tap_window_frames
            )
            candidate.transient_metrics = transient_metrics
            if candidate.window_start_frame_index < self._history.start_frame_index:
                return self._complete_history_rejection(
                    candidate,
                    "required_history_unavailable",
                )

        assert candidate.window_end_frame_index_exclusive is not None
        if (
            self._processed_frame_count
            < candidate.window_end_frame_index_exclusive
        ):
            return None
        assert candidate.window_start_frame_index is not None
        if not self._history.contains(
            candidate.window_start_frame_index,
            candidate.window_end_frame_index_exclusive,
        ):
            return self._complete_history_rejection(
                candidate,
                "required_history_unavailable",
            )

        window = self._history.copy_range(
            candidate.window_start_frame_index,
            candidate.window_end_frame_index_exclusive,
        )
        clipped_count = int(
            np.count_nonzero(
                np.abs(window) >= float(self.config.clipping_threshold)
            )
        )
        reasons: tuple[str, ...] = ()
        status = "detected"
        if self.config.reject_near_clipping and clipped_count:
            status = "rejected"
            reasons = ("near_clipping",)

        metrics = self._result_metrics(candidate)
        metrics.update(
            {
                "candidate_frame_count": int(window.shape[0]),
                "candidate_channel_count": int(window.shape[1]),
                "candidate_peak_absolute": float(np.max(np.abs(window))),
                "clipping_threshold": float(self.config.clipping_threshold),
                "clipped_sample_count": clipped_count,
                "clipped_sample_fraction": float(clipped_count / window.size),
            }
        )
        return self._complete_candidate(
            candidate,
            status=status,
            rejection_reasons=reasons,
            candidate_window=window,
            metrics=metrics,
        )

    def _complete_history_rejection(
        self,
        candidate: _PendingCandidate,
        reason: str,
    ) -> DetectionResult:
        metrics = self._result_metrics(candidate)
        metrics.update(
            {
                "retained_history_start_frame_index": (
                    self._history.start_frame_index
                ),
                "retained_history_end_frame_index_exclusive": (
                    self._history.end_frame_index_exclusive
                ),
            }
        )
        return self._complete_candidate(
            candidate,
            status="rejected",
            rejection_reasons=(reason,),
            candidate_window=None,
            metrics=metrics,
        )

    def _result_metrics(self, candidate: _PendingCandidate) -> dict[str, Any]:
        metrics: dict[str, Any] = dict(candidate.onset_metrics)
        metrics.update(
            {
                "center_search_start_frame_index": (
                    candidate.search_start_frame_index
                ),
                "center_search_end_frame_index_exclusive": (
                    candidate.search_end_frame_index_exclusive
                ),
                "center_search_pre_onset_frames": (
                    self.config.center_search_pre_onset_frames
                ),
                "center_search_post_onset_frames": (
                    self.config.center_search_post_onset_frames
                ),
                "transient_energy_window_frames": (
                    self.config.transient_energy_window_frames
                ),
                "candidate_selection_is_phase2a_identical": False,
            }
        )
        if candidate.transient_metrics is not None:
            metrics.update(candidate.transient_metrics)
        if candidate.center_frame_index is not None:
            latency_frames = (
                self._processed_frame_count - candidate.center_frame_index
            )
            metrics.update(
                {
                    "detector_latency_frames": int(latency_frames),
                    "detector_latency_seconds": float(
                        latency_frames / float(self.config.sample_rate_hz)
                    ),
                }
            )
        return metrics

    def _complete_candidate(
        self,
        candidate: _PendingCandidate,
        *,
        status: str,
        rejection_reasons: tuple[str, ...],
        candidate_window: np.ndarray[Any, Any] | None,
        metrics: dict[str, Any],
    ) -> DetectionResult:
        anchor = (
            candidate.center_frame_index
            if candidate.center_frame_index is not None
            else candidate.onset_frame_index
        )
        refractory_until = anchor + self.config.refractory_frames
        metrics.update(
            {
                "refractory_anchor": (
                    "selected transient center"
                    if candidate.center_frame_index is not None
                    else "detected onset because no center was available"
                ),
                "refractory_anchor_frame_index": int(anchor),
                "refractory_frames": self.config.refractory_frames,
                "refractory_until_frame_index_exclusive": int(
                    refractory_until
                ),
                "refractory_timing_note": (
                    "Onset evaluation resumes on the first fixed detector "
                    "block boundary at or after this absolute frame."
                ),
            }
        )
        result = DetectionResult(
            status=status,
            rejection_reasons=rejection_reasons,
            stream_epoch=self._stream_epoch,
            onset_frame_index=candidate.onset_frame_index,
            center_frame_index=candidate.center_frame_index,
            window_start_frame_index=candidate.window_start_frame_index,
            window_end_frame_index_exclusive=(
                candidate.window_end_frame_index_exclusive
            ),
            finalized_after_frame_index_exclusive=self._processed_frame_count,
            candidate_window=candidate_window,
            metrics=metrics,
            state_before=DetectorState.COLLECTING,
            state_after=DetectorState.REFRACTORY,
        )
        self._refractory_until_frame_index_exclusive = refractory_until
        self._candidate = None
        self._state = DetectorState.REFRACTORY
        return result


def _block_metrics(
    block: np.ndarray[Any, Any],
) -> tuple[float, float, float, np.ndarray[Any, Any]]:
    working = np.asarray(block, dtype=np.float64)
    per_frame_power = np.mean(np.square(working), axis=1)
    block_rms = float(math.sqrt(max(0.0, float(np.mean(per_frame_power)))))
    frame_peaks = np.max(np.abs(working), axis=1)
    block_peak = float(np.max(frame_peaks))
    if block_rms <= np.finfo(np.float64).tiny:
        crest_factor = 0.0 if block_peak == 0.0 else math.inf
    else:
        crest_factor = float(block_peak / block_rms)
    return block_rms, block_peak, crest_factor, frame_peaks


def _strongest_transient_center(
    search_audio: np.ndarray[Any, Any],
    energy_window_frames: int,
) -> tuple[int, dict[str, Any]]:
    """Select an earliest-tie strongest center on a processed copy."""

    working = np.asarray(search_audio, dtype=np.float64)
    centered = working - np.mean(working, axis=0, dtype=np.float64)
    pooled_power = np.mean(np.square(centered), axis=1)
    prefix = np.empty(pooled_power.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(pooled_power, dtype=np.float64, out=prefix[1:])
    moving_energy = (
        prefix[energy_window_frames:] - prefix[:-energy_window_frames]
    )
    energy_start = int(np.argmax(moving_energy))
    energy_end = energy_start + energy_window_frames
    peak_offset = int(np.argmax(pooled_power[energy_start:energy_end]))
    center = energy_start + peak_offset
    return center, {
        "transient_energy_region_start_offset": energy_start,
        "transient_energy_region_end_offset_exclusive": energy_end,
        "transient_energy_window_mean_square": float(
            moving_energy[energy_start] / energy_window_frames
        ),
        "transient_center_offset_in_search": center,
        "transient_center_pooled_power": float(pooled_power[center]),
        "transient_selection_dc_removed_copy": True,
        "transient_ties_choose_earliest": True,
    }


def _validated_float32_chunk(
    samples: Any, expected_channels: int
) -> np.ndarray[Any, Any]:
    try:
        initial = np.asarray(samples)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Audio chunks must contain real numeric samples.") from error
    if initial.ndim != 2:
        raise ValueError("Audio chunks must be two-dimensional frames-by-channels arrays.")
    if initial.shape[1] != expected_channels:
        raise ValueError(
            f"Audio chunk has {initial.shape[1]} channel(s); expected "
            f"{expected_channels}."
        )
    if np.issubdtype(initial.dtype, np.complexfloating):
        raise ValueError("Audio chunks must contain real, not complex, samples.")
    try:
        converted = np.array(initial, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Audio chunks must contain real numeric samples.") from error
    if not np.all(np.isfinite(converted)):
        raise ValueError("Audio chunks must contain only finite samples.")
    return converted


def _positive_finite(value: Any, label: str) -> float:
    converted = _nonnegative_finite(value, label)
    if converted <= 0.0:
        raise ValueError(f"{label.capitalize()} must be positive and finite.")
    return converted


def _nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label.capitalize()} must be finite and non-negative.")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{label.capitalize()} must be finite and non-negative."
        ) from error
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label.capitalize()} must be finite and non-negative.")
    return converted
