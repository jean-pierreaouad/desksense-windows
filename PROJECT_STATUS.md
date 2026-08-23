# DeskSense Project Status

Status captured: 2026-08-23

## Project purpose

DeskSense is a Windows-focused project whose long-term goal is to turn acoustic
desk taps into programmable input. The intended system would use a laptop's
built-in microphones to detect taps, determine tap zones, and trigger Windows
actions while adapting to different laptop audio hardware.

The project is inspired by the MIT-licensed macOS acoustic-input project
[Holo](https://github.com/JustinGamer191/Holo). DeskSense is a separate project;
Holo compatibility or equivalent behavior is not assumed.

## Evidence boundary

The project currently establishes Windows audio feasibility and provides tools
for microphone-channel characterization. It does **not** yet establish tap
localization, localization accuracy, cross-laptop compatibility, real-time tap
detection, or reliable action triggering.

Real-hardware observations in this document apply only to the Lenovo Windows 11
development laptop and the tested endpoints and procedures.

## Development environment

| Component | Current environment |
| --- | --- |
| Operating system | Windows 11 AMD64 |
| Python | CPython 3.14.4, 64-bit |
| Virtual environment | `.venv` |
| Audio library | sounddevice 0.5.6 |
| Audio backend | PortAudio V19.7.0-devel |
| Numerical library | NumPy 2.5.2 |
| Test framework | pytest 9.1.1 |

## Delivered milestones

### Milestone 1 — Windows audio feasibility

Status: **complete**

Implemented:

- Windows and Python runtime diagnostics.
- Audio-input enumeration, default-device selection, and host API reporting.
- Maximum input-channel and default sample-rate reporting.
- 44.1 kHz and 48 kHz capability checks.
- Optional approximately three-second in-memory recording.
- Per-channel peak and RMS measurements.
- Machine-readable JSON diagnostic reports without raw audio.
- Hardware-independent tests using mocked sounddevice access.

Verification at completion:

- 25 tests passed.
- `pip check`, byte compilation, and CLI checks passed.

### Milestone 1.5 — microphone characterization

Status: **tooling complete; endpoint investigation ongoing**

Implemented:

- Explicit input selection by sounddevice device index.
- Approximately five-second in-memory characterization capture.
- Per-channel peak, RMS, DC offset, and standard-deviation measurements.
- Reviewable heuristic active/inactive assessment using absolute and relative
  signal levels.
- Pearson correlation and normalized difference-signal energy.
- Relative AC channel level.
- Bounded maximum normalized cross-correlation and lag in samples and
  microseconds.
- Exploratory strongest-transient window analysis.
- Machine-readable characterization reports without raw audio.
- Numerical, mocked audio, CLI, and JSON tests.

Verification at completion:

- Complete suite: 56 tests passed, including the Milestone 1 tests.
- `pip check`, byte compilation, and CLI checks passed.
- Generated reports remained ignored by Git.
- Previous Milestone 1 behavior was preserved.

## Lenovo audio endpoints observed

The Intel Smart Sound Technology microphone array is exposed through multiple
Windows audio APIs. The following indices describe one enumeration session and
must not be treated as stable identifiers.

| Host API | Observed device | Reported input configuration |
| --- | ---: | --- |
| MME | 1 | 4 channels, default 44.1 kHz; 44.1/48 kHz supported |
| DirectSound | 5 | 4 channels, default 44.1 kHz |
| WASAPI | 9 | 2 channels, default 48 kHz |
| WDM-KS | 18 | 2 channels at 48 kHz |
| WDM-KS | 19 | 4 channels at 16 kHz |
| WDM-KS | 20 | 4 channels at 16 kHz |

Device indices can change between machines, boots, device changes, and
PortAudio enumeration sessions. Reports therefore preserve the index together
with the endpoint name and host API.

## Real-hardware findings

### Initial MME recording and quiet baseline

The default MME endpoint exposed four channels. Channels 1 and 2 carried
meaningful signal; channels 3 and 4 remained near approximately one normalized
16-bit LSB and appeared effectively inactive.

| Capture | Ch1 peak / RMS | Ch2 peak / RMS | Ch3 and Ch4 |
| --- | --- | --- | --- |
| Fingertip desk tap | 0.015900 / 0.001093 | 0.015869 / 0.001092 | Peak about 0.000031; RMS about 0.000015 |
| Quiet baseline | 0.002808 / 0.000496 | 0.002808 / 0.000496 | Peak about 0.000031; RMS about 0.000015 |

This establishes only that an ordinary fingertip desk tap was measurable above
the observed quiet baseline on this laptop and endpoint.

### Controlled MME characterization

The laptop remained stationary on the same wooden desk. One moderate
fingertip-pad tap was made at a marked point approximately 7–10 cm left of the
front-left/palm-rest area. The strongest transient was centered near 2.393
seconds in the five-second capture.

- Capture: MME device 1, 44.1 kHz, four reported channels.
- Channels 1 and 2: active, with peaks of approximately 0.230896.
- Channels 3 and 4: effectively inactive.
- Whole-recording Ch1/Ch2 Pearson correlation: approximately 0.999992.
- Whole-recording normalized difference energy: approximately 0.000008.
- Whole-recording relative level: approximately 0 dB.
- Maximum normalized cross-correlation: approximately 0.999992.
- Estimated lag: 0 samples / 0 microseconds.
- Exploratory heuristic label: `duplicate_like`.
- Transient-window Ch1/Ch2 Pearson correlation: approximately 1.000000, with
  approximately zero difference energy, 0 dB relative level, and zero lag.

Current interpretation: for this Lenovo and the tested MME endpoint, channels 1
and 2 appear effectively duplicate-like or heavily shared-processed and do not
currently expose useful independent spatial cues. This does not prove that the
laptop has only one physical microphone; Windows, driver, or array DSP may be
responsible. MME is therefore not currently the preferred endpoint for
inter-channel localization investigation.

## Current technical question

The primary unresolved question is:

> Which Windows microphone endpoint, if any, exposes sufficiently independent
> spatial information from the Lenovo microphone array for reliable tap
> localization?

Planned characterization order, subject to new evidence:

1. WASAPI Intel SST endpoint.
2. WDM-KS microphone-array endpoints.
3. DirectSound, if it remains useful after the earlier results.

Not every endpoint must be tested if earlier evidence changes the architecture.
If no endpoint exposes useful independent channels, future investigation may
consider single-channel acoustic or spectral signatures, deeper Windows audio
access, device-specific array access, or a different sensing strategy. These
are possibilities, not current design decisions.

## Roadmap

| Phase | Status |
| --- | --- |
| 1 — Windows hardware feasibility | Complete |
| 1.5 — characterization tooling | Complete; endpoint investigation ongoing |
| 2 — controlled labeled tap dataset and localization feasibility | Not started |
| 3 — tap detection, features, classification, confidence/rejection | Not started |
| 4 — real-time DeskSense and Windows action mapping | Not started |
| 5 — cross-laptop hardware adaptation and testing | Not started |
| 6 — installer/UI if justified, benchmarks, demo, and release material | Not started |

No localization-accuracy claim should be made until controlled labeled data
demonstrates it.

## Repository and data-handling state

- Runtime dependencies remain limited to sounddevice and NumPy; pytest is the
  test dependency.
- Raw recordings are kept in memory and are not included in JSON reports.
- Generated reports under `reports/` are intentionally ignored by Git while
  `reports/.gitkeep` preserves the directory location.
- Earlier diagnostic and characterization reports may exist locally and are
  not project source artifacts.
- The detailed experiment chronology and evidence-retention notes are in
  [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).
