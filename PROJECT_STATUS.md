# DeskSense Project Status

Status captured: 2026-08-25

## Project purpose

DeskSense is a Windows-focused project whose long-term goal is to turn acoustic
desk taps into programmable input. The intended system would use a laptop's
built-in microphones to detect taps, determine tap zones, and trigger Windows
actions while adapting to different laptop audio hardware.

The project is inspired by the MIT-licensed macOS acoustic-input project
[Holo](https://github.com/JustinGamer191/Holo). DeskSense is a separate project;
Holo compatibility or equivalent behavior is not assumed.

## Evidence boundary

The project currently establishes Windows audio feasibility, provides tools for
microphone-channel characterization, and has produced initial evidence that
left/right tap position affects measured channel features on one Lenovo
endpoint. It does **not** yet establish reliable tap localization, localization
accuracy, cross-laptop compatibility, real-time tap detection, or reliable
action triggering.

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

## Current checkpoint result

WDM-KS device 18 exposes two active, meaningfully different endpoint channels,
and repeated controlled observations provide initial evidence that left/right
tap position changes measurable channel features. Classification and
localization accuracy have **not** been measured.

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

Status: **complete for initial Lenovo microphone/backend feasibility**

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

| Host API | Observed device | Reported input configuration | Current evidence |
| --- | ---: | --- | --- |
| MME | 1 | 4 channels, default 44.1 kHz; 44.1/48 kHz supported | Ch1/Ch2 duplicate-like; Ch3/Ch4 inactive |
| DirectSound | 5 | 4 channels, default 44.1 kHz | Enumerated; further testing paused |
| WASAPI | 9 | 2 channels, default 48 kHz | Two numerically identical, duplicate-like channels |
| WDM-KS | 18 | 2 channels at 48 kHz | Two active, meaningfully different channels; currently preferred |
| WDM-KS | 19 | 4 channels at 16 kHz | Enumerated; further testing paused |
| WDM-KS | 20 | 4 channels at 16 kHz | Enumerated; further testing paused |

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

### Controlled WASAPI characterization

A controlled tap was recorded through WASAPI device 9 at 48 kHz with two
active channels. The strongest transient was near 2.861 seconds. Channels 1 and
2 were numerically identical in both the whole recording and transient window:
Pearson correlation 1.000000, normalized difference energy 0.000000, 0.00 dB
level difference, and zero reported lag. The exploratory heuristic label was
`duplicate_like`.

Current interpretation: the normal WASAPI endpoint exposes duplicate-like
processed channels on this tested Lenovo and does not provide useful
inter-channel spatial information.

### WDM-KS device 18 characterization

WDM-KS device 18 (`Microphone Array 1 (Intel Smart Sound Technology / Intel SST
Microphone)`) exposes two active channels at 48 kHz. Repeated controlled
recordings showed substantial, broadly consistent channel differences.

A representative controlled left-side tap produced:

- Ch2 relative RMS versus Ch1: approximately -4.55 dB over the whole recording
  and -4.79 dB in the transient window.
- Transient Pearson correlation: approximately +0.135.
- Transient normalized difference energy: approximately 0.884.
- Reported transient lag: -48 samples.
- Exploratory heuristic label: `meaningfully_different`.

WDM-KS device 18 is therefore the current preferred Lenovo endpoint. This
means only that the endpoint exposes substantially different channel
information; it does not prove that its channels map directly to separate raw
physical microphones. Driver or DSP processing may still be involved.

### Initial left/right spatial evidence

A mirrored right-side tap through WDM-KS device 18 differed substantially from
the controlled left-side observations. Its strongest transient was near 2.498
seconds, with transient Ch2 relative RMS approximately -2.67 dB, Pearson
approximately -0.297, normalized difference energy approximately 1.283, and a
reported lag of +44 samples.

Four subsequent valid alternating LEFT/RIGHT recordings showed the same broad
grouping tendency:

- Left-side taps: Ch2 substantially weaker than Ch1, low positive Pearson
  correlation, normalized difference energy below approximately 1, and
  negative reported lag.
- Right-side taps: a smaller inter-channel RMS difference, negative Pearson
  correlation, normalized difference energy above approximately 1, and
  positive reported lag.

This is initial evidence that tap position affects measurable channel features
on this endpoint. It is not a classifier-accuracy result: the sample is very
small, the separation was observed after viewing the data, no held-out
evaluation exists, tap force and position varied, the transient finder is
exploratory, and the waveform samples were discarded.

### Open lag-analysis issue

The current normalized cross-correlation search is bounded to +/-48 samples
(+/-1 ms at 48 kHz). Several left-side results reached exactly -48 samples,
while some right-side results approached or reached the positive boundary.
These values must not be interpreted as established physical acoustic
time-of-arrival measurements.

Future work should consider a several-millisecond exploratory search, explicit
boundary-hit flags, and analysis of retained waveform windows to determine
whether lag is genuinely useful.

## Current technical question

The primary engineering question is now:

> Can unseen desk taps be reliably classified into spatial zones?

Broad Windows endpoint exploration is paused. WDM-KS device 18 at 48 kHz with
two active, meaningfully different channels is the selected endpoint for the
next Lenovo experiments. DirectSound and WDM-KS devices 19 and 20 should not be
tested unless later evidence provides a reason.

The next milestone is **Phase 2A — guided labeled tap dataset collection**. Its
initial target is approximately 20 LEFT and 20 RIGHT taps with natural small
within-zone variation. The collection workflow should:

- Use explicit countdown and tap cues.
- Control or alternate collection order to reduce time/order bias.
- Retain short labeled multichannel waveform windows locally for later feature
  extraction.
- Store reproducibility metadata and basic capture-quality results.
- Keep raw/local dataset artifacts out of Git by default.
- Avoid prematurely introducing a GUI or complex machine-learning model.
- Evaluate held-out samples rather than calling training-set separation
  accuracy.

Phase 2 should study and adapt suitable ideas from the MIT-licensed Holo project
with attribution where useful instead of rebuilding algorithms unnecessarily.

## Roadmap

| Phase | Status |
| --- | --- |
| 1 — Windows hardware feasibility | Complete |
| 1.5 — microphone/backend characterization | Complete for initial Lenovo feasibility |
| 2A — guided labeled LEFT/RIGHT tap dataset | Next |
| 2B — held-out spatial-feasibility analysis | Not started |
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
- Phase 2A is expected to retain short waveform windows locally; those future
  dataset artifacts must remain out of Git by default.
- The detailed experiment chronology and evidence-retention notes are in
  [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).
