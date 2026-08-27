# DeskSense Project Status

Status captured: 2026-08-27

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
microphone-channel characterization, and includes a guided labeled-dataset
collector that has passed a small real-hardware pilot. Retained pilot waveforms
provide additional evidence that left/right tap position affects measured
channel features on one Lenovo endpoint. The project does **not** yet establish
reliable tap localization, localization accuracy, cross-laptop compatibility,
real-time tap detection, or reliable action triggering.

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

WDM-KS device 18 exposes two active, meaningfully different endpoint channels.
The Phase 2A collector is implemented and has successfully retained an
alternating 2 LEFT + 2 RIGHT pilot with complete multichannel captures and
tap-centered windows. The pilot adds encouraging spatial evidence, but the main
20 LEFT + 20 RIGHT dataset has not been collected. Classification and
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

### Phase 2A — guided labeled tap dataset collection

Status: **collector implemented and pilot-validated; main dataset collection
pending**

Implemented:

- Alternating LEFT/RIGHT collection with a silent terminal countdown.
- Microphone capture begins before `TAP NOW`, with an intended 0.200-second
  pre-cue interval inside the 1.5-second capture and a nominal 1.3 seconds
  remaining after the cue.
- Complete lossless float32 multichannel captures and exploratory 200 ms
  centered tap windows stored in local NPZ artifacts.
- Manual retry support; rejected attempts do not advance accepted sample
  numbering.
- Validation that all expected spatial channels are active.
- Conservative quiet/transient, clipping, finite-data, and boundary checks.
- Reproducible session, manifest, sample, endpoint, cue, and quality metadata.
- Local dataset storage under a Git-ignored dataset root; no upload or automatic
  commit behavior.

Automated verification before physical testing:

- Complete suite: 92 tests passed.
- Focused Phase 2A suite: 44 tests passed.
- `pip check`, byte compilation/import checks, CLI help, and
  `git diff --check` passed.
- `datasets/` was confirmed ignored by Git.
- Automated tests did not access real microphone hardware.

Physical validation:

- One guided alternating 2 LEFT + 2 RIGHT pilot completed successfully on
  WDM-KS device 18.
- All four captures were accepted on their first attempt and their retained
  artifacts were structurally verified.
- This pilot validates the collector workflow on the tested endpoint; it does
  not complete the intended main dataset or measure classification accuracy.

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
exploratory, and the waveform samples from those characterization runs were
discarded.

### Phase 2A guided 2+2 pilot

The collector was physically validated through WDM-KS device 18 at 48 kHz with
two channels. Session `20260826T215711.391122Z-4ed5556c` requested two LEFT and
two RIGHT samples in alternating order. LEFT #1, RIGHT #1, LEFT #2, and RIGHT #2
were accepted on attempts 1–4 respectively, with no retries.

Each accepted NPZ artifact contained:

- A 72,000 x 2 float32 complete capture, representing 1.5 seconds.
- A 9,600 x 2 float32 exploratory tap window, representing 200 ms.
- Embedded metadata matching the session manifest.

Capture began before the visual `TAP NOW` cue. The intended cue offset was
0.200 seconds / 9,600 samples, leaving a nominal 1.3 seconds after the cue. This
is a controlled intended offset, not a measurement of terminal rendering or
human reaction latency.

| Accepted sample | Strongest transient center | Approximate delay after intended cue | Transient/background contrast |
| --- | ---: | ---: | ---: |
| LEFT #1 | 1.164 s | 0.964 s | 34.47 dB |
| RIGHT #1 | 1.118 s | 0.918 s | 34.48 dB |
| LEFT #2 | 0.910 s | 0.710 s | 33.89 dB |
| RIGHT #2 | 0.843 s | 0.643 s | 34.43 dB |

All four taps occurred well inside their captures. Both channels were active,
no samples were clipped, and every attempt passed the conservative quality
checks. The 200 ms windows contained approximately 96–97% of the observed
full-capture AC signal energy in this pilot. This supports 200 ms as a
reasonable exploratory window for the next dataset, not as a proven optimum.

Offline measurements from the retained tap windows were:

| Accepted sample | Ch2 RMS vs Ch1 | Ch2 peak vs Ch1 |
| --- | ---: | ---: |
| LEFT #1 | -3.96 dB | -5.35 dB |
| RIGHT #1 | -0.18 dB | +1.18 dB |
| LEFT #2 | -3.93 dB | -4.92 dB |
| RIGHT #2 | +0.16 dB | +1.11 dB |

Same-zone waveform repeats were highly similar after small alignment, while
cross-zone similarity was materially lower. These four samples provide
encouraging additional evidence that position affects the two-channel acoustic
signature. They do not constitute a classifier evaluation or accuracy result.

### Lag-analysis correction and open issue

The earlier characterization analysis searched only +/-48 samples (+/-1 ms at
48 kHz). Several left-side results reached exactly -48 samples, while some
right-side results approached or reached the positive boundary. The resulting
apparent LEFT-negative / RIGHT-positive lag pattern should not be treated as a
reliable spatial cue.

Offline analysis of the newly retained 200 ms windows with a wider search found
the strongest inter-channel offset at approximately 123–126 samples
(approximately 2.56–2.63 ms) in the same direction across all four pilot
samples. The bounded earlier pattern was therefore likely influenced by the
search limit, and this pilot's raw lag results do not show useful left/right
sign separation.

The approximately 2.6 ms offset must not be interpreted as proven physical
microphone time of arrival because Windows, driver, or array DSP processing may
contribute. Phase 2B should determine whether lag has predictive value after a
larger dataset is available.

## Current technical question

The primary engineering question is now:

> Can unseen desk taps be reliably classified into spatial zones?

Broad Windows endpoint exploration is paused. WDM-KS device 18 at 48 kHz with
two active, meaningfully different channels is the selected endpoint for the
next Lenovo experiments. DirectSound and WDM-KS devices 19 and 20 should not be
tested unless later evidence provides a reason.

The Phase 2A collector implementation and small physical pilot are complete.
The immediate work is to checkpoint the validated collector and then use it to
collect the main dataset of approximately 20 LEFT and 20 RIGHT taps with
natural small within-zone variation. The delivered workflow:

- Uses explicit countdown and tap cues, with capture active before the cue.
- Alternates collection order to reduce time/order bias.
- Retains complete captures and short labeled multichannel waveform windows
  locally for later feature extraction.
- Stores reproducibility metadata and basic capture-quality results.
- Keeps raw/local dataset artifacts out of Git by default.

After the main dataset is collected, Phase 2B should evaluate held-out samples
before any accuracy claim; training-set separation must not be reported as
accuracy.

Phase 2 should study and adapt suitable ideas from the MIT-licensed Holo project
with attribution where useful instead of rebuilding algorithms unnecessarily.

## Roadmap

| Phase | Status |
| --- | --- |
| 1 — Windows hardware feasibility | Complete |
| 1.5 — microphone/backend characterization | Complete for initial Lenovo feasibility |
| 2A — guided labeled LEFT/RIGHT tap dataset | Collector complete and pilot-validated; main 20+20 dataset pending |
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
- Phase 2A explicitly retains accepted complete captures and tap windows in
  local dataset artifacts. The pilot session exists locally under `datasets/`,
  which is ignored by Git by default.
- Diagnostic and characterization commands continue to omit raw audio from
  their JSON reports; waveform retention occurs only in explicit dataset
  collection sessions.
- The detailed experiment chronology and evidence-retention notes are in
  [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).
