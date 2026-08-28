# DeskSense Project Status

Status captured: 2026-08-28

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
microphone-channel characterization and guided labeled-dataset collection, and
includes a reproducible offline LEFT/RIGHT evaluation pipeline. The main
20 LEFT + 20 RIGHT development dataset is complete, and a predeclared simple
baseline showed strong within-session separation of the two observed
interaction conditions.

That dataset contains a material hand/location confound: LEFT means left hand
at the left location, while RIGHT means right hand at the right location. The
result therefore does **not** establish pure spatial localization,
hand-independent localization accuracy, external-session generalization,
cross-laptop compatibility, real-time tap detection, or reliable action
triggering.

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
The Phase 2A collector and main alternating 20 LEFT + 20 RIGHT dataset are
complete. Phase 2B offline analysis is implemented and reviewed. On the main
development session, the predeclared Ch2/Ch1 peak-ratio baseline classified all
10 examples in a within-session chronological holdout correctly and all 40
examples across within-session leave-one-pair-out cross-validation correctly.

These results are scoped evidence for separation of the recorded LEFT/RIGHT
interaction conditions on one user, laptop, desk, setup, and session. They are
not pure left/right position accuracy or independent external validation. The
next evidence gate is a separately collected, untouched same-hand external
session evaluated with a baseline frozen beforehand.

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

Status: **collector implemented and validated; main 20 LEFT + 20 RIGHT dataset
complete**

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
- Main session `20260827T171528.349289Z-c0e2caa7` collected 20 accepted LEFT
  and 20 accepted RIGHT samples in alternating order.
- One RIGHT attempt was manually rejected and retried; the rejected waveform
  was not retained or treated as an accepted sample.
- All 40 accepted artifacts contained 72,000 x 2 float32 complete captures and
  9,600 x 2 float32 tap windows at 48 kHz. Both expected channels were active,
  no accepted capture clipped, and transient/background separation was strong.

### Phase 2B — reproducible offline spatial-feasibility analysis

Status: **pipeline implemented and reviewed; formal main-session
within-session analysis complete**

Implemented:

- Read-only dataset loading with session, manifest, NPZ, embedded-metadata,
  sample-order, shape, dtype, finite-data, sample-rate, and channel-count
  integrity validation.
- Deterministic NumPy-only feature extraction.
- A predeclared Ch2/Ch1 peak-amplitude ratio in dB as the primary baseline.
- Training-only midpoint threshold fitting with class direction inferred from
  training means.
- Within-session chronological holdout and leave-one-pair-out
  cross-validation.
- Machine-readable JSON reporting without waveform arrays.
- Explicit analyst-supplied evidence context, including the current
  hand/location confound.
- An offline CLI path that does not initialize microphone hardware.

Automated verification before the real analysis:

- Complete suite: 154 tests passed.
- Focused Phase 2B/CLI tests: 62 passed, 16 deselected.
- `pip check`, byte compilation/import checks, CLI help, and
  `git diff --check` passed.
- `datasets/` and generated reports remained ignored by Git.
- No microphone access occurred during analysis.

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

### Phase 2B main-session analysis

The offline pipeline validated all 40 accepted artifacts in session
`20260827T171528.349289Z-c0e2caa7`: 20 LEFT, 20 RIGHT, no orphan NPZ files, and
a dataset complete against the session request. The session used WDM-KS device
18 at collection time, 48 kHz, and two channels. The generated evidence report
is `reports/phase2b-main-session.json` and remains ignored by Git.

The predeclared primary feature was Ch2/Ch1 peak-amplitude ratio in dB:

```text
20 * log10(channel_2_peak_absolute / channel_1_peak_absolute)
```

It was selected from the earlier 2+2 pilot before formal analysis of the main
20+20 dataset. No spectral search or wider-lag feature was used to choose or
fit the primary result.

All-sample descriptive distributions were:

| Feature | LEFT | RIGHT |
| --- | --- | --- |
| Peak ratio dB | n=20; mean -3.101798; std 1.676876; min -5.669502; median -2.933005; max -0.281766 | n=20; mean +3.357663; std 1.025112; min +1.203436; median +3.502110; max +5.721189 |
| RMS ratio dB | mean -4.061981; min -5.536680; max -2.422510 | mean -0.359265; min -2.049678; max +1.290794 |
| Zero-lag Pearson | mean -0.029403 | mean -0.212044 |
| Normalized channel-difference energy | mean 1.028559 | mean 1.210321 |

The observed all-sample peak-ratio ranges did not overlap; the closest
descriptive separation was approximately 1.485202 dB. The observed RMS-ratio
ranges also did not overlap. These are descriptive statistics over all accepted
samples, not independent test results, and the secondary features were not
promoted to classifiers.

The **within-session chronological holdout** used accepted LEFT/RIGHT #1-#15
for training and #16-#20 for testing. The 30-sample training means were
approximately -2.882771 dB for LEFT and +3.683416 dB for RIGHT, producing a
training-only midpoint threshold of approximately +0.400322 dB. LEFT was below
the threshold and RIGHT was at or above it. The result was **10/10 on the
10-sample within-session chronological holdout**: 5/5 LEFT and 5/5 RIGHT, with
no held-out sample used to fit the threshold. The smallest absolute held-out
margin was approximately 0.682089 dB for LEFT #17; RIGHT #20 was another close
example at approximately 0.803113 dB.

The **within-session leave-one-pair-out cross-validation** excluded LEFT #N
and RIGHT #N in each of 20 folds, refitted the threshold on the other 38
samples, and classified only the excluded pair. The aggregate was 40/40: 20/20
LEFT and 20/20 RIGHT. Fold thresholds ranged from approximately +0.0644 dB to
+0.2513 dB. This reduces dependence on any single training pair but is not an
independent external test.

Some within-session shift was visible. The first 15 samples per class had means
of approximately -2.882771 dB for LEFT and +3.683416 dB for RIGHT; the last
five had means of approximately -3.758878 dB and +2.380403 dB respectively.
Every last-five example remained on the correct side of the training-only
threshold. This small-session observation does not establish drift mechanics;
it reinforces the need for session-level external validation.

The formal interpretation is strong within-session evidence that the two
observed LEFT/RIGHT interaction conditions are separable with a simple,
interpretable amplitude feature. It is not proof of pure spatial localization:
LEFT combined the left hand with the left location, RIGHT combined the right
hand with the right location, and all samples came from one user, laptop, desk,
setup, and session.

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

> Does the predeclared interaction-condition separation generalize to an
> untouched session when the same hand and finger tap both locations?

Broad Windows endpoint exploration is paused. WDM-KS device 18 at 48 kHz with
two active, meaningfully different channels is the selected endpoint for the
next Lenovo experiments. DirectSound and WDM-KS devices 19 and 20 should not be
tested unless later evidence provides a reason.

The collector, main development dataset, offline analysis pipeline, and formal
within-session analysis are complete. The next sequence is:

1. Checkpoint and freeze the reviewed Phase 2B implementation.
2. Implement a small reproducible frozen-baseline/external-evaluation path.
3. Fit the fixed midpoint rule using all 40 accepted samples from development
   session `20260827T171528.349289Z-c0e2caa7`, now that feature and model
   selection are complete.
4. Save the feature name, LEFT/RIGHT training means, threshold, learned
   direction, source session ID, and fitting rule, then freeze them.
5. Only afterward, collect an untouched session using the same hand and finger
   for both LEFT and RIGHT locations.
6. Apply the frozen baseline without refitting the threshold, changing the
   feature or direction, or selecting new features from the external results.

That future result must be reported separately as cross-session, same-hand
external validation. No external threshold or outcome is claimed yet.

Phase 2 should study and adapt suitable ideas from the MIT-licensed Holo project
with attribution where useful instead of rebuilding algorithms unnecessarily.

## Roadmap

| Phase | Status |
| --- | --- |
| 1 — Windows hardware feasibility | Complete |
| 1.5 — microphone/backend characterization | Complete for initial Lenovo feasibility |
| 2A — guided labeled LEFT/RIGHT tap dataset | Collector and main alternating 20+20 development dataset complete |
| 2B — reproducible offline spatial-feasibility analysis | Pipeline and formal main-session within-session analysis complete |
| External validation evidence gate | Frozen-baseline capability and untouched same-hand session planned |
| 3 — tap detection, features, classification, confidence/rejection | Not started |
| 4 — real-time DeskSense and Windows action mapping | Not started |
| 5 — cross-laptop hardware adaptation and testing | Not started |
| 6 — installer/UI if justified, benchmarks, demo, and release material | Not started |

No pure localization-accuracy claim should be made until an untouched
same-hand external session isolates location more cleanly and demonstrates
cross-session generalization.

## Repository and data-handling state

- Runtime dependencies remain limited to sounddevice and NumPy; pytest is the
  test dependency.
- Raw recordings are kept in memory and are not included in JSON reports.
- Generated reports under `reports/` are intentionally ignored by Git while
  `reports/.gitkeep` preserves the directory location.
- Earlier diagnostic and characterization reports may exist locally and are
  not project source artifacts.
- Phase 2A explicitly retains accepted complete captures and tap windows in
  local dataset artifacts. The pilot and main development sessions exist
  locally under `datasets/`, which is ignored by Git by default.
- The Phase 2B report `reports/phase2b-main-session.json` contains metrics and
  metadata but no waveform arrays. It is generated evidence and remains
  ignored by Git.
- Diagnostic and characterization commands continue to omit raw audio from
  their JSON reports; waveform retention occurs only in explicit dataset
  collection sessions.
- The detailed experiment chronology and evidence-retention notes are in
  [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).
