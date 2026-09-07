# DeskSense Project Status

Status captured: 2026-09-05

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
interaction conditions. Phase 2C now provides a reproducible frozen-baseline
artifact and a no-refit cross-session evaluation path. The baseline was
committed before external collection, and the first separate same-hand session
was evaluated without refitting: 39 of 40 taps were classified correctly.

The development dataset contains a material hand/location confound: LEFT means
left hand at the left location, while RIGHT means right hand at the right
location. The Phase 2B result therefore does **not** establish pure spatial
localization,
hand-independent localization accuracy, external-session generalization,
cross-laptop compatibility, real-time tap detection, or reliable action
triggering. Freezing the development baseline preserves the planned evaluation
rule. The same-hand external result provides stronger evidence for
location-dependent acoustic information in the tested setup, but it does not
establish general DeskSense accuracy.

Real-hardware observations in this document apply only to the Lenovo Windows 11
development laptop and the tested endpoints and procedures.

Phase 3 robustness work now separates high-recall candidate generation,
TAP/NON_TAP validation, and the unchanged frozen LEFT/RIGHT classifier.
Development Session A was used to select the Phase 3B.2 Stage 1 recovery route,
Stage 2 feature/model design, L2 value, and operating threshold. Every Phase
3B.2 result from that session is therefore development evidence. The complete
frozen pipeline was then evaluated on untouched Session B. It passed the
predeclared intended-tap gates but failed the negative false-accept gate, so it
must not be described as robust or production-ready.

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

Phase 2C implementation and automated validation are complete. The reviewed
artifact `baselines/lenovo-left-right-v1.json` freezes the same predeclared
feature using all 40 accepted development samples. Git checkpoint `9bea99f`
committed and pushed that artifact before external collection. The separate
same-hand session then produced 39/40 correct (97.5%; two-sided 95% Wilson
interval 87.1183%–99.5573%) with the stored threshold and direction unchanged.

The earlier Phase 2B results remain scoped to the hand/location-confounded
development session. The Phase 2C result is a true cross-session evaluation
that used the same right index finger at both locations, controlling tapping
hand more cleanly. It still represents one user, Lenovo laptop, wooden
desk/setup, two zones, and one external session—not general, cross-device, or
product-level accuracy.

Phase 3A.1 and Phase 3A.2 are complete and checkpointed. They provide the pure
streaming state machine, injected `sounddevice.InputStream` adapter, bounded
callback-to-main-thread transport, and terminal `--sense` path. Phase 3A.3
diagnostics made candidate gates and missed-event evidence observable and were
sufficient to expose that the original detector had both poor intended-tap
recall and serious mechanical false-positive behavior. No Windows action path
exists yet.

Checkpoint `93776418aef34558a114df7ab5d16e0cf2512b30` (`Add Phase 3B robustness
evidence pipeline`) added the local guided robustness dataset and deterministic
offline replay. Development
Session A (`20260901T131308.362205Z-f9e2b1ec`) contains 30 intended taps and 14
labeled negative segments. The old detector produced candidates for only 8/30
intended taps and 47 completed false events over 140 labeled negative seconds
(20.143/min), while its conditional frozen spatial result remained 8/8.

Phase 3B.1 used Session A only as development evidence. Phase 3B.2 now keeps
the ordinary RMS/peak/crest route unchanged and adds one non-overlapping
strong-impact recovery route requiring both RMS gate ratio >= 6.0 and peak
gate ratio >= 8.0. Production replay gives 29/30 Stage 1 candidates and 50
negative candidates over 140 seconds (21.429/min). Overlap, adaptive-floor,
ordinary-crest, center, refractory, clipping, and 9,600-by-2 candidate changes
were deliberately deferred.

The new versioned `baselines/lenovo-tapness-v1.json` artifact applies an
independent nine-feature L2-logistic TAP/NON_TAP model before the unchanged
spatial classifier. It was fitted from 29 TAP and 50 NON_TAP Session A
candidates with L2 0.01 and an uncalibrated-score threshold of
0.4495211534633274. Grouped development OOF accepted 28/29 TAP candidates and
5/50 NON_TAP candidates. Final all-development replay accepted 28/30 intended
taps, falsely accepted 4/50 negative candidates (1.714/min), and retained
28/28 conditional spatial correctness. These are tuned development and
resubstitution results, not validation. Phase 3B.2 was checkpointed in
`dc7b8eb3b387cdeda4b5b45ff0687dba6a761cc2` (`Implement Phase 3B tapness
pipeline`) before untouched Session B collection.

The Phase 3B external-validation harness defines a prediction-independent
30-tap positive protocol and an exact 300-second, seven-activity negative
protocol, binds collection to the frozen Stage 1/Stage 2/Stage 3 identities,
refuses tapness fitting from sessions marked `external_validation`, rejects
incomplete external collections for evaluation, and reports each pipeline
stage separately.

The frozen positive protocol uses the same right index finger and fleshy
fingertip pad for both established lower/front LEFT and RIGHT locations, with
one intended tap per cue. The negative denominator is 300 labeled seconds
across seven separately recorded activity segments; each segment has its own
excluded quiet warm-up and completion tail, so it is not one continuous
five-minute live run.

Official Session B (`20260904T171453.458221Z-4ecb16ad`) was collected and
evaluated as untouched external evidence for the frozen Phase 3B.2 pipeline.
Stage 1 generated candidates for 30/30 intended taps and Stage 2 accepted all
30/30 as TAP. The unchanged Stage 3 spatial classifier produced 28/30 correct
intended-zone outputs: all 15 LEFT taps were classified LEFT, while 13/15 RIGHT
taps were classified RIGHT and two were classified LEFT. Over the exact 300
labeled negative seconds, Stage 1 generated 74 candidates (14.8/min) and Stage
2 falsely accepted eight (1.6/min). The predeclared overall engineering gate
therefore **failed solely because the required maximum was one false accept**;
the Stage 1, Stage 2, and end-to-end positive gates all passed.

Session B remains the official first external validation of the complete
Phase 3B.2 pipeline, including its unchanged FAIL result. If its waveforms,
events, or results inform any later feature, threshold, model, calibration, or
Stage 1/2/3 change, it becomes development evidence for that later pipeline and
a new untouched Session C is required before another external-validation claim.

Stage 3 remains `baselines/lenovo-left-right-v1.json` without refitting or
modification: `peak_ratio_db_ch2_minus_ch1`, threshold
+0.12793235855251162 dB, LEFT below and RIGHT at or above, over the exact raw
48 kHz × 2-channel × 9,600-frame domain. The historical untouched same-hand
Phase 2C result remains 39/40.

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

Status: **collector implemented and validated; development and first external
20 LEFT + 20 RIGHT sessions complete**

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
- External session `20260829T101459.893269Z-566a8435` collected 20 accepted
  LEFT and 20 accepted RIGHT samples using the same right index finger for both
  zones, with no rejected or retried attempts.

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

### Phase 2C — frozen baseline and external evaluation

Status: **implementation and frozen baseline complete; baseline precommitted;
first same-hand cross-session external evaluation complete**

Implemented:

- Offline creation of the fixed LEFT/RIGHT midpoint baseline using every
  accepted sample in a validated development session.
- Deterministic exact-byte SHA-256 fingerprinting of `session.json`,
  `manifest.jsonl`, and each accepted NPZ in manifest order.
- Strict schema/version, feature, direction, midpoint, membership, fingerprint,
  finite-value, and waveform-exclusion validation for frozen artifacts.
- Cross-session evaluation that loads the stored feature, threshold, direction,
  and tie rule unchanged and never refits them from external data.
- Separate development and external interaction-context metadata.
- Confusion matrix, per-sample threshold margins, per-class results, and a
  two-sided 95% Wilson accuracy interval in local JSON reports.
- Offline CLI paths that do not initialize microphone hardware.

Verification before generating the real baseline:

- Complete suite: 217 tests passed.
- Focused Phase 2C tests: 63 passed, 28 deselected.
- `pip check`, byte compilation, imports, CLI help, and `git diff --check`
  passed.
- Imports completed without sounddevice being imported.
- `datasets/` and `reports/` remained ignored; `baselines/` remained eligible
  for Git tracking.
- No microphone hardware was accessed.

Reviewed frozen artifact:

- Path: `baselines/lenovo-left-right-v1.json`.
- Pre-external Git checkpoint: `9bea99f` (`Add frozen baseline and external
  validation pipeline`).
- Artifact type: `desksense_frozen_left_right_midpoint_baseline`, schema 1.
- Source session: `20260827T171528.349289Z-c0e2caa7`.
- Source context: `hand-location-confounded` — LEFT combined the left hand and
  left location; RIGHT combined the right hand and right location.
- Training membership: all 20 LEFT and 20 RIGHT accepted samples; no external
  samples.
- Primary feature: `peak_ratio_db_ch2_minus_ch1`, definition version 1,
  calculated over the complete retained `tap_window` array.
- LEFT mean: -3.1017978964848574 dB.
- RIGHT mean: +3.3576626135898806 dB.
- Frozen midpoint threshold: +0.12793235855251162 dB.
- Direction: LEFT below the threshold; RIGHT at or above it.
- Tie rule: `feature_value >= threshold_db predicts higher_feature_zone`.
- Source dataset SHA-256:
  `6c4ac4c3881faecbab430d593b6b217e03f58a13ca07522b679ee3d19706516f`.
- Artifact file SHA-256 recorded by the external report:
  `42ca902b06cc840b19df409d2869bad21db3f0d6b78c3850510938b31e13b588`.
- Fingerprint components: one `session.json`, one `manifest.jsonl`, and 40
  accepted NPZ artifacts. Rejected-attempt metadata is excluded.
- The baseline stores derived metadata and no raw waveform arrays.

The +0.12793235855251162 dB frozen threshold was derived reproducibly from all
40 accepted development samples after feature and model selection were
complete. It is distinct from the historical Phase 2B chronological-holdout
threshold of approximately +0.400322 dB, which used only accepted #1–#15 per
class.

For development-data context only, all 20 LEFT values were below the frozen
threshold and all 20 RIGHT values were above it. Observed LEFT values ranged
from approximately -5.6695 to -0.2818 dB; RIGHT values ranged from
approximately +1.2034 to +5.7212 dB. LEFT #17 was closest to the threshold at
approximately -0.281766 dB, about 0.409699 dB below it. These are
training/development observations, not external-validation results or a new
accuracy measurement.

First frozen cross-session external evaluation:

- External session: `20260829T101459.893269Z-566a8435`.
- External dataset SHA-256:
  `23f4c56a708ac43b570abef4437760923af1eb542636ceb857dbbcd24cfb945a`.
- Collection: 20 LEFT and 20 RIGHT accepted samples; no rejected or retried
  attempts.
- Protocol: the same right index finger made every LEFT and RIGHT tap on the
  same Lenovo laptop and wooden desk/setup, with the intended zone geometry,
  Windows WDM-KS endpoint, 48 kHz, two channels, and 200 ms retained windows.
- The baseline was not viewed or tuned during collection.
- Result: 39/40 correct (97.5%).
- Two-sided 95% Wilson score interval: 87.1183%–99.5573%.
- LEFT: 20/20; RIGHT: 19/20.
- Confusion matrix: actual LEFT → 20 LEFT, 0 RIGHT; actual RIGHT → 1 LEFT,
  19 RIGHT.
- No external threshold fitting, direction learning, feature selection, or
  normalization occurred.

The one misclassification is official external evidence and must be retained:
RIGHT #11 (`20260829T101459.893269Z-566a8435-right_011`) had a feature value of
+0.050166168713707354 dB and was predicted LEFT because it fell
0.07776618983880426 dB below the frozen +0.12793235855251162 dB threshold.
RIGHT #15 and RIGHT #9 were correctly classified but had small positive
margins of approximately 0.014163 dB and 0.035988 dB respectively. Margins are
threshold distances, not probabilities.

Post-prediction descriptive statistics showed non-overlapping external ranges:
LEFT mean -5.630947313147696 dB, range -8.668097139293714 to
-2.2228560154426775 dB; RIGHT mean +1.2090723599046973 dB, range
+0.050166168713707354 to +2.070086215063918 dB. The observed range gap was
approximately 2.273022 dB. These summaries did not alter the official
predictions.

Relative to development, the external LEFT and RIGHT means shifted downward by
approximately 2.529149 dB and 2.148590 dB respectively. Strong LEFT/RIGHT
separation remained, but the unchanged development threshold sat slightly
above one external RIGHT sample. This motivates later investigation of
calibration or session-offset handling; no calibrated solution is implemented
or claimed, and the official frozen result remains unchanged.

### Phase 3A.1 — pure streaming detector and label-free frozen inference

Status: **implementation and code review complete; Checkpoint #7 complete;
hardware-independent core ready for live integration**

Implemented:

- A pure `StreamingTapDetector` with `LEARNING`, `ARMED`, `COLLECTING`, and
  `REFRACTORY` states.
- Arbitrary sequential frames-by-channels input partitioned internally into
  fixed five-millisecond analysis blocks, independent of caller chunk sizes.
- Startup noise learning over an effective 0.75 seconds before any onset may
  trigger.
- A phase-insensitive onset gate using pooled multichannel energy and
  per-channel peak evidence without averaging channel waveforms together.
- A bounded causal center search spanning 12 ms before the detected onset to
  25 ms after it.
- Strongest-transient center refinement using pooled squared multichannel
  energy, a five-millisecond local-energy region, and earliest-frame tie
  resolution.
- Exact raw candidate extraction as
  `[center - 4,800 : center + 4,800]` at 48 kHz, producing a 9,600 x 2
  float32 window with channel order and sample values preserved.
- A sample-indexed 250 ms refractory interval after completed or structurally
  rejected candidates.
- Reset and discontinuity handling that clears history, partial blocks,
  candidate and refractory state, and learned noise state before restarting
  `LEARNING`.
- A pure label-free `classify_peak_ratio_value()` helper shared with the
  existing offline prediction path.

The intended default domain is 48 kHz, two channels, and finite
float-compatible input. Its main frame geometry is:

| Parameter | Default duration | Frames at 48 kHz |
| --- | ---: | ---: |
| Internal detector block | 5 ms | 240 |
| Startup learning | 0.75 s | 36,000 |
| Center-search pre-onset context | 12 ms | 576 |
| Center-search post-onset context | 25 ms | 1,200 |
| Transient-energy region | 5 ms | 240 |
| Frozen-feature candidate window | 200 ms | 9,600 |
| Refractory interval | 250 ms | 12,000 |

These onset thresholds, search dimensions, and refractory duration are initial
Phase 3A engineering defaults. Synthetic tests exercise them, but continuous
Lenovo microphone operation has not physically validated or optimized them.

The causal center-selection rule is deliberately distinct from Phase 2A. The
collector could search globally over a completed 1.5-second attempt; an
unbounded live stream cannot reproduce that operation causally. Phase 3A.1
instead opens a fixed local region around an onset, optionally removes DC from
a separate selection copy, and uses that copy only to choose a center. It does
not filter, normalize, average, or otherwise modify the exact raw classifier
candidate.

Bounded-memory and structural-safety behavior:

- The production circular-history capacity is 11,040 frames. Its two-channel
  float32 storage is 88,320 bytes, approximately 88 KB; partial fixed-block
  buffering is separately bounded below one 240-frame block.
- Returned candidates own copied memory and do not change when circular
  history advances.
- Caller arrays are copied before processing and are not modified.
- Rank, channel-count, numeric-conversion, and finite-value validation occurs
  before detector state mutation.
- A candidate is rejected rather than padded, shifted, or truncated if its
  required history is unavailable.
- A known discontinuity starts a new continuous epoch. No candidate may join
  samples from before and after the gap.

The frozen primary feature remains unchanged:

```text
peak_ratio_db_ch2_minus_ch1 =
    20 * log10(channel_2_peak_absolute / channel_1_peak_absolute)
```

The label-free helper accepts a feature value, frozen threshold, lower-feature
zone, and higher-feature zone. Values below the threshold predict the lower
zone; values at or above it predict the higher zone. It supports either
LEFT/RIGHT direction, requires finite values and valid class identities, and
performs no fitting, normalization, calibration, or label-dependent scoring.
Its margins are distances on the dB feature axis, not probabilities.
`analysis.predict_peak_ratio()` delegates only this threshold decision to the
shared helper, preserving Phase 2B and Phase 2C output semantics.

Synthetic verification includes:

- An impulse at non-block-aligned frame 203 produced identical onset, center,
  window indexes, and exact candidate bytes when supplied as one full chunk,
  irregular chunks, or one frame at a time.
- A two-stage event crossed the onset threshold at frame 203 and contained a
  stronger transient at frame 210. Refinement selected center 210 and the
  small-test candidate `[110:310]`, demonstrating that causal onset and
  strongest-transient center are separate concepts.
- Configuration rejects odd-frame tap windows rather than silently producing
  an asymmetric centered candidate.
- Tests cover startup suppression, quiet input, clipping, exact extraction,
  channel preservation, input immutability, owned output memory, refractory
  behavior, reset, discontinuity, invalid-input atomicity, bounded memory,
  feature compatibility, and offline/live decision parity.

Verification:

- Streaming and inference tests: 63 passed (33 streaming, 30 inference).
- Feature, analysis, and frozen-baseline regression selection: 89 passed.
- Complete suite: 280 passed.
- `pip check`, `compileall`, pure import checks, and `git diff --check` passed.
- Importing the streaming and inference modules did not import sounddevice.
- No microphone hardware was accessed.

This is a software/DSP architecture milestone, not evidence that live sensing
works on the Lenovo. Unknowns include WDM-KS callback behavior, physical onset
thresholds, adaptive-floor behavior under real noise, causal alignment on real
taps, weak-tap recall, speech/typing/desk-bump false positives, sustained-noise
behavior, callback sizes, overflow/discontinuity handling, callback/main-thread
scheduling, and end-to-end latency.

### Phase 3A.2a — injected live audio adapter and terminal sensing

Status: **implementation and fake validation complete; independently
code-reviewed and checkpointed in Checkpoint #8**

Implemented live path:

```text
explicit Windows input endpoint
  -> injected sounddevice.InputStream
  -> lightweight PortAudio callback
  -> one owned contiguous float32 copy
  -> bounded Queue(maxsize=8)
  -> main-thread consumer
  -> StreamingTapDetector
  -> exact accepted 9,600 x 2 candidate
  -> extract_two_channel_features()
  -> unchanged frozen peak-ratio baseline
  -> classify_peak_ratio_value()
  -> structured terminal event
```

There is no additional worker thread. `realtime.py` does not import
sounddevice; the CLI retains lazy backend loading and injects the backend into
the live runner. The frozen live domain is 48 kHz, two channels, float32, and a
9,600-frame / 200 ms candidate window. The selected endpoint name and host API
must match the frozen artifact, including Windows WDM-KS for the current Lenovo
baseline. The historical numeric device index is not treated as stable, so a
different current index is allowed when endpoint identity and the feature
domain still match.

The initial `InputStream` configuration uses the explicit current device,
48,000 Hz, two channels, float32, `blocksize=0`, the input callback, and a
finished callback. Allowing PortAudio to choose callback sizes is intentional;
the Phase 3A.1 detector accepts arbitrary caller chunk sizes and internally
reblocks them onto fixed 240-frame / 5 ms boundaries. No explicit low-latency
mode is requested before physical WDM-KS behavior is measured.

Callback and bounded-transport behavior:

- Callback sequence numbers are zero-based and monotonically increasing.
- Each callback takes one owned contiguous float32 copy and snapshots primitive
  PortAudio time/status data plus Python callback-arrival and enqueue-attempt
  monotonic timestamps.
- Queue insertion is non-blocking. The callback performs no detector DSP,
  feature extraction, classification, terminal output, or filesystem work.
- The initial queue capacity is eight callback packets. This is a reviewable
  engineering default, not a physically optimized value.
- A full queue drops the current/newest callback packet while retaining older
  queued packets in order. Known dropped callback/frame counts accumulate, and
  the next successfully retained packet carries explicit discontinuity
  metadata.

Known continuity loss includes queue overflow, PortAudio input overflow,
callback sequence gaps, and other relevant PortAudio status problems. Before
the first retained post-gap packet is processed, the main thread calls
`detector.notify_discontinuity()` exactly once for that boundary. Multiple
reasons still cause one reset. Detector history, partial blocks, pending
candidate state, refractory state, and learned noise state are cleared, a new
epoch begins in `LEARNING`, and no candidate may intentionally bridge the gap.
The adapter does not guess how many frames PortAudio itself discarded.

Only a `DetectionResult` whose status is `detected` reaches feature extraction
and classification. Rejected candidates are not classified even when they
carry waveform data, such as a near-clipping rejection. Accepted candidates
use the existing complete-window `peak_ratio_db_ch2_minus_ch1` feature and the
stored frozen threshold, direction, and tie rule without fitting,
normalization, calibration, or model modification. An undefined primary
feature produces a structured rejection rather than a fabricated LEFT/RIGHT
label. Reported margin is a dB threshold distance, not a probability.

Timing evidence includes, where available, PortAudio `inputBufferAdcTime` and
callback `currentTime`, Python callback-arrival and enqueue-attempt monotonic
timestamps, main-loop processing start, detector-result availability,
feature/inference completion, stream-reported latency, queue dwell, detector
lookahead, and approximate onset/center ADC mapping. PortAudio timestamps are
not subtracted directly from Python performance-counter timestamps because
their origins may differ. No precise physical impact-to-terminal latency is
claimed.

The live CLI requires both explicit inputs:

```powershell
python -m desksense --sense --device 18 --baseline baselines/lenovo-left-right-v1.json
```

`--sense` is mutually exclusive with other operational modes. Its structured
terminal output covers startup/learning, armed, discontinuity/relearning,
detected LEFT/RIGHT taps, and rejected candidates. It saves no audio, writes no
automatic report, and executes no Windows action.

Verification:

- Realtime tests: 39 passed.
- Focused `--sense` CLI tests: 21 passed, 52 deselected.
- Complete CLI tests: 73 passed.
- Streaming, inference, features, analysis, and frozen-baseline regression
  selection: 152 passed.
- Complete suite: 340 passed.
- `pip check`, `compileall`, lazy import checks, and `git diff --check` passed.
- Importing realtime/CLI did not import sounddevice.
- No microphone hardware was accessed and no physical `--sense` command ran.

The fake tests cover the exact stream domain, callback ownership and PortAudio
buffer reuse, channel order, bounded queue/drop propagation, PortAudio
overflow, fatal callback errors, endpoint/host-API compatibility, changed
device indexes, variable callback sizes, discontinuity-before-processing,
cross-gap history prevention, detected-only classification, near-clipping and
undefined-feature rejection, unchanged frozen inference, event ordering,
startup/armed/relearning events, unexpected termination, Ctrl+C-style cleanup,
startup failure, cleanup-error precedence, and absence of waveform/report
persistence.

At Phase 3A.2a completion these remained software-integration unknowns. The
Phase 3A.2b pilot below supplies only initial positive-case observations;
callback sizes/cadence, long-run PortAudio and queue behavior, detector
noise-floor robustness, typing/speech/movement false triggers, weak-tap recall,
causal center behavior across varied taps, and defensible end-to-end latency
remain unmeasured.

### Phase 3A.2b — first physical continuous live Lenovo pilot

Status: **first short positive-case pilot completed; broad live robustness not
established**

On 2026-08-30, the first continuous `--sense` run opened the Lenovo's
`Microphone Array 1 (Intel® Smart Sound Technology (Intel® SST) Microphone)`
through Windows WDM-KS at the current device index 18. The stream used 48 kHz,
two channels, and float32. Startup reported 47.00 ms stream latency, completed
the 0.75-second noise-learning period, and reached `Armed (detector epoch 0)`.

The command was:

```powershell
python -m desksense --sense --device 18 --baseline baselines\lenovo-left-right-v1.json
```

The established tap points were approximately 7–10 cm outside the respective
laptop edges, toward the user/touchpad side rather than vertically centered
beside the laptop: approximately lower/front-left and lower/front-right. The
same wooden desk setup was used. Every tap used the fleshy pad of the right
index finger with a moderate natural impact, so the same hand and finger were
used at both locations.

The intended sequence was three LEFT taps followed by four RIGHT taps. The
fourth RIGHT was added because the user initially thought the preceding RIGHT
tap had not appeared, although the terminal later showed that it had been
detected. This is an early terminal-attention/UX observation and does not by
itself establish excessive detection latency.

Observed live results:

| # | Intended/output zone | Feature (dB) | Margin (dB) | Onset | Center | Queue dwell (ms) | Detector lookahead (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LEFT | -7.106153 | 7.234085 | 213540 | 214185 | 0.13 | 102.81 |
| 2 | LEFT | -8.166028 | 8.293960 | 360098 | 360754 | 0.10 | 104.29 |
| 3 | LEFT | -4.018297 | 4.146229 | 826800 | 827949 | 0.11 | 101.06 |
| 4 | RIGHT | +2.447690 | 2.319758 | 1223811 | 1224472 | 0.11 | 100.17 |
| 5 | RIGHT | +2.196171 | 2.068238 | 1494843 | 1495495 | 0.28 | 103.85 |
| 6 | RIGHT | +1.736659 | 1.608726 | 1709463 | 1710152 | 0.05 | 101.83 |
| 7 | RIGHT | +2.936812 | 2.808879 | 1842295 | 1842326 | 0.08 | 103.21 |

All seven feature values fell comfortably on the expected side of the stored
approximately +0.127932 dB threshold. The exact frozen threshold and direction
remained unchanged: LEFT below the threshold and RIGHT at or above it. No
refitting, normalization, calibration, or threshold change occurred, and these
seven samples were not used as a new fitted dataset.

The defensible result is: the first continuous live Lenovo WDM-KS pilot
detected and correctly classified all seven intended same-hand LEFT/RIGHT taps
using the previously frozen classifier without refitting or calibration. This
means 3/3 intended LEFT taps were output as LEFT and 4/4 intended RIGHT taps
were output as RIGHT in this short run. It is not a 100% localization-accuracy,
production-accuracy, or generalization claim.

No `Tap rejected`, `Audio discontinuity`, `queue_overflow`, or PortAudio
`input_overflow` event was observed. The detector remained in epoch 0 and
Ctrl+C shutdown completed cleanly. The 0.05–0.28 ms observed queue-dwell values
provide no evidence of queue backlog in this short run, but do not establish
that queue capacity eight is generally sufficient. The 100.17–104.29 ms
detector-lookahead values describe algorithm timing and are not precise
end-to-end user-perceived latency. No false-positive rate, callback size,
callback cadence, weak-tap recall, or true end-to-end latency can be inferred.

The run also exposed a timing instrumentation defect. Terminal values such as
`approx. onset-to-result=82331177.79 ms` and
`approx. onset-to-result=82331163.10 ms` are physically impossible and invalid.
The PortAudio stream-time value and ADC-time estimate were not safely
comparable under the current calculation on this backend/run, although the
exact root cause is not yet established. These raw observations must not be
used in latency claims, clamped, or silently reinterpreted. Phase 3A.3 should
make this metric fail safely to `unavailable` when its inputs cannot be used.
The software change below addresses that requirement; physical validation is
still pending.

### Phase 3A.3 — clock-origin-safe live timing instrumentation

Status: **initial software fix implemented and tested; later physical evidence
showed the combined metric remained invalid, so normal live output omits it**

The first physical pilot's impossible approximately 82-million-ms value came
from a derived timing path conceptually equivalent to:

```text
stream.time - estimated_onset_adc_time
```

That historical observation remains valid evidence that the two absolute
values were not safely comparable under the calculation used on that WDM-KS
run. The exact backend or root cause is not claimed to be proven.

The revised calculation composes durations whose endpoints stay within their
respective clock domains. For onset timing:

```text
PortAudio onset-to-callback duration =
    callback_current_time_seconds - estimated_onset_adc_time_seconds

Python callback-to-result duration =
    (result_available_monotonic_ns - callback_arrival_monotonic_ns) / 1e9

approximate onset-to-result duration =
    PortAudio onset-to-callback duration
    + Python callback-to-result duration
```

Center-to-result uses the equivalent center ADC estimate. The implementation
does not subtract a PortAudio absolute timestamp from a Python performance-
counter timestamp. `stream.time` remains available as raw backend diagnostic
evidence only and no longer drives either derived onset/center duration.

If a required component is missing, non-finite, negative, or otherwise
unusable, the corresponding derived value becomes `None`/unavailable. There is
no clamping, inferred clock offset, calibration, or fallback that invents a
latency. The result remains approximate instrumentation, not precise physical
impact-to-terminal or user-perceived latency.

The change is limited to timing instrumentation. It does not alter detector
thresholds, startup learning, refractory timing, center search, candidate
window, queue capacity or drop policy, callback design, frozen baseline,
peak-ratio feature, LEFT/RIGHT rule, endpoint compatibility, or `InputStream`
configuration.

Verification:

- Focused timing selection: 12 passed, 38 deselected.
- Complete realtime suite: 50 passed.
- Complete suite: 351 passed.
- `pip check`, `compileall`, lazy import checks, and `git diff --check` passed.
- No microphone hardware was accessed and `--sense` was not executed.

A later physical Lenovo WDM-KS run still emitted an impossible approximately
84.7-million-ms combined value. The cross-clock composition therefore did not
produce a physically valid end-to-end metric on the tested path. Normal
`--sense` output now omits the combined onset/center-to-result fields rather
than attempting another transformation. Stream-reported latency, queue dwell,
and sample-domain detector lookahead remain separate raw or algorithmic
evidence; none is called precise user-perceived latency.

Subsequent Phase 3A.3 observability added opt-in gate counters,
closest-to-trigger evidence, candidate-start routes, and discontinuity-scoped
detector epochs without changing decisions. Controlled physical diagnostics
then exposed both missed intended taps and substantial false triggering from
mechanical non-tap activity. This completed enough diagnostic work to define
the Phase 3A-to-3B boundary; it did not establish robust detection.

### Phase 3B.0 — robustness evidence and offline replay foundation

Status: **complete and checkpointed in `93776418`; Development Session A
collected**

Implemented:

- A distinct local Phase 3 robustness schema for guided positive attempts and
  labeled negative activity segments.
- Positive captures that preserve every structurally valid intended attempt,
  including detector misses.
- Negative captures with machine-readable warm-up, labeled-activity, and
  post-activity completion-tail boundaries.
- Deterministic offline replay through the production detector without loading
  microphone hardware or modifying dataset files.
- Separate Stage 1 recall, negative event-rate, conditional spatial, candidate
  association, and descriptive tapness-feature reporting.

Development Session A, `20260901T131308.362205Z-f9e2b1ec`, contains 30
LEFT/RIGHT × light/normal/firm intended taps and 14 negative segments. Its raw
48 kHz, two-channel recordings remain local. The old detector produced 8/30
associated positive candidates and 47 negative completed events over 140
labeled seconds. This established the need to separate candidate generation,
tapness validation, and spatial classification.

### Phase 3B.1 — Session A offline detector-design study

Status: **development study complete**

Session A waveform analysis compared the existing non-overlapping detector,
an overlapping counterfactual, strong-impact recovery alternatives, adaptive
floor behavior, and deterministic Stage 2 feature gates. The selected smallest
Stage 1 change retained the fixed 5 ms timeline and added only the RMS >= 6x
and peak >= 8x recovery route. Overlapping blocks, floor-policy changes, and
global crest relaxation were deferred. Feature overlap between true taps and
mechanical negative events supported a small regularized binary model rather
than a growing set of brittle hard gates.

### Phase 3B.2 — versioned Stage 1 and frozen tapness baseline

Status: **complete and checkpointed; frozen pipeline externally evaluated on
Session B**

Implemented:

- Versioned Stage 1 ordinary-or-strong-impact candidate generation with
  ordinary diagnostic precedence and no duplicate candidate.
- One shared, immutable nine-feature tapness schema and extractor that preserves
  the raw 9,600-by-2 candidate.
- Deterministic NumPy L2-logistic fitting, grouped development folds,
  training-fold-only preprocessing, and a documented recall-constrained
  operating-point rule.
- Strict artifact validation, exact source-dataset fingerprinting and candidate
  membership, complete material Stage 1 compatibility binding, exclusive JSON
  writing, and explicit non-convergence failure.
- Live and offline ordering of Stage 1 → frozen Stage 2 → unchanged frozen
  Stage 3, with Stage 2 rejection preventing spatial inference.

Frozen Stage 2 artifact:

- Path: `baselines/lenovo-tapness-v1.json`.
- Model: L2-regularized TAP/NON_TAP logistic regression; L2 = 0.01.
- Features: pre-onset RMS, impact-window RMS, impact peak, onset contrast,
  peak-dominant contrast, effective energy duration, early-energy fraction,
  late-to-impact RMS ratio, and strong-sample fraction.
- Uncalibrated-score threshold: 0.4495211534633274.
- Training membership: 29 TAP and 50 NON_TAP candidates, all from Session A.
- Dataset SHA-256:
  `88a003966141d15858a2afec390c42e567439eb5f538b958e7aa60ee7638ab83`.
- No LEFT/RIGHT feature, spatial threshold, spatial margin, predicted zone, or
  waveform array is stored or used by Stage 2.

### Phase 3B Session B — external validation

Status: **complete; official overall FAIL due to negative false accepts**

Untouched Session B, `20260904T171453.458221Z-4ecb16ad`, used the same right
index finger and fleshy fingertip pad for all 30 guided LEFT/RIGHT taps. Stage 1
generated 30/30 associated candidates and frozen Stage 2 accepted 30/30 as TAP.
Frozen Stage 3 classified 28/30 to the intended zone (LEFT 15/15; RIGHT 13/15).

Across exactly 300 labeled negative seconds in seven separately recorded
activity segments, Stage 1 produced 74 candidates and Stage 2 falsely accepted
eight: typing 1, trackpad 1, laptop movement 2, and desk/object interaction 4;
quiet, speech, and hand movement produced no Stage 2 false accepts. The four
predeclared hard gates required Stage 1 >=27/30, Stage 2 >=27/30, end-to-end
correct intended-zone outputs >=27/30, and no more than one Stage 2 false accept.
The first three passed; the negative gate failed, making the official overall
result a FAIL. The per-cell preference also passed at 5/5 Stage 2 acceptance in
every side-by-strength cell, but it was not a hard gate.

This result applies only to the tested user, Lenovo laptop, wooden desk, and
session. It demonstrates high intended-tap survival in this session, not general
robustness. Mechanical false-positive rejection, especially for desk/object and
laptop interactions, is the next development problem. Windows actions remain
deferred.

Development-only results:

- Versioned Stage 1: 29/30 intended candidates; LEFT 15/15, RIGHT 14/15,
  light 10/10, normal 9/10, firm 10/10.
- Negative Stage 1 volume: 50 candidates over 140 labeled seconds,
  21.429/min.
- Grouped OOF Stage 2: 28/29 TAP accepted and 5/50 NON_TAP false accepted.
- Final all-Session-A replay: 28/30 intended taps accepted; 4/50 negative
  candidates false accepted, or 1.714/min; two laptop-movement and two
  desk/object-interaction events.
- Frozen Stage 3 remained unchanged and was correct for all 28 Stage 2-accepted
  positives.

Session A selected the Stage 1 policy, Stage 2 design, L2, and threshold.
Accordingly, none of these figures is untouched validation. Artifact
regeneration using its stored timestamp reproduced its means, scales,
coefficients, intercept, threshold, memberships, and fingerprint exactly.
Validation completed with 40 tapness tests, 165 combined
tapness/streaming/realtime/robustness tests, 94 CLI/robustness-CLI tests, and a
462-test full suite. `pip check`, `compileall`, CLI `--help`, lazy imports with
`sounddevice_loaded=False`, and `git diff --check` passed without microphone or
live-sensing access; the diff check emitted only normal LF-to-CRLF advisory
warnings.

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

> How can DeskSense reject realistic mechanical non-tap impacts—especially
> desk/object and laptop interactions—while preserving the now-observed high
> intended-tap recall and Stage 2 survival, before freezing a revised pipeline
> for untouched Session C?

Broad Windows endpoint exploration is paused. WDM-KS device 18 at 48 kHz with
two active, meaningfully different channels is the selected endpoint for the
next Lenovo experiments. DirectSound and WDM-KS devices 19 and 20 should not be
tested unless later evidence provides a reason.

The frozen Phase 3B.2 pipeline has now been evaluated on untouched Session B.
It passed Stage 1 recall (30/30), Stage 2 positive survival (30/30), and
end-to-end correct intended-zone output (28/30), but eight Stage 2 false accepts
over the exact 300 labeled negative seconds exceeded the predeclared maximum of
one. The official overall result is therefore FAIL; the gate is not revised
after seeing the data.

Phase 3B.3 used Sessions A and B as development evidence to select and track a
two-feature Stage 2 v2 research representation. A fixed provisional A+B
research artifact and prediction-independent Development Replication R1
collection/evaluation harness are now implemented. They are not part of the
production or live pipeline, R1 has not been collected, and v2 has not been
externally validated. If Session B informs any change, a new untouched Session
C is required before a subsequent external robustness claim. The historical
Session B result and its official FAIL remain preserved rather than being
replaced.
The historical Phase 2C same-hand result remains the official 39/40 spatial
external result and is not replaced by Phase 3 development replay.

Windows actions, hotkeys, and GUI behavior remain deferred until reliable
real-time sensing is demonstrated.

The observed cross-session downward shift motivates later study of calibration
or session adaptation, but no calibration method is currently selected or
validated. Additional sessions and cross-device experiments remain important
future robustness work rather than the immediate implementation step.

The completed external session must be preserved as the official 39/40 result.
Once its measurements inform calibration or model changes, it becomes
development evidence for those changes. Any modified classifier requires
another newly collected untouched session before a new external-validation
claim can be made.

Phase 3 should continue adapting suitable ideas from the MIT-licensed Holo
project with attribution only where DeskSense evidence supports them.

## Roadmap

| Phase | Status |
| --- | --- |
| 1 — Windows hardware feasibility | Complete |
| 1.5 — microphone/backend characterization | Complete for initial Lenovo feasibility |
| 2A — guided labeled LEFT/RIGHT tap dataset | Collector plus 20+20 development and same-hand external sessions complete |
| 2B — reproducible offline spatial-feasibility analysis | Pipeline and formal main-session within-session analysis complete |
| 2C — frozen baseline and external evaluation | Complete; baseline precommitted; first same-hand cross-session result 39/40 |
| External validation evidence gate | First scoped session complete; further untouched sessions required for modified models or broader claims |
| 3A.1 — pure streaming detector and label-free inference | Complete, code-reviewed, and checkpointed |
| 3A.2 — injected live adapter and physical sensing pilots | Complete and checkpointed; positive live path demonstrated without a broad robustness claim |
| 3A.3 — observability and diagnostic work | Complete enough to expose missed-tap and false-positive failure modes |
| 3B.0 — robustness evidence and replay pipeline | Complete and checkpointed in `93776418` |
| 3B.1 — Session A offline design study | Complete; development evidence only |
| 3B.2 — high-recall Stage 1 and frozen Stage 2 tapness model | Complete, checkpointed in `dc7b8eb3`, and externally evaluated |
| 3B Session B external validation | Complete; official overall FAIL because eight Stage 2 false accepts exceeded the maximum of one |
| 3B.3 — Stage 2 v2 research and replication | Two-feature research representation, provisional A+B artifact, and R1 harness implemented; R1 not collected; no live/default change |
| Development Replication R1 | Next; fixed prediction-independent 30-positive/300-second protocol; development evidence, not external validation |
| Session C external validation | Required after any pipeline change; not yet collected |
| 4 — Windows action mapping | Deferred until robustness evidence passes |
| 5 — cross-laptop adaptation and testing | Deferred |
| 6 — polish, demo, and release work | Deferred |

The first same-hand external session provides strong evidence for
location-dependent acoustic information and cross-session performance in the
tested setup. It does not establish a general localization-accuracy figure;
broader claims require additional users, sessions, desks, devices, and zones.

## Repository and data-handling state

- Runtime dependencies remain limited to sounddevice and NumPy; pytest is the
  test dependency.
- Normal diagnostic/live-sensing audio remains in memory and is not included
  in JSON reports. Raw waveform retention occurs only in explicit local dataset
  collection modes.
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
- The reviewed Phase 2C artifact `baselines/lenovo-left-right-v1.json` contains
  only derived model, membership, compatibility, fingerprint, and evidence
  metadata. It contains no waveform arrays and was committed in checkpoint
  `9bea99f` before external collection.
- External session `20260829T101459.893269Z-566a8435` remains local under the
  Git-ignored dataset root. Its generated evaluation report remains under the
  Git-ignored reports root and contains no waveform arrays.
- Diagnostic and characterization commands continue to omit raw audio from
  their JSON reports; waveform retention occurs only in explicit dataset
  collection sessions.
- Phase 3A.2 keeps streaming history and candidate windows in memory only. The
  physical pilot wrote no waveform or report and executed no action. The
  callback is supplied by PortAudio; detector and frozen inference work remains
  on the main thread without an additional worker.
- The first live pilot's approximately 82-million-ms onset-to-result values are
  retained only as evidence of a timing instrumentation defect and must not be
  treated as physical latency measurements.
- Normal Phase 3A.3 live output omits the physically invalid combined
  onset/center-to-result timing metric. Stream-reported latency, queue dwell,
  and detector lookahead remain separate evidence and are not combined into a
  claimed end-to-end latency.
- Phase 3B Development Session A remains local under the Git-ignored
  `datasets/` root. Its raw Phase 3 robustness recordings are not committed.
- Phase 3 Stage 2 fitting, replay, and inference remain local. Generated replay
  reports remain under the ignored `reports/` root and contain no waveform
  arrays.
- `baselines/lenovo-tapness-v1.json` contains derived model, feature-schema,
  training-membership, fingerprint, Stage 1 compatibility, and development
  evidence metadata only. It contains no raw waveform arrays and was
  checkpointed before untouched Session B collection.
- Phase 3B Session B remains local under the ignored `datasets/` root. Its
  waveform-free external report remains under ignored `reports/`. Evaluation
  verified the dataset fingerprint and both frozen-artifact hashes were
  identical before and after replay; neither baseline was altered.
- The detailed experiment chronology and evidence-retention notes are in
  [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).
