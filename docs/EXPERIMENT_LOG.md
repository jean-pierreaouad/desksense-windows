# DeskSense Engineering Experiment Log

This log records experiments in execution order. Exact calendar dates and some
original commands were not retained in the current checkpoint; they are marked
as such rather than reconstructed as facts. Results apply only to the Lenovo
Windows 11 development laptop and the named audio endpoint.

## Experiment 1 — hardware enumeration and first recording

**Purpose**

Confirm that Windows/PortAudio exposes usable laptop microphone inputs and that
ordinary desk taps produce measurable samples.

**Setup and procedure**

- Lenovo Windows 11 laptop using its Intel Smart Sound Technology microphone
  array.
- Audio devices were enumerated across the available Windows host APIs.
- A short in-memory recording was made through the default MME input.
- The exact tap position and timing were not retained for this initial run.

**Command**

The exact invocation was not retained. Equivalent current commands are:

```powershell
python -m desksense
python -m desksense --record --device 1
```

**Key measurements**

- MME device 1: four channels, default 44.1 kHz, with 44.1/48 kHz supported.
- DirectSound device 5: four channels, default 44.1 kHz.
- WASAPI device 9: two channels, default 48 kHz.
- WDM-KS device 18: two channels at 48 kHz.
- WDM-KS devices 19 and 20: four channels at 16 kHz.
- Initial MME tap recording:
  - Ch1 peak approximately 0.015900; RMS approximately 0.001093.
  - Ch2 peak approximately 0.015869; RMS approximately 0.001092.
  - Ch3/Ch4 peak approximately 0.000031; RMS approximately 0.000015.

**Interpretation**

The Windows audio stack exposed multiple input representations of the
microphone array. Channels 1 and 2 captured the event, while channels 3 and 4
were near a very low fixed level. Comparison with the later quiet baseline
showed the tap was measurable above baseline.

**Limitations**

- This was not a controlled localization experiment.
- Device indices are enumeration-session specific.
- A single recording cannot establish repeatability or physical channel
  independence.
- The host API or driver may apply undisclosed processing.

**Resulting decision**

Proceed with a quiet-baseline measurement and build dedicated multichannel
characterization tooling before attempting localization or machine learning.

## Experiment 2 — quiet baseline

**Purpose**

Measure the short-recording noise floor so the initial tap levels could be
interpreted against a no-deliberate-tap capture.

**Setup and procedure**

- Same Lenovo microphone system and default MME endpoint.
- Approximately three seconds were recorded without a deliberate tap.
- Exact ambient-noise conditions were not formally logged.

**Command**

The exact invocation was not retained. Equivalent current command:

```powershell
python -m desksense --record --device 1
```

**Key measurements**

- Ch1 peak approximately 0.002808; RMS approximately 0.000496.
- Ch2 peak approximately 0.002808; RMS approximately 0.000496.
- Ch3/Ch4 peak approximately 0.000031; RMS approximately 0.000015.

**Interpretation**

The initial fingertip tap was measurably above this observed baseline on
channels 1 and 2. Channels 3 and 4 did not show a useful change and continued
to appear inactive.

**Limitations**

- One short baseline does not characterize varying room noise, fan noise,
  automatic gain control, or long-term drift.
- This result does not establish a reliable detection threshold.
- It provides no localization evidence.

**Resulting decision**

Treat acoustic tap capture as feasible on this laptop, while leaving detection
thresholds and spatial feasibility unresolved.

## Experiment 3 — initial MME characterization

**Purpose**

Use the Milestone 1.5 metrics to assess activity and independence across the
four channels exposed by MME device 1.

**Setup and procedure**

- Approximately five-second MME characterization capture.
- A deliberate tap was intended, but the strongest detected transient appeared
  at approximately 0.669 seconds and could not be trusted as the controlled tap
  event.
- The exact physical tap timing/procedure was not sufficiently controlled for
  this run.

**Command**

The exact invocation was not retained. Equivalent current command:

```powershell
python -m desksense --characterize --device 1 --save-report reports\lenovo-mme.json
```

**Key measurements**

- Strongest exploratory transient center: approximately 0.669 seconds.
- No additional numeric result from this run is promoted here as controlled
  tap evidence.

**Interpretation**

The strongest-window logic found an event, but its timing did not provide a
reliable association with the intended tap. The window may have represented
incidental contact, handling noise, or another transient.

**Limitations**

- Tap timing and placement were not adequately controlled.
- The strongest-energy window is exploratory and is not a tap detector.
- Channel-similarity measurements around an unverified event cannot support a
  localization conclusion.

**Resulting decision**

Do not use this run as the primary controlled MME result. Repeat with the laptop
stationary, a marked tap point, one deliberate tap, and known approximate tap
timing.

## Experiment 4 — controlled MME repeat

**Purpose**

Repeat MME characterization with enough physical and temporal control to assess
whether exposed channels contain independent information during a known tap.

**Setup and procedure**

- Lenovo remained stationary on the same wooden desk.
- One marked tap point was approximately 7–10 cm left of the laptop's
  front-left/palm-rest area.
- One deliberate moderate fingertip-pad tap was used, not a fingernail tap.
- The tap occurred around 2.4 seconds into the approximately five-second
  recording.
- Endpoint: MME device 1, 44.1 kHz, four reported channels.

**Command**

```powershell
python -m desksense --characterize --device 1 --save-report reports\lenovo-mme.json
```

**Key measurements**

- Strongest transient center: approximately 2.393 seconds.
- Ch1: active; peak approximately 0.230896.
- Ch2: active; peak approximately 0.230896.
- Ch3 and Ch4: effectively inactive.
- Whole-recording Ch1 versus Ch2:
  - Pearson correlation approximately 0.999992.
  - Normalized difference energy approximately 0.000008.
  - Relative level approximately 0 dB.
  - Maximum normalized cross-correlation approximately 0.999992.
  - Lag 0 samples / 0 microseconds.
  - Heuristic label `duplicate_like`.
- Transient-window Ch1 versus Ch2:
  - Pearson correlation approximately 1.000000.
  - Normalized difference energy approximately 0.000000.
  - Relative level approximately 0 dB.
  - Lag 0 samples / 0 microseconds.

**Interpretation**

For this controlled tap and the tested MME endpoint, channels 1 and 2 were
effectively duplicate-like or heavily shared-processed. Channels 3 and 4 did
not provide useful signal. MME therefore did not expose useful independent
inter-channel spatial cues in this experiment.

This does not establish the number of physical microphones. Driver, Windows,
or array DSP behavior may account for the duplicate-like endpoint channels.

**Limitations**

- One controlled tap at one location is insufficient for localization or
  repeatability claims.
- The transient window is a subset of the whole recording, not independent
  corroboration.
- MME may expose a processed mix rather than raw physical microphone channels.
- No comparison endpoint was characterized under the same procedure in this
  experiment.

**Evidence retention**

The characterization completed and its terminal output remains the evidence
for this run. The final JSON was **not saved** because
`reports\lenovo-mme.json` already existed. The program correctly refused to
overwrite the existing file, so that file must not be treated as the saved
record of this controlled repeat.

**Resulting decision**

- Do not prefer MME for the next inter-channel localization investigation.
- Characterize the WASAPI Intel SST endpoint next, followed by relevant WDM-KS
  microphone-array endpoints if the evidence still supports that direction.
- Consider DirectSound only if it remains informative after those results.
- Do not begin localization-accuracy claims or final dataset collection yet.

## Experiment 5 — controlled WASAPI characterization

**Purpose**

Determine whether the normal WASAPI representation of the Intel SST microphone
array exposes more independent channel information than the duplicate-like MME
endpoint.

**Endpoint**

- Device 9: `Microphone Array (Intel Smart Sound Technology / Intel SST)`.
- Host API: Windows WASAPI.
- 48 kHz, two channels.

**Setup and procedure**

- A controlled tap was made during an approximately five-second
  characterization recording.
- The strongest exploratory transient was near 2.861 seconds.
- Further physical-placement details were not retained in this checkpoint.

**Command**

The exact invocation was not retained. Equivalent current command:

```powershell
python -m desksense --characterize --device 9
```

**Filename**

Valid controlled run: `lenovo-wasapi-02.json`.

The earlier `lenovo-wasapi-01.json` attempt was a false start and must not be
treated as valid experimental evidence.

**Key measurements**

- Both channels were active and numerically identical.
- Whole-recording Pearson correlation: 1.000000.
- Whole-recording normalized difference energy: 0.000000.
- Whole-recording relative level: 0.00 dB.
- Reported lag: 0 samples.
- Transient-window measurements were also identical.
- Exploratory heuristic label: `duplicate_like`.

**Interpretation**

The normal WASAPI endpoint exposed duplicate-like processed channels in this
test and did not provide useful inter-channel spatial information.

**Limitations**

- This was one controlled recording on one laptop and endpoint.
- Identical endpoint channels do not establish the physical microphone count.
- Windows, driver, or array DSP processing may determine the exposed channel
  layout.

**Resulting decision**

Do not prefer normal WASAPI for inter-channel localization on this Lenovo.
Proceed to the WDM-KS microphone-array representations.

## Experiment 6 — WDM-KS device 18 discovery and repeat

**Purpose**

Determine whether a WDM-KS endpoint exposes active channels with meaningfully
different information, unlike the tested MME and WASAPI endpoints.

**Endpoint**

- Device 18: `Microphone Array 1 (Intel Smart Sound Technology / Intel SST
  Microphone)`.
- Host API: Windows WDM-KS.
- 48 kHz, two channels.

**Setup and procedure**

- Controlled recordings were made with a tap on the left side of the laptop.
- Both endpoint channels were active.
- A representative run was repeated; the repeat showed broadly consistent
  channel differences.

**Command**

The exact invocations were not retained. Equivalent current command:

```powershell
python -m desksense --characterize --device 18
```

**Filename**

The discovery/repeat filenames were not recorded in this checkpoint.

**Key measurements — representative left-side run**

- Ch2 relative RMS versus Ch1: approximately -4.55 dB over the whole recording.
- Transient Ch2 relative RMS versus Ch1: approximately -4.79 dB.
- Transient Pearson correlation: approximately +0.135.
- Transient normalized difference energy: approximately 0.884.
- Reported transient lag: -48 samples.
- Exploratory heuristic label: `meaningfully_different`.

**Interpretation**

WDM-KS device 18 exposed substantially different channel information and was
the first tested endpoint to do so consistently. It became the leading Lenovo
endpoint for spatial-feasibility work.

The result does not prove that the two endpoint channels correspond directly
to separate raw physical microphones. Driver or DSP processing may remain in
the path.

**Limitations**

- Only a small number of controlled taps had been recorded.
- Exact tap force and position varied and were not instrumented.
- Raw waveform samples were discarded, preventing deeper offline analysis.
- The -48-sample result reached the current cross-correlation search boundary
  and is not an established physical arrival-time measurement.

**Resulting decision**

Select WDM-KS device 18 for the next experiment and compare a mirrored
right-side tap against the controlled left-side observations.

## Experiment 7 — first controlled right-side spatial test

**Purpose**

Test whether moving the tap from the controlled left-side location to a
mirrored right-side location changes the relationship between the two WDM-KS
device 18 channels.

**Endpoint**

- WDM-KS device 18 at 48 kHz with two active channels.

**Setup and procedure**

- The same endpoint and laptop/desk arrangement were used.
- A controlled tap was made at the mirrored right-side location.
- The strongest exploratory transient was near 2.498 seconds.

**Command**

The exact invocation was not retained. Equivalent current command:

```powershell
python -m desksense --characterize --device 18
```

**Filename**

Valid first right-side run: `lenovo-wdm18-right-01.json`.

**Key measurements**

- Transient Ch2 relative RMS versus Ch1: approximately -2.67 dB.
- Transient Pearson correlation: approximately -0.297.
- Transient normalized difference energy: approximately 1.283.
- Reported lag: +44 samples.

**Interpretation**

These measurements differed substantially from the controlled left-side run:
the channel-level gap was smaller, Pearson changed from low positive to
negative, normalized difference energy moved from below 1 to above 1, and the
reported lag changed sign. This was the first direct indication that tap
position may affect the measured relationship between the endpoint channels.

**Limitations**

- A single left/right comparison cannot establish repeatability or accuracy.
- The observed distinctions were interpreted after viewing the measurements.
- Tap force and exact position were not instrumented.
- The +44-sample lag was close to the bounded +48-sample search limit and is
  not a proven physical time-of-arrival result.

**Resulting decision**

Run an alternating LEFT/RIGHT experiment to determine whether the broad feature
differences repeat across additional controlled taps.

## Experiment 8 — alternating LEFT/RIGHT repeatability test

**Purpose**

Check whether the initial left/right channel-feature differences repeat when
valid LEFT and RIGHT taps are collected in alternating order.

**Endpoint**

- WDM-KS device 18 at 48 kHz with two active channels.

**Setup and procedure**

- Controlled taps alternated between the established left-side and mirrored
  right-side locations.
- Several false-start recordings occurred. They were excluded from
  interpretation and are **not** valid experimental samples.
- File numbering therefore contains gaps and does not equal sample order.

**Command**

The exact invocations were not retained. Each valid run used characterization
on device 18 and was saved under the corresponding filename below.

**Corrected valid filename mapping**

- `spatial-left-02.json` = valid Left #1.
- `spatial-right-02.json` = valid Right #1.
- `spatial-left-03.json` = valid Left #2.
- `spatial-right-03.json` = valid Right #2.

Earlier attempts with lower numbers were false starts and must not be treated
as data.

**Key measurements**

| Valid sample | Strongest transient | Ch2 RMS vs Ch1 | Pearson | Normalized difference energy | Reported lag | Channel heuristic |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Left #1 — `spatial-left-02.json` | 0.400 s | -5.88 dB | +0.094 | 0.924 | -48 samples | `meaningfully_different` |
| Right #1 — `spatial-right-02.json` | 2.505 s | -2.65 dB | -0.046 | 1.044 | +48 samples | `meaningfully_different` |
| Left #2 — `spatial-left-03.json` | 2.025 s | -4.16 dB | +0.181 | 0.838 | -48 samples | `meaningfully_different` |
| Right #2 — `spatial-right-03.json` | 2.838 s | -0.53 dB | -0.136 | 1.136 | +37 samples | `meaningfully_different` |

Left #1 selected an atypically early transient near 0.400 seconds. It remains
in the valid set because its channel features are consistent with the other
left-side observations, but its timing is an explicit limitation and should be
revisited during guided collection.

The `meaningfully_different` heuristic describes the relationship between the
two channels within each recording. It is not a left/right classification
result.

**Interpretation**

Across this small valid set and the preceding controlled observations:

- Left-side taps tended to make Ch2 substantially weaker than Ch1, produce low
  positive Pearson correlation, normalized difference energy below
  approximately 1, and a negative reported lag that frequently reached the
  -48-sample boundary.
- Right-side taps tended to show a smaller inter-channel RMS difference,
  negative Pearson correlation, normalized difference energy above
  approximately 1, and a positive reported lag.

Several measured features therefore formed visibly different left/right groups
in the observations collected so far. This is initial evidence of left/right
spatial feasibility, not measured classifier accuracy.

**Limitations**

- The valid alternating set contains only four samples.
- Candidate feature thresholds were observed after seeing the measurements.
- No held-out evaluation was performed.
- Tap force and exact position varied.
- The strongest-transient finder is exploratory; Left #1 selected an unusually
  early event.
- Actual waveform samples were discarded, preventing offline verification and
  alternative feature extraction.
- Cross-correlation was limited to +/-48 samples (+/-1 ms at 48 kHz). Multiple
  results hit the boundary, so the reported lag values must not be interpreted
  as established acoustic time-of-arrival measurements.

**Open lag-analysis issue**

Future work should consider expanding the exploratory lag search to several
milliseconds, explicitly flagging boundary hits, and determining whether lag
remains useful when retained waveform windows can be analyzed directly.

**Resulting decision**

- Pause broad Windows endpoint exploration.
- Use WDM-KS device 18 as the preferred Lenovo endpoint unless later evidence
  gives a reason to revisit the decision.
- Do not test DirectSound or WDM-KS devices 19 and 20 merely for completeness.
- Change the main question from whether Windows exposes useful different
  channel information to whether unseen taps can be classified reliably into
  spatial zones.
- Make Phase 2A, guided labeled tap dataset collection, the next milestone.
  Begin with approximately 20 LEFT and 20 RIGHT taps, controlled collection
  order, explicit tap cues, locally retained labeled waveform windows,
  metadata, and capture-quality checks.
- Keep dataset artifacts out of Git and evaluate future methods on held-out
  samples before reporting accuracy.
- Study and adapt suitable ideas from the MIT-licensed Holo project with
  attribution where useful, without prematurely choosing a GUI or complex
  machine-learning model.

## Experiment 9 — Phase 2A guided 2+2 collector pilot

**Purpose**

Physically validate the guided labeled-dataset collector on the selected Lenovo
endpoint, including alternating labels, cue timing, capture-quality gates, and
lossless local persistence. The pilot also provided the first retained waveform
windows for limited offline spatial inspection; it was not designed to measure
classifier accuracy.

**Implementation verification before physical testing**

- Complete automated suite: 92 tests passed.
- Focused Phase 2A suite: 44 tests passed.
- `pip check`, byte compilation/import checks, CLI help, and
  `git diff --check` passed.
- `datasets/` was confirmed ignored by Git.
- Automated tests used injected or synthetic audio and did not access the real
  microphone.

**Endpoint**

- Device 18: `Microphone Array 1 (Intel Smart Sound Technology / Intel SST
  Microphone)`.
- Host API: Windows WDM-KS.
- 48 kHz, two channels.

The numeric device index describes this enumeration session and is not a
portable endpoint identifier.

**Session**

- Session ID: `20260826T215711.391122Z-4ed5556c`.
- Requested samples: two LEFT and two RIGHT.
- Collection order: alternating LEFT, RIGHT, LEFT, RIGHT.

**Setup and procedure**

- The guided collector requested confirmation before each attempt and used a
  silent terminal countdown; no audible cue was used.
- Microphone capture began before `TAP NOW` was displayed.
- The intended pre-cue interval was 0.200 seconds / 9,600 samples within a
  1.5-second capture, leaving a nominal 1.3 seconds after the cue.
- The intended cue offset records collection configuration. It does not measure
  terminal rendering latency or human reaction time exactly.
- Four alternating taps were collected. All were accepted on attempts 1–4,
  with no retries.

**Command**

The recorded session configuration corresponds to:

```powershell
python -m desksense --collect-dataset --device 18 --samples-per-zone 2
```

**Accepted samples and timing**

| Collection order | Accepted sample | Strongest transient center | Approximate delay after intended cue | Transient/background contrast |
| ---: | --- | ---: | ---: | ---: |
| 1 | LEFT #1 | 1.164 s | 0.964 s | 34.47 dB |
| 2 | RIGHT #1 | 1.118 s | 0.918 s | 34.48 dB |
| 3 | LEFT #2 | 0.910 s | 0.710 s | 33.89 dB |
| 4 | RIGHT #2 | 0.843 s | 0.643 s | 34.43 dB |

All four strongest transients occurred well inside the 1.5-second captures.
Both channels were active in every accepted attempt, no samples were clipped,
and all four attempts passed the conservative quality checks.

**Artifact verification**

Each accepted NPZ contained:

- `capture`: 72,000 x 2 float32 samples, representing the complete 1.5-second
  guided capture.
- `tap_window`: 9,600 x 2 float32 samples, representing an exploratory 200 ms
  centered window.
- Embedded metadata matching the manifest record.

The 200 ms windows contained approximately 96–97% of the observed full-capture
AC signal energy in this pilot. This is pilot evidence that the current window
is reasonable for exploratory analysis, not proof that 200 ms is optimal.

**Preliminary offline spatial measurements**

| Accepted sample | Ch2 RMS vs Ch1 | Ch2 peak amplitude vs Ch1 |
| --- | ---: | ---: |
| LEFT #1 | -3.96 dB | -5.35 dB |
| RIGHT #1 | -0.18 dB | +1.18 dB |
| LEFT #2 | -3.93 dB | -4.92 dB |
| RIGHT #2 | +0.16 dB | +1.11 dB |

Same-zone waveform repeats were highly similar after small alignment, while
cross-zone waveform similarity was materially lower. This is encouraging
additional evidence that LEFT and RIGHT position affects the two-channel
acoustic signature. With only four samples collected for collector validation,
it is not a classifier-accuracy result.

**Lag correction and refinement**

The earlier characterization results in Experiments 6–8 used a bounded
cross-correlation search of only +/-48 samples (+/-1 ms at 48 kHz). Several
measurements reached that limit. Offline analysis of the retained pilot windows
with a wider search found the strongest inter-channel offset at approximately
123–126 samples (approximately 2.56–2.63 ms) in the same direction across all
four samples.

This refines the earlier interpretation rather than erasing the historical
measurements: the apparent LEFT-negative / RIGHT-positive lag pattern was
likely influenced by the +/-1 ms boundary, and the current raw lag evidence
does not provide useful left/right sign separation in this pilot. The wider
offset must not be treated as proven physical microphone time of arrival;
Windows, driver, or microphone-array DSP processing may contribute. Phase 2B
should determine whether lag has predictive value.

**Interpretation**

The collector completed the requested alternating sequence on the selected
hardware, placed each observed tap inside its capture, retained the intended
float32 arrays and matching metadata, and accepted both active channels without
clipping. The waveform measurements add limited spatial-feasibility evidence,
but they do not demonstrate reliable localization.

**Limitations**

- The pilot contains only four samples and was explicitly intended to validate
  the collector.
- No retry occurred, so this run does not by itself exercise real-world retry
  frequency or all rejection paths.
- Tap force and exact within-zone position were not instrumented.
- The intended cue offset does not measure display scheduling or human reaction
  latency.
- The exploratory transient finder and 200 ms window have not been shown to be
  optimal.
- No held-out classifier evaluation was performed.
- The wider lag result may include driver or DSP effects and is not established
  acoustic time of arrival.

**Resulting decision**

- Treat the Phase 2A collector implementation as physically validated on the
  tested Lenovo endpoint, while leaving the overall data-collection milestone
  open.
- Checkpoint the validated implementation before the main collection run.
- Collect approximately 20 LEFT and 20 RIGHT taps with natural small
  within-zone variation.
- Proceed afterward to Phase 2B held-out spatial-feasibility analysis.
- Do not report localization or classifier accuracy before that evaluation.

## Experiment 10 — Phase 2B main-session offline analysis

**Purpose**

Evaluate a predeclared, interpretable LEFT/RIGHT baseline on the completed main
development dataset using reproducible offline integrity checks, a
training-only chronological holdout, and within-session cross-validation. This
experiment evaluates the two observed interaction conditions; it was not an
external or hand-independent localization test.

**Implementation verification before the real analysis**

- Complete automated suite: 154 tests passed.
- Focused Phase 2B/CLI tests: 62 passed, 16 deselected.
- `pip check`, byte compilation/import checks, CLI help, and
  `git diff --check` passed.
- `datasets/` and generated reports remained ignored by Git.
- The offline analysis path did not initialize microphone hardware.

**Dataset and endpoint**

- Session ID: `20260827T171528.349289Z-c0e2caa7`.
- Device index at collection time: 18.
- Endpoint: `Microphone Array 1 (Intel Smart Sound Technology / Intel SST
  Microphone)` through Windows WDM-KS.
- Configuration: 48 kHz, two channels.
- Accepted dataset: 20 LEFT and 20 RIGHT samples, collected in alternating
  order.
- One RIGHT attempt was manually rejected and retried. Its waveform was not
  retained or analyzed as an accepted sample.

Every accepted artifact contained a 72,000 x 2 float32 complete capture and a
9,600 x 2 float32, 200 ms tap window. Both expected channels were active and no
accepted sample clipped.

**Experimental context**

- LEFT samples were made with the left hand at the left location.
- RIGHT samples were made with the right hand at the right location.

Tapping hand/impact mechanics and location therefore changed together. The
dataset can test LEFT/RIGHT interaction-condition separation, but it cannot
isolate spatial location alone.

**Command and procedure**

The reproducible offline command for the retained report is:

```powershell
python -m desksense --analyze-dataset datasets\20260827T171528.349289Z-c0e2caa7 --interaction-context hand-location-confounded --save-report reports\phase2b-main-session.json
```

The analyzer loaded `session.json`, `manifest.jsonl`, and every accepted NPZ;
validated their identity, numbering, order, metadata, array structure, dtype,
sample rate, channel count, finiteness, and tap-window consistency; and then
computed features from the retained tap windows. It did not modify the dataset
or access the microphone.

**Integrity result and evidence retention**

- Integrity validation passed.
- All 40 accepted manifest records had matching validated NPZ artifacts.
- No orphan NPZ artifacts were found.
- Accepted counts matched the requested 20 LEFT + 20 RIGHT dataset.
- Generated report: `reports/phase2b-main-session.json`.

The report is local generated evidence, contains no waveform arrays, and is
intentionally ignored by Git.

**Predeclared primary feature**

The primary baseline was the Ch2/Ch1 peak-amplitude ratio in dB:

```text
20 * log10(channel_2_peak_absolute / channel_1_peak_absolute)
```

This feature was selected from the earlier 2+2 pilot before formal analysis of
the main 20+20 dataset. No spectral feature search or wider-lag feature was
used to select the primary result.

**All-sample descriptive measurements**

These summaries use all accepted samples and are not independent test results.

| Feature | LEFT | RIGHT |
| --- | --- | --- |
| Peak ratio dB | n=20; mean -3.101798; std 1.676876; min -5.669502; median -2.933005; max -0.281766 | n=20; mean +3.357663; std 1.025112; min +1.203436; median +3.502110; max +5.721189 |
| RMS ratio dB | mean -4.061981; min -5.536680; max -2.422510 | mean -0.359265; min -2.049678; max +1.290794 |
| Zero-lag Pearson | mean -0.029403 | mean -0.212044 |
| Normalized channel-difference energy | mean 1.028559 | mean 1.210321 |

The observed peak-ratio ranges did not overlap. Their closest descriptive
separation was approximately 1.485202 dB, calculated as the RIGHT minimum
(+1.203436 dB) minus the LEFT maximum (-0.281766 dB). The observed RMS-ratio
ranges also did not overlap in this session, but RMS ratio, Pearson, and
normalized difference energy remained secondary descriptive features rather
than additional classifiers.

**Within-session chronological holdout**

- Training: accepted LEFT #1-#15 and RIGHT #1-#15, 30 samples total.
- Test: accepted LEFT #16-#20 and RIGHT #16-#20, 10 samples total.
- The split used accepted sample number, not filesystem or manifest order.
- Held-out examples were not used to fit the threshold.
- Training LEFT mean: approximately -2.882771 dB.
- Training RIGHT mean: approximately +3.683416 dB.
- Training-only midpoint threshold: approximately +0.400322 dB.
- Learned direction: LEFT below the threshold; RIGHT at or above it.

The result was **10/10 on the 10-sample within-session chronological holdout**:
5/5 LEFT and 5/5 RIGHT.

| Actual class | Predicted LEFT | Predicted RIGHT |
| --- | ---: | ---: |
| LEFT | 5 | 0 |
| RIGHT | 0 | 5 |

The smallest held-out absolute margin from the learned threshold was
approximately 0.682089 dB for LEFT #17. RIGHT #20 was another relatively close
example at approximately 0.803113 dB. No held-out sample crossed the threshold.

**Within-session leave-one-pair-out cross-validation**

For each accepted number N, LEFT #N and RIGHT #N were excluded together, the
midpoint threshold was refitted using the remaining 38 samples, and only the
excluded pair was classified. Across 20 folds, the aggregate was 40/40: 20/20
LEFT and 20/20 RIGHT. Fold thresholds ranged from approximately +0.0644 dB to
+0.2513 dB.

This within-session cross-validation shows that the result was not dependent
on one single included training pair. It is not an independent external test
and does not establish cross-session generalization.

**Session-shift observation**

The first 15 samples per class had means of approximately -2.882771 dB for
LEFT and +3.683416 dB for RIGHT. The last five had means of approximately
-3.758878 dB for LEFT and +2.380403 dB for RIGHT. Despite this shift, every
last-five sample remained correctly separated by the training-only threshold.

This is a small-session observation, not proof of a drift mechanism. Its main
implication is that session-level external validation remains necessary.

**Interpretation**

The experiment provides strong within-session evidence that the two observed
LEFT/RIGHT interaction conditions are separable using a simple, interpretable
two-channel amplitude feature. The predeclared baseline achieved 10/10 on its
chronological held-out subset and 40/40 in within-session leave-one-pair-out
cross-validation.

These results do not establish pure spatial localization. Tapping hand and
location were confounded, and all data came from one user, laptop, desk, setup,
and session. They are not hand-independent, external-session, cross-device, or
final DeskSense accuracy results.

**Limitations**

- LEFT combined left hand with left location; RIGHT combined right hand with
  right location.
- All 40 samples came from one collection session and physical setup.
- The chronological holdout was defined after dataset collection, although its
  held-out samples were not used to fit the threshold.
- Leave-one-pair-out folds share the same session and are not independent
  external datasets.
- Descriptive all-sample range separation must not be presented as held-out
  performance.
- The observed session shift may matter across sessions, users, desks, or
  devices and has not yet been characterized.

**Resulting decision**

- Treat Phase 2B implementation and formal main-session within-session analysis
  as complete.
- Checkpoint and freeze the reviewed Phase 2B implementation.
- Before collecting an external dataset, implement a reproducible
  frozen-baseline/external-evaluation path.
- Fit the already selected midpoint rule on all 40 development samples and
  preserve its feature name, LEFT/RIGHT means, threshold, direction, source
  session ID, and fitting rule.
- Freeze that baseline before collecting a new untouched session using the same
  hand and finger for both LEFT and RIGHT locations.
- Apply the frozen baseline without refitting, redefining the feature or
  direction, or selecting features from the external result.
- Report the future result separately as cross-session, same-hand external
  validation. No such outcome is claimed yet.

## Experiment 11 — Phase 2C frozen development baseline

**Purpose**

Freeze the already selected simple LEFT/RIGHT midpoint model from the complete
development session before collecting any external same-hand data. The goal
was to make the future cross-session test auditable and prevent its feature,
threshold, direction, or tie rule from being changed after external examples
are observed.

**Implementation verification before baseline generation**

- Complete automated suite: 217 tests passed.
- Focused Phase 2C tests: 63 passed, 28 deselected.
- `pip check`, byte compilation, imports, CLI help, and
  `git diff --check` passed.
- Imports passed without sounddevice being imported.
- `datasets/` and `reports/` were confirmed ignored; `baselines/` was confirmed
  eligible for Git tracking.
- No microphone hardware was accessed.

**Source development session and evidence context**

- Session ID: `20260827T171528.349289Z-c0e2caa7`.
- Accepted samples: 20 LEFT, 20 RIGHT, 40 total.
- All 40 accepted samples were used; no external samples were used.
- Source interaction context: `hand-location-confounded`.
- LEFT condition: left hand at the left location.
- RIGHT condition: right hand at the right location.

The source session therefore does not isolate location alone. It remains
development evidence from one user, Lenovo laptop, desk, setup, and session.

**Procedure**

The offline freeze path loaded the source session through the existing strict
dataset validator, extracted the predeclared primary feature from every
accepted retained tap window, computed the LEFT and RIGHT arithmetic means,
and stored their midpoint with the direction learned from those means. It did
not access a microphone, modify the source dataset, search thresholds, select
another feature, or use any external sample.

The reproducible command corresponding to the reviewed artifact is:

```powershell
python -m desksense --freeze-baseline datasets\20260827T171528.349289Z-c0e2caa7 --interaction-context hand-location-confounded --save-baseline baselines\lenovo-left-right-v1.json
```

**Frozen artifact**

- Path: `baselines/lenovo-left-right-v1.json`.
- Artifact type: `desksense_frozen_left_right_midpoint_baseline`.
- Baseline schema version: 1.
- Primary feature: `peak_ratio_db_ch2_minus_ch1`.
- Feature definition version: 1.
- Feature definition:

  ```text
  20 * log10(channel_2_peak_absolute / channel_1_peak_absolute)
  ```

- Source window: complete retained `tap_window` array.
- LEFT development mean: -3.1017978964848574 dB.
- RIGHT development mean: +3.3576626135898806 dB.
- Frozen midpoint threshold: +0.12793235855251162 dB.
- Direction: LEFT below the threshold; RIGHT at or above it.
- Tie rule: `feature_value >= threshold_db predicts higher_feature_zone`.

The threshold was derived reproducibly from all 40 accepted development
samples; it was not manually hard-coded. It must not be confused with the
historical Phase 2B chronological-holdout threshold of approximately
+0.400322 dB, which was fitted only on accepted #1–#15 per class for that
within-session experiment.

**Source dataset fingerprint and evidence retention**

The frozen source dataset SHA-256 is:

```text
6c4ac4c3881faecbab430d593b6b217e03f58a13ca07522b679ee3d19706516f
```

The exact-byte fingerprint contains 42 deterministic components:

- One `session.json`.
- One `manifest.jsonl`.
- Forty accepted NPZ artifacts in manifest order.

Rejected-attempt metadata is excluded. The baseline stores the fingerprint,
training membership, compatibility information, derived model values, and
evidence limitations, but no raw waveform arrays.

**Development-data observation**

All 20 development LEFT feature values were below the frozen threshold and all
20 development RIGHT values were above it. Observed LEFT values ranged from
approximately -5.6695 to -0.2818 dB; observed RIGHT values ranged from
approximately +1.2034 to +5.7212 dB. LEFT #17 was closest to the threshold at
approximately -0.281766 dB, about 0.409699 dB below it.

These are training/development observations. They are not a new held-out
result, external-validation result, or accuracy measurement.

**Planned external protocol**

The future external session must be a different session collected only after
this implementation and frozen artifact are checkpointed. It should use the
same Lenovo laptop, intended microphone endpoint/backend, sample rate and
channel configuration, 200 ms retained tap-window design, and LEFT/RIGHT zone
geometry. The same hand and finger must tap both zones so the previous
tapping-hand variable is controlled more cleanly.

External evaluation must apply unchanged:

- `peak_ratio_db_ch2_minus_ch1`, definition version 1.
- Threshold +0.12793235855251162 dB.
- LEFT-below / RIGHT-at-or-above direction.
- The stored greater-than-or-equal tie rule.

It must not refit the threshold, relearn direction, select another feature,
normalize from external class statistics, tune from external labels, or replace
the frozen artifact after viewing the result. If performance is poor, the
result must be preserved. Any model change makes that session development
evidence, and another untouched future session is required for a new external
test.

**Interpretation and limitations**

The generated and independently reviewed artifact freezes the intended
development rule before external collection. This is a reproducibility and
experimental-design checkpoint, not an external performance result. The
source session remains hand/location-confounded, and no hand-independent,
cross-session, cross-device, or final DeskSense accuracy has been established.

No microphone capture was performed while creating or reviewing the baseline.
No external dataset had been collected or evaluated at this checkpoint. The
baseline contains no waveform data.

**Resulting decision**

- Treat Phase 2C implementation and real-baseline generation as complete.
- Complete final validation and Checkpoint #5 commit/push containing the Phase
  2C implementation, tests, README changes, checkpoint documentation, and
  `baselines/lenovo-left-right-v1.json`.
- Do not collect the same-hand external session until after that checkpoint.
- Apply the frozen baseline unchanged to the future external session and report
  the result separately, whether successful or unsuccessful.

## Experiment 12 — first frozen same-hand cross-session external validation

**Purpose**

Test whether the predeclared LEFT/RIGHT baseline learned entirely from the
development session could classify a separately collected session without any
refitting. The external collection used the same hand and finger for both zones
to control the earlier hand/location confound more cleanly.

**Precommitment and frozen source model**

The Phase 2C implementation and frozen baseline were committed and pushed
before the external session was collected:

- Git checkpoint: `9bea99f` — `Add frozen baseline and external validation
  pipeline`.
- Baseline: `baselines/lenovo-left-right-v1.json`.
- Baseline artifact file SHA-256 recorded by the external report:
  `42ca902b06cc840b19df409d2869bad21db3f0d6b78c3850510938b31e13b588`.
- Source session: `20260827T171528.349289Z-c0e2caa7`.
- Source interaction context: `hand-location-confounded`.
- Source condition: LEFT = left hand + left location; RIGHT = right hand +
  right location.
- Source membership: 20 LEFT + 20 RIGHT accepted samples, all 40 used.
- Source dataset SHA-256:
  `6c4ac4c3881faecbab430d593b6b217e03f58a13ca07522b679ee3d19706516f`.
- Primary feature: `peak_ratio_db_ch2_minus_ch1`.
- Frozen threshold: +0.12793235855251162 dB.
- Frozen direction: LEFT below the threshold; RIGHT at or above it.
- Tie rule: `feature_value >= threshold_db predicts higher_feature_zone`.

The feature is defined as:

```text
20 * log10(channel_2_peak_absolute / channel_1_peak_absolute)
```

**External session and physical protocol**

- Session ID: `20260829T101459.893269Z-566a8435`.
- External dataset SHA-256:
  `23f4c56a708ac43b570abef4437760923af1eb542636ceb857dbbcd24cfb945a`.
- Accepted samples: 20 LEFT and 20 RIGHT, 40 total.
- Rejected or retried attempts: zero.
- Interaction context: `same-hand`.
- The same right index finger made every LEFT and RIGHT tap.
- The Lenovo laptop, wooden desk/setup, intended LEFT/RIGHT geometry, Windows
  WDM-KS endpoint, 48 kHz sample rate, two-channel configuration, and 200 ms
  retained tap-window design were held consistent with the intended protocol.
- The classifier was not viewed or tuned during collection.

The external dataset did not exist when the frozen baseline was committed.

**No-refit evaluation procedure**

The external evaluator loaded the committed baseline and applied its stored
feature definition, +0.12793235855251162 dB threshold, direction, and tie rule
to all 40 accepted external samples. The report explicitly records:

- External samples used to fit threshold: false.
- Threshold refit performed: false.
- Direction relearned: false.
- Feature selection performed: false.
- External normalization fitted: false.

No external statistic influenced an official prediction.

**Official external result**

- Correct: 39/40.
- Accuracy: 97.5%.
- Two-sided 95% Wilson score interval: 87.1183%–99.5573%.
- LEFT: 20/20 correct.
- RIGHT: 19/20 correct.

| Actual class | Predicted LEFT | Predicted RIGHT |
| --- | ---: | ---: |
| LEFT | 20 | 0 |
| RIGHT | 1 | 19 |

This result is labeled the first frozen-baseline cross-session same-hand
external evaluation. It is not a training result, within-session estimate, or
general DeskSense accuracy figure.

**Official misclassification and near-threshold observations**

The only misclassification was retained unchanged:

- Accepted sample: RIGHT #11.
- Sample ID: `20260829T101459.893269Z-566a8435-right_011`.
- Actual label: RIGHT.
- Predicted label: LEFT.
- Feature value: +0.050166168713707354 dB.
- Frozen threshold: +0.12793235855251162 dB.
- Actual-class margin: -0.07776618983880426 dB.

RIGHT #11 is a valid official external sample and a near-threshold miss. It was
not removed, relabeled, or used to tune the threshold.

Two correctly classified RIGHT samples were even closer to the threshold on
its positive side:

- RIGHT #15: feature approximately +0.142095 dB; margin approximately
  +0.014163 dB.
- RIGHT #9: feature approximately +0.163920 dB; margin approximately
  +0.035988 dB.

These margins show that the frozen absolute threshold lies near the lower edge
of the observed external RIGHT distribution. Margins are threshold distances,
not calibrated probabilities.

**Post-prediction descriptive distributions**

These statistics were calculated for interpretation after predictions and did
not alter the official result:

| Class | Mean | Observed range |
| --- | ---: | ---: |
| LEFT | -5.630947313147696 dB | -8.668097139293714 to -2.2228560154426775 dB |
| RIGHT | +1.2090723599046973 dB | +0.050166168713707354 to +2.070086215063918 dB |

The external class ranges did not overlap. The observed gap between the LEFT
maximum and RIGHT minimum was approximately 2.273022 dB. This descriptive
separation must not be used to revise the already recorded predictions.

**Cross-session shift**

| Class | Development mean | External mean | Approximate shift |
| --- | ---: | ---: | ---: |
| LEFT | -3.1017978964848574 dB | -5.630947313147696 dB | -2.529149 dB |
| RIGHT | +3.3576626135898806 dB | +1.2090723599046973 dB | -2.148590 dB |

LEFT/RIGHT separation remained strong, but both distributions shifted downward
between sessions. The one error occurred because the unchanged development
threshold sat slightly above one external RIGHT sample. This motivates future
study of calibration or session-offset handling; it does not establish a
calibration method, and the official frozen baseline and 39/40 result remain
unchanged.

**Interpretation**

This experiment is stronger evidence than the Phase 2B within-session result:

1. The external session did not exist when the model was frozen and committed.
2. The primary feature, threshold, direction, and tie rule remained unchanged.
3. The data came from a separate collection session.
4. The same right index finger was used at both locations, controlling the
   previous tapping-hand variable more cleanly.
5. The frozen model retained 39/40 performance.

The result provides strong evidence that location-dependent acoustic
information is present in the tested setup.

**Limitations**

- One user.
- One Lenovo laptop and microphone endpoint.
- One wooden desk/setup.
- Two LEFT/RIGHT zones.
- One external collection session.
- The development session used different tapping hands across zones, although
  the external session controlled hand with the same right index finger.
- The Wilson interval describes finite-sample uncertainty; it is not a promise
  of future performance.

The result does not establish general 97.5% DeskSense accuracy, cross-user,
cross-device, cross-desk, multi-zone, or product-level robustness.

**Resulting decision**

- Preserve RIGHT #11, the frozen baseline, and the complete 39/40 external
  result unchanged.
- Move the immediate engineering direction toward Phase 3: robust tap/event
  detection, real-time feature extraction, real-time frozen classification,
  and a confidence/rejection strategy, followed later by Windows action
  mapping.
- Treat calibration/session adaptation as a motivated future investigation,
  not a completed solution or change to the Phase 2C result.
- Treat this external session as development evidence once its measurements
  are used to change calibration or model design.
- Require another newly collected untouched session before making any future
  external-validation claim for a modified classifier.
- Leave additional robustness sessions and cross-device experiments as later
  validation work rather than the immediate next implementation action.

## Experiment 13 — Phase 3A.1 pure streaming detector and inference parity

**Purpose**

Implement and verify the hardware-independent DSP/state-machine boundary needed
before adding PortAudio callback behavior. The goal was to separate detector,
candidate-alignment, chunk-partition, and frozen-inference correctness from the
future risks introduced by a live audio adapter, callback queue, worker
scheduling, and physical microphone behavior.

This was an engineering/software milestone. No microphone capture, dataset
collection, external evaluation, or physical latency measurement was performed.

**Implemented architecture**

The pure streaming detector has four states:

1. `LEARNING` — accumulate startup noise statistics; triggering is disabled.
2. `ARMED` — evaluate fixed detector blocks for a causal onset.
3. `COLLECTING` — complete the bounded local center search and wait for the
   exact post-center samples required by the classifier window.
4. `REFRACTORY` — suppress nearby duplicate triggers before returning
   deterministically to `ARMED`.

The default intended input domain is finite float-compatible audio at 48 kHz
with two ordered channels. Caller chunks may have arbitrary lengths, but the
detector carries incomplete input forward and performs noise updates, onset
decisions, and state counters on fixed five-millisecond blocks. At 48 kHz each
internal block contains 240 frames. This makes caller or future callback
partitioning external to the defined detector timeline.

Startup noise learning lasts an effective 0.75 seconds / 36,000 frames at
48 kHz. No event can trigger during this state. The initial thresholds and
adaptation settings are explicit engineering defaults; they were not tuned
from Phase 2C external labels and are not yet validated against a continuous
Lenovo microphone stream.

**Causal onset and candidate-center rule**

An onset opens a bounded local search with the initial geometry:

- 12 ms / 576 frames before the onset.
- 25 ms / 1,200 frames after the onset.

The center search uses pooled squared multichannel energy, so opposite-polarity
channels cannot cancel through waveform averaging. It evaluates a five-
millisecond local-energy region, chooses the strongest such region, and then
chooses the largest instantaneous pooled-power frame inside it. Exact ties
resolve to the earliest frame.

A per-channel DC-removed copy may be used for this selection calculation only.
The raw retained samples are not filtered, normalized, aligned, averaged, or
otherwise changed.

Once the center is fixed and enough future data exists, the candidate is copied
exactly as:

```text
[center - 4,800 : center + 4,800]
```

At 48 kHz the result is exactly 9,600 x 2 float32 frames, representing 200 ms.
The tap-window frame count must be even; an odd count is rejected at
configuration time rather than creating an asymmetric centered window.

This causal selection rule is new. It is not claimed to reproduce Phase 2A's
global strongest-transient search over a complete 1.5-second guided attempt.
It preserves the validated raw 200 ms feature domain while replacing the
noncausal global search with a bounded onset-relative search suitable for a
future live stream.

The default refractory interval is 250 ms / 12,000 frames at 48 kHz. It is
sample-indexed and applies after both completed detections and structured
candidate rejections. This is an initial engineering setting, not a measured
optimum.

**Bounded memory and discontinuity behavior**

- Raw history uses a fixed circular float32 buffer.
- Default production history capacity: 11,040 frames.
- Two-channel history storage: 88,320 bytes, approximately 88 KB.
- The incomplete internal-block buffer is separately bounded below 240 frames.
- Caller arrays are validated and copied before processing and are not
  modified.
- Emitted candidate arrays own copied memory and remain unchanged as circular
  history advances.
- Invalid rank, channel count, numeric conversion, or finite-value input fails
  before detector state mutation.
- Missing required history produces a structured rejection; candidates are not
  padded, shifted, or truncated.
- `reset()` returns to deterministic initial state.
- A notified discontinuity clears history, partial blocks, pending candidates,
  refractory state, and learned noise state, starts a new continuous epoch,
  and restarts `LEARNING`. A candidate cannot span a known audio gap.

**Synthetic onset/center distinction**

A deterministic two-stage synthetic event first crossed the onset threshold at
frame 203. A stronger transient occurred later at frame 210 inside the local
search interval. The detector recorded:

- Onset frame: 203.
- Refined strongest-transient center: 210.
- Small-test candidate: `[110:310]`.

The candidate was centered on frame 210 rather than the earlier causal onset,
and its bytes matched the corresponding source slice. This verifies the
intended software distinction between onset detection and center refinement;
it is not evidence about alignment on real desk taps.

**Chunk-partition invariance**

Another deterministic impulse was placed at non-internal-block-aligned frame
203 in the small test configuration. The same continuous waveform was supplied
as:

- One complete chunk.
- Irregularly sized chunks.
- One frame per call.

All three tested partitions produced identical onset and center indexes, exact
window start/end indexes, and candidate bytes. Additional tests cover fixed
chunks and events crossing caller boundaries. This establishes deterministic
behavior for the tested synthetic inputs, not for an untested PortAudio stream.

**Label-free frozen inference**

The primary feature remains unchanged:

```text
peak_ratio_db_ch2_minus_ch1 =
    20 * log10(channel_2_peak_absolute / channel_1_peak_absolute)
```

The new pure helper is:

```text
classify_peak_ratio_value(
    feature_value_db,
    threshold_db,
    lower_feature_zone,
    higher_feature_zone,
)
```

Its decision rule is unchanged from the frozen model:

- `feature < threshold` predicts `lower_feature_zone`.
- `feature >= threshold` predicts `higher_feature_zone`.

Exact ties therefore predict the higher-feature zone. The helper supports both
normal and reversed LEFT/RIGHT direction, validates finite feature and
threshold values and the two class identities, requires no actual label, and
performs no fitting, normalization, or calibration. Reported margins are dB
distances from the threshold, not probabilities.

The existing offline `analysis.predict_peak_ratio()` now delegates only this
threshold decision to the shared helper. Regression tests verified parity for
normal direction, reverse direction, both sides of the threshold, exact ties,
and invalid/non-finite values. Phase 2B and Phase 2C report semantics remain
unchanged.

**Automated verification**

- Streaming and inference focused tests: 63 passed.
  - Streaming: 33 passed.
  - Inference: 30 passed.
- Feature, analysis, and frozen-baseline regression selection: 89 passed.
- Complete automated suite: 280 passed.
- `pip check`: passed.
- `compileall`: passed.
- Pure import check: passed; importing streaming/inference did not import
  sounddevice.
- `git diff --check`: passed.
- No automated test accessed microphone hardware.

Tests also cover startup trigger suppression, quiet input, a single isolated
event, exact window length and center placement, channel preservation, input
immutability, owned candidate memory, clipping rejection, unavailable history,
refractory behavior, reset/replay, discontinuity, invalid-input atomicity,
bounded history, direct compatibility with
`extract_two_channel_features()`, and deterministic earliest-frame tie
handling.

**Limitations and unknowns**

- No `sounddevice.InputStream` adapter or live `--sense` CLI exists.
- No real WDM-KS callback size, timing, or overflow behavior has been measured.
- Initial onset thresholds, adaptive noise behavior, center-search geometry,
  and refractory duration have synthetic coverage but no continuous Lenovo
  validation.
- Causal center alignment on real taps is unknown.
- False positives from speech, typing, desk bumps, and handling noise are
  unknown.
- Weak-tap recall and sustained-noise behavior are unknown.
- Callback-to-worker queue behavior, Python scheduling, and end-to-end latency
  are not implemented or measured.
- No Windows action, hotkey, GUI, confidence cutoff, calibration change, or new
  classifier was introduced.

No real-time sensing accuracy, detection recall, false-positive rate, or
latency result is claimed from these synthetic tests.

**Resulting decision**

- Treat Phase 3A.1 pure detector and label-free inference implementation as
  complete and independently code-reviewed.
- Complete Checkpoint #7 before adding hardware integration.
- Proceed next to Phase 3A.2: an injected `sounddevice.InputStream` adapter,
  bounded callback-to-worker transport, discontinuity propagation, and a live
  terminal `--sense` path that loads the existing frozen baseline unchanged.
- Continue deferring Windows action execution until physical real-time sensing
  is reliable.

## Experiment 14 — Phase 3A.2a injected live adapter and fake-tested sensing CLI

**Purpose**

Connect the reviewed Phase 3A.1 detector and label-free frozen inference path
to an injectable PortAudio-compatible input stream while keeping callback work
bounded and testable. This was an engineering/software milestone, not a
physical microphone experiment. No `--sense` stream was opened, no microphone
hardware was accessed, and no physical callback, latency, detection, or
classification result was produced.

**Implemented architecture**

The live path is:

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

No additional worker thread was introduced. The realtime module does not
import sounddevice; the CLI preserves lazy loading and injects the selected
backend. The stream is constructed with the explicit current device, 48 kHz,
two channels, float32, `blocksize=0`, the input callback, and a finished
callback. PortAudio may therefore choose host-appropriate callback sizes, while
the pure detector retains its deterministic internal 5 ms / 240-frame blocks.

**Frozen-domain compatibility**

Live operation requires the existing 48 kHz, two-channel, float32,
9,600-frame / 200 ms candidate domain. The current endpoint name and host API
must agree with the frozen artifact; Windows WDM-KS is expected for the current
Lenovo baseline. The historical numeric device index is treated as
session-dependent, so a changed current index is allowed when endpoint
identity and all feature-domain requirements still match.

The baseline feature, threshold, direction, and tie rule remain unchanged. No
live fitting, normalization, calibration, or model modification occurs.

**Callback and bounded transport**

Each callback receives a zero-based monotonically increasing sequence number,
takes one owned contiguous float32 copy, snapshots primitive PortAudio
time/status fields and Python monotonic callback/enqueue-attempt timestamps,
and attempts a non-blocking queue insertion. It performs no detector DSP,
feature extraction, inference, terminal output, JSON work, or filesystem work.

The initial queue holds at most eight callback packets. If it is full, the
current/newest packet is dropped so the callback does not block and older
queued packets retain order. Known dropped callback and frame counts
accumulate, and the next packet successfully retained carries explicit loss
metadata. Eight packets is an initial engineering value; its adequacy on the
Lenovo has not been measured.

**Discontinuity and reset behavior**

Queue overflow, PortAudio input overflow, callback sequence gaps, and other
relevant PortAudio status problems identify known continuity loss. Before the
first retained post-gap packet reaches the detector, the main thread calls
`detector.notify_discontinuity()` exactly once for that boundary. Multiple
reason codes still cause a single reset.

The reset clears history, partial internal blocks, pending candidate state,
refractory state, and learned noise state, then begins a new stream epoch in
`LEARNING`. No candidate intentionally joins samples from opposite sides of a
known gap. The adapter records callback/frame losses it knows about but does
not fabricate a count for frames discarded internally by PortAudio.

**Classification and terminal events**

Only `DetectionResult.status == "detected"` enters feature extraction and
frozen classification. A rejected result is not classified even if it retains
a candidate window, including a near-clipping rejection. A detected exact
9,600 x 2 candidate uses the existing complete-window
`peak_ratio_db_ch2_minus_ch1` feature and label-free frozen decision helper. If
the feature is undefined, the system emits a structured rejection without
inventing a zone. Margin is reported as a dB distance from the threshold, not
as a probability.

The `--sense` CLI requires an explicit device and baseline and is mutually
exclusive with the existing operational modes. It formats startup/learning,
armed, discontinuity/relearning, detected LEFT/RIGHT, and rejected-candidate
events. It writes no audio or report and executes no Windows action.

**Timing instrumentation**

The adapter preserves, where available, PortAudio `inputBufferAdcTime`,
callback `currentTime`, Python callback-arrival and enqueue-attempt monotonic
timestamps, main-loop processing start, detector-result availability,
feature/inference completion, stream-reported latency, queue dwell, detector
lookahead, and approximate onset/center ADC mapping. It does not directly
subtract PortAudio absolute times from Python performance-counter values
because their clock origins may differ. No precise impact-to-terminal latency
is inferred from fake testing.

**Hardware-independent verification**

- Realtime tests: 39 passed.
- Focused `--sense` CLI tests: 21 passed, 52 deselected.
- Complete CLI tests: 73 passed.
- Streaming, inference, features, analysis, and frozen-baseline regression
  selection: 152 passed.
- Complete suite: 340 passed.
- `pip check`, `compileall`, lazy import checks, and `git diff --check` passed.
- Importing realtime/CLI did not import sounddevice.
- No microphone hardware was accessed and no physical `--sense` command ran.

Fake coverage includes exact `InputStream` arguments, settings validation
before start, endpoint and host-API compatibility, changed current device
indexes, opened-stream mismatch rejection, callback ownership and PortAudio
buffer reuse, channel order, queue bounds and drop propagation, PortAudio
overflow, fatal callback errors, ordered variable-size packets,
discontinuity-before-processing, prevention of cross-gap history, detected-only
classification, near-clipping and undefined-feature rejection, unchanged
frozen inference, result/event order, startup/armed/relearning transitions,
unexpected stream termination, deterministic stop, Ctrl+C-style cleanup,
startup failure, cleanup-error precedence, and absence of audio/report
persistence.

**Limitations**

Actual Lenovo WDM-KS callback sizes and cadence, PortAudio status behavior,
queue high-water and overflow frequency, physical noise-floor adaptation,
continuous-audio onset thresholds, false triggers from typing/speech/desk
movement, weak-tap recall, causal center alignment, frozen feature behavior on
live causal windows, observed margins, and end-to-end latency remain unknown.
Fake tests establish software behavior only; they do not establish physical
live sensing performance.

**Resulting decision**

- Treat Phase 3A.2a implementation and fake validation as complete and
  independently code-reviewed.
- Complete Checkpoint #8 before opening the first live microphone stream.
- Proceed next to Phase 3A.2b, the first physical live Lenovo sensing pilot.
- Continue deferring Windows actions, hotkeys, GUI behavior, calibration, and
  model changes until physical sensing behavior is measured.

## Experiment 15 — first physical continuous live sensing pilot

**Date**

2026-08-30.

**Purpose**

Run the Checkpoint #8 continuous realtime pipeline against the actual Lenovo
microphone for the first time and verify basic stream startup, detector arming,
positive tap detection, unchanged frozen LEFT/RIGHT inference, transport
behavior visible in the terminal, and clean shutdown. This was a short initial
pilot, not a robustness, false-positive-rate, weak-tap-recall, or formal
real-time accuracy experiment.

**Physical setup**

- Laptop: Lenovo Windows laptop.
- Endpoint: `Microphone Array 1 (Intel® Smart Sound Technology (Intel® SST)
  Microphone)`.
- Current device index: 18.
- Host API: Windows WDM-KS.
- Runtime domain: 48 kHz, two channels, float32.
- Desk: the same wooden desk setup used in the preceding experiments.
- Baseline: `baselines/lenovo-left-right-v1.json`.
- Frozen source session: `20260827T171528.349289Z-c0e2caa7`.
- Frozen threshold: approximately +0.127932 dB; the exact stored value was
  unchanged.
- Decision rule: LEFT below the threshold; RIGHT at or above it.
- No refitting, normalization, calibration, or threshold modification.

The established tap points were approximately 7–10 cm outside the respective
laptop edges. They were not centered vertically beside the laptop. Both were
toward the user/touchpad side and away from the screen, approximately
lower/front-left and lower/front-right relative to the laptop. These were the
same established tap locations used in the relevant earlier experiments.

Every tap used the right hand and the fleshy pad of the right index finger with
a moderate natural impact. The same hand and finger therefore produced both
LEFT and RIGHT taps.

**Command and startup**

```powershell
python -m desksense --sense --device 18 --baseline baselines\lenovo-left-right-v1.json
```

Startup output confirmed the intended endpoint, Windows WDM-KS, 48 kHz, two
channels, float32, a reported stream latency of 47.00 ms, and the 0.75-second
initial noise-learning period. The detector armed successfully in its original
continuous epoch:

```text
Armed (detector epoch 0).
```

**Procedure**

The intended physical sequence was:

1. LEFT
2. LEFT
3. LEFT
4. RIGHT
5. RIGHT
6. RIGHT
7. RIGHT

The seventh tap was an extra RIGHT tap. The user added it because they initially
thought the preceding RIGHT tap had not been detected; later inspection of the
terminal showed that the preceding tap had in fact produced a detection. This
is an early UX/terminal-attention observation, not by itself evidence that
detection latency was excessive.

**Observed live events**

| # | Intended zone | Output zone | Feature (dB) | Margin (dB) | Onset | Center | Queue dwell (ms) | Detector lookahead (ms) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LEFT | LEFT | -7.106153 | 7.234085 | 213540 | 214185 | 0.13 | 102.81 |
| 2 | LEFT | LEFT | -8.166028 | 8.293960 | 360098 | 360754 | 0.10 | 104.29 |
| 3 | LEFT | LEFT | -4.018297 | 4.146229 | 826800 | 827949 | 0.11 | 101.06 |
| 4 | RIGHT | RIGHT | +2.447690 | 2.319758 | 1223811 | 1224472 | 0.11 | 100.17 |
| 5 | RIGHT | RIGHT | +2.196171 | 2.068238 | 1494843 | 1495495 | 0.28 | 103.85 |
| 6 | RIGHT | RIGHT | +1.736659 | 1.608726 | 1709463 | 1710152 | 0.05 | 101.83 |
| 7 | RIGHT | RIGHT | +2.936812 | 2.808879 | 1842295 | 1842326 | 0.08 | 103.21 |

All seven intended taps produced live detections. The three intended LEFT taps
were output as LEFT, and the four intended RIGHT taps were output as RIGHT. All
observed feature values fell comfortably on the expected side of the unchanged
frozen threshold. The seven events were not used to fit a new threshold or
replace the frozen artifact.

No `Tap rejected` event, `Audio discontinuity` event, `queue_overflow` warning,
or PortAudio `input_overflow` warning was observed. The detector remained in
epoch 0, and Ctrl+C shutdown completed cleanly.

**Transport and timing observations**

The stream-reported latency was 47.00 ms. Queue-dwell values were 0.13, 0.10,
0.11, 0.11, 0.28, 0.05, and 0.08 ms. They provide no evidence of queue backlog
during this short run, but do not validate queue capacity eight as universally
sufficient.

Detector-lookahead values were 102.81, 104.29, 101.06, 100.17, 103.85, 101.83,
and 103.21 ms. These are detector/algorithm timing evidence and are not precise
end-to-end user-perceived latency.

The terminal also emitted values such as:

```text
approx. onset-to-result=82331177.79 ms
approx. onset-to-result=82331163.10 ms
```

These approximately 82-million-ms values are physically impossible and
invalid. They are preserved here as evidence of a timing instrumentation
defect, not as latency measurements. The likely high-level interpretation is
that the PortAudio stream-time value and ADC-time estimate were not safely
comparable under the current calculation on this physical backend/run. The
exact root cause has not been established.

This metric must not be clamped, silently reinterpreted, or used in project
claims. Phase 3A.3 should validate clock compatibility and physical
plausibility, falling safely to `unavailable` rather than emitting an invalid
duration.

**Interpretation and limitations**

The defensible result is:

> The first continuous live Lenovo WDM-KS pilot detected and correctly
> classified all 7 intended same-hand LEFT/RIGHT taps using the previously
> frozen classifier without refitting or calibration.

This is not a 100% localization-accuracy, production-accuracy, or generalizable
performance claim. The run was short and positive-case-focused. It does not
measure a false-positive rate, weak-tap recall, callback size or cadence,
long-run queue behavior, background-noise robustness, causal-window equivalence
to the offline collector, or true end-to-end latency. The absence of a warning
or rejection applies only to the observed run and must not be generalized to
unobserved conditions.

**Resulting decision**

- Treat Phase 3A.2a as complete and checkpointed.
- Record Phase 3A.2b as a successful first physical pilot, not a broad
  robustness result.
- Proceed to Phase 3A.3 live robustness and measurement work.
- First make invalid cross-clock onset-to-result timing fail safely to
  `unavailable` before relying on that metric.
- Then perform controlled live robustness experiments rather than immediately
  refitting or recalibrating the frozen classifier.

## Experiment 16 — Phase 3A.3 clock-origin-safe timing fix

**Purpose**

Correct the live timing instrumentation defect exposed by Experiment 15 before
using approximate onset/center timing in further physical claims. This was an
engineering/software change only. No microphone hardware was accessed and
`--sense` was not executed.

**Historical defect retained as evidence**

The first physical WDM-KS pilot emitted values including:

```text
approx. onset-to-result=82331177.79 ms
approx. onset-to-result=82331163.10 ms
```

Those physically impossible approximately 82-million-ms observations remain
part of Experiment 15 and are not rewritten, clamped, or removed. They showed
that the previous derived timing path was invalid on that physical run. The
old calculation was conceptually equivalent to:

```text
stream.time - estimated_onset_adc_time
```

The WDM-KS result demonstrated that those absolute values could not safely be
treated as directly comparable in that calculation. The exact backend/root
cause has not been proven.

**Revised duration composition**

The implementation now constructs the estimate from two durations rather than
mixing absolute clock origins:

```text
PortAudio onset-to-callback duration =
    callback_current_time_seconds - estimated_onset_adc_time_seconds

Python callback-to-result duration =
    (result_available_monotonic_ns - callback_arrival_monotonic_ns) / 1e9

approximate onset-to-result duration =
    PortAudio onset-to-callback duration
    + Python callback-to-result duration
```

Center-to-result uses the corresponding center ADC estimate. PortAudio
`callback_current_time_seconds` and the ADC estimate are compared only within
the PortAudio timing domain. Callback-arrival and result-availability values
are compared only within the injected Python monotonic/performance-counter
domain. No PortAudio absolute timestamp is directly subtracted from a Python
absolute timestamp.

`stream.time` remains recorded as raw backend diagnostic evidence. It no
longer participates in the approximate onset-to-result or center-to-result
calculation.

**Fail-safe behavior**

If any required timing component is missing, non-finite, negative, or otherwise
unusable, the corresponding approximate metric becomes `None`/unavailable.
The implementation does not clamp an implausible value, infer a clock offset,
calibrate clock domains, guess missing timestamps, or fall back to another
absolute-clock subtraction.

The resulting metrics remain approximate instrumentation. They are not precise
physical impact-to-terminal latency or user-perceived latency.

**Unchanged behavior**

The fix does not change detector thresholds, startup learning, refractory
timing, center-search logic, the 200 ms candidate window, queue capacity or
drop policy, callback responsibilities, frozen artifact, peak-ratio feature,
LEFT/RIGHT decision rule, endpoint matching, or `InputStream` settings.

**Hardware-independent verification**

- Focused timing selection: 12 passed, 38 deselected.
- Complete realtime suite: 50 passed.
- Complete suite: 351 passed.
- `pip check`: passed.
- `compileall`: passed.
- Lazy import check: passed.
- `git diff --check`: passed.
- No microphone hardware was accessed.
- No `--sense` command was executed.

Tests cover the expected composition of a 0.100-second PortAudio duration and
a 0.020-second Python duration into a 0.120-second onset-to-result estimate,
the equivalent center calculation, independent absolute clock origins, an
unrelated huge `stream.time`, missing ADC/callback timing, non-finite values,
negative/reversed durations, unchanged queue dwell and detector lookahead, and
CLI omission when approximate timing is unavailable.

**Limitations and resulting decision**

This fake-tested change has not yet run against the physical Lenovo WDM-KS
backend. It therefore does not establish that the defect is physically
resolved or provide a new latency result.

- Treat the timing instrumentation fix as implemented, fake-tested, and
  independently code-reviewed.
- Checkpoint the change before physical validation.
- Rerun a short controlled Lenovo live pilot afterward.
- Require the revised values to be physically sane or fail safely to
  unavailable before using them as timing evidence.
- Do not refit the classifier or change detector behavior as part of this
  timing-only work.

## Experiment 17 — Phase 3B Development Session A and offline detector-design study

**Date:** September 1, 2026

**Purpose and evidence boundary**

Collect the first dedicated Phase 3 robustness development evidence and use it
to study candidate-generation and tap/non-tap validation designs without
changing the production detector or frozen spatial classifier. Session
`20260901T131308.362205Z-f9e2b1ec` is development evidence only. Its local raw
waveforms remain under `datasets/`; the existing replay report
`reports/phase3b0-development-a-replay.json` is waveform-free.

The complete session contains 30 intended positive taps and 14 labeled
negative segments. Strict loading validated all 44 referenced NPZ artifacts,
manifest/embedded metadata agreement, contiguous ordering, finite float32
stereo data, 96,000-frame positive captures, 540,000-frame negative captures,
and the 48 kHz/two-channel domain.

**Unchanged-detector replay**

- Positive Stage 1 candidate-start recall: 8/30 (26.67%).
- LEFT: 6/15; RIGHT: 2/15.
- Light: 0/10; normal: 4/10; firm: 4/10.
- All eight associated completions were detections; none was rejected.
- Frozen LEFT/RIGHT classification was 8/8 among those detected positives.
  This is conditional spatial performance; end-to-end positive success was
  only 8/30.
- Negative labeled duration: 140 seconds.
- Candidate starts/completed false events: 47/47, or 20.143/min.
- Rates by activity: quiet 0/min, speech 0/min, typing 42/min, trackpad
  21/min, hand movement 12/min, laptop movement 12/min, and desk/object
  interaction 54/min.

No Stage 2 tap/non-tap validator existed. The frozen spatial feature,
threshold, direction, and artifact were unchanged.

**Development-only offline findings**

An association-window oracle used the strongest pooled-multichannel 5 ms
energy region only to inspect the saved positive waveforms; it is not a live
algorithm. An ordered counterfactual attribution of the 22 current misses
associated six primarily with fixed-block phase, two with adaptive-floor
contamination, thirteen with crest-only failure despite adequate RMS/peak
evidence, and one with combined RMS/peak/crest failure. This attribution is a
development diagnostic, not physical ground truth.

Evaluating the unchanged 5 ms gate every 2.5 ms raised positive recall from
8/30 to 14/30, while negative candidates rose from 47 to 59
(20.143 to 25.286/min). A smaller candidate-generator alternative retained the
existing non-overlapping 5 ms timeline and added a development-selected strong
RMS/peak route: RMS gate ratio at least 6 and peak gate ratio at least 8,
without globally lowering crest. It produced candidates for 29/30 positives
and 50 negative events (21.429/min). All 29 positive candidates retained the
existing raw 9,600-by-2 window geometry and were conditionally classified to
the intended side by the unchanged frozen spatial baseline. Those 29/29
results are development-only and were not used as a tapness score.

Failed high-energy blocks often raised the current EWMA floor: 125 such
positive-interval updates and 122 negative-interval updates increased it by at
least 10% in one step. Freezing floor updates on RMS-and-peak-passing blocks
raised overlapping-window positive recall only from 14/30 to 16/30 while
raising negative candidates from 25.286 to 39.0/min. An exploratory four-times
upward observation cap recovered no additional positives. Floor-policy changes
were therefore deferred from the first Stage 1 implementation.

All nine existing descriptive tapness features had overlapping positive and
negative ranges. On the 29 positive and 50 negative candidates from the
non-overlapping strong-route variant, a development-selected contrast and
shape envelope accepted 29/29 positives but also 8 negatives (3.429/min).
Adding an absolute impact-RMS floor retained 29/29 but still accepted two
desk/object events (0.857/min), and introduced a session-level amplitude
dependency. These deterministic rules do not meet the planned untouched
negative gate and are not validation results.

**Resulting decision**

- Keep Stage 3 frozen LEFT/RIGHT classification and its 200 ms raw feature
  domain unchanged.
- For the next implementation, add the single strong RMS/peak recovery route
  on the existing non-overlapping timeline; do not add overlap or change the
  adaptive floor in the same iteration.
- Treat the remaining Stage 2 overlap as justification to study a small,
  regularized, interpretable tap/non-tap model over a predeclared version of
  the existing descriptive feature vector rather than accumulating brittle
  hard gates. Session A is its development source, not its validation set.
- Freeze the complete Stage 1/Stage 2 design before collecting a new untouched
  Session B. Session B must separately measure Stage 1 recall, Stage 2 positive
  survival, Stage 2 false accepts over a five-minute scripted negative run,
  and conditional frozen spatial correctness.

**Phase 3B.2 implementation and development fit**

The production candidate generator now retains the ordinary RMS/peak/crest
route unchanged and adds exactly one non-overlapping strong-impact recovery
route: RMS gate ratio at least 6 and peak gate ratio at least 8. Ordinary-route
diagnostic labeling takes precedence when both routes pass. No overlapping
window, floor-policy, center-refinement, refractory, candidate-geometry, or
spatial-classifier change was made.

Production replay reproduced the reviewed Session A Stage 1 set exactly:
29/30 intended taps generated associated candidates (LEFT 15/15, RIGHT 14/15;
light 10/10, normal 9/10, firm 10/10), while 50 negative candidates over 140
labeled seconds corresponded to 21.429/min.

The versioned `baselines/lenovo-tapness-v1.json` artifact fits deterministic
L2-regularized binary logistic regression to those 29 TAP and 50 NON_TAP
candidates. Its fixed nine-feature vector contains no spatial feature, margin,
threshold, or predicted zone. Positive scale/ratio/duration descriptors use
fixed numerically safe natural-log transforms; the two bounded fractions stay
linear. Standardization is stored in the artifact. The L2 value of 0.01 was
selected from a small Session A development-only comparison of 1.0, 0.1, and
0.01; it is therefore tuned development evidence.

Five-fold grouped development evaluation kept every event from one recording
in one fold and fitted means, scales, and model parameters on training folds
only. The deterministic recall-constrained operating-point rule selected an
uncalibrated model-output threshold of `0.4495211534633274`. Out-of-fold, it
accepted 28/29 TAP candidates and falsely accepted 5/50 NON_TAP candidates.
This is grouped development cross-validation, not external validation.

After refitting the fixed specification on all Session A candidates, replay of
the frozen artifact accepted 28/29 generated positive candidates, or 28/30
intended attempts overall. Breakdown by intended condition was LEFT 15/15,
RIGHT 13/15, light 9/10, normal 9/10, and firm 10/10. Stage 2 falsely accepted
4/50 negative candidates: two from laptop movement and two from desk/object
interaction, with none from quiet, typing, speech, trackpad, or hand movement.
That is 1.714 false accepts/min over the 140 labeled development seconds.
Frozen spatial classification remained unchanged and was correct for all
28 Stage 2-accepted positives, giving a Session A development-only end-to-end
result of 28/30. These final-fit replay numbers are resubstitution evidence and
must not be substituted for untouched Session B validation.

The compact artifact binds the exact Session A source files with dataset
SHA-256 `88a003966141d15858a2afec390c42e567439eb5f538b958e7aa60ee7638ab83`,
records complete candidate membership and negative recording groups, contains
no waveform arrays, and declares that no external samples were used. The next
evidence gate remains a new untouched Session B collected only after the full
Stage 1/Stage 2 pipeline is reviewed and frozen.

## Experiment 18 — frozen Phase 3B.2 Session B external validation

**Date:** September 4, 2026

**Purpose and evidence boundary**

Evaluate the complete frozen Phase 3B.2 pipeline on a new untouched session,
without fitting, refitting, calibration, feature selection, threshold changes,
or Stage 1 policy changes. Session `20260904T171453.458221Z-4ecb16ad` was
collected and evaluated with evidence role `external_validation`; its official
local waveform-free report is `reports/phase3b-session-b-external.json`.

Collection used Microphone Array 1 (Intel Smart Sound Technology (Intel SST)
Microphone), Windows WDM-KS device index 18, at 48 kHz, two channels, and
float32. The same right index finger and fleshy fingertip pad were used for
exactly one intended tap per cue at both established lower/front zones,
approximately 7–10 cm outside the corresponding laptop edge and toward the
user/touchpad side. The Lenovo laptop and wooden-desk setup were unchanged, and
the user reported no known protocol irregularities.

The complete dataset contains 30 positive records and seven negative activity
records. The negative denominator is exactly 300 labeled seconds across seven
separately recorded and replayed segments, each with its own excluded warm-up
and completion tail. Dataset fingerprint
`6b6b7cff6ff4f597d0c4c9bc8ccea3544cda99e4d25bf9be1cc732ee6b9ee5ac`
was identical before and after replay. The frozen Stage 2 artifact SHA-256 was
`2c980a0d3ae05bf8f74d9bf3b9ba4a2f35cb9e679e9273f6f55eaf4e6778dbde`;
the frozen Stage 3 artifact SHA-256 was
`42ca902b06cc840b19df409d2869bad21db3f0d6b78c3850510938b31e13b588`.
Both artifact hashes were identical before and after replay.

**Official results**

- Stage 1 generated an associated candidate for 30/30 intended taps: LEFT
  15/15, RIGHT 15/15, and light/normal/firm 10/10 each. Every side-by-strength
  cell was 5/5.
- Frozen Stage 2 accepted 30/30 intended taps as TAP, with the same complete
  side, strength, and cell counts.
- Frozen Stage 3 classified 28/30 accepted intended taps correctly (93.33%).
  The confusion matrix was actual LEFT: 15 LEFT, 0 RIGHT; actual RIGHT: 2 LEFT,
  13 RIGHT. Both errors were actual RIGHT taps predicted LEFT.
- End-to-end correct intended-zone output was 28/30 (93.33%).
- Stage 1 generated 74 negative candidates over 300 labeled seconds, or
  14.8/min. Stage 2 falsely accepted eight, or 1.6/min.
- Negative breakdown was: quiet 30 s, 0 candidates/0 false accepts; typing
  45 s, 12/1; speech 45 s, 0/0; trackpad 45 s, 12/1; hand movement 45 s,
  20/0; laptop movement 45 s, 10/2; desk/object interaction 45 s, 20/4.

**Predeclared gate result**

The hard gates were Stage 1 candidate recall at least 27/30, Stage 2 intended
TAP acceptance at least 27/30, end-to-end correct intended-zone output at least
27/30, and at most one Stage 2 false accept over the exact 300 labeled negative
seconds. The first three passed; the negative gate failed with eight false
accepts. The official overall engineering gate is therefore **FAIL**, caused by
negative false acceptance rather than intended-tap recall. The preference for
at least 4/5 Stage 2 acceptance in every side-by-strength cell was met at 5/5,
but it was not a hard gate and does not alter the overall result.

**Interpretation and next decision**

The Stage 1 redesign addressed the earlier intended-tap recall failure in this
untouched session, and Stage 2 preserved all intended taps. The remaining
official robustness failure is mechanical non-tap rejection, concentrated in
desk/object and laptop interaction, with one false accept each from typing and
trackpad. Quiet and speech produced no Stage 1 candidates. Hand movement
produced 20 Stage 1 candidates, all rejected by Stage 2, showing useful
downstream discrimination. These findings apply only to this user, setup, and
session and do not establish general reliability. Windows actions remain
deferred.

Session B is permanently preserved as the first external validation of the
frozen Phase 3B.2 pipeline, including this official FAIL. From this point, if
its waveforms, events, or results are used to select, modify, tune, or train any
Stage 1, Stage 2, Stage 3, feature, threshold, model, or calibration, it becomes
development evidence for that later pipeline. The historical result is not
erased or replaced, and any modified pipeline requires a new untouched Session
C before another external-validation claim. The next step is read-only Phase
3B.3 failure analysis using Sessions A and B as development evidence.

## Local report handling

Earlier generated diagnostic and characterization JSON reports exist locally.
Generated files under `reports/` are intentionally ignored by Git and are not
part of the source history. Raw audio is not stored in those reports.

The four corrected spatial filenames above are the valid repeatability records;
false-start recordings are excluded from interpretation even if files remain
locally. Phase 2A waveform datasets are a distinct local artifact type and are
ignored by Git by default. The pilot dataset remains local unless explicitly
moved or shared by the user.

Future entries should preserve the endpoint name and host API as well as the
session-local device index, physical setup, tap timing/location, exact command,
report filename or terminal-only evidence status, measurements, limitations,
and resulting decision.
