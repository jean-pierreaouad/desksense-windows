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
