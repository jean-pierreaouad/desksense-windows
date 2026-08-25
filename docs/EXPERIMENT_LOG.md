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

## Local report handling

Earlier generated diagnostic and characterization JSON reports exist locally.
Generated files under `reports/` are intentionally ignored by Git and are not
part of the source history. Raw audio is not stored in those reports.

The four corrected spatial filenames above are the valid repeatability records;
false-start recordings are excluded from interpretation even if files remain
locally. Future Phase 2A waveform datasets will be a distinct local artifact
type and must also be ignored by Git by default.

Future entries should preserve the endpoint name and host API as well as the
session-local device index, physical setup, tap timing/location, exact command,
report filename or terminal-only evidence status, measurements, limitations,
and resulting decision.
