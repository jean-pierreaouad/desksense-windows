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

## Local report handling

Earlier generated diagnostic and characterization JSON reports exist locally.
Generated files under `reports/` are intentionally ignored by Git and are not
part of the source history. Raw audio is not stored in those reports.

Future entries should preserve the endpoint name and host API as well as the
session-local device index, physical setup, tap timing/location, exact command,
report filename or terminal-only evidence status, measurements, limitations,
and resulting decision.
