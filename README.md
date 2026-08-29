# DeskSense for Windows

DeskSense aims to become a hardware-adaptive acoustic tap input system for
Windows laptops. The long-term goal is to use built-in microphones to detect
desk taps, determine their zones, and map them to programmable Windows actions.
The project is inspired by the MIT-licensed [Holo project](https://github.com/JustinGamer191/Holo).

## Current milestone

Milestone 1 established the microphone and audio-hardware feasibility probe.
It enumerates input devices, identifies the default input and host API, checks
common sample rates, and can measure a short in-memory recording. Milestone 1.5
adds exploratory channel and endpoint characterization. Phase 2A adds guided,
labeled local waveform collection. Phase 2B adds reproducible offline
LEFT/RIGHT feature analysis and within-session evaluation. Phase 2C adds the
offline ability to freeze that already selected simple baseline and later apply
it unchanged to a different session. The first precommitted, same-hand
cross-session evaluation classified 39 of 40 taps correctly (97.5%) on the
tested Lenovo laptop and desk setup, without refitting. It does not implement
the final real-time tap detector or classifier, production localization, a
GUI, hotkeys, or action mapping.

That 97.5% result is one scoped external experiment, not general DeskSense,
cross-user, cross-desk, or cross-device accuracy. Diagnostic and
characterization commands never save raw audio; only the separate, explicit
dataset-collection command retains waveforms.

## Environment setup

From the repository root in PowerShell, activate the existing Python 3.14
virtual environment and install the package with its test dependency:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

Run the automated tests with:

```powershell
python -m pytest
```

## Run the diagnostic

Inventory the available microphone hardware without recording:

```powershell
python -m desksense
```

Run the approximately three-second recording test and save a timestamped JSON
report under `reports/`:

```powershell
python -m desksense --record --save-report
```

The recording uses a validated supported configuration, preferring the
selected microphone's default sample rate, prints peak absolute amplitude and
RMS for each captured channel, then discards the samples. Use `--device INDEX`
to select a listed input other than the default. You may also give
`--save-report` an explicit `.json` path:

```powershell
python -m desksense --device 2 --record --save-report reports\my-laptop.json
```

## Characterize microphone channels

Milestone 1.5 adds exploratory microphone-channel characterization so that the
same physical microphone exposed through different Windows audio host APIs can
be examined before localization or classification work begins. First run the
inventory to find an input device index, then characterize that endpoint:

```powershell
python -m desksense --characterize --device INDEX
```

Characterization records approximately five seconds using a supported channel
count and sample rate. Samples remain in memory and are discarded after the
tool calculates per-channel levels, activity estimates, whole-recording channel
comparisons, and comparisons around the strongest short transient. These
metrics are exploratory and do not demonstrate tap localization.

Device indices are assigned by PortAudio and are only reliable for the current
device enumeration/session. Re-run the inventory before a later comparison.
Use distinct explicit JSON paths when characterizing several endpoints so the
reports can be compared without overwriting one another:

```powershell
python -m desksense --characterize --device 3 --save-report reports\mme.json
python -m desksense --characterize --device 7 --save-report reports\wasapi.json
```

The JSON reports contain device information and calculated metrics, never raw
audio.

## Collect a guided labeled dataset (Phase 2A)

Phase 2A provides an explicit terminal workflow for collecting reproducible
LEFT/RIGHT tap examples for later offline analysis. Re-run the hardware
inventory first because PortAudio device indices can change, then start a
session with the intended input endpoint:

```powershell
python -m desksense --collect-dataset --device INDEX --samples-per-zone 20
```

The collector has been physically validated on the current Lenovo endpoint.
That validates the collection workflow, not localization performance or
cross-laptop behavior.

The default order alternates `LEFT`, `RIGHT`, `LEFT`, `RIGHT`, and so on to
reduce simple time/order bias. Before every attempt, press Enter when ready;
the tool prints a silent `3`, `2`, `1` countdown, starts the microphone capture,
waits through a 200 ms pre-cue interval, and then displays `TAP NOW` while the
same capture is still running. This is a controlled guided cue; terminal
rendering latency and human reaction time are not measured. A quality failure
is explained and retries the same zone without advancing its accepted sample
number. After a usable capture, press Enter to accept it or type `R` to discard
it and retry, which protects labels when a false start still contains sound.

Each attempt captures about 1.5 seconds as float32 multichannel audio. An
accepted `.npz` artifact stores that full guided capture and an exact 200 ms
window centered on the strongest exploratory short-energy region. The 200 ms
choice preserves useful context but is not claimed to be optimal or a final tap
detector. Conservative checks reject obviously inactive, non-finite,
near-clipping, low-contrast, or boundary-truncated attempts. Every configured
spatial channel must be active under the existing channel-activity heuristic;
lag and observed LEFT/RIGHT feature values are not acceptance criteria.

Sessions are created without overwriting existing data under the Git-ignored
`datasets/` directory:

```text
datasets/<session-id>/
  session.json
  manifest.jsonl
  rejected_attempts.jsonl
  samples/
    left_001.npz
    right_001.npz
```

`session.json` records the system, PortAudio/backend, endpoint identity, actual
capture configuration, intended cue offset, targets, and quality parameters.
`manifest.jsonl` indexes accepted samples and repeats the intended guided-cue
timing; rejected-attempt metadata contains no waveform.
Accepted samples remain on this computer unless you explicitly move or upload
the dataset. Nothing is uploaded automatically, and `datasets/` must remain
untracked because it contains microphone recordings. If you use
`--dataset-root` to choose another path inside the repository, add that path to
`.gitignore` before collecting.

If Windows denies microphone access, enable it under **Settings > Privacy &
security > Microphone** and retry.

## Analyze a labeled dataset offline (Phase 2B)

Phase 2B validates and analyzes one completed Phase 2A session without opening
the microphone or changing the dataset files:

```powershell
python -m desksense --analyze-dataset datasets\<SESSION-ID> `
  --interaction-context hand-location-confounded
```

The command always prints a readable summary and creates a timestamped JSON
report under the Git-ignored `reports/` directory. Use an explicit destination
when a stable experiment filename is helpful:

```powershell
python -m desksense --analyze-dataset datasets\<SESSION-ID> `
  --interaction-context hand-location-confounded `
  --save-report reports\phase2b-main-session.json
```

The interaction context is required because tapping-hand information is not
stored in Phase 2A artifacts and must not be guessed from waveforms. Use
`hand-location-confounded` for the current 20+20 session. The `same-hand`
choice is reserved for a session whose documented procedure used the same
hand/finger for both zones; selecting it does not by itself make that session
an independent external validation.

Before calculating features, the loader verifies `session.json`, every
`manifest.jsonl` record, and every referenced NPZ. It checks sample identity and
numbering, embedded metadata equality, float32 array shape and finiteness,
sample rate/channel consistency, declared tap-window slices, and unexpected
unindexed NPZ artifacts. It fails without repairing or rewriting inconsistent
data. Rejected-attempt records are counted for diagnostics but are never used
as accepted labeled samples.

The fixed per-window feature set includes channel RMS and peaks, Ch2/Ch1 ratios,
zero-lag Pearson correlation, and normalized channel-difference energy. The
predeclared primary baseline is Ch2/Ch1 peak ratio in dB. For the 20+20
protocol, LEFT/RIGHT accepted samples 1–15 fit a midpoint threshold and samples
16–20 form the test set. A separate deterministic leave-one-pair-out estimate
fits a new threshold after excluding LEFT #N and RIGHT #N in each fold. Both
results are within-session evaluations, not independent external validation or
production accuracy.

The current dataset has an important experimental confound:

- `LEFT` means left desk location **and** the user's left hand.
- `RIGHT` means right desk location **and** the user's right hand.

Consequently, Phase 2B can measure separation between those observed
interaction conditions but cannot isolate spatial location from tapping-hand
or impact-mechanics effects. Phase 2C therefore used the same right index
finger for both zones in a separately collected session. Analysis reports
contain features, metadata, evaluation results, and limitations—but no
waveform arrays. The source recordings remain local and nothing is uploaded
automatically.

## Freeze and externally evaluate the baseline (Phase 2C)

Phase 2C separates model creation from cross-session testing:

```text
existing development session
  -> fit the fixed midpoint rule using every accepted development sample
  -> write and review a frozen baseline artifact
  -> checkpoint that artifact before collecting external data
  -> collect an untouched same-hand LEFT/RIGHT session
  -> evaluate every external sample without refitting
```

This is stronger than repeatedly tuning on the same dataset because the primary
feature, threshold rule, learned direction, and tie rule are fixed before the
external examples exist. The historical Phase 2B chronological holdout remains
a separate within-session result. Once the feature and model family were fixed,
all accepted samples in the existing development session became development
data for the frozen baseline.

The reviewed baseline can be reproduced from its source development session
with:

```powershell
python -m desksense --freeze-baseline datasets\<DEVELOPMENT-SESSION-ID> `
  --interaction-context hand-location-confounded `
  --save-baseline baselines\lenovo-left-right-v1.json
```

The command is entirely offline, validates the dataset, extracts the existing
`peak_ratio_db_ch2_minus_ch1` feature, computes the LEFT and RIGHT means from
all accepted development samples, and freezes their midpoint and learned
direction. It never reuses the old Phase 2B 1-15 training subset. The baseline
JSON contains compact derived metadata and no waveform arrays, refuses
overwrite, and is intended to be eligible for Git review and tracking;
`baselines/` is deliberately not ignored.

The artifact also records every training sample ID and primary-feature value,
the source endpoint/configuration, and a deterministic SHA-256 fingerprint. The
fingerprint covers the exact `session.json` and `manifest.jsonl` bytes followed
by each manifest-referenced accepted NPZ in logical collection order, with
explicit file names and length framing. Rejected-attempt metadata is not model
input and is excluded from that fingerprint. Source dataset files are never
rewritten.

Evaluate a different session using the already frozen baseline with:

```powershell
python -m desksense `
  --evaluate-frozen-baseline datasets\<EXTERNAL-SESSION-ID> `
  --baseline baselines\lenovo-left-right-v1.json `
  --interaction-context same-hand `
  --save-report reports\phase2c-external.json
```

External evaluation loads the stored feature definition, threshold, direction,
and tie rule unchanged and classifies every accepted external sample. It does
not fit class means, move the threshold, relearn direction, normalize from
external labels, run model selection, or use external samples in any training
step. It rejects the source session itself, an exact full-dataset fingerprint
match, and incompatible endpoint/channel/window configurations. Baseline and
external-session timestamps are retained as provenance and the report states
whether their nominal order matches the planned precommitment workflow, but
clock metadata is not treated as trusted proof or a brittle hard-failure rule.
The baseline should still be reviewed and checkpointed before external data is
collected.

The terminal and JSON outputs include explicit confusion counts, per-class and
per-sample results, threshold margins, and a two-sided 95% Wilson interval for
finite-sample accuracy. Margins are threshold distances, not probabilities,
and the Wilson interval is not a guarantee of future performance. Generated
external reports remain under the Git-ignored `reports/` directory and contain
no waveform arrays.

The external context controls the previous hand/location confound more cleanly
because the same hand/finger taps both locations. Any such result remains
evidence from its documented users, laptops, desks, and sessions—not automatic
cross-device or final DeskSense accuracy.

## First frozen cross-session validation

The frozen baseline was committed in Git checkpoint `9bea99f` before the
external session was collected. The new session used the same right index
finger for every LEFT and RIGHT tap, while keeping the same Lenovo laptop,
wooden desk/setup, WDM-KS endpoint, 48 kHz two-channel configuration, and
200 ms tap windows.

The unchanged frozen baseline classified **39/40 taps correctly (97.5%)**:

| Actual class | Predicted LEFT | Predicted RIGHT |
| --- | ---: | ---: |
| LEFT | 20 | 0 |
| RIGHT | 1 | 19 |

The two-sided 95% Wilson interval was 87.1183%–99.5573%. No external sample was
used to refit the threshold, relearn direction, select the feature, or fit a
normalization. The single error, RIGHT #11, was preserved as part of the
official result: its +0.050166 dB feature value fell just below the frozen
+0.127932 dB threshold.

This is stronger evidence for location-dependent acoustic information than the
earlier within-session result because the model was committed first, the
collection session was separate, and tapping hand was held constant across
zones. It is still only one user, Lenovo laptop, wooden desk/setup, two zones,
and one external session. It does not establish general 97.5% DeskSense
accuracy or cross-user, cross-device, cross-desk, multi-zone, or product-level
robustness.
