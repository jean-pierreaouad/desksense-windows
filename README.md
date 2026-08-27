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
labeled local waveform collection for later offline feasibility work. It does
not implement the final tap detector or classifier, localization, machine
learning, a GUI, hotkeys, or action mapping.

No cross-laptop compatibility or tap-classification accuracy is claimed at
this stage. Diagnostic and characterization commands never save raw audio;
only the separate, explicit dataset-collection command retains waveforms.

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

This collection tooling still requires physical review on the target laptop;
its presence does not mean a valid dataset or localization result exists yet.

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
