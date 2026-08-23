# DeskSense for Windows

DeskSense aims to become a hardware-adaptive acoustic tap input system for
Windows laptops. The long-term goal is to use built-in microphones to detect
desk taps, determine their zones, and map them to programmable Windows actions.
The project is inspired by the MIT-licensed [Holo project](https://github.com/JustinGamer191/Holo).

## Current milestone

Milestone 1 established the microphone and audio-hardware feasibility probe.
It enumerates input devices, identifies the default input and host API, checks
common sample rates, and can measure a short in-memory recording. Milestone 1.5
adds exploratory channel and endpoint characterization. Neither milestone
implements tap detection or classification, localization, machine learning, a
GUI, hotkeys, or action mapping.

No cross-laptop compatibility or tap-classification accuracy is claimed at
this stage. Raw audio is never saved by this diagnostic.

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

If Windows denies microphone access, enable it under **Settings > Privacy &
security > Microphone** and retry.
