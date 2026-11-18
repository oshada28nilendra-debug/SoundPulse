# SoundPulse

SoundPulse learns the sound of normal machine operation from WAV recordings. Given a new recording, it scores how unusual the clip is and shows which time windows contributed to the score.

**Status:** CPU research prototype, validated on 17 November 2026.

[Detailed setup and reproduction commands](02-machine-health-sound/README.md) · [Validation report](02-machine-health-sound/VALIDATION.md) · [License](02-machine-health-sound/LICENSE)

## What it does

- Fits a model on normal machine audio and calibrates the alert threshold on a separate set of normal recordings.
- Scores mono or stereo WAV clips and exports a spectrogram and a timeline of anomaly scores.
- Tests how gain changes, added noise, and unfamiliar recording conditions affect alerts.

## Quick start

Use **Python 3.10+**. From the root of a local clone:

```bash
cd 02-machine-health-sound
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux or macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -e .
sound-health demo --output outputs/my-demo
```

Open `outputs/my-demo/evaluation/report.html` after the command finishes. The demo generates synthetic motor sounds and anomalies. Choose a fresh output directory for each run. The model runs locally on a CPU after dependencies are installed.

## Use your own data

Put good recordings from one machine and operating regime into separate training and calibration folders. Use independent recordings where possible. The default threshold setting requires at least nine calibration clips.

```bash
sound-health fit --normal-dir data/train --calibration-dir data/calibration --model outputs/model.npz
sound-health predict --model outputs/model.npz --audio data/query.wav --output outputs/query
```

Prediction writes `prediction.json` and `spectrogram_and_scores.png`.

## How it works

The pipeline resamples WAV clips to 16 kHz and extracts log-mel spectrogram statistics from overlapping windows. It compares each window with its five closest normal examples and averages the highest-scoring windows to produce a clip score.

## Measured results

The [DCASE 2025 fan experiment](https://zenodo.org/records/15097779) used 160 normal source recordings for training, 40 separate normal source recordings for calibration, and 200 test clips. The table shows the original recordings scored with absolute-level features.

| Recording domain | AUROC | False alarms among normal clips | Anomalies detected |
| --- | ---: | ---: | ---: |
| Source | 0.5408 | 8/50 | 10/50 |
| Target | 0.6216 | 50/50 | 50/50 |
| Combined | 0.5429 | 58/100 | 60/100 |

The source-calibrated threshold flagged every target clip. Level normalization stabilized scores under a simple gain change, while unfamiliar recording conditions still caused excessive false alarms. Results cover the fan subset under a custom protocol. [Full noise tests and limitations](02-machine-health-sound/VALIDATION.md).

## Repository guide

| Path | Contents |
| --- | --- |
| [`soundhealth/`](02-machine-health-sound/soundhealth) | Model, CLI, and evaluation code |
| [`scripts/`](02-machine-health-sound/scripts) | Dataset preparation |
| [`tests/`](02-machine-health-sound/tests) | Behavioral tests |
| [`reports/`](02-machine-health-sound/reports) | Measured results, per-sample scores, and dataset split metadata |

From the project folder, run `python -m unittest discover -s tests -v`. All **12 tests** passed in the recorded validation.
