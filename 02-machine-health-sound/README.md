# 02 — Machine Health Monitoring Through Sound

A runnable, local CPU research prototype for detecting changes in a machine's
operating sound from **normal-only training recordings**. It includes audio
preprocessing, log-mel spectrogram features, an anomaly-scoring model, a CLI,
time-score visualizations, tests, and a reproducible noise/domain sensitivity study.

Implemented and validated on **17 November 2026**.

## Quick start

Python 3.10+ is required; validation used Python 3.12.14 on Linux CPU. From this folder:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -e .
sound-health --help
sound-health demo --output outputs/my-demo
```

Open `outputs/my-demo/evaluation/report.html`. The demo creates **synthetic**
motor-like sounds and deliberately inserts artificial anomalies. It is a smoke
test of the implementation, not a real-machine benchmark. Use a new output
directory for each demo run; existing generated data are not overwritten.

No neural-network weights, API keys, paid services, or GPU are required. All
inference is local after installing dependencies. This version deliberately uses
a small spectrogram/statistical machine-learning baseline rather than a large
pretrained audio model, to make CPU installation and experiments reproducible.
No pretrained embeddings or deep-network training are claimed.

For the exact directly used validation versions:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
```

## Train on a normal machine and inspect a recording

Place independent, defect-free WAV recordings of one machine/operating regime in
`data/my-machine/train`, and **different** defect-free recordings in
`data/my-machine/calibration`. Do not put faulty recordings in either directory.
Use at least 9 calibration recordings for the default alpha=.1; 40 were used in
the real-data experiment. Prefer distinct recording sessions over adjacent crops
from the same recording.

```bash
sound-health fit --normal-dir data/my-machine/train \
  --calibration-dir data/my-machine/calibration --model outputs/my-machine.npz
sound-health predict --model outputs/my-machine.npz \
  --audio data/my-machine/query.wav --output outputs/query
```

Prediction writes:

- `prediction.json`: clip score, normal-calibrated threshold, anomaly decision,
  waveform RMS, clipping fraction, and window times/scores.
- `spectrogram_and_scores.png`: observed log-mel spectrum and anomaly-distance
  trace. The clip threshold is not misrepresented as a frame-level threshold.

The `.npz` model is portable and loaded with `allow_pickle=False`. It contains
normal acoustic-window features, scaling parameters, configuration, and calibration
metadata. It requires no source WAV files at inference time. A large distance
means unfamiliar sound, not a specific diagnosed mechanical fault.

For gain-invariant features, add `--level-normalize` to `fit`. Evaluate it on
held-out conditions; RMS normalization can remove useful changes in loudness.

## Model pipeline

1. Read mono or stereo PCM/float WAV; correctly scale integer audio before stereo
   averaging, and resample to 16 kHz using anti-aliased polyphase resampling.
2. Reject nonfinite samples, clips shorter than one second or longer than 30 seconds,
   and silence/near-silence. Silence is an input error, never a healthy prediction.
3. Compute a Hann-window STFT (1,024 samples, 256-sample hop). Average power into
   64 triangular HTK-mel bands with unit-sum filters; convert power to dB with a
   fixed floor. This is a documented custom feature recipe, not a bit-identical
   reproduction of the official DCASE front end.
4. Form overlapping approximately one-second contexts. For each mel band, extract
   temporal mean, standard deviation, 90th percentile, and mean absolute temporal
   difference: **256 features per context**.
5. Learn feature means/scales from normal training contexts. Standard deviations
   have a one-dB numerical floor. Keep up to 4,096 normal windows, sampled with a
   fixed seed if necessary. No anomalous labels are used to fit the model.
6. Score each test window by the mean Euclidean distance to its five closest normal
   windows. Aggregate the highest 10% of window scores into a clip anomaly score.
7. Calibrate a fixed clip threshold using a disjoint normal set: order statistic
   `ceil((n+1)*(1-alpha))`. With n=40 and alpha=.1, use the 37th sorted score.

This is a normal-memory k-nearest-neighbor model, **not an autoencoder**. No
gradient-based training was performed. It can miss subtle temporal faults because
summary statistics discard detailed event order.

The calibration rule depends on comparable/exchangeable normal recordings. It
does **not** guarantee a 10% deployed false-alarm rate under a new microphone,
different background noise, changed load, or other distribution shifts. Scores
and traces are distances, not probabilities or calibrated fault severities.

## Reproduce the real-data experiment

Dataset: **DCASE 2025 Task 2 development set, fan**, official record:
https://zenodo.org/records/15097779

Review the record's **CC BY-NC-SA 4.0** terms. Download `dev_fan.zip` from the
official record (about 356 MB). The preparation script verifies its official MD5
`61b6aae6d5ebb8cb937faa86741fbf47`, selects the declared subset, and records
per-file checksums. Raw real recordings and real-data models are not included
in the delivery archive.

```bash
python scripts/prepare_dcase.py --archive /path/to/dev_fan.zip \
  --output data/dcase-fan --train-count 160 --calibration-count 40 --seed 17
sound-health evaluate --manifest data/dcase-fan/manifest.json \
  --output outputs/dcase-fan --seed 17
```

The experiment uses 160 source-domain normal training clips, 40 separate source
normal calibration clips, and **all 200 fan development-test clips**: 50 normal
and 50 anomalous clips in each of source and target domains. Target normal training
clips and supplemental recordings are deliberately excluded to expose an unseen-
condition failure mode. This is a **custom source-only protocol**, not an official
challenge submission or a result averaged over seven machine types.

The seeded training/calibration selection is stored in `reports/dcase_manifest.json`.
The preparer keeps original official train/test boundaries. The manifest loader
rejects exact decoded/resampled recording duplication within or across splits.
It cannot identify near-duplicate crops or establish independent physical machines;
no cross-machine generalization claim is made.

## Noise and gain comparison

The evaluator fits two separately calibrated models: absolute-level features and
per-clip RMS-normalized features. It evaluates each under:

- Original observed recordings.
- A 0.5 waveform gain multiplier.
- Added white Gaussian noise at 10 dB SNR relative to the **observed whole waveform**.
- Added white Gaussian noise at 0 dB relative to that waveform.

Noise is deterministic per filename/seed and identical between models. It is
added in floating point without clipping, so the stated added-noise SNR is
maintained. These are digital stress tests, not new factory recordings. Original
recordings already contain environmental noise; these labels are not estimates
of the dataset's original machine-only SNR.

Thresholds remain those calibrated on original normal recordings. Source/target
labels are used only for reporting, never to pick separate thresholds. Test labels
never fit models, thresholds, or preprocessing parameters. The script reports
every prespecified variant, including failures.

Reports include AUROC, average precision, scikit-learn's **standardized partial
AUROC at maximum FPR .1**, false-alarm counts/rates, recall, and CPU inference
time. The partial AUROC is not an unstandardized area or the official DCASE
aggregate score. See `VALIDATION.md` for actual results and interpretation.

## Tests and example inference

```bash
python -m unittest discover -s tests -v
sound-health predict --model outputs/my-demo/evaluation/model_absolute_level.npz \
  --audio outputs/my-demo/data/test/020.wav --output outputs/bundled-example
```

The tests use synthetic signals, not labeled industrial test recordings. They
cover integer-stereo amplitude handling, resampling frequency preservation,
gain invariance, exact added-noise level, serialization, normal-only thresholding,
duplicate/split leakage rejection, silence and invalid-input rejection, metric
accounting, feature dimensions, and detection of an inserted synthetic defect.

## Research grounding and gap

- **DCASE 2025 Task 2** frames normal-only acoustic anomaly detection under
  changing operating/acoustic conditions; official baselines include log-mel
  autoencoder and selective Mahalanobis approaches. This project uses a different
  simple kNN backend. https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring
- **Nishida et al., DCASE 2026 Task 2 overview** makes background-noise robustness
  explicit and introduces near/far microphone recordings. This prototype instead
  uses mono DCASE 2025 audio and controlled noise additions, so it does not
  implement or evaluate the 2026 two-microphone setup. https://arxiv.org/abs/2606.01578
- **DCASE 2025 development dataset**, Nishida et al., published 1 April 2025;
  source/target metadata and normal/anomalous labels. https://doi.org/10.5281/zenodo.15097779
- **Training-Free Model Selection and Domain-Aware Score Calibration for First-Shot
  Anomalous Sound Detection**, 2026, motivates careful treatment of calibration
  under domain shifts; its method is not reproduced here. https://arxiv.org/abs/2607.04526

The prototype's question is whether gain normalization stabilizes fixed-threshold
inspection and whether that benefit survives background noise or real source/target
shift. The experiment tests this hypothesis; it does not assume robustness.

## Limits and next research step

- The source-only fan experiment is narrow and challenging. Read false alarms
  alongside recall: flagging every target recording is not useful detection.
- Normal training data may not represent all healthy speeds, loads, microphones,
  or environments. Train one model per supported machine/regime and document it.
- RMS normalization removes overall level changes but does not separate machine
  sounds from environmental sounds. Loudness can itself be a fault signal.
- This is clip-based offline inference, not continuous streaming, remaining-life
  prediction, root-cause diagnosis, or a verified preventive-maintenance system.
- No pretrained audio embedding, deep autoencoder, multiple-machine study,
  multi-seed confidence interval, or official challenge leaderboard reproduction
  was completed. The bundled synthetic model is only for demonstration.
- A useful next experiment is to compare pretrained audio embeddings on these
  identical frozen splits and add known-normal target-condition recordings using
  a declared adaptation protocol. Do not tune on anomaly test labels.


## Project files

The CLI generates synthetic inputs, fitted demo models, and visual reports locally.
