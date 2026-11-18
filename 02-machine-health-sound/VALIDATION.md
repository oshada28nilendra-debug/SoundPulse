# Validation — 17 November 2026

## Executed validation

- 12 behavioral tests passed; see the captured test log below.
- Editable installation and the `sound-health` entry point succeeded.
- Synthetic data generation, model fitting, normal-only calibration, noise evaluation, model save/load, and CLI inference completed.
- Real-data evaluation completed on all 200 official DCASE 2025 fan development-test clips using a declared source-only training subset.
- Python compilation of source, preparation script, and tests passed.

## Real-data protocol

160 source normal training clips and 40 disjoint source normal calibration clips, selected using seed 17. The official fan test set contains 50 normal and 50 anomalous source clips, and the same counts in the target domain.

Target normal training recordings and supplemental sounds were excluded. This is not the official full-training challenge protocol or an average across machine types. Exact clip selections/checksums are in `reports/dcase_manifest.json`.

## Measured original-condition results

| Model | Domain | AUROC | False alarms / normal | Detected / anomalous |
|---|---|---:|---:|---:|
| absolute_level | all | 0.5429 | 58/100 (58.0%) | 60/100 (60.0%) |
| absolute_level | source | 0.5408 | 8/50 (16.0%) | 10/50 (20.0%) |
| absolute_level | target | 0.6216 | 50/50 (100.0%) | 50/50 (100.0%) |
| rms_normalized | all | 0.5431 | 58/100 (58.0%) | 60/100 (60.0%) |
| rms_normalized | source | 0.5404 | 8/50 (16.0%) | 10/50 (20.0%) |
| rms_normalized | target | 0.6228 | 50/50 (100.0%) | 50/50 (100.0%) |

## All fixed-threshold stress tests (both domains combined)

| Model | Condition | AUROC | False alarms / 100 normal | Detected / 100 anomalous | Median inference |
|---|---|---:|---:|---:|---:|
| absolute_level | original | 0.5429 | 58 | 60 | 24.4 ms |
| absolute_level | gain_0.5 | 0.5550 | 85 | 85 | 28.4 ms |
| absolute_level | added_white_noise_10dB | 0.5264 | 58 | 60 | 26.1 ms |
| absolute_level | added_white_noise_0dB | 0.4938 | 71 | 73 | 24.9 ms |
| rms_normalized | original | 0.5431 | 58 | 60 | 26.3 ms |
| rms_normalized | gain_0.5 | 0.5431 | 58 | 60 | 29.0 ms |
| rms_normalized | added_white_noise_10dB | 0.5249 | 56 | 60 | 46.0 ms |
| rms_normalized | added_white_noise_0dB | 0.5134 | 33 | 34 | 23.5 ms |

## Interpretation

The original-condition absolute-level model has combined AUROC 0.5429 and detects 60/100 anomalous clips, but falsely flags 58/100 normal clips. It is a runnable baseline, not a reliable machine-health monitor.

The source/target breakdown exposes the main problem: at the fixed source-calibrated threshold, all 50 normal target clips are flagged. Detecting all 50 target anomalies in that situation does not demonstrate usable detection.

RMS normalization makes inference invariant to a simple gain multiplier, as the implementation tests and experiment confirm. It does not repair the real target-domain false alarms, and added-noise conditions remain close to chance discrimination. Lower false alarm counts under strong noise may be accompanied by lower recall, rather than representing robustness.

These results motivate richer representations and a declared known-normal target-adaptation experiment. No variant was selected or tuned using test labels; every prespecified variant is included.

## Synthetic validation

The synthetic experiment uses 32 normal training clips, 20 independent normal calibration clips, and 80 test clips split equally between two artificial domains. Inserted high-frequency rubbing/impacts are deliberately easy. Its results verify pipeline behavior only and must not be substituted for real-machine performance.

## What was and was not trained

Normal feature scaling and a nearest-neighbor memory bank were fitted. No pretrained embeddings, deep network, autoencoder, remaining-life model, or fault-specific diagnosis was trained or claimed.

## Reproducibility and practical limits

- The model and calibration never see test anomaly labels.
- Exact decoded/resampled duplicates across the splits are rejected, but near duplicates and physical-machine independence cannot be established automatically.
- Additional white-noise SNR is measured relative to the full original observed waveform, not the isolated machine sound. Noise is added without clipping.
- Domain labels are used for evaluation only. A single clean-calibration threshold is reused across all stress conditions for each model.
- Reported partial AUROC uses scikit-learn standardization with max_fpr=.1; it is not an official DCASE aggregate score.
- Inference timing includes feature extraction and scoring on CPU, excluding file decoding. Timings depend on hardware and system load.
- One seed and one machine type were evaluated; no confidence interval or cross-machine generalization claim is made.
- Model window timestamps were aligned to STFT window centers after evaluation; this changes only time labels, not features, scores, thresholds, or reported metrics.

## Actual test log

```text
test_calibration_and_training_overlap_rejected (test_soundhealth.SoundHealthTests.test_calibration_and_training_overlap_rejected) ... ok
test_detects_synthetic_defect_signal (test_soundhealth.SoundHealthTests.test_detects_synthetic_defect_signal) ... ok
test_duplicate_content_cannot_cross_manifest_splits (test_soundhealth.SoundHealthTests.test_duplicate_content_cannot_cross_manifest_splits) ... ok
test_feature_trace_dimensions_and_mel_support (test_soundhealth.SoundHealthTests.test_feature_trace_dimensions_and_mel_support) ... ok
test_gain_normalized_features_are_invariant (test_soundhealth.SoundHealthTests.test_gain_normalized_features_are_invariant) ... ok
test_metrics_keep_false_alarms_separate_from_recall (test_soundhealth.SoundHealthTests.test_metrics_keep_false_alarms_separate_from_recall) ... ok
test_noise_level_and_determinism (test_soundhealth.SoundHealthTests.test_noise_level_and_determinism) ... ok
test_nonfinite_audio_rejected (test_soundhealth.SoundHealthTests.test_nonfinite_audio_rejected) ... ok
test_resamples_integer_stereo_without_amplitude_error (test_soundhealth.SoundHealthTests.test_resamples_integer_stereo_without_amplitude_error) ... ok
test_serialization_preserves_score_and_trace (test_soundhealth.SoundHealthTests.test_serialization_preserves_score_and_trace) ... ok
test_silence_and_short_audio_are_not_healthy_predictions (test_soundhealth.SoundHealthTests.test_silence_and_short_audio_are_not_healthy_predictions) ... ok
test_threshold_is_normal_only_order_statistic (test_soundhealth.SoundHealthTests.test_threshold_is_normal_only_order_statistic) ... ok

----------------------------------------------------------------------
Ran 12 tests in 1.341s

OK
```

