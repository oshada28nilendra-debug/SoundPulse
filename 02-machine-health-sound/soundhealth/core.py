"""Log-mel context statistics and an exact normal-window nearest-neighbor model.

No pretrained embeddings or neural fine-tuning are used. All learned parameters
come from normal recordings; a disjoint normal set calibrates the clip threshold.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from scipy.spatial.distance import cdist


@dataclass
class Config:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 64
    window_frames: int = 64
    window_step: int = 32
    neighbors: int = 5
    max_windows: int = 4096
    level_normalize: bool = False
    top_fraction: float = 0.1
    seed: int = 17
    max_seconds: float = 30.0

    def validate(self):
        if self.sample_rate < 8000 or self.sample_rate > 48000:
            raise ValueError("sample_rate must be between 8000 and 48000")
        if not 128 <= self.n_fft <= 4096 or not 1 <= self.hop <= self.n_fft:
            raise ValueError("invalid FFT or hop")
        if not 8 <= self.n_mels <= 128 or self.window_frames < 2 or self.window_step < 1:
            raise ValueError("invalid mel/context configuration")
        if self.neighbors < 1 or self.max_windows < self.neighbors or not 0 < self.top_fraction <= 1:
            raise ValueError("invalid neighbor/memory/aggregation configuration")
        if not 1 <= self.max_seconds <= 300:
            raise ValueError("max_seconds must be in [1,300]")


def read_audio(path: str | Path, config: Config | None = None) -> np.ndarray:
    config = config or Config()
    rate, raw = wavfile.read(path)
    if raw.ndim not in [1, 2] or not raw.size or rate <= 0:
        raise ValueError("expected a nonempty mono/stereo WAV")
    # Integer PCM values must be scaled before stereo downmixing.
    if raw.dtype == np.uint8:
        wave = (raw.astype(np.float64) - 128) / 128
    elif np.issubdtype(raw.dtype, np.signedinteger):
        wave = raw.astype(np.float64) / (float(np.iinfo(raw.dtype).max) + 1)
    elif np.issubdtype(raw.dtype, np.floating):
        wave = raw.astype(np.float64)
    else:
        raise ValueError("unsupported WAV encoding; use PCM or floating-point WAV")
    if not np.isfinite(wave).all() or np.max(np.abs(wave)) > 1.001:
        raise ValueError("WAV must have finite audio samples in [-1,1]")
    if wave.ndim == 2:
        if wave.shape[1] != 2:
            raise ValueError("only mono/stereo WAV is supported")
        wave = wave.mean(axis=1)
    if len(wave) > round(config.max_seconds * rate):
        raise ValueError("clip is too long; split it into shorter WAV recordings")
    if rate != config.sample_rate:
        divisor = math.gcd(int(rate), config.sample_rate)
        wave = resample_poly(wave, config.sample_rate // divisor, int(rate) // divisor)
    return wave.astype(np.float32)


def audio_hash(path: str | Path, config: Config | None = None):
    """Hash decoded/resampled samples so copied recordings cannot cross splits."""
    return hashlib.sha256(read_audio(path, config).tobytes()).hexdigest()


def mel_filterbank(config: Config):
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    mel_to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(20), hz_to_mel(config.sample_rate / 2), config.n_mels + 2))
    frequencies = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    bank = np.zeros((config.n_mels, len(frequencies)), np.float64)
    for i in range(config.n_mels):
        left, center, right = edges[i:i+3]
        bank[i] = np.maximum(0, np.minimum((frequencies-left)/(center-left),
                                            (right-frequencies)/(right-center)))
        bank[i] /= max(bank[i].sum(), 1e-12)
    return bank


def log_mel(wave: np.ndarray, config: Config):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("expected finite mono waveform")
    if len(wave) < config.sample_rate:
        raise ValueError("at least one second of audio is required")
    if len(wave) > config.max_seconds * config.sample_rate:
        raise ValueError("clip exceeds configured maximum duration")
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    if rms < 1e-5:
        raise ValueError("insufficient signal: silence or near-silence")
    if config.level_normalize:
        wave = wave * (0.1 / rms)
    _, times, spectrum = stft(wave, fs=config.sample_rate, nperseg=config.n_fft,
                              noverlap=config.n_fft-config.hop, boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    mel = np.maximum(mel_filterbank(config) @ power, 1e-12)
    return (10 * np.log10(mel)).astype(np.float32), times


def features(wave: np.ndarray, config: Config):
    spec, times = log_mel(wave, config)
    if spec.shape[1] < config.window_frames:
        spec = np.pad(spec, ((0, 0), (0, config.window_frames-spec.shape[1])), mode="edge")
    starts = list(range(0, spec.shape[1]-config.window_frames+1, config.window_step))
    if starts[-1] != spec.shape[1]-config.window_frames:
        starts.append(spec.shape[1]-config.window_frames)
    rows, centers = [], []
    for start in starts:
        context = spec[:, start:start+config.window_frames]
        rows.append(np.concatenate([context.mean(axis=1), context.std(axis=1),
                                    np.quantile(context, .9, axis=1),
                                    np.abs(np.diff(context, axis=1)).mean(axis=1)]))
        center = ((start + (config.window_frames-1)/2) * config.hop + config.n_fft/2) / config.sample_rate
        centers.append(min(center,len(wave)/config.sample_rate))
    return np.asarray(rows, np.float32), np.asarray(centers), spec


class SoundMonitor:
    def __init__(self, config: Config | None = None):
        self.config = config or Config(); self.config.validate()
        self.memory = self.mean = self.scale = self.threshold = None
        self.metadata = {"schema": 1, "method": "logmel_context_stats_exact_knn",
                         "neural_training": False, "pretrained_embeddings": False}

    def fit(self, normal_paths: list[Path]):
        if not normal_paths:
            raise ValueError("normal training recordings are required")
        hashes = [audio_hash(p, self.config) for p in normal_paths]
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate normal training audio")
        matrix = np.concatenate([features(read_audio(p, self.config), self.config)[0] for p in normal_paths])
        self.mean = matrix.mean(axis=0)
        # A one-dB scale floor prevents near-constant features dominating distances.
        self.scale = np.maximum(matrix.std(axis=0), 1.0)
        matrix = (matrix-self.mean)/self.scale
        if len(matrix) > self.config.max_windows:
            matrix = matrix[np.random.default_rng(self.config.seed).choice(
                len(matrix), self.config.max_windows, replace=False)]
        self.memory = matrix.astype(np.float32)
        self.threshold = None
        self.metadata.update({"training_hashes": hashes, "training_count": len(normal_paths),
                              "normal_windows": len(matrix)})
        return self

    def predict(self, wave: np.ndarray):
        if self.memory is None:
            raise ValueError("fit or load a model first")
        wave = np.asarray(wave, dtype=np.float32)
        matrix, centers, spec = features(wave, self.config)
        matrix = (matrix-self.mean)/self.scale
        distances = cdist(matrix, self.memory, metric="sqeuclidean")
        k = min(self.config.neighbors, len(self.memory))
        nearest = np.partition(distances, k-1, axis=1)[:, :k]
        window_scores = np.sqrt(np.maximum(nearest, 0)).mean(axis=1)
        count = max(1, math.ceil(self.config.top_fraction * len(window_scores)))
        score = float(np.partition(window_scores, -count)[-count:].mean())
        return {"score": score, "threshold": self.threshold,
                "anomalous": None if self.threshold is None else bool(score > self.threshold),
                "window_times_seconds": centers.tolist(), "window_scores": window_scores.tolist(),
                "rms": float(np.sqrt(np.mean(wave.astype(np.float64)**2))),
                "clipping_fraction": float((np.abs(wave) >= .999).mean()),
                "duration_seconds": len(wave)/self.config.sample_rate,
                "log_mel": spec}

    def calibrate(self, normal_paths: list[Path], alpha=.1):
        if self.memory is None:
            raise ValueError("fit first")
        if not normal_paths or not 0 < alpha < 1:
            raise ValueError("normal calibration recordings and alpha in (0,1) required")
        hashes = [audio_hash(p, self.config) for p in normal_paths]
        if len(set(hashes)) != len(hashes) or set(hashes)&set(self.metadata["training_hashes"]):
            raise ValueError("calibration audio must be unique and disjoint from training")
        rank = math.ceil((len(normal_paths)+1)*(1-alpha))
        if rank > len(normal_paths):
            raise ValueError("not enough normal calibration clips for requested alpha")
        scores = [self.predict(read_audio(p, self.config))["score"] for p in normal_paths]
        self.threshold = float(np.sort(scores)[rank-1])
        self.metadata.update({"calibration_hashes": hashes, "calibration_count": len(normal_paths),
                              "alpha": alpha, "threshold_rank": rank,
                              "calibration_note": "normal-score order statistic; no guarantee under domain shift"})
        return self

    def save(self, path: Path):
        if self.memory is None:
            raise ValueError("cannot save unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "config": asdict(self.config), "threshold": self.threshold}
        with path.open("wb") as f:
            np.savez_compressed(f, memory=self.memory, mean=self.mean, scale=self.scale,
                                metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as bundle:
            meta = json.loads(str(bundle["metadata"]))
            if meta.get("schema") != 1:
                raise ValueError("unsupported model schema")
            model = cls(Config(**meta["config"]))
            model.memory, model.mean, model.scale = [np.array(bundle[k], np.float32) for k in ["memory", "mean", "scale"]]
        d = model.config.n_mels*4
        if model.memory.ndim != 2 or not len(model.memory) or model.memory.shape[1] != d:
            raise ValueError("invalid normal memory")
        if model.mean.shape != (d,) or model.scale.shape != (d,) or np.any(model.scale <= 0):
            raise ValueError("invalid feature normalization")
        if not all(np.isfinite(a).all() for a in [model.memory, model.mean, model.scale]):
            raise ValueError("nonfinite model arrays")
        model.threshold, model.metadata = meta["threshold"], meta
        if model.threshold is not None and (not np.isfinite(model.threshold) or model.threshold < 0):
            raise ValueError("invalid threshold")
        return model
