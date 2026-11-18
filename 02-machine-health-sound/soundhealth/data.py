"""Audio manifests with explicit synthetic/real provenance and split checks."""
from pathlib import Path
import json
import numpy as np
from scipy.io import wavfile
from .core import Config, audio_hash


def wavs(directory: Path):
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() == ".wav")


def synthetic(root: Path, seed=17):
    if (root/"manifest.json").exists():
        raise ValueError("synthetic output already exists; choose a new directory")
    rng = np.random.default_rng(seed)
    sample_rate = 16000
    t = np.arange(4*sample_rate)/sample_rate
    manifest = {"kind": "synthetic_smoke_test", "seed": seed,
                "description": "Procedural motor-like harmonics, synthetic rubbing/impact defects and artificial target domain; not real machine audio",
                "train": [], "calibration": [], "test": []}
    for split,count in [("train",32),("calibration",20),("test",80)]:
        (root/split).mkdir(parents=True,exist_ok=True)
        for i in range(count):
            target = split == "test" and i >= count//2
            bad = split == "test" and i % (count//2) >= count//4
            f = 220 + rng.uniform(-2,2)
            phase = rng.uniform(0,2*np.pi)
            wave = (.16*np.sin(2*np.pi*f*t+phase) + .06*np.sin(2*np.pi*2*f*t+phase)
                    + .025*np.sin(2*np.pi*3*f*t+phase))
            wave *= 1 + .08*np.sin(2*np.pi*3*t)
            wave += rng.normal(0,.004 if not target else .014,len(t))
            if bad:
                envelope = (np.sin(2*np.pi*5*t) > .5).astype(float)
                wave += .07*envelope*np.sin(2*np.pi*1800*t)
                centers = rng.choice(np.arange(1000,len(t)-1000),10,replace=False)
                for center in centers:
                    wave[center:center+80] += .16*np.exp(-np.arange(80)/20)
            if target:
                wave = .6*wave + .025*np.sin(2*np.pi*60*t)
            wave = np.clip(wave,-.98,.98)
            name = f"{split}/{i:03}.wav"
            wavfile.write(root/name,sample_rate,np.int16(wave*32767))
            if split == "test":
                manifest[split].append({"audio":name,"label":int(bad),
                                        "domain":"target" if target else "source"})
            else:
                manifest[split].append(name)
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return root/"manifest.json"


def read_manifest(path: Path, config: Config | None=None):
    payload = json.loads(path.read_text())
    if not all(k in payload for k in ["kind","train","calibration","test"]):
        raise ValueError("manifest requires kind, train, calibration and test")
    resolve = lambda p: Path(p) if Path(p).is_absolute() else path.parent/p
    train,cal = [[resolve(p) for p in payload[k]] for k in ["train","calibration"]]
    test = []
    for row in payload["test"]:
        if row.get("label") not in [0,1] or row.get("domain") not in ["source","target"]:
            raise ValueError("test records need binary labels and source/target domains")
        test.append({**row,"audio":resolve(row["audio"])})
    seen=set()
    for group in [train,cal,[r["audio"] for r in test]]:
        if not group:
            raise ValueError("manifest splits must be nonempty")
        hashes=[audio_hash(p,config) for p in group]
        if len(hashes)!=len(set(hashes)) or seen.intersection(hashes):
            raise ValueError("duplicate recording or decoded-audio split leakage")
        seen.update(hashes)
    return payload,train,cal,test
