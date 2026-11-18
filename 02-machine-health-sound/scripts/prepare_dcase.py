"""Prepare a source-only normal-training subset from the official DCASE 2025 fan archive.

Source: https://zenodo.org/records/15097779 (CC BY-NC-SA 4.0).
No raw dataset recordings are redistributed in the project delivery.
"""
from pathlib import Path,PurePosixPath
import argparse
import hashlib
import json
import zipfile
import numpy as np


def prepare(archive:Path,output:Path,train_count=160,calibration_count=40,seed=17):
    if (output/"manifest.json").exists():
        raise ValueError("destination manifest already exists")
    if train_count<1 or calibration_count<1:
        raise ValueError("training/calibration counts must be positive")
    digest=hashlib.md5(archive.read_bytes()).hexdigest()
    expected="61b6aae6d5ebb8cb937faa86741fbf47"
    if digest!=expected:
        raise ValueError("archive does not match official DCASE 2025 dev_fan.zip checksum")
    output.mkdir(parents=True,exist_ok=True)
    payload={"kind":"real_dcase2025_fan_source_only_subset","seed":seed,
             "description":"Real DCASE 2025 fan recordings; a custom subset of source-domain normal training, all 200 official development-test fan clips; not an official challenge submission",
             "source":"https://zenodo.org/records/15097779","license":"CC BY-NC-SA 4.0",
             "archive_md5":digest,"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
             "training_policy":"source normal only; target normal training and supplemental clips excluded",
             "train":[],"calibration":[],"test":[],"file_sha256":{}}
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if n.endswith(".wav")]
        candidates=sorted(n for n in names if '/train/' in n and '_source_train_normal_' in n)
        if len(candidates)<train_count+calibration_count:
            raise ValueError("not enough source normal training files")
        order=np.random.default_rng(seed).permutation(len(candidates))
        training=[candidates[i] for i in order[:train_count]]
        calibration=[candidates[i] for i in order[train_count:train_count+calibration_count]]
        test=sorted(n for n in names if '/test/' in n)
        for split,selection in [("train",training),("calibration",calibration),("test",test)]:
            (output/split).mkdir(exist_ok=True)
            for member in selection:
                name=PurePosixPath(member).name
                relative=f"{split}/{name}"
                raw=z.read(member)
                (output/relative).write_bytes(raw)
                payload["file_sha256"][relative]=hashlib.sha256(raw).hexdigest()
                if split=="test":
                    if '_normal_' not in name and '_anomaly_' not in name:
                        raise ValueError("unrecognized test label")
                    domain='source' if '_source_' in name else 'target' if '_target_' in name else None
                    if domain is None:raise ValueError("unrecognized domain")
                    payload[split].append({"audio":relative,"label":int('_anomaly_' in name),"domain":domain})
                else:
                    payload[split].append(relative)
    (output/"manifest.json").write_text(json.dumps(payload,indent=2)+"\n")
    return {k:len(payload[k]) for k in ["train","calibration","test"]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-count',type=int,default=160)
    p.add_argument('--calibration-count',type=int,default=40)
    p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    print(json.dumps(prepare(a.archive,a.output,a.train_count,a.calibration_count,a.seed)))
