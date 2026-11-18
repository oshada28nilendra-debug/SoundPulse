from pathlib import Path
from dataclasses import asdict
import csv
import hashlib
import html
import importlib.metadata
import json
import platform
import time
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from .core import Config,SoundMonitor,read_audio
from .data import read_manifest

CONDITIONS = ["original","gain_0.5","added_white_noise_10dB","added_white_noise_0dB"]


def perturb(wave,condition,identity,seed=17):
    if condition == "original":
        return wave.copy()
    if condition == "gain_0.5":
        return wave*.5
    if condition not in CONDITIONS:
        raise ValueError("unknown stress-test condition")
    snr = 10 if condition.endswith("10dB") else 0
    number = int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8],"little")
    noise = np.random.default_rng(number).normal(size=len(wave))
    noise /= max(np.sqrt(np.mean(noise**2)),1e-12)
    rms = np.sqrt(np.mean(wave.astype(np.float64)**2))
    # SNR is relative to the complete observed waveform, including existing noise.
    # This adds simulated noise; it is not the dataset's physical machine-only SNR.
    return (wave + noise*rms/(10**(snr/20))).astype(np.float32)


def metrics(rows,threshold):
    labels=np.array([r["label"] for r in rows]);scores=np.array([r["score"] for r in rows])
    normal,anomaly=labels==0,labels==1
    both=len(np.unique(labels))==2
    flags=scores>threshold
    return {"clips":len(rows),"normal_clips":int(normal.sum()),"anomalous_clips":int(anomaly.sum()),
            "auroc":float(roc_auc_score(labels,scores)) if both else None,
            "standardized_partial_auroc_fpr_0.1":float(roc_auc_score(labels,scores,max_fpr=.1)) if both else None,
            "average_precision":float(average_precision_score(labels,scores)) if both else None,
            "false_alarm_rate":float(flags[normal].mean()) if normal.any() else None,
            "recall":float(flags[anomaly].mean()) if anomaly.any() else None,
            "false_alarms":int(flags[normal].sum()),"detected_anomalies":int(flags[anomaly].sum())}


def evaluate(manifest:Path,output:Path,seed=17):
    payload,train,cal,test=read_manifest(manifest)
    output.mkdir(parents=True,exist_ok=True)
    report={"dataset_kind":payload["kind"],"description":payload.get("description"),
            "seed":seed,"training_count":len(train),"calibration_count":len(cal),"test_count":len(test),
            "notes":["Independent spectrogram-statistics kNN baseline, not an official DCASE system",
                     "Only normal training recordings fit the model; separate normal clips calibrate thresholds",
                     "Original-condition calibration threshold is fixed across all test conditions",
                     "Added noise is synthetic white noise relative to the observed audio, not a new physical factory recording",
                     "Source/target labels are used for evaluation only, not to choose separate thresholds",
                     "No pretrained audio embeddings or neural-network training"],
            "environment":{"python":platform.python_version(),"platform":platform.platform(),
                           **{p:importlib.metadata.version(p) for p in ["numpy","scipy","scikit-learn","matplotlib"]}},
            "results":[],"model_configuration":{},"fit_seconds":{}}
    predictions=[]
    cached=[read_audio(r["audio"]) for r in test]
    for normalize in [False,True]:
        mode="rms_normalized" if normalize else "absolute_level"
        model=SoundMonitor(Config(level_normalize=normalize,seed=seed))
        t0=time.perf_counter()
        model.fit(train).calibrate(cal)
        report["fit_seconds"][mode]=time.perf_counter()-t0
        report["model_configuration"][mode]=asdict(model.config)
        model.save(output/f"model_{mode}.npz")
        for condition in CONDITIONS:
            rows=[];durations=[]
            for record,wave in zip(test,cached):
                transformed=perturb(wave,condition,record["audio"].name,seed)
                t0=time.perf_counter();result=model.predict(transformed);durations.append(time.perf_counter()-t0)
                row={"mode":mode,"condition":condition,"audio":record["audio"].name,
                     "domain":record["domain"],"label":record["label"],"score":result["score"],
                     "threshold":model.threshold,"anomalous":result["anomalous"]}
                rows.append(row);predictions.append(row)
            for domain in ["all","source","target"]:
                selection=[r for r in rows if domain=="all" or r["domain"]==domain]
                if not selection:continue
                entry={"mode":mode,"condition":condition,"domain":domain,"threshold":model.threshold,
                       "median_inference_seconds":float(np.median(durations)),**metrics(selection,model.threshold)}
                report["results"].append(entry)
                print(json.dumps(entry),flush=True)
    (output/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    with (output/"predictions.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    render_report(report,output)
    return report


def render_report(report,output):
    columns=["mode","condition","domain","auroc","false_alarm_rate","recall","clips"]
    rows="".join("<tr>"+"".join("<td>"+html.escape(str(round(r[k],4) if isinstance(r[k],float) else r[k]))+"</td>" for k in columns)+"</tr>" for r in report["results"])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Machine Health Sound — Validation</title><style>body{max-width:1200px;margin:32px auto;padding:24px;font:16px/1.6 system-ui;background:#0d1725;color:#e7f0ff}h1{font-size:34px}td,th{padding:10px;border:1px solid #41526c;text-align:left}table{border-collapse:collapse;font-size:13px}.scroll{overflow:auto}.notice{padding:16px;background:#203047;border-left:4px solid #9ecaff}a{color:#9ecaff}</style><h1>Machine Health Monitoring Through Sound</h1>'''
    page+='<p class="notice">Dataset: <strong>'+html.escape(report["dataset_kind"])+f'</strong>. {report["training_count"]} normal training clips, {report["calibration_count"]} calibration clips, {report["test_count"]} test clips.</p>'
    page+='<p>'+html.escape(str(report["description"]))+'</p><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in report["notes"])+"</ul>"
    page+='<div class="scroll"><table><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'</tr>'+rows+'</table></div><p>Scores measure departure from normal acoustic windows; they do not diagnose a specific mechanical fault. Results on one machine type are not an official challenge aggregate.</p></html>'
    (output/"report.html").write_text(page)


def plot_prediction(result,path:Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,6),layout="constrained")
    axes[0].imshow(result["log_mel"],origin="lower",aspect="auto",cmap="magma",
                   extent=[0,result["duration_seconds"],0,result["log_mel"].shape[0]])
    axes[0].set(title="Observed sound: log-mel spectrogram",xlabel="Seconds",ylabel="Mel band")
    axes[1].plot(result["window_times_seconds"],result["window_scores"],color="#126c85")
    axes[1].set(title="Window anomaly distances (not fault probabilities)",xlabel="Seconds",ylabel="kNN distance")
    # Clip-level calibration applies to an aggregation; never draw it as a
    # purported frame-level detection threshold.
    fig.suptitle(f"Clip score: {result['score']:.3f} | Clip threshold: {result['threshold']}",fontsize=11)
    fig.savefig(path,dpi=140);plt.close(fig)
