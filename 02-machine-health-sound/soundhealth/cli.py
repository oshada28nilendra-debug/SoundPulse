import argparse
import json
from pathlib import Path
from .core import Config,SoundMonitor,read_audio
from .data import synthetic,wavs
from .evaluate import evaluate,plot_prediction


def main(argv=None):
    p=argparse.ArgumentParser(description="Normal-only machine sound anomaly scoring on CPU")
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("demo",help="Generate synthetic sounds, fit and run noise stress tests")
    d.add_argument("--output",type=Path,default=Path("outputs/synthetic-demo"))
    d.add_argument("--seed",type=int,default=17)
    f=sub.add_parser("fit",help="Fit normal recordings and calibrate on disjoint normal recordings")
    f.add_argument("--normal-dir",type=Path,required=True)
    f.add_argument("--calibration-dir",type=Path,required=True)
    f.add_argument("--model",type=Path,required=True)
    f.add_argument("--level-normalize",action="store_true")
    f.add_argument("--alpha",type=float,default=.1)
    i=sub.add_parser("predict",help="Score a WAV clip and output time trace and spectrogram")
    i.add_argument("--model",type=Path,required=True)
    i.add_argument("--audio",type=Path,required=True)
    i.add_argument("--output",type=Path,required=True)
    e=sub.add_parser("evaluate",help="Evaluate fixed-threshold noise/domain sensitivity")
    e.add_argument("--manifest",type=Path,required=True)
    e.add_argument("--output",type=Path,required=True)
    e.add_argument("--seed",type=int,default=17)
    a=p.parse_args(argv)
    try:
        if a.command=="demo":
            manifest=synthetic(a.output/"data",a.seed)
            evaluate(manifest,a.output/"evaluation",a.seed)
        elif a.command=="fit":
            model=SoundMonitor(Config(level_normalize=a.level_normalize))
            model.fit(wavs(a.normal_dir)).calibrate(wavs(a.calibration_dir),a.alpha)
            model.save(a.model)
            print(json.dumps({"model":str(a.model),"threshold":model.threshold,
                              "training_count":model.metadata["training_count"]}))
        elif a.command=="predict":
            model=SoundMonitor.load(a.model)
            result=model.predict(read_audio(a.audio,model.config))
            a.output.mkdir(parents=True,exist_ok=True)
            plot_prediction(result,a.output/"spectrogram_and_scores.png")
            result.pop("log_mel")
            (a.output/"prediction.json").write_text(json.dumps(result,indent=2)+"\n")
            print(json.dumps({k:result[k] for k in ["score","threshold","anomalous","duration_seconds","clipping_fraction"]}))
        elif a.command=="evaluate":
            evaluate(a.manifest,a.output,a.seed)
    except (ValueError,FileNotFoundError) as exc:
        p.error(str(exc))


if __name__=="__main__":
    main()
