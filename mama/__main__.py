import argparse
import json
import uuid
from .download import download_from_modelscope
from .config import load_config, project_path
from .evaluation import evaluate
from .pipeline import infer, load_jsonl, prepare, read_samples, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Multimodal image enhancement experiment")
    parser.add_argument("command", choices=["download", "prepare", "infer", "evaluate", "all"])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--manifest", help="JSONL dataset, required for prepare/all")
    parser.add_argument("--prepared", help="Prepared JSONL, required for infer")
    parser.add_argument("--predictions", help="Predictions JSONL, required for evaluate")
    parser.add_argument("--run-id", help="Optional run identifier")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "download":
        download_from_modelscope(cfg, parser)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    run_dir = project_path(cfg["inference"]["output_dir"]) / run_id
    if args.command in {"prepare", "all"}:
        if not args.manifest:
            parser.error("--manifest is required")
        samples = read_samples(args.manifest)
        prepared = prepare(samples, cfg, run_id)
        save_jsonl(run_dir / "prepared.jsonl", prepared)
        print(f"Prepared {len(prepared)} samples: {run_dir / 'prepared.jsonl'}")
    if args.command == "infer":
        if not args.prepared:
            parser.error("--prepared is required")
        prepared = load_jsonl(args.prepared)
    if args.command in {"infer", "all"}:
        predictions = infer(prepared, cfg)
        save_jsonl(run_dir / "predictions.jsonl", predictions)
        print(f"Predictions: {run_dir / 'predictions.jsonl'}")
    if args.command == "evaluate":
        if not args.predictions:
            parser.error("--predictions is required")
        predictions = load_jsonl(args.predictions)
    if args.command in {"evaluate", "all"}:
        report = evaluate(predictions)
        out = project_path(cfg["evaluation"]["output_dir"]) / f"{run_id}_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Metrics: {out}")


if __name__ == "__main__":
    main()
