import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .config import project_path
from .enhancements import METHODS, enhance
from .imaging import preprocess


def read_samples(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    samples = []
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict) or not isinstance(sample.get("images"), list) or not sample["images"]:
                raise ValueError(f"Line {line_number}: images must be a nonempty list")
            if not isinstance(sample.get("question"), str) or not sample["question"].strip():
                raise ValueError(f"Line {line_number}: question required")
            if not isinstance(sample.get("target"), str) or not sample["target"].strip():
                raise ValueError(f"Line {line_number}: target required")
            sample_id = sample.get("id") or str(uuid.uuid4())
            if not isinstance(sample_id, str) or sample_id in ids:
                raise ValueError(f"Line {line_number}: duplicate or invalid id")
            ids.add(sample_id)
            sample["id"] = sample_id
            paths = []
            for image in sample["images"]:
                p = Path(image)
                p = p if p.is_absolute() else path.parent / p
                if not p.is_file():
                    raise FileNotFoundError(f"Line {line_number}: {p}")
                paths.append(str(p.resolve()))
            sample["images"] = paths
            truth = sample.get("truth")
            if truth is not None:
                if not isinstance(truth, list) or len(truth) != len(paths):
                    raise ValueError(f"Line {line_number}: truth length must match images")
                for label in truth:
                    if not isinstance(label, dict) or type(label.get("count")) is not int or label["count"] < 0:
                        raise ValueError(f"Line {line_number}: truth requires nonnegative count")
            samples.append(sample)
    if not samples:
        raise ValueError("Manifest is empty")
    return samples


def prepare(samples: list[dict], cfg: dict, run_id: str) -> list[dict]:
    output_dir = project_path(cfg["enhancement"]["output_dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    workers = int(cfg["enhancement"].get("workers", 4))
    if workers < 1:
        raise ValueError("enhancement.workers must be positive")
    prepared = []
    for sample in samples:
        conditions = {"original": []}
        for method in METHODS:
            for level in cfg["enhancement"]["strengths"]:
                conditions[f"{method}_{level}"] = []
        for source in sample["images"]:
            image_id = uuid.uuid4().hex
            original = preprocess(source, cfg["preprocess"])
            original_path = output_dir / f"{image_id}_original.png"
            original.save(original_path)
            conditions["original"].append(str(original_path))

            def process(method):
                full = enhance(original, method, cfg["enhancement"])
                def save_level(level, strength):
                    path = output_dir / f"{image_id}_{method}_{level}.png"
                    Image.blend(original, full, float(strength)).save(path)
                    return f"{method}_{level}", str(path)

                with ThreadPoolExecutor(max_workers=3) as levels:
                    futures = [levels.submit(save_level, level, strength)
                               for level, strength in cfg["enhancement"]["strengths"].items()]
                    return dict(future.result() for future in futures)

            # Independent OpenCV methods run in parallel. Torch methods run in the
            # main thread so they do not compete for GPU memory.
            classic = [name for name in METHODS if name not in {"zero_dce", "waternet"}]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process, name): name for name in classic}
                for future in as_completed(futures):
                    for condition, path in future.result().items():
                        conditions[condition].append(path)
            for name in ("zero_dce", "waternet"):
                for condition, path in process(name).items():
                    conditions[condition].append(path)
        if any(len(paths) != len(sample["images"]) for paths in conditions.values()):
            raise RuntimeError("Condition image count mismatch")
        prepared.append({"id": sample["id"], "question": sample["question"], "target": sample["target"],
                         "sources": sample["images"], "truth": sample.get("truth"), "conditions": conditions})
    return prepared


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def infer(prepared: list[dict], cfg: dict) -> list[dict]:
    from .model import VisionModel, parse_answer

    model = VisionModel(cfg["model"])
    results = []
    for sample in prepared:
        for condition, paths in sample["conditions"].items():
            if condition == "original" and not cfg["inference"].get("include_original", True):
                continue
            if condition != "original":
                method, separator, level = condition.rpartition("_")
                if not separator or method not in METHODS or level not in cfg["enhancement"]["strengths"]:
                    raise ValueError(f"Invalid condition: {condition}")
            if not paths or any(not Path(path).stem.endswith("_" + condition) for path in paths):
                raise ValueError(f"Condition {condition} contains mismatched images")
            record = {"id": sample["id"], "target": sample["target"], "question": sample["question"],
                      "condition": condition, "images": paths, "truth": sample.get("truth")}
            try:
                record["raw"] = model.predict(paths, sample["question"], sample["target"])
                record["prediction"] = parse_answer(record["raw"], len(paths))
                record["error"] = None
            except (ValueError, RuntimeError) as exc:
                record["prediction"] = None
                record["error"] = str(exc)
            results.append(record)
    return results
