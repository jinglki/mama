from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    required = ("model", "preprocess", "enhancement", "inference", "evaluation")
    for key in required:
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f"Missing config section: {key}")
    model = cfg["model"]
    if model.get("type") not in {"qwen35", "llava", "custom"}:
        raise ValueError("model.type must be qwen35, llava or custom")
    if not model.get("path"):
        raise ValueError("model.path is required")
    strengths = cfg["enhancement"].get("strengths", {})
    if set(strengths) != {"Low", "Med", "High"} or any(
        not isinstance(v, (int, float)) or not 0 < v <= 1 for v in strengths.values()
    ):
        raise ValueError("enhancement.strengths must define Low, Med, High in (0, 1]")
    for key in ("width", "height"):
        if int(cfg["preprocess"].get(key, 0)) <= 0:
            raise ValueError(f"preprocess.{key} must be positive")
    return cfg


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path
