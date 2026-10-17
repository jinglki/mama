from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def preprocess(path: str | Path, config: dict) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    degrees = float(config.get("rotation_degrees", 0))
    if degrees:
        image = image.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=True)
    size = (int(config["width"]), int(config["height"]))
    image = ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
    sharpness = float(config.get("sharpness", 1.0))
    if sharpness <= 0:
        raise ValueError("preprocess.sharpness must be positive")
    return ImageEnhance.Sharpness(image).enhance(sharpness)


def image_paths(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
