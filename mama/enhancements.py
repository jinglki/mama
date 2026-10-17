"""Eight enhancement slots; each returns a full-strength RGB image."""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import ROOT, project_path

METHODS = ("gamma", "dcp", "zero_dce", "retinex", "clahe", "white_balance", "waternet", "reserved_identity")


def _u8(value):
    return np.clip(value, 0, 255).astype(np.uint8)


def _gamma(rgb):
    return _u8(255 * np.power(rgb.astype(np.float32) / 255, 0.65))


def _dcp(rgb):
    image = rgb.astype(np.float32) / 255
    dark = cv2.erode(np.min(image, axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    indices = np.argpartition(dark.ravel(), -max(1, dark.size // 100))[-max(1, dark.size // 100):]
    atmosphere = image.reshape(-1, 3)[indices].mean(axis=0).clip(0.05, 1)
    normalized = image / atmosphere
    transmission = 1 - 0.90 * cv2.erode(np.min(normalized, axis=2), np.ones((15, 15), np.uint8))
    transmission = cv2.bilateralFilter(transmission.astype(np.float32), 9, 0.1, 12).clip(0.15, 1)
    restored = (image - atmosphere) / transmission[..., None] + atmosphere
    return _u8(restored * 255)


def _retinex(rgb):
    image = rgb.astype(np.float32) + 1
    logs = [np.log(image) - np.log(cv2.GaussianBlur(image, (0, 0), sigma) + 1) for sigma in (15, 80, 250)]
    result = sum(logs) / len(logs)
    # Shared percentiles preserve channel balance.
    low, high = np.percentile(result, [1, 99])
    return _u8((result - low) * 255 / max(high - low, 1e-6))


def _clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _white_balance(rgb):
    image = rgb.astype(np.float32)
    means = np.mean(image, axis=(0, 1)).clip(1)
    balanced = image * (np.mean(means) / means)
    return _u8(balanced)


def _waternet(rgb, config):
    import torch

    vendor = ROOT / "enhance" / "waternet"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from waternet.net import WaterNet

    weights = project_path(config["waternet_weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"WaterNet weights missing: {weights}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaterNet().to(device).eval()
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    # Use robust local transforms; the vendor white balance divides by zero on black images.
    wb, gc, ce = _white_balance(rgb), _gamma(rgb), _clahe(rgb)
    tensors = [torch.from_numpy(x.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255 for x in (rgb, wb, ce, gc)]
    with torch.inference_mode():
        output = model(*tensors)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    del model
    return _u8(output * 255)


def _zero_dce(rgb, config):
    """Per-image zero-reference curve optimization; no external training labels or weights."""
    import torch
    import torch.nn.functional as F

    class CurveNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.Conv2d(16, 16, 3, padding=1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Conv2d(32, 3, 3, padding=1),
            ])

        def forward(self, x):
            a = F.relu(self.layers[0](x))
            b = F.relu(self.layers[1](a))
            c = F.relu(self.layers[2](torch.cat((a, b), 1)))
            d = F.relu(self.layers[3](torch.cat((b, c), 1)))
            e = F.relu(self.layers[4](torch.cat((c, d), 1)))
            f = F.relu(self.layers[5](torch.cat((d, e), 1)))
            return torch.tanh(self.layers[6](torch.cat((e, f), 1)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Estimate a smooth curve map directly, without supervised target images.
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    height, width = image.shape[-2:]
    small = F.interpolate(image, size=(min(height, 256), min(width, 256)), mode="bilinear", align_corners=False)
    network = CurveNet().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    steps = int(config.get("zero_dce_steps", 20))
    if steps < 1:
        raise ValueError("zero_dce_steps must be positive")
    for _ in range(steps):
        optimizer.zero_grad()
        alpha = network(small)
        enhanced = small
        for _ in range(8):
            enhanced = enhanced + alpha * (enhanced * enhanced - enhanced)
        pool = F.avg_pool2d(enhanced.mean(1, keepdim=True), 16, stride=16)
        exposure = (pool - 0.6).abs().mean()
        color = (enhanced.mean((2, 3)) - enhanced.mean((1, 2, 3), keepdim=False)[:, None]).square().mean()
        smooth = (alpha[:, :, 1:] - alpha[:, :, :-1]).abs().mean() + (alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).abs().mean()
        (exposure + 0.1 * color + 0.05 * smooth).backward()
        optimizer.step()
    with torch.no_grad():
        alpha = F.interpolate(network(small), size=(height, width), mode="bilinear", align_corners=False)
        enhanced = image
        for _ in range(8):
            enhanced = enhanced + alpha * (enhanced * enhanced - enhanced)
        output = enhanced[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return _u8(output * 255)


def enhance(image: Image.Image, method: str, config: dict) -> Image.Image:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    rgb = np.asarray(image.convert("RGB"))
    functions = {"gamma": _gamma, "dcp": _dcp, "retinex": _retinex, "clahe": _clahe,
                 "white_balance": _white_balance, "reserved_identity": lambda x: x.copy()}
    if method == "waternet":
        result = _waternet(rgb, config)
    elif method == "zero_dce":
        result = _zero_dce(rgb, config)
    else:
        result = functions[method](rgb)
    return Image.fromarray(result, "RGB")
