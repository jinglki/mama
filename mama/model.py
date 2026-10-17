"""Single-model Hugging Face multimodal inference."""
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .config import project_path

SYSTEM_PROMPT = """你是严谨的图像物体识别器。只根据提供的图像判断用户指定的目标物体；不要依据常识猜测被遮挡或看不清的物体。
每张图独立判断，按输入顺序输出。相同物体的可分辨实例分别计数；部分遮挡但可识别的实例计入数量。不能确定时回答不存在，并在位置写空数组。
只输出一个 JSON 对象，不要 Markdown、解释或额外键。格式严格为：
{"images":[{"image_index":0,"present":true,"count":2,"locations":["左上","中央"]}]}
images 长度必须等于输入图片数，image_index 从 0 连续递增。present 为布尔值，count 为非负整数；present 当且仅当 count > 0。locations 每个物体一项，长度必须等于 count；使用简短的相对方位，如左上、右下、中央。若没有目标物体，count=0 且 locations=[]。"""


class VisionModel:
    def __init__(self, config: dict):
        kind = config["type"]
        self.kind = kind
        path = project_path(config["path"])
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory missing: {path}. Download weights first.")
        requested = config.get("device", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; set model.device: cpu for a smoke test")
        self.device = requested
        dtype_name = config.get("dtype", "bfloat16")
        if dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError("model.dtype must be bfloat16, float16 or float32")
        dtype = getattr(torch, dtype_name)
        if requested == "cpu" and dtype != torch.float32:
            dtype = torch.float32
        trust = bool(config.get("trust_remote_code", False))
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust)
        if kind == "qwen35":
            from transformers import Qwen3_5ForConditionalGeneration as model_class
        elif kind == "llava":
            from transformers import LlavaForConditionalGeneration as model_class
        else:
            model_class = AutoModelForImageTextToText
        print(requested)
        self.model = model_class.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=trust, low_cpu_mem_usage=True
        ).to(requested).eval()
        self.dtype = dtype
        self.max_new_tokens = int(config.get("max_new_tokens", 256))
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

    def predict(self, paths: list[str], question: str, target: str) -> str:
        if not paths:
            raise ValueError("At least one image is required")
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": f"目标物体：{target}\n用户问题：{question}\n请按指定 JSON 格式回答。"})
        # LLaVA 1.5 chat templates generally have no separate system role.
        if self.kind == "llava":
            content[-1]["text"] = SYSTEM_PROMPT + "\n用户问题：" + content[-1]["text"]
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        template_kwargs = {"enable_thinking": False} if self.kind == "qwen35" else {}
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", **template_kwargs
        )
        inputs = {key: value.to(self.device, dtype=self.dtype if value.is_floating_point() else None)
                  if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = tokens[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return raw


def parse_answer(raw: str, n_images: int) -> dict:
    candidate = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.I)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not return JSON")
        data = json.loads(candidate[start:end + 1])
    if not isinstance(data, dict) or set(data) != {"images"} or not isinstance(data["images"], list) or len(data["images"]) != n_images:
        raise ValueError("Invalid images array")
    for index, item in enumerate(data["images"]):
        if not isinstance(item, dict) or set(item) != {"image_index", "present", "count", "locations"}:
            raise ValueError(f"Invalid image object at index {index}")
        if type(item["image_index"]) is not int or item["image_index"] != index:
            raise ValueError("image_index must match input order")
        if type(item["present"]) is not bool or type(item["count"]) is not int or item["count"] < 0:
            raise ValueError("Invalid present or count")
        if item["present"] != (item["count"] > 0):
            raise ValueError("present/count mismatch")
        if not isinstance(item["locations"], list) or len(item["locations"]) != item["count"] or any(not isinstance(x, str) for x in item["locations"]):
            raise ValueError("Invalid locations")
    return data
