"""推理后端：本地 transformers 或远端 vLLM HTTP 服务。

两者接口一致：recognize(image, prompt, max_tokens) -> str
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

import torch
from PIL import Image

MODEL_ID = "SoMarkAI/OXR-1.0"


class LocalBackend:
    """在本进程内加载模型（transformers）。"""

    name = "local"

    def __init__(self, model_id: str = MODEL_ID):
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise SystemExit("CUDA 不可用：请在 systemd 服务里运行")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="cuda:0", low_cpu_mem_usage=True,
        )
        self.model.eval()

    @staticmethod
    def _to_cuda(x):
        if x is None:
            return None
        if isinstance(x, (tuple, list)):
            return torch.cat(list(x), dim=0).to("cuda")
        return x.to("cuda")

    def recognize(self, image: Image.Image, prompt: str, max_tokens: int = 2048) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(messages)
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids.to("cuda"),
                pixel_values=self._to_cuda(pixel_values),
                grid_thws=self._to_cuda(grid_thws),
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        return self.model.text_tokenizer.decode(out[0], skip_special_tokens=True).strip()


class VLLMBackend:
    """调用已启动的 vLLM OpenAI 兼容服务。"""

    name = "vllm"

    def __init__(self, base_url: str = "http://127.0.0.1:8001/v1",
                 model: str = "oxr", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _encode(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def recognize(self, image: Image.Image, prompt: str, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64," + self._encode(image)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.URLError as e:
            raise SystemExit(f"vLLM 服务不可用（{self.base_url}）：{e}") from e
        return data["choices"][0]["message"]["content"].strip()


def make_backend(kind: str = "local", base_url: str = "http://127.0.0.1:8001/v1"):
    if kind == "vllm":
        return VLLMBackend(base_url)
    return LocalBackend()
