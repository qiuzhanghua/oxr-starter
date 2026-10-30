"""加载 OXR-1.0，报告参数量与显存占用（不推理）。

用途：确认模型能在本机 16 GB 显存上加载，作为后续推理脚本的前置检查。

运行（沙箱内看不到 GPU，必须走 systemd 服务）：
    systemd-run --user --collect --wait --quiet --unit=oxr-load \
      --working-directory=/home/q/space/python/oxr-starter \
      --setenv=HF_HOME=/home/q/cot/cache/huggingface \
      --setenv=HF_HUB_OFFLINE=1 \
      /home/q/space/python/oxr-starter/.venv/bin/python -m oxr_starter.load_model
"""

from __future__ import annotations

import time

import torch
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

from .vendor.oxr_processor import OXRProcessor

MODEL_ID = "SoMarkAI/OXR-1.0"


def gb(n: int) -> float:
    return round(n / 1024**3, 2)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用：请在 systemd 服务里运行（沙箱内看不到 GPU）")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  总显存 {gb(props.total_memory)} GB")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    processor = OXRProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_pad_token="<|image_pad|>",
    )
    print(f"processor 就绪（{time.time() - t0:.1f}s）")

    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"模型加载完成（{time.time() - t1:.1f}s，总计 {time.time() - t0:.1f}s）")

    n_params = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated()
    print(f"参数量: {n_params / 1e9:.3f} B")
    print(f"权重显存: {gb(torch.cuda.memory_allocated())} GB")
    print(f"显存峰值: {gb(peak)} GB")
    print(f"剩余可用: {gb(props.total_memory - peak)} GB")
    print("OK: 模型可以加载")


if __name__ == "__main__":
    main()
