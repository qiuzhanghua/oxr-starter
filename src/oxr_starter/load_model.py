"""加载 OXR-1.0，报告参数量与显存占用（不推理）。

用途：确认模型能在本机加载，作为后续推理脚本的前置检查。

设备：自动选 CUDA → MPS → CPU，可用 `--device` / `--dtype` 覆盖。

    uv run python -m oxr_starter.load_model
    uv run python -m oxr_starter.load_model --device mps --dtype fp16
    # Linux/GPU 机器上若想限制在某个卡：--device cuda
"""

from __future__ import annotations

import argparse
import time

from . import devices

MODEL_ID = "SoMarkAI/OXR-1.0"


def main() -> None:
    ap = argparse.ArgumentParser(description="加载 OXR-1.0 并报告资源占用")
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--dtype", default="auto", help="auto / bf16 / fp16 / fp32")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = devices.pick_device(args.device)
    dtype = devices.resolve_dtype(dev, args.dtype)
    print(f"设备: {devices.describe(dev, dtype)}")
    devices.prepare(dev)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer 就绪（{time.time() - t0:.1f}s，vocab {len(tokenizer)}）")

    t1 = time.time()
    with devices.PeakTracker(dev) as peak:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **devices.device_kwargs(dev),
        )
        model = devices.move_model(model, dev)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"模型加载完成（{time.time() - t1:.1f}s，总计 {time.time() - t0:.1f}s）")
        print(devices.memory_line(dev))

    print(f"参数量: {n_params / 1e9:.3f} B")
    print(f"峰值显存: {peak.peak_gb} GB")
    if dev == devices.MPS:
        limit = devices.gb(torch.mps.recommended_max_memory())
        print(f"推荐上限: {limit} GB（余 {limit - peak.peak_gb:.2f} GB 给 KV cache）")
    print("OK: 模型可以加载")


if __name__ == "__main__":
    main()
