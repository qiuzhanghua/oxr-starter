"""OXR-1.0 最小 transformers 推理示例（单张图片 → 文本/Markdown）。

走模型自带的 preprocess_inputs + generate，不依赖 vendored processor：
    messages -> apply_chat_template -> input_ids/pixel_values/grid_thws -> generate

设备：自动选 CUDA → MPS → CPU，可用 `--device` / `--dtype` 覆盖。

    uv run python -m oxr_starter.generate_demo
    uv run python -m oxr_starter.generate_demo ~/page.png --device mps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from . import devices
from .backend import resolve_max_pixels

MODEL_ID = "SoMarkAI/OXR-1.0"

# 官方提示词（SoMarkAI/OXR 仓库 oxr/model/prompts.py）
PROMPT_WHOLE_PAGE = "Recognize the whole page content of the image."
PROMPT_TEXT = "Recognize the content of the image."
PROMPT_TABLE = "Recognize the table of the image into html format."


def make_sample_image() -> Image.Image:
    img = Image.new("RGB", (900, 620), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 40), "Quarterly Report 2026", fill="black")
    d.text((60, 110), "Revenue grew 18% year over year, driven by", fill="black")
    d.text((60, 140), "strong demand in the enterprise segment.", fill="black")
    d.rectangle((60, 220, 840, 420), outline="black", width=2)
    for y in (270, 320, 370):
        d.line((60, y, 840, y), fill="black", width=2)
    d.line((360, 220, 360, 420), fill="black", width=2)
    d.text((80, 235), "Region", fill="black")
    d.text((380, 235), "Revenue", fill="black")
    d.text((80, 285), "North", fill="black")
    d.text((380, 285), "12.4M", fill="black")
    d.text((80, 335), "South", fill="black")
    d.text((380, 335), "9.8M", fill="black")
    d.text((80, 385), "Total", fill="black")
    d.text((380, 385), "22.2M", fill="black")
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="OXR 单图推理")
    ap.add_argument("image", nargs="?", type=Path, help="图片路径（缺省用内置样例）")
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--dtype", default="auto", help="auto / bf16 / fp16 / fp32")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="视觉编码最大像素数（如 1200000）；0=模型默认 1344×1792")
    args = ap.parse_args()

    try:
        max_pixels = resolve_max_pixels(args.max_pixels)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    from transformers import AutoModelForCausalLM

    dev = devices.pick_device(args.device)
    dtype = devices.resolve_dtype(dev, args.dtype)
    print(f"设备: {devices.describe(dev, dtype)}")

    img = Image.open(args.image).convert("RGB") if args.image else make_sample_image()
    px_info = f"，max_pixels {max_pixels}" if max_pixels else ""
    print(f"输入图: {img.size}{px_info}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True, **devices.device_kwargs(dev),
    )
    model = devices.move_model(model, dev)
    model.eval()
    print(f"模型就绪（{time.time() - t0:.1f}s）")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT_WHOLE_PAGE},
        ],
    }]

    tdev = devices.torch_device(dev)
    t1 = time.time()
    preprocess_kwargs = {} if max_pixels is None else {"max_pixels": max_pixels}
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages, **preprocess_kwargs)
    pixel_values = devices.to_device(pixel_values, tdev)
    grid_thws = devices.to_device(grid_thws, tdev)
    input_ids = input_ids.to(tdev)
    print(f"预处理完成：{input_ids.shape[-1]} tokens（{time.time() - t1:.1f}s）")

    with devices.PeakTracker(dev) as peak:
        t2 = time.time()
        with torch.inference_mode():
            out = model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=args.max_tokens,
                do_sample=False,
            )
        if dev == devices.MPS:
            torch.mps.synchronize()
        dt = time.time() - t2

    new_ids = out[0]  # 模型只返回新生成的 token（不含 prompt）
    text = model.text_tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"生成：{len(new_ids)} tokens，{dt:.1f}s（{len(new_ids)/max(dt,1e-6):.1f} tok/s）")
    print(f"显存: {'无（CPU）' if dev == devices.CPU else f'峰值 {peak.peak_gb} GB'}")
    print("---- 输出 ----")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
