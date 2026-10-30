"""OXR-1.0 最小 transformers 推理示例（单张图片 → 文本/Markdown）。

走模型自带的 preprocess_inputs + generate，不依赖 vendored processor：
    messages -> apply_chat_template -> input_ids/pixel_values/grid_thws -> generate

运行（必须走 systemd 服务，沙箱里看不到 GPU）：
    systemd-run --user --collect --wait --quiet --unit=oxr-demo \
      --working-directory=/home/q/space/python/oxr-starter \
      --setenv=HF_HOME=/home/q/cot/cache/huggingface \
      --setenv=HF_HUB_OFFLINE=1 \
      /home/q/space/python/oxr-starter/.venv/bin/python -m oxr_starter.generate_demo [图片路径]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM

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
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用：请在 systemd 服务里运行")

    img = Image.open(Path(sys.argv[1])).convert("RGB") if len(sys.argv) > 1 else make_sample_image()
    print(f"输入图: {img.size}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="cuda:0", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"模型就绪（{time.time() - t0:.1f}s）")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT_WHOLE_PAGE},
        ],
    }]

    t1 = time.time()
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages)
    def to_cuda(x):
        if x is None:
            return None
        if isinstance(x, (tuple, list)):
            return torch.cat(list(x), dim=0).to("cuda")
        return x.to("cuda")

    pixel_values = to_cuda(pixel_values)
    grid_thws = to_cuda(grid_thws)
    input_ids = input_ids.to("cuda")
    print(f"预处理完成：{input_ids.shape[-1]} tokens（{time.time() - t1:.1f}s）")

    t2 = time.time()
    with torch.inference_mode():
        out = model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            max_new_tokens=512,
            do_sample=False,
        )
    dt = time.time() - t2

    new_ids = out[0]  # 模型只返回新生成的 token（不含 prompt）
    text = model.text_tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"生成：{len(new_ids)} tokens，{dt:.1f}s（{len(new_ids)/max(dt,1e-6):.1f} tok/s）")
    print(f"显存峰值: {round(torch.cuda.max_memory_allocated()/1024**3, 2)} GB")
    print("---- 输出 ----")
    print(text)


if __name__ == "__main__":
    main()
