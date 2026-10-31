"""MPS 性能回归测试（默认跳过）。

`docs/mac.md` 的核心结论：视觉编码是固定开销的大头，而视觉耗时随 token 数
（即 `max_pixels`）平方增长。这个用例把结论变成可执行的阈值：

    OXR_SLOW=1 uv run pytest tests/test_slow_perf.py -s

可调阈值（都带默认值，换机器时按需覆盖）：

    OXR_VISION_MAX_SECONDS  默认 2.4M 像素下单次视觉编码上限（默认 8 s）
    OXR_VISION_SPEEDUP_MIN  1.2M 相对 2.4M 至少快多少倍（默认 1.3）
    OXR_MIN_TOK_S           解码吞吐下限（默认 20 tok/s）

只在 Apple Silicon（MPS）上运行；需要本地已缓存 SoMarkAI/OXR-1.0。
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from PIL import Image, ImageDraw

from oxr_starter import devices
from oxr_starter.backend import DEFAULT_MAX_PIXELS, LocalBackend

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("OXR_SLOW") != "1",
        reason="真实模型慢测试：设置 OXR_SLOW=1 才运行",
    ),
    pytest.mark.skipif(not devices.mps_available(), reason="需要 Apple Silicon / MPS"),
]


def _sync() -> None:
    torch.mps.synchronize()


def _bench(fn, repeats: int = 3) -> float:
    fn()
    _sync()
    t0 = time.time()
    for _ in range(repeats):
        fn()
    _sync()
    return (time.time() - t0) / repeats


def _page_image() -> Image.Image:
    """与 OmniDocBench 论文页同尺寸的合成页（文字只影响预处理，不影响视觉耗时）。"""
    img = Image.new("RGB", (1654, 2174), "white")
    draw = ImageDraw.Draw(img)
    for y in range(80, 2100, 36):
        draw.text((90, y), f"line {y}: performance regression fixture", fill="black")
    return img


def _preprocess(backend: LocalBackend, img: Image.Image, max_pixels: int):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Recognize the whole page content of the image."},
            ],
        }
    ]
    ids, pixel_values, grid = backend.model.preprocess_inputs(
        messages, max_pixels=max_pixels
    )
    dev = devices.torch_device(backend.device)
    return ids.to(dev), pixel_values.to(dev), grid.to(dev)


def _vision(backend: LocalBackend, pixel_values, grid):
    with torch.inference_mode():
        out = backend.model.visual_tokenizer.vit(
            pixel_values, grid, output_hidden_states=True, return_dict=True
        )
    return out.last_hidden_state


def test_mps_vision_and_decode_budget() -> None:
    backend = LocalBackend(device="mps")
    try:
        img = _page_image()
        _, pv_big, grid_big = _preprocess(backend, img, DEFAULT_MAX_PIXELS)
        ids_small, pv_small, grid_small = _preprocess(backend, img, 1_200_000)

        def run_big():
            _vision(backend, pv_big, grid_big)

        def run_small():
            _vision(backend, pv_small, grid_small)

        big_sec = _bench(run_big)
        small_sec = _bench(run_small)
        max_big = float(os.environ.get("OXR_VISION_MAX_SECONDS", "8"))
        speedup_min = float(os.environ.get("OXR_VISION_SPEEDUP_MIN", "1.3"))
        tok_s_min = float(os.environ.get("OXR_MIN_TOK_S", "20"))

        # 解码吞吐：先付一次 prefill，再用 (65 - 1) 个 token 的差值算纯解码速度。
        # 直接测 64 token 会被 prefill 稀释（实测只有 ~19 tok/s）。
        def generate(n: int):
            torch.mps.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                out = backend.model.generate(
                    inputs=ids_small,
                    pixel_values=pv_small,
                    grid_thws=grid_small,
                    max_new_tokens=n,
                    do_sample=False,
                )
            torch.mps.synchronize()
            return out, time.time() - t0

        generate(2)  # 预热 shader / 分配缓存
        _, t_one = generate(1)
        out, t_many = generate(65)
        decode_sec = max(t_many - t_one, 1e-9)
        generated = 64
        tok_s = generated / decode_sec
        text = backend.model.text_tokenizer.decode(out[0], skip_special_tokens=True)

        print(
            f"\n[MPS perf] tokens {pv_big.shape[0]} -> {pv_small.shape[0]}；"
            f"vision {big_sec:.2f}s -> {small_sec:.2f}s（{big_sec / small_sec:.1f}×）；"
            f"decode {generated} tok / {decode_sec:.1f}s = {tok_s:.1f} tok/s"
        )

        assert big_sec <= max_big, f"视觉编码 {big_sec:.2f}s 超过上限 {max_big}s"
        assert big_sec / small_sec >= speedup_min, (
            f"max_pixels 降档只快了 {big_sec / small_sec:.2f}×，"
            f"低于阈值 {speedup_min}×"
        )
        assert tok_s >= tok_s_min, f"解码 {tok_s:.1f} tok/s 低于阈值 {tok_s_min}"
        assert text.strip()
    finally:
        backend.close()
