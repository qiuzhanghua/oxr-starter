"""真实模型冒烟测试（默认跳过）。

这些用例会加载 SoMarkAI/OXR-1.0（约 3.5-6 GB 内存），耗时从几十秒到几分钟，
所以只在显式开启时运行：

    OXR_SLOW=1 uv run pytest -m slow

可调环境变量：

    OXR_DEVICE       auto / cuda / mps / cpu（默认 auto）
    OXR_DTYPE        auto / bf16 / fp16 / fp32（默认 auto）
    OXR_DPI          渲染 DPI（默认 150）
    OXR_MAX_TOKENS   生成上限（默认 256）
    OXR_MAX_SECONDS  单页耗时上限，超过即失败（默认 600）
"""

from __future__ import annotations

import os
import time

import pytest

from oxr_starter import OXRClient, devices
from tests.helpers import make_pdf

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("OXR_SLOW") != "1",
        reason="真实模型慢测试：设置 OXR_SLOW=1 才运行",
    ),
]


def test_real_model_parses_synthetic_page(tmp_path):
    pdf = make_pdf(tmp_path / "page.pdf", pages=1)
    client = OXRClient(
        backend="local",
        device=os.environ.get("OXR_DEVICE", "auto"),
        dtype=os.environ.get("OXR_DTYPE", "auto"),
    )
    t0 = time.time()
    try:
        doc = client.parse(
            pdf,
            dpi=int(os.environ.get("OXR_DPI", "150")),
            max_tokens=int(os.environ.get("OXR_MAX_TOKENS", "256")),
        )
    finally:
        client.close()
    elapsed = time.time() - t0

    assert client.backend.device in devices.available_devices()
    assert doc.failed_pages == []
    assert len(doc.pages) == 1
    assert doc.pages[0].markdown.strip()
    assert elapsed < float(os.environ.get("OXR_MAX_SECONDS", "600"))
