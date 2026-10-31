"""测试共用的小工具：合成 PDF、记录调用的假后端、临时 HTTP 服务。

这里刻意不加载真实模型：所有测试都用 `FakeBackend` / `FakeRunner` 替换推理，
所以整套测试在 CPU 上、无网络、无 HuggingFace 缓存时也能跑完。
"""

from __future__ import annotations

import io
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image

from oxr_starter import server

# 与 parse_pdf.PROMPT_TABLE 保持一致；单独写一份，避免 helper 依赖业务模块。
TABLE_PROMPT = "Recognize the table of the image into html format."


def png_bytes(width: int = 120, height: int = 80, color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def make_pdf(
    path: Path,
    pages: int = 1,
    *,
    images: bool = False,
    tiny_image: bool = False,
    text: str = "hello",
) -> Path:
    """生成一份真实（但内容极简）的 PDF，供 PyMuPDF 相关测试使用。"""
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((40, 60), f"{text} page {n + 1}")
        if images:
            page.insert_image(pymupdf.Rect(60, 200, 180, 280), stream=png_bytes())
        if tiny_image:
            page.insert_image(pymupdf.Rect(250, 250, 260, 260), stream=png_bytes(10, 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


@dataclass(frozen=True)
class BackendCall:
    prompt: str
    max_tokens: int
    size: tuple[int, int]
    thread: str


class FakeBackend:
    """实现 `recognize(image, prompt, max_tokens)` 的假后端。

    - `reply=None`：第 n 次调用返回 `page text n`
    - `reply="..."`：每次返回同一段文本
    - `reply=callable`：`reply(call_index)` 决定返回内容
    - `fail_calls`：指定第几次调用（含表格调用）抛异常
    """

    def __init__(
        self,
        name: str = "vllm",
        reply: str | Callable[[int], str] | None = None,
        table_reply: str = "<table><tr><td>cell</td></tr></table>",
        fail_calls: tuple[int, ...] = (),
        fail_tables: bool = False,
        delay: float = 0.0,
    ):
        self.name = name
        self.reply = reply
        self.table_reply = table_reply
        self.fail_calls = set(fail_calls)
        self.fail_tables = fail_tables
        self.delay = delay
        self.calls: list[BackendCall] = []
        self.closed = False
        self._lock = threading.Lock()

    def recognize(self, image: Image.Image, prompt: str, max_tokens: int = 2048) -> str:
        with self._lock:
            index = len(self.calls) + 1
            self.calls.append(
                BackendCall(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    size=image.size,
                    thread=threading.current_thread().name,
                )
            )
        if self.delay:
            time.sleep(self.delay)
        if index in self.fail_calls:
            raise RuntimeError(f"recognize failed on call {index}")
        if prompt == TABLE_PROMPT:
            if self.fail_tables:
                raise RuntimeError("table recognition failed")
            return self.table_reply
        if callable(self.reply):
            return self.reply(index)
        if self.reply is not None:
            return self.reply
        return f"page text {index}"

    def close(self) -> None:
        self.closed = True

    @property
    def prompts(self) -> list[str]:
        return [c.prompt for c in self.calls]


class FakeRunner:
    """替身执行器：直接返回文本，不做推理。"""

    def __init__(
        self,
        text: str = "recognized markdown",
        error: Exception | None = None,
        stats: dict | None = None,
    ):
        self.text = text
        self.error = error
        self.calls: list[dict] = []
        self.stats_data = {
            "instances": 1,
            "queue": 0,
            "inflight": 0,
            "ready": 1,
            "warmup_sec": 0.0,
        }
        if stats:
            self.stats_data.update(stats)

    def submit(
        self, image_bytes: bytes, prompt: str, max_tokens: int, timeout: float = 600.0
    ) -> str:
        self.calls.append(
            {
                "image": image_bytes,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
        )
        if self.error:
            raise self.error
        return self.text

    def stats(self) -> dict:
        return dict(self.stats_data)


def handler_for(
    runner,
    *,
    device: str = "cpu",
    dtype_name: str = "float32",
    model_name: str = "oxr",
):
    """生成一个不打印访问日志的 Handler 子类（类属性按测试隔离）。"""

    class TestHandler(server.Handler):
        def log_message(self, fmt: str, *args) -> None:  # 静音
            pass

    TestHandler.runner = runner
    TestHandler.device = device
    TestHandler.dtype_name = dtype_name
    TestHandler.model_name = model_name
    TestHandler.started_at = time.time()
    return TestHandler


@contextmanager
def serve_http(handler_cls):
    """在随机端口起一个真实的 ThreadingHTTPServer，退出时关闭。"""
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    httpd.daemon_threads = True
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        host, port = httpd.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def http_json(
    base: str,
    path: str,
    payload: dict | None = None,
    *,
    raw: bytes | None = None,
    method: str | None = None,
) -> tuple[int, dict]:
    """发请求并解析 JSON；4xx/5xx 也返回 (status, body) 而不是抛异常。"""
    body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(
        base + path,
        data=body,
        headers=headers,
        method=method or ("POST" if body is not None else "GET"),
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def free_port() -> int:
    """找一个当前没人监听的端口（用于测试连接失败）。"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def data_url(payload: bytes, mime: str = "image/png") -> str:
    import base64

    return f"data:{mime};base64," + base64.b64encode(payload).decode()
