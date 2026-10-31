"""OpenAI 兼容的 OXR 推理服务（CUDA / MPS / CPU 通用）。

目的：让 `--backend vllm --workers N` 在**没有 vLLM 的机器**（如 Apple Silicon）
上也能用——客户端代码零改动，服务端协议与 vLLM 一致。

接口：

    GET  /health               设备/精度/队列/显存
    GET  /v1/models            模型列表（客户端会先探这个）
    POST /v1/chat/completions  image_url(data URI) + text，返回 markdown

用法：

    uv run python -m oxr_starter.server                    # 自动选设备，:8001
    uv run python -m oxr_starter.server --port 8002
    uv run python -m oxr_starter.server --instances 3      # 3 个进程各持一份模型

并发说明（M1 Max 实测）：MPS 单流已基本吃满 GPU，**同进程多线程无收益**
（两个线程 8.48s+8.48s 完全串行）；真并行要靠**多进程**：2 进程约 1.07×、
3 进程约 1.38×（每进程多占 3.5~6 GB 统一内存）。所以 `--instances` 用 spawn
子进程，默认 1（= 进程内单工作线程，无序列化开销）。
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from PIL import Image, ImageDraw

from . import devices
from .backend import resolve_max_pixels

PROMPT_WARMUP = "Recognize the whole page content of the image."


def _log(msg: str) -> None:
    print(f"[oxr-server] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 图片解析
# --------------------------------------------------------------------------- #
def decode_data_url(url: str) -> bytes:
    """`data:image/png;base64,....` → 原始字节。"""
    if not url.startswith("data:"):
        raise ValueError("只支持 data URI 形式的 image_url")
    header, _, payload = url.partition(",")
    if ";base64" not in header:
        raise ValueError("只支持 base64 编码的 data URI")
    return base64.b64decode(payload)


def extract_request(payload: dict) -> tuple[bytes, str]:
    """从 OpenAI 请求体里取出 (图片字节, 提示词)。"""
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 不能为空")

    image_bytes: bytes | None = None
    texts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for item in content or []:
            kind = item.get("type")
            if kind in ("image_url", "image"):
                url = item.get("image_url")
                url = url.get("url") if isinstance(url, dict) else url
                if url and image_bytes is None:
                    image_bytes = decode_data_url(url)
            elif kind == "text" and item.get("text"):
                texts.append(item["text"])
    if image_bytes is None:
        raise ValueError("请求里没有图片（需要 image_url 的 data URI）")
    return image_bytes, "\n".join(texts).strip() or PROMPT_WARMUP


def open_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def warmup(backend, tokens: int = 8) -> float:
    """跑一次极短生成，把 Metal shader 编译等一次性开销提前付掉。"""
    img = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(img).text((24, 24), "warmup", fill="black")
    t = time.time()
    backend.recognize(img, PROMPT_WARMUP, tokens)
    return time.time() - t


# --------------------------------------------------------------------------- #
# 执行器：进程内单线程（instances=1）
# --------------------------------------------------------------------------- #
class ThreadRunner:
    """进程内一个工作线程串行跑推理（MPS 上这是最优形态）。"""

    def __init__(self, backend, warmup_first: bool = True):
        self.backend = backend
        self.q: queue.Queue = queue.Queue()
        self.inflight = 0
        self._lock = threading.Lock()
        self.warmup_sec = warmup(backend) if warmup_first else 0.0
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def _loop(self) -> None:
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                job["text"] = self.backend.recognize(
                    open_image(job["image"]), job["prompt"], job["max_tokens"]
                )
            except Exception as e:  # noqa: BLE001 - 单请求失败不拖垮服务
                job["error"] = f"{type(e).__name__}: {e}"
            finally:
                job["event"].set()

    def submit(
        self, image_bytes: bytes, prompt: str, max_tokens: int, timeout: float = 600.0
    ) -> str:
        job = {
            "image": image_bytes,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "event": threading.Event(),
            "text": None,
            "error": None,
        }
        with self._lock:
            self.inflight += 1
        self.q.put(job)
        try:
            if not job["event"].wait(timeout):
                raise TimeoutError(f"推理超时（>{timeout:.0f}s）")
        finally:
            with self._lock:
                self.inflight -= 1
        if job["error"]:
            raise RuntimeError(job["error"])
        return job["text"]

    def stats(self) -> dict:
        with self._lock:
            inflight = self.inflight
        return {
            "instances": 1,
            "queue": self.q.qsize(),
            "inflight": inflight,
            "ready": 1,
            "warmup_sec": round(self.warmup_sec, 1),
        }

    def close(self) -> None:
        self.q.put(None)
        self.worker.join(timeout=2)
        self.backend.close()


# --------------------------------------------------------------------------- #
# 执行器：多进程（instances>1）
# --------------------------------------------------------------------------- #
def _worker_main(in_q, out_q, model_id: str, device: str, dtype: str,
                 max_pixels: int | None = None) -> None:
    """子进程：加载模型 → 预热 → 循环取任务。"""
    from .backend import LocalBackend

    pid = os.getpid()
    t0 = time.time()
    try:
        backend = LocalBackend(model_id=model_id, device=device, dtype=dtype,
                               max_pixels=max_pixels)
        warm = warmup(backend)
    except Exception as e:  # noqa: BLE001 - 子进程要把启动失败报回主进程
        out_q.put(("fail", pid, f"{type(e).__name__}: {e}", 0.0))
        return
    out_q.put(("ready", pid, None, time.time() - t0))
    _log(f"worker {pid} 就绪（加载 {time.time() - t0:.1f}s，预热 {warm:.1f}s）")

    while True:
        job = in_q.get()
        if job is None:
            return
        jid, image_bytes, prompt, max_tokens = job
        try:
            text = backend.recognize(open_image(image_bytes), prompt, max_tokens)
            out_q.put(("done", jid, (text, None), 0.0))
        except Exception as e:  # noqa: BLE001 - 错误文本回传，不中断 worker
            out_q.put(("done", jid, (None, f"{type(e).__name__}: {e}"), 0.0))


class ProcessRunner:
    """N 个 spawn 子进程，各持一份模型；主进程只做分发。"""

    def __init__(
        self,
        instances: int,
        model_id: str,
        device: str,
        dtype: str,
        max_pixels: int | None = None,
        start_timeout: float = 300.0,
    ):
        ctx = mp.get_context("spawn")
        self.in_q = ctx.Queue()
        self.out_q = ctx.Queue()
        self.instances = instances
        self.ready = 0
        self.warmup_sec = 0.0
        self.failed: list[str] = []
        self.pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        # macOS 上 mp.Queue.qsize() 会抛 NotImplementedError（没有 sem_getvalue），自己计数
        self._submitted = 0
        self._finished = 0
        self.procs = [
            ctx.Process(
                target=_worker_main,
                args=(self.in_q, self.out_q, model_id, device, dtype, max_pixels),
                daemon=True,
            )
            for _ in range(instances)
        ]
        for p in self.procs:
            p.start()
        self.collector = threading.Thread(target=self._collect, daemon=True)
        self.collector.start()
        self._wait_ready(start_timeout)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        last = 0
        while time.time() < deadline and len(self.failed) < self.instances:
            time.sleep(0.2)
            with self._lock:
                ready = self.ready
            if ready != last:
                _log(f"就绪 {ready}/{self.instances}")
                last = ready
            if ready + len(self.failed) >= self.instances:
                return
        raise RuntimeError(
            f"worker 启动超时/失败：ready={self.ready} failed={self.failed}"
        )

    def _collect(self) -> None:
        while True:
            kind, key, value, extra = self.out_q.get()
            if kind == "ready":
                with self._lock:
                    self.ready += 1
                    self.warmup_sec = max(self.warmup_sec, float(extra))
                continue
            if kind == "fail":
                with self._lock:
                    self.failed.append(str(value))
                _log(f"worker {key} 启动失败: {value}")
                continue
            with self._lock:
                job = self.pending.pop(key, None)
                if job is not None:
                    self._finished += 1
            if job is None:
                continue
            job["text"], job["error"] = value
            job["event"].set()

    def submit(
        self, image_bytes: bytes, prompt: str, max_tokens: int, timeout: float = 600.0
    ) -> str:
        jid = f"j{next(self._ids)}"
        job = {"event": threading.Event(), "text": None, "error": None}
        with self._lock:
            self.pending[jid] = job
            self._submitted += 1
        self.in_q.put((jid, image_bytes, prompt, max_tokens))
        if not job["event"].wait(timeout):
            with self._lock:
                self.pending.pop(jid, None)
            raise TimeoutError(f"推理超时（>{timeout:.0f}s）")
        if job["error"]:
            raise RuntimeError(job["error"])
        return job["text"]

    def stats(self) -> dict:
        with self._lock:
            inflight = len(self.pending)
            done = self._finished
            submitted = self._submitted
            ready = self.ready
            failed = list(self.failed)
        # 排队数 = 已提交 - 已完成 - 正在跑（正在跑的数量最多等于就绪 worker 数）
        busy = min(inflight, ready)
        return {
            "instances": self.instances,
            "queue": max(0, submitted - done - busy),
            "inflight": inflight,
            "completed": done,
            "ready": ready,
            "failed": failed,
            "warmup_sec": round(self.warmup_sec, 1),
        }

    def close(self) -> None:
        for _ in self.procs:
            self.in_q.put(None)
        for p in self.procs:
            p.join(timeout=5)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "oxr-mps/0.1"
    runner = None  # type: ignore[assignment]
    model_name = "oxr"
    device = "cpu"
    dtype_name = ""
    started_at = time.time()

    # --- 工具 ------------------------------------------------------------- #
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str, etype: str = "invalid_request_error") -> None:
        self._send(code, {"error": {"message": msg, "type": etype}})

    def log_message(self, fmt: str, *args) -> None:  # 简洁访问日志
        _log(f"{self.address_string()} {fmt % args}")

    # --- 路由 ------------------------------------------------------------- #
    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/healthz"):
            try:
                st = self.runner.stats()
            except Exception as e:  # noqa: BLE001 - 健康检查本身不能 500
                st = {"stats_error": f"{type(e).__name__}: {e}"}
            body = {
                "status": "ok" if st.get("ready", 0) > 0 else "starting",
                "model": self.model_name,
                "device": self.device,
                "dtype": self.dtype_name,
                "uptime_sec": round(time.time() - self.started_at, 1),
                **st,
            }
            if self.device in (devices.CUDA, devices.MPS):
                body["memory_gb"] = devices.driver_gb(self.device)
                if st.get("instances", 1) > 1:
                    body["memory_scope"] = "仅主进程（子进程各占 3.5–6 GB，未计入）"
            self._send(200, body)
        elif path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_name,
                            "object": "model",
                            "created": int(self.started_at),
                            "owned_by": "oxr-starter",
                        }
                    ],
                },
            )
        else:
            self._error(404, f"未知路径 {path}", "not_found")

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        if path != "/v1/chat/completions":
            self._error(
                404, f"未知路径 {path}（只支持 /v1/chat/completions）", "not_found"
            )
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:  # noqa: BLE001 - 任何解析失败都该回 400
            self._error(400, f"请求体不是合法 JSON：{e}")
            return
        if not isinstance(payload, dict):
            self._error(400, "请求体必须是 JSON 对象")
            return
        if payload.get("stream"):
            self._error(400, "不支持 stream=true（本服务为同步返回）")
            return
        try:
            image_bytes, prompt = extract_request(payload)
        except Exception as e:  # noqa: BLE001 - 请求体校验错误统一回 400
            self._error(400, str(e))
            return

        try:
            max_tokens = int(payload.get("max_tokens") or 2048)
        except (TypeError, ValueError):
            self._error(400, "max_tokens 必须是整数")
            return
        t0 = time.time()
        try:
            text = self.runner.submit(image_bytes, prompt, max_tokens)
        except TimeoutError as e:
            self._error(504, str(e), "timeout")
            return
        except Exception as e:  # noqa: BLE001 - 兜底成 500，保持服务存活
            self._error(500, str(e), "server_error")
            return

        dt = time.time() - t0
        _log(
            f"完成 1 请求 {dt:.1f}s（{max_tokens} tok 上限，图片 {len(image_bytes) / 1024:.0f} KB）"
        )
        self._send(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                # token 数不做精确统计，只给耗时，避免误导
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "elapsed_sec": round(dt, 2),
                },
            },
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="OXR OpenAI 兼容服务（MPS/CUDA/CPU）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="SoMarkAI/OXR-1.0")
    ap.add_argument("--served-model-name", default="oxr")
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--dtype", default="auto", help="auto / bf16 / fp16 / fp32")
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="视觉编码最大像素数（如 1200000）；0=模型默认 1344×1792",
    )
    ap.add_argument(
        "--instances",
        type=int,
        default=1,
        help="模型实例数（>1 用子进程；每实例多占约 3.5-6 GB）",
    )
    ap.add_argument(
        "--no-warmup", action="store_true", help="不预热（首请求会慢 3-4s）"
    )
    args = ap.parse_args()

    dev = devices.pick_device(args.device)
    dtype = devices.resolve_dtype(dev, args.dtype)
    dtype_name = str(dtype).replace("torch.", "")
    try:
        max_pixels = resolve_max_pixels(args.max_pixels)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    px_info = f"；max_pixels {max_pixels}" if max_pixels else ""
    _log(f"设备 {devices.describe(dev, dtype)}；实例 {args.instances}{px_info}")

    t0 = time.time()
    if args.instances > 1:
        runner = ProcessRunner(args.instances, args.model, dev, dtype_name, max_pixels)
    else:
        from .backend import LocalBackend

        backend = LocalBackend(model_id=args.model, device=dev, dtype=dtype_name,
                               max_pixels=max_pixels)
        runner = ThreadRunner(backend, warmup_first=not args.no_warmup)
    _log(f"就绪（{time.time() - t0:.1f}s）→ http://{args.host}:{args.port}/v1")

    Handler.runner = runner
    Handler.model_name = args.served_model_name
    Handler.device = dev
    Handler.dtype_name = dtype_name
    Handler.started_at = time.time()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True

    def _stop(signum, _frame):
        _log(f"收到信号 {signum}，退出中…")
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        runner.close()
        _log("已停止")


if __name__ == "__main__":
    main()
