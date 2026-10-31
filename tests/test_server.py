"""server.py：OpenAI 兼容协议、请求校验、两种执行器。

HTTP 用例会起真实的 ThreadingHTTPServer（随机端口），但执行器是 FakeRunner，
不会加载模型；ThreadRunner 用例用 FakeBackend 顶替推理。
"""

from __future__ import annotations

import itertools
import queue
import threading
import time

import pytest

from oxr_starter import backend as backend_module
from oxr_starter import devices, server
from tests.helpers import (
    FakeBackend,
    FakeRunner,
    data_url,
    handler_for,
    http_json,
    png_bytes,
    serve_http,
)


# --------------------------------------------------------------------------- #
# data URI / 请求体解析
# --------------------------------------------------------------------------- #
def test_decode_data_url_roundtrip():
    payload = png_bytes()
    assert server.decode_data_url(data_url(payload)) == payload


def test_decode_data_url_rejects_plain_url():
    with pytest.raises(ValueError, match="data URI"):
        server.decode_data_url("https://example.com/a.png")


def test_decode_data_url_requires_base64():
    with pytest.raises(ValueError, match="base64"):
        server.decode_data_url("data:image/png,raw-bytes")


def test_extract_request_reads_image_and_text():
    payload = png_bytes()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(payload)}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    image, prompt = server.extract_request({"messages": messages})
    assert image == payload
    assert prompt == "describe"


def test_extract_request_accepts_image_alias_and_string_url():
    payload = png_bytes()
    messages = [
        {
            "content": [
                {"type": "image", "image_url": data_url(payload)},
                {"type": "text", "text": "  "},
            ]
        }
    ]
    image, prompt = server.extract_request({"messages": messages})
    assert image == payload
    assert prompt == server.PROMPT_WARMUP  # 空白提示词回退到默认值


def test_extract_request_joins_text_parts():
    payload = png_bytes()
    messages = [
        {"content": "first"},
        {
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(payload)}},
                {"type": "text", "text": "second"},
            ]
        },
    ]
    _, prompt = server.extract_request({"messages": messages})
    assert prompt == "first\nsecond"


def test_extract_request_uses_first_image_only():
    one, two = png_bytes(color="red"), png_bytes(color="blue")
    messages = [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(one)}},
                {"type": "image_url", "image_url": {"url": data_url(two)}},
            ]
        }
    ]
    image, _ = server.extract_request({"messages": messages})
    assert image == one


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"messages": []},
        {"messages": "not-a-list"},
        {"messages": [{"content": [{"type": "text", "text": "no image"}]}]},
        {"messages": [{"content": "no image either"}]},
    ],
)
def test_extract_request_rejects_missing_image(payload):
    with pytest.raises(ValueError):
        server.extract_request(payload)


def test_open_image_converts_to_rgb():
    img = server.open_image(png_bytes(color="red"))
    assert img.mode == "RGB"
    assert img.size == (120, 80)


def test_warmup_pays_one_short_generation():
    backend = FakeBackend(name="local")
    elapsed = server.warmup(backend, tokens=8)
    assert elapsed >= 0.0
    call = backend.calls[0]
    assert call.prompt == server.PROMPT_WARMUP
    assert call.max_tokens == 8
    assert call.size == (448, 448)


# --------------------------------------------------------------------------- #
# ThreadRunner（instances=1）
# --------------------------------------------------------------------------- #
def test_thread_runner_returns_result_and_stats():
    backend = FakeBackend(name="local", reply="hello")
    runner = server.ThreadRunner(backend, warmup_first=False)
    try:
        assert runner.submit(png_bytes(), "prompt", 16) == "hello"
        stats = runner.stats()
        assert stats == {
            "instances": 1,
            "queue": 0,
            "inflight": 0,
            "ready": 1,
            "warmup_sec": 0.0,
        }
    finally:
        runner.close()
    assert backend.closed is True


def test_thread_runner_warmup_runs_once():
    backend = FakeBackend(name="local")
    runner = server.ThreadRunner(backend)
    try:
        assert runner.stats()["warmup_sec"] >= 0.0
        warmup_calls = [c for c in backend.calls if c.max_tokens == 8]
        assert len(warmup_calls) == 1
    finally:
        runner.close()


def test_thread_runner_reports_backend_error():
    backend = FakeBackend(name="local", fail_calls=(1,))
    runner = server.ThreadRunner(backend, warmup_first=False)
    try:
        with pytest.raises(RuntimeError, match="recognize failed on call 1"):
            runner.submit(png_bytes(), "prompt", 16)
    finally:
        runner.close()


def test_thread_runner_times_out():
    backend = FakeBackend(name="local", delay=0.2)
    runner = server.ThreadRunner(backend, warmup_first=False)
    try:
        with pytest.raises(TimeoutError):
            runner.submit(png_bytes(), "prompt", 16, timeout=0.05)
    finally:
        runner.close()


# --------------------------------------------------------------------------- #
# ProcessRunner（instances>1）：不真的 spawn，只验证统计口径
# --------------------------------------------------------------------------- #
def test_process_runner_stats_counts_queue():
    runner = server.ProcessRunner.__new__(server.ProcessRunner)
    runner.instances = 2
    runner.ready = 2
    runner.failed = []
    runner.warmup_sec = 3.0
    runner.pending = {"j1": {}, "j2": {}}
    runner._finished = 1
    runner._submitted = 4
    runner._lock = threading.Lock()

    assert runner.stats() == {
        "instances": 2,
        "queue": 1,  # 4 提交 - 1 完成 - 2 正在跑
        "inflight": 2,
        "completed": 1,
        "ready": 2,
        "failed": [],
        "warmup_sec": 3.0,
    }


# --------------------------------------------------------------------------- #
# HTTP：GET
# --------------------------------------------------------------------------- #
def test_health_reports_ready_state():
    runner = FakeRunner(stats={"instances": 2, "queue": 1, "ready": 2, "warmup_sec": 1.5})
    with serve_http(handler_for(runner)) as base:
        status, body = http_json(base, "/health")

    assert status == 200
    assert body["status"] == "ok"
    assert body["model"] == "oxr"
    assert body["device"] == "cpu"
    assert body["dtype"] == "float32"
    assert body["instances"] == 2
    assert body["queue"] == 1
    assert body["warmup_sec"] == 1.5
    assert body["uptime_sec"] >= 0
    assert "memory_gb" not in body


def test_health_is_starting_before_first_worker():
    runner = FakeRunner(stats={"ready": 0})
    with serve_http(handler_for(runner)) as base:
        status, body = http_json(base, "/health")

    assert status == 200
    assert body["status"] == "starting"


def test_health_survives_stats_failure():
    class BrokenRunner(FakeRunner):
        def stats(self):
            raise RuntimeError("stats exploded")

    with serve_http(handler_for(BrokenRunner())) as base:
        status, body = http_json(base, "/health")

    assert status == 200
    assert body["status"] == "starting"
    assert "stats exploded" in body["stats_error"]


def test_health_reports_memory_for_mps(monkeypatch):
    monkeypatch.setattr(devices, "driver_gb", lambda device: 6.2)
    runner = FakeRunner(stats={"instances": 2, "ready": 2})
    with serve_http(handler_for(runner, device="mps", dtype_name="float16")) as base:
        status, body = http_json(base, "/health")

    assert status == 200
    assert body["memory_gb"] == 6.2
    assert "主进程" in body["memory_scope"]


def test_health_accepts_trailing_slash():
    with serve_http(handler_for(FakeRunner())) as base:
        status, _ = http_json(base, "/health/")
    assert status == 200


def test_models_lists_served_name():
    with serve_http(handler_for(FakeRunner(), model_name="my-oxr")) as base:
        status, body = http_json(base, "/v1/models")

    assert status == 200
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "my-oxr"
    assert body["data"][0]["owned_by"] == "oxr-starter"


def test_unknown_get_returns_404():
    with serve_http(handler_for(FakeRunner())) as base:
        status, body = http_json(base, "/nope")
    assert status == 404
    assert body["error"]["type"] == "not_found"


# --------------------------------------------------------------------------- #
# HTTP：POST /v1/chat/completions
# --------------------------------------------------------------------------- #
def _completion_payload(text: str | None = "describe", max_tokens: int | None = 64) -> dict:
    content = [{"type": "image_url", "image_url": {"url": data_url(png_bytes())}}]
    if text is not None:
        content.append({"type": "text", "text": text})
    payload = {"model": "oxr", "messages": [{"role": "user", "content": content}]}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def test_completion_roundtrip():
    runner = FakeRunner(text="  # title\nbody  ")
    with serve_http(handler_for(runner)) as base:
        status, body = http_json(
            base, "/v1/chat/completions/", _completion_payload(max_tokens=64)
        )

    assert status == 200
    assert body["object"] == "chat.completion"
    assert body["model"] == "oxr"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "  # title\nbody  ",
    }
    assert body["usage"]["total_tokens"] == 0
    assert body["usage"]["elapsed_sec"] >= 0

    call = runner.calls[0]
    assert call["image"] == png_bytes()
    assert call["prompt"] == "describe"
    assert call["max_tokens"] == 64


def test_completion_uses_default_prompt_and_max_tokens():
    runner = FakeRunner()
    with serve_http(handler_for(runner)) as base:
        status, _ = http_json(base, "/v1/chat/completions", _completion_payload(text=None, max_tokens=None))

    assert status == 200
    assert runner.calls[0]["prompt"] == server.PROMPT_WARMUP
    assert runner.calls[0]["max_tokens"] == 2048


def test_completion_rejects_stream():
    with serve_http(handler_for(FakeRunner())) as base:
        status, body = http_json(
            base, "/v1/chat/completions", {"stream": True, "messages": []}
        )
    assert status == 400
    assert "stream" in body["error"]["message"]


def test_completion_rejects_invalid_json():
    with serve_http(handler_for(FakeRunner())) as base:
        status, body = http_json(base, "/v1/chat/completions", raw=b"{not-json")
    assert status == 400
    assert "合法 JSON" in body["error"]["message"]


def test_completion_rejects_non_object_json():
    with serve_http(handler_for(FakeRunner())) as base:
        status, body = http_json(base, "/v1/chat/completions", raw=b"[]")
    assert status == 400
    assert "JSON 对象" in body["error"]["message"]


def test_completion_rejects_missing_image():
    with serve_http(handler_for(FakeRunner())) as base:
        status, body = http_json(
            base,
            "/v1/chat/completions",
            {"messages": [{"content": [{"type": "text", "text": "hi"}]}]},
        )
    assert status == 400
    assert "没有图片" in body["error"]["message"]


def test_completion_rejects_bad_max_tokens():
    payload = _completion_payload()
    payload["max_tokens"] = "many"
    with serve_http(handler_for(FakeRunner())) as base:
        status, body = http_json(base, "/v1/chat/completions", payload)
    assert status == 400
    assert "max_tokens" in body["error"]["message"]


def test_completion_maps_timeout_to_504():
    runner = FakeRunner(error=TimeoutError("推理超时（>600s）"))
    with serve_http(handler_for(runner)) as base:
        status, body = http_json(base, "/v1/chat/completions", _completion_payload())
    assert status == 504
    assert body["error"]["type"] == "timeout"


def test_completion_maps_backend_error_to_500():
    runner = FakeRunner(error=RuntimeError("CUDA out of memory"))
    with serve_http(handler_for(runner)) as base:
        status, body = http_json(base, "/v1/chat/completions", _completion_payload())
    assert status == 500
    assert body["error"]["type"] == "server_error"
    assert "CUDA out of memory" in body["error"]["message"]


def test_unknown_post_returns_404():
    with serve_http(handler_for(FakeRunner())) as base:
        status, body = http_json(base, "/v1/embeddings", {"messages": []})
    assert status == 404
    assert body["error"]["type"] == "not_found"


# --------------------------------------------------------------------------- #
# ProcessRunner（多进程）：用假队列覆盖 worker / 分发 / 回收
# --------------------------------------------------------------------------- #
class FakeQueue:
    """只实现测试用到的 get/put；取空后抛 SystemExit 结束线程。"""

    def __init__(self, items=()):
        self.items = list(items)
        self.puts: list = []

    def get(self):
        if not self.items:
            raise SystemExit
        return self.items.pop(0)

    def put(self, item):
        self.puts.append(item)


def _new_process_runner(**attrs):
    runner = server.ProcessRunner.__new__(server.ProcessRunner)
    runner.instances = 1
    runner.ready = 0
    runner.failed = []
    runner.warmup_sec = 0.0
    runner.pending = {}
    runner._lock = threading.Lock()
    runner._ids = itertools.count(1)
    runner._submitted = 0
    runner._finished = 0
    for key, value in attrs.items():
        setattr(runner, key, value)
    return runner


def test_worker_main_reports_ready_and_dispatches_jobs(monkeypatch):
    created = {}

    class WorkerBackend:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def recognize(self, image, prompt, max_tokens):
            if prompt == "boom":
                raise RuntimeError("bad prompt")
            return "text"

    monkeypatch.setattr(backend_module, "LocalBackend", WorkerBackend)
    in_q = queue.Queue()
    out_q = queue.Queue()
    in_q.put(("j1", png_bytes(), "hello", 8))
    in_q.put(("j2", png_bytes(), "boom", 8))
    in_q.put(None)

    server._worker_main(in_q, out_q, "m", "cpu", "fp32", max_pixels=1_200_000)

    assert created["max_pixels"] == 1_200_000

    messages = []
    while not out_q.empty():
        messages.append(out_q.get())
    assert messages[0][0] == "ready"
    assert [m for m in messages if m[0] == "done"] == [
        ("done", "j1", ("text", None), 0.0),
        ("done", "j2", (None, "RuntimeError: bad prompt"), 0.0),
    ]


def test_worker_main_reports_startup_failure(monkeypatch):
    class BrokenBackend:
        def __init__(self, **kwargs):
            raise RuntimeError("no model")

    monkeypatch.setattr(backend_module, "LocalBackend", BrokenBackend)
    out_q = queue.Queue()

    server._worker_main(queue.Queue(), out_q, "m", "cpu", "fp32")

    kind, _pid, message, elapsed = out_q.get()
    assert kind == "fail"
    assert "no model" in message
    assert elapsed == 0.0


def test_process_runner_wait_ready_returns_when_all_accounted_for():
    runner = _new_process_runner(instances=2, ready=1, failed=["boom"])
    runner._wait_ready(5.0)  # 1 ready + 1 failed == instances


def test_process_runner_wait_ready_times_out():
    runner = _new_process_runner(instances=2)
    with pytest.raises(RuntimeError, match="启动超时"):
        runner._wait_ready(0.01)


def test_process_runner_collect_dispatches_messages():
    job = {"event": threading.Event(), "text": None, "error": None}
    runner = _new_process_runner(pending={"j1": job})
    runner.out_q = FakeQueue(
        [
            ("ready", 1, None, 2.5),
            ("fail", 2, "load failed", 0.0),
            ("done", "j1", ("result", None), 0.0),
            ("done", "unknown", (None, "ignored"), 0.0),
        ]
    )

    def drain():
        try:
            runner._collect()
        except SystemExit:  # FakeQueue 用 SystemExit 模拟“队列关闭”
            pass

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    thread.join(timeout=2)

    assert runner.ready == 1
    assert runner.warmup_sec == 2.5
    assert runner.failed == ["load failed"]
    assert runner._finished == 1
    assert job["text"] == "result"
    assert job["event"].is_set()


def test_process_runner_submit_waits_for_collector():
    runner = _new_process_runner(in_q=FakeQueue())
    runner._ids = itertools.count(1)
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("text", runner.submit(png_bytes(), "p", 8))
    )
    thread.start()

    deadline = time.time() + 2
    while not runner.pending and time.time() < deadline:
        time.sleep(0.001)
    job = runner.pending.pop("j1")
    job["text"] = "ok"
    job["event"].set()
    thread.join(timeout=2)

    assert result["text"] == "ok"
    assert runner._submitted == 1
    assert runner.in_q.puts[0][0] == "j1"


def test_process_runner_submit_times_out():
    runner = _new_process_runner(in_q=FakeQueue())
    with pytest.raises(TimeoutError):
        runner.submit(png_bytes(), "p", 8, timeout=0.01)
    assert runner.pending == {}
    assert runner._submitted == 1


def test_process_runner_close_sends_one_sentinel_per_worker():
    class Proc:
        def __init__(self):
            self.joined = None

        def join(self, timeout):
            self.joined = timeout

    procs = [Proc(), Proc()]
    runner = _new_process_runner(in_q=FakeQueue(), procs=procs)

    runner.close()

    assert runner.in_q.puts == [None, None]
    assert [p.joined for p in procs] == [5, 5]


def test_process_runner_submit_raises_worker_error():
    runner = _new_process_runner(in_q=FakeQueue())
    result = {}

    def call():
        try:
            runner.submit(png_bytes(), "p", 8)
        except RuntimeError as e:
            result["error"] = str(e)

    thread = threading.Thread(target=call)
    thread.start()
    deadline = time.time() + 2
    while not runner.pending and time.time() < deadline:
        time.sleep(0.001)
    job = runner.pending.pop("j1")
    job["error"] = "CUDA out of memory"
    job["event"].set()
    thread.join(timeout=2)

    assert result["error"] == "CUDA out of memory"
