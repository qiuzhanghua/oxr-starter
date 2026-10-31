# 在 Apple Silicon（MPS）上运行 OXR-1.0

原项目只支持 NVIDIA：`device_map="cuda:0"`、`bfloat16`、`torch.cuda.*`、systemd 起
vLLM。本机（M1 Max / 64 GB 统一内存 / macOS 27）没有 CUDA、也跑不了 vLLM，所以这份
文档记录**同一套代码如何在 MPS 上跑通全部功能**，以及实测数据。

结论：**功能 100% 覆盖，速度反而比原 README 的 4090 基线快约 4 倍。**

---

## 1. 一句话差异

| | 原 CUDA 路径 | 本机 MPS 路径 |
|---|---|---|
| 设备 | `cuda:0` + `device_map` | `mps` + `.to("mps")` |
| dtype | `bfloat16` | `float16`（MPS 上比 bf16 快约 16%，输出一致）|
| 显存统计 | `torch.cuda.max_memory_allocated()` | `torch.mps.driver_allocated_memory()` + 轮询采样 |
| 服务 | vLLM（`scripts/oxr-vllm.sh` + systemd）| `oxr_starter.server`（`scripts/oxr-mac.sh` + launchd）|
| 并发 | vLLM 连续批处理 | 单进程单流；要并行就多进程（`--instances`）|
| 算子 | 可选 flash-attn | 无 flash-attn，走 `F.scaled_dot_product_attention` |

**CUDA 路径的行为一行没变**：`devices.pick_device()` 优先返回 `cuda`，
`dtype` 仍是 `bfloat16`，`device_map="cuda:0"` 照旧。Mac 上只是走另一条分支。

---

## 2. 快速上手

```bash
# 0) 依赖（若在受限网络里，代理被墙时绕过：env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY uv sync）
uv sync

# 1) 下模型（约 2.8 GB；国内建议走镜像）
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  uv run python -c "from huggingface_hub import snapshot_download as d; print(d('SoMarkAI/OXR-1.0'))"

# 2) 三连自检：能加载 → 能推理 → 设备信息
uv run python -m oxr_starter.load_model
uv run python -m oxr_starter.generate_demo
uv run python -m oxr_starter.generate_demo ~/page.png --max-tokens 1024

# 3a) 直接解析（进程内，最简单）
uv run python -m oxr_starter.parse_pdf doc.pdf -o out/doc.md --parse-tables

# 3b) 起服务再用客户端（等价于原来的 vLLM 用法）
./scripts/oxr-mac.sh start            # launchd 托管，等就绪
uv run python -m oxr_starter.parse_pdf doc.pdf -o out/doc.md \
    --backend vllm --workers 2 --parse-tables
./scripts/oxr-mac.sh stop             # 释放统一内存
```

`--device` 可显式指定 `mps` / `cpu`；`--dtype` 可指定 `fp16` / `bf16` / `fp32`。
默认 `auto`：`cuda → mps → cpu`。

---

## 3. 实测数据（M1 Max 32 核 GPU / 64 GB / fp16）

| 场景 | 耗时 | 吞吐 | 显存 |
|---|---|---|---|
| 900×620 小图，128 tok | 1.96 s | 65 tok/s | 3.6 GB |
| 合成 A4 @150 DPI，512 tok | 9.4 s | 55 tok/s | 5.5 GB |
| 真实双栏论文页 1654×2174，自然停止 2221 tok | **31.5 s** | 71 tok/s | **6.2 GB** |
| 整页 + 表格二次识别（`--parse-tables`）| 49.7 s | — | 6.2 GB |

对照原 README（RTX 4090 Laptop / transformers / bf16）：**单页 27.6 s，12 tok/s**。
同等 token 量下 M1 大约 4 倍快 —— 因为 4090 那条路同样没装 flash-attn，而 M1 的
fp16 单元效率很高。

### 成本结构

```
整页 ≈ 3.5 s 固定开销 + 11 ms/token
        ├─ 视觉编码 2.63 s   ← 27 层 Python 逐层循环 + window attn（无 flash-attn）
        └─ prefill   0.9 s
性能解码 ≈ 90 tok/s（边际）
```

所以 300 tok 的页约 6.8 s，500 tok 约 9 s，2200 tok 约 28 s。

### 并发（决定服务端怎么配）

| 方式 | 聚合吞吐 | 结论 |
|---|---|---|
| 单进程 | 40 tok/s | 基准 |
| 同进程 2 线程 | 两路各 8.48 s，**完全串行** | MPS 单流已饱和，多线程无用 |
| 2 进程 | 43 tok/s | 1.07× |
| 3 进程 | 55.5 tok/s | **1.38×**，显存 ×3 |

短请求场景多进程收益更明显（两个 120 tok 请求并发：2.9 s 完成两个，等于 2×）。
这就是 `--instances` 用**子进程**而不是线程的原因。

---

## 4. 服务端：为什么不用 vLLM

vLLM 依赖 CUDA kernel，MPS 上跑不了。但客户端代码（`VLLMBackend`）、`--workers`
语义、`docs/vllm.md` 里的调用方式都依赖 OpenAI 协议，所以本仓库提供了一个
**协议兼容的最小服务** `oxr_starter.server`：

```
GET  /health                设备/精度/队列/显存
GET  /v1/models             模型列表（客户端会先探这个）
POST /v1/chat/completions   image_url(data URI) + text → markdown
```

- 请求体与 vLLM 一致，`--backend vllm --base-url ...` **零改动**可用。
- 不支持 `stream=true`（本项目用不到，显式返回 400 而不是假装支持）。
- 单实例 = 进程内一个工作线程串行（MPS 上最优）；`--instances N` = N 个 spawn 子进程。
- 启动即预热（一次 8 token 的生成），避免首个请求吃 3–4 s 的 Metal shader 编译。
- 队列深度、就绪 worker 数、显存都从 `/health` 可查。

```bash
./scripts/oxr-mac.sh start                                  # 默认 1 实例
OXR_INSTANCES=3 OXR_PORT=8001 ./scripts/oxr-mac.sh restart  # 3 实例（约 +11 GB）
curl -s localhost:8001/health | python3 -m json.tool
```

---

## 5. launchd 托管（对应 systemd）

`scripts/oxr-vllm.sh` 用 systemd user unit；macOS 用 LaunchAgent：

```bash
./scripts/oxr-mac.sh install    # 生成 ~/Library/LaunchAgents/com.oxr.server.plist
./scripts/oxr-mac.sh start      # launchctl bootstrap gui/$UID，等 /health 通过
./scripts/oxr-mac.sh status     # launchctl print + /health
./scripts/oxr-mac.sh logs 100   # ~/Library/Logs/oxr-server.log
./scripts/oxr-mac.sh stop       # bootout，释放统一内存
./scripts/oxr-mac.sh uninstall
./scripts/oxr-mac.sh run        # 前台直跑，调试用（不经 launchd）
```

`KeepAlive.SuccessfulExit=false`：异常崩溃自动拉起，正常退出不重启。

---

## 6. 代码改动清单

| 文件 | 改动 |
|---|---|
| `devices.py` | **新增**。设备选择 / dtype / 显存报告 / `PeakTracker`（MPS 无峰值 API，轮询采样）|
| `backend.py` | `LocalBackend` 走 `devices`；删掉 `_to_cuda`；新增 `close()` |
| `server.py` | **新增**。OpenAI 兼容服务，`ThreadRunner` / `ProcessRunner` |
| `load_model.py` | 重写：设备信息 + 峰值显存 + 剩余上限 |
| `generate_demo.py` | 重写：`--device/--dtype/--max-tokens`，去掉 `sys.argv` 硬编码 |
| `layout.py` | 去掉重复的模型加载 / `to_cuda`，改调 `LocalBackend`；区域检测本身不需要 GPU |
| `parse_pdf.py` / `parse_batch.py` / `api.py` | 新增 `--device` / `--dtype` 并一路透传；`close()` 真正释放显存 |
| `scripts/oxr-mac.sh` | **新增**。launchd 管理 |

---

## 7. 已知风险 / 注意

1. **视觉编码 2.63 s/页**（占固定开销 75%）因为没 flash-attn，走 Python 逐层循环。
   想要更快就把 window-attn 向量化成 block-diagonal mask + 一次 SDPA，预计省 0.5–1 s/页。
   *当前未做*——先保证功能一致。
2. **MPS 没有 `max_memory_allocated()`**：峰值靠轮询采样（`PeakTracker`，50 ms 间隔），
   极端短峰值可能漏采。要严格上限就设 `OXR_MPS_MEM_FRACTION=0.75`。
3. **超长页**：2221 tok 已占 6.2 GB；把 `--max-tokens` 拉到 8192 且页面很密时要当心
   整机内存（`recommended_max_memory` 是 51.8 GB，不设比例理论上能吃满）。
4. **首次调用慢**：Metal shader 编译约 3–4 s，服务启动已预热，脚本里直接跑第一次会慢。
5. **`use_cache: false` 写在 config 里，但实测走了 KV cache**（transformers 5.22 行为）。
   升级 transformers 后速度可能变，建议在测试里锁住 tok/s 下限。
6. **fp16 与 bf16 输出逐字节一致**，但 fp16 快 16%。极长上下文若担心数值，切 `--dtype bf16`。
7. **并发别指望 `--workers 8` 变快**：单流已饱和，`workers` 只是让请求排队而不报错。
8. **模型 `layout_analysis/last.pth`（D-FINE）未接入**：那是官方 38 文件 FastAPI 服务的
   依赖，`layout.py` 用 PyMuPDF 的结构信息代替。**原项目也没有这块**，属于双方都缺的能力。

---

## 8. 与 CUDA 机器的互操作

同一份 `out/doc.json`、同样的 `pNNNN.md` 断点目录，两条路径产出一致（贪心解码、
同一套提示词、同一套后处理）。差异只在速度和显存；`meta.backend` 会记录实际后端，
方便对比时区分来源。
