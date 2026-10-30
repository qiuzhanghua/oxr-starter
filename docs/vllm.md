# 用 vLLM 跑 OXR（本机实测可行）

> 2026-10-30 实测：RTX 4090 Laptop 16 GB，Ubuntu 24.04，驱动 580.178.04

## 结论

**可行，且比 transformers 快约 21 倍。**

| 指标 | transformers | vLLM |
|---|---|---|
| 整页识别 | 27.6 s | **1.3 s** |
| 生成速度 | ~12 tok/s | **239.5 tok/s** |
| 显存 | 3.9 GB | 约 8.8 GB（utilization 0.55）|

KV cache 实测量：**5.68 GiB / 53,120 tokens**，16 GB 仍余约 7 GB。

## 环境

独立 venv，与 `oxr-starter`（transformers 环境）隔离：

```
~/space/python/oxr-vllm     9.1 GB
  vllm           0.25.1
  torch          2.11.0+cu130
  transformers   5.22.0
  ninja          已装（必需）
  oxr-vllm-plugin 0.2.0（--no-deps 安装）
```

安装：

```bash
uv venv --python 3.12 ~/space/python/oxr-vllm
VIRTUAL_ENV=~/space/python/oxr-vllm uv pip install 'vllm==0.25.1'
VIRTUAL_ENV=~/space/python/oxr-vllm uv pip install ninja

# 插件在模型仓库里（HF 缓存快照的 plugins/vllm）
cp -rL <snapshot>/plugins/vllm /tmp/oxr-plugin
VIRTUAL_ENV=~/space/python/oxr-vllm uv pip install --no-deps /tmp/oxr-plugin
```

> `--no-deps` 是为了保留 transformers 5.22.0（插件 pyproject 里 pin 的是 5.13.1，
> 但 vLLM 0.25.1 与 5.22.0 配合正常）。

## 启动

```bash
MODEL=$(ls -d ~/cot/cache/huggingface/hub/models--SoMarkAI--OXR-1.0/snapshots/*/)

VLLM_PLUGINS=oxr \
CPATH=/home/q/cot/lib/python_3.12.16_linux_amd64_gnu/include/python3.12 \
PATH=~/space/python/oxr-vllm/bin:$PATH \
~/space/python/oxr-vllm/bin/vllm serve "$MODEL" \
  --trust-remote-code --served-model-name oxr \
  --chat-template "$MODEL/chat_template.jinja" \
  --dtype bfloat16 --gpu-memory-utilization 0.55 --max-model-len 8192 \
  --host 127.0.0.1 --port 8001
```

启动约 **80 秒**（加载权重 + CUDA graph capture）。

## 两个必踩的坑

### 1. `Python.h: 没有那个文件或目录`

vLLM 的 torch inductor 会 JIT 编译 CUDA 扩展，需要 Python 开发头文件。
系统 Python（`/usr/bin/python3.12`）没装 `python3.12-dev`。

**不必装 apt 包**——uv 管理的 Python 自带头文件，用 `CPATH` 指过去即可：

```bash
CPATH=/home/q/cot/lib/python_3.12.16_linux_amd64_gnu/include/python3.12
```

### 2. `FileNotFoundError: ninja`

修好头文件后立刻暴露的下一个错误。装 `ninja`（pip 包，自带二进制）：

```bash
uv pip install ninja
```

> vLLM 服务进程的 `PATH` 里要能找到 ninja——如果用 systemd 启动，
> 记得把 `~/space/python/oxr-vllm/bin` 放进 `PATH`。

## 验证

```bash
curl -s http://127.0.0.1:8001/v1/models
# {"data":[{"id":"oxr",...}]}

curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"oxr","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},
        {"type":"text","text":"Recognize the whole page content of the image."}]}],
      "max_tokens":512,"temperature":0}'
```

## 常驻服务：用脚本管理（推荐）

`scripts/oxr-vllm.sh` 把下面这一整套封装好了：

```bash
./scripts/oxr-vllm.sh check       # 检查 venv / 模型 / 插件 / 头文件 / 端口
./scripts/oxr-vllm.sh install     # 生成单元 + 开机自启 + 等待就绪
./scripts/oxr-vllm.sh status      # systemctl status
./scripts/oxr-vllm.sh logs [行数]  # journalctl
./scripts/oxr-vllm.sh restart | stop | start
./scripts/oxr-vllm.sh run         # 前台直接跑（调试，不经 systemd）
./scripts/oxr-vllm.sh uninstall
```

`install` 会**根据当前环境自动生成**单元文件（模型快照路径、venv 路径、头文件路径
都写成实际值），所以换机器/换缓存目录时不用手改配置。

可用环境变量覆盖：

| 变量 | 默认 | 作用 |
|---|---|---|
| `OXR_VLLM_VENV` | `~/space/python/oxr-vllm` | vLLM 环境 |
| `HF_HOME` | `~/cot/cache/huggingface` | 模型缓存 |
| `OXR_VLLM_PORT` | `8001` | 服务端口 |
| `OXR_PY_HEADERS` | `~/cot/lib/python_3.12.16_*/include/python3.12` | Python 头文件 |
| `OXR_GPU_UTIL` | `0.55` | 显存占用比例 |
| `OXR_MAX_LEN` | `8192` | 上下文长度 |

实测安装到就绪约 **48 秒**。

## 常驻服务：手写单元（参考）

临时验证用 `systemd-run`；要常驻就写用户单元：

```ini
# ~/.config/systemd/user/oxr-vllm.service
[Unit]
Description=OXR vLLM server
After=graphical-session.target

[Service]
Environment=VLLM_PLUGINS=oxr
Environment=CPATH=/home/q/cot/lib/python_3.12.16_linux_amd64_gnu/include/python3.12
Environment=PATH=/home/q/space/python/oxr-vllm/bin:/usr/local/bin:/usr/bin:/bin
Environment=HF_HOME=/home/q/cot/cache/huggingface
Environment=HF_HUB_OFFLINE=1
ExecStart=/home/q/space/python/oxr-vllm/bin/vllm serve <模型目录> \
  --trust-remote-code --served-model-name oxr \
  --chat-template <模型目录>/chat_template.jinja \
  --dtype bfloat16 --gpu-memory-utilization 0.55 --max-model-len 8192 \
  --host 127.0.0.1 --port 8001
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

## 参数调优

| 参数 | 当前 | 说明 |
|---|---|---|
| `--gpu-memory-utilization` | 0.55 | 显存占用比例；16 GB 下 0.55 → 约 8.8 GB |
| `--max-model-len` | 8192 | 上下文上限；调大能处理更长页面，但吃 KV cache |
| `--max-num-seqs` | 默认 | 并发请求数；显存够可以调大提升吞吐 |

如果只跑短页面，`--max-model-len 4096` 能把 KV cache 用在更多并发上。
