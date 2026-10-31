# oxr-starter

在本机（RTX 4090 Laptop 16 GB / CUDA 13.2）用 **transformers** 跑
[SoMarkAI/OXR-1.0](https://huggingface.co/SoMarkAI/OXR-1.0) 的最小工程。

同一套代码也支持 **Apple Silicon（MPS）**：自动选设备，不需 CUDA 与 vLLM，
实测比上面 4090 的 transformers 基线快约 4 倍。见 [docs/mac.md](docs/mac.md)。

OXR 是文档解析模型（1.379 B 参数，BF16）：图片/PDF 页面 → Markdown 或 JSON，
带版面坐标。

## 环境

| 项 | 值 |
|---|---|
| Python | 3.12（`uv` 管理）|
| torch | 2.15.0+cu132 |
| transformers | 5.22.0 |
| 模型缓存 | `$HF_HOME`，默认 `~/cot/cache/huggingface` |
| 镜像 | `HF_ENDPOINT=https://hf-mirror.com` |
| 显存 | 模型峰值约 2.9 GB |

模型下载（走镜像）：

```bash
hf download SoMarkAI/OXR-1.0
```

## 用法

```bash
uv sync

# 只检查能否加载，报告参数量与显存
uv run python -m oxr_starter.load_model

# 推理：内置测试图
uv run python -m oxr_starter.generate_demo

# 推理：指定图片
uv run python -m oxr_starter.generate_demo ~/some-page.png

# PDF → Markdown（逐页渲染后识别）
uv run python -m oxr_starter.parse_pdf doc.pdf -o doc.md
uv run python -m oxr_starter.parse_pdf doc.pdf --pages 1-3,7 --dpi 150

# 表格区域精细识别（替换整页结果里的 <table>）
uv run python -m oxr_starter.parse_pdf doc.pdf --parse-tables

# 视觉编码上限（默认 1344×1792；MPS 上降到 1200000 可显著提速）
uv run python -m oxr_starter.parse_pdf doc.pdf --max-pixels 1200000

# 走 vLLM 服务（前提：vLLM 已启动，见 docs/vllm.md）
uv run python -m oxr_starter.parse_pdf doc.pdf --backend vllm
uv run python -m oxr_starter.parse_batch ~/papers --backend vllm --parse-tables

# 并发（仅 vllm 后端生效；渲染仍在主线程，识别请求并行）
uv run python -m oxr_starter.parse_pdf doc.pdf --backend vllm --workers 4

# 批量：目录下所有 PDF → Markdown
uv run python -m oxr_starter.parse_batch ~/papers -o ~/papers/md
uv run python -m oxr_starter.parse_batch ~/papers --skip-existing --limit 3
uv run python -m oxr_starter.parse_batch ~/papers --parse-tables   # 表格精细识别

# 版面区域检测 / 表格单独识别
uv run python -m oxr_starter.layout doc.pdf --page 3
uv run python -m oxr_starter.layout doc.pdf --page 3 --parse-tables
```

## Python API

```python
from oxr_starter import parse

doc = parse("paper.pdf", backend="vllm", workers=4, parse_tables=True)
print(doc.markdown)              # 整份 markdown
print(doc.pages[0].markdown)     # 单页文本
print(doc.pages[0].tables)       # [Region(bbox=[...], html="<table>…")]
doc.save("out.md", json_path=True)
```

批量复用后端（避免重复建连接）：

```python
from oxr_starter import OXRClient

with OXRClient(backend="vllm", workers=4) as client:
    for pdf in pdfs:
        doc = client.parse(pdf, parse_tables=True)
        doc.save(f"out/{pdf.stem}.md")
```

数据类型：

| 类型 | 字段 |
|---|---|
| `Document` | `source` / `pages` / `meta` / `images_dir` / `markdown` / `failed_pages` |
| `Page` | `number` / `markdown` / `tables` / `images` / `failed` |
| `Region` | `bbox` / `html` |
| `ImageRef` | `file` / `bbox` / `width` / `height` / `bytes` |

`Document` 还提供 `to_dict()`、`to_json()`、`save(path, json_path=)`。

## vLLM 服务

```bash
./scripts/oxr-vllm.sh install    # 安装单元文件（不启用、不启动）
./scripts/oxr-vllm.sh start      # 启动并等待就绪（约 30-50s）
./scripts/oxr-vllm.sh stop       # 停止，释放显存
./scripts/oxr-vllm.sh status     # 看状态
./scripts/oxr-vllm.sh check      # 环境自检
```

默认不开机自启；需要时 `enable` / `disable` 切换。

详见 [docs/vllm.md](docs/vllm.md)。服务就绪后即可 `--backend vllm` 调用。

## Apple Silicon（MPS）

没有 NVIDIA 卡也能跑全部功能：设备自动选 `cuda → mps → cpu`（MPS 用 `float16`），
vLLM 换成仓库自带的 OpenAI 兼容服务，systemd 换成 launchd。

```bash
uv run python -m oxr_starter.load_model          # 打印设备与显存（3.5 GB / 1.379 B）
uv run python -m oxr_starter.generate_demo       # 单图推理

# 想用 --backend vllm --workers 的统一调用方式：起本机服务（协议与 vLLM 一致）
./scripts/oxr-mac.sh start                       # launchd 托管，等 /health 通过
uv run python -m oxr_starter.parse_pdf doc.pdf --backend vllm --workers 2 --parse-tables
./scripts/oxr-mac.sh stop                        # 释放统一内存
```

实测（M1 Max 32 核 GPU / 64 GB）：小图 1.96 s / 128 tok；A4 整页 500 tok 约 9 s；
真实双栏论文页（自然停止 2221 tok）31.5 s、峰值 6.2 GB、约 71 tok/s。

视觉编码约占固定开销 75%，且耗时随像素数平方增长。MPS 上想更快可以降
`--max-pixels`（默认行为不变）：2.4M→1.2M 时视觉编码 2.8 s→1.0 s、整页 36 s→29 s，
同一页输出相似度 0.992。实测表与注意事项见 [docs/mac.md](docs/mac.md)。

并发注意：MPS 单流已基本吃满 GPU，**同进程多线程无收益**；要并行得用多进程
（`--instances 3` 约 1.38×，每实例多占 3.5–6 GB）。所以 `--workers` 只保证排队不报错。

详见 [docs/mac.md](docs/mac.md)（含成本结构、风险清单与移植细节）。

## 后端与并发

| 参数 | 取值 | 说明 |
|---|---|---|
| `--backend` | `local` / `vllm` | local 进程内加载（3.9 GB 显存）；vllm 走 HTTP 服务 |
| `--base-url` | 默认 `http://127.0.0.1:8001/v1` | vLLM 服务地址 |
| `--workers` | 默认 1 | 并发页数，**仅 vllm 生效**；local 自动忽略 |
| `--device` | `auto` / `cuda` / `mps` / `cpu` | 仅 local 生效；默认 `auto` = cuda→mps→cpu |
| `--dtype` | `auto` / `bf16` / `fp16` / `fp32` | 默认：CUDA→bf16，MPS→fp16，CPU→fp32 |
| `--max-pixels` | 默认 0（模型值 1344×1792）| 视觉编码最大像素数；视觉耗时随 token 数平方增长，MPS 上降档收益明显 |
| `OXR_MAX_PIXELS` | 环境变量 | 与 `--max-pixels` 相同，CLI / Python API / 服务端全局生效 |

并发实现：PyMuPDF 渲染留在主线程（非线程安全），识别请求在线程池里并行发出；
表格裁剪后同样并发识别。每批最多 `workers` 张图驻留内存。

## JSON 结构化输出

`--json` 在生成 markdown 的同时，额外产出同名 `.json`：

```bash
uv run python -m oxr_starter.parse_pdf doc.pdf --backend vllm \
  --parse-tables --extract-images --json
# → doc.md + doc.json
```

结构：

```json
{
  "source": "doc.pdf",
  "pages": [
    {
      "page": 3,
      "markdown": "表1：三种编译引擎的对比…",
      "tables": [{"bbox": [195, 156, 399, 225], "html": "<table>…</table>"}],
      "images": [{"file": "p0003-img01.png", "bbox": [...], "width": 310, "height": 220}],
      "failed": false
    }
  ],
  "meta": {"dpi": 150, "backend": "vllm", "workers": 2,
           "parse_tables": true, "elapsed_sec": 5.6}
}
```

- `markdown`：该页的 markdown（与 `.md` 文件里的内容一致）
- `tables`：表格的区域坐标与独立识别的 HTML
- `images`：提取出的图片文件与坐标（需 `--extract-images`）
- `failed`：该页是否识别失败（便于下游决定是否重试）

断点缓存也会保留表格信息（`pNNNN.json`），续跑时 JSON 里的 bbox 不会丢。

## 图片提取

```bash
# 解析时顺带提取（图片存到 <输出>.images/，markdown 里插引用）
uv run python -m oxr_starter.parse_pdf doc.pdf --extract-images

# 只提取，不解析
uv run python -m oxr_starter.images doc.pdf -o out/images
```

导出的是 PDF 里**原始嵌入的位图**（`doc.extract_image`），不是页面渲染截图，
所以保留原始分辨率和格式；同时生成 `manifest.json`：

```json
{"page": 1, "file": "p0001-img01.png", "bbox": [109.1, 100.0, 390.9, 300.0],
 "width": 310, "height": 220, "bytes": 1024}
```

过滤规则：面积占页面 < 0.5%、或边长 < 24 px 的图片会被跳过（图标、装饰线）。
用 `--min-area` 调整。

> 纯矢量文档（TeX/TikZ 绘图）没有位图，工具会提示"没有可提取的位图"。

## 页码级断点

每页识别成功后会立刻写入 `<输出名>.parts/p0001.md`；中断或部分失败后，
**重跑同一条命令即可续传**，已完成的页不会再识别。

```bash
uv run python -m oxr_starter.parse_pdf doc.pdf --backend vllm --workers 4
# 中断后再跑同一条命令
# 断点续跑: doc.parts（已有 37 页）
```

| 参数 | 作用 |
|---|---|
| `--no-resume` | 忽略缓存，全部重做 |
| `--clean-parts` | 全部成功后删掉 `.parts` 目录 |

未完成的页会在输出 markdown 里留 `<!-- 第 N 页 识别失败，重跑可续传 -->`，
命令以退出码 1 结束，方便脚本判断。

批量模式每个文件各自维护 `<文件名>.parts/`。

## 关于版面分析

OXR 本体**不输出布局框**：用官方 layout 提示词实测，模型仍然返回整页内容。
真正的版面检测是独立的 **D-FINE Stage-2 B2** 模型（`layout_analysis/last.pth`，
12 类），需要官方 38 个文件的 FastAPI 服务（`services/layout_analyze/`）。

本项目的折中方案：用 **PyMuPDF** 的结构信息检测区域（文本块 / 图片 / 表格），
再对**表格区域**裁剪后单独送给模型，用官方表格提示词识别。

`parse_pdf --parse-tables` 会把整页结果里的 `<table>` 块替换成表格区域单独识别的
结果（按 PyMuPDF 检测到的表格顺序匹配）。代价是每张表多一次生成：实测单页
28s → 55s。

实测效果（`ctex-demo.pdf` 第 3 页，4 行 × 3 列表格）：

```html
<table><tr><td>引擎</td><td>适用场景</td><td>中文支持</td></tr>
<tr><td>pdflatex</td><td>纯英文、老模板</td><td>弱</td></tr>
<tr><td>xelatex</td><td>中文、字体自由</td><td>强</td></tr>
<tr><td>lualatex</td><td>复杂脚本、Lua 扩展</td><td>强</td></tr></table>
```

## 性能（本机实测）

| 场景 | transformers | vLLM |
|---|---|---|
| 单张小图（900×620）| 6.9 s | — |
| PDF A4 单页 @150 DPI | 27.6 s | **1.3–2.1 s** |
| 4 页 PDF 全文 | 112.8 s | **5.5 s** |
| 4 页 PDF（并发 `--workers 4`）| — | **2.2 s** |
| 单页 + 表格替换 | 55.2 s | **2.1 s** |

vLLM 约快 20 倍，显存占用 8.8 GB（`--gpu-memory-utilization 0.55`）。
搭建方法见 [docs/vllm.md](docs/vllm.md)。

PDF 每页约 28 秒，主要花在视觉 token 上的自回归生成。想加速可以：
降低 `--dpi`、减小 `--max-tokens`，或改用官方 vLLM 路径。

## 实现要点

1. **不要用 `AutoProcessor`**：仓库的 `preprocessor_config.json` 指向普通
   `SiglipProcessor`，缺视觉占位符逻辑。用模型自带的
   `model.preprocess_inputs(messages)`。
2. `generate()` 的入参名是 **`inputs`**（不是 `input_ids`），并且需要显式传
   `pixel_values` / `grid_thws`。
3. `generate()` **只返回新生成的 token**，直接 decode `out[0]` 即可。
4. 提示词来自官方仓库 `oxr/model/prompts.py`：
   - `Recognize the whole page content of the image.`（整页）
   - `Recognize the content of the image.`（纯文本）
   - `Recognize the table of the image into html format.`（表格）

## 局限

官方生产路径是 **vLLM + 三服务**（解析 API / 模型服务 / 版面分析），
完整流水线在 [SoMarkAI/OXR](https://github.com/SoMarkAI/OXR) 仓库的 `services/`。
本项目只做**单图/单页**的本地验证，没有 PDF 分页与版面分析模型。

## 测试

```bash
uv sync                 # 默认安装 dev 组（pytest / ruff）
uv run pytest           # 快速用例：不加载模型、不联网，约 3 s
uv run ruff check tests src
```

用例全部用假后端（`tests/helpers.py` 里的 `FakeBackend` / `FakeRunner`）替换推理，
PDF 也由 PyMuPDF 现场合成，所以没下模型、没有 GPU 的机器也能跑：

| 文件 | 覆盖 |
|---|---|
| `test_devices.py` | cuda→mps→cpu 选择、dtype 解析、显存口径、`PeakTracker`（monkeypatch torch）|
| `test_server.py` | `/health`、`/v1/models`、`/v1/chat/completions`、400/404/500/504、data URI、`ThreadRunner` 串行/超时、`ProcessRunner` 队列统计 |
| `test_parse_pdf.py` | 页码解析、渲染/裁剪、表格替换、并发、失败隔离、断点续跑、CLI 的 md/json/图片输出 |
| `test_parse_batch.py` | 递归收集、`--skip-existing`、`--limit`、`--pages`、单文件失败计数 |
| `test_api.py` | `Document` 的 markdown/to_dict/save、`OXRClient` 复用、resume、图片提取 |
| `test_backend.py` | 本地模型在 CUDA/MPS/CPU 上的放置（假 transformers）、vLLM 请求体、连接失败 |
| `test_images_layout.py` | 位图提取与 manifest、区域检测、`layout` CLI |
| `test_demo_load.py` | `generate_demo` / `load_model` 的 CLI 输出 |
| `test_slow_perf.py` | MPS 性能回归：2.4M→1.2M 的视觉加速比、解码 tok/s、单次视觉耗时上限 |

真实模型的冒烟测试默认跳过，需要显式开启：

```bash
OXR_SLOW=1 uv run pytest -m slow
# 可选：OXR_DEVICE=mps OXR_DTYPE=fp16 OXR_MAX_SECONDS=120 ...
```

它会在合成页面上跑一次完整的 `OXRClient.parse()`，断言解析成功且耗时不超过
`OXR_MAX_SECONDS`（默认 600 s），用来兜住 transformers 升级后的明显性能回退。

MPS 性能回归单独跑（会打印 vision / decode 实测值）：

```bash
OXR_SLOW=1 uv run pytest tests/test_slow_perf.py -s
# 阈值：OXR_VISION_MAX_SECONDS（默认 8）/ OXR_VISION_SPEEDUP_MIN（默认 1.3）
#       / OXR_MIN_TOK_S（默认 20）
```

## 目录

```
src/oxr_starter/
├── devices.py          # 设备自适应：cuda / mps / cpu + dtype + 显存报告
├── load_model.py       # 加载 + 显存报告
├── generate_demo.py    # 单图推理
├── parse_pdf.py        # PDF 逐页解析 → Markdown
├── parse_batch.py      # 目录批量处理
├── api.py              # Python API：parse() / OXRClient / Document
├── layout.py           # 区域检测 + 表格精细识别
├── images.py           # 位图提取 + manifest
├── backend.py          # 后端抽象：local(transformers) / vllm
└── server.py           # OpenAI 兼容服务（MPS 上的 vLLM 替代品）

scripts/
├── oxr-vllm.sh         # Linux：systemd 起 vLLM
└── oxr-mac.sh          # macOS：launchd 起 oxr_starter.server

tests/                  # pytest 用例（假后端 + 合成 PDF，不加载模型）
```
