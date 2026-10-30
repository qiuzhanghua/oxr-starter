# oxr-starter

在本机（RTX 4090 Laptop 16 GB / CUDA 13.2）用 **transformers** 跑
[SoMarkAI/OXR-1.0](https://huggingface.co/SoMarkAI/OXR-1.0) 的最小工程。

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

## 后端与并发

| 参数 | 取值 | 说明 |
|---|---|---|
| `--backend` | `local` / `vllm` | local 进程内加载（3.9 GB 显存）；vllm 走 HTTP 服务 |
| `--base-url` | 默认 `http://127.0.0.1:8001/v1` | vLLM 服务地址 |
| `--workers` | 默认 1 | 并发页数，**仅 vllm 生效**；local 自动忽略 |

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

## 目录

```
src/oxr_starter/
├── load_model.py       # 加载 + 显存报告
├── generate_demo.py    # 单图推理
├── parse_pdf.py        # PDF 逐页解析 → Markdown
├── parse_batch.py      # 目录批量处理
├── layout.py           # 区域检测 + 表格精细识别
├── images.py           # 位图提取 + manifest
└── backend.py          # 后端抽象：local(transformers) / vllm
```
