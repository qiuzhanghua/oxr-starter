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

# 批量：目录下所有 PDF → Markdown
uv run python -m oxr_starter.parse_batch ~/papers -o ~/papers/md
uv run python -m oxr_starter.parse_batch ~/papers --skip-existing --limit 3

# 版面区域检测 / 表格单独识别
uv run python -m oxr_starter.layout doc.pdf --page 3
uv run python -m oxr_starter.layout doc.pdf --page 3 --parse-tables
```

## 关于版面分析

OXR 本体**不输出布局框**：用官方 layout 提示词实测，模型仍然返回整页内容。
真正的版面检测是独立的 **D-FINE Stage-2 B2** 模型（`layout_analysis/last.pth`，
12 类），需要官方 38 个文件的 FastAPI 服务（`services/layout_analyze/`）。

本项目的折中方案：用 **PyMuPDF** 的结构信息检测区域（文本块 / 图片 / 表格），
再对**表格区域**裁剪后单独送给模型，用官方表格提示词识别。

实测效果（`ctex-demo.pdf` 第 3 页，4 行 × 3 列表格）：

```html
<table><tr><td>引擎</td><td>适用场景</td><td>中文支持</td></tr>
<tr><td>pdflatex</td><td>纯英文、老模板</td><td>弱</td></tr>
<tr><td>xelatex</td><td>中文、字体自由</td><td>强</td></tr>
<tr><td>lualatex</td><td>复杂脚本、Lua 扩展</td><td>强</td></tr></table>
```

## 性能（本机实测）

| 场景 | 耗时 | 显存峰值 |
|---|---|---|
| 单张小图（900×620）| 6.9 s（512 tok，74.7 tok/s）| 2.91 GB |
| PDF A4 单页 @150 DPI（1241×1754）| 约 28 s | 3.93 GB |

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
└── layout.py           # 区域检测 + 表格精细识别
```
