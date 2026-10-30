# OXR 本机基准测试

> 2026-10-30 · RTX 4090 Laptop 16 GB · vLLM 后端 · `--workers 3~4`

## 测试数据来源

OXR 官方仓库的 [`docs/frozen-100-pages.json`](https://github.com/SoMarkAI/OXR/blob/main/docs/frozen-100-pages.json)
给出了评测所用页面的确切清单：

```json
{"dataset": "OpenDataLab/OmniDocBench", "files": ["images/PPT_...png", ...]}
```

即 **OmniDocBench** 的 100 页抽样，按类型分布：

| 类型 | 页数 | 说明 |
|---|---|---|
| jiaocaineedrop / jiaocai | 12 | 教材扫描件 |
| newspaper | 10 | 报纸（多栏、密集）|
| notes / yanbaor2 / yanbaopptmerge | 24 | 笔记与报告 |
| PPT | 7 | 幻灯片 |
| docstructbench | 6 | 结构化文档 |
| exam | 6 | 试卷 |
| scihub | 3 | **学术论文（双栏）** |
| book | 5 | 书籍 |
| eastmoney | 4 | 财经报表 |
| color | 4 | 彩色版面 |
| page-<uuid> | 16 | 学术类页面 |

下载（走 hf-mirror）：

```bash
BASE="https://hf-mirror.com/datasets/opendatalab/OmniDocBench/resolve/main/"
curl -L -o page.jpg "${BASE}images/scihub_ijop.12033.pdf_5.jpg"
```

> 文件名里带 `%2F`（原路径含 `/`）的要再转义一次：`%2F` → `%252F`。

---

## 测试 1：PPT 幻灯片（4 页）

样本：`PPT_B14_Claeys...`、`PPT_Keuk Chan Narith...`、`PPT_LEP...`、`PPT_ch7_page_030`。

```
输入: bench.pdf（4 页，150 DPI，后端 vllm，并发 4）
完成：2.3s 总计（含 Python 启动 4.2s）
```

### 质量

| 页面 | 内容 | 结果 |
|---|---|---|
| p1 | 图片为主的幻灯片 | 标题 + 一条要点 + 署名行，均正确 |
| p3 | 政策说明（7 条要点）| 自动转成 markdown 无序列表 |
| p4 | 数学教材（行内公式）| 公式转 LaTeX：`$\mathbf{AS} = S\mathbf{A} = c\mathbf{A}.$`，下标 `$I_{n}$` 正确 |

**结论**：文字、列表、行内公式都能结构化输出。

---

## 测试 2：双栏学术论文（5 页）

样本：3 页 `scihub`（心理学/工程期刊）+ 2 页 `page-<uuid>` 学术页。

```
输入: academic.pdf（5 页，150 DPI，后端 vllm，并发 3，--parse-tables）
  第 1/5 页：3059 字符
  第 2/5 页：864 字符
  第 3/5 页：5495 字符
  第 4/5 页：5823 字符
  第 5/5 页：5865 字符
完成：23.0s 总计
```

### 难点页（第 4 页）

原图：跨栏统计表（13 列 × 12 行）+ 双栏正文 + 右栏小标题。

**① 表格数值零错误**

```html
Mediator | β | Lower | Upper | β | Lower | Upper | β | Lower | Upper | β | Lower | Upper
Integrative Self-Knowledge (ISK) | -.149 | -.195 | -.109 | .118 | .074 | .167 | .165 | .120 | .215
Total | -.326 | -.385 | -.271 | -.147 | -.211 | -.085 | .309 | .255 | .366 | .248 | .194 | .304
```

逐项对照原图：数值、符号、空单元格位置全部正确。这类**多级表头 + 部分单元格为空**的表格
最容易错位，结果没有错。

**② 跨栏阅读顺序正确**

左栏末尾 "…Only this measure," 与右栏开头 "alone or in interaction with Self-Control,"
在输出里无缝衔接，没有出现跨栏错乱。

**③ 结构保留**

表题 `TABLE 2`、表注、右栏小标题 `Limitations and conclusions`、页脚版权行都在正确位置。

---

## 测试 3：扫描件（4 页）

样本：`jiaocaineedrop` 的 4 页教材扫描件（化学、英语词汇表、地理）。

```
输入: scan.pdf（4 页，150 DPI，后端 vllm，并发 4）
  第 1/4 页：1850 字符
  第 2/4 页：750 字符
  第 3/4 页：4913 字符
  第 4/4 页：3217 字符
完成：7.2s 总计
```

### 质量

**正文转录准确**，科学记号被正确转成 LaTeX：

| 原文 | 输出 |
|---|---|
| Δ_O（八面体场分裂能）| `$\Delta_0$` |
| d⁵ 电子 | `$d^5$` |
| 反应式 `C₁₂H₂₂O₁₁(s) + heat → 12 C(s) + 11 H₂O(g)` | `$\mathrm{C_{12}H_{22}O_{11}(s) + heat \rightarrow 12\,C(s) + 11\,H_2O(g)}$` |
| Figure 7 图注（含 Δ_O、d₄-d₇）| 整段转录，下标转 LaTeX |

**词汇表页**（连续条目）按行保留，每条含页码引用：

```
orthodoxy traditional beliefs, especially in religion (p. 469)
ozone layer a thin layer of gas in the upper atmosphere ... (p. 970)
```

### 三个观察

**① 图内标注不会被转录**

第 1 页的轨道图里有 `x²-y²`、`z²`、`xy`、`xz`、`yz`、`Small Δ_O (Weak Field)`、
`Large Δ_O (Strong Field)` 等标注，**全部缺失**——OXR 只转录正文与图注，不 OCR 图形内部的文字。

需要图中文字时，要么单独裁剪该区域请求识别，要么接受缺失。

**② 下标 O 与数字 0 的混淆**

原图 `Δ_O`（octahedral，字母 O）被识别成 `\Delta_0`（数字 0）。
这类**下标的 0/O、1/l、5/S** 在低分辨率区域容易混，属 OCR 共性问题。

**③ 页眉页脚未保留**

`LibreTexts™` 标志、页码 `10.3.6`、页脚 URL 都没有出现在输出里。
对正文提取是好事（去噪），但如果需要保留页码引用则要另做处理。

> 注意：这批样本是"电子版 PDF 转图"，噪点很少。真正的**拍照/复印扫描**
> （倾斜、阴影、污点）还没测，准确率预期会更低。

---

## 已知局限

### 1. `--parse-tables` 对图片型 PDF 无效

PyMuPDF 的 `find_tables()` 依赖**矢量线条与文字层**。扫描件/图片型 PDF 没有这些结构：

```
第 4/5 页：5823 字符，表格 0     ← 期望 "表格 1"，实际为 0
```

| PDF 类型 | 表格识别路径 |
|---|---|
| 原生数字 PDF | PyMuPDF 定位 → 裁剪 → OXR 精细识别 |
| 扫描件 / 图片 PDF | 无法定位 → 由整页 OXR 直接输出表格 |

后者依然可用（上表就是整页识别结果），只是少了"区域重识别"这道增强。

### 2. 耗时随文字密度变化大

| 内容 | 单页耗时 |
|---|---|
| PPT（少字）| 约 0.6 s |
| 学术论文（密排版）| 约 4.6 s |

差异来自生成 token 数：PPT 一页 ~130 字符，论文一页 ~5000 字符。

### 3. 未做定量指标

本文只做了定性核对（抽样对照原图）。没有计算编辑距离、表格 TEDS 等指标——
那需要下载 OmniDocBench 的完整 ground truth（`OmniDocBench.json`），
属于评测工程，不是日常使用必需。

---

## 复现步骤

```bash
# 1. 启动后端（手动启停，不常驻）
./scripts/oxr-vllm.sh start

# 2. 下载样本页（见"测试数据来源"）
# 3. 拼成 PDF
uv run python - <<'PY'
import pymupdf, glob
doc = pymupdf.open()
for f in sorted(glob.glob('/tmp/oxr-bench2/*.jpg') + glob.glob('/tmp/oxr-bench2/*.png')):
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, filename=f, keep_proportion=True)
doc.save('/tmp/oxr-bench2/academic.pdf')
PY

# 4. 解析
uv run python -m oxr_starter.parse_pdf /tmp/oxr-bench2/academic.pdf \
  -o /tmp/oxr-bench2/out.md --backend vllm --workers 3 --parse-tables --json

# 5. 用完停掉
./scripts/oxr-vllm.sh stop
```

---

## 换模型版本时的对比基线

| 指标 | 本次结果（OXR-1.0 @ vLLM 0.25.1）|
|---|---|
| PPT 4 页 | 2.3 s |
| 学术 5 页 | 23.0 s |
| 扫描件 4 页 | 7.2 s |
| 跨栏表格数值准确率 | 抽样 2 行，零错误 |
| 跨栏阅读顺序 | 正确 |
| 公式转 LaTeX | 正确（含上下标）|
| 扫描件正文 OCR | 正确，科学记号转 LaTeX |
| 扫描件图内标注 | **不转录** |
| 显存 | 8.8 GB（utilization 0.55）|
