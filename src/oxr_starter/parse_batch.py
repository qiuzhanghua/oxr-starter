"""批量把目录下的 PDF 转成 Markdown。

用法：
    uv run python -m oxr_starter.parse_batch ~/papers -o ~/papers/md
    uv run python -m oxr_starter.parse_batch ~/papers --pattern '*.pdf' --skip-existing
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pymupdf
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from .parse_pdf import MODEL_ID, PROMPT_WHOLE_PAGE, replace_tables, to_cuda

DEFAULT_PATTERN = "*.pdf"


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def parse_page(model, image: Image.Image, max_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_WHOLE_PAGE},
        ],
    }]
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages)
    with torch.inference_mode():
        out = model.generate(
            inputs=input_ids.to("cuda"),
            pixel_values=to_cuda(pixel_values),
            grid_thws=to_cuda(grid_thws),
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    return model.text_tokenizer.decode(out[0], skip_special_tokens=True)


def collect_pdfs(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description="批量 PDF → Markdown（OXR）")
    ap.add_argument("input", type=Path, help="输入目录")
    ap.add_argument("-o", "--output", type=Path, help="输出目录（默认与输入同级 md/）")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="文件匹配，默认 *.pdf")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--pages", type=str, default="", help="每份只处理指定页，如 1-3")
    ap.add_argument("--skip-existing", action="store_true", help="跳过已存在且不为空的 .md")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几个文件（调试用）")
    ap.add_argument("--parse-tables", action="store_true",
                    help="表格区域单独精细识别（每张表多一次生成，较慢）")
    args = ap.parse_args()

    root = args.input.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"不是目录: {root}")
    out_dir = (args.output or root / "md").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(root, args.pattern)
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit(f"在 {root} 下没找到 {args.pattern}")

    page_filter = set()
    if args.pages:
        for part in args.pages.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                page_filter.update(range(int(a), int(b) + 1))
            else:
                page_filter.add(int(part))

    print(f"目录: {root}")
    print(f"输出: {out_dir}")
    print(f"找到 {len(pdfs)} 个文件")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="cuda:0", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"模型就绪（{time.time() - t0:.1f}s）\n")

    done, skipped, failed = 0, 0, 0
    for n, pdf in enumerate(pdfs, 1):
        rel = pdf.relative_to(root)
        dst = out_dir / rel.with_suffix(".md")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and dst.exists() and dst.stat().st_size > 0:
            print(f"[{n}/{len(pdfs)}] 跳过（已存在）: {rel}")
            skipped += 1
            continue

        try:
            doc = pymupdf.open(pdf)
            indices = [i for i in range(1, doc.page_count + 1)
                       if not page_filter or i in page_filter]
            parts = [f"# {pdf.stem}\n"]
            t1 = time.time()
            tables_total = 0
            for i in indices:
                page = doc[i - 1]
                img = render_page(page, args.dpi)
                text = parse_page(model, img, args.max_tokens).strip()
                if args.parse_tables:
                    text, n = replace_tables(model, page, text, args.dpi, args.max_tokens)
                    tables_total += n
                parts.append(f"\n<!-- 第 {i} 页 -->\n\n{text}\n")
            dst.write_text("".join(parts), encoding="utf-8")
            dt = time.time() - t1
            extra = f"，表格 {tables_total}" if args.parse_tables else ""
            print(f"[{n}/{len(pdfs)}] {rel} → {len(indices)} 页，{dt:.1f}s{extra}")
            done += 1
        except Exception as e:  # 单个失败不影响其余文件
            print(f"[{n}/{len(pdfs)}] 失败: {rel}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n完成：成功 {done}，跳过 {skipped}，失败 {failed}，总计 {time.time() - t0:.1f}s")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
