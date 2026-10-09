"""Per-page router for PDFs: use the text layer when it is good, OCR only the pages that need it.

usage:
  python route_pdf.py FILE.pdf                         # auto mode (default)
  python route_pdf.py FILE.pdf --mode manual --ocr-pages 1,3-5
  python route_pdf.py FILE.pdf --dry-run               # only print the routing table

auto   : the script decides per page (little text / garbled text / large image -> OCR).
manual : YOU list the pages to OCR with --ocr-pages; every other page uses the text layer.

output: route_out/<pdf-stem>/page_N.txt, all_pages.txt, routing.csv
NOTE: OCR uses PP-OCRv5 (Thai). Do not import torch in this process (paddle/torch DLL conflict).
"""
import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(HERE / "paddle_models"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "th_PP-OCRv5_mobile_rec"
GARBLED_RATIO = 0.10  # share of replacement / private-use characters that marks text as garbled


@dataclass(frozen=True)
class PageInfo:
    number: int  # 1-based
    chars: int
    image_ratio: float  # share of the page area covered by images
    garbled_ratio: float
    text: str


@dataclass(frozen=True)
class Decision:
    page: PageInfo
    use_ocr: bool
    reason: str


def parse_pages(spec: str, total: int) -> set[int]:
    pages: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        start, end = int(lo), int(hi or lo)
        pages.update(range(start, end + 1))
    bad = {p for p in pages if not 1 <= p <= total}
    if bad:
        raise SystemExit(f"--ocr-pages has pages outside 1..{total}: {sorted(bad)}")
    return pages


def inspect_page(page: pymupdf.Page, number: int) -> PageInfo:
    text = page.get_text().strip()
    compact = re.sub(r"\s+", "", text)
    weird = sum(1 for c in compact if c == "�" or "" <= c <= "")
    page_area = page.rect.width * page.rect.height or 1.0
    covered = 0.0
    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            covered += (rect & page.rect).get_area()
    return PageInfo(
        number=number,
        chars=len(compact),
        image_ratio=min(covered / page_area, 1.0),
        garbled_ratio=weird / len(compact) if compact else 0.0,
        text=text,
    )


def decide_auto(info: PageInfo, min_chars: int, image_area: float) -> Decision:
    if info.chars < min_chars:
        return Decision(info, True, f"little/no text ({info.chars} chars)")
    if info.garbled_ratio >= GARBLED_RATIO:
        return Decision(info, True, f"garbled text ({info.garbled_ratio:.0%} bad chars)")
    if info.image_ratio >= image_area:
        return Decision(info, True, f"large image ({info.image_ratio:.0%} of page)")
    return Decision(info, False, "text layer looks fine")


def decide_manual(info: PageInfo, ocr_pages: set[int], min_chars: int) -> Decision:
    if info.number in ocr_pages:
        return Decision(info, True, "you chose OCR")
    note = "you chose text layer"
    if info.chars < min_chars:
        note += f" (WARNING: only {info.chars} chars in text layer)"
    return Decision(info, False, note)


def print_table(decisions: list[Decision]) -> None:
    print(f"{'page':>4} {'chars':>6} {'image%':>7}  {'route':<5} reason")
    for d in decisions:
        route = "OCR" if d.use_ocr else "text"
        print(f"{d.page.number:>4} {d.page.chars:>6} {d.page.image_ratio:>7.0%}  {route:<5} {d.reason}")


def ocr_pages(doc: pymupdf.Document, numbers: list[int], out_dir: Path, dpi: int) -> dict[int, str]:
    from paddleocr import PaddleOCR  # imported lazily so --dry-run needs no OCR engine

    ocr = PaddleOCR(
        text_detection_model_name=DET_MODEL,
        text_recognition_model_name=REC_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    texts: dict[int, str] = {}
    for n in numbers:
        png = out_dir / f"page_{n}.png"
        doc[n - 1].get_pixmap(dpi=dpi).save(png)
        result = ocr.predict(str(png))[0]
        texts[n] = "\n".join(result["rec_texts"])
        print(f"OCR page {n}: {len(result['rec_texts'])} lines", flush=True)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--ocr-pages", default="", help="manual mode: pages to OCR, e.g. 1,3-5")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--image-area", type=float, default=0.20, help="auto: image share of page that triggers OCR")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    with pymupdf.open(pdf_path) as doc:
        infos = [inspect_page(p, i) for i, p in enumerate(doc, 1)]
        if args.mode == "manual":
            if not args.ocr_pages:
                raise SystemExit("manual mode needs --ocr-pages (use '--ocr-pages none' for no OCR)")
            chosen = set() if args.ocr_pages == "none" else parse_pages(args.ocr_pages, len(infos))
            decisions = [decide_manual(i, chosen, args.min_chars) for i in infos]
        else:
            decisions = [decide_auto(i, args.min_chars, args.image_area) for i in infos]

        print_table(decisions)
        if args.dry_run:
            return

        out_dir = HERE / "route_out" / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [d.page.number for d in decisions if d.use_ocr]
        ocr_text = ocr_pages(doc, todo, out_dir, args.dpi) if todo else {}

    combined = []
    with open(out_dir / "routing.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "chars", "image_ratio", "route", "reason"])
        for d in decisions:
            n = d.page.number
            text = ocr_text[n] if d.use_ocr else d.page.text
            (out_dir / f"page_{n}.txt").write_text(text, encoding="utf-8")
            combined.append(f"=== page {n} ({'OCR' if d.use_ocr else 'text'}) ===\n{text}")
            writer.writerow([n, d.page.chars, f"{d.page.image_ratio:.2f}", "OCR" if d.use_ocr else "text", d.reason])
    (out_dir / "all_pages.txt").write_text("\n\n".join(combined), encoding="utf-8")
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
