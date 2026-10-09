"""Text extraction for office-style documents, using only the standard library.

Each reader returns a list of segments: {"label": str | None, "text": str}
  label  = how a chunk should be cited ("สไลด์ 3", "ชีต Sheet1"); None -> numbered "ส่วนที่ k" later.
PDFs and images are handled by app.py (they may need OCR).
"""
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

TEXT_EXT = {".txt", ".md", ".csv", ".tsv", ".json", ".log", ".xml", ".yaml", ".yml"}
HTML_EXT = {".html", ".htm"}
OFFICE_EXT = {".docx", ".pptx", ".xlsx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXT = {".pdf"}
SUPPORTED = TEXT_EXT | HTML_EXT | OFFICE_EXT | IMAGE_EXT | PDF_EXT
LEGACY_HINT = {
    ".doc": "ไฟล์ .doc แบบเก่า ไม่รองรับ · เปิดแล้วบันทึกเป็น .docx ก่อน",
    ".ppt": "ไฟล์ .ppt แบบเก่า ไม่รองรับ · เปิดแล้วบันทึกเป็น .pptx ก่อน",
    ".xls": "ไฟล์ .xls แบบเก่า ไม่รองรับ · เปิดแล้วบันทึกเป็น .xlsx ก่อน",
}
MAX_UNZIPPED = 300 * 1024 * 1024  # guard against zip bombs
MAX_TEXT = 3_000_000  # characters kept per document

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def kind_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        return "pdf"
    if ext in IMAGE_EXT:
        return "image"
    return "office" if ext in OFFICE_EXT else "text"


def decode_bytes(raw: bytes) -> str:
    """UTF-8 first, then UTF-16 (with BOM), then Thai Windows code page, finally Latin-1."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "cp874"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


class _HtmlText(HTMLParser):
    SKIP = {"script", "style", "head", "noscript"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def clean(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()[:MAX_TEXT]


def _check_zip(zf: zipfile.ZipFile) -> None:
    if sum(i.file_size for i in zf.infolist()) > MAX_UNZIPPED:
        raise ValueError("ไฟล์ใหญ่ผิดปกติหลังแตกไฟล์ (อาจเป็นไฟล์อันตราย)")


def read_plain(path: Path) -> list[dict]:
    text = decode_bytes(path.read_bytes())
    if path.suffix.lower() in HTML_EXT:
        parser = _HtmlText()
        parser.feed(text)
        text = "".join(parser.parts)
    return [{"label": None, "text": clean(text)}]


def read_docx(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        _check_zip(zf)
        root = ET.fromstring(zf.read("word/document.xml"))
    paragraphs = ["".join(t.text or "" for t in p.iter(W + "t")) for p in root.iter(W + "p")]
    return [{"label": None, "text": clean("\n".join(paragraphs))}]


def read_pptx(path: Path) -> list[dict]:
    segments = []
    with zipfile.ZipFile(path) as zf:
        _check_zip(zf)
        slides = sorted(
            (n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)),
        )
        for number, name in enumerate(slides, 1):
            root = ET.fromstring(zf.read(name))
            lines = ["".join(t.text or "" for t in p.iter(A + "t")) for p in root.iter(A + "p")]
            text = clean("\n".join(lines))
            if text:
                segments.append({"label": f"สไลด์ {number}", "text": text})
    return segments


def read_xlsx(path: Path) -> list[dict]:
    segments = []
    with zipfile.ZipFile(path) as zf:
        _check_zip(zf)
        names = set(zf.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).iter(S + "si"):
                shared.append("".join(t.text or "" for t in si.iter(S + "t")))
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet_names = [s.get("name", "") for s in wb.iter(S + "sheet")]
        for number, sheet_name in enumerate(sheet_names, 1):
            part = f"xl/worksheets/sheet{number}.xml"
            if part not in names:
                continue
            rows = []
            for row in ET.fromstring(zf.read(part)).iter(S + "row"):
                cells = []
                for c in row.iter(S + "c"):
                    v = c.find(S + "v")
                    if c.get("t") == "s" and v is not None and v.text and v.text.isdigit() and int(v.text) < len(shared):
                        cells.append(shared[int(v.text)])
                    elif c.get("t") == "inlineStr":
                        cells.append("".join(t.text or "" for t in c.iter(S + "t")))
                    elif v is not None and v.text:
                        cells.append(v.text)
                if cells:
                    rows.append(" | ".join(cells))
            text = clean("\n".join(rows))
            if text:
                segments.append({"label": f"ชีต {sheet_name}", "text": text})
    return segments


def read_office(path: Path) -> list[dict]:
    reader = {".docx": read_docx, ".pptx": read_pptx, ".xlsx": read_xlsx}[path.suffix.lower()]
    return reader(path)


def read_text_like(path: Path) -> list[dict]:
    return read_plain(path)
