"""Test tool: chunk a folder of documents EXACTLY the way the web upload does -> test_corpus/corpus.jsonl

It does not re-implement anything: the reader (extractors.py), the splitter and the row format are the app's own
(app.make_rows), and each file is named like the web names a dropped folder (<folder name>/<relative path>),
so chunk texts and ids match what the web produces for the same folder.

usage: python build_corpus.py [folder=dataset/nitibench-statute/upload]
Reads txt md csv json html docx pptx xlsx (PDF and images need OCR: upload those on the web page).
Output goes to test_corpus/, never to the app's own data files.
"""
import json
import sys
from pathlib import Path

import app
import extractors as ex

HERE = Path(__file__).parent
OUT = HERE / "test_corpus"
READABLE = ex.TEXT_EXT | ex.HTML_EXT | ex.OFFICE_EXT


def web_name(folder: Path, file: Path) -> str:
    """Name the web gives a file inside a dropped folder: '<folder>/<sub>/<file>' passed through safe_relpath."""
    return app.safe_relpath(f"{folder.name}/{file.relative_to(folder).as_posix()}")


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "dataset" / "nitibench-statute" / "upload"
    files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in READABLE)
    if not files:
        raise SystemExit(f"no readable files in {folder}")
    OUT.mkdir(exist_ok=True)
    total = 0
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as out:
        for file in files:
            segments = ex.read_office(file) if file.suffix.lower() in ex.OFFICE_EXT else ex.read_text_like(file)
            for row in app.make_rows(web_name(folder, file), [s for s in segments if s["text"].strip()]):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    print(f"{len(files)} files -> {total} chunks in {OUT / 'corpus.jsonl'}")


if __name__ == "__main__":
    main()
