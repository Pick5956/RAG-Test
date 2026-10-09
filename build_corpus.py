"""Build the Thai-law test corpus from NitiBench-Statute -> corpus.jsonl

NitiBench-Statute: 5,127 sections of 35 Thai statutes (mainly corporate/commercial law), MIT license.
Each dataset row is one section; `section_content` already starts with the law name and section number.

usage: python build_corpus.py
"""
import json
import re
from pathlib import Path

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).parent
DATASET_IDS = ("vistec-AI/nitibench-statute", "VISAI-AI/nitibench-statute")  # card and URL disagree; try both
SPLIT = "ccl"

CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 100  # characters shared between neighbouring chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # paragraph -> line -> space -> character (Thai has no spaces between words)
    separators=["\n\n", "\n", " ", ""],
)


def load_sections():
    last_error = None
    for dataset_id in DATASET_IDS:
        try:
            return load_dataset(dataset_id, split=SPLIT)
        except Exception as exc:  # noqa: BLE001 - try the next repo id
            last_error = exc
            print(f"could not load {dataset_id}: {type(exc).__name__}", flush=True)
    raise SystemExit(f"dataset not available: {last_error}")


def clean(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def chunk_section(row: dict) -> list[dict]:
    text = clean(row["section_content"])
    if not text:
        return []
    law, number = row["law_name"], row["section_num"]
    pieces = splitter.split_text(text)
    rows = []
    for k, piece in enumerate(pieces):
        # the first piece already carries the law name and section; give later pieces the same context
        body = piece if k == 0 else f"{law} มาตรา {number}: {piece}"
        rows.append(
            {
                "id": f"law-{row['law_code']}-{number}-c{k}",
                "src": "law",
                "law_name": law,
                "section_num": number,
                "text": body,
            }
        )
    return rows


def main() -> None:
    sections = load_sections()
    rows = [chunk for row in sections for chunk in chunk_section(row)]
    with open(HERE / "corpus.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    laws = {r["law_name"] for r in rows}
    lengths = sorted(len(r["text"]) for r in rows)
    print(f"sections: {len(sections)} | chunks: {len(rows)} | laws: {len(laws)}")
    print(f"chunk length min/median/max: {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}")


if __name__ == "__main__":
    main()
