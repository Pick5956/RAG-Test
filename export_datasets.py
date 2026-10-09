"""Turn the raw datasets in dataset/ into files you can upload on the web page (no embedding step here: the app does that).

dataset/
  nitibench-statute/   Thai statutes (NitiBench-Statute, VISAI-AI)
    raw/               as downloaded
    upload/            one .txt per law (drag the folder onto the upload tab)
  beir-scifact/        scientific abstracts (BEIR / SciFact)
    raw/               as downloaded
    upload-sample/     a few dozen abstracts + questions.txt (the answers are in the listed files)
    scifact-all.zip    every abstract in one zip (use this for the "lots of files" case)

usage: python export_datasets.py [nitibench|scifact ...]   (default: both; works offline from dataset/*/raw)
"""
import random
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent / "dataset"
SAMPLE_QUESTIONS, SAMPLE_EXTRA = 15, 10  # SciFact sample: questions to answer, plus unrelated abstracts as distractors


def slug(text: str, limit: int = 60) -> str:
    return re.sub(r"[^\w฀-๿]+", "-", text.lower()).strip("-")[:limit] or "doc"


def export_nitibench() -> None:
    base = ROOT / "nitibench-statute"
    frame = pd.concat(pd.read_parquet(f) for f in sorted((base / "raw").rglob("*.parquet")))
    out = base / "upload"
    out.mkdir(exist_ok=True)
    for (code, name), rows in frame.groupby(["law_code", "law_name"], sort=False):
        # section_content already starts with the law name and the section number; blank lines keep sections apart
        text = "\n\n".join(str(t).strip() for t in rows["section_content"] if str(t).strip())
        (out / f"{slug(name)}.txt").write_text(text + "\n", encoding="utf-8")
    print(f"nitibench: {frame['law_code'].nunique()} laws, {len(frame)} sections -> {out}")


def read_scifact(raw: Path) -> tuple[pd.DataFrame, dict[str, str], dict[str, list[str]]]:
    corpus = pd.read_parquet(next(raw.rglob("corpus*.parquet")))
    queries = pd.read_parquet(next(raw.rglob("queries*.parquet")))
    qrels = pd.read_csv(raw / "qrels-test.tsv", sep="\t")
    relevant: dict[str, list[str]] = {}
    for q, d, score in qrels[["query-id", "corpus-id", "score"]].itertuples(index=False):
        if score > 0:
            relevant.setdefault(str(q), []).append(str(d))
    return corpus, {str(i): t for i, t in zip(queries["_id"], queries["text"])}, relevant


def doc_text(row: dict) -> str:
    return f"{row['title']}\n\n{row['text']}\n" if row["title"] else f"{row['text']}\n"


def export_scifact() -> None:
    base = ROOT / "beir-scifact"
    corpus, queries, relevant = read_scifact(base / "raw")
    rows = {str(i): {"title": t, "text": x} for i, t, x in zip(corpus["_id"], corpus["title"], corpus["text"])}
    file_of = {i: f"scifact-{i}-{slug(r['title'], 50)}.txt" for i, r in rows.items()}

    with zipfile.ZipFile(base / "scifact-all.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for i, r in rows.items():
            z.writestr(file_of[i], doc_text(r))

    picked = [q for q in relevant if q in queries and all(d in rows for d in relevant[q])][:SAMPLE_QUESTIONS]
    chosen = dict.fromkeys(d for q in picked for d in relevant[q])
    rest = [i for i in rows if i not in chosen]
    chosen.update(dict.fromkeys(random.Random(0).sample(rest, min(SAMPLE_EXTRA, len(rest)))))
    sample = base / "upload-sample"
    sample.mkdir(exist_ok=True)
    for i in chosen:
        (sample / file_of[i]).write_text(doc_text(rows[i]), encoding="utf-8")
    lines = ["Questions to ask after uploading upload-sample/ (the file with the answer is in brackets):", ""]
    for q in picked:
        lines += [f"- {queries[q]}", "    [" + ", ".join(file_of[d] for d in relevant[q]) + "]"]
    (base / "questions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"scifact: {len(rows)} abstracts -> scifact-all.zip | {len(chosen)} files in upload-sample/ | {len(picked)} questions")


if __name__ == "__main__":
    wanted = set(sys.argv[1:]) or {"nitibench", "scifact"}
    if "nitibench" in wanted:
        export_nitibench()
    if "scifact" in wanted:
        export_scifact()
