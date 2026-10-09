"""Test tool: embed test_corpus/corpus.jsonl with the app's own embedding code -> test_corpus/embeddings.npy

Uses app.INDEX.embed_texts (same model load, fp16, truncation, left padding, last-token pooling, batch of 16),
called once per document like the web does for a single uploaded file, so the vectors are the ones the web would compute.
Caveat: the web groups documents differently for a zip (several files in one batch); batch composition changes padding,
so vectors can differ in the last few decimals (fp16) - not in meaning.

usage: python embed_corpus.py            embed test_corpus/corpus.jsonl
       python embed_corpus.py --check    compare with what the web stored (user_docs.jsonl / user_vecs.npy): same ids,
                                         same text, how far apart the vectors are
"""
import json
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

import app

HERE = Path(__file__).parent
OUT = HERE / "test_corpus"


def load_rows() -> list[dict]:
    path = OUT / "corpus.jsonl"
    if not path.exists():
        raise SystemExit("test_corpus/corpus.jsonl not found - run build_corpus.py first")
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def embed(rows: list[dict]) -> np.ndarray:
    app.INDEX.keep_resident = True  # one model load for the whole run
    parts = []
    for n, (_, group) in enumerate(groupby(rows, key=lambda r: r["doc"]), 1):
        parts.append(app.INDEX.embed_texts([r["text"] for r in group]))  # one call per document, like the web
        if n % 10 == 0:
            print(f"  {n} documents", flush=True)
    return np.concatenate(parts)


def check(rows: list[dict], vecs: np.ndarray) -> None:
    web_rows = [json.loads(line) for line in open(app.USER_DOCS, encoding="utf-8")]
    web_vecs = np.load(app.USER_VECS)
    web = {r["id"]: (r, v) for r, v in zip(web_rows, web_vecs)}
    both = [(r, v) for r, v in zip(rows, vecs) if r["id"] in web]
    print(f"test chunks: {len(rows)} | web chunks: {len(web)} | same id in both: {len(both)}")
    if not both:
        raise SystemExit("nothing to compare (upload the same folder on the web page first)")
    text_diff = sum(1 for r, _ in both if r["text"] != web[r["id"]][0]["text"])
    cos = np.array([float(v @ web[r["id"]][1]) for r, v in both])
    print(f"text different: {text_diff} | cosine min/mean: {cos.min():.6f} / {cos.mean():.6f}")
    print("OK: same chunks, same vectors (fp16 noise only)" if not text_diff and cos.min() > 0.999
          else "DIFFERENT: look at the numbers above")


def main() -> None:
    rows = load_rows()
    if "--check" in sys.argv:
        vecs_path = OUT / "embeddings.npy"
        if not vecs_path.exists():
            raise SystemExit("run embed_corpus.py first")
        return check(rows, np.load(vecs_path))
    vecs = embed(rows)
    np.save(OUT / "embeddings.npy", vecs)
    print(f"{len(rows)} chunks -> {OUT / 'embeddings.npy'} {vecs.shape}")


if __name__ == "__main__":
    main()
