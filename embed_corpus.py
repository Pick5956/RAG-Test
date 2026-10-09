"""Embed corpus.jsonl with Qwen3-Embedding-0.6B -> embeddings.npy, then run a few sample searches.

usage: python embed_corpus.py [batch=8] [limit]
  batch  documents per forward pass (16 is fine on a 6 GB GPU; lower it if you hit out-of-memory)
  limit  only embed the first N chunks (quick trial; the app needs the full file, so run it once without a limit)
Uses the GPU when CUDA is available (fp16), otherwise the CPU (fp32, much slower).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).parent
MODEL_DIR = HERE / "Qwen3-Embedding-0.6B"
MAX_LEN = 512
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
TASK = "Given a web search query, retrieve relevant passages that answer the query"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUERIES = [
    "การจดทะเบียนบริษัทมหาชนจำกัดต้องทำอย่างไร",
    "หลักเกณฑ์การจัดซื้อจัดจ้างภาครัฐ",
    "พนักงานรัฐวิสาหกิจถูกเลิกจ้างได้เมื่อใด",
    "What is a public limited company?",
]


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL_DIR, dtype=dtype).to(DEVICE).eval()
    return tok, model


@torch.inference_mode()
def embed(tok, model, texts):
    batch = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    out = model(**batch).last_hidden_state[:, -1]  # left padding -> last token
    return F.normalize(out.float(), dim=-1).cpu().numpy()


def main():
    if not (HERE / "corpus.jsonl").exists():
        raise SystemExit("corpus.jsonl not found - run build_corpus.py first")
    docs = [json.loads(line) for line in open(HERE / "corpus.jsonl", encoding="utf-8")]
    if LIMIT:
        docs = docs[:LIMIT]
    order = np.argsort([len(d["text"]) for d in docs])  # sort by length so padding waste is small

    t0 = time.time()
    tok, model = load_model()
    print(f"device: {DEVICE} | model loaded in {time.time() - t0:.1f}s", flush=True)

    vecs = np.zeros((len(docs), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(docs), BATCH):
        idx = order[i : i + BATCH]
        vecs[idx] = embed(tok, model, [docs[j]["text"] + tok.eos_token for j in idx])
        if (i // BATCH) % 25 == 0:
            print(f"  {i + len(idx)}/{len(docs)}  {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    vram = f", peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if DEVICE == "cuda" else ""
    print(f"embedded {len(docs)} chunks in {dt:.1f}s = {len(docs) / dt:.1f}/s{vram}, dim={vecs.shape[1]}")
    np.save(HERE / "embeddings.npy", vecs)

    for q in QUERIES:
        qv = embed(tok, model, [f"Instruct: {TASK}\nQuery:{q}" + tok.eos_token])[0]
        top = np.argsort(-(vecs @ qv))[:3]
        print(f"\nQ: {q}")
        for j in top:
            print(f"  {vecs[j] @ qv:.3f} {docs[j]['text'][:90]}")


if __name__ == "__main__":
    main()
