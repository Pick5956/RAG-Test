"""RAG lab: document upload + semantic search + ask-AI in one page.   usage: python app.py   ->  http://127.0.0.1:8765

search : Qwen3-Embedding over your uploaded documents (an optional base corpus in corpus.jsonl / embeddings.npy is also read)
upload : files or whole folders (pdf txt md csv json html docx pptx xlsx + images) are extracted (OCR when needed),
         chunked, embedded and added to the map as new points - one background queue, one file at a time.
OCR runs in a child process (ocr_worker.py). Do not import paddle in this process.

env (optional): APP_PORT, APP_USER_DIR (where uploads and the user library live), OCR_PYTHON, OCR_DEVICE
"""
import gc
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

import extractors as ex
import llm_client as llm
import route_pdf as rp

HERE = Path(__file__).parent
USER_DIR = Path(os.environ.get("APP_USER_DIR", HERE))
CORPUS, EMBEDDINGS = HERE / "corpus.jsonl", HERE / "embeddings.npy"  # optional base corpus; normally absent: everything comes from uploads
USER_DOCS, USER_VECS = USER_DIR / "user_docs.jsonl", USER_DIR / "user_vecs.npy"  # uploaded documents (this app)
UPLOADS, OUT_ROOT, PAGE_HTML = USER_DIR / "uploads", USER_DIR / "route_out", HERE / "app_page.html"
MODEL_DIR = HERE / "Qwen3-Embedding-0.6B"
HOST, PORT = "127.0.0.1", int(os.environ.get("APP_PORT", 8765))
TASK = "Given a web search query, retrieve relevant passages that answer the query"
TOP_K = 10
EMBED_DIM = 1024  # Qwen3-Embedding-0.6B; only used while there is no base corpus to read the dimension from
ZONE_POINT_CAP = 8000  # most points the 3D zone view receives for one zone (a random sample beyond that)
STATIC_DIR = HERE / "static"
STATIC_FILES = {"globe.js": "text/javascript; charset=utf-8", "three.module.min.js": "text/javascript; charset=utf-8"}
ZONE_MAX_GROUPS, ZONE_MIN_POINTS = 6, 60  # no base corpus and few groups: colour by content zone instead of by file
EMPTY_BOX = {"x0": -1.0, "x1": 1.0, "y0": -1.0, "y1": 1.0}
MIN_CHARS, IMAGE_AREA = 50, 0.20  # PDF routing thresholds (auto)
MAX_UPLOAD, MAX_PAGES, MAX_CHUNKS = 100 * 1024 * 1024, 300, 5000
# Many files: dragging thousands of files is refused in the page; they go in one .zip instead (one request, one job).
MAX_DROP_FILES = 300  # files per drop / waiting in the queue at once (enforced by the page)
MAX_ZIP, MAX_ZIP_UNPACKED, MAX_ZIP_ENTRIES = 500 * 1024 * 1024, 2 * 1024**3, 20000
ZIP_GROUP = 250  # documents embedded and saved together while a zip is being processed
ZIP_READABLE = ex.TEXT_EXT | ex.HTML_EXT | ex.OFFICE_EXT  # PDF and images inside a zip are skipped (they need OCR: upload them directly)
ZIP_HIDDEN = re.compile(r"(^|/)(\.[^/]*|~\$[^/]*|thumbs\.db|desktop\.ini|__MACOSX)(/|$)", re.I)
RESULT_MARK = "@@RESULT@@ "
# GPU policy. Normal mode: the chat LLM waits in VRAM and the embedding model is loaded only for the moment a query
# needs it (~2-3 s). While documents are being ingested the roles swap: LLM unloaded, embedding model resident.
# Set EMBED_RESIDENT=1 to keep the embedding model on the GPU all the time (fast search, ~1.2 GB more VRAM).
EMBED_RESIDENT = os.environ.get("EMBED_RESIDENT", "0") == "1"
MAX_SIDE, RENDER_DPI = 4000, 200  # PaddleOCR downsizes anything above MAX_SIDE anyway


def venv_python(folder: str) -> Path | None:
    """Interpreter of a venv living next to this file (Windows or POSIX layout), if it exists."""
    for rel in ("Scripts/python.exe", "bin/python"):
        candidate = HERE / folder / rel
        if candidate.exists():
            return candidate
    return None


GPU_PYTHON = venv_python(".venv-paddle")  # paddlepaddle-gpu venv (optional; no pymupdf needed there)
# OCR engines tried in order; the worker only needs paddleocr. Override with OCR_PYTHON / OCR_DEVICE.
OCR_ENGINES = (
    [(os.environ["OCR_PYTHON"], os.environ.get("OCR_DEVICE", "cpu"))]
    if os.environ.get("OCR_PYTHON")
    else ([(str(GPU_PYTHON), "gpu")] if GPU_PYTHON else []) + [(sys.executable, "cpu")]
)
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOBQ: "queue.Queue[str]" = queue.Queue()
# Embedded documents wait here and are saved to the index in batches: saving rewrites the whole user library, so doing it
# per file would be quadratic for thousands of files.
COMMIT: list[tuple[dict, list[dict], np.ndarray]] = []
COMMIT_BATCH = 200  # files per save
COMMIT_QUIET_SECONDS = 1.5  # save early when no new upload has arrived for this long
LAST_UPLOAD = 0.0
ingest_dirty = False  # something was ingested since the last return to normal mode
chat_model_unloaded = False  # the chat LLM was already freed for the current ingest (do not ask Ollama again per file)


# ---------------------------------------------------------------- index (law data + uploaded documents)
def doc_name(d: dict) -> str:
    return d.get("law_name") or d.get("doc") or "(ไม่ระบุ)"


def group_name(d: dict) -> str:
    """Legend / library entry a chunk belongs to: the zip name for files from a zip, otherwise the document itself."""
    return d.get("group") or doc_name(d)


def content_zones(vecs: np.ndarray, docs: list[dict], seed: int = 0) -> tuple[list[int], list[dict]]:
    """Spherical k-means on the embeddings (unit vectors): returns a zone index per point and the zone legend entries.
    A zone is named after the chunk closest to its centre, so the name comes from the data itself (any language)."""
    n = len(vecs)
    k = int(min(12, max(4, round((n / 150) ** 0.5))))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    centres, closest = [vecs[first]], vecs @ vecs[first]
    for _ in range(k - 1):  # k-means++ start: far-away points are likelier to open a new zone
        weight = np.clip(1 - closest, 0, None) ** 2
        nxt = int(rng.choice(n, p=weight / weight.sum()))
        centres.append(vecs[nxt])
        closest = np.maximum(closest, vecs @ vecs[nxt])
    centres = np.stack(centres)
    for _ in range(25):
        labels = (vecs @ centres.T).argmax(axis=1)
        moved = np.stack([vecs[labels == j].mean(axis=0) if (labels == j).any() else centres[j] for j in range(k)])
        moved /= np.linalg.norm(moved, axis=1, keepdims=True) + 1e-9
        if np.allclose(moved, centres, atol=1e-5):
            break
        centres = moved
    labels = (vecs @ centres.T).argmax(axis=1)
    order = [int(j) for j in np.argsort(-np.bincount(labels, minlength=k)) if (labels == j).any()]  # biggest zone first
    renumber = {old: new for new, old in enumerate(order)}
    zones = []
    for old in order:
        members = np.flatnonzero(labels == old)
        centre = members[int(np.argmax(vecs[members] @ centres[old]))]
        title = docs[int(centre)]["text"].split(": ", 1)[-1].strip().replace("\n", " ")[:48]
        zones.append({"name": f"โซน {renumber[old] + 1} · {title}", "count": int(len(members)), "user": False})
    return [renumber[int(j)] for j in labels], zones


def doc_label(d: dict) -> str:
    if d.get("kind") == "user":
        return d.get("label", "")
    return f"มาตรา {d['section_num']}" if d.get("section_num") else ""


class SearchIndex:
    """Lazy index: loads on first use, again when the law files change, and takes uploaded documents live."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._gpu = threading.Lock()  # one GPU job (query / document embedding) at a time
        self._stamp: tuple | None = None
        self._model = self._tok = None  # the embedding model lives on the GPU only while it is needed
        self._device = "cpu"
        self.keep_resident = EMBED_RESIDENT  # False: free the GPU after every embedding call (normal mode)
        self.law_docs: list[dict] = []
        self.law_vecs = self.mean = self.basis = None
        self._zone_cache: tuple | None = None  # (version, labels, payload) for the 3D globe
        self.zoned = False  # the legend shows content zones instead of files
        self.fixed_basis = False  # True with a base corpus: its map never moves; False: the map is fitted to the uploads
        self.box: dict = dict(EMPTY_BOX)
        self.user_docs: list[dict] = []
        self.user_vecs = np.zeros((0, 0), dtype=np.float32)
        self.docs: list[dict] = []
        self.vecs = None
        self.points: list[list[float]] = []
        self.law_idx: list[int] = []
        self.laws: list[dict] = []
        self.labels: list[str] = []
        self.version = 0

    @staticmethod
    def missing() -> str | None:
        """The base corpus (law) is optional: with neither file the app starts empty and the map is built from uploads."""
        if CORPUS.exists() and not EMBEDDINGS.exists():
            return "มี corpus.jsonl แต่ไม่มี embeddings.npy · ลบ corpus.jsonl ออกหรือใส่ embeddings.npy ให้ครบคู่"
        if EMBEDDINGS.exists() and not CORPUS.exists():
            return "มี embeddings.npy แต่ไม่มี corpus.jsonl · ลบ embeddings.npy ออกหรือใส่ corpus.jsonl ให้ครบคู่"
        return None

    def status(self) -> dict:
        reason = self.missing()
        return {"available": reason is None, "reason": reason, "docs": len(self.docs), "version": self.version,
                "user_docs": len({d["doc"] for d in self.user_docs})}

    def ensure(self) -> str | None:
        """Return an error message, or None when the index is ready."""
        with self._lock:
            reason = self.missing()
            if reason:
                return reason
            stamp = tuple(f.stat().st_mtime_ns if f.exists() else None for f in (CORPUS, EMBEDDINGS))
            if stamp == self._stamp:
                return None
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001 - e.g. embeddings file still being written
                return f"อ่านไฟล์ข้อมูลไม่ได้ ({type(exc).__name__}) · ถ้าไฟล์ยังถูกเขียนอยู่ให้รอให้เสร็จ"
            self._stamp = stamp
            return None

    def _load(self) -> None:
        if CORPUS.exists():
            docs = [json.loads(line) for line in open(CORPUS, encoding="utf-8")]
            vecs = np.load(EMBEDDINGS)
            if len(docs) != len(vecs):
                raise ValueError(f"corpus has {len(docs)} chunks but embeddings has {len(vecs)}; the two files must be a matching pair")
            mean = vecs.mean(axis=0)
            _, _, vt = np.linalg.svd(vecs - mean, full_matrices=False)
            basis = vt[:2].T
            self.box = self._box((vecs - mean) @ basis)
            self.law_docs, self.law_vecs, self.mean, self.basis, self.fixed_basis = docs, vecs, mean, basis, True
        else:
            self.law_docs, self.law_vecs, self.fixed_basis = [], None, False
        dim = self.law_vecs.shape[1] if self.law_vecs is not None else EMBED_DIM
        if self.law_vecs is None:
            self.law_vecs = np.zeros((0, dim), dtype=np.float32)
        self._load_user(dim)
        self._rebuild()

    @staticmethod
    def _box(points: np.ndarray) -> dict:
        if len(points) < 3:
            return dict(EMPTY_BOX)
        q = lambda a, f: float(np.quantile(a, f))  # noqa: E731
        return {"x0": q(points[:, 0], .01), "x1": q(points[:, 0], .99), "y0": q(points[:, 1], .01), "y1": q(points[:, 1], .99)}

    def _fit_basis(self, vecs: np.ndarray) -> None:
        """No base corpus: fit the 2-D map to everything uploaded so far (the layout refits as the library grows)."""
        dim = vecs.shape[1]
        self.mean = vecs.mean(axis=0) if len(vecs) else np.zeros(dim, dtype=np.float32)
        self.basis = np.zeros((dim, 2), dtype=np.float32)
        if len(vecs) >= 3:
            centred = (vecs - self.mean).astype(np.float64)
            _, vectors = np.linalg.eigh(centred.T @ centred)  # 1024 x 1024: much cheaper than an SVD of the whole library
            basis = vectors[:, -2:][:, ::-1]
            basis *= np.sign(basis[np.abs(basis).argmax(axis=0), range(2)])  # fixed sign, so the map does not flip between saves
            self.basis = basis.astype(np.float32)
        self.box = self._box((vecs - self.mean) @ self.basis)

    def _load_user(self, dim: int) -> None:
        self.user_docs, self.user_vecs = [], np.zeros((0, dim), dtype=np.float32)
        if USER_DOCS.exists() and USER_VECS.exists():
            rows = [json.loads(line) for line in open(USER_DOCS, encoding="utf-8")]
            vecs = np.load(USER_VECS)
            if len(rows) == len(vecs) and (not len(rows) or vecs.shape[1] == dim):
                self.user_docs, self.user_vecs = rows, vecs.astype(np.float32)

    def _rebuild(self) -> None:
        """Recompute everything derived from law + user data (the PCA basis stays fixed so law points never move)."""
        self.docs = self.law_docs + self.user_docs
        self.vecs = np.vstack([self.law_vecs, self.user_vecs]) if self.user_docs else self.law_vecs
        if not self.fixed_basis:
            self._fit_basis(self.vecs)
        self.points = np.round((self.vecs - self.mean) @ self.basis, 4).tolist()
        n_law = len(self.law_docs)
        law_counts = Counter(doc_name(d) for d in self.law_docs)
        groups = [(name, False) for name, _ in law_counts.most_common()]
        groups += [(name, True) for name in dict.fromkeys(group_name(d) for d in self.user_docs)]
        rank = {g: i for i, g in enumerate(groups)}
        user_counts = Counter(group_name(d) for d in self.user_docs)
        self.law_idx = [rank[(group_name(d), i >= n_law)] for i, d in enumerate(self.docs)]
        self.laws = [{"name": n, "count": (user_counts if u else law_counts)[n], "user": u} for n, u in groups]
        self.labels = [doc_label(d) for d in self.docs]
        self.zoned = not self.fixed_basis and len(self.laws) < ZONE_MAX_GROUPS and len(self.docs) >= ZONE_MIN_POINTS
        if self.zoned:  # one colour per file tells nothing when there are only a few files: show the content zones
            self.law_idx, self.laws = content_zones(np.asarray(self.vecs, dtype=np.float32), self.docs)
        self.version += 1

    def graph(self) -> dict:
        return {"points": self.points, "law": self.law_idx, "laws": self.laws, "section": self.labels,
                "box": self.box, "version": self.version, "base": self.fixed_basis}

    def doc(self, i: int) -> dict | None:
        if not 0 <= i < len(self.docs):
            return None
        d = self.docs[i]
        return {"law": doc_name(d), "section": self.labels[i], "text": d["text"][:2000]}

    def _ensure_model(self) -> None:
        """Load the embedding model onto the GPU if it is not there. Caller holds self._gpu."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
        self._model = AutoModel.from_pretrained(MODEL_DIR, dtype=dtype).to(self._device).eval()

    def _drop_model(self) -> None:
        """Free the GPU memory held by the embedding model. Caller holds self._gpu."""
        if self._model is None:
            return
        import torch

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_model(self) -> None:
        with self._gpu:
            self._drop_model()

    def embed_texts(self, texts: list[str], batch: int = 16) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        out = []
        with self._gpu:
            self._ensure_model()
            for i in range(0, len(texts), batch):
                enc = self._tok([t + self._tok.eos_token for t in texts[i : i + batch]], padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(self._device)
                with torch.inference_mode():
                    vec = self._model(**enc).last_hidden_state[:, -1]
                out.append(F.normalize(vec.float(), dim=-1).cpu().numpy())
            if not self.keep_resident:
                self._drop_model()
        return np.concatenate(out).astype(np.float32)

    def search(self, query: str, text_chars: int = 420) -> dict:
        """Top-K chunks. `text` is cut to text_chars (420 = enough for the result list; the LLM path asks for the full chunk)."""
        qv = self.embed_texts([f"Instruct: {TASK}\nQuery:{query}"])[0]
        scores = self.vecs @ qv
        top = np.argsort(-scores)[:TOP_K]
        return {
            "results": [
                {"i": int(i), "score": float(scores[i]), "law": doc_name(self.docs[i]), "section": self.labels[i],
                 "user": self.docs[i].get("kind") == "user", "text": self.docs[i]["text"][:text_chars]}
                for i in top
            ],
        }

    # ---- 3D globe: zones (top layer) and the points inside one zone (second layer)
    @staticmethod
    def _sphere_dirs(centres: np.ndarray) -> np.ndarray:
        """Place zone centres on a sphere: their top-3 principal directions, then pushed apart so labels do not overlap."""
        k = len(centres)
        centred = centres - centres.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        xyz = centred @ vt[:3].T
        if xyz.shape[1] < 3:
            xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])))
        xyz += np.random.default_rng(0).normal(scale=1e-3, size=xyz.shape)
        dirs = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        for _ in range(60):  # light relaxation: neighbours closer than ~35 degrees repel
            for i in range(k):
                push = np.zeros(3)
                for j in range(k):
                    if i != j:
                        d = float(dirs[i] @ dirs[j])
                        if d > 0.82:
                            push += (dirs[i] - dirs[j]) * (d - 0.82)
                dirs[i] = dirs[i] + push * 0.5
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs

    def zones(self) -> dict:
        """Top layer of the globe: content zones with a direction on the sphere. Cached per index version."""
        with self._lock:
            if self._zone_cache and self._zone_cache[0] == self.version:
                return self._zone_cache[2]
            n = len(self.docs)
            if n < ZONE_MIN_POINTS:
                payload = {"version": self.version, "n": n, "zones": [], "zone_of": []}
                self._zone_cache = (self.version, np.zeros(0, dtype=int), payload)
                return payload
            vecs = np.asarray(self.vecs, dtype=np.float32)
            labels, zones = content_zones(vecs, self.docs)
            labels_arr = np.asarray(labels)
            centres = np.stack([vecs[labels_arr == j].mean(axis=0) for j in range(len(zones))])
            for j, (zone, direction) in enumerate(zip(zones, self._sphere_dirs(centres))):
                zone.update(id=j, dir=[round(float(x), 4) for x in direction])
            payload = {"version": self.version, "n": n, "zones": zones, "zone_of": labels}
            self._zone_cache = (self.version, labels_arr, payload)
            return payload

    def zone_detail(self, zone_id: int) -> dict | None:
        """Second layer: the points of one zone in a local 3-D layout, grouped into sub-zones."""
        payload = self.zones()
        with self._lock:
            labels = self._zone_cache[1]
            if not 0 <= zone_id < len(payload["zones"]):
                return None
            members = np.flatnonzero(labels == zone_id)
            total = len(members)
            if total > ZONE_POINT_CAP:
                members = np.sort(np.random.default_rng(0).choice(members, ZONE_POINT_CAP, replace=False))
            vecs = np.asarray(self.vecs[members], dtype=np.float32)
            centred = (vecs - vecs.mean(axis=0)).astype(np.float64)
            _, vectors = np.linalg.eigh(centred.T @ centred)
            xyz = centred @ vectors[:, -3:][:, ::-1]
            xyz /= np.quantile(np.abs(xyz), 0.99, axis=0) + 1e-9  # most points inside [-1, 1] on every axis
            xyz = np.clip(xyz, -1.3, 1.3)
            docs = [self.docs[i] for i in members]
            if total >= ZONE_MIN_POINTS:
                sub, subs = content_zones(vecs, docs)
            else:
                sub, subs = [0] * len(members), [{"name": payload["zones"][zone_id]["name"], "count": len(members)}]
            names = list(dict.fromkeys(group_name(d) if d.get("kind") == "user" else doc_name(d) for d in docs))
            lookup = {n: i for i, n in enumerate(names)}
            return {"id": zone_id, "name": payload["zones"][zone_id]["name"], "total": total, "shown": len(members),
                    "idx": members.tolist(), "xyz": np.round(xyz, 3).tolist(), "sub": sub, "subs": subs, "names": names,
                    "nm": [lookup[group_name(d) if d.get("kind") == "user" else doc_name(d)] for d in docs],
                    "lab": [self.labels[i] for i in members]}

    # ---- uploaded documents
    def add_user_docs(self, items: list[tuple[str, list[dict], np.ndarray]]) -> None:
        """Add several documents with ONE save and ONE rebuild (re-uploading a name replaces the old copy)."""
        fresh = {name: (rows, vecs) for name, rows, vecs in items}  # a name repeated in the batch: the last one wins
        with self._lock:
            keep = [i for i, d in enumerate(self.user_docs) if d["doc"] not in fresh]
            docs = [self.user_docs[i] for i in keep]
            vecs = [self.user_vecs[keep]] if keep else []
            for rows, v in fresh.values():
                docs += rows
                vecs.append(v)
            self.user_docs, self.user_vecs = docs, np.vstack(vecs).astype(np.float32)
            self._save_user()
            self._rebuild()

    def remove_user_doc(self, name: str) -> bool:
        with self._lock:
            keep = [i for i, d in enumerate(self.user_docs) if d["doc"] != name and d.get("group") != name]
            if len(keep) == len(self.user_docs):
                return False
            self.user_docs = [self.user_docs[i] for i in keep]
            self.user_vecs = self.user_vecs[keep]
            self._save_user()
            self._rebuild()
            return True

    def library(self) -> list[dict]:
        chunks = Counter(group_name(d) for d in self.user_docs)
        files = {g: len({d["doc"] for d in self.user_docs if group_name(d) == g}) for g in chunks} if len(chunks) < 50 else {}
        return [{"name": n, "chunks": c, "files": files.get(n, 1)} for n, c in chunks.items()]

    def _save_user(self) -> None:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        tmp_docs, tmp_vecs = USER_DOCS.with_suffix(".tmp"), USER_VECS.with_suffix(".tmp")
        with open(tmp_docs, "w", encoding="utf-8") as f:
            for r in self.user_docs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(tmp_vecs, "wb") as f:
            np.save(f, self.user_vecs)
        os.replace(tmp_docs, USER_DOCS)
        os.replace(tmp_vecs, USER_VECS)


INDEX = SearchIndex()


# ---------------------------------------------------------------- OCR plumbing
def update_job(job: dict, **fields) -> None:
    with JOBS_LOCK:
        job.update(fields)


def render_pages(path: Path, out_dir: Path, numbers: list[int]) -> list[tuple[int, Path]]:
    """Render the pages that need OCR to PNG (at most MAX_SIDE px on the long side)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    with pymupdf.open(path) as doc:
        for number in numbers:
            page = doc[number - 1]
            zoom = min(RENDER_DPI / 72, MAX_SIDE / max(page.rect.width, page.rect.height))
            png = out_dir / f"page_{number}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(png)
            items.append((number, png))
    return items


def run_ocr(job: dict, items: list[tuple[int, Path]]) -> None:
    """Try each OCR engine in turn (GPU first); a failed engine hands its unfinished pages to the next one."""
    last_error: Exception | None = None
    for python, device in OCR_ENGINES:
        todo = [(n, p) for n, p in items if str(n) not in job["pages"]]
        if not todo:
            return
        update_job(job, engine=device)
        try:
            run_ocr_worker(job, python, device, todo)
            return
        except RuntimeError as exc:
            last_error = exc
            print(f"OCR engine '{device}' failed: {exc}", flush=True)
    raise last_error or RuntimeError("no OCR engine configured")


def run_ocr_worker(job: dict, python: str, device: str, items: list[tuple[int, Path]]) -> None:
    cmd = [python, str(HERE / "ocr_worker.py"), device, *[f"{n}|{p}" for n, p in items]]
    proc = subprocess.Popen(
        cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
    )
    tail: list[str] = []
    for line in proc.stdout:
        if line.startswith(RESULT_MARK):
            item = json.loads(line[len(RESULT_MARK) :])
            with JOBS_LOCK:
                job["pages"][str(item["page"])] = {"route": "ocr", "text": item["text"], "confidence": item["confidence"]}
                job["done"] += 1
        else:
            tail = (tail + [line.rstrip()])[-12:]
    if proc.wait() != 0 or any(str(n) not in job["pages"] for n, _ in items):
        raise RuntimeError("OCR worker failed: " + " | ".join(t for t in tail if t.strip())[-600:])


# ---------------------------------------------------------------- extraction per file type
def extract_pdf(job: dict, path: Path) -> list[dict]:
    with pymupdf.open(path) as doc:
        if len(doc) > MAX_PAGES:
            raise ValueError(f"PDF มี {len(doc)} หน้า เกินขีดจำกัด {MAX_PAGES} หน้า")
        decisions = [rp.decide_auto(rp.inspect_page(p, n), MIN_CHARS, IMAGE_AREA) for n, p in enumerate(doc, 1)]
        plan = [{"n": d.page.number, "chars": d.page.chars, "image_ratio": round(d.page.image_ratio, 3),
                 "route": "ocr" if d.use_ocr else "text", "reason": d.reason} for d in decisions]
        update_job(job, plan=plan, total=len(plan), status="running")
        for d in decisions:  # pages with a good text layer need no OCR
            if not d.use_ocr:
                with JOBS_LOCK:
                    job["pages"][str(d.page.number)] = {"route": "text", "text": d.page.text, "confidence": None}
                    job["done"] += 1
    ocr_pages = [d.page.number for d in decisions if d.use_ocr]
    if ocr_pages:
        run_ocr(job, render_pages(path, OUT_ROOT / path.stem, ocr_pages))
    return [{"label": f"หน้า {n}", "text": job["pages"][str(n)]["text"]} for n in range(1, len(decisions) + 1)]


def extract_image(job: dict, path: Path) -> list[dict]:
    update_job(job, plan=[{"n": 1, "chars": 0, "image_ratio": 1.0, "route": "ocr", "reason": "image file"}],
               total=1, status="running")
    run_ocr(job, [(1, path)])
    return [{"label": "รูปภาพ", "text": job["pages"]["1"]["text"]}]


def extract_segments(job: dict, path: Path) -> list[dict]:
    kind = job["kind"]
    if kind == "pdf":
        return extract_pdf(job, path)
    if kind == "image":
        return extract_image(job, path)
    segments = ex.read_office(path) if kind == "office" else ex.read_text_like(path)
    update_job(job, total=1, done=1, status="running",
               preview="\n\n".join(s["text"] for s in segments)[:3000])
    return segments


def make_rows(name: str, segments: list[dict]) -> list[dict]:
    """Chunk every segment; each chunk is prefixed with file name + label so it is understandable on its own."""
    doc_id = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    rows, k = [], 0
    for seg in segments:
        for piece in SPLITTER.split_text(seg["text"]):
            k += 1
            label = seg["label"] or f"ส่วนที่ {k}"
            rows.append({"id": f"user-{doc_id}-{k}", "kind": "user", "doc": name, "label": label,
                         "text": f"{name} · {label}: {piece}"})
            if len(rows) >= MAX_CHUNKS:
                return rows
    return rows


def enter_ingest_mode() -> None:
    """Ingest mode: wait for a running answer, free VRAM from the chat LLM, keep the embedding model on the GPU."""
    global chat_model_unloaded
    llm.GATE.wait_for_answers()
    if not chat_model_unloaded:  # chat stays blocked until the queue is empty
        llm.unload_model()
        chat_model_unloaded = True
    INDEX.keep_resident = True
    error = INDEX.ensure()
    if error:
        raise RuntimeError(error)


def read_zip_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo, work: Path) -> list[dict]:
    ext = Path(info.filename).suffix.lower()
    data = zf.open(info).read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise ValueError("too large")
    tmp = work / f"entry{ext}"
    tmp.write_bytes(data)
    return ex.read_office(tmp) if ext in ex.OFFICE_EXT else ex.read_text_like(tmp)


def save_zip_group(group: list[tuple[str, list[dict]]]) -> int:
    """Embed and save a group of (name, rows) documents with one save; returns the number of chunks."""
    vecs = INDEX.embed_texts([r["text"] for _, rows in group for r in rows])
    items, start = [], 0
    for name, rows in group:
        items.append((name, rows, vecs[start : start + len(rows)]))
        start += len(rows)
    INDEX.add_user_docs(items)
    return start


def process_zip_job(job: dict) -> None:
    """One zip = one job: read every text/office/html file inside, chunk, embed and save in groups."""
    path = Path(job["path"])
    stem = safe_relpath(Path(job["name"]).stem)
    work = path.parent / f"{job['id']}-unzip"
    try:
        update_job(job, status="analyzing")
        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            entries = [i for i in zf.infolist() if not i.is_dir() and not ZIP_HIDDEN.search(i.filename)]
            if len(entries) > MAX_ZIP_ENTRIES:
                raise ValueError(f"zip มี {len(entries)} ไฟล์ เกินขีดจำกัด {MAX_ZIP_ENTRIES}")
            if sum(i.file_size for i in entries) > MAX_ZIP_UNPACKED:
                raise ValueError("zip ใหญ่เกินไปเมื่อแตกไฟล์ (เกิน 2 GB)")
            update_job(job, total=len(entries), done=0, status="running", note="กำลังอ่านไฟล์ใน zip")
            enter_ingest_mode()
            group, docs, chunks, skipped = [], 0, 0, Counter()
            for n, info in enumerate(entries, 1):
                ext = Path(info.filename).suffix.lower()
                if ext in ex.PDF_EXT | ex.IMAGE_EXT:
                    skipped["PDF/ภาพ (อัปแยกเพื่อใช้ OCR)"] += 1
                elif ext not in ZIP_READABLE:
                    skipped["ชนิดที่ไม่รองรับ"] += 1
                elif info.file_size == 0:
                    skipped["ไฟล์ว่าง"] += 1
                else:
                    try:
                        segments = read_zip_entry(zf, info, work)
                        rows = make_rows(f"{stem}/{safe_relpath(info.filename)}", [s for s in segments if s["text"].strip()])
                        if rows:
                            for r in rows:
                                r["group"] = stem  # one legend entry / library item for the whole zip
                            group.append((rows[0]["doc"], rows))
                        else:
                            skipped["ไม่มีข้อความ"] += 1
                    except Exception as exc:  # noqa: BLE001 - one broken file must not stop the other thousands
                        print(f"zip entry {info.filename!r} failed: {exc!r}", flush=True)
                        skipped["อ่านไม่ได้"] += 1
                if len(group) >= ZIP_GROUP:
                    docs, chunks = docs + len(group), chunks + save_zip_group(group)
                    group = []
                update_job(job, done=n, chunks=chunks, note=f"เพิ่มแล้ว {docs} ไฟล์ · {chunks} จุด")
            if group:
                docs, chunks = docs + len(group), chunks + save_zip_group(group)
        if not docs:
            raise ValueError("ไม่พบไฟล์ที่อ่านได้ใน zip (รองรับ txt md csv json html docx pptx xlsx)")
        skip_note = f" · ข้าม {sum(skipped.values())} ไฟล์ ({', '.join(f'{k} {v}' for k, v in skipped.items())})" if skipped else ""
        update_job(job, status="done", chunks=chunks, note=f"เพิ่ม {docs} ไฟล์ {chunks} จุดบนแผนที่แล้ว{skip_note}")
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        print(f"zip job {job['id']} failed: {exc!r}", flush=True)
        update_job(job, status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)  # only the temporary folder this job made
        llm.GATE.job_finished()


def process_job(job_id: str) -> None:
    job = JOBS[job_id]
    if job["kind"] == "zip":
        return process_zip_job(job)
    path = Path(job["path"])
    try:
        update_job(job, status="analyzing")
        segments = extract_segments(job, path)
        rows = make_rows(job["name"], [s for s in segments if s["text"].strip()])
        if not rows:
            raise ValueError("ไม่พบข้อความในไฟล์นี้")
        update_job(job, status="embedding", note=f"หั่นได้ {len(rows)} ชิ้น กำลังสร้างเวกเตอร์")
        enter_ingest_mode()
        vecs = INDEX.embed_texts([r["text"] for r in rows])
        update_job(job, note=f"สร้างเวกเตอร์แล้ว {len(rows)} ชิ้น รอบันทึกเข้าคลัง")
        COMMIT.append((job, rows, vecs))
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        print(f"job {job_id} failed: {exc!r}", flush=True)
        update_job(job, status="error", error=f"{type(exc).__name__}: {exc}")
        llm.GATE.job_finished()


def flush_commits() -> None:
    """Save every embedded document waiting in COMMIT to the index in one go, then mark those jobs done."""
    batch = COMMIT[:]
    del COMMIT[:]
    if not batch:
        return
    try:
        INDEX.add_user_docs([(job["name"], rows, vecs) for job, rows, vecs in batch])
        for job, rows, _ in batch:
            capped = " (ตัดที่ %d ชิ้น)" % MAX_CHUNKS if len(rows) >= MAX_CHUNKS else ""
            update_job(job, status="done", chunks=len(rows), note=f"เพิ่ม {len(rows)} จุดบนแผนที่แล้ว{capped}")
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        print(f"saving {len(batch)} documents failed: {exc!r}", flush=True)
        for job, _, _ in batch:
            update_job(job, status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        for _ in batch:
            llm.GATE.job_finished()  # chat unlocks only after the documents are really searchable


def settle_after_ingest() -> None:
    """Back to normal mode once the queue is empty: embedding model off the GPU, chat LLM loaded again."""
    global chat_model_unloaded
    chat_model_unloaded = False
    INDEX.keep_resident = EMBED_RESIDENT
    if not EMBED_RESIDENT:
        INDEX.release_model()
    threading.Thread(target=llm.warm_model, daemon=True).start()


def worker_loop() -> None:
    global ingest_dirty
    while True:
        try:
            job_id = JOBQ.get(timeout=1.0)
        except queue.Empty:
            if COMMIT and time.time() - LAST_UPLOAD > COMMIT_QUIET_SECONDS:
                flush_commits()
            if ingest_dirty and not COMMIT and llm.GATE.pending == 0:
                ingest_dirty = False
                settle_after_ingest()
            continue
        ingest_dirty = True
        process_job(job_id)
        if len(COMMIT) >= COMMIT_BATCH:
            flush_commits()


def summary(job: dict) -> dict:
    plan = job.get("plan", [])
    return {"id": job["id"], "name": job["name"], "kind": job["kind"], "status": job["status"], "total": job["total"],
            "done": job["done"], "chunks": job.get("chunks", 0), "engine": job.get("engine", ""),
            "note": job.get("note", ""), "error": job.get("error", ""),
            "ocr_pages": sum(1 for p in plan if p["route"] == "ocr"), "text_pages": sum(1 for p in plan if p["route"] != "ocr")}


def export_text(job: dict) -> str:
    if job["pages"]:
        parts = [f"=== page {k} ({'OCR' if job['pages'][k]['route'] == 'ocr' else 'text'}) ===\n{job['pages'][k]['text']}"
                 for k in sorted(job["pages"], key=int)]
        return "\n\n".join(parts)
    return job.get("preview", "")


def thumbnail(path: Path, number: int, width: int) -> bytes | None:
    with pymupdf.open(path) as doc:  # pymupdf opens PDFs and images alike
        if not 1 <= number <= len(doc):
            return None
        page = doc[number - 1]
        zoom = max(width, 80) / page.rect.width
        return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


def safe_relpath(raw: str) -> str:
    parts = [re.sub(r"[^\w\-. ฀-๿()\[\],]", "_", p).strip(" .") for p in re.split(r"[\\/]+", raw)]
    parts = [p for p in parts if p and p != ".."]
    return "/".join(parts)[-200:] or "upload"


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, status: int = 200, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        try:
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:  # the browser reloaded / closed the tab mid-response; nothing to do
            pass

    def _json(self, data, status: int = 200) -> None:
        self._send(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def _job(self, job_id: str) -> dict | None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            return json.loads(json.dumps(job, default=str)) if job else None

    def _answer(self, question: str) -> None:
        if not question:
            return self._json({"error": "ใส่คำถามก่อน"}, 400)
        if not llm.GATE.start_answer():
            return self._json({"error": "กำลังลงเอกสารใหม่อยู่ · ถาม AI ได้เมื่อลงเสร็จ (ค้นหาธรรมดายังใช้ได้)", "busy": True}, 409)
        try:
            error = INDEX.ensure()
            if error:
                return self._json({"error": error}, 503)
            hits = INDEX.search(question, text_chars=llm.SOURCE_CHARS)["results"][: llm.TOP_SOURCES]
            reply = llm.ask_adaptive(question, hits)  # 4 chunks first, 8 if the model says "not found"
            reply["sources"] = [{"n": n, "i": h["i"], "law": h["law"], "section": h["section"], "user": h["user"],
                                 "text": h["text"][:240]} for n, h in enumerate(reply["sources"], 1)]
            return self._json(reply)
        except llm.LlmError as exc:
            return self._json({"error": str(exc)}, exc.status)
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        finally:
            llm.GATE.end_answer()

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self._send(PAGE_HTML.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/status":
            return self._json({**INDEX.status(), "ingesting": llm.GATE.pending > 0, "pending": llm.GATE.pending,
                               "llm_model": llm.OLLAMA_MODEL})
        if url.path == "/api/answer":
            return self._answer(q.get("q", "").strip()[:500])
        if url.path == "/api/config":
            return self._json({"extensions": sorted(ex.SUPPORTED | {".zip"}), "legacy": ex.LEGACY_HINT, "max_mb": MAX_UPLOAD >> 20,
                               "max_zip_mb": MAX_ZIP >> 20, "max_drop_files": MAX_DROP_FILES})
        if url.path.startswith("/static/"):
            name = url.path[len("/static/"):]
            target = STATIC_DIR / name
            if name not in STATIC_FILES:
                return self._json({"error": "not found"}, 404)
            if not target.exists():
                return self._json({"error": f"missing static/{name}"}, 404)
            return self._send(target.read_bytes(), STATIC_FILES[name])
        if url.path in ("/api/zones", "/api/zone"):
            error = INDEX.ensure()
            if error:
                return self._json({"error": error}, 503)
            if url.path == "/api/zones":
                return self._json(INDEX.zones())
            zone_id = q.get("id", "")
            detail = INDEX.zone_detail(int(zone_id)) if zone_id.isdigit() else None
            return self._json(detail if detail else {"error": "zone not found"}, 200 if detail else 404)
        if url.path == "/api/library":
            INDEX.ensure()  # the library is read from disk on first load; an error just means an empty list
            return self._json({"docs": INDEX.library()})
        if url.path in ("/api/search", "/api/points", "/api/doc"):
            error = INDEX.ensure()
            if error:
                return self._json({"error": error}, 503)
            if url.path == "/api/points":
                return self._json(INDEX.graph())
            if url.path == "/api/doc":
                try:
                    doc = INDEX.doc(int(q.get("i", -1)))
                except ValueError:
                    doc = None
                return self._json(doc) if doc else self._json({"error": "not found"}, 404)
            query = q.get("q", "").strip()[:500]
            if not query:
                return self._json({"results": []})
            try:
                return self._json(INDEX.search(query))
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        if url.path == "/api/jobs":
            with JOBS_LOCK:
                jobs = [summary(JOBS[i]) for i in q.get("ids", "").split(",") if i in JOBS]
            return self._json({"jobs": jobs})
        if url.path in ("/api/job", "/api/export", "/api/thumb"):
            snapshot = self._job(q.get("id", ""))
            if snapshot is None:
                return self._json({"error": "job not found"}, 404)
            if url.path == "/api/job":
                return self._json(snapshot)
            if url.path == "/api/export":
                return self._send(export_text(snapshot).encode("utf-8"), "text/plain; charset=utf-8",
                                  extra={"Content-Disposition": 'attachment; filename="extracted_text.txt"'})
            if snapshot["kind"] not in ("pdf", "image"):
                return self._json({"error": "no thumbnail"}, 404)
            try:
                png = thumbnail(Path(snapshot["path"]), int(q.get("page", 1)), int(q.get("w", 360)))
            except ValueError:
                return self._json({"error": "bad parameter"}, 400)
            if png is None:
                return self._json({"error": "page not found"}, 404)
            return self._send(png, "image/png", extra={"Cache-Control": "max-age=600"})
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        global LAST_UPLOAD
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/api/delete":
            return self._json({"ok": INDEX.remove_user_doc(q.get("name", ""))})
        if url.path != "/api/upload":
            return self._json({"error": "not found"}, 404)
        name = safe_relpath(q.get("name", "upload"))
        ext = Path(name).suffix.lower()
        if ext in ex.LEGACY_HINT:
            return self._json({"error": ex.LEGACY_HINT[ext]}, 415)
        if ext not in ex.SUPPORTED | {".zip"}:
            return self._json({"error": f"ไม่รองรับไฟล์ชนิด {ext or '(ไม่มีนามสกุล)'}"}, 415)
        length = int(self.headers.get("Content-Length", 0))
        if ext == ".zip":
            return self._upload_zip(name, length)
        if not 0 < length <= MAX_UPLOAD:
            return self._json({"error": "ไฟล์ว่างหรือใหญ่เกิน 100 MB"}, 413)
        data = self.rfile.read(length)
        if (ext == ".pdf" and not data.startswith(b"%PDF")) or (ext in ex.OFFICE_EXT and not data.startswith(b"PK")):
            return self._json({"error": "เนื้อไฟล์ไม่ตรงกับนามสกุล"}, 400)
        job_id = uuid.uuid4().hex[:12]
        UPLOADS.mkdir(parents=True, exist_ok=True)
        path = UPLOADS / f"{job_id}-{Path(name).name}"
        path.write_bytes(data)
        job = {"id": job_id, "status": "queued", "name": name, "kind": ex.kind_of(path), "path": str(path),
               "engine": "", "total": 0, "done": 0, "plan": [], "pages": {}, "preview": "", "chunks": 0, "note": ""}
        with JOBS_LOCK:
            JOBS[job_id] = job
        llm.GATE.job_added()  # chat is refused from now until this job (and any others) finish
        LAST_UPLOAD = time.time()
        JOBQ.put(job_id)
        self._json({"job": job_id, "name": name})

    def _upload_zip(self, name: str, length: int) -> None:
        """A zip can be large, so it is streamed to disk instead of being held in memory."""
        global LAST_UPLOAD
        if not 0 < length <= MAX_ZIP:
            return self._json({"error": f"zip ว่างหรือใหญ่เกิน {MAX_ZIP >> 20} MB"}, 413)
        job_id = uuid.uuid4().hex[:12]
        UPLOADS.mkdir(parents=True, exist_ok=True)
        path = UPLOADS / f"{job_id}-{Path(name).name}"
        left = length
        with open(path, "wb") as f:
            while left > 0:
                block = self.rfile.read(min(1 << 20, left))
                if not block:
                    break
                f.write(block)
                left -= len(block)
        if left or not zipfile.is_zipfile(path):
            path.unlink(missing_ok=True)  # the partial file this request just wrote
            return self._json({"error": "ไฟล์ zip เสียหรือส่งไม่ครบ"}, 400)
        job = {"id": job_id, "status": "queued", "name": name, "kind": "zip", "path": str(path),
               "engine": "", "total": 0, "done": 0, "plan": [], "pages": {}, "preview": "", "chunks": 0, "note": ""}
        with JOBS_LOCK:
            JOBS[job_id] = job
        llm.GATE.job_added()
        LAST_UPLOAD = time.time()
        JOBQ.put(job_id)
        self._json({"job": job_id, "name": name})

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=llm.warm_model, daemon=True).start()  # normal mode: the chat LLM waits in VRAM
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"ready: http://{HOST}:{PORT}", flush=True)
    server.serve_forever()
