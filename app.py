"""Thai-law RAG demo + document upload in one page.   usage: python app.py   ->  http://127.0.0.1:8765

search : Qwen3-Embedding over corpus.jsonl / embeddings.npy (law) + your uploaded documents
upload : files or whole folders (pdf txt md csv json html docx pptx xlsx + images) are extracted (OCR when needed),
         chunked, embedded and added to the map as new points - one background queue, one file at a time.
OCR runs in a child process (ocr_worker.py). Do not import paddle in this process.

env (optional): APP_PORT, APP_USER_DIR (where uploads and the user library live), OCR_PYTHON, OCR_DEVICE
"""
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

import extractors as ex
import route_pdf as rp

HERE = Path(__file__).parent
USER_DIR = Path(os.environ.get("APP_USER_DIR", HERE))
CORPUS, EMBEDDINGS = HERE / "corpus.jsonl", HERE / "embeddings.npy"  # law data (build_corpus.py / embed_corpus.py)
USER_DOCS, USER_VECS = USER_DIR / "user_docs.jsonl", USER_DIR / "user_vecs.npy"  # uploaded documents (this app)
UPLOADS, OUT_ROOT, PAGE_HTML = USER_DIR / "uploads", USER_DIR / "route_out", HERE / "app_page.html"
MODEL_DIR = HERE / "Qwen3-Embedding-0.6B"
HOST, PORT = "127.0.0.1", int(os.environ.get("APP_PORT", 8765))
TASK = "Given a web search query, retrieve relevant passages that answer the query"
TOP_K = 10
MIN_CHARS, IMAGE_AREA = 50, 0.20  # PDF routing thresholds (auto)
MAX_UPLOAD, MAX_PAGES, MAX_CHUNKS = 100 * 1024 * 1024, 300, 5000
RESULT_MARK = "@@RESULT@@ "
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


# ---------------------------------------------------------------- index (law data + uploaded documents)
def doc_name(d: dict) -> str:
    return d.get("law_name") or d.get("doc") or "(ไม่ระบุ)"


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
        self._model = self._tok = None
        self._device = "cpu"
        self.law_docs: list[dict] = []
        self.law_vecs = self.mean = self.basis = None
        self.box: dict = {}
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
        if not CORPUS.exists():
            return "ยังไม่มี corpus.jsonl · รัน build_corpus.py ก่อน"
        if not EMBEDDINGS.exists():
            return "ยังไม่มี embeddings.npy · รัน embed_corpus.py ก่อน"
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
            stamp = (CORPUS.stat().st_mtime_ns, EMBEDDINGS.stat().st_mtime_ns)
            if stamp == self._stamp:
                return None
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001 - e.g. embeddings file still being written
                return f"อ่านไฟล์ข้อมูลไม่ได้ ({type(exc).__name__}) · ถ้า embed_corpus.py ยังรันอยู่ให้รอให้เสร็จ"
            self._stamp = stamp
            return None

    def _load(self) -> None:
        docs = [json.loads(line) for line in open(CORPUS, encoding="utf-8")]
        vecs = np.load(EMBEDDINGS)
        if len(docs) != len(vecs):
            raise ValueError(f"corpus has {len(docs)} chunks but embeddings has {len(vecs)}; rerun embed_corpus.py")
        mean = vecs.mean(axis=0)
        _, _, vt = np.linalg.svd(vecs - mean, full_matrices=False)
        basis = vt[:2].T
        law_points = (vecs - mean) @ basis
        q = lambda a, f: float(np.quantile(a, f))  # noqa: E731
        self.box = {"x0": q(law_points[:, 0], .01), "x1": q(law_points[:, 0], .99),
                    "y0": q(law_points[:, 1], .01), "y1": q(law_points[:, 1], .99)}
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
            self._model = AutoModel.from_pretrained(MODEL_DIR, dtype=dtype).to(self._device).eval()
        self.law_docs, self.law_vecs, self.mean, self.basis = docs, vecs, mean, basis
        self._load_user(vecs.shape[1])
        self._rebuild()

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
        self.points = np.round((self.vecs - self.mean) @ self.basis, 4).tolist()
        n_law = len(self.law_docs)
        law_counts = Counter(doc_name(d) for d in self.law_docs)
        groups = [(name, False) for name, _ in law_counts.most_common()]
        groups += [(name, True) for name in dict.fromkeys(d["doc"] for d in self.user_docs)]
        rank = {g: i for i, g in enumerate(groups)}
        user_counts = Counter(d["doc"] for d in self.user_docs)
        self.law_idx = [rank[(doc_name(d), i >= n_law)] for i, d in enumerate(self.docs)]
        self.laws = [{"name": n, "count": (user_counts if u else law_counts)[n], "user": u} for n, u in groups]
        self.labels = [doc_label(d) for d in self.docs]
        self.version += 1

    def graph(self) -> dict:
        return {"points": self.points, "law": self.law_idx, "laws": self.laws, "section": self.labels,
                "box": self.box, "version": self.version}

    def doc(self, i: int) -> dict | None:
        if not 0 <= i < len(self.docs):
            return None
        d = self.docs[i]
        return {"law": doc_name(d), "section": self.labels[i], "text": d["text"][:2000]}

    def embed_texts(self, texts: list[str], batch: int = 16) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        out = []
        with self._gpu:
            for i in range(0, len(texts), batch):
                enc = self._tok([t + self._tok.eos_token for t in texts[i : i + batch]], padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(self._device)
                with torch.inference_mode():
                    vec = self._model(**enc).last_hidden_state[:, -1]
                out.append(F.normalize(vec.float(), dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def search(self, query: str) -> dict:
        qv = self.embed_texts([f"Instruct: {TASK}\nQuery:{query}"])[0]
        scores = self.vecs @ qv
        top = np.argsort(-scores)[:TOP_K]
        return {
            "results": [
                {"i": int(i), "score": float(scores[i]), "law": doc_name(self.docs[i]), "section": self.labels[i],
                 "user": self.docs[i].get("kind") == "user", "text": self.docs[i]["text"][:420]}
                for i in top
            ],
        }

    # ---- uploaded documents
    def add_user_doc(self, name: str, rows: list[dict], vecs: np.ndarray) -> None:
        with self._lock:
            keep = [i for i, d in enumerate(self.user_docs) if d["doc"] != name]  # re-uploading replaces the old copy
            self.user_docs = [self.user_docs[i] for i in keep] + rows
            self.user_vecs = np.vstack([self.user_vecs[keep], vecs]) if keep else vecs
            self._save_user()
            self._rebuild()

    def remove_user_doc(self, name: str) -> bool:
        with self._lock:
            keep = [i for i, d in enumerate(self.user_docs) if d["doc"] != name]
            if len(keep) == len(self.user_docs):
                return False
            self.user_docs = [self.user_docs[i] for i in keep]
            self.user_vecs = self.user_vecs[keep]
            self._save_user()
            self._rebuild()
            return True

    def library(self) -> list[dict]:
        counts = Counter(d["doc"] for d in self.user_docs)
        return [{"name": n, "chunks": c} for n, c in counts.items()]

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


def process_job(job_id: str) -> None:
    job = JOBS[job_id]
    path = Path(job["path"])
    try:
        update_job(job, status="analyzing")
        segments = extract_segments(job, path)
        rows = make_rows(job["name"], [s for s in segments if s["text"].strip()])
        if not rows:
            raise ValueError("ไม่พบข้อความในไฟล์นี้")
        update_job(job, status="embedding", note=f"หั่นได้ {len(rows)} ชิ้น กำลังสร้างเวกเตอร์")
        error = INDEX.ensure()
        if error:
            raise RuntimeError(error)
        INDEX.add_user_doc(job["name"], rows, INDEX.embed_texts([r["text"] for r in rows]))
        capped = " (ตัดที่ %d ชิ้น)" % MAX_CHUNKS if len(rows) >= MAX_CHUNKS else ""
        update_job(job, status="done", chunks=len(rows), note=f"เพิ่ม {len(rows)} จุดบนแผนที่แล้ว{capped}")
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        print(f"job {job_id} failed: {exc!r}", flush=True)
        update_job(job, status="error", error=f"{type(exc).__name__}: {exc}")


def worker_loop() -> None:
    while True:
        process_job(JOBQ.get())


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

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self._send(PAGE_HTML.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/status":
            return self._json(INDEX.status())
        if url.path == "/api/config":
            return self._json({"extensions": sorted(ex.SUPPORTED), "legacy": ex.LEGACY_HINT, "max_mb": MAX_UPLOAD >> 20})
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
        if ext not in ex.SUPPORTED:
            return self._json({"error": f"ไม่รองรับไฟล์ชนิด {ext or '(ไม่มีนามสกุล)'}"}, 415)
        length = int(self.headers.get("Content-Length", 0))
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
        JOBQ.put(job_id)
        self._json({"job": job_id, "name": name})

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    threading.Thread(target=worker_loop, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"ready: http://{HOST}:{PORT}", flush=True)
    server.serve_forever()
