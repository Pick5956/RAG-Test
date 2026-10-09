"""Generation step: answer a question from retrieved chunks with a local LLM served by Ollama.

Also holds ChatGate: while documents are being ingested (embedding needs the GPU memory), chat is refused and the
LLM is unloaded from VRAM. Standard library only.

env: OLLAMA_URL (default http://127.0.0.1:11434), OLLAMA_MODEL (default iapp/chinda-qwen3-4b)
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "iapp/chinda-qwen3-4b")
# Adaptive retrieval: start with the first 4 chunks; if the model says "not found", retry with 8 (env: ANSWER_STEPS="4,8")
ADAPTIVE_STEPS = tuple(int(x) for x in os.environ.get("ANSWER_STEPS", "4,8").split(",") if x.strip().isdigit()) or (4, 8)
TOP_SOURCES = max(ADAPTIVE_STEPS)  # how many ranked chunks the caller should fetch
SOURCE_CHARS = 900  # per chunk
CHARS_PER_TOKEN = 1.6  # conservative: measured 1.77 over the law corpus, 1.59 for the token-heaviest 5% of chunks
PROMPT_MARGIN_TOKENS = 150
NOT_FOUND = "ไม่พบข้อมูลในเอกสารที่ให้"
ANSWER_CTX = 4096  # keep small: the KV cache must fit next to the embedding model on a 6 GB GPU
ANSWER_MAX_TOKENS = 450
REQUEST_TIMEOUT = 240
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")  # how long the model stays in VRAM after the last request

SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามจากเอกสารที่ให้ ตอบจากแหล่งข้อมูลที่ให้เท่านั้น ห้ามใช้ความรู้นอกเหนือจากนั้น "
    "อ้างอิงแหล่งที่มาด้วยเลขในวงเล็บเหลี่ยม เช่น [1] [2] ตามหมายเลขของแหล่งข้อมูล "
    "ห้ามเพิ่มความรู้นอกเหนือจากแหล่งข้อมูลแม้เพียงเล็กน้อย "
    "ถ้าผู้ใช้พิมพ์เป็นข้อความยืนยัน ไม่ใช่คำถาม ให้ตอบว่าแหล่งข้อมูลสนับสนุน ขัดแย้ง หรือสนับสนุนบางส่วน พร้อมอ้างอิง [n] และสรุปสิ่งที่แหล่งข้อมูลพูดถึงเรื่องนั้น "
    "ถ้าแหล่งข้อมูลไม่พูดถึงเรื่องที่ถามเลยให้ตอบเพียงประโยคเดียวว่า 'ไม่พบข้อมูลในเอกสารที่ให้' โดยไม่ต้องอธิบายเพิ่ม "
    "ตอบเป็นภาษาไทยให้กระชับ /no_think"
)


class LlmError(RuntimeError):
    def __init__(self, message: str, status: int = 503) -> None:
        super().__init__(message)
        self.status = status


class ChatGate:
    """Chat is allowed only when no document is queued or being processed."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.pending = 0  # jobs queued or running
        self.active = 0  # answers being generated

    def job_added(self) -> None:
        with self._cond:
            self.pending += 1

    def job_finished(self) -> None:
        with self._cond:
            self.pending = max(0, self.pending - 1)
            self._cond.notify_all()

    def start_answer(self) -> bool:
        with self._cond:
            if self.pending:
                return False
            self.active += 1
            return True

    def end_answer(self) -> None:
        with self._cond:
            self.active = max(0, self.active - 1)
            self._cond.notify_all()

    def wait_for_answers(self, timeout: float = 180.0) -> bool:
        """Block until no answer is being generated (an ingestion job calls this before it unloads the model)."""
        with self._cond:
            return self._cond.wait_for(lambda: self.active == 0, timeout)


GATE = ChatGate()


def _post(path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def unload_model() -> bool:
    """Ask Ollama to drop the model from VRAM. Never raises: if Ollama is not running there is nothing to free."""
    try:
        _post("/api/generate", {"model": OLLAMA_MODEL, "keep_alive": 0, "prompt": ""}, timeout=10)
        return True
    except Exception:  # noqa: BLE001 - best effort
        return False


def warm_model() -> bool:
    """Load the model into VRAM now (empty prompt) so the first question does not pay the load time.
    Uses the same num_ctx as real requests so Ollama does not reload it. Never raises."""
    try:
        _post("/api/generate", {"model": OLLAMA_MODEL, "prompt": "", "keep_alive": KEEP_ALIVE,
                                "options": {"num_ctx": ANSWER_CTX}}, timeout=REQUEST_TIMEOUT)
        return True
    except Exception:  # noqa: BLE001 - Ollama not running yet is fine; the first question will load it
        return False


def build_messages(question: str, sources: list[dict]) -> list[dict]:
    blocks = []
    for n, s in enumerate(sources, 1):
        where = f"{s['law']} {s['section']}".strip()
        blocks.append(f"[{n}] ({where})\n{s['text'][:SOURCE_CHARS]}")
    user = "แหล่งข้อมูล:\n" + "\n\n".join(blocks) + f"\n\nคำถาม: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def strip_think(text: str) -> str:
    """Remove <think>…</think> blocks, including an unterminated one (the answer was cut off while 'thinking')."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" not in text:
        # with /no_think this model opens an EMPTY block ("<think>\n\n") and then answers without ever closing it
        text = re.sub(r"^\s*<think>[ \t\r]*\n[ \t\r]*\n", "", text)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


def ask(question: str, sources: list[dict]) -> dict:
    payload = {
        "model": OLLAMA_MODEL, "stream": False, "think": False, "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": ANSWER_CTX, "num_predict": ANSWER_MAX_TOKENS, "temperature": 0.2},
        "messages": build_messages(question, sources),
    }
    t0 = time.time()
    try:
        reply = _post("/api/chat", payload, REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise LlmError(f"ไม่พบโมเดล {OLLAMA_MODEL} ใน Ollama · ดึงด้วย: ollama pull {OLLAMA_MODEL}", 404) from exc
        raise LlmError(f"Ollama ตอบ error {exc.code}", 502) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise LlmError(f"เชื่อมต่อ Ollama ไม่ได้ที่ {OLLAMA_URL} · เปิดแอป Ollama ก่อน", 503) from exc
    raw = reply.get("message", {}).get("content", "")
    answer = strip_think(raw)
    if not answer:
        raise LlmError("โมเดลยังคิดไม่จบ ลองถามอีกครั้ง (หรือถามให้เจาะจงขึ้น)", 502)
    eval_s = (reply.get("eval_duration") or 0) / 1e9
    return {"answer": answer, "model": OLLAMA_MODEL, "seconds": round(time.time() - t0, 1),
            "tokens": reply.get("eval_count", 0), "tok_per_s": round(reply["eval_count"] / eval_s, 1) if eval_s else None}


def fit_sources(question: str, sources: list[dict]) -> list[dict]:
    """Keep the best-ranked sources that fit the context window (always at least one); the lowest ranks are dropped."""
    room_tokens = ANSWER_CTX - ANSWER_MAX_TOKENS - PROMPT_MARGIN_TOKENS
    budget = int(room_tokens * CHARS_PER_TOKEN) - len(SYSTEM_PROMPT) - len(question) - 60
    used, total = [], 0
    for s in sources:
        cost = min(len(s["text"]), SOURCE_CHARS) + 40  # + the "[n] (law section)" label
        if used and total + cost > budget:
            break
        used.append(s)
        total += cost
    return used


def is_refusal(answer: str) -> bool:
    """A refusal says 'not found' and cites nothing; an answer that cites [n] counts as an answer even if it adds a caveat."""
    return NOT_FOUND in answer and not re.search(r"\[\d+\]", answer)


def ask_adaptive(question: str, hits: list[dict]) -> dict:
    """Ask with the first ADAPTIVE_STEPS[0] chunks; while the model refuses, retry with more (next step) until the last step.
    A final refusal is reduced to the bare refusal sentence so no outside knowledge sneaks in after it."""
    t0 = time.time()
    rounds: list[dict] = []
    final: tuple[dict, list[dict]] | None = None
    total_tokens = 0
    for step in sorted(set(s for s in ADAPTIVE_STEPS if s > 0)):
        used = fit_sources(question, hits[: min(step, len(hits))])
        if rounds and len(used) <= rounds[-1]["used"]:
            break  # nothing new to add (too few hits, or the context budget is full)
        if not used:
            break
        reply = ask(question, used)
        refused = is_refusal(reply["answer"])
        rounds.append({"k": step, "used": len(used), "refused": refused, "seconds": reply["seconds"]})
        total_tokens += reply["tokens"]
        final = (reply, used)
        if not refused:
            break
    if final is None:
        return {"answer": NOT_FOUND, "model": OLLAMA_MODEL, "seconds": round(time.time() - t0, 1), "tokens": 0,
                "tok_per_s": None, "used_k": 0, "rounds": [], "sources": []}
    reply, used = final
    answer = NOT_FOUND if rounds[-1]["refused"] else reply["answer"]
    return {"answer": answer, "model": OLLAMA_MODEL, "seconds": round(time.time() - t0, 1), "tokens": total_tokens,
            "tok_per_s": reply.get("tok_per_s"), "used_k": len(used), "rounds": rounds, "sources": used}
