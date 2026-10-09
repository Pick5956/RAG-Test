"""OCR worker (child process of app.py): PP-OCRv5 Thai on page images.

usage: python ocr_worker.py DEVICE "1|C:\\path\\page_1.png" "3|C:\\path\\page_3.png" ...
  DEVICE = gpu | cpu
Prints one line per finished page:  @@RESULT@@ {"page": N, "text": "...", "confidence": 0.93}
Runs in its own process so paddle and torch never share a process (DLL conflict on Windows).
Needs only paddleocr (no pymupdf), so it can run from .venv-paddle (paddlepaddle-gpu).
"""
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(HERE / "paddle_models"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

MARK = "@@RESULT@@ "
DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "th_PP-OCRv5_mobile_rec"


def main() -> None:
    device = sys.argv[1]
    items = [(int(a.split("|", 1)[0]), a.split("|", 1)[1]) for a in sys.argv[2:]]

    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        device=device,
        text_detection_model_name=DET_MODEL,
        text_recognition_model_name=REC_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    for number, image_path in items:
        res = engine.predict(image_path)[0]
        scores = [float(s) for s in res["rec_scores"]]
        payload = {
            "page": number,
            "text": "\n".join(res["rec_texts"]),
            "confidence": round(statistics.fmean(scores), 3) if scores else None,
        }
        print(MARK + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
