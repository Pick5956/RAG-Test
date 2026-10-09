# RAG Lab

ต้นแบบระบบ **ค้นหาเอกสารด้วยความหมาย (RAG ฝั่ง Retrieval)** ที่รันในเครื่องตัวเองทั้งหมด ไม่ส่งข้อมูลออกอินเทอร์เน็ต

- ค้น **กฎหมายไทย 36 ฉบับ** (ชุดข้อมูล NitiBench-Statute, 5,127 มาตรา) ด้วยโมเดล embedding `Qwen3-Embedding-0.6B`
- **อัปโหลดเอกสารของตัวเอง** (ไฟล์หรือทั้งโฟลเดอร์) แล้วระบบดึงข้อความ → หั่น → สร้างเวกเตอร์ → เพิ่มเป็นจุดบน **แผนที่ความหมาย** และค้นหาเจอทันที
- รองรับ: `pdf` `txt` `md` `csv` `json` `html` `xml` `yaml` `docx` `pptx` `xlsx` และรูปภาพ (`png` `jpg` `tif` `bmp` `webp`) ถ้าเป็นสแกนหรือรูปจะอ่านด้วย OCR (PP-OCRv5 ภาษาไทย)

## ทำถึงไหนแล้ว / ยังไม่ได้ทำ

| ส่วน | สถานะ |
|---|---|
| Retrieval: หั่นเอกสาร, embedding, ค้นหา, แผนที่ความหมาย | ทำแล้ว |
| อัปโหลดหลายชนิดไฟล์, OCR, คลังเอกสารของผู้ใช้ (ลบ/อัปซ้ำได้) | ทำแล้ว |
| **Generation: ส่งชิ้นที่ค้นเจอให้ LLM ตอบ** | **ยังไม่ได้ทำ** |
| **LLM ท้องถิ่น / Agent** | **ยังไม่ได้ทำ** (ดูหัวข้อ "ต่อยอดเป็น Agent") |

## ความต้องการ

- Windows 10/11 (พัฒนาและทดสอบบน Windows 11), **Python 3.11**
- การ์ดจอ NVIDIA แนะนำ (ทดสอบบน RTX 4050 Laptop VRAM 6 GB) โค้ดฝั่ง embedding มีทางสำรองให้ใช้ CPU ได้แต่ **ยังไม่ได้ทดสอบ** และจะช้ามาก
- พื้นที่ว่างราว 10 GB (PyTorch CUDA ~2.5 GB, โมเดล ~1.2 GB, Paddle GPU ~1 GB ขึ้นไป) และอินเทอร์เน็ตตอนติดตั้งครั้งแรก
- **ที่เก็บโปรเจกต์ต้องเป็น path ภาษาอังกฤษล้วน** (เช่น `C:\work\rag-lab`) PaddleOCR โหลดโมเดลจาก path ที่มีอักษรไทยไม่ได้ (เจอจริงและยืนยันแล้ว)

## เริ่มใช้งาน

### ทางลัด (สคริปต์)

```powershell
git clone https://github.com/Pick5956/RAG-Test.git rag-lab
cd rag-lab
powershell -ExecutionPolicy Bypass -File setup.ps1 -GpuOcr     # ดูแผนก่อนด้วย -DryRun
powershell -ExecutionPolicy Bypass -File run.ps1
```

เปิด http://127.0.0.1:8765

ตัวเลือกของ `setup.ps1`: `-GpuOcr` (venv แยกสำหรับ OCR บน GPU, เร็ว) · `-CpuOcr` (OCR บน CPU ใน venv หลัก, ช้า) · `-SkipData` (ข้ามการสร้างข้อมูลกฎหมาย) · `-DryRun` (พิมพ์คำสั่งทั้งหมดโดยไม่รัน) ถ้าไม่ใส่ `-GpuOcr`/`-CpuOcr` แอปใช้งานได้กับไฟล์ข้อความทั้งหมด แต่ PDF สแกนและรูปภาพจะขึ้นข้อผิดพลาด

### ทำเองทีละคำสั่ง

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # ไม่มี NVIDIA: ตัด --index-url ออก
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# โมเดล embedding (~1.2 GB)
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-Embedding-0.6B', local_dir='Qwen3-Embedding-0.6B')"

# ข้อมูลกฎหมาย: ดาวน์โหลด ~2 MB แล้วหั่นเป็นชิ้น -> corpus.jsonl, จากนั้นสร้างเวกเตอร์ -> embeddings.npy
.\.venv\Scripts\python.exe build_corpus.py
.\.venv\Scripts\python.exe embed_corpus.py 16

# OCR แบบ GPU (ทางเลือก)
python -m venv .venv-paddle
.\.venv-paddle\Scripts\python.exe -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
.\.venv-paddle\Scripts\python.exe -m pip install paddleocr==3.7.0

.\.venv\Scripts\python.exe app.py
```

หมายเหตุ: `embed_corpus.py 16` ใช้ batch 16 (ลดเป็น 8 ถ้า VRAM ไม่พอ) ครั้งแรกที่ Paddle ทำงานจะดาวน์โหลดโมเดล OCR ขนาดเล็กลง `paddle_models/` เอง

## การใช้งาน

- **แท็บ "ค้นหา · แผนที่":** พิมพ์คำถาม ได้ 10 ชิ้นที่ใกล้ที่สุด แผนที่ซูมไปที่ผลลัพธ์ให้เอง ชี้เมาส์ดูข้อความ คลิกจุดเพื่อปักหมุด ซูมด้วยล้อเมาส์ ลากเพื่อเลื่อน คลิกชื่อกฎหมายด้านล่างเพื่อเน้นกลุ่มนั้น (จุดที่มีวงขาว = เอกสารที่อัปโหลด)
- **แท็บ "อัปโหลดเอกสาร":** ลากไฟล์/โฟลเดอร์มาวาง ระบบประมวลผลทีละไฟล์ตามคิว เสร็จแล้วเป็นจุดใหม่บนแผนที่ทันที PDF ตัดสินใจรายหน้า (มีข้อความฝังก็ดึงตรง ถ้าเป็นรูปหรือข้อความน้อยกว่า 50 ตัวอักษรหรือรูปกินหน้า ≥ 20% จึงใช้ OCR) มีคลังเอกสารให้ดูและลบ
- ขีดจำกัด: 100 MB ต่อไฟล์, PDF ไม่เกิน 300 หน้า, ไม่เกิน 5,000 ชิ้นต่อไฟล์, `.doc` `.ppt` `.xls` แบบเก่าไม่รองรับ (ต้องบันทึกเป็น `.docx` `.pptx` `.xlsx` ก่อน)

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | เซิร์ฟเวอร์เดียวจบ: ดัชนีค้นหา, คิวอัปโหลด, OCR, HTTP API |
| `app_page.html` | หน้าเว็บ (ค้นหา/แผนที่/อัปโหลด) |
| `extractors.py` | ตัวอ่าน txt/csv/html/docx/pptx/xlsx (ใช้ไลบรารีมาตรฐานล้วน) |
| `ocr_worker.py` | ตัวอ่าน OCR รันเป็นโปรเซสลูก (แยกเพื่อไม่ให้ paddle ชนกับ torch) |
| `route_pdf.py` | ตรวจรายหน้าของ PDF ว่าใช้ข้อความฝังหรือ OCR (มี CLI ในตัว) |
| `build_corpus.py` | ดาวน์โหลด NitiBench-Statute → หั่น → `corpus.jsonl` |
| `embed_corpus.py` | `corpus.jsonl` → `embeddings.npy` |
| `setup.ps1` / `run.ps1` | ติดตั้ง / เปิดแอป |
| `requirements*.txt` | แพ็กเกจที่ล็อกเวอร์ชันตามที่ทดสอบ |

ข้อมูลที่ถูกสร้างขึ้นเอง (ไม่ถูก commit): `.venv*/`, `Qwen3-Embedding-0.6B/`, `paddle_models/`, `corpus.jsonl`, `embeddings.npy`, `uploads/`, `route_out/`, `user_docs.jsonl`, `user_vecs.npy` ข้อมูลกฎหมายกับเอกสารผู้ใช้เก็บแยกไฟล์กัน รัน `build_corpus.py`/`embed_corpus.py` ใหม่แล้วเอกสารที่อัปโหลดไม่หาย

## HTTP API (สำหรับต่อกับ Agent)

เซิร์ฟเวอร์ผูกที่ `127.0.0.1` เท่านั้นและ **ไม่มีระบบยืนยันตัวตน** อย่าเปิดสู่ภายนอก

| คำขอ | ผลลัพธ์ |
|---|---|
| `GET /api/search?q=<คำถาม>` | `{"results": [{"i", "score", "law", "section", "user", "text"}]}` 10 รายการ (`law` = ชื่อกฎหมายหรือชื่อไฟล์, `section` = "มาตรา N" / "หน้า N" / "สไลด์ N", `user` = true ถ้ามาจากเอกสารที่อัปโหลด) |
| `GET /api/doc?i=<i>` | `{"law", "section", "text"}` ข้อความเต็มของชิ้นนั้น |
| `POST /api/upload?name=<path/ชื่อไฟล์>` | body = ไบต์ของไฟล์ → `{"job", "name"}` |
| `GET /api/jobs?ids=a,b` | สถานะงานอัปโหลด |
| `GET /api/library` · `POST /api/delete?name=` | รายการ/ลบเอกสารในคลัง |
| `GET /api/points` | ข้อมูลแผนที่ทั้งหมด |

ตัวอย่างเรียกจาก Python:

```python
import json, urllib.parse, urllib.request
q = urllib.parse.quote("หลักเกณฑ์การจัดซื้อจัดจ้างภาครัฐ")
hits = json.load(urllib.request.urlopen(f"http://127.0.0.1:8765/api/search?q={q}"))["results"]
for h in hits[:3]:
    print(h["score"], h["law"], h["section"], h["text"][:80])
```

## ต่อยอดเป็น Agent

ส่วนที่ขาดคือ **Generation** แนวทางที่ต่อได้ง่ายจากของที่มี (ยังไม่ได้ทำ ไม่ได้ทดสอบ):

1. รัน LLM ในเครื่อง (เช่น Ollama หรือ llama.cpp) แล้วเขียนตัวกลางที่เรียก `GET /api/search` → นำ `text` + `law` + `section` ของ top-k ใส่ในพรอมต์ → ให้ LLM ตอบโดยอ้างมาตรา/หน้า และตอบว่า "ไม่พบข้อมูล" ถ้าบริบทไม่พอ
2. ทำเป็น **เครื่องมือ (tool)** ของ Agent: `search_documents(query)` เรียก API ข้างบน และ `read_chunk(i)` เรียก `/api/doc`
3. จุดเสียบในโค้ด: `SearchIndex.search()` ใน `app.py` คืนชิ้นที่ใกล้ที่สุดพร้อมคะแนน ส่วนการหั่นอยู่ที่ `SPLITTER` (800 ตัวอักษร ทับซ้อน 100) ถ้าเปลี่ยนต้องสร้าง embedding ใหม่

## ข้อควรรู้ / ปัญหาที่เคยเจอ

- **Path ภาษาไทยทำให้ PaddleOCR พัง** (ยืนยันจากการทดลองควบคุมตัวแปร) ใช้ path อังกฤษล้วน
- **paddle กับ torch อยู่ในโปรเซสเดียวกันบน Windows แล้วชนกันได้** (เจอตอน import paddle ก่อน torch) จึงแยก OCR เป็นโปรเซสลูก อย่านำ `import paddle` ไปไว้ใน `app.py`
- ตอน OCR บน GPU อาจเห็นคำเตือน cuDNN เวอร์ชันไม่ตรงใน log ในเครื่องที่ทดสอบ PP-OCRv5 ยังทำงานได้ปกติ
- **คุณภาพ OCR ภาษาไทย:** PP-OCRv5 อ่านได้ใช้งานได้ แต่มีข้อผิดพลาด เช่น สระ/วรรณยุกต์หาย และตัวอักษรไทยแทรกในคำอังกฤษ จากการทดสอบเล็กๆ (17 คำถาม 11 หน้า) ข้อผิดพลาดแบบนี้กระทบการค้นหาน้อย แต่ **ไม่ใช่การพิสูจน์** สำหรับเอกสารจริงจำนวนมาก
- ข้อความ OCR ที่อ่านผิดจะเข้าไปอยู่ในระบบค้นหาด้วย เอกสารที่ต้องแม่นยำควรให้คนตรวจเทียบต้นฉบับ
- OCR บน CPU ช้า (~10 วินาทีต่อหน้า) บน GPU เร็วกว่าหลายเท่า (ทดสอบ PDF 11 หน้าเสร็จในไม่กี่สิบวินาที)

## สิ่งที่ยังไม่ได้ทดสอบ (ตรงไปตรงมา)

- `setup.ps1` **ยังไม่เคยรันครบบนเครื่องสะอาด** (ตรวจแล้วด้วย `-DryRun` และตรวจไวยากรณ์) ถ้าติดขัดตรงไหนแจ้งได้ การติดตั้ง Paddle GPU ใน venv ใหม่แบบ `paddleocr` เปล่าๆ ก็ยังไม่ได้ทดสอบจากศูนย์
- Linux/macOS และโหมด CPU ล้วนของ embedding ยังไม่ได้ทดสอบ
- ตัวอ่าน `docx/pptx/xlsx` ทดสอบกับไฟล์ที่สร้างแบบย่อ ไม่ใช่ไฟล์จาก Word/PowerPoint/Excel จริง ส่วนที่ยังไม่ดึง: หัว-ท้ายกระดาษ กล่องข้อความ คอมเมนต์ ข้อความในรูป
- การเลือกไฟล์/ลากทั้งโฟลเดอร์ในเบราว์เซอร์ ทดสอบผ่าน API เท่านั้น ส่วน UI ของคิวอัปโหลดยังไม่ได้ทดสอบด้วยการใช้งานจริง

## ข้อมูลและใบอนุญาต

- ข้อมูลกฎหมาย: [NitiBench-Statute](https://huggingface.co/datasets/VISAI-AI/nitibench-statute) (การ์ดระบุใบอนุญาต MIT) มาจากกฎหมายที่เผยแพร่สาธารณะ
- โมเดล: `Qwen/Qwen3-Embedding-0.6B` ตรวจเงื่อนไขการใช้งานบนหน้าโมเดลก่อนนำไปใช้เชิงธุรกิจ
- **อย่า commit เอกสารจริงขององค์กร** โฟลเดอร์ `uploads/` และไฟล์คลังผู้ใช้ถูกใส่ใน `.gitignore` ไว้แล้ว
- ยังไม่ได้ระบุใบอนุญาตของโค้ดนี้ (ผู้ดูแล repo ควรเลือกและเพิ่มไฟล์ `LICENSE`)
