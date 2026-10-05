# MiniPJ_AI_YOLO — Patongko Detection (Brand vs Street)

จำแนกปาท่องโก๋แบรนด์ vs ร้านทั่วไป + สุก/ไหม้ + ทรงดี/ชำรุด ด้วย YOLOv8 Detection (6 คลาส)

**สถานะ (5 ต.ค. 2026):** Label + Export dataset บน Roboflow เสร็จแล้ว (~70%) — เหลือเทรน + วัดผล ดูรายละเอียดใน `report.md`

## Dataset (พร้อมเทรน)

- Source: Roboflow `DTPG v1` → `mixed/` (480 ภาพ, YOLOv8 format, 512x512 stretch, ไม่มี augment)
- Split: `train 341 / valid 93 / test 46`
- Classes (6): `brand_golden_defect`, `brand_golden_good`, `street_burnt_defect`, `street_burnt_good`, `street_golden_defect`, `street_golden_good`
- แผนเดิม 8 คลาส (ใน `ชื่อ Class บน Roboflow.docx`) แต่ `brand_burnt_*` ไม่มีตัวอย่างจริงจึงเหลือ 6

```
mixed/
├── data.yaml
├── train/images + train/labels (341)
├── valid/images + valid/labels (93)
└── test/images + test/labels (46)
```

## เทรนต่อ (env พร้อมแล้ว: Python 3.13 + ultralytics 8.4 + torch 2.14)

```powershell
.\.env\Scripts\python.exe -m ultralytics detect train model=yolov8n.pt data=mixed/data.yaml epochs=100 imgsz=512 batch=16 patience=20
.\.env\Scripts\python.exe -m ultralytics detect val model=runs/detect/train/weights/best.pt data=mixed/data.yaml
.\.env\Scripts\python.exe -m ultralytics detect predict model=runs/detect/train/weights/best.pt source=mixed/test/images save=True
```

## โครง repo

- `01-extract/_frames.py` — สกัดเฟรมจากวิดีโอ
- `raw_images/` — ภาพดิบ 1,322 + วิดีโอ 2 ไฟล์ (ไม่ commit .mp4)
- `mixed/` — YOLO dataset หลัก
- `streed_pt_images/` — ชุด classification เก่า 231 ภาพ (สำรอง)
- `report.md` — รายงานความคืบหน้าฉบับเต็ม
