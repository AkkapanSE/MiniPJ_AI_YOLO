# MiniPJ_AI_YOLO — Patongko Detection (Brand vs Street)

จำแนกปาท่องโก๋แบรนด์ vs ร้านทั่วไป + สุก/ไหม้ + ทรงดี/ชำรุด ด้วย YOLOv8 Detection (6 คลาส)

**สถานะ (5 ต.ค. 2026):** เทรนรอบ 3 เสร็จ (label แก้แล้วครบ 100%) — test **mAP50 0.879**
รายละเอียดใน `report.md` / ขั้นตอนแลปใน `report2.md`

## Dataset

- `mixed2/` — DTPG v2 จาก Roboflow (852 ภาพ: train 713 / valid 93 / test 46, YOLO bbox 512px)
- 6 classes: `brand_golden_defect`, `brand_golden_good`, `street_burnt_defect`, `street_burnt_good`, `street_golden_defect`, `street_golden_good`
- `raw_images/` — ภาพดิบ 1,322 ใบ + วิดีโอ 2 ไฟล์ (local only, ไม่ push)

## โครงสคริปต์ (ตามลำดับ)

| ไฟล์ | ทำอะไร | วิธีรัน |
|---|---|---|
| `01-extract/_frames.py` | สกัดเฟรมจากวิดีโอ | `python 01-extract/_frames.py` |
| `02-train/02-train.py` | เทรน yolov8n 80e (`cls=1.0` + copy_paste/mixup ชดเชยคลาสน้อย) | `python 02-train/02-train.py` |
| `03-predict.py` | ทดสอบภาพนิ่ง (default: test set) | `python 03-predict.py [path/0]` |
| `04-predict_video.py` | ทดสอบวิดีโอ (สตรีมทีละเฟรม กันแรมเต็ม) | `python 04-predict_video.py [path/0]` |
| `05-evaluate.py` | ตาราง P/R/F1/mAP + confusion matrix + กราฟทีละภาพ | `python 05-evaluate.py` |
| `eval_utils.py` | helper ของ evaluate | — |
| `06-webcam_realtime.py` | realtime ผ่าน webcam (q = ออก) | `python 06-webcam_realtime.py [เลขกล้อง]` |
| `07-pseudolabel/pseudo_label.py` | pseudo-label ภาพดิบด้วย best.pt | `python 07-pseudolabel/pseudo_label.py` |

รันด้วย venv: `.\.env\Scripts\python.exe <สคริปต์>` | env: Python 3.13 + torch cu126 + ultralytics 8.4 (GPU RTX 4060)

## ผลลัพธ์ (รอบ 3)

- Val: P 0.806 / R 0.896 / **mAP50 0.895**
- Test: P 0.942 / R 0.857 / **mAP50 0.879**
- Weights: `runs/detect/pathongko_mixed3_fixed/weights/best.pt` (local only)
- จุดอ่อน: `street_burnt_defect` test R 0.6 — เก็บตัวอย่างเพิ่มได้ถ้าจะดันต่อ
