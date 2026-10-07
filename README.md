# MiniPJ_AI_YOLO — Patongko Detection (Brand vs Street)

จำแนกปาท่องโก๋แบรนด์ vs ร้านทั่วไป + สุก/ไหม้ + ทรงดี/ชำรุด ด้วย YOLOv8 Detection (6 คลาส)

**สถานะ (6 ต.ค. 2026):** เทรนทั้งหมด 5 รอบ — รอบ 3 ดีสุด (label แก้แล้วครบ 100%) — test **mAP50 0.879**
รายละเอียดใน `report.md` / ขั้นตอนแลปใน `report2.md`

## Dataset

- `mixed2/` — DTPG v2 จาก Roboflow (852 ภาพ: train 713 / valid 93 / test 46, YOLO bbox 512px)
- 6 classes: `brand_golden_defect`, `brand_golden_good`, `street_burnt_defect`, `street_burnt_good`, `street_golden_defect`, `street_golden_good`
  (แผนเดิม 8 คลาส — `brand_burnt_good`/`brand_burnt_defect` 0 ภาพ 0 กล่องทั้งชุด เพราะแบรนด์คุมไฟดี ไม่พบเคสไหม้จริง โมเดล `nc=6` จึงจำแนก 2 คลาสนี้ไม่ได้)
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

## ผลลัพธ์ (เทรน 5 รอบ — รอบ 3 ดีสุด)

- เทรนทั้งหมด 5 รอบ: รอบ1 baseline `yolov8n 50e` val 0.844 / รอบ2 `yolov8n 80e` test 0.867 / **รอบ3 `yolov8n 80e fixed` test 0.879 ดีสุด** / รอบ4 `yolov8s` test 0.872 / รอบ5 oversample `yolov8s 100e` test 0.878
- Val (รอบ3): P 0.806 / R 0.896 / **mAP50 0.895**
- Test: P 0.942 / R 0.857 / **mAP50 0.879**
- Weights: `runs/detect/pathongko_mixed3_fixed/weights/best.pt` (local only)
- จุดอ่อน: `street_burnt_defect` test R 0.6 — เก็บตัวอย่างเพิ่มได้ถ้าจะดันต่อ
- ข้อจำกัด: `brand_burnt_good`/`brand_burnt_defect` ไม่มีข้อมูลเลย — `03/04/06` เปิดกล้องจ่อแบรนด์ไหม้ก็ทายเป็น 1 ใน 6 คลาสที่มีเท่านั้น
