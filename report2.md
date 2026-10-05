# เอกสารแลป: Patongko Detection ด้วย YOLOv8 (6 คลาส)

**วิชา/โครงงาน:** MiniPJ_AI_YOLO — จำแนกปาท่องโก๋แบรนด์ vs ร้านทั่วไป (สุก/ไหม้ × ทรงดี/ชำรุด)
**เครื่องมือ:** Python 3.13 + PyTorch cu126 + Ultralytics 8.4 + Roboflow (GPU RTX 4060 Laptop)
**รันทุกสคริปต์ด้วย:** `.\.env\Scripts\python.exe <สคริปต์>`

---

## 1. วัตถุประสงค์

1. สร้างชุดข้อมูล YOLO 6 คลาสจากวิดีโอหน้าร้านจริง
2. เทรน YOLOv8n ให้ได้ test mAP50 > 0.85
3. ประเมินผล (P/R/F1/mAP + confusion matrix) และทดสอบภาพนิ่ง/วิดีโอ/webcam

## 2. ข้อมูล

| ชุด | ที่มา | จำนวน |
|---|---|---|
| วิดีโอดิบ | `raw_images/vdo/` (brand 110MB + street 247MB) | 2 คลิป |
| ภาพดิบ | `01-extract/_frames.py` สกัดทุก 10 เฟรม | 1,322 ใบ (local only) |
| `mixed/` (v1) | Roboflow DTPG v1, label บนเว็บ | 480 ภาพ (superseded) |
| `mixed2/` (v2, ชุดหลัก) | DTPG v2 = v1 + pseudo-label 524 ภาพที่ผ่าน review | **852 ภาพ** (train 713 / valid 93 / test 46) |

6 คลาส: `brand_golden_defect/good`, `street_burnt_defect/good`, `street_golden_defect/good`
(แผนเดิม 8 คลาส แต่ `brand_burnt_*` ไม่มีตัวอย่างจริง)

## 3. ขั้นตอนการทดลอง

### 3.1 เตรียมข้อมูล (`01-extract`)
สกัดเฟรม → อัปโหลด Roboflow → ตีกรอบ 480 ภาพ → export `mixed/`

### 3.2 เทรน baseline (`02-train/02-train.py` ฉบับแรก)
yolov8n 50e บน `mixed/` → val mAP50 0.844 (อ่อนสุด `street_burnt_defect` 0.591)

### 3.3 เพิ่มข้อมูล (`07-pseudolabel/pseudo_label.py`)
เอา best.pt baseline predict ภาพดิบ conf 0.5 → ได้ร่าง 524 ภาพ (ทิ้งภาพเปล่า 318 + ซ้ำชุดเดิม 480)
→ review บน Roboflow → export `mixed2/` (train 341→713)

### 3.4 เทรนรอบ 2 (`02-train/02-train.py` ฉบับปัจจุบัน)
yolov8n **80e**, `cls=1.0` + `copy_paste=0.3` + `mixup=0.2` ชดเชยคลาสน้อย
(`street_burnt_good` มีแค่ 47 กล่อง เทียบ `street_golden_defect` 298)

### 3.5 ประเมิน (`05-evaluate.py` + `eval_utils.py`)
ตาราง P/R/F1/mAP50/mAP50-95 รายคลาส (คลาสที่ไม่มีใน test ขึ้น N/A),
confusion matrix custom (heatmap จำนวนดิบ + %), กราฟ matplotlib ทีละภาพกลางจอ

### 3.6 ทดสอบใช้งาน
`03-predict.py` (ภาพนิ่ง) / `04-predict_video.py` (วิดีโอ, stream กันแรมเต็ม) /
`06-webcam_realtime.py` (webcam, q=ออก)

## 4. ผลการทดลอง

| ชุด | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| valid (93) | 0.806 | 0.896 | ~0.85 | **0.895** | 0.868 |
| test (46) | 0.942 | 0.857 | ~0.90 | **0.879** | 0.862 |

รายคลาส (test รอบ 3): brand ทั้งคู่ ~0.9+, `street_burnt_defect` 0.605 (R 0.6 — จุดอ่อนใหม่),
`street_golden_defect` 0.995, `street_golden_good` 0.909 — ผ่านเป้า 0.85 ✅
จุดอ่อนคงเหลือ: `street_burnt_good` (ตัวอย่างน้อย) แต่ recall ดีขึ้น 0.667→0.889 แล้ว

## 5. บั๊กที่พบและการแก้ไข

**label ผสม bbox + polygon:** `mixed2` มี polygon ปน (train 262 / valid 69 / test 32 ไฟล์)
ultralytics เจอไฟล์ผสมแล้วทิ้ง polygon เงียบๆ → เทรน/วัดใช้ข้อมูลไม่ครบ
(เลข test mAP50 0.935 รอบแรกวัดแค่ 60 กล่อง) — แก้โดยแปลง polygon→bbox 363 ไฟล์
เลขจริงบนข้อมูลครบคือ test mAP50 **0.867**

## 6. สรุปผล

ได้โมเดลรอบ 3 `runs/detect/pathongko_mixed3_fixed/weights/best.pt` ผ่านเป้า
(val mAP50 0.895 / test 0.879, เทรนบน label ที่แก้แล้วครบ 100%)
งานต่อ: เก็บ `street_burnt_good`/`street_burnt_defect` ฝั่ง test เพิ่มถ้าจะดัน `street_burnt_defect` test 0.605

## ภาคผนวก: คำสั่งที่ใช้บ่อย

```powershell
.\.env\Scripts\python.exe 02-train/02-train.py       # เทรน
.\.env\Scripts\python.exe 05-evaluate.py              # ประเมิน + กราฟ
.\.env\Scripts\python.exe 03-predict.py <path/0>      # ทดสอบภาพ
.\.env\Scripts\python.exe 04-predict_video.py <path>  # ทดสอบวิดีโอ
.\.env\Scripts\python.exe 06-webcam_realtime.py       # webcam
git add -A; git commit -m "<msg>"; git push origin Home2
```
