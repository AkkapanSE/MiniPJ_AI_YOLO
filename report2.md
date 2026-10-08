# เอกสารแลป: Patongko Detection ด้วย YOLOv8 (6 คลาส, 9 รอบเทรน)

**ผู้จัดทำ:** 1. นาย เอกพันธ์ ทศทิศรังสรรค์ 67543210050-0 SE-sec1
2. นาย ณฐภาพ สายหล้า 675432154-2 SE-sec1
**วิชา/โครงงาน:** MiniPJ_AI_YOLO — จำแนกปาท่องโก๋แบรนด์ vs ร้านทั่วไป (สุก/ไหม้ × ทรงดี/ชำรุด)
**เครื่องมือ:** Python 3.13 + PyTorch cu126 + Ultralytics 8.4 + Roboflow (GPU RTX 4060 Laptop)
**รันทุกสคริปต์ด้วย:** `.\.env\Scripts\python.exe <สคริปต์>`

---

## 1. วัตถุประสงค์

1. สร้างชุดข้อมูล YOLO 6 คลาสจากวิดีโอหน้าร้านจริง (เป้า: ครบ train/valid/test)
2. เทรน YOLOv8 ให้ได้ test mAP50 > 0.85 (ได้ 0.930 รอบ 9 ✅)
3. ประเมินผล (P/R/F1/mAP + confusion matrix) และทดสอบภาพนิ่ง/วิดีโอ/webcam/ภาพครูนอกชุดเทรน

## 2. ข้อมูล

| ชุด | ที่มา | จำนวน |
|---|---|---|
| วิดีโอดิบ | `raw_images/vdo/` (เก่า 1004 ×3 + ใหม่ 813113849 ×2) | 5 คลิป |
| ภาพดิบใหม่ | `01-extract/_frames.py` สกัดทุก 15 เฟรม | 1,082 ใบ (`new_017632` 482 + `new_101088` 600) |
| `mixed2/` (v2) | DTPG v2 = v1 480 + pseudo 524 ผ่าน review | 852 ภาพ (รอบ 1-5) |
| `mixed5/` (v5) | DTPG v5 = mixed2 + ภาพใหม่ 1,011 ผ่าน review | 1,863 ภาพ (รอบ 8) |
| `mixed6/` (v6, **ชุดหลัก**) | DTPG v6 = v5 แก้รูพรุน `good→defect` + เกลี่ย street | **1,863 ภาพ** (train 1,714 / valid 93 / test 56) |

train ต่อคลาส (v6): `0:244 / 1:574 / 2:751 / 3:406 / 4:335 / 5:463`
(6 คลาส: `brand_golden_defect/good`, `street_burnt_defect/good`, `street_golden_defect/good`
— แผนเดิม 8 คลาส แต่ `brand_burnt_*` 0 ภาพ 0 กล่องเพราะแบรนด์คุมไฟดี โมเดล `nc=6` จำแนกไม่ได้)

## 3. ขั้นตอนการทดลอง

### 3.1 เตรียมข้อมูล (`01-extract` → Roboflow)
สกัดเฟรมคลิปใหม่ → pseudo-label ด้วย best รอบ 3 (conf 0.5, ได้ร่าง 659 กล่อง) →
อัปโหลด 1,082 ภาพ + label ร่าง → **ตรวจมือแก้บน Roboflow** → export v5
(พบ pseudo เอนเอียง `brand_good` 76% — ต้องแก้ก่อนเทรน ไม่งั้นโมเดลจำว่า "ทุกอย่างคือของดี")

### 3.2 เทรนรอบ 1-5 (ยุค mixed2)
baseline `yolov8n 50e` (val 0.844) → รอบ 2-3 `80e + cls=1.0/copy_paste/mixup` (test 0.867→**0.879**)
→ รอบ 4 `yolov8s` overfit → รอบ 5 oversample ไม่ช่วย — สรุปใช้รอบ 3

### 3.3 เทรนรอบ 6-9 (ยุค mixed3-6)
รอบ 6-7 ตัวกลาง (test ~0.917) → **รอบ 8** fine-tune จาก `best_round3` บน mixed5 (test **0.935** สูงสุด)
→ พบรูพรุนถูกสอนเป็น `good` → แก้ label v6 → **รอบ 9** fine-tune จาก `best_round8` (test **0.930** ✅ใช้จริง)

### 3.4 ประเมิน (`05-evaluate.py` + `eval_utils.py`)
ตาราง P/R/F1/mAP50/mAP50-95 รายคลาส, confusion matrix custom, กราฟทีละภาพ
(รอบ 9: P 0.967 / R 0.929 / F1 0.947 / mAP50 **0.930** — ทุกคลาส ≥0.875)

### 3.5 ทดสอบใช้งาน (ดูเต็มใน `README2.md`)
`03-predict.py` (ภาพนิ่ง, conf 0.5) / `04-predict_video.py` (วิดีโอ stream 7,229 เฟรม เจอ 70%) /
`06-webcam_realtime.py` (webcam ถูก 6/7 ใบ) / ภาพครู C–G (สูงสุด 5/6 คลาส)

## 4. ผลการทดลอง (รอบ 9 ใช้จริง)

| ชุด | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| valid (93) | 0.86 | 0.803 | ~0.83 | **0.881** | 0.851 |
| test (56) | 0.967 | 0.929 | 0.947 | **0.930** | 0.903 |

รายคลาส test: defect 0.895, good 0.982, **burnt_def 0.921 (R 1.0 — จุดอ่อนเดิม 0.605 หายแล้ว)**,
**burnt_good 0.875 (เคยไม่มีใน test)**, golden_def 0.995, golden_good 0.915 — ผ่านเป้า 0.85 ✅

## 5. บั๊กที่พบและการแก้ไข

1. **label ผสม bbox + polygon (v2):** ultralytics ทิ้ง polygon เงียบ → แปลงเป็น bbox 363 ไฟล์
2. **class เพี้ยน `['2','3',...]` (v4):** export ผิด order → rename + เรียงใหม่ตั้งแต่ v5
3. **pseudo bias brand_good 76%:** ตรวจมือก่อนเทรน (บทเรียนหลักของรอบ 8)
4. **รูพรุน = good (v5):** นิยามไม่ตรงกับครู → แก้ `good→defect` 97 กล่องใน v6 → รอบ 9 พลิกถูกบนของจริง (WC2: defect 0.87)
5. **`street_0379` ปน segment+box:** โดนข้ามตอน eval — รอแก้บน Roboflow (ไม่บล็อกผล)

## 6. สรุปผล

เทรนทั้งหมด **9 รอบ** ได้โมเดลรอบ 9 `all_model/best_round9_v6.pt`
(val 0.881 / test **0.930**) ผ่านเป้าทุกข้อ ยกเว้น `brand_burnt_*` (0 ข้อมูล — ข้อจำกัดถาวร)
งานที่เหลือข้อเดียว: ภาพ `street_burnt_defect` 1 ใบจากครูให้เทสครบนอกชุดเทรน 6/6 (ตอนนี้ 5/6)

## ภาคผนวก: คำสั่งที่ใช้บ่อย

```powershell
.\.env\Scripts\python.exe 02-train/02-train-mixed6.py  # เทรนรอบ 9
.\.env\Scripts\python.exe 05-evaluate.py                 # ประเมิน + กราฟ
.\.env\Scripts\python.exe 03-predict.py <path> 0.5 960 --tta  # ทดสอบภาพ (ครบสุด)
.\.env\Scripts\python.exe 04-predict_video.py <path>    # ทดสอบวิดีโอ
.\.env\Scripts\python.exe 06-webcam_realtime.py 0       # webcam (q=ออก)
```
