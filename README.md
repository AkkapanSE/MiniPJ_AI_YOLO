# MiniPJ_AI_YOLO — Patongko Detection (Brand vs Street)

**ผู้จัดทำมินิโปรเจค:**
1. นาย เอกพันธ์ ทศทิศรังสรรค์ 67543210050-0 SE-sec1
2. นาย ณฐภาพ สายหล้า 675432154-2 SE-sec1

จำแนกปาท่องโก๋ **แบรนด์ vs ร้านทั่วไป + สุก/ไหม้ + ทรงดี/ชำรุด** ด้วย YOLOv8 Detection (6 คลาส)
ถ่ายวิดีโอหน้าร้านจริง → สกัดเฟรม → label บน Roboflow → เทรน 9 รอบ → โมเดลรอบ 9 ใช้จริง

**สถานะ (8 ต.ค. 2026):** เทรนทั้งหมด 9 รอบ — รอบ 8 test สูงสุด 0.935, รอบ 9 test 0.930 + แก้รูพรุนแล้ว (ใช้จริง)
test **mAP50 0.930 ผ่านเป้า 0.85** — ผลเทสภาพรวมภาพถ่ายใหม่/วิดีโอ/กล้องดูใน **[README2.md](README2.md) 👈 คลิก**

รายละเอียดเต็มใน `report.md` / ขั้นตอนแลปใน `report2.md` / ฉบับทฤษฎี+โค้ดใน `report3.md`

## 1. Dataset `mixed6/` คืออะไร (ใช้จริง)

- `mixed6/` — DTPG v6 จาก Roboflow (1,863 ภาพ: train 1,714 / valid 93 / test 56, YOLO bbox 512px)
- ที่มา: `mixed2/` v2 852 ภาพ + วิดีโอใหม่ 2 คลิปสกัดทุก 15 เฟรม 1,082 ภาพ (กรองว่าง/เบลอ + ตรวจมือบน Roboflow เหลือ 1,011 ภาพ)
- v5→v6 แก้ label รูพรุน `brand_good→defect` (+97) + เกลี่ย street ใหม่ (valid/test ชุดเดิม)
- `raw_images/` — ภาพดิบ + วิดีโอ + ชุดเทสครู + `RP_images/` ภาพหลักฐาน (local only, ไม่ push ยกเว้น `RP_images/`)

6 classes ที่เทรนจริง (`nc=6`, นับกล่องชุด train):

| id | class | train | valid | test | หมายเหตุ |
|----|---|---|---|---|---|
| 0 | brand_golden_defect | 244 | 18 | 10 | แก้รูพรุนเพิ่มจาก 147 |
| 1 | brand_golden_good | 574 | 26 | 17 | ลดจาก 808 (ย้ายไป defect) |
| 2 | street_burnt_defect | 751 | 23 | 26 | จาก 191 — จุดอ่อนเดิมหายแล้ว (R 1.0) |
| 3 | street_burnt_good | 406 | 9 | 8 | จาก 47 — ตอนนี้มีใน test แล้ว |
| 4 | street_golden_defect | 335 | 25 | 13 | - |
| 5 | street_golden_good | 463 | 26 | 13 | - |

> แผนเดิม 8 คลาส แต่ `brand_burnt_good`/`brand_burnt_defect` **0 ภาพ 0 กล่องทั้งชุด** เพราะแบรนด์คุมไฟดี ไม่พบเคสไหม้จริง
> โมเดล `nc=6` จึงจำแนก 2 คลาสนี้ไม่ได้ — เอาแบรนด์ไหม้จ่อกล้องก็ทายเป็น 1 ใน 6 ที่มีเท่านั้น

![สกัดเฟรมรอบแรก ทุก 15 เฟรม](<raw_images/RP_images/w1.png>)
![สกัดเฟรมรอบสอง ทุก 10 เฟรม](<raw_images/RP_images/w2.png>)
![ตีกรอบบน Roboflow](<raw_images/RP_images/label.png>)
![สถิติชุดข้อมูล](<raw_images/RP_images/labe2.png>)

## 2. สคริปต์ทำอะไรบ้าง (รันตามลำดับ 01→07)

รันด้วย venv: `.\.env\Scripts\python.exe <สคริปต์>` | env: Python 3.13 + torch cu126 + ultralytics 8.4 (GPU RTX 4060)

| ไฟล์ | หน้าที่แบบเข้าใจง่าย | วิธีรัน |
|---|---|---|
| `01-extract/_frames.py` | เปิดวิดีโอด้วย OpenCV เซฟทุก 15 เฟรม | `python 01-extract/_frames.py` |
| `02-train/02-train-mixed5.py` | fine-tune จาก `best_round3` บน `mixed5` ได้รอบ 8 (`cls=1.0` + `copy_paste/mixup`, `patience=20`) | `python 02-train/02-train-mixed5.py` |
| `02-train/02-train-mixed6.py` | fine-tune จาก `best_round8` บน `mixed6` ได้รอบ 9 (สูตรเดิม) | `python 02-train/02-train-mixed6.py` |
| `02-train/oversample.py` | ก็อปภาพ train คลาสน้อย (ของเก่ารอบ 5) | `python 02-train/oversample.py` |
| `07-pseudolabel/pseudo_label.py` | เอา best รอบแรกทายภาพดิบ `conf 0.5` ทำร่าง label (กัน leakage ชื่อซ้ำ) | `python 07-pseudolabel/pseudo_label.py` |
| `03-predict.py` | ทายภาพนิ่ง default `mixed5/test/images` 56 ภาพ (conf 0.5, `--w 8/9`) เซฟภาพ+txt | `python 03-predict.py [path] [conf] [imgsz] [--w 8]` |
| `04-predict_video.py` | ทายวิดีโอแบบ `stream=True` ทีละเฟรม กันแรมเต็ม (รอบ 9) | `python 04-predict_video.py [path]` |
| `05-evaluate.py` + `eval_utils.py` | `model.val(split=test)` ตาราง P/R/F1/mAP รายคลาส + confusion matrix + กราฟทีละภาพ | `python 05-evaluate.py` |
| `06-webcam_realtime.py` | เปิดกล้องทาย realtime รอบ 9 กด `q` ออก | `python 06-webcam_realtime.py [0]` |

## 3. ผลลัพธ์ (เทรน 9 รอบ — รอบ 9 ใช้จริง)

- รอบ1-5 (mixed2): baseline 50e → **รอบ3 ดีสุด test 0.879** → รอบ4 `yolov8s` 0.872 (overfit) → รอบ5 oversample 0.878
- รอบ6-7 (mixed3): test ~0.917 → รอบ8 (mixed5, fine-tune จากรอบ3) **test 0.935 สูงสุด** → **รอบ9 (mixed6, แก้รูพรุน) test 0.930 ใช้จริง**
- **รอบ 9:** Val P 0.86 / R 0.803 / **mAP50 0.881** | Test P 0.967 / R 0.929 / F1 0.947 / **mAP50 0.930**
- รายคลาส test: `brand_defect` 0.895, `brand_good` 0.982, **`burnt_defect` 0.921 (R 1.0)**, **`burnt_good` 0.875** (เคย N/A), `golden_defect` 0.995, `golden_good` 0.915
- Weights: `all_model/best_round9_v6.pt` (local only, ไม่ push)
- เทสของจริงนอกชุดเทรน (ภาพครู C–G / วิดีโอ 7,229 เฟรม / กล้อง realtime): ดู **[README2.md](README2.md)** — ภาพรวมสูงสุด 5/6 คลาส ขาดแค่ `street_burnt_defect` ในภาพครู

![กราฟเทรนรอบ 9](<raw_images/RP_images/v6_results.png>)
![confusion matrix รอบ 9](<raw_images/RP_images/v6_confusion_matrix.png>)
![ผลรัน 05-evaluate (รอบ 3)](<raw_images/RP_images/eval_mean.png>)

## 4. ตัวอย่าง detect (รอบ 9, `mixed5/test` 56 ภาพ ได้ 79 กล่อง)

- ครบ 6 คลาสทุกตัว ≥0.875, เคสหลุดเดิม (`street_0379` 8 ชิ้น) กลับมาจับได้
- 2 ภาพ close-up (`street_0424/0428`) ยังไม่มี detection — ต้องถ่ายระยะเห็นชิ้นเต็ม
- ตัวอย่างรอบ 3 (เก็บไว้เทียบ): brand 3 กล่องถูก, street 3 กล่องถูก

![predict brand](<raw_images/RP_images/predict_brand_0213.jpg>)
![predict street](<raw_images/RP_images/predict_street_0100.jpg>)
![เฟรมวิดีโอ 4 คลาส](<raw_images/RP_images/RP_f04559_4cls_5box.jpg>)

## 5. สรุปเป้าหมาย สำเร็จ / ไม่สำเร็จ

| เป้าหมาย | เกณฑ์ | ผลจริง | สถานะ |
|---|---|---|---|
| 1. ชุดข้อมูล 6 คลาสจากวิดีโอจริง | ครบ train/valid/test | 1,863 ภาพ (1,714/93/56) + label ครบทุกรูป | ✅ สำเร็จ |
| 2. เทรนผ่านเป้า | test mAP50 > 0.85 | รอบ 9 test **0.930** (สูงสุดรอบ 8: 0.935) | ✅ สำเร็จ |
| 3. ประเมิน + ใช้งานจริง | P/R/F1/mAP + ภาพนิ่ง/วิดีโอ/webcam | test ครบ + วิดีโอ 70% เฟรม + กล้อง realtime 6/7 ใบ — ดู [README2.md](README2.md) | ✅ สำเร็จ |
| 4. ครบ 8 คลาสตามแผน | มี `brand_burnt_*` | 0 ภาพ 0 กล่อง (แบรนด์ไม่ไหม้จริง) จำแนกไม่ได้ | ❌ ไม่สำเร็จ (ข้อจำกัดข้อมูล) |
| 5. ดันคลาสไหม้ | `street_burnt_defect` R ดี + มี `street_burnt_good` ใน test | `burnt_defect` 0.921 (R 1.0) + คลาส 3 ใน test 8 กล่อง 0.875 | ✅ สำเร็จ (รอบ 9) |
| 6. ภาพรวมครบนอกชุดเทรน | ภาพถ่ายใหม่ขึ้นครบ 6 คลาส | สูงสุด 5/6 (ขาด `street_burnt_defect` ในภาพครู) — ดู [README2.md](README2.md) | 🟡 เหลือชิ้นไหม้ทรงเสีย 1 ชิ้น |
