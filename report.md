# รายงานความคืบหน้าโปรเจกต์ Patongko Detection ด้วย YOLO

**วันที่อัปเดต:** 6 ตุลาคม 2026 (เทรนครบ 5 รอบ — โมเดลดีสุดรอบ 3)
**ที่ตั้งโปรเจกต์:** `C:\Users\chano\MiniPJ_AI_YOLO`
**สถานะโดยรวม:** 🟢 **~95% — เทรน + ประเมินเสร็จ โมเดลรอบ 3 test mAP50 0.879 ผ่านเป้า 0.85 พร้อมใช้**

---

## 1. สรุปสั้นๆ ว่าทำถึงขั้นไหนแล้ว

| Phase | สถานะ | หมายเหตุ (อัปเดต 5 ต.ค. 2026) |
|-------|--------|-------------------------------|
| 1. เก็บวิดีโอดิบ (Data Collection) | ✅ เสร็จ | มี 2 คลิปใน `raw_images/vdo/` (`1004(1).mp4` brand, `1004.mp4` street) |
| 2. สกัดเฟรมจากวิดีโอ (Frame Extraction) | ✅ เสร็จ | `brand1 280 + street1 1,042 = 1,322 ภาพ` ผ่าน `01-extract/_frames.py` |
| 3. ทำความสะอาดข้อมูล / คัดภาพ (Data Cleaning) | ✅ เสร็จบางส่วน (บน Roboflow) | คัดเหลือ 480 ภาพที่ label ได้จริง ที่เหลือเป็นภาพเปล่า/เบลอถูกตัดออกตอนอัปโหลด Roboflow |
| 4. ตีกรอบ Annotation สำหรับ YOLO (Labeling) | ✅ เสร็จ (บน Roboflow) | Label บน Roboflow โปรเจกต์ `dtpg` version 1 ครบ 480 ภาพ ฟอร์แมต YOLOv8 |
| 5. Export YOLO Dataset (`mixed/`) | ✅ เสร็จ | `train 341 / valid 93 / test 46` รวม 480 ภาพ + `data.yaml` 6 คลาส |
| 6. Train / Val YOLO Detection | ✅ เสร็จ (รอบ 3 ดีสุด) | เทรน 5 รอบบน `mixed2/` — รอบ 3 `yolov8n 80e` ดีสุด val mAP50 0.895 / `runs/detect/pathongko_mixed3_fixed/weights/best.pt` (รอบ 4 s 0.89 / รอบ 5 oversample 0.88 ไม่ชนะ — ดู 7.7–7.8) |
| 7. ประเมินผล + ทดสอบ Detection | ✅ เสร็จ | `05-evaluate.py` รายงาน test mAP50 0.879 (P 0.942 / R 0.857 / F1 ~0.90) + confusion matrix + กราฟทีละภาพ; `03/04/06` ทดสอบภาพนิ่ง/วิดีโอ/webcam แล้ว |
| 8. ชุด classification สำรอง `streed_pt_images` | 🟡 มีอยู่ 231 ภาพ | `train 138 / val 46 / test 47` คลาสเดียว `Pa_Thong_Ko` — ไม่ได้ใช้ในสาย YOLO หลักแล้ว |

> **สรุป:** **~95% — Train/Eval เสร็จแล้ว (รอบ 3 ดีสุด test 0.879)** — ข้ามขั้น Label มาแล้ว รายละเอียดรอบ 4–5 ดูข้อ 7.7–7.8

---

## 2. สิ่งที่มีอยู่จริงในโฟลเดอร์ (สำรวจ 5 ต.ค. 2026)

```
MiniPJ_AI_YOLO/
├── 01-extract/
│   └── _frames.py              # สกัดเฟรม OpenCV (frame_interval=15 รอบแรก, =10 รอบสอง)
├── raw_images/
│   ├── vdo/ (2 ไฟล์ ~358 MB)
│   ├── brand1/ 280 ภาพ
│   ├── street1/ 1,042 ภาพ
│   ├── person1/ ว่าง, person2/ ว่าง
├── mixed/  ← NEW: Roboflow export 5 ต.ค. 2026 14:36 (DTPG v1)
│   ├── data.yaml (nc=6)
│   ├── train/images 341 + train/labels 341
│   ├── valid/images 93 + valid/labels 93
│   ├── test/images 46 + test/labels 46
│   ├── README.dataset.txt / README.roboflow.txt
├── streed_pt_images/ 231 ภาพ (ชุดเก่า classification)
├── brand_pt_images/ ว่าง
├── ReportImages/ w1.png, w2.png + 1 ไฟล์
├── ชื่อ Class บน Roboflow.docx (แผน 8 คลาส)
├── .env/ (venv Python 3.13.15, ultralytics 8.4.172, torch 2.14.1 พร้อมเทรน)
├── README.md / report.md
```

### 2.1 ชุด `mixed/` (Roboflow DTPG v1) — พร้อมเทรน

- **ที่มา:** https://universe.roboflow.com/akp-hyk4u/dtpg/dataset/1 (CC BY 4.0)
- **ภาพรวม:** 480 ภาพ, annotate YOLOv8 ครบ
- **Split:** train 341 (71%) / valid 93 (19.4%) / test 46 (9.6%)
- **Pre-processing:** Auto-orient + Resize 512x512 (Stretch) — **ไม่มี Augmentation**
- **`data.yaml`:** `nc=6`, names = `['brand_golden_defect', 'brand_golden_good', 'street_burnt_defect', 'street_burnt_good', 'street_golden_defect', 'street_golden_good']`
- **หมายเหตุ path:** ค่า default `train: ../train/images` เป็น relative จากตำแหน่ง yaml — ใช้เทรน local ได้เลยถ้าชี้ `--data mixed/data.yaml`

#### ตาราง classes (นับ instances จากไฟล์ `.txt`, 5 ต.ค. 2026)

| id | class name | train | valid | test | รวม | หมายเหตุ |
|----|------------|-------|-------|------|-----|----------|
| 0 | brand_golden_defect | 47 | 18 | 10 | 75 |  |
| 1 | brand_golden_good | 127 | 26 | 17 | 170 | เยอะสุด |
| 2 | street_burnt_defect | 87 | 32 | 15 | 134 |  |
| 3 | street_burnt_good | 33 | 9 | 0 | 42 | **น้อยสุด + ไม่มีใน test** |
| 4 | street_golden_defect | 76 | 27 | 13 | 116 |  |
| 5 | street_golden_good | 98 | 27 | 13 | 138 |  |
| - | ภาพ background (label ว่าง) | 5 | 0 | 2 | 7 | ใช้เป็น negative ได้ |
| - | **รวม instances** | 468 | 139 | 68 | **675** | เฉลี่ย ~1.4 กล่อง/ภาพ |

#### เทียบกับแผน 8 คลาสใน docx

ไฟล์ `ชื่อ Class บน Roboflow.docx` วางไว้ 8 คลาส (brand/street × golden/burnt × good/defect) แต่ export จริงมี **6 คลาส — ขาด `brand_burnt_good`, `brand_burnt_defect`** สาเหตุที่สมเหตุสมผล: ปาท่องโก๋แบรนด์คุมไฟดี ไม่พบเคสไหม้จริง จึงไม่มีตัวอย่างให้ label — **ไม่ต้องฝืนสร้างคลาสเปล่า** ให้คง 6 คลาสนี้ตอนเทรน แล้วระบุในรายงานว่า brand_burnt ไม่มีข้อมูล

### 2.2 สภาพแวดล้อมพร้อมเทรนแล้ว

- `.env` Python 3.13.15 + `ultralytics 8.4.172`, `torch 2.14.1`, `opencv 5.0.0.93` — รัน `yolo detect train` ได้เลย
- ยังไม่มี `runs/`, `*.pt`, `*.onnx` ใน repo (ต้องเทรน local รอบนึงถึงจะมี)

### 2.3 ของเดิม (รอบ 1-3) ยังเก็บไว้

- ภาพดิบ 1,322 ภาพ + วิดีโอ 2 ไฟล์ + สคริปต์ `_frames.py` + ภาพ w1/w2 เหมือนเดิม
- `streed_pt_images` 231 ภาพ (ชื่อสะกดผิด `streed` ควรเป็น `street` ถ้าจะ rename ให้ทำตอนจัด dataset ใหม่)

---

## 3. สิ่งที่ยังขาดสำหรับ YOLO Detection (อัปเดต 6 ต.ค. 2026 — Train/Eval เสร็จแล้ว)

1. **Weights/metrics:** ✅ มีแล้ว — `runs/detect/pathongko_mixed3_fixed/weights/best.pt` (val 0.895 / test 0.879) + `05-evaluate.py` รายงาน P/R/F1/mAP รายคลาส
2. **Local train:** ✅ รันแล้ว 5 รอบ (baseline + รอบ 2–5) — โมเดลใช้งานจริงคือรอบ 3
3. **สคริปต์เทรนมาตรฐาน:** ✅ มี `02-train/02-train.py` + `02-train/oversample.py` (รอบ 5)
4. **Git:** ✅ มี commit แล้ว (branch `Home2`) — เหลือ merge `Home2` → `main` ถ้าจะปิดงาน
5. **วิดีโอใหญ่:** ✅ `raw_images/vdo/*.mp4` ใส่ gitignore แล้ว (local only)

---

## 4. ขั้นตอนถัดไปที่แนะนำ (Next Steps)

### Step 6 — เทรน YOLO local (พร้อมทำได้ทันที)
```powershell
.\.env\Scripts\python.exe -m ultralytics detect train model=yolov8n.pt data=mixed/data.yaml epochs=100 imgsz=512 batch=16 patience=20 workers=4
```
- ถ้า GPU ไม่พอ ลดเป็น `yolov8n.pt epochs=50 imgsz=512 batch=8`
- ถ้าอยากเทียบรุ่น: ลอง `yolov8s.pt` อีกรอบหลัง n ผ่าน
- เป้าหมายเบื้องต้น: `mAP50 > 0.85`, `Precision/Recall > 0.8` (ระวังคลาส 3 `street_burnt_good` มีน้อย ค่า AP อาจต่ำกว่าคลาสอื่น)

### Step 7 — ประเมิน + รายงานผล
- [ ] ดู `runs/detect/train/results.csv`, `confusion_matrix.png`, `F1_curve.png`, `PR_curve.png`
- [ ] รัน `yolo detect val model=runs/detect/train/weights/best.pt data=mixed/data.yaml` + `yolo detect predict` กับภาพใน `mixed/test/images` และวิดีโอ `1004.mp4`
- [ ] ถ้ามี Roboflow Train metrics ให้เอามาเทียบตารางเดียวกัน
- [ ] อัปเดตไฟล์นี้ด้วย metrics + ตัวอย่างภาพ predict

### Step 8 — เก็บงานลง git
- [ ] แก้ `.gitignore` (เลิกใช้ `*`): ignore `.env/`, `*.mp4`, `runs/`, `__pycache__/`
- [ ] `git add report.md README.md mixed/data.yaml "ชื่อ Class บน Roboflow.docx" 01-extract/_frames.py` + dataset sample (อย่า push ภาพ 480 ทั้งหมดถ้า repo หนัก — พิจารณา DVC หรือเก็บแค่ yaml + link Roboflow)
- [ ] commit แรก

---

## 5. ความเสี่ยง / ข้อควรระวัง (อัปเดต)

1. **คลาสไม่สมดุล:** `brand_golden_good 170` vs `street_burnt_good 42` (~4:1) และคลาส 3 ไม่มีใน test — วัดผล test จะไม่มีตัวแทนคลาส 3 เลย ควรระบุในรายงาน / รอบหน้า re-split แบบ stratified หรือย้ายบางภาพจาก train → test
2. **Resize แบบ Stretch 512:** ภาพถูกยืดสัดส่วน อาจทำให้ทรงปาท่องโก๋เพี้ยนเล็กน้อย — รับได้สำหรับรอบแรก ถ้า mAP ต่ำค่อย export ใหม่แบบ Letterbox/Fit
3. **ไม่มี Augmentation:** export มาตัวเปล่า — ชดเชยด้วย augmentation ตอนเทรน (`hsv_h, flipud, mosaic` default ของ ultralytics เปิดอยู่แล้ว)
4. **ภาพ background 7 ใบ:** มีประโยชน์เป็น negative แต่อย่าลบ — เก็บไว้ลด false positive
5. **วิดีโอ + dataset ใหญ่:** อย่า commit `.mp4` / ภาพ 960 ไฟล์ทั้งหมดถ้าไม่จำเป็น — repo จะบวม
6. **`brand_burnt_*` ไม่มีข้อมูลเลย (แผน 8 → ทำจริง 6):** `brand_burnt_good` / `brand_burnt_defect` มี 0 ภาพ 0 กล่องทั้ง `mixed2` (train/valid/test) เพราะแบรนด์คุมไฟดี ไม่พบเคสไหม้จริง — โมเดล `best.pt (nc=6)` จึงจำแนก 2 คลาสนี้ไม่ได้เลย (`03/04/06` เปิดกล้องจ่อแบรนด์ไหม้ก็จะทายเป็น `street_burnt_*`/`brand_golden_*` หรือหลุด) ไม่นับเป็นเป้าไม่ถึง แต่เป็นข้อจำกัดที่ต้องระบุในรายงาน/ตอนพรีเซนต์

---

## 6. คำตอบสั้นๆ: ตอนนี้ถึงขั้นไหน?

> **จบการเทรน 5 รอบ — โมเดลดีสุดคือรอบ 3 (`pathongko_mixed3_fixed`, yolov8n):
> val mAP50 0.895 / test 0.879 ผ่านเป้า 0.85 — พร้อมใช้ คิดเป็น ~95%
> (รอบ 4 yolov8s / รอบ 5 oversample ไม่ชนะรอบ 3 — ดู 7.7–7.8)**

---

## 7. รอบ 5 ต.ค. 2026 (เย็น): เทรน baseline + pseudo-label + เทรนรอบ 2 ✅

> **อัปเดตความถูกต้อง (หัวข้อ 7.4):** ตัวเลข test รอบแรก (mAP50 0.935) วัดบน label ผสมที่ ultralytics
> ตัด polygon ทิ้งเงียบๆ — ค่าจริงบน label ที่แก้แล้วคือ test **mAP50 0.867** / F1 0.858

### 7.1 Baseline (`runs/detect/pathongko_mixed_model`, yolov8n 50e, สคริปต์ `02-train/02-train.py` ฉบับแรก)
- Val (92 ภาพ): P 0.819 / R 0.822 / **mAP50 0.844** / mAP50-95 0.797
- อ่อนสุด `street_burnt_defect` 0.591 → ที่มาของแผนเพิ่มข้อมูล

### 7.2 Pseudo-label ภาพดิบ (`07-pseudolabel/pseudo_label.py`, best.pt conf 0.5)
- ได้ label ร่าง **524 ภาพ** → review บน Roboflow → export **DTPG v2 (`mixed2/`)**
- ตัดภาพซ้ำ mixed 480 + ภาพเปล่า 318 ใบทิ้ง (local only, gitignore)

### 7.3 รอบ 2 (`02-train/02-train.py` ฉบับปัจจุบัน, yolov8n **80e**, `cls=1.0` + `copy_paste=0.3` + `mixup=0.2`)
- `mixed2/`: train **713** (+372) / valid 93 / test 46 (valid/test ชุดเดิม เทียบ baseline ได้ตรง)
- Val: P 0.875 / R 0.832 / **mAP50 0.895** / mAP50-95 0.831 ✅ ผ่านเป้า 0.85
- Test (`split=test`): P 0.93 / R 0.897 / **mAP50 0.935** ⚠️ เลขนี้วัดบน label ผสม
  (polygon โดน ultralytics ทิ้งเงียบๆ เหลือ 60 กล่อง — ดู 7.4, ค่าจริง 0.867)
- `street_burnt_defect`: 0.591 → **0.767** (+0.176) / brand ทั้งคู่ 0.995
- จุดอ่อนคงเหลือ: `street_burnt_good` recall 0.667 (กล่อง train แค่ 47)
- Weights รอบ 2: `runs/detect/pathongko_mixed2_80e/weights/best.pt` (superseded — ปัจจุบันใช้รอบ 3)
- ภาพ demo predict 46 ใบ: `runs/detect/runs/detect/predict_test/` (local only)

### 7.4 แก้บั๊ก label ผสม bbox + polygon (สำคัญ)
- `mixed2` มี label polygon ปน (train 262 / valid 69 / test 32 ไฟล์) — ultralytics เจอไฟล์ผสมแล้ว
  **ทิ้ง polygon เงียบๆ ใช้แค่ bbox** ทำให้เทรนรอบ 2 ใช้ข้อมูลไม่ครบ + test mAP50 0.935 วัดแค่ 60 กล่อง
- แก้: แปลง polygon→bbox (min/max) 363 ไฟล์ เป็น bbox 5 คอลัมน์ล้วน
- วัดใหม่บน label ที่แก้แล้ว: test **mAP50 0.867** (P 0.93 / R ~0.87 / F1 0.858, ครบ 68 กล่อง) = ค่าจริง,
  valid **mAP50 0.895** เท่าเดิม
- บทเรียน: รอบ 3 ควรเทรนใหม่บน label ที่แก้แล้วเพื่อใช้ข้อมูลครบ 100%

### 7.5 สคริปต์ตามโครงอาจารย์ (branch Home2)
- `03-predict.py` (ภาพนิ่ง) / `04-predict_video.py` (วิดีโอ, stream ทีละเฟรมกันแรมเต็ม)
- `05-evaluate.py` (+`eval_utils.py`): ตาราง P/R/F1/mAP รายคลาส, confusion matrix custom heatmap,
  กราฟ matplotlib ทีละภาพกลางจอ — ย่อเหลือ ~50 บรรทัดแล้ว
- `06-webcam_realtime.py` (realtime webcam, q=ออก) / `07-pseudolabel/` (เลื่อนจาก 03→07)
- merge Home2→main แล้ว 2 รอบ, สถานะปัจจุบันดู `git log`

### 7.6 เทรนรอบ 3 บน label ที่แก้แล้ว (ใช้ข้อมูลครบ 100%) ✅
- สคริปต์เดิม `02-train/02-train.py` เปลี่ยนชื่อผลเป็น `pathongko_mixed3_fixed` (แยกจากรอบ 2)
- Val: P 0.806 / R 0.896 / **mAP50 0.895** (เท่าเดิม) — `street_burnt_good` recall 0.667→**0.889**,
  `street_golden_good` 0.796→0.836
- Test: P 0.942 / R 0.857 / **mAP50 0.879** (รอบ 2: 0.867, +0.012)
- สคริปต์ 03/04/05/06 ย้ายมาใช้ `runs/detect/pathongko_mixed3_fixed/weights/best.pt` แล้ว

### 7.7 เทรนรอบ 4: ขยับเป็น yolov8s (ไม่ชนะรอบ 3)
- เหตุผล: ดัน `street_burnt_defect` (test R 0.6) ด้วยโมเดลใหญ่ขึ้น (11M params)
- Val mAP50 **0.89** / Test **0.872** (รอบ 3: 0.895/0.879) — แพ้รอบ 3 เล็กน้อย สรุปไม่ใช้

### 7.8 เทรนรอบ 5: oversample คลาสน้อย (`02-train/oversample.py` → `mixed2_os/`, local only)
- train 713→**1,164** ภาพ กล่อง 0:204 / 1:174 / 2:394 / 3:235 / 4:309 / 5:315 (สัดส่วน 6.3:1→2.3:1)
- yolov8s 100e — Val mAP50 **0.88** / Test **0.878** (เสมอรอบ 3, `street_burnt_defect` test 0.624/R 0.6)
- สรุป: oversampling ไม่ขยับ test — **โมเดลใช้งานจริงยังคงรอบ 3**
- ทางที่เหลือถ้าจะดัน `street_burnt_defect` ต่อ: เก็บตัวอย่างไหม้เพิ่มจากหน้างานจริงเท่านั้น
