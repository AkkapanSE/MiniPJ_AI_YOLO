# รายงานแลปที่ 3: Patongko Detection ด้วย YOLOv8 (6 คลาส)

**โปรเจกต์:** MiniPJ_AI_YOLO — จำแนกปาท่องโก๋แบรนด์ vs ร้านทั่วไป + สุก/ไหม้ + ทรงดี/ชำรุด
**เครื่องมือ:** Python 3.13 + PyTorch cu126 + Ultralytics 8.4 + Roboflow (GPU RTX 4060 Laptop)
**ชุดหลัก:** `mixed2/` 852 ภาพ (train 713 / valid 93 / test 46) | **เทรนทั้งหมด 5 รอบ — รอบ 3 ดีสุด val 0.895 / test 0.879**
**รันทุกสคริปต์:** `.\.env\Scripts\python.exe <สคริปต์>`

---

## 1. ทฤษฎีที่เกี่ยวข้อง

### 1.1 YOLOv8 Detection (One-Stage Detector)
YOLO มองทั้งภาพครั้งเดียวแล้วทายกรอบพร้อมคลาสเลย ไม่ต้อง propose region ก่อนเหมือน Faster R-CNN
เอาต์พุตต่อ 1 กล่อง = `class + x,y,w,h (normalized) + conf`
- `conf >= 0.8` ถึงตีกรอบ (ใช้ใน `03/04/06`, ล็อกตามอาจารย์)
- กล่องซ้ำซ้อนถูกกรองด้วย NMS (Non-Maximum Suppression) โดยใช้ IoU เป็นเกณฑ์
- โมเดลที่ใช้: `yolov8n` (nano ~3M params, เร็ว/พอดีข้อมูลเล็ก) เทียบกับ `yolov8s` (~11M params รอบ 4–5)

### 1.2 การแบ่งข้อมูล train / valid / test
- `train 713` มีไว้เรียน, `valid 93` มีไว้เลือกโมเดล/early stopping, `test 46` มีไว้สอบครั้งเดียวห้ามเห็นตอนเทรน
- ห้าม leakage: `07-pseudolabel` กันชื่อไฟล์ซ้ำ `mixed/` แล้ว (`xxx_jpg.rf.<hash>.jpg → xxx.jpg`) ภาพซ้ำ/valid+test ถูก skip
- `mixed2_os/` (รอบ 5) oversample เฉพาะ train ส่วน valid/test ชี้ชุดเดิมกัน leakage

### 1.3 IoU, P/R/F1, mAP (ทำไมไม่มี Accuracy)
- `IoU = พื้นที่ทับซ้อน / พื้นที่รวม` เกณฑ์ `IoU>=0.5` ถือว่าทายถูก (ใช้ใน `eval_utils.py:70`)
- `Precision = TP/(TP+FP)` ทายแม่นแค่ไหน, `Recall = TP/(TP+FN)` จับได้ครบแค่ไหน
- `F1 = 2PR/(P+R)` ค่ากลาง, งาน Detection **ไม่มี True Negative** (นับจุดว่างที่ทายถูกไม่ได้) เลยใช้ F1/mAP แทน Accuracy
- `mAP50` = เฉลี่ย AP ทุกคลาสที่ IoU 0.5, `mAP50-95` = เฉลี่ย IoU 0.5–0.95 (เข้มกว่า) เป้าโปรเจกต์ `mAP50 > 0.85`

### 1.4 Augmentation + การชดเชยคลาสน้อย
- Ultralytics เปิด `mosaic/hsv/flip` ให้อยู่แล้ว รอบ 2–5 เพิ่ม `cls=1.0 + copy_paste=0.3 + mixup=0.2` ให้โมเดลเห็นคลาสน้อยบ่อยขึ้น
- `patience=20` early stopping ถ้า valid ไม่ดีขึ้น 20 epochs
- รอบ 5 oversample (`02-train/oversample.py`): ภาพมีคลาส 3→x5, คลาส 0→x4, คลาส 2→x2 (train 713→1,164) แต่ก็อปภาพเดิมซ้ำไม่เพิ่ม information ใหม่ → test ไม่ขยับ (0.878)

### 1.5 ฟอร์แมต Label (บทเรียน polygon)
- YOLO train ต้องการ bbox 5 ค่า `class x y w h` ต่อบรรทัด
- `mixed2` มี polygon ปน (train 262 / valid 69 / test 32 ไฟล์) ultralytics เจอไฟล์ผสมแล้ว**ทิ้ง polygon เงียบๆ** → รอบ 2 เทรน/วัดไม่ครบ (test 0.935 วัดแค่ 60 กล่อง) แก้โดยแปลง polygon→bbox (min/max) 363 ไฟล์ ค่าจริงคือ test 0.867–0.879 (68 กล่องครบ)

### 1.6 Confusion Matrix + background
แถว=true / หลัก=pred ช่อง `background` คือทายเกิน (FP) กับหลุด (FN) (`eval_utils.build_confusion_matrix` จับคู่ greedy IoU)

---

## 2. Python Code (โค้ดหลักที่ใช้จริง)

โครงสคริปต์ตามลำดับ `01 → 02 → 07 → 03/04/05/06` (ดู `README.md`)

**`01-extract/_frames.py` — สกัดเฟรม:**
```python
extract_frames(video_path="raw_images/vdo/1004(1).mp4",
    output_folder="raw_images/brand1", frame_interval=15, prefix="brand")
# while cap.read(): if frame_count % frame_interval == 0: cv2.imwrite(...)
```

**`02-train/02-train.py` — เทรน (ฉบับรอบ 5, รอบ 3 เปลี่ยนแค่ data/epochs/name):**
```python
model = YOLO('yolov8s.pt')
results = model.train(data='mixed2_os/data_local.yaml', epochs=100,
    imgsz=640, batch=16, device=0, workers=4, patience=20,
    cls=1.0, copy_paste=0.3, mixup=0.2, name='pathongko_mixed5_os')
```

**`02-train/oversample.py` — ก็อปคลาสน้อย:**
```python
MULT = {3: 5, 0: 4, 2: 2}  # คลาส 3→x5, 0→x4, 2→x2
copies = max([MULT.get(c, 1) for c in cls] or [1])
```

**`07-pseudolabel/pseudo_label.py` — ทำ label ร่าง:**
```python
results = model.predict(source=[str(p) for p in imgs], conf=0.5, imgsz=640)
# skip ชื่อซ้ำ train/valid+test กัน leakage → เซฟ "cls x y w h" + copy ภาพเข้า pseudo/
```

**`03-predict.py` — ทดสอบภาพนิ่ง:**
```python
for r in model.predict(source=source, conf=0.8, imgsz=640,
    save=True, save_txt=True, stream=True,
    project="runs/detect", name="predict_demo"):
    n_img += 1; n_box += len(r.boxes)
```

**`04-predict_video.py` — ทดสอบวิดีโอ (stream กันแรมเต็ม):**
```python
for r in model.predict(source=source, conf=0.8, imgsz=640,
    save=True, stream=True, project="runs/detect", name="predict_video_demo"):
    n_frames += 1; n_box += len(r.boxes)
```

**`05-evaluate.py` + `eval_utils.py` — ประเมิน:**
```python
m = model.val(data="mixed2/data_local.yaml", split="test", imgsz=640)
present = U.present_classes(BASE/"mixed2"/"test"/"labels")  # คลาสไหนไม่มีใน test → N/A
cm = U.build_confusion_matrix(model, m.names, img_dir, lbl_dir)  # greedy IoU>=0.5
```

**`06-webcam_realtime.py` — webcam:**
```python
annotated = model.predict(frame, conf=0.8, imgsz=640, verbose=False)[0].plot()
cv2.imshow("Patongko Realtime (q=ออก)", annotated)  # กด q ออก
```

---

## 3. คำอธิบาย Code

| ไฟล์ | ทำอะไร | พารามิเตอร์สำคัญ |
|---|---|---|
| `01-extract/_frames.py` | อ่านวิดีโอด้วย OpenCV เซฟทุก `frame_interval` เฟรม | `frame_interval=15` รอบแรก (รอบสอง =10) ได้ดิบ 1,322 ใบ |
| `02-train/02-train.py` | สั่ง `YOLO.train()` บันทึก `runs/detect/<name>/weights/best.pt` | `epochs 50/80/100, imgsz 640, batch 16, patience 20, cls 1.0, copy_paste 0.3, mixup 0.2` |
| `02-train/oversample.py` | ก็อปภาพ train ที่มีคลาสน้อย 713→1,164 สร้าง `mixed2_os/` | `MULT {3:5, 0:4, 2:2}` valid/test ชี้เดิม |
| `07-pseudolabel/pseudo_label.py` | เอา best baseline predict ดิบ `conf 0.5` ได้ร่าง 524 ภาพ → review → `mixed2/` | กัน leakage ด้วย basename, ภาพเปล่า 318 ลง `pseudo_empty.txt` |
| `03-predict.py` | predict ภาพนิ่ง default `mixed2/test/images` เซฟภาพ+txt | `conf 0.8, stream=True` ประหยัดแรม |
| `04-predict_video.py` | predict วิดีโอ default `raw_images/vdo/1004.mp4` | `stream=True` ทีละเฟรม ไม่โหลดทั้งคลิป |
| `05-evaluate.py` / `eval_utils.py` | `model.val(split=test)` ตาราง P/R/F1/mAP + CM custom + โชว์กราฟทีละภาพ | `IOU 0.5, CONF 0.8` คลาสไม่อยู่ใน test ขึ้น `N/A` |
| `06-webcam_realtime.py` | เปิดกล้อง `cv2.VideoCapture` ทาย realtime | `cam 0 default, q=ออก` |

---

## 4. การทำงาน (Workflow)

1. **ถ่ายวิดีโอหน้าร้านจริง 2 คลิป** `raw_images/vdo/` (brand + street) → **สกัดเฟรม** `01-extract` ได้ดิบ 1,322 ใบ
2. **Label บน Roboflow** 480 ภาพ → export `mixed/` (v1, 6 คลาส) → **เทรน baseline รอบ 1** `yolov8n 50e` val 0.844
3. **Pseudo-label** เอา best รอบ 1 ทายภาพดิบ `conf 0.5` ได้ร่าง 524 ภาพ → คนตรวจบน Roboflow → export `mixed2/` 852 ภาพ (train 713 / valid 93 / test 46)
4. **แก้บั๊ก polygon** แปลง 363 ไฟล์เป็น bbox ล้วน (ไม่งั้น ultralytics ทิ้งเงียบ)
5. **เทรนรอบ 2–5:** รอบ 2 `n 80e` → รอบ 3 `n 80e fixed` (ข้อมูลครบ 100%, ดีสุด) → รอบ 4 `s` (overfit) → รอบ 5 oversample `s 100e` (เสมอตัว)
6. **ประเมิน** `05-evaluate` บน test ที่ไม่เคยเห็น → ตารางรายคลาส + CM + กราฟ
7. **ทดสอบใช้งาน** `03` ภาพนิ่ง / `04` วิดีโอ / `06` webcam (stream ทีละเฟรมกันแรมเต็ม)

---

## 5. ผลการทดลอง (แนบรูป)

### 5.1 ชุดข้อมูล `mixed2` (6 คลาส — แผนเดิม 8 แต่ `brand_burnt_*` 0 ข้อมูล)

| id | class | train | valid | test | รวม |
|----|---|---|---|---|---|
| 0 | brand_golden_defect | 51 | 18 | 10 | 79 |
| 1 | brand_golden_good | 147 | 26 | 17 | 190 |
| 2 | street_burnt_defect | 191 | 23 | 15 | 229 |
| 3 | street_burnt_good | 47 | 9 | **0** | 56 |
| 4 | street_golden_defect | 298 | 25 | 13 | 336 |
| 5 | street_golden_good | 267 | 26 | 13 | 306 |
| - | background (label ว่าง) | - | - | 2 | - |

> `brand_burnt_good/defect` ไม่มี 0 กล่องทั้งชุด (แบรนด์คุมไฟดี) โมเดล `nc=6` จำแนกไม่ได้ + คลาส 3 ไม่มีใน test (`05-evaluate` ขึ้น `N/A`)

![สกัดเฟรมรอบแรก](<raw_images/RP_images/w1.png>)
![สกัดเฟรมรอบสอง](<raw_images/RP_images/w2.png>)
![ตีกรอบบน Roboflow](<raw_images/RP_images/label.png>)
![สถิติชุดข้อมูล labels.jpg](<raw_images/RP_images/labe2.png>)
![กราฟ label train รอบ 3](<raw_images/RP_images/mixed3fixed_labels.jpg>)

### 5.2 เทรนทั้งหมด 5 รอบ (รอบ 3 ดีสุด)

| รอบ | โมเดล | ข้อมูล | val mAP50 | test mAP50 |
|---|---|---|---|---|
| 1 baseline | yolov8n 50e | mixed 480 | 0.844 | - |
| 2 | yolov8n 80e | mixed2 (label ผสม) | 0.895 | 0.867 (ค่าจริง, 0.935 คือเลขลวง 60 กล่อง) |
| **3 ใช้จริง** | **yolov8n 80e** | **mixed2 label แก้แล้ว 100%** | **0.895** | **0.879** |
| 4 | yolov8s | mixed2 แก้แล้ว | 0.89 | 0.872 (overfit) |
| 5 | yolov8s 100e | mixed2_os oversample 1,164 | 0.88 | 0.878 (เสมอตัว) |

![กราฟเทรน results.png](<raw_images/RP_images/mixed3fixed_results.png>)
![เทียบ label กับ pred ชุด valid](<raw_images/RP_images/mixed3fixed_val_batch0_pred.jpg>)
![train batch ตัวอย่าง](<raw_images/RP_images/mixed3fixed_train_batch0.jpg>)

### 5.3 ประเมิน test 46 ภาพ (รอบ 3: P 0.942 / R 0.857 / F1 ~0.90 / mAP50 0.879 ผ่านเป้า 0.85)

รายคลาส: brand ทั้งคู่ ~0.9+, `street_golden_defect` 0.995, `street_golden_good` 0.909, จุดอ่อน `street_burnt_defect` 0.605 (R 0.6)

![ผลรัน 05-evaluate](<raw_images/RP_images/eval_mean.png>)
![confusion matrix](<raw_images/RP_images/mixed3fixed_confusion_matrix.png>)
![confusion matrix normalized](<raw_images/RP_images/mixed3fixed_confusion_matrix_normalized.png>)
![PR curve](<raw_images/RP_images/mixed3fixed_BoxPR_curve.png>)
![F1 curve](<raw_images/RP_images/mixed3fixed_BoxF1_curve.png>)

### 5.4 ทดสอบใช้งาน `03/04/06` (46 ภาพ ได้ 59 กล่อง)

- `03-predict.py` บน test: 44/46 ภาพมีผล อีก 2 ภาพ close-up (`street_0424/0428`) ไม่มี detection
- ตัวอย่างทายถูก: brand 3 กล่อง (`defect 0.71 + good 0.91/0.93`), street 3 กล่อง (`good 0.97 + defect 0.94/0.96`)
- เคสหลุด: `street_0379` มี ~8 ชิ้นแต่จับได้แค่ `burnt_defect 0.86/0.84` 2 กล่องซ้าย — ตรงจุดอ่อน `burnt_defect` R 0.6
- `04` วิดีโอ + `06` webcam ใช้ `stream=True` ทีละเฟรม กันแรมเต็ม

![predict brand 3 กล่อง](<raw_images/RP_images/predict_brand_0213.jpg>)
![predict street 3 กล่อง](<raw_images/RP_images/predict_street_0100.jpg>)
![predict burnt หลุดบางกล่อง](<raw_images/RP_images/predict_street_0379.jpg>)
![เคสไม่มี detection](<raw_images/RP_images/predict_street_0424.jpg>)
![รัน predict วิดีโอ](<raw_images/RP_images/detect_vdo.png>)

### 5.5 ข้อจำกัด
1. `brand_burnt_*` จำแนกไม่ได้เลย (0 ข้อมูล) — จ่อกล้องก็ได้แค่ 1 ใน 6 ที่มี
2. `street_burnt_good` ไม่มีใน test — วัด test ไม่ครอบคลุมคลาสนี้
3. `street_burnt_defect` test R 0.6 — ต้องเก็บตัวอย่างไหม้จริงเพิ่มเท่านั้น (oversample ไม่ช่วย)
4. test มีแค่ 46 ภาพ — เล็กไป ถ้าจะดันต่อควร re-split แบบ stratified
