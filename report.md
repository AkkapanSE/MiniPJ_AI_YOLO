# รายงานความคืบหน้าโปรเจกต์ Patongko Detection ด้วย YOLO

**ผู้จัดทำ:** 1. นาย เอกพันธ์ ทศทิศรังสรรค์ 67543210050-0 SE-sec1
2. นาย ณฐภาพ สายหล้า 675432154-2 SE-sec1
**วันที่อัปเดต:** 8 ตุลาคม 2026 (เทรนครบ 9 รอบ — โมเดลใช้จริงรอบ 9)
**ที่ตั้งโปรเจกต์:** `C:\Users\chano\MiniPJ_AI_YOLO`
**สถานะโดยรวม:** 🟢 **~99% — เทรน + ประเมิน + เทสของจริง (ภาพครู/วิดีโอ/กล้อง) เสร็จ รอบ 9 test mAP50 0.930 ผ่านเป้า 0.85**

> ผลเทสภาพรวมนอกชุดเทรนทั้งหมดดูใน **[README2.md](README2.md)** (ภาพครู C–G / วิดีโอ / webcam + ตาราง conf)

---

## 1. สรุปสั้นๆ ว่าทำถึงขั้นไหนแล้ว

| Phase | สถานะ | หมายเหตุ (อัปเดต 8 ต.ค. 2026) |
|-------|--------|-------------------------------|
| 1. เก็บวิดีโอดิบ (Data Collection) | ✅ เสร็จ | 5 คลิปใน `raw_images/vdo/` (เก่า 1004 ×3 + ใหม่ 813113849 ×2, เก็บเพิ่ม 8 ต.ค.) |
| 2. สกัดเฟรมจากวิดีโอ (Frame Extraction) | ✅ เสร็จ | คลิปใหม่สกัดทุก 15 เฟรมได้ 1,082 ภาพ (`new_017632` 482 + `new_101088` 600) |
| 3. ทำความสะอาดข้อมูล / คัดภาพ (Data Cleaning) | ✅ เสร็จ | กรองว่าง/เบลอ (pseudo ยิงแล้วว่าง 423 ใบ) เหลืออัปโหลด 1,082 → review บน Roboflow เหลือใช้จริง 1,011 ภาพ |
| 4. ตีกรอบ Annotation สำหรับ YOLO (Labeling) | ✅ เสร็จ | Label บน Roboflow โปรเจกต์ `dtpg` — v5 (1,863 ภาพ) → แก้รูพรุน `good→defect` + เกลี่ย street → v6 |
| 5. Export YOLO Dataset (`mixed6/`) | ✅ เสร็จ | **train 1,714 / valid 93 / test 56** รวม 1,863 ภาพ + `data.yaml` 6 คลาส (512px, no aug) |
| 6. Train / Val YOLO Detection | ✅ เสร็จ (รอบ 9 ใช้จริง) | เทรน 9 รอบ — รอบ 8 test สูงสุด 0.935, รอบ 9 test 0.930 + แก้รูพรุน (`all_model/best_round9_v6.pt`) |
| 7. ประเมินผล + ทดสอบ Detection | ✅ เสร็จ | test 0.930 (P 0.967 / R 0.929 / F1 0.947) + confusion matrix; `03/04/06` ทดสอบภาพนิ่ง/วิดีโอ/webcam แล้ว |
| 8. เทสของจริงนอกชุดเทรน | ✅ เสร็จ (ดู README2.md) | ภาพครู C–G สูงสุด 5/6 คลาส, วิดีโอเจอ 70% เฟรม, กล้องถูก 6/7 ใบ |

---

## 2. สิ่งที่มีอยู่จริงในโฟลเดอร์ (สำรวจ 8 ต.ค. 2026)

```
MiniPJ_AI_YOLO/
├── 01-extract/_frames.py            # สกัดเฟรม OpenCV (frame_interval=15)
├── 02-train/
│   ├── 02-train.py                  # รอบเก่า (mixed3)
│   ├── 02-train-mixed5.py           # รอบ 8: fine-tune จาก best_round3 บน mixed5
│   ├── 02-train-mixed6.py           # รอบ 9: fine-tune จาก best_round8 บน mixed6 ✅ใช้จริง
│   └── oversample.py                # ของเก่ารอบ 5
├── 03-predict.py                    # ภาพนิ่ง (default รอบ 8, conf 0.5, mixed5)
├── 04-predict_video.py              # วิดีโอ stream (รอบ 9)
├── 05-evaluate.py + eval_utils.py   # ประเมิน test + confusion matrix + กราฟ
├── 06-webcam_realtime.py            # webcam realtime (รอบ 9, conf 0.5)
├── 07-pseudolabel/pseudo_label.py   # ทำร่าง label conf 0.5 (กัน leakage)
├── mixed2/ (tracked)                # DTPG v2: 852 ภาพ (ของเก่ารอบ 1-5)
├── mixed3/ mixed4/ mixed5/ mixed6/  # DTPG v3-v6 (local only, gitignore — ใช้ mixed6)
├── all_model/                       # best_round3/6/7/8/9 + yolov8n/s (local only, *.pt ignore)
├── runs/detect/                     # pathongko_v6 (รอบ 9) + teacherC/D/E/F/G + predict_* (ignore)
├── raw_images/
│   ├── vdo/ (ใหม่ 2 คลิป ~123MB + old/)
│   ├── new_017632/ + new_101088/    # เฟรมสกัด 1,082 ใบ
│   ├── upload_ready/                # ชุดอัปโหลด Roboflow 1,082 ภาพ + pseudo 659
│   ├── teacher_test(A)-(G)/         # ชุดเทสครู (local only)
│   └── RP_images/                   # ภาพหลักฐาน 35 ใบ (push ขึ้น repo)
├── ReportImages/                    # ภาพประกอบรอบเก่า (tracked)
├── DTPG.v5/v6.yolov8.zip            # export ดิบ (local only, *.zip ignore)
└── README.md / README2.md / report.md / report2.md / report3.md
```

### 2.1 ชุด `mixed6/` (Roboflow DTPG v6) — ชุดใช้จริง

- **ที่มา:** https://universe.roboflow.com/akp-hyk4u/dtpg/dataset/6 (CC BY 4.0)
- **ภาพรวม:** 1,863 ภาพ (train 1,714 / valid 93 / test 56)
- **Pre-processing:** Auto-orient + Resize 512x512 (Stretch) — **ไม่มี Augmentation** (augment ตอนเทรนแทน: `cls=1.0` + `copy_paste/mixup`)
- **`data.yaml`:** `nc=6`, names 6 ชื่อเดิมเรียงถูก (v4 เคยเพี้ยนเป็น `['2','3',...]` — แก้แล้วตั้งแต่ v5)

#### ตาราง classes (นับ instances จากไฟล์ `.txt`, 8 ต.ค. 2026)

| id | class name | train | valid | test | หมายเหตุ |
|----|------------|-------|-------|------|----------|
| 0 | brand_golden_defect | 244 | 18 | 10 | v5 มี 147 → v6 แก้รูพรุน +97 |
| 1 | brand_golden_good | 574 | 26 | 17 | v5 มี 808 → ย้ายรูพรุนออก -234 |
| 2 | street_burnt_defect | 751 | 23 | 26 | จาก mixed2 แค่ 191 — จุดอ่อนเดิมหาย |
| 3 | street_burnt_good | 406 | 9 | 8 | จาก 47 — มีใน test แล้ว |
| 4 | street_golden_defect | 335 | 25 | 13 | - |
| 5 | street_golden_good | 463 | 26 | 13 | - |

#### เทียบกับแผน 8 คลาสใน docx

`docs/ชื่อ Class บน Roboflow.docx` วางไว้ 8 คลาส แต่ export จริงมี **6 คลาส — ขาด `brand_burnt_*`**
เพราะแบรนด์คุมไฟดี ไม่พบเคสไหม้จริง — คง 6 คลาสตอนเทรนตามเดิม

---

## 3. ประวัติเทรน 9 รอบ (ย่อ)

| รอบ | ข้อมูล | โมเดลตั้งต้น | val mAP50 | test mAP50 | หมายเหตุ |
|---|---|---|---|---|---|
| 1 | mixed/ 480 | yolov8n 50e | 0.844 | - | baseline |
| 2 | mixed2 852 | yolov8n 80e | 0.895 | 0.867 | +pseudo 524, แก้ polygon |
| 3 | mixed2 (label แก้ 100%) | yolov8n 80e | 0.895 | **0.879** | ดีสุดยุคแรก |
| 4 | mixed2 | yolov8s | 0.89 | 0.872 | overfit ไม่ใช้ |
| 5 | mixed2_os 1,164 | yolov8s 100e | 0.88 | 0.878 | oversample ไม่ช่วย |
| 6-7 | mixed3 | yolov8n | ~0.89 | ~0.917 | ตัวกลาง |
| 8 | mixed5 1,863 | best_round3 80e | 0.889 | **0.935** | สูงสุด (แต่รูพรุนยังเป็น good) |
| 9 | mixed6 1,863 | best_round8 80e | 0.881 | **0.930** | ✅ใช้จริง (แก้รูพรุนแล้ว) |

**รอบ 9 รายคลาส test:** defect 0.895, good 0.982, burnt_def **0.921 (R 1.0)**,
burnt_good **0.875** (เคย N/A), golden_def 0.995, golden_good 0.915

---

## 4. เทสของจริงนอกชุดเทรน (ย่อ — เต็มดู README2.md)

- **ภาพครู C–G:** C 1/6 → D 3/6 → E/F/G **5/6** (ขาดแค่ 2 `street_burnt_defect`), ใบเดี่ยวดีสุด `S__33120296` 4/6 (conf 0.5 พอดี 0.52)
- **วิดีโอ 7,229 เฟรม:** เจอ 70.4% เฟรม 6,964 กล่อง, เฟรม `RP_f04559` ได้ 4 คลาสในเฟรมเดียว
- **กล้อง realtime:** ถูก 6/7 ใบ ครบ 5/6 คลาส — WC2 พิสูจน์รูพรุนพลิกเป็น defect 0.87 แล้ว
- สาเหตุที่เจอ: pseudo bias → ภาพแคปบีบอัด → ภาพเบลอ/ว่าง → นิยามรูพรุนไม่ตรง (แก้ v6 แล้ว)

---

## 5. ความเสี่ยง / ข้อควรระวัง (อัปเดต)

1. **ภาพครูเป็น out-of-domain** (แคปแชทบีบอัด/แสง/พื้นหลังต่างจากคลิปเทรน) — คลาสไหม้แยกยากสุด ต้องถ่ายกล้องจริง
2. **อย่าจ่อของใกล้เกิน** (เต็มเฟรม = หลุด, พิสูจน์แล้ว WC5) — ถ่ายให้เห็นชิ้นเต็ม
3. **`street_0379` label ปน segment+box** — โดนข้ามตอน eval 1 ภาพ กลับไปแก้บน Roboflow ได้แต่ไม่บล็อกผล
4. **ดาต้าเซ็ตหนัก** — mixed3-6 + zip + raw_images local only (gitignore), push แค่โค้ด+รีพอร์ต+`RP_images/`
5. **`brand_burnt_*` จำแนกไม่ได้** (0 ข้อมูล) — ข้อจำกัดถาวรจนกว่าจะเจอของจริง
