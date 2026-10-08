MiniPJ_AI_YOLO — Patongko Detection (Brand vs Street)

จำแนกปาท่องโก๋ แบรนด์ vs ร้านทั่วไป + สุก/ไหม้ + ทรงดี/ชำรุด ด้วย YOLOv8 Detection (6 คลาส)

ถ่ายวิดีโอหน้าร้านจริง 2 คลิป → สกัดเฟรม → Label บน Roboflow → เทรน 5 รอบ → ได้โมเดลรอบ 3 ดีสุด

สถานะ (6 ต.ค. 2026): เทรนทั้งหมด 5 รอบ — รอบ 3 ดีสุด (Label แก้แล้วครบ 100%) — Test mAP50 = 0.879 ผ่านเป้าหมาย 0.85

รายละเอียดเต็มใน report.md / ขั้นตอนแลปใน report2.md / ฉบับทฤษฎี+โค้ดใน report3.md / บทพูดพรีเซนต์ใน presentscript.md

⸻

ผู้จัดทำโครงงาน

ลำดับ	ชื่อ-นามสกุล	รหัสนักศึกษา
1	นาย ณฐภาพ สายหล้า	67543210054-2
2	นาย เอกพันธุ์ ทศทิศรังสรรค์	67543210050-0

⸻

1. Dataset mixed2/ คืออะไร

* mixed2/ — DTPG v2 จาก Roboflow (852 ภาพ: train 713 / valid 93 / test 46, YOLO bbox 512px)
* ที่มา: mixed/ v1 480 ภาพ + pseudo-label 524 ภาพที่ผ่านคนตรวจ (ทิ้งภาพซ้ำ 480 + ภาพเปล่า 318)
* raw_images/ — ภาพดิบ 1,322 ใบ + วิดีโอ 2 ไฟล์ (local only, ไม่ push เพราะ ~800MB)

6 classes ที่เทรนจริง (nc=6):

id	class	train	valid	test	หมายเหตุ
0	brand_golden_defect	51	18	10	แบรนด์สุกชำรุด
1	brand_golden_good	147	26	17	เยอะสุดฝั่งแบรนด์
2	street_burnt_defect	191	23	15	จุดอ่อน R 0.6
3	street_burnt_good	47	9	0	น้อยสุด + ไม่มีใน test → 05 ขึ้น N/A
4	street_golden_defect	298	25	13	เยอะสุด
5	street_golden_good	267	26	13	-

แผนเดิม 8 คลาส แต่ brand_burnt_good / brand_burnt_defect 0 ภาพ 0 กล่องทั้งชุด เพราะแบรนด์คุมไฟดี ไม่พบเคสไหม้จริง
โมเดล nc=6 จึงจำแนก 2 คลาสนี้ไม่ได้ — เอาแบรนด์ไหม้จ่อกล้องก็ทายเป็น 1 ใน 6 คลาสที่มีเท่านั้น

⸻

2. สคริปต์ทำอะไรบ้าง (รันตามลำดับ 01→07)

รันด้วย venv:

.\.env\Scripts\python.exe <สคริปต์>

Environment:

* Python 3.13
* PyTorch CUDA 12.6 (torch cu126)
* Ultralytics 8.4
* GPU: NVIDIA RTX 4060

ไฟล์	หน้าที่แบบเข้าใจง่าย	วิธีรัน
01-extract/_frames.py	เปิดวิดีโอด้วย OpenCV เซฟทุก 15 เฟรม (รอบสองทุก 10 เฟรม) ได้ภาพดิบ 1,322 ใบ	python 01-extract/_frames.py
02-train/02-train.py	สั่ง YOLO.train() ได้ best.pt (cls=1.0 + copy_paste/mixup ชดเชยคลาสน้อย, patience=20)	python 02-train/02-train.py
02-train/oversample.py	ก็อปภาพ train คลาสน้อยจาก 713 → 1,164 ภาพ สร้าง mixed2_os/ สำหรับรอบ 5	python 02-train/oversample.py
07-pseudolabel/pseudo_label.py	เอา best รอบแรกทายภาพดิบที่ conf 0.5 ได้ร่าง 524 ภาพ และกัน Data Leakage ด้วยชื่อไฟล์ซ้ำ	python 07-pseudolabel/pseudo_label.py
03-predict.py	ทายภาพนิ่ง Default mixed2/test/images จำนวน 46 ภาพ ได้ 59 กล่อง พร้อมเซฟภาพ + txt	python 03-predict.py [path/0]
04-predict_video.py	ทายวิดีโอแบบ stream=True ทีละเฟรม ป้องกัน RAM เต็ม	python 04-predict_video.py [path]
05-evaluate.py + eval_utils.py	model.val(split=test) แสดง P/R/F1/mAP รายคลาส + Confusion Matrix + กราฟทีละภาพ	python 05-evaluate.py
06-webcam_realtime.py	เปิดกล้องตรวจจับแบบ Realtime กด q เพื่อออก	python 06-webcam_realtime.py

⸻

3. ผลลัพธ์ (เทรน 5 รอบ — รอบ 3 ดีสุด)

ผลการเทรนแต่ละรอบ

* รอบ 1: Baseline yolov8n 50e → Val mAP50 = 0.844
* รอบ 2: yolov8n 80e → Test mAP50 = 0.867
* รอบ 3: yolov8n 80e fixed → Test mAP50 = 0.879 (ดีที่สุด)
* รอบ 4: yolov8s → Test mAP50 = 0.872 (Overfit)
* รอบ 5: Oversample yolov8s 100e → Test mAP50 = 0.878 (ผลใกล้เคียงเดิม)

รอบ 3 — โมเดลที่เลือกใช้งานจริง

Validation

* Precision (P) = 0.806
* Recall (R) = 0.896
* mAP50 = 0.895

Test

* Precision (P) = 0.942
* Recall (R) = 0.857
* mAP50 = 0.879

ผลรายคลาสบน Test Set

* brand_golden_defect → ประสิทธิภาพประมาณ 0.9+
* brand_golden_good → ประสิทธิภาพประมาณ 0.9+
* street_golden_defect → mAP50 = 0.995
* street_golden_good → mAP50 = 0.909
* street_burnt_defect → จุดอ่อนของโมเดล โดยมี Recall = 0.6
* street_burnt_good → ไม่มีข้อมูลใน Test Set จึงไม่สามารถประเมินผลได้

Weights

โมเดลที่ใช้งานจริง:

runs/detect/pathongko_mixed3_fixed/weights/best.pt

best.pt เก็บไว้แบบ Local Only และไม่ได้ Push ขึ้น Repository

จุดอ่อนของโมเดล

street_burnt_defect มี Test Recall เพียง 0.6

วิธีปรับปรุงที่เหมาะสมคือเก็บตัวอย่าง ปาท่องโก๋ไหม้จริงเพิ่ม เนื่องจากการ Copy หรือ Oversample ภาพเดิมซ้ำ ๆ ไม่ได้เพิ่มความหลากหลายของข้อมูลจริง

ข้อจำกัดเพิ่มเติม:

* brand_burnt_good ไม่มีข้อมูล
* brand_burnt_defect ไม่มีข้อมูล
* street_burnt_good ไม่มีข้อมูลใน Test Set

⸻

4. ตัวอย่าง Detection จาก 03-predict.py

ทดสอบกับภาพทั้งหมด 46 ภาพ จาก Test Set

ผลรวมตรวจพบทั้งหมด 59 Bounding Boxes

ตัวอย่างที่ตรวจจับได้ถูกต้อง

Brand

ตรวจพบ 3 กล่อง:

* defect confidence 0.71
* good confidence 0.91
* good confidence 0.93

Street

ตรวจพบ 3 กล่อง:

* good confidence 0.97
* defect confidence 0.94
* defect confidence 0.96

ตัวอย่างเคสที่โมเดลตรวจจับพลาด

street_0379

ในภาพมีปาท่องโก๋ประมาณ 8 ชิ้น แต่โมเดลตรวจพบ burnt_defect เพียง 2 กล่อง

ปัญหานี้สอดคล้องกับผลประเมินที่พบว่า street_burnt_defect มี Recall เพียง 0.6

นอกจากนี้มีภาพ Close-up 2 ภาพที่ไม่เกิด Detection:

* street_0424
* street_0428

⸻

5. สรุปเป้าหมาย สำเร็จ / ไม่สำเร็จ

เป้าหมาย	เกณฑ์	ผลจริง	สถานะ
1. ชุดข้อมูล 6 คลาสจากวิดีโอจริง	ครบ train/valid/test	852 ภาพ (713/93/46) + Label ครบทุกรูป	✅ สำเร็จ
2. เทรนผ่านเป้า	Test mAP50 > 0.85	รอบ 3 Test 0.879 (Val 0.895)	✅ สำเร็จ
3. ประเมิน + ใช้งานจริง	P/R/F1/mAP + ภาพนิ่ง/วิดีโอ/Webcam	05 ครบ + 03 59 กล่อง + 04/06 Stream ไม่ทำให้ RAM เต็ม	✅ สำเร็จ
4. ครบ 8 คลาสตามแผน	มี brand_burnt_*	0 ภาพ 0 กล่อง เพราะไม่พบแบรนด์ไหม้จริง ทำให้จำแนกไม่ได้	❌ ไม่สำเร็จ (ข้อจำกัดข้อมูล)
5. ดันคลาสไหม้	street_burnt_defect Recall ดี + มี street_burnt_good ใน Test	Recall เพียง 0.6 + Class 3 ไม่มีใน Test (Train 47 / Valid 9)	🟡 ไม่ถึงเป้าย่อย

⸻

สรุป

โครงงานนี้สามารถพัฒนาโมเดล YOLOv8 สำหรับตรวจจับและจำแนกปาท่องโก๋จากภาพและวิดีโอจริงได้ โดยโมเดลสามารถแยกประเภทระหว่าง ปาท่องโก๋แบรนด์และร้านทั่วไป รวมถึงลักษณะสุก/ไหม้ และทรงดี/ชำรุด

จากการเทรนทั้งหมด 5 รอบ พบว่าโมเดลจาก รอบที่ 3 (yolov8n 80e fixed) ให้ผลดีที่สุด โดยมีค่า Test mAP50 = 0.879 ซึ่งสูงกว่าเป้าหมายที่กำหนดไว้ที่ 0.85

อย่างไรก็ตาม โมเดลยังมีข้อจำกัดด้านจำนวนและความหลากหลายของข้อมูล โดยเฉพาะกลุ่มปาท่องโก๋ไหม้ ดังนั้นแนวทางพัฒนาต่อไปคือการเก็บข้อมูลปาท่องโก๋ไหม้จากสถานการณ์จริงเพิ่มเติม เพื่อเพิ่มความสามารถในการตรวจจับและลดปัญหาคลาสที่มีข้อมูลไม่สมดุล