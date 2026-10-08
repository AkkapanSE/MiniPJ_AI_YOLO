🥖 MiniPJ_AI_YOLO — Patongko Detection

Brand vs Street Patongko Detection with YOLOv8

โปรเจกต์สำหรับ ตรวจจับและจำแนกปาท่องโก๋ด้วย YOLOv8 Object Detection โดยจำแนกจากแหล่งที่มาและลักษณะรูปทรงของปาท่องโก๋

โมเดลแบ่งปาท่องโก๋ออกเป็น 4 Classes

* 🏪 brand_good — ปาท่องโก๋แบรนด์ รูปทรงดี
* ⚠️ brand_defect — ปาท่องโก๋แบรนด์ รูปทรงชำรุด
* 🛒 street_good — ปาท่องโก๋ร้านทั่วไป รูปทรงดี
* ⚠️ street_defect — ปาท่องโก๋ร้านทั่วไป รูปทรงชำรุด

โปรเจกต์เริ่มจากการถ่ายวิดีโอหน้าร้านจริงจำนวน 2 คลิป

จากนั้นดำเนินการตามขั้นตอน

ถ่ายวิดีโอจริง
      ↓
สกัดเฟรม
      ↓
Label บน Roboflow
      ↓
สร้าง Dataset
      ↓
Train YOLOv8
      ↓
Evaluate
      ↓
Predict Image / Video / Webcam

⸻

📌 Project Status

สถานะล่าสุด: 6 ตุลาคม 2026

* Dataset ปรับเหลือ 4 Classes
* เทรนโมเดลทั้งหมด 5 รอบ
* ตรวจสอบและแก้ไข Label แล้ว
* โมเดลที่ดีที่สุดคือ รอบที่ 3
* Test mAP50 = 0.879
* เป้าหมายที่กำหนดไว้คือ mAP50 > 0.85
* ✅ ผ่านเป้าหมายของโครงงาน

เอกสารเพิ่มเติมภายใน Repository

ไฟล์	รายละเอียด
report.md	รายงานฉบับเต็ม
report2.md	ขั้นตอนการทำ Lab
report3.md	ทฤษฎีที่เกี่ยวข้องและ Source Code
presentscript.md	บทพูดสำหรับนำเสนอโปรเจกต์

⸻

👨‍💻 ผู้จัดทำโครงงาน

ลำดับ	ชื่อ-นามสกุล	รหัสนักศึกษา
1	นาย ณฐภาพ สายหล้า	67543210054-2
2	นาย เอกพันธุ์ ทศทิศรังสรรค์	67543210050-0

⸻

🎯 วัตถุประสงค์ของโครงงาน

1. สร้าง Dataset ปาท่องโก๋จากภาพและวิดีโอที่เก็บจากสถานการณ์จริง
2. พัฒนาโมเดล YOLOv8 สำหรับตรวจจับและจำแนกปาท่องโก๋
3. จำแนกปาท่องโก๋ระหว่าง ร้านแบรนด์ และ ร้านทั่วไป
4. จำแนกลักษณะของปาท่องโก๋เป็น รูปทรงดี และ รูปทรงชำรุด
5. ประเมินประสิทธิภาพโมเดลด้วย Precision, Recall, F1-score และ mAP
6. ทดลองนำโมเดลไปใช้งานกับภาพนิ่ง วิดีโอ และ Webcam แบบ Real-time

⸻

🧠 Model Overview

โปรเจกต์นี้ใช้

YOLOv8 Object Detection

สำหรับตรวจจับตำแหน่งปาท่องโก๋ด้วย Bounding Box พร้อมกับจำแนก Class ของปาท่องโก๋แต่ละชิ้น

Dataset เวอร์ชันปัจจุบันใช้

nc: 4

จำนวนทั้งหมด 4 Classes

แนวคิดการจำแนกคือ

                Patongko
                   │
          ┌────────┴────────┐
        Brand             Street
          │                  │
     ┌────┴────┐        ┌────┴────┐
    Good     Defect     Good     Defect

⸻

📂 1. Dataset — mixed2/

Dataset หลักที่ใช้ในการเทรนคือ

mixed2/

เป็น Dataset จาก Roboflow ในรูปแบบ YOLO Bounding Box

ขนาดภาพ

512 px

Dataset มีข้อมูลจากภาพที่สกัดจากวิดีโอหน้าร้านจริง

แบ่งออกเป็น

Train
Validation
Test

ข้อมูลดิบต้นฉบับอยู่ใน

raw_images/

ประกอบด้วย

* ภาพดิบประมาณ 1,322 ภาพ
* วิดีโอต้นฉบับจำนวน 2 ไฟล์
* ขนาดรวมประมาณ 800 MB

raw_images/ เก็บไว้เฉพาะ Local และไม่ได้ Push ขึ้น Repository เนื่องจากมีขนาดค่อนข้างใหญ่

⸻

🏷️ 2. Classes ที่ใช้ในการเทรน

จำนวน Class ที่ใช้จริง

nc: 4

ID	Class	ความหมาย
0	brand_defect	ปาท่องโก๋แบรนด์ รูปทรงชำรุด
1	brand_good	ปาท่องโก๋แบรนด์ รูปทรงดี
2	street_defect	ปาท่องโก๋ร้านทั่วไป รูปทรงชำรุด
3	street_good	ปาท่องโก๋ร้านทั่วไป รูปทรงดี

ตัวอย่างการกำหนด Class ใน data.yaml

nc: 4
names:
  0: brand_defect
  1: brand_good
  2: street_defect
  3: street_good

⸻

🔄 3. การปรับจาก 6 Classes เหลือ 4 Classes

Dataset ก่อนหน้านี้แบ่งออกเป็น 6 Classes

brand_golden_defect
brand_golden_good
street_burnt_defect
street_burnt_good
street_golden_defect
street_golden_good

แต่พบว่าการแบ่งตามระดับ Golden / Burnt ทำให้เกิดปัญหาเรื่องจำนวนข้อมูลในแต่ละ Class ไม่สมดุล

โดยเฉพาะข้อมูลกลุ่ม

street_burnt_good
street_burnt_defect

มีจำนวนข้อมูลน้อยกว่ากลุ่มอื่น และบาง Class มีข้อมูลใน Test Set ไม่เพียงพอสำหรับการประเมินอย่างเหมาะสม

จึงปรับโครงสร้าง Dataset ใหม่ให้เหลือ 4 Classes

โดยรวมข้อมูล golden และ burnt เข้าด้วยกัน แล้วให้โมเดลเน้นจำแนกสองคุณลักษณะหลักคือ

1. แหล่งที่มา
   Brand / Street
2. สภาพรูปทรง
   Good / Defect

ดังนั้น

brand_golden_good
→ brand_good
brand_golden_defect
→ brand_defect
street_golden_good
street_burnt_good
→ street_good
street_golden_defect
street_burnt_defect
→ street_defect

ข้อดีของการปรับเหลือ 4 Classes คือ

* ลดปัญหา Class Imbalance
* ลดความซับซ้อนในการจำแนก
* เพิ่มจำนวนตัวอย่างต่อ Class
* ลดความสับสนระหว่าง Golden และ Burnt
* ทำให้ Dataset เหมาะกับข้อมูลจริงมากขึ้น
* ทำให้โมเดลมีโอกาส Generalize ได้ดีขึ้น

⸻

🛠️ 4. Workflow ของโปรเจกต์

ขั้นตอนการพัฒนาโมเดล

วิดีโอหน้าร้านจริง 2 คลิป
          ↓
OpenCV Extract Frames
          ↓
ภาพดิบ 1,322 ภาพ
          ↓
Label ด้วย Roboflow
          ↓
ตรวจสอบ / แก้ไข Label
          ↓
แบ่ง Dataset
Train / Valid / Test
          ↓
Train YOLOv8
          ↓
Evaluate Model
          ↓
เลือก Best Model
          ↓
Predict Image
          ↓
Predict Video
          ↓
Webcam Realtime

⸻

💻 5. Environment

Environment ที่ใช้ในการพัฒนาและเทรนโมเดล

Component	Version / Hardware
Python	3.13
PyTorch	CUDA 12.6 (cu126)
Ultralytics	8.4
GPU	NVIDIA GeForce RTX 4060 Laptop GPU

รันผ่าน Virtual Environment

.\.env\Scripts\python.exe <script>

ตัวอย่าง

.\.env\Scripts\python.exe 03-predict.py

⸻

📜 6. Scripts

Script	หน้าที่	วิธีรัน
01-extract/_frames.py	สกัดภาพจากวิดีโอด้วย OpenCV	python 01-extract/_frames.py
02-train/02-train.py	Train YOLOv8 และสร้าง best.pt	python 02-train/02-train.py
02-train/oversample.py	Oversample Class ที่มีข้อมูลน้อย	python 02-train/oversample.py
03-predict.py	Predict ภาพนิ่ง	python 03-predict.py [path/0]
04-predict_video.py	Predict วิดีโอ	python 04-predict_video.py [path]
05-evaluate.py	Evaluate โมเดลบน Test Set	python 05-evaluate.py
eval_utils.py	Utility สำหรับการประเมินผล	ใช้ร่วมกับ 05-evaluate.py
06-webcam_realtime.py	ตรวจจับผ่าน Webcam แบบ Real-time	python 06-webcam_realtime.py
07-pseudolabel/pseudo_label.py	สร้าง Pseudo Label	python 07-pseudolabel/pseudo_label.py

⸻

01-extract/_frames.py

ใช้ OpenCV สำหรับเปิดวิดีโอและสกัดภาพออกมาเป็น Frame

รอบแรก

Save ทุก 15 Frames

รอบที่สอง

Save ทุก 10 Frames

รวมภาพดิบประมาณ

1,322 images

⸻

02-train/02-train.py

ใช้สำหรับ Train YOLOv8 ผ่าน

YOLO.train()

มีการทดลองปรับ Hyperparameters เช่น

cls = 1.0
copy_paste
mixup
patience = 20

โมเดลที่ได้หลัง Training

best.pt

⸻

02-train/oversample.py

ใช้สำหรับเพิ่มจำนวนข้อมูลของ Class ที่มีข้อมูลน้อย โดยนำข้อมูลใน Training Set มาทำ Oversampling

วัตถุประสงค์คือช่วยลดผลกระทบจาก

Class Imbalance

อย่างไรก็ตาม การ Oversample ภาพเดิมเพียงอย่างเดียวไม่ได้เพิ่มความหลากหลายของข้อมูลจริง จึงยังควรเก็บภาพใหม่เพิ่มเติม

⸻

07-pseudolabel/pseudo_label.py

ใช้โมเดลที่ผ่านการ Training แล้วมาช่วยสร้าง Label เบื้องต้นให้กับภาพดิบ

กำหนด Confidence

conf = 0.5

จากนั้นนำผลที่โมเดลตรวจจับได้ไปตรวจสอบและแก้ไข Label ด้วยคนอีกครั้ง

เพื่อให้ Dataset ที่นำไปใช้ Training มีความถูกต้องมากขึ้น

⸻

03-predict.py

ใช้สำหรับ Predict ภาพนิ่ง

ค่า Default

mixed2/test/images

โปรแกรมสามารถบันทึกผลลัพธ์ทั้ง

Prediction Images
+
YOLO TXT Results

⸻

04-predict_video.py

ใช้สำหรับตรวจจับปาท่องโก๋จากวิดีโอ

ใช้

stream=True

เพื่อประมวลผลวิดีโอทีละ Frame

ช่วยลดการใช้ RAM และป้องกันปัญหาการโหลดวิดีโอทั้งหมดเข้า Memory พร้อมกัน

⸻

05-evaluate.py

ใช้

model.val(split="test")

สำหรับประเมินโมเดลกับ Test Dataset

Metric ที่ใช้ ได้แก่

* Precision
* Recall
* F1-score
* mAP50
* mAP50-95
* Confusion Matrix
* ผลลัพธ์ราย Class

⸻

06-webcam_realtime.py

ใช้สำหรับตรวจจับปาท่องโก๋ผ่าน Webcam แบบ Real-time

กด

q

เพื่อหยุดการทำงาน

⸻

📊 7. ผลการ Training

โปรเจกต์มีการทดลอง Training ทั้งหมด 5 รอบ

รอบ	Model / การทดลอง	ผลลัพธ์
1	Baseline YOLOv8n 50 epochs	Val mAP50 = 0.844
2	YOLOv8n 80 epochs	Test mAP50 = 0.867
3	YOLOv8n 80 epochs + Fixed Labels	Test mAP50 = 0.879
4	YOLOv8s	Test mAP50 = 0.872
5	Oversample + YOLOv8s 100 epochs	Test mAP50 = 0.878

ตารางผลด้านบนเป็นผลจากการทดลองโมเดลในขั้นตอนการพัฒนาโครงงาน ก่อนและระหว่างการปรับปรุง Dataset

⸻

🏆 8. Best Model

โมเดลที่ให้ผลดีที่สุดจากการทดลองคือ

YOLOv8n
80 Epochs
Fixed Labels
Training Round 3

ผล Validation

Metric	Result
Precision	0.806
Recall	0.896
mAP50	0.895

ผล Test

Metric	Result
Precision	0.942
Recall	0.857
mAP50	0.879

เป้าหมายของโปรเจกต์คือ

Test mAP50 > 0.85

ผลที่ได้

Test mAP50 = 0.879

ดังนั้น

✅ PASS

⸻

⚖️ 9. Model Weights

Weights ของโมเดลที่ดีที่สุด

runs/detect/pathongko_mixed3_fixed/weights/best.pt

ไฟล์ best.pt เก็บไว้เฉพาะ Local และไม่ได้ Push ขึ้น Repository

⸻

🔍 10. การจำแนกผลลัพธ์

โมเดลจะตรวจจับปาท่องโก๋แต่ละชิ้นแล้วระบุหนึ่งใน 4 Classes

ตัวอย่าง

brand_good 0.93

หมายถึง

ปาท่องโก๋จากร้านแบรนด์
รูปทรงดี
Confidence = 93%

ตัวอย่าง

street_defect 0.94

หมายถึง

ปาท่องโก๋จากร้านทั่วไป
รูปทรงชำรุด
Confidence = 94%

⸻

🖼️ 11. Prediction

สามารถทดสอบโมเดลกับภาพนิ่งได้ด้วย

python 03-predict.py

หรือกำหนด Path เอง

python 03-predict.py path/to/image.jpg

โมเดลจะแสดง

Bounding Box
Class
Confidence Score

ของปาท่องโก๋แต่ละชิ้นที่ตรวจพบ

⸻

🎥 12. Video Detection

สามารถตรวจจับจากวิดีโอได้ด้วย

python 04-predict_video.py path/to/video.mp4

โปรแกรมใช้

stream=True

เพื่อประมวลผลทีละ Frame

⸻

📹 13. Webcam Real-time

เปิด Webcam และตรวจจับแบบ Real-time

python 06-webcam_realtime.py

กด

q

เพื่อออกจากโปรแกรม

⸻

⚠️ 14. ข้อจำกัดของโมเดล

แม้โมเดลจะสามารถตรวจจับและจำแนกปาท่องโก๋ได้ แต่ยังมีข้อจำกัดบางประการ

1. จำนวนข้อมูลแต่ละ Class ไม่เท่ากัน

Dataset ยังมีความแตกต่างของจำนวนข้อมูลในแต่ละ Class

จึงอาจทำให้โมเดลเรียนรู้บาง Class ได้ดีกว่า Class อื่น

2. ภาพ Close-up

ภาพที่ถ่ายใกล้มากอาจทำให้ลักษณะของปาท่องโก๋แตกต่างจากข้อมูลส่วนใหญ่ใน Training Dataset

จึงอาจเกิดกรณี

No Detection

3. มุมกล้อง

หากปาท่องโก๋อยู่ในมุมที่โมเดลไม่เคยเห็นระหว่าง Training ประสิทธิภาพอาจลดลง

4. สภาพแสง

แสงที่แตกต่างจาก Dataset เช่น

มืดมาก
สว่างมาก
แสงสีเหลือง
เงาสะท้อน

อาจส่งผลต่อความแม่นยำ

5. รูปทรง Good / Defect

การแบ่งระหว่าง good และ defect ขึ้นอยู่กับเกณฑ์การ Label ของผู้จัดทำ

จึงต้องกำหนดมาตรฐานการ Label ให้ชัดเจนและสม่ำเสมอ

⸻

💡 15. แนวทางการพัฒนาต่อ

แนวทางการพัฒนา Dataset และโมเดลในอนาคต

1. เก็บข้อมูลของทั้ง 4 Classes เพิ่ม
2. ทำจำนวนข้อมูลแต่ละ Class ให้ใกล้เคียงกัน
3. เพิ่มภาพจากหลายร้าน
4. เพิ่มมุมกล้องที่หลากหลาย
5. เพิ่มภาพ Close-up
6. เพิ่มระยะใกล้ กลาง และไกล
7. เพิ่มสภาพแสงที่แตกต่างกัน
8. เพิ่ม Background ที่หลากหลาย
9. ทดลอง Data Augmentation เพิ่มเติม
10. ทดลอง Hyperparameter Tuning
11. ทดสอบกับร้านที่ไม่เคยอยู่ใน Training Dataset
12. เพิ่ม Dataset จากสถานการณ์จริงแทนการ Oversample ภาพเดิม

⸻

✅ 16. สรุปเป้าหมายของโครงงาน

เป้าหมาย	เกณฑ์	ผลลัพธ์	สถานะ
Dataset ปาท่องโก๋จากข้อมูลจริง	มี Train / Valid / Test	สร้าง Dataset สำเร็จ	✅ สำเร็จ
จำแนก 4 Classes	Brand/Street + Good/Defect	nc=4	✅ สำเร็จ
Training ผ่านเป้าหมาย	Test mAP50 > 0.85	0.879	✅ สำเร็จ
ประเมินผลโมเดล	P / R / F1 / mAP	มี 05-evaluate.py	✅ สำเร็จ
Predict ภาพ	ตรวจจับภาพนิ่งได้	03-predict.py	✅ สำเร็จ
Predict วิดีโอ	ตรวจจับวิดีโอได้	04-predict_video.py	✅ สำเร็จ
Webcam Real-time	ตรวจจับจากกล้องได้	06-webcam_realtime.py	✅ สำเร็จ

⸻

🏁 Conclusion

โครงงานนี้สามารถพัฒนาโมเดล YOLOv8 Object Detection สำหรับตรวจจับและจำแนกปาท่องโก๋จากภาพและวิดีโอจริงได้สำเร็จ

โมเดลถูกปรับให้จำแนกทั้งหมด 4 Classes

brand_good
brand_defect
street_good
street_defect

โดยพิจารณาจากสองคุณลักษณะหลัก ได้แก่

แหล่งที่มา
Brand / Street
+
สภาพรูปทรง
Good / Defect

การปรับจาก 6 Classes เหลือ 4 Classes ช่วยลดความซับซ้อนของปัญหา และลดผลกระทบจากจำนวนข้อมูลของกลุ่มปาท่องโก๋ไหม้ที่มีน้อย

จากการทดลอง Training ทั้งหมด 5 รอบ พบว่าโมเดลที่ให้ผลดีที่สุดคือ

YOLOv8n
80 Epochs
Fixed Labels
Round 3

โดยให้ผล

Validation mAP50 = 0.895
Test mAP50       = 0.879

สูงกว่าเป้าหมายที่กำหนดไว้

mAP50 > 0.85

จึงถือว่าโมเดลสามารถบรรลุเป้าหมายหลักของโครงงานได้

แนวทางสำคัญในการพัฒนาต่อคือการเพิ่มจำนวนข้อมูลจริงของทั้ง 4 Classes ให้มีความสมดุลมากขึ้น รวมถึงเพิ่มความหลากหลายของมุมกล้อง ระยะการถ่าย แสง และสภาพแวดล้อม เพื่อเพิ่มความสามารถในการตรวจจับกับสถานการณ์จริง

⸻

📷 Project Result Images

Frame Extraction

⸻

Roboflow Labeling

⸻

Training Results

⸻

Confusion Matrix

⸻

Evaluation

⸻

Prediction — Brand

⸻

Prediction — Street

⸻

Video Detection

⸻

📌 Final Result

Classes    : 4
0 : brand_defect
1 : brand_good
2 : street_defect
3 : street_good
Best Model : YOLOv8n — Round 3
Val mAP50  : 0.895
Test mAP50 : 0.879
Target     : > 0.850
Result     : ✅ PASS