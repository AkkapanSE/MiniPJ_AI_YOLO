🥖 MiniPJ_AI_YOLO — Patongko Detection

Brand vs Street Patongko Detection with YOLOv8

Mini Project ด้าน Artificial Intelligence และ Object Detection สำหรับตรวจจับและจำแนกปาท่องโก๋ด้วย YOLOv8

ระบบสามารถตรวจจับปาท่องโก๋แต่ละชิ้นด้วย Bounding Box และจำแนกตาม 2 คุณลักษณะหลัก ได้แก่

* แหล่งที่มา: Brand / Street
* สภาพรูปทรง: Good / Defect

โมเดลเวอร์ชันปัจจุบันออกแบบให้จำแนกทั้งหมด 4 Classes

Class	ความหมาย
brand_good	ปาท่องโก๋แบรนด์ รูปทรงดี
brand_defect	ปาท่องโก๋แบรนด์ รูปทรงชำรุด
street_good	ปาท่องโก๋ร้านทั่วไป รูปทรงดี
street_defect	ปาท่องโก๋ร้านทั่วไป รูปทรงชำรุด

⸻

📌 Project Overview

กระบวนการพัฒนาเริ่มจากการถ่ายวิดีโอปาท่องโก๋จากสถานการณ์จริงจำนวน 2 คลิป จากนั้นนำมาสกัดเป็นภาพ ทำ Label บน Roboflow สร้าง Dataset และ Train ด้วย YOLOv8

วิดีโอหน้าร้านจริง
        ↓
Extract Frames
        ↓
Label บน Roboflow
        ↓
ตรวจสอบและแก้ไข Label
        ↓
สร้าง Dataset
        ↓
Train YOLOv8
        ↓
Evaluate Model
        ↓
Predict Image / Video / Webcam

⸻

👨‍💻 ผู้จัดทำโครงงาน

ลำดับ	ชื่อ-นามสกุล	รหัสนักศึกษา
*1	นาย ณฐภาพ สายหล้า	67543210054-2
*2	นาย เอกพันธุ์ ทศทิศรังสรรค์	67543210050-0

⸻

🎯 วัตถุประสงค์

1. สร้าง Dataset ปาท่องโก๋จากภาพและวิดีโอจริง
2. พัฒนาโมเดล YOLOv8 สำหรับตรวจจับและจำแนกปาท่องโก๋
3. จำแนกปาท่องโก๋ระหว่าง Brand และ Street
4. จำแนกรูปทรงเป็น Good และ Defect
5. ประเมินโมเดลด้วย Precision, Recall, F1-score และ mAP
6. ทดลองใช้งานกับภาพนิ่ง วิดีโอ และ Webcam แบบ Real-time

⸻

📂 1. Dataset

Dataset หลักของโปรเจกต์จัดเตรียมผ่าน Roboflow และ Export ในรูปแบบ YOLO Bounding Box

ข้อมูลต้นฉบับมาจากวิดีโอที่ถ่ายจากสถานการณ์จริง และนำมาสกัดเป็นภาพด้วย OpenCV

Raw Data

โฟลเดอร์

raw_images/

ประกอบด้วย

* ภาพดิบประมาณ 1,322 ภาพ
* วิดีโอต้นฉบับ 2 ไฟล์
* ขนาดรวมประมาณ 800 MB

raw_images/ เก็บไว้เฉพาะ Local และไม่ได้ Push ขึ้น Repository เนื่องจากมีขนาดไฟล์ค่อนข้างใหญ่

Dataset แบ่งออกเป็น

* Train
* Validation
* Test

⸻

🖼️ Frame Extraction

การสกัดภาพแบ่งเป็น 2 รอบ

รอบที่ 1 : Save ทุก 15 Frames
รอบที่ 2 : Save ทุก 10 Frames

ภาพตัวอย่างจากการสกัดเฟรม

⸻

🏷️ Roboflow Labeling

ใช้ Roboflow สำหรับ

* สร้าง Bounding Box
* กำหนด Class
* ตรวจสอบ Label
* แบ่ง Train / Validation / Test
* Export Dataset สำหรับ YOLO

⸻

🏷️ 2. Class Structure

โมเดลเวอร์ชันปัจจุบันใช้ทั้งหมด 4 Classes

nc: 4
names:
  0: brand_defect
  1: brand_good
  2: street_defect
  3: street_good

ID	Class	ประเภท	รูปทรง
0	brand_defect	Brand	Defect
1	brand_good	Brand	Good
2	street_defect	Street	Defect
3	street_good	Street	Good

โครงสร้างการจำแนก

                  Patongko
                     │
           ┌─────────┴─────────┐
         Brand               Street
           │                    │
      ┌────┴────┐          ┌────┴────┐
    Good      Defect      Good      Defect

⸻

🔄 3. การปรับจาก 6 Classes เหลือ 4 Classes

ในช่วงแรก Dataset ถูกออกแบบเป็น 6 Classes

brand_golden_defect
brand_golden_good
street_burnt_defect
street_burnt_good
street_golden_defect
street_golden_good

หลังจากทดลองพบว่าการแยก Golden และ Burnt ทำให้จำนวนข้อมูลในแต่ละ Class ไม่สมดุล โดยเฉพาะข้อมูลกลุ่มปาท่องโก๋ไหม้

บาง Class มีตัวอย่างน้อยมาก และบาง Class ไม่มีข้อมูลเพียงพอใน Test Set ทำให้ประเมินผลได้ไม่สมบูรณ์

จึงปรับโครงสร้างใหม่ให้เหลือ 4 Classes

Class Mapping

Class เดิม	Class ใหม่
brand_golden_good	brand_good
brand_golden_defect	brand_defect
street_golden_good	street_good
street_burnt_good	street_good
street_golden_defect	street_defect
street_burnt_defect	street_defect

โมเดลจึงเน้นเพียง 2 คุณลักษณะหลัก

Brand / Street
      +
Good / Defect

ข้อดีของการปรับเหลือ 4 Classes

* ลดความซับซ้อนในการจำแนก
* ลดผลกระทบจาก Class Imbalance
* เพิ่มจำนวนตัวอย่างต่อ Class
* ลดความสับสนระหว่าง Golden และ Burnt
* ทำให้เกณฑ์การ Label ชัดเจนขึ้น
* เหมาะกับข้อมูลจริงที่สามารถเก็บได้
* ช่วยให้โมเดลมีโอกาส Generalize ได้ดีขึ้น

⸻

🛠️ 4. Project Workflow

วิดีโอจริง 2 คลิป
        ↓
OpenCV Extract Frames
        ↓
ภาพดิบประมาณ 1,322 ภาพ
        ↓
Roboflow Labeling
        ↓
ตรวจสอบและแก้ไข Label
        ↓
Train / Validation / Test
        ↓
YOLOv8 Training
        ↓
Model Evaluation
        ↓
เลือก Best Model
        ↓
Image Prediction
        ↓
Video Prediction
        ↓
Webcam Real-time

⸻

💻 5. Environment

Component	Version / Hardware
Python	3.13
PyTorch	CUDA 12.6 (cu126)
Ultralytics	8.4
GPU	NVIDIA GeForce RTX 4060 Laptop GPU
Labeling	Roboflow
Image Processing	OpenCV

รันโปรแกรมผ่าน Virtual Environment

.\.env\Scripts\python.exe <script>

ตัวอย่าง

.\.env\Scripts\python.exe 03-predict.py

⸻

📜 6. Project Scripts

Script	หน้าที่
01-extract/_frames.py	สกัดภาพจากวิดีโอด้วย OpenCV
02-train/02-train.py	Train โมเดล YOLOv8
02-train/oversample.py	เพิ่มจำนวนข้อมูลของ Class ที่มีข้อมูลน้อย
03-predict.py	ตรวจจับปาท่องโก๋จากภาพนิ่ง
04-predict_video.py	ตรวจจับปาท่องโก๋จากวิดีโอ
05-evaluate.py	ประเมินประสิทธิภาพโมเดล
eval_utils.py	Utility สำหรับการประเมินผล
06-webcam_realtime.py	ตรวจจับจาก Webcam แบบ Real-time
07-pseudolabel/pseudo_label.py	ช่วยสร้าง Pseudo Label

⸻

6.1 01-extract/_frames.py

ใช้ OpenCV เปิดวิดีโอและสกัด Frame ออกมาเป็นภาพ

รอบที่ 1 : Save ทุก 15 Frames
รอบที่ 2 : Save ทุก 10 Frames

ได้ภาพดิบรวมประมาณ

1,322 images

⸻

6.2 02-train/02-train.py

ใช้สำหรับ Train โมเดลผ่าน Ultralytics YOLO

YOLO.train()

มีการทดลองปรับ Hyperparameters เช่น

cls = 1.0
copy_paste
mixup
patience = 20

หลัง Train จะได้ Weight หลัก

best.pt

⸻

6.3 02-train/oversample.py

ใช้สำหรับเพิ่มจำนวนข้อมูลของ Class ที่มีตัวอย่างน้อยใน Training Set

วัตถุประสงค์หลักคือช่วยลดผลกระทบจาก

Class Imbalance

อย่างไรก็ตาม Oversampling เป็นการนำภาพเดิมกลับมาใช้ซ้ำ จึงไม่สามารถทดแทนการเก็บข้อมูลจริงใหม่ได้ทั้งหมด

⸻

6.4 07-pseudolabel/pseudo_label.py

ใช้โมเดลที่ Train แล้วช่วยสร้าง Label เบื้องต้นให้ภาพที่ยังไม่มี Annotation

ค่าที่ใช้

conf = 0.5

จากนั้นตรวจสอบและแก้ไข Label ด้วยคนอีกครั้งก่อนนำข้อมูลไปใช้ Training

⸻

6.5 03-predict.py

ใช้สำหรับตรวจจับปาท่องโก๋จากภาพนิ่ง

python 03-predict.py

หรือระบุ Path

python 03-predict.py path/to/image.jpg

ผลลัพธ์ประกอบด้วย

* Bounding Box
* Class
* Confidence Score
* Prediction Image
* YOLO TXT Result

⸻

6.6 04-predict_video.py

ใช้สำหรับตรวจจับปาท่องโก๋จากวิดีโอ

python 04-predict_video.py path/to/video.mp4

ใช้

stream=True

เพื่อประมวลผลทีละ Frame และช่วยลดการใช้ RAM

⸻

6.7 05-evaluate.py

ใช้

model.val(split="test")

สำหรับประเมินโมเดลด้วย Test Dataset

Metrics ที่ใช้ประกอบด้วย

* Precision
* Recall
* F1-score
* mAP50
* mAP50-95
* Confusion Matrix
* ผลลัพธ์ราย Class

⸻

6.8 06-webcam_realtime.py

ใช้ตรวจจับปาท่องโก๋ผ่าน Webcam แบบ Real-time

python 06-webcam_realtime.py

กด

q

เพื่อหยุดโปรแกรม

⸻

📊 7. Training Experiments

ในขั้นตอนพัฒนาโมเดลมีการทดลอง Train ทั้งหมด 5 รอบ

รอบ	การทดลอง	ผลลัพธ์
1	Baseline YOLOv8n, 50 Epochs	Val mAP50 = 0.844
2	YOLOv8n, 80 Epochs	Test mAP50 = 0.867
3	YOLOv8n, 80 Epochs + Fixed Labels	Test mAP50 = 0.879
4	YOLOv8s	Test mAP50 = 0.872
5	Oversample + YOLOv8s, 100 Epochs	Test mAP50 = 0.878

หมายเหตุ: ผลลัพธ์ส่วนนี้เป็นผลจากการทดลองก่อนปรับ Dataset รุ่นสุดท้ายจาก 6 Classes เหลือ 4 Classes ดังนั้นค่าของโมเดล 4 Classes รุ่นใหม่ควรถูกนำมาอัปเดตในส่วนนี้หลัง Train ใหม่

⸻

🏆 8. Previous Best Experiment

จากการทดลองเดิมทั้ง 5 รอบ โมเดลที่ให้ผลดีที่สุดคือ

Model  : YOLOv8n
Epochs : 80
Labels : Fixed
Round  : 3

Validation Results

Metric	Result
Precision	0.806
Recall	0.896
mAP50	0.895

Test Results

Metric	Result
Precision	0.942
Recall	0.857
mAP50	0.879

เป้าหมายของการทดลอง

Test mAP50 > 0.85

ผลที่ได้

Test mAP50 = 0.879

ผลการทดลองเดิม

✅ PASS

⸻

📈 Training Results

⸻

📊 Confusion Matrix

⸻

📋 Evaluation Result

⸻

⚖️ 9. Model Weights

Weight ของ Best Experiment เดิม

runs/detect/pathongko_mixed3_fixed/weights/best.pt

best.pt เก็บไว้เฉพาะ Local และไม่ได้ Push ขึ้น Repository

หลังจาก Train Dataset 4 Classes ใหม่ ควรเปลี่ยน Path นี้เป็น Weight ของโมเดลรุ่นล่าสุด

⸻

🔍 10. Prediction Output

โมเดล 4 Classes จะให้ผลในรูปแบบ

Class + Confidence

ตัวอย่าง

brand_good 0.93

หมายถึง

* ปาท่องโก๋จากร้านแบรนด์
* รูปทรงดี
* Confidence = 93%

อีกตัวอย่าง

street_defect 0.94

หมายถึง

* ปาท่องโก๋จากร้านทั่วไป
* รูปทรงชำรุด
* Confidence = 94%

⸻

🖼️ 11. Prediction Examples

Brand

Street

Video Detection

⸻

⚠️ 12. Limitations

12.1 Class Imbalance

จำนวนข้อมูลของแต่ละ Class อาจไม่เท่ากัน ทำให้โมเดลเรียนรู้บาง Class ได้ดีกว่า Class อื่น

12.2 Close-up Images

ภาพที่ถ่ายใกล้มากอาจมี Scale แตกต่างจากข้อมูลส่วนใหญ่ใน Training Dataset ทำให้มีโอกาสเกิด

No Detection

12.3 Camera Angle

มุมกล้องที่แตกต่างจากข้อมูลที่ใช้ Train อาจทำให้ Accuracy ลดลง

12.4 Lighting Conditions

สภาพแสงที่แตกต่างจาก Dataset เช่น

* ภาพมืด
* ภาพสว่างมาก
* แสงสีเหลือง
* เงาสะท้อน
* แสงจากหลายทิศทาง

อาจส่งผลต่อความแม่นยำ

12.5 Good / Defect Definition

การแบ่งระหว่าง Good และ Defect ขึ้นอยู่กับเกณฑ์การ Label ของผู้จัดทำ

จึงควรกำหนดมาตรฐานการ Label ให้ชัดเจนและใช้เกณฑ์เดียวกันตลอด Dataset

⸻

💡 13. Future Improvements

แนวทางพัฒนาต่อ

1. เพิ่มจำนวนข้อมูลของทั้ง 4 Classes
2. ทำจำนวนข้อมูลแต่ละ Class ให้สมดุลมากขึ้น
3. เพิ่มข้อมูลจากหลายร้าน
4. เพิ่มมุมกล้องที่หลากหลาย
5. เพิ่มภาพ Close-up
6. เพิ่มภาพระยะใกล้ กลาง และไกล
7. เพิ่มสภาพแสงหลายรูปแบบ
8. เพิ่ม Background ที่หลากหลาย
9. ทดลอง Data Augmentation เพิ่มเติม
10. ทดลอง Hyperparameter Tuning
11. ทดสอบกับร้านที่ไม่เคยอยู่ใน Training Dataset
12. เพิ่มข้อมูลจริงแทนการ Oversample ภาพเดิม
13. Train และ Evaluate Dataset 4 Classes ใหม่ทั้งหมด

⸻

✅ 14. Project Goals

เป้าหมาย	เกณฑ์	สถานะ
สร้าง Dataset จากข้อมูลจริง	มี Train / Validation / Test	✅ สำเร็จ
จำแนก Brand / Street	จำแนกแหล่งที่มาได้	✅ สำเร็จ
จำแนก Good / Defect	จำแนกรูปทรงได้	✅ สำเร็จ
ปรับโครงสร้างเป็น 4 Classes	nc=4	✅ สำเร็จ
Predict ภาพนิ่ง	03-predict.py	✅ สำเร็จ
Predict วิดีโอ	04-predict_video.py	✅ สำเร็จ
Webcam Real-time	06-webcam_realtime.py	✅ สำเร็จ
Evaluate โมเดล	P / R / F1 / mAP	✅ มีระบบประเมิน
Evaluate โมเดล 4 Classes รุ่นสุดท้าย	Train / Test ใหม่	🔄 รออัปเดต

⸻

📚 15. Project Documents

File	Description
report.md	รายงานโปรเจกต์ฉบับเต็ม
report2.md	ขั้นตอนการทำ Lab และการทดลอง
report3.md	ทฤษฎีที่เกี่ยวข้องและ Source Code
presentscript.md	บทพูดสำหรับนำเสนอ Mini Project

⸻

🏁 Conclusion

โปรเจกต์นี้พัฒนาระบบ YOLOv8 Object Detection สำหรับตรวจจับและจำแนกปาท่องโก๋จากข้อมูลจริง

โครงสร้าง Class ปัจจุบันถูกปรับให้เหลือทั้งหมด 4 Classes

brand_defect
brand_good
street_defect
street_good

โมเดลพิจารณาคุณลักษณะหลัก 2 ส่วน

Brand / Street
      +
Good / Defect

การปรับจาก 6 Classes เหลือ 4 Classes ช่วยลดความซับซ้อนในการจำแนก ลดผลกระทบจากจำนวนข้อมูลกลุ่ม Burnt ที่มีไม่เพียงพอ และทำให้โครงสร้าง Dataset เหมาะกับข้อมูลจริงมากขึ้น

จากการทดลองก่อนปรับ Dataset รอบที่ให้ผลดีที่สุดคือ

YOLOv8n
80 Epochs
Fixed Labels
Round 3

โดยให้ผล

Validation mAP50 = 0.895
Test mAP50       = 0.879

สูงกว่าเป้าหมาย

mAP50 > 0.85

อย่างไรก็ตาม ผลดังกล่าวเป็นผลจาก Dataset รุ่นก่อนหน้า

หลังจากปรับโครงสร้างเป็น 4 Classes ควร Train และ Evaluate ใหม่ เพื่อให้ค่า Precision, Recall, F1-score และ mAP สะท้อนประสิทธิภาพของโมเดล 4 Classes อย่างถูกต้อง

แนวทางสำคัญในการพัฒนาต่อคือเพิ่มจำนวนและความหลากหลายของข้อมูลจริงจากหลายร้าน หลายมุมกล้อง หลายระยะ และหลายสภาพแสง เพื่อเพิ่มความสามารถในการนำโมเดลไปใช้งานในสถานการณ์จริง

⸻

📌 Current Configuration

Classes : 4
0 : brand_defect
1 : brand_good
2 : street_defect
3 : street_good

⸻

📌 Previous Best Experiment

Model      : YOLOv8n
Round      : 3
Epochs     : 80
Val mAP50  : 0.895
Test mAP50 : 0.879
Target     : > 0.850
Result     : ✅ PASS

ผลด้านบนเป็นผลจาก Dataset รุ่นก่อนปรับเป็น 4 Classes และควรอัปเดตด้วยผลจากโมเดล 4 Classes รุ่นสุดท้ายหลัง Train ใหม่