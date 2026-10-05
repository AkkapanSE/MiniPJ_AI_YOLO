"""สคริปต์ทดสอบโมเดล (inference) ด้วย best.pt รอบ 3 (mixed2 label แก้แล้ว, mAP50 0.895).

วิธีใช้:
    .\.env\Scripts\python.exe 03-predict.py                        # ทดสอบกับ mixed2/test/images
    .\.env\Scripts\python.exe 03-predict.py <path-รูป/วิดีโอ/0>    # ทดสอบกับไฟล์หรือกล้อง (0 = webcam)

ผลลัพธ์: runs/detect/predict_demo/ (ภาพตีกรอบ + labels)
"""
import sys
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "runs" / "detect" / "pathongko_mixed3_fixed" / "weights" / "best.pt"

# โมเดลรอบ 3: yolov8n เทรนบน mixed2 label แก้แล้ว 80 epochs (train 713 / val mAP50 0.895 / test 0.879)
# ถ้ายังไม่เคยเทรน ให้รัน 02-train/02-train.py ก่อน

if __name__ == "__main__":
    assert WEIGHTS.exists(), f"ไม่พบ weights: {WEIGHTS} — รัน 02-train/02-train.py ก่อน"

    source = sys.argv[1] if len(sys.argv) > 1 else str(BASE / "mixed2" / "test" / "images")
    print(f"🔍 ทดสอบโมเดล: {WEIGHTS.name} กับ {source}")

    model = YOLO(str(WEIGHTS))
    # stream=True: ประมวลผลทีละภาพ ไม่เก็บผลทั้งหมดไว้ในแรม (กันแรมเต็ม)
    n_img, n_box = 0, 0
    for r in model.predict(
        source=source,
        conf=0.5,          # มั่นใจ >= 50% ถึงตีกรอบ
        imgsz=640,
        save=True,         # เซฟภาพผลลัพธ์
        save_txt=True,     # เซฟ label ที่ predict ได้
        stream=True,       # สตรีมทีละภาพ ประหยัดแรม
        project=str(BASE / "runs" / "detect"),
        name="predict_demo",
    ):
        n_img += 1
        n_box += len(r.boxes)

    print(f"✅ เสร็จ: {n_img} ภาพ, เจอ {n_box} กล่อง → runs/detect/predict_demo/")
