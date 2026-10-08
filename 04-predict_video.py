"""สคริปต์ทดสอบโมเดลกับวิดีโอ (inference) ด้วย best.pt รอบ 9 (mixed6 DTPG v6, test mAP50 0.930).

วิธีใช้:
    .\.env\Scripts\python.exe 04-predict_video.py                          # ใช้คลิป 813113849.017632.mp4
    .\.env\Scripts\python.exe 04-predict_video.py <path-วิดีโอ/0>          # วิดีโออื่นหรือ webcam (0)

ผลลัพธ์: วิดีโอตีกรอบใน runs/detect/predict_video_demo/
"""
import sys
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "all_model" / "best_round9_v6.pt"

if __name__ == "__main__":
    assert WEIGHTS.exists(), f"ไม่พบ weights: {WEIGHTS} — รัน 02-train/02-train-mixed6.py ก่อน"

    source = sys.argv[1] if len(sys.argv) > 1 else str(BASE / "raw_images" / "vdo" / "813113849.017632.mp4")
    print(f"🎬 ทดสอบวิดีโอ: {source}")

    model = YOLO(str(WEIGHTS))
    # stream=True: ประมวลผลทีละเฟรม ไม่เก็บผลทั้งคลิปไว้ในแรม (กันแรมเต็ม)
    n_frames, n_box = 0, 0
    for r in model.predict(
        source=source,
        conf=0.8,          # มั่นใจ >= 80% ถึงตีกรอบ
        imgsz=640,
        save=True,         # เซฟวิดีโอผลลัพธ์
        stream=True,       # สตรีมทีละเฟรม ประหยัดแรม
        project=str(BASE / "runs" / "detect"),
        name="predict_video_demo",
    ):
        n_frames += 1
        n_box += len(r.boxes)

    print(f"✅ เสร็จ: ประมวลผล {n_frames} เฟรม เจอ {n_box} กล่อง → runs/detect/predict_video_demo/")
