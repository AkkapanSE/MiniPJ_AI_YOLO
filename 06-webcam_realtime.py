"""สคริปต์ทดสอบโมเดล realtime ด้วย webcam ด้วย best.pt รอบ 2 (mixed2).

รัน: .\.env\Scripts\python.exe 06-webcam_realtime.py [เลขกล้อง]   (default 0)
กด q เพื่อออกจากหน้าต่าง
"""
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "runs" / "detect" / "pathongko_mixed2_80e" / "weights" / "best.pt"
CONF = 0.5
IMGSZ = 640

if __name__ == "__main__":
    assert WEIGHTS.exists(), f"ไม่พบ weights: {WEIGHTS} — รัน 02-train/02-train.py ก่อน"

    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model = YOLO(str(WEIGHTS))
    cap = cv2.VideoCapture(cam)
    assert cap.isOpened(), f"เปิดกล้อง {cam} ไม่ได้ — เช็คกล้องหรือลองเลขอื่น (1, 2, ...)"

    print(f"📷 realtime ด้วยกล้อง {cam} (conf>={CONF}) — กด q เพื่อออก")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ อ่านเฟรมไม่ได้ — จบ")
            break
        annotated = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0].plot()
        cv2.imshow("Patongko Realtime (q=ออก)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("👋 ปิดกล้องแล้ว")
