"""สคริปต์ประเมินโมเดล best.pt รอบ 2 บนชุด test (ไม่เคยเห็นตอนเทรน).

รัน: .\.env\Scripts\python.exe 05-evaluate.py
ผล: ตาราง P/R/mAP50/mAP50-95 รายคลาส + ตรวจผ่านเป้า mAP50 > 0.85
"""
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "runs" / "detect" / "pathongko_mixed2_80e" / "weights" / "best.pt"
DATA = BASE / "mixed2" / "data_local.yaml"
TARGET_MAP50 = 0.85

if __name__ == "__main__":
    assert WEIGHTS.exists(), f"ไม่พบ weights: {WEIGHTS} — รัน 02-train/02-train.py ก่อน"
    assert DATA.exists(), f"ไม่พบ dataset: {DATA}"

    print(f"📊 ประเมิน {WEIGHTS.name} บนชุด test ...")
    model = YOLO(str(WEIGHTS))
    m = model.val(data=str(DATA), split="test", imgsz=640)

    ok = m.box.map50 >= TARGET_MAP50
    print(f"\n{'✅ ผ่านเป้า' if ok else '❌ ต่ำกว่าเป้า'} "
          f"mAP50 = {m.box.map50:.3f} (เป้า {TARGET_MAP50}) | "
          f"P = {m.box.mp:.3f} R = {m.box.mr:.3f} mAP50-95 = {m.box.map:.3f}")
