"""สคริปต์ทดสอบโมเดล (inference) ด้วย best.pt รอบ 8 (mixed5 DTPG v5, test mAP50 0.935).

วิธีใช้:
    python 03-predict.py [source] [conf] [imgsz] [--w R] [--tta] [--iou X]
      source : path รูป/โฟลเดอร์/วิดีโอ/0 (default: mixed5/test/images)
      conf   : threshold (default 0.5 เน้นเรียกของครบ, 0.8 ตามกติกาครูถ้าต้องการ)
      imgsz  : ขนาดภาพ (default 640, ลอง 960 กับของชิ้นใหญ่/close-up)
      --w R  : เลือก weights รอบ R: 3/6/7/8 (default 8)
      --tta  : เปิด test-time augmentation (flip+multiscale ช้าหน่อยแต่เรียกของกลับ)
      --iou X: NMS IoU (default 0.7, ลดเหลือ 0.5 เพื่อยุบกล่องซ้อน)

ผลลัพธ์: runs/detect/predict_demo/ (ภาพตีกรอบ + labels)
"""
import sys
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

# โมเดลรอบ 8: fine-tune จาก best_round3 บน mixed5 80 epochs (train 1714 / test mAP50 0.935)
# ถ้ายังไม่เคยเทรน ให้รัน 02-train/02-train-mixed5.py ก่อน

ROUND_W = {"3": "best_round3.pt", "6": "best_round6.pt", "7": "best_round7.pt", "8": "best_round8_mixed5.pt"}


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    source = pos[0] if len(pos) > 0 else str(BASE / "mixed5" / "test" / "images")
    conf = float(pos[1]) if len(pos) > 1 else 0.5   # default 0.5 เน้นเรียกของครบ 6 คลาส
    imgsz = int(pos[2]) if len(pos) > 2 else 640
    weights = BASE / "all_model" / ROUND_W.get(_arg("--w", "8"), ROUND_W["8"])
    assert weights.exists(), f"ไม่พบ weights: {weights} — รัน 02-train/02-train.py ก่อน"

    tta = "--tta" in sys.argv
    iou = float(_arg("--iou", 0.7))
    print(f"🔍 {weights.name} | {source} | conf={conf} imgsz={imgsz} tta={tta} iou={iou}")

    model = YOLO(str(weights))
    # stream=True: ประมวลผลทีละภาพ ไม่เก็บผลทั้งหมดไว้ในแรม (กันแรมเต็ม)
    n_img, n_box = 0, 0
    for r in model.predict(
        source=source,
        conf=conf,
        imgsz=imgsz,
        augment=tta,     # TTA: flip + multiscale ช่วยของนอกโดเมน/สเกลเพี้ยน
        iou=iou,         # NMS: ต่ำลง = ยุบกล่องซ้อนกันมากขึ้น
        save=True,       # เซฟภาพผลลัพธ์
        save_txt=True,   # เซฟ label ที่ predict ได้
        stream=True,     # สตรีมทีละภาพ ประหยัดแรม
        project=str(BASE / "runs" / "detect"),
        name="predict_demo",
    ):
        n_img += 1
        n_box += len(r.boxes)

    print(f"✅ เสร็จ: {n_img} ภาพ, เจอ {n_box} กล่อง → runs/detect/predict_demo/")
