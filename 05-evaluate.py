"""สคริปต์ประเมินโมเดล best.pt รอบ 3 บนชุด test (ไม่เคยเห็นตอนเทรน).

รัน: .\\.env\\Scripts\\python.exe 05-evaluate.py
ผล: ตาราง P / R / F1 / mAP50 / mAP50-95 รายคลาส + ตรวจผ่านเป้า + โชว์กราฟทีละภาพ
หมายเหตุ: Detection ไม่มีค่า Accuracy (ไม่มี True Negative) จึงใช้ F1 แทน
"""
from pathlib import Path
from ultralytics import YOLO
import eval_utils as U

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "runs" / "detect" / "pathongko_mixed3_fixed" / "weights" / "best.pt"
DATA = BASE / "mixed2" / "data_local.yaml"
TARGET_MAP50 = 0.85

if __name__ == "__main__":
    assert WEIGHTS.exists(), f"ไม่พบ weights: {WEIGHTS} — รัน 02-train/02-train.py ก่อน"
    assert DATA.exists(), f"ไม่พบ dataset: {DATA}"

    print(f"📊 ประเมิน {WEIGHTS.name} บนชุด test ...")
    model = YOLO(str(WEIGHTS))
    m = model.val(data=str(DATA), split="test", imgsz=640)  # plots=True (default)
    b = m.box
    present = U.present_classes(BASE / "mixed2" / "test" / "labels")

    print(f"\n{'class':<24}{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>8}{'mAP50-95':>10}")
    f1s = []
    for c, name in m.names.items():
        if c in present:
            i = present.index(c)  # อาร์เรย์ p/r/f1/ap50 มีเฉพาะคลาสที่อยู่ใน test
            f1s.append(float(b.f1[i]))
            print(f"{name:<24}{b.p[i]:>7.3f}{b.r[i]:>7.3f}"
                  f"{b.f1[i]:>7.3f}{b.ap50[i]:>8.3f}{b.maps[c]:>10.3f}")
        else:
            print(f"{name:<24}{'N/A':>7}{'N/A':>7}{'N/A':>7}{'N/A':>8}{'N/A':>10}  (ไม่มีใน test)")
    mean_f1 = sum(f1s) / len(f1s)
    print(f"{'mean':<24}{b.mp:>7.3f}{b.mr:>7.3f}{mean_f1:>7.3f}"
          f"{b.map50:>8.3f}{b.map:>10.3f}")

    ok = b.map50 >= TARGET_MAP50
    print(f"\n{'✅ ผ่านเป้า' if ok else '❌ ต่ำกว่าเป้า'} "
          f"mAP50 = {b.map50:.3f} (เป้า {TARGET_MAP50}) | F1 เฉลี่ย = {mean_f1:.3f}")

    print("\n🧮 สร้าง confusion matrix แบบ custom ...")
    cm = U.build_confusion_matrix(model, m.names,
                                  BASE / "mixed2" / "test" / "images",
                                  BASE / "mixed2" / "test" / "labels")
    labels = [m.names[i] for i in range(len(m.names))] + ["background"]
    print("แถว=true / หลัก=pred (ช่อง background = ทายเกิน/หลุด):")
    print(f"{'':<24}{' '.join(f'{l[:6]:>7}' for l in labels)}")
    for i, l in enumerate(labels):
        print(f"{l:<24}{' '.join(f'{cm[i, j]:>7d}' for j in range(len(labels)))}")

    curves = ["F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png",
              "results.png", "confusion_matrix.png"]
    U.show_one_by_one(cm, labels,
                      [str(m.save_dir / p) for p in curves if (m.save_dir / p).exists()],
                      m.save_dir,
                      f"Evaluate {WEIGHTS.parent.parent.name} "
                      f"(test mAP50={b.map50:.3f}, F1={mean_f1:.3f})")
