"""สคริปต์ประเมินโมเดล best.pt รอบ 2 บนชุด test (ไม่เคยเห็นตอนเทรน).

รัน: .\.env\Scripts\python.exe 05-evaluate.py
ผล: ตาราง P / R / F1 / mAP50 / mAP50-95 รายคลาส + ตรวจผ่านเป้า + โชว์กราฟ
หมายเหตุ: Detection ไม่มีค่า Accuracy (ไม่มี True Negative) จึงใช้ F1 แทน
"""
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "runs" / "detect" / "pathongko_mixed2_80e" / "weights" / "best.pt"
DATA = BASE / "mixed2" / "data_local.yaml"
TARGET_MAP50 = 0.85


def present_classes() -> list:
    """class id ที่มีอยู่จริงใน test labels (เรียงน้อย→มาก)"""
    ids = set()
    for f in (BASE / "mixed2" / "test" / "labels").glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(int(line.split()[0]))
    return sorted(ids)


if __name__ == "__main__":
    assert WEIGHTS.exists(), f"ไม่พบ weights: {WEIGHTS} — รัน 02-train/02-train.py ก่อน"
    assert DATA.exists(), f"ไม่พบ dataset: {DATA}"

    print(f"📊 ประเมิน {WEIGHTS.name} บนชุด test ...")
    model = YOLO(str(WEIGHTS))
    m = model.val(data=str(DATA), split="test", imgsz=640)  # plots=True (default)
    b, present = m.box, present_classes()

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

    # โชว์กราฟที่ ultralytics เซฟไว้ (confusion matrix + curves)
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    pngs = ["confusion_matrix.png", "F1_curve.png", "P_curve.png",
            "R_curve.png", "PR_curve.png", "results.png"]
    paths = [m.save_dir / p for p in pngs if (m.save_dir / p).exists()]
    print(f"กราฟเซฟที่: {m.save_dir}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Evaluate {WEIGHTS.parent.parent.name} (test mAP50={b.map50:.3f})")
    for ax, p in zip(axes.flat, paths):
        ax.imshow(imread(p))
        ax.set_title(p.name)
        ax.axis("off")
    for ax in axes.flat[len(paths):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
