"""สคริปต์ประเมินโมเดล best.pt รอบ 2 บนชุด test (ไม่เคยเห็นตอนเทรน).

รัน: .\.env\Scripts\python.exe 05-evaluate.py
ผล: ตาราง P / R / F1 / mAP50 / mAP50-95 รายคลาส + ตรวจผ่านเป้า + โชว์กราฟ
หมายเหตุ: Detection ไม่มีค่า Accuracy (ไม่มี True Negative) จึงใช้ F1 แทน
"""
from pathlib import Path
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "runs" / "detect" / "pathongko_mixed2_80e" / "weights" / "best.pt"
DATA = BASE / "mixed2" / "data_local.yaml"
TARGET_MAP50 = 0.85
IOU_THRESH = 0.5
CONF = 0.5


def present_classes() -> list:
    """class id ที่มีอยู่จริงใน test labels (เรียงน้อย→มาก)"""
    ids = set()
    for f in (BASE / "mixed2" / "test" / "labels").glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(int(line.split()[0]))
    return sorted(ids)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU ระหว่างกล่อง xyxy ชุด a (N,4) กับ b (M,4) -> (N,M)"""
    inter_x1 = np.maximum(a[:, None, 0], b[:, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[:, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[:, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b - inter + 1e-9)


def build_confusion_matrix(model, names: dict) -> np.ndarray:
    """จับคู่ pred (conf>=CONF) กับ GT แบบ greedy IoU>=0.5 ได้เมทริกซ์ 7x7 (6 คลาส + background)"""
    from PIL import Image

    nc, bg = len(names), len(names)
    cm = np.zeros((nc + 1, nc + 1), dtype=int)  # แถว=true, หลัก=pred
    img_dir = BASE / "mixed2" / "test" / "images"
    lbl_dir = BASE / "mixed2" / "test" / "labels"
    imgs = sorted(img_dir.glob("*.jpg"))
    for r in model.predict(source=[str(p) for p in imgs], conf=CONF,
                           imgsz=640, verbose=False, stream=True):
        gt = []
        lf = lbl_dir / f"{Path(r.path).stem}.txt"
        if lf.exists():
            w, h = Image.open(r.path).size
            for line in lf.read_text().splitlines():
                vals = [float(v) for v in line.split()]
                if not vals:
                    continue
                c, pts = int(vals[0]), vals[1:]
                if len(pts) == 4:
                    # bbox: x_center y_center w h (normalize)
                    x, y, bw, bh = pts
                    x1, y1 = (x - bw / 2) * w, (y - bh / 2) * h
                    x2, y2 = (x + bw / 2) * w, (y + bh / 2) * h
                else:
                    # polygon: หา min/max เป็น bbox
                    xs, ys = pts[0::2], pts[1::2]
                    x1, y1, x2, y2 = min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h
                gt.append([c, x1, y1, x2, y2])
        gt = np.array(gt, dtype=float).reshape(-1, 5)
        pb = r.boxes
        if len(pb):
            order = np.argsort(-pb.conf.cpu().numpy())
            preds = [(int(pb.cls[i]), *map(float, pb.xyxy[i])) for i in order]
        else:
            preds = []
        matched = np.zeros(len(gt), dtype=bool)
        for pc, *pxyxy in preds:
            best, bi = 0.0, -1
            for j, (gc, *gxyxy) in enumerate(gt):
                if not matched[j] and gc == pc:
                    iou = box_iou(np.array([pxyxy]), np.array([gxyxy]))[0, 0]
                    if iou > best:
                        best, bi = iou, j
            if best >= IOU_THRESH:
                matched[bi] = True
                cm[int(gt[bi][0]), pc] += 1
            else:
                cm[bg, pc] += 1  # ทายเกิน (false positive)
        for j, (gc, *_) in enumerate(gt):
            if not matched[j]:
                cm[int(gc), bg] += 1  # หลุด (false negative)
    return cm


def plot_confusion_matrices(cm: np.ndarray, labels: list, save_dir: Path):
    """heatmap 2 แบบ: จำนวนดิบ + % รายแถว (normalize)"""
    import matplotlib.pyplot as plt

    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float),
                     where=row_sum != 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.6))
    fig.suptitle("Confusion Matrix แบบ custom (IoU>=0.5, conf>=0.5)", fontsize=13, y=0.99)
    for ax, mat, fmt, title in [
        (ax1, cm, "d", "จำนวนดิบ (count)"),
        (ax2, norm, ".0%", "% รายแถว (แถว=true class)"),
    ]:
        ax.imshow(mat, cmap="Blues", aspect="equal")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(title, fontsize=11)
        thresh = mat.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = f"{mat[i, j]:{fmt}}"
                ax.text(j, i, v, ha="center", va="center", fontsize=9,
                        color="white" if mat[i, j] > thresh else "black")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(save_dir / "confusion_matrix_custom.png", dpi=150)
    print(f"confusion matrix แบบ custom: {save_dir / 'confusion_matrix_custom.png'}")
    try:
        plt.get_current_fig_manager().window.state("zoomed")
    except Exception:
        try:
            plt.get_current_fig_manager().window.showMaximized()
        except Exception:
            pass
    plt.show()


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

    # confusion matrix แบบ custom: heatmap จำนวนดิบ + % (เทียบกับแบบ ultralytics)
    print("\n🧮 สร้าง confusion matrix แบบ custom ...")
    cm = build_confusion_matrix(model, m.names)
    labels = [m.names[i] for i in range(len(m.names))] + ["background"]
    print("แถว=true / หลัก=pred (ช่อง background = ทายเกิน/หลุด):")
    print(f"{'':<24}{' '.join(f'{l[:6]:>7}' for l in labels)}")
    for i, l in enumerate(labels):
        print(f"{l:<24}{' '.join(f'{cm[i, j]:>7d}' for j in range(len(labels)))}")
    plot_confusion_matrices(cm, labels, m.save_dir)

    # โชว์กราฟที่ ultralytics เซฟไว้ (confusion matrix + curves)
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    pngs = ["confusion_matrix.png", "F1_curve.png", "P_curve.png",
            "R_curve.png", "PR_curve.png", "results.png"]
    paths = [m.save_dir / p for p in pngs if (m.save_dir / p).exists()]
    print(f"กราฟเซฟที่: {m.save_dir}")
    # figsize 12.8x7.2 (= 1280x720 px) พอดีจอโน้ตบุ๊ก ไม่ล้น (เซฟไฟล์แยกที่ dpi สูงกว่า)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2))
    fig.suptitle(f"Evaluate {WEIGHTS.parent.parent.name} (test mAP50={b.map50:.3f})",
                 fontsize=13, y=0.98)
    for ax, p in zip(axes.flat, paths):
        ax.imshow(imread(p), aspect="equal")
        ax.set_title(p.name, fontsize=10, pad=4)
        ax.axis("off")
    for ax in axes.flat[len(paths):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])  # เว้นที่ให้ suptitle ขอบไม่ถูกตัด
    fig.savefig(m.save_dir / "evaluate_summary.png", dpi=150)
    print(f"รวมกราฟเซฟที่: {m.save_dir / 'evaluate_summary.png'}")
    # ขยายหน้าต่างกราฟเต็มจอ (รองรับทั้ง Tk / Qt)
    try:
        plt.get_current_fig_manager().window.state("zoomed")  # TkAgg (default Windows)
    except Exception:
        try:
            plt.get_current_fig_manager().window.showMaximized()  # Qt
        except Exception:
            pass
    plt.show()
