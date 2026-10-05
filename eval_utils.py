"""Helper สำหรับ 05-evaluate.py: confusion matrix + โชว์กราฟทีละภาพกลางจอ."""
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
IOU_THRESH = 0.5
CONF = 0.5


def present_classes(lbl_dir: Path) -> list:
    """class id ที่มีอยู่จริงใน labels (เรียงน้อย→มาก)"""
    ids = set()
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(int(line.split()[0]))
    return sorted(ids)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU ระหว่างกล่อง xyxy ชุด a (N,4) กับ b (M,4) -> (N,M)"""
    inter = (np.clip(np.minimum(a[:, None, 2], b[:, 2]) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
             * np.clip(np.minimum(a[:, None, 3], b[:, 3]) - np.maximum(a[:, None, 1], b[:, 1]), 0, None))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b - inter + 1e-9)


def _parse_gt(line: str, w: float, h: float):
    """รองรับทั้ง label bbox (5 ค่า) และ polygon (min/max เป็น bbox)"""
    vals = [float(v) for v in line.split()]
    if not vals:
        return None
    c, pts = int(vals[0]), vals[1:]
    if len(pts) == 4:
        x, y, bw, bh = pts
        return [c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
    xs, ys = pts[0::2], pts[1::2]
    return [c, min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h]


def build_confusion_matrix(model, names: dict, img_dir: Path, lbl_dir: Path) -> np.ndarray:
    """จับคู่ pred กับ GT แบบ greedy IoU ได้เมทริกซ์ (nc+1)x(nc+1) รวม background"""
    from PIL import Image

    nc = len(names)
    cm = np.zeros((nc + 1, nc + 1), dtype=int)  # แถว=true, หลัก=pred
    imgs = sorted(img_dir.glob("*.jpg"))
    for r in model.predict(source=[str(p) for p in imgs], conf=CONF,
                           imgsz=640, verbose=False, stream=True):
        gt = []
        lf = lbl_dir / f"{Path(r.path).stem}.txt"
        if lf.exists():
            w, h = Image.open(r.path).size
            for line in lf.read_text().splitlines():
                if (g := _parse_gt(line, w, h)) is not None:
                    gt.append(g)
        gt = np.array(gt, dtype=float).reshape(-1, 5)
        pb = r.boxes
        preds = [(int(pb.cls[i]), *map(float, pb.xyxy[i]))
                 for i in np.argsort(-pb.conf.cpu().numpy())] if len(pb) else []
        matched = np.zeros(len(gt), dtype=bool)
        for pc, *pxy in preds:
            cand = [(box_iou(np.array([pxy]), np.array([g[1:]]))[0, 0], j)
                    for j, g in enumerate(gt) if not matched[j] and int(g[0]) == pc]
            if cand and max(cand)[0] >= IOU_THRESH:
                bi = max(cand)[1]
                matched[bi] = True
                cm[int(gt[bi][0]), pc] += 1
            else:
                cm[nc, pc] += 1  # ทายเกิน
        for j, g in enumerate(gt):
            if not matched[j]:
                cm[int(g[0]), nc] += 1  # หลุด
    return cm


def center_window():
    """จัดหน้าต่างกราฟไว้กลางจอ (รองรับทั้ง Tk / Qt)"""
    import matplotlib.pyplot as plt

    try:
        win = plt.get_current_fig_manager().window
        win.state("zoomed")
        win.eval("tk::PlaceWindow . center")
    except Exception:
        try:
            plt.get_current_fig_manager().window.showMaximized()
        except Exception:
            pass


def show_one_by_one(cm: np.ndarray, labels: list, curve_pngs: list,
                    save_dir: Path, title_base: str):
    """โชว์ทีละกราฟกลางจอ: heatmap custom 2 ภาพ + curves ทีละภาพ"""
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    def figure(title):
        fig, ax = plt.subplots(figsize=(9.6, 7.2))
        fig.suptitle(f"{title_base}\n{title}", fontsize=12, y=0.98)
        fig.tight_layout(rect=[0, 0.02, 1, 0.90])
        return fig, ax

    def heatmap(mat, fmt, title, fname):
        fig, ax = figure(title)
        ax.imshow(mat, cmap="Blues", aspect="equal")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        thresh = mat.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=10, color="white" if mat[i, j] > thresh else "black")
        fig.savefig(save_dir / fname, dpi=150)
        center_window()
        plt.show()

    row_sum = cm.sum(axis=1, keepdims=True)
    heatmap(cm, "d", "Confusion Matrix — จำนวนดิบ", "confusion_matrix_custom.png")
    heatmap(np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0),
            ".0%", "Confusion Matrix — % รายแถว", "confusion_matrix_norm.png")

    for p in curve_pngs:
        fig, ax = figure(Path(p).name)
        ax.imshow(imread(p), aspect="equal")
        ax.axis("off")
        center_window()
        plt.show()
    print(f"เซฟ heatmap custom ใน {save_dir}")
