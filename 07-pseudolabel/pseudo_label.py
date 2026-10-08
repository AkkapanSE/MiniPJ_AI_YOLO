"""Pseudo-label ภาพดิบ raw_images/brand1 + street1 ด้วย best.pt baseline.

ขั้นตอน:
  1. อ่านรายชื่อภาพต้นฉบับที่ใช้ใน mixed/ แล้ว (train/valid/test) จากชื่อไฟล์
     `xxx_jpg.rf.<hash>.jpg` -> `xxx.jpg` เพื่อกัน leakage / งานซ้ำ
  2. predict ภาพดิบ conf=0.5 -> เซฟ label YOLO (5 คอลัมน์) + copy ภาพเข้า pseudo/
     - ข้ามภาพที่อยู่ใน mixed train (มี human label แล้ว) และ valid/test (กัน leakage)
     - ภาพที่ไม่มี detection เลย -> ลงรายการ pseudo_empty.txt (ส่วนใหญ่คือภาพเปล่า)
  3. สรุปจำนวนต่อโฟลเดอร์/ต่อคลาส -> pseudo_report.txt

รัน: .\.env\Scripts\python.exe 03-pseudolabel\pseudo_label.py
"""
from pathlib import Path
from shutil import copy2
from collections import Counter
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
RAW_DIRS = [BASE / "raw_images" / "brand1", BASE / "raw_images" / "street1"]
MIXED = BASE / "mixed"
OUT_IMG = BASE / "pseudo" / "images"
OUT_LBL = BASE / "pseudo" / "labels"
WEIGHTS = BASE / "all_model" / "best_round3.pt"
CONF = 0.5
IMGSZ = 640


def original_basenames(split: str) -> set:
    """brand_0000_jpg.rf.<hash>.jpg -> brand_0000.jpg"""
    names = set()
    for p in (MIXED / split / "images").glob("*.jpg"):
        stem = p.name.split("_jpg.rf.")[0]  # brand_0000
        names.add(stem + ".jpg")
    return names


def main():
    assert WEIGHTS.exists(), f"ไม่พบ weights: {WEIGHTS} — รัน 02-train ก่อน"
    train_used = original_basenames("train")
    holdout = original_basenames("valid") | original_basenames("test")
    print(f"mixed ใช้แล้ว: train {len(train_used)} / valid+test {len(holdout)}")

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))
    kept, skipped_train, skipped_holdout, empty = 0, 0, 0, []
    cls_counter = Counter()

    for raw_dir in RAW_DIRS:
        imgs = sorted(raw_dir.glob("*.jpg"))
        print(f"predict {raw_dir.name}: {len(imgs)} ภาพ ...")
        results = model.predict(source=[str(p) for p in imgs], conf=CONF,
                                imgsz=IMGSZ, verbose=False)
        for img_path, r in zip(imgs, results):
            if img_path.name in holdout:
                skipped_holdout += 1
                continue
            if img_path.name in train_used:
                skipped_train += 1
                continue
            if len(r.boxes) == 0:
                empty.append(f"{raw_dir.name}/{img_path.name}")
                continue
            # เซฟ label 5 คอลัมน์ (class x y w h) แบบ normalize
            lines = []
            for b in r.boxes:
                cls = int(b.cls[0])
                x, y, w, h = (float(v) for v in b.xywhn[0])
                lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                cls_counter[cls] += 1
            (OUT_LBL / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
            copy2(img_path, OUT_IMG / img_path.name)
            kept += 1

    (BASE / "pseudo" / "pseudo_empty.txt").write_text("\n".join(empty) + "\n")
    names = model.names
    report = [
        f"conf={CONF} imgsz={IMGSZ} weights={WEIGHTS.name}",
        f"kept(pseudo): {kept}",
        f"skip: train-used {skipped_train} / valid+test {skipped_holdout}",
        f"empty(no detection): {len(empty)}",
        "per-class pseudo boxes:",
        *[f"  {names[c]}: {n}" for c, n in sorted(cls_counter.items())],
    ]
    (BASE / "pseudo" / "pseudo_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
