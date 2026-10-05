"""Oversample คลาสน้อยใน mixed2/train แล้วสร้าง mixed2_os/ (valid/test ชี้ของเดิม กัน leakage).

สัดส่วนกล่องเดิม train — 0:51 / 1:147 / 2:191 / 3:47 / 4:298 / 5:267
ตัวคูณ: ภาพที่มีคลาส 3 → x5, คลาส 0 → x4, คลาส 2 → x2 (ก็อปไฟล์เติม suffix _osN,
augmentation ตอนเทรนทำให้แต่ละสำเนาเห็นต่างกัน)

รัน: .\.env\Scripts\python.exe 02-train\oversample.py
"""
from pathlib import Path
from shutil import copy2
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
SRC_IMG = BASE / "mixed2" / "train" / "images"
SRC_LBL = BASE / "mixed2" / "train" / "labels"
DST = BASE / "mixed2_os"
# (คลาสเป้าหมาย, จำนวนสำเนารวม)
MULT = {3: 5, 0: 4, 2: 2}

NAMES = ['brand_golden_defect', 'brand_golden_good', 'street_burnt_defect',
         'street_burnt_good', 'street_golden_defect', 'street_golden_good']


def main():
    (DST / "train" / "images").mkdir(parents=True, exist_ok=True)
    (DST / "train" / "labels").mkdir(parents=True, exist_ok=True)
    counter, n_img = Counter(), 0
    for img in sorted(SRC_IMG.glob("*.jpg")):
        lines = (SRC_LBL / f"{img.stem}.txt").read_text().splitlines()
        cls = {int(l.split()[0]) for l in lines if l.strip()}
        copies = max([MULT.get(c, 1) for c in cls] or [1])
        for k in range(copies):
            tag = "" if k == 0 else f"_os{k}"
            copy2(img, DST / "train" / "images" / f"{img.stem}{tag}.jpg")
            (DST / "train" / "labels" / f"{img.stem}{tag}.txt").write_text(
                "\n".join(lines) + "\n")
            n_img += 1
        for l in lines:
            if l.strip():
                counter[int(l.split()[0])] += copies
    (DST / "data_local.yaml").write_text(
        f"train: {(DST / 'train' / 'images').as_posix()}\n"
        f"val: {(BASE / 'mixed2' / 'valid' / 'images').as_posix()}\n"
        f"test: {(BASE / 'mixed2' / 'test' / 'images').as_posix()}\n"
        f"nc: 6\nnames: {NAMES}\n")
    print(f"train {len(list(SRC_IMG.glob('*.jpg')))} to {n_img} images")
    print("boxes after oversample:", {NAMES[c]: counter[c] for c in sorted(counter)})


if __name__ == "__main__":
    main()
