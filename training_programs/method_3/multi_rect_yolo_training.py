#!/usr/bin/env python3
# verify_windows.py
# ------------------------------------------------------------------
#  Edit ROOT_PATH below → run the script.
#  It will look recursively for
#      .../images/**/<file>.tiff (png, jpg, …)
#      .../labels/**/<file>.txt
#  and show 10 random crops with their rectangles.
# ------------------------------------------------------------------
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time 

# edit this one line
ROOT_PATH = Path("data/YOLO_training_set/dataset_large_images_randomwindows")

# ---------- helper to convert YOLO → pixel rect -------------------
def yolo_to_rect(line: str, w: int, h: int):
    _cls, xc, yc, bw, bh = map(float, line.split())
    x0 = (xc - bw / 2) * w
    y0 = (yc - bh / 2) * h
    return x0, y0, bw * w, bh * h

# ---------- find all image/label pairs ----------------------------
image_exts = (".tiff", ".tif", ".png", ".jpg", ".jpeg")
pairs = []

for img_path in ROOT_PATH.rglob("*"):
    if img_path.suffix.lower() not in image_exts:
        continue
    # swap "/images/" with "/labels/" and change suffix to .txt
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
    except ValueError:
        continue                     # skip files outside the images tree
    parts[idx] = "labels"
    lbl_path = Path(*parts).with_suffix(".txt")
    if lbl_path.exists():
        pairs.append((img_path, lbl_path))

if not pairs:
    print(f"No image/label pairs found under {ROOT_PATH}")
    exit(0)

# pick up to 10 samples
random.shuffle(pairs)
samples = pairs[:10]

# ---------- plot ---------------------------------------------------
cols = 5
rows = (len(samples) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
if rows == 1:
    axes = np.expand_dims(axes, 0)  # make 2-D for indexing

for ax, (img_path, lbl_path) in zip(axes.flatten(), samples):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        ax.axis("off")
        continue
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]
    ax.imshow(img_rgb)

    for ln in lbl_path.read_text().splitlines():
        if not ln.strip():
            continue
        x0, y0, bw, bh = yolo_to_rect(ln, w, h)
        ax.add_patch(
            patches.Rectangle(
                (x0, y0), bw, bh,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
        )

    ax.set_title(img_path.stem, fontsize=8)
    ax.axis("off")

# turn off any unused axes
for ax in axes.flatten()[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# Grayscale YOLOv8 training + 10-image val preview
import shutil, random, torch, cv2, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from pathlib import Path

# ---------- paths ---------------------------------------------------
DATA_ROOT = ROOT_PATH                     # from viewer section
IMG_DIR, LBL_DIR = DATA_ROOT/'images', DATA_ROOT/'labels'
TRAIN_I, VAL_I   = IMG_DIR/'train', IMG_DIR/'val'
TRAIN_L, VAL_L   = LBL_DIR/'train', LBL_DIR/'val'
for d in (TRAIN_I, VAL_I, TRAIN_L, VAL_L):
    d.mkdir(parents=True, exist_ok=True)

# ---------- robust split: refresh every run -------------------------
def sync_split(img_pairs, train_i, val_i, train_l, val_l, ratio=0.8):
    """Ensure train/ and val/ contain the desired split every run."""
    # 1) clear existing sub-dirs
    for d in (train_i, val_i, train_l, val_l):
        for p in d.glob("*"):
            p.unlink()
    # 2) compute fresh split
    random.shuffle(img_pairs)
    cut = int(ratio * len(img_pairs))
    for i, (img_p, lbl_p) in enumerate(img_pairs):
        tgt_img = train_i if i < cut else val_i
        tgt_lbl = train_l if i < cut else val_l
        # copy only if source and dest are different paths
        if img_p.resolve() != (tgt_img / img_p.name).resolve():
            shutil.copy2(img_p, tgt_img / img_p.name)
            shutil.copy2(lbl_p, tgt_lbl / lbl_p.name)
    print(f"[split] {cut} train, {len(img_pairs)-cut} val images ready")

sync_split(pairs, TRAIN_I, VAL_I, TRAIN_L, VAL_L)


# ---------- dataset.yaml (channels:1) -------------------------------
yaml_path = DATA_ROOT / 'dataset.yaml'
yaml_path.write_text(f"""\
path: {DATA_ROOT}
train: images/train
val: images/val
channels: 1
names:
  0: dislocation
""")

# ---------- build & save 1-channel backbone -------------------------
ckpt = DATA_ROOT / 'yolov8n_gray.pt'
if not ckpt.exists():                     # build only once
    base = YOLO('yolov8n.pt')
    conv0 = base.model.model[0]
    w = conv0.conv.weight.data
    gray = torch.nn.Conv2d(1, w.shape[0],
                           conv0.conv.kernel_size,
                           conv0.conv.stride,
                           conv0.conv.padding,
                           bias=False)
    with torch.no_grad():
        gray.weight[:] = w.mean(1, keepdim=True)
    conv0.conv = gray
    base.save(ckpt)
    print(f"[YOLO] 1-channel weights saved → {ckpt.name}")

# ---------- train ---------------------------------------------------
model = YOLO(str(ckpt))
print("\n[YOLO] training …\n")
# build a unique folder name: e.g. "fullimg_20250603_142530"
run_name = f"fullimg_{time.strftime('%Y%m%d_%H%M%S')}"

model.train(
    data=str(yaml_path),
    epochs=10000,
    imgsz=640,
    batch=8,
    rect=True,
    workers=8,
    project=str(DATA_ROOT / "runs"),   # <- all runs go here
    name=run_name,                     # <- this run's sub-folder
    exist_ok=False                     # error if the name already exists
)

best_path = model.trainer.best           # ↩︎ safe handles
last_path = model.trainer.last
print(f"[YOLO] best checkpoint: {best_path}")

best = YOLO(best_path)                   # load for preview

# ---------- preview grid --------------------------------------------
val_pairs = []
for img_path in sorted(VAL_I.iterdir()):
    lbl_path = VAL_L / img_path.with_suffix('.txt').name
    if lbl_path.exists():
        val_pairs.append((img_path, lbl_path))

samples = random.sample(val_pairs, k=min(10, len(val_pairs)))

cols, rows = 5, (len(samples) + 4) // 5
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
if rows == 1:
    axes = np.expand_dims(axes, 0)

def y2rect(line, w, h):
    _, xc, yc, bw, bh = map(float, line.split())
    return ( (xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h )

for ax, (img_p, lbl_p) in zip(axes.flatten(), samples):
    img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        ax.axis("off")
        continue
    h, w = img.shape
    ax.imshow(img, cmap="gray")

    # -------- ground-truth (green) ----------
    for ln in Path(lbl_p).read_text().splitlines():
        if ln.strip():
            x0, y0, bw, bh = y2rect(ln, w, h)
            ax.add_patch(
                patches.Rectangle((x0, y0), bw, bh,
                                  edgecolor="lime", facecolor="none", lw=2)
            )

    # -------- predictions (red) -------------
    res = best.predict(img, verbose=False, conf=0.01)[0]   # ← 0.01 !
    if res.boxes is not None and len(res.boxes):
        for xc, yc, bw_n, bh_n in res.boxes.xywhn.cpu().numpy():
            x0, y0 = (xc - bw_n / 2) * w, (yc - bh_n / 2) * h
            ax.add_patch(
                patches.Rectangle((x0, y0), bw_n * w, bh_n * h,
                                  edgecolor="red", facecolor="none", lw=1)
            )

    ax.set_title(img_p.stem, fontsize=7)
    ax.axis("off")

for ax in axes.flatten()[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
