#!/usr/bin/env python3
# yolo_window_exporter.py - Export all annotation types (detect, pose, segment, obb)
# -------------------------------------------------------------------
#  – Loads ALL annotation types for each image
#  – Displays all annotations like the visualizer
#  – Exports windows with all applicable annotation types preserved
#  – Maintains the same folder structure for all label types
# -------------------------------------------------------------------
import sys, random, cv2
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from shapely.geometry import box, Point
from shapely.ops import unary_union
from PySide6.QtCore import Qt, QRectF, QTimer, QPointF
from PySide6.QtCore import QObject, QEvent
from PySide6.QtGui import QColor, QPen, QPixmap, QAction, QKeySequence, QPainter, QBrush, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGraphicsScene, QGraphicsView,
    QLabel, QMainWindow, QMessageBox, QToolBar, QDialog,
    QSlider, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSizePolicy,
    QCheckBox
)

try:
    from poisson_disc import Bridson_sampling
    HAVE_POISSON = True
except ImportError:
    HAVE_POISSON = False
    print("Warning: poisson_disc not available. Install with: pip install poisson-disc")

# Adjustable parameters
TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN = 0, 20, 0, 0
MULTIPLIER   = 2.0
MIN_INSIDE   = 0.75
DATA_ROOT = Path("data/YOLO_training_set/comprehensive_dataset")
EXPORT_DIR   = DATA_ROOT / "exported_windows"
RANDOM_VAR = 0.2  # ±20% variability in random windows size

# Class mapping
CLASS_NAMES = ['edge', 'screw', 'mixed', 'unknown']
CLASS_COLORS = {
    'edge': (255, 50, 50),     # Red (BGR for OpenCV)
    'screw': (50, 255, 50),    # Green
    'mixed': (50, 50, 255),    # Blue
    'unknown': (255, 255, 50)  # Yellow
}

@dataclass
class AllLabels:
    """Container for all label types"""
    detect: List[List[float]]
    pose: List[List[float]]
    segment: List[List[float]]
    obb: List[List[float]]

def load_all_labels(img_path: Path, img_w: int, img_h: int) -> AllLabels:
    """Load all annotation types for an image"""
    stem = img_path.stem
    
    def load_file(label_type: str):
        label_path = DATA_ROOT / 'labels' / label_type / 'all' / f'{stem}.txt'
        if not label_path.exists():
            return []
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    labels.append([float(x) for x in parts])
        return labels
    
    return AllLabels(
        detect=load_file('detect'),
        pose=load_file('pose'),
        segment=load_file('segment'),
        obb=load_file('obb')
    )

def get_bbox_from_label(label: List[float], label_type: str, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Extract bounding box from any label type"""
    if label_type in ['detect', 'pose']:
        # Format: class xc yc w h ...
        xc, yc, w, h = label[1:5]
        xmin = (xc - w/2) * img_w
        ymin = (yc - h/2) * img_h
        xmax = (xc + w/2) * img_w
        ymax = (yc + h/2) * img_h
        return xmin, ymin, xmax, ymax
    
    elif label_type == 'segment':
        # Format: class x1 y1 x2 y2 x3 y3 ...
        points = label[1:]
        xs = [points[i] * img_w for i in range(0, len(points), 2)]
        ys = [points[i+1] * img_h for i in range(0, len(points)-1, 2)]
        return min(xs), min(ys), max(xs), max(ys)
    
    elif label_type == 'obb':
        # Format: class x1 y1 x2 y2 x3 y3 x4 y4
        points = label[1:9]
        xs = [points[i] * img_w for i in range(0, 8, 2)]
        ys = [points[i+1] * img_h for i in range(0, 8, 2)]
        return min(xs), min(ys), max(xs), max(ys)
    
    return 0, 0, 0, 0

def draw_all_annotations(scene: QGraphicsScene, all_labels: AllLabels, img_w: int, img_h: int):
    """Draw all annotation types on the scene"""
    
    # Draw detect boxes
    for label in all_labels.detect:
        cls = int(label[0])
        xmin, ymin, xmax, ymax = get_bbox_from_label(label, 'detect', img_w, img_h)
        color = QColor(*CLASS_COLORS[CLASS_NAMES[cls]][::-1])  # BGR to RGB
        scene.addRect(QRectF(xmin, ymin, xmax-xmin, ymax-ymin), QPen(color, 1))
    
    # Draw pose keypoints
    for label in all_labels.pose:
        cls = int(label[0])
        xmin, ymin, xmax, ymax = get_bbox_from_label(label, 'pose', img_w, img_h)
        color = QColor(*CLASS_COLORS[CLASS_NAMES[cls]][::-1])
        scene.addRect(QRectF(xmin, ymin, xmax-xmin, ymax-ymin), QPen(color, 1))
        
        # Draw keypoints
        keypoints = label[5:]
        for i in range(0, len(keypoints), 3):
            if i+2 < len(keypoints):
                kx, ky, vis = keypoints[i:i+3]
                if vis > 0:
                    px, py = kx * img_w, ky * img_h
                    kpt_color = QColor(255, 0, 255) if (i//3) % 2 == 0 else QColor(0, 255, 255)
                    scene.addEllipse(px-2, py-2, 4, 4, QPen(kpt_color, 1), QBrush(kpt_color))
    
    # Draw segment polygons
    for label in all_labels.segment:
        cls = int(label[0])
        points = label[1:]
        poly_points = []
        for i in range(0, len(points), 2):
            if i+1 < len(points):
                poly_points.append(QPointF(points[i] * img_w, points[i+1] * img_h))
        
        if len(poly_points) >= 3:
            color = QColor(*CLASS_COLORS[CLASS_NAMES[cls]][::-1])
            color.setAlpha(100)
            scene.addPolygon(QPolygonF(poly_points), QPen(color, 1), QBrush(color))
    
    # Draw OBB boxes
    for label in all_labels.obb:
        cls = int(label[0])
        points = label[1:9]
        corners = []
        for i in range(0, 8, 2):
            corners.append(QPointF(points[i] * img_w, points[i+1] * img_h))
        
        if len(corners) == 4:
            color = QColor(*CLASS_COLORS[CLASS_NAMES[cls]][::-1])
            scene.addPolygon(QPolygonF(corners), QPen(color, 1))

# Cluster/Random window generation (same as before)
def build_cluster_windows(all_labels: AllLabels, w: int, h: int):
    """Build cluster windows using detect labels"""
    usable = box(LEFT_MARGIN, TOP_MARGIN, w - RIGHT_MARGIN, h - BOTTOM_MARGIN)
    
    # Use detect labels to build clusters
    valid = []
    for label in all_labels.detect:
        xmin, ymin, xmax, ymax = get_bbox_from_label(label, 'detect', w, h)
        if xmin >= LEFT_MARGIN and ymin >= TOP_MARGIN and xmax <= w - RIGHT_MARGIN and ymax <= h - BOTTOM_MARGIN:
            valid.append((xmin, ymin, xmax, ymax))
    
    if not valid:
        return []
    
    # Calculate average size
    widths = [xmax - xmin for xmin, ymin, xmax, ymax in valid]
    heights = [ymax - ymin for xmin, ymin, xmax, ymax in valid]
    avg = (sum(widths) + sum(heights)) / (2 * len(valid))
    dmax = MULTIPLIER * avg
    
    # Create initial windows
    wins = []
    for xmin, ymin, xmax, ymax in valid:
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        w_box = xmax - xmin
        h_box = ymax - ymin
        wins.append(box(cx - 1.5 * w_box, cy - 1.5 * h_box, cx + 1.5 * w_box, cy + 1.5 * h_box))
    
    # Cluster
    clusters, seen = [], set()
    for i in range(len(valid)):
        if i in seen:
            continue
        grp, q = [i], [i]
        seen.add(i)
        
        while q:
            cur = q.pop()
            xmin1, ymin1, xmax1, ymax1 = valid[cur]
            cx1, cy1 = (xmin1 + xmax1) / 2, (ymin1 + ymax1) / 2
            
            for j, (xmin2, ymin2, xmax2, ymax2) in enumerate(valid):
                if j not in seen:
                    cx2, cy2 = (xmin2 + xmax2) / 2, (ymin2 + ymax2) / 2
                    if hypot(cx2 - cx1, cy2 - cy1) <= dmax:
                        seen.add(j)
                        q.append(j)
                        grp.append(j)
        
        clusters.append(unary_union([wins[k] for k in grp]).intersection(usable))
    
    return clusters

def build_random_windows(w: int, h: int, base_w: int, base_h: int,
                         variation: float = RANDOM_VAR, density: float = 1.0):
    """Generate random windows"""
    inner_w = max(0, w - LEFT_MARGIN - RIGHT_MARGIN)
    inner_h = max(0, h - TOP_MARGIN - BOTTOM_MARGIN)
    
    if inner_w == 0 or inner_h == 0:
        return []
    
    base_w = min(base_w, inner_w)
    base_h = min(base_h, inner_h)
    
    area_ratio = (inner_w * inner_h) / (base_w * base_h)
    expected_windows = area_ratio * density
    total = max(1, np.random.poisson(lam=expected_windows))
    
    centres = []
    if HAVE_POISSON:
        r = max(1, min(base_w, base_h))
        try:
            centres = Bridson_sampling(inner_w, inner_h, r=r, k=30)
        except Exception:
            pass
    
    if len(centres) < total:
        need = total - len(centres)
        extra = np.column_stack([
            np.random.uniform(0, inner_w, need),
            np.random.uniform(0, inner_h, need)
        ])
        centres.extend(map(tuple, extra))
    
    centres = centres[:total]
    
    windows = []
    for cx, cy in centres:
        rw = int(base_w * random.uniform(1 - variation, 1 + variation))
        rh = int(base_h * random.uniform(1 - variation, 1 + variation))
        rw = min(rw, inner_w)
        rh = min(rh, inner_h)
        
        x0 = int(cx - rw / 2) + LEFT_MARGIN
        y0 = int(cy - rh / 2) + TOP_MARGIN
        x0 = min(max(x0, LEFT_MARGIN), w - RIGHT_MARGIN - rw)
        y0 = min(max(y0, TOP_MARGIN), h - BOTTOM_MARGIN - rh)
        
        windows.append(box(x0, y0, x0 + rw, y0 + rh))
    
    return windows

def crop_labels_to_window(all_labels: AllLabels, window_box, img_w: int, img_h: int) -> AllLabels:
    """Crop all label types to fit within a window"""
    x0, y0, x1, y1 = map(int, window_box.bounds)
    tw, th = x1 - x0, y1 - y0
    
    cropped = AllLabels(detect=[], pose=[], segment=[], obb=[])
    
    # Crop detect labels
    for label in all_labels.detect:
        xmin, ymin, xmax, ymax = get_bbox_from_label(label, 'detect', img_w, img_h)
        inter = box(xmin, ymin, xmax, ymax).intersection(window_box)
        if inter and inter.area / ((xmax - xmin) * (ymax - ymin)) >= MIN_INSIDE:
            # Convert to window-relative normalized coords
            cls = int(label[0])
            xc = ((xmin + xmax) / 2 - x0) / tw
            yc = ((ymin + ymax) / 2 - y0) / th
            w = (xmax - xmin) / tw
            h = (ymax - ymin) / th
            cropped.detect.append([cls, xc, yc, w, h])
    
    # Crop pose labels
    for label in all_labels.pose:
        xmin, ymin, xmax, ymax = get_bbox_from_label(label, 'pose', img_w, img_h)
        inter = box(xmin, ymin, xmax, ymax).intersection(window_box)
        if inter and inter.area / ((xmax - xmin) * (ymax - ymin)) >= MIN_INSIDE:
            cls = int(label[0])
            xc = ((xmin + xmax) / 2 - x0) / tw
            yc = ((ymin + ymax) / 2 - y0) / th
            w = (xmax - xmin) / tw
            h = (ymax - ymin) / th
            
            new_label = [cls, xc, yc, w, h]
            
            # Crop keypoints
            keypoints = label[5:]
            for i in range(0, len(keypoints), 3):
                if i+2 < len(keypoints):
                    kx, ky, vis = keypoints[i:i+3]
                    # Check if keypoint is inside window
                    px, py = kx * img_w, ky * img_h
                    if x0 <= px <= x1 and y0 <= py <= y1:
                        kx_new = (px - x0) / tw
                        ky_new = (py - y0) / th
                        new_label.extend([kx_new, ky_new, vis])
                    else:
                        new_label.extend([0.0, 0.0, 0])  # invisible
                else:
                    break
            
            # Pad to 4 keypoints
            while len(new_label) < 5 + 12:  # 5 (cls + bbox) + 12 (4 keypoints × 3)
                new_label.extend([0.0, 0.0, 0])
            
            cropped.pose.append(new_label)
    
    # Crop segment labels
    for label in all_labels.segment:
        cls = int(label[0])
        points = label[1:]
        
        # Check if polygon intersects window
        poly_points = []
        for i in range(0, len(points), 2):
            if i+1 < len(points):
                poly_points.append((points[i] * img_w, points[i+1] * img_h))
        
        if not poly_points:
            continue
        
        # Simple check: if any point is inside window
        inside = any(x0 <= px <= x1 and y0 <= py <= y1 for px, py in poly_points)
        if inside:
            # Convert to window-relative coords
            new_label = [cls]
            for px, py in poly_points:
                new_label.append((px - x0) / tw)
                new_label.append((py - y0) / th)
            cropped.segment.append(new_label)
    
    # Crop OBB labels
    for label in all_labels.obb:
        xmin, ymin, xmax, ymax = get_bbox_from_label(label, 'obb', img_w, img_h)
        inter = box(xmin, ymin, xmax, ymax).intersection(window_box)
        if inter and inter.area / ((xmax - xmin) * (ymax - ymin)) >= MIN_INSIDE:
            cls = int(label[0])
            points = label[1:9]
            
            new_label = [cls]
            for i in range(0, 8, 2):
                px = points[i] * img_w
                py = points[i+1] * img_h
                new_label.append((px - x0) / tw)
                new_label.append((py - y0) / th)
            cropped.obb.append(new_label)
    
    return cropped

def export_windows(img: np.ndarray, img_path: Path, all_labels: AllLabels, 
                   windows, status: QLabel, mode: str = "random"):
    """Export windows with all annotation types"""
    folder_suffix = "cluster_windows" if mode == "cluster" else "random_windows"
    base_dir = EXPORT_DIR / folder_suffix
    
    # Create folder structure for all label types
    img_dir = base_dir / "images" / "train_val"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    label_dirs = {}
    for label_type in ['detect', 'pose', 'segment', 'obb']:
        label_dir = base_dir / "labels" / label_type / "train_val"
        label_dir.mkdir(parents=True, exist_ok=True)
        label_dirs[label_type] = label_dir
    
    base = img_path.stem
    nxt = _next_index(img_dir, base)
    
    h, w = img.shape[:2]
    saved, total = 0, len(windows)
    
    for w_box in windows:
        x0, y0, x1, y1 = map(int, w_box.bounds)
        if x1 <= x0 or y1 <= y0:
            continue
        
        tile = img[y0:y1, x0:x1]
        
        # Crop all labels to this window
        cropped_labels = crop_labels_to_window(all_labels, w_box, w, h)
        
        # Only save if there's at least one label of any type
        has_labels = (len(cropped_labels.detect) > 0 or 
                     len(cropped_labels.pose) > 0 or
                     len(cropped_labels.segment) > 0 or
                     len(cropped_labels.obb) > 0)
        
        if not has_labels:
            continue
        
        name = f"{base}_win{nxt + saved}"
        
        # Save image
        cv2.imwrite(str(img_dir / f"{name}.tiff"), tile)
        
        # Save each label type
        for label_type, labels in [
            ('detect', cropped_labels.detect),
            ('pose', cropped_labels.pose),
            ('segment', cropped_labels.segment),
            ('obb', cropped_labels.obb)
        ]:
            if labels:
                label_lines = []
                for label in labels:
                    label_str = ' '.join(f"{x:.6f}" if isinstance(x, float) else str(int(x)) 
                                        for x in label)
                    label_lines.append(label_str)
                
                (label_dirs[label_type] / f"{name}.txt").write_text('\n'.join(label_lines))
        
        saved += 1
        msg = f"saved {saved}/{total} → {folder_suffix}/.../{name}.tiff"
        print(msg)
        status.setText(msg)
        QApplication.processEvents()
    
    final_msg = f"Exported {saved} windows to {folder_suffix}"
    print(f"\n{final_msg}\n")
    status.setText(final_msg)

def _next_index(folder: Path, stem: str):
    """Find next available index"""
    taken = {int(p.stem.split("_")[-1][3:]) 
             for p in folder.glob(f"{stem}_win*.tiff") 
             if p.stem.split("_")[-1].startswith("win")}
    i = 0
    while i in taken:
        i += 1
    return i

# UI classes (View, SizeSelector) - same as before
class View(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
    
    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

class SizeSelector(QDialog):
    def __init__(self, parent, img_w: int, img_h: int, init_w: int = None, 
                 init_h: int = None, init_density: float = None):
        super().__init__(parent)
        self.setWindowTitle("Select Window Size, Density & Preview")
        self.resize(parent.width() * 0.8, parent.height() * 0.8)
        
        self.img_w, self.img_h = img_w, img_h
        self.base_w = min(init_w if init_w else 100, img_w)
        self.base_h = min(init_h if init_h else 100, img_h)
        self.density = init_density if init_density else 3.0
        self.preview_count = 500
        
        main = QVBoxLayout(self)
        
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setRenderHints(self.view.renderHints() | QPainter.Antialiasing)
        self.scene.addPixmap(parent.pixmap_item.pixmap())
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        main.addWidget(self.view)
        
        self.window_rect = self.scene.addRect(
            0, 0, self.base_w, self.base_h, QPen(QColor("orange"), 2)
        )
        
        self.random_items: list = []
        
        self.w_slide = None
        self.h_slide = None
        self._build_sliders(main)
        
        self.lock_cb = QCheckBox("Square (W = H)")
        self.lock_cb.setChecked(self.base_w == self.base_h)
        main.addWidget(self.lock_cb, alignment=Qt.AlignRight)
        
        self.win_lbl = QLabel()
        font = self.win_lbl.font()
        font.setBold(True)
        self.win_lbl.setFont(font)
        main.addWidget(self.win_lbl, alignment=Qt.AlignCenter)
        
        bh = QHBoxLayout()
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept)
        cn = QPushButton("Cancel")
        cn.clicked.connect(self.reject)
        bh.addWidget(ok)
        bh.addWidget(cn)
        main.addLayout(bh)
        
        self._refresh_all()
    
    def _build_sliders(self, layout: QVBoxLayout):
        for label, attr, minv, maxv, step in [
            ("Width",   "base_w",         1, self.img_w,  1),
            ("Height",  "base_h",         1, self.img_h,  1),
            ("Density", "density",      0.1, 10.0,      0.1),
            ("Preview", "preview_count",  1, 500,        1),
        ]:
            row = QWidget()
            hbox = QHBoxLayout(row)
            txt = QLabel(f"{label}:")
            slid = QSlider(Qt.Horizontal)
            
            if attr == "density":
                slid.setRange(int(minv * 10), int(maxv * 10))
                slid.setValue(int(self.density * 10))
            else:
                slid.setRange(minv, maxv)
                slid.setValue(getattr(self, attr))
            
            val = QLabel(str(getattr(self, attr)))
            slid.valueChanged.connect(
                lambda v, a=attr, l=val: self._on_slide(a, v, l)
            )
            
            hbox.addWidget(txt)
            hbox.addWidget(slid)
            hbox.addWidget(val)
            layout.addWidget(row)
            
            if attr == "base_w":
                self.w_slide = slid
            elif attr == "base_h":
                self.h_slide = slid
    
    def _on_slide(self, attr: str, value: int, lbl: QLabel):
        if attr == "density":
            self.density = value / 10.0
            lbl.setText(f"{self.density:.1f}")
        else:
            if self.lock_cb.isChecked() and attr in ("base_w", "base_h"):
                partner = self.h_slide if attr == "base_w" else self.w_slide
                partner.blockSignals(True)
                partner.setValue(value)
                partner.blockSignals(False)
                self.base_w = self.base_h = value
                
                lbl.setText(str(value))
                other_lbl = partner.parent().findChildren(QLabel)[-1]
                other_lbl.setText(str(value))
            else:
                setattr(self, attr, value)
                lbl.setText(str(value))
        
        self._refresh_all()
    
    def _refresh_all(self):
        self.window_rect.setRect(
            (self.img_w - self.base_w) / 2,
            (self.img_h - self.base_h) / 2,
            self.base_w,
            self.base_h,
        )
        
        area_ratio = (self.img_w * self.img_h) / (self.base_w * self.base_h)
        est = int(area_ratio * self.density)
        self.win_lbl.setText(f"≈ {est} windows at export")
        
        for item in self.random_items:
            self.scene.removeItem(item)
        self.random_items.clear()
        
        sample = build_random_windows(
            self.img_w, self.img_h, self.base_w, self.base_h,
            variation=RANDOM_VAR, density=self.density
        )
        pen = QPen(QColor("cyan"), 2, Qt.DashLine)
        for w in sample[:min(self.preview_count, len(sample))]:
            x0, y0, x1, y1 = w.bounds
            self.random_items.append(
                self.scene.addRect(x0, y0, x1 - x0, y1 - y0, pen)
            )

# Main window
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Multi-Format Window Exporter")
        self.view = View()
        self.setCentralWidget(self.view)
        self.status = QLabel("Choose image (Ctrl+O) - all annotation types will be loaded")
        self.statusBar().addPermanentWidget(self.status)
        
        self.last_folder = DATA_ROOT / 'images' / 'all'
        self.last_base_w = 100
        self.last_base_h = 100
        self.last_density = 3.0
        
        tb = QToolBar()
        self.addToolBar(tb)
        
        load_one = QAction("Load Image (Auto-find All Labels)", self)
        load_one.setShortcut(QKeySequence("Ctrl+O"))
        load_one.triggered.connect(self.open_files)
        tb.addAction(load_one)
        
        load_dir = QAction("Load Folder (Recursive)", self)
        load_dir.setShortcut(QKeySequence("Ctrl+Shift+O"))
        load_dir.triggered.connect(self.open_folder)
        tb.addAction(load_dir)
        
        quit_ = QAction("Quit", self)
        quit_.setShortcut(QKeySequence("Ctrl+Q"))
        quit_.triggered.connect(self.close)
        tb.addAction(quit_)
        
        self.resize(1280, 800)
        QTimer.singleShot(0, self.open_files)
    
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose folder (will search recursively)", str(self.last_folder)
        )
        if not folder:
            return
        folder = Path(folder)
        self.last_folder = folder
        
        mode_dlg = QMessageBox(self)
        mode_dlg.setWindowTitle("Export Mode")
        mode_dlg.setText("Choose export mode for ALL images:")
        cbtn = mode_dlg.addButton("Cluster windows", QMessageBox.AcceptRole)
        rbtn = mode_dlg.addButton("Random windows", QMessageBox.AcceptRole)
        xbtn = mode_dlg.addButton("Cancel", QMessageBox.RejectRole)
        mode_dlg.exec()
        
        if mode_dlg.clickedButton() == xbtn:
            self.status.setText("Canceled")
            return
        
        export_mode = "cluster" if mode_dlg.clickedButton() == cbtn else "random"
        
        img_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        
        self.status.setText("Searching for images recursively...")
        QApplication.processEvents()
        
        all_images = sorted(
            p for p in folder.rglob("*") 
            if p.suffix.lower() in img_exts and p.is_file()
        )
        
        if not all_images:
            QMessageBox.information(self, "No images", "No supported images found.")
            return
        
        self.status.setText(f"Found {len(all_images)} images. Processing...")
        QApplication.processEvents()
        
        processed = 0
        skipped = 0
        
        for idx, img_path in enumerate(all_images, 1):
            self.status.setText(f"Processing image {idx}/{len(all_images)}: {img_path.name}")
            QApplication.processEvents()
            
            # Check if any labels exist
            stem = img_path.stem
            has_labels = any(
                (DATA_ROOT / 'labels' / lt / 'all' / f'{stem}.txt').exists()
                for lt in ['detect', 'pose', 'segment', 'obb']
            )
            
            if not has_labels:
                print(f"Skipping {img_path.name} - no labels found")
                skipped += 1
                continue
            
            self.process(img_path, export_mode)
            processed += 1
        
        self.status.setText(f"Completed: {processed} processed, {skipped} skipped (no labels)")
    
    def open_files(self):
        img_filters = "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"
        img_path, _ = QFileDialog.getOpenFileName(
            self, "Choose image (all labels will be found automatically)", 
            str(self.last_folder), img_filters
        )
        if not img_path:
            return
        
        img_path = Path(img_path)
        self.last_folder = img_path.parent
        
        self.process(img_path, None)
    
    def process(self, img_path: Path, export_mode=None):
        pix = QPixmap(str(img_path))
        if pix.isNull():
            QMessageBox.critical(self, "Error", f"Cannot load {img_path}")
            return
        
        # Load ALL labels
        all_labels = load_all_labels(img_path, pix.width(), pix.height())
        
        # Check if any labels exist
        has_any = (len(all_labels.detect) > 0 or len(all_labels.pose) > 0 or
                   len(all_labels.segment) > 0 or len(all_labels.obb) > 0)
        
        if not has_any:
            print(f"Warning: No labels found for {img_path.name}")
            return
        
        print(f"\nLoaded labels for {img_path.name}:")
        print(f"  Detect: {len(all_labels.detect)}")
        print(f"  Pose: {len(all_labels.pose)}")
        print(f"  Segment: {len(all_labels.segment)}")
        print(f"  OBB: {len(all_labels.obb)}")
        
        scene = QGraphicsScene(self)
        scene.addPixmap(pix)
        self.view.setScene(scene)
        self.pixmap_item = scene.items()[0]
        
        img_np = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            QMessageBox.critical(self, "Error", f"Cannot read {img_path} with OpenCV")
            return
        
        if export_mode is None:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Export Options")
            dlg.setText("Choose export mode:")
            cbtn = dlg.addButton("Cluster windows", QMessageBox.AcceptRole)
            rbtn = dlg.addButton("Random windows", QMessageBox.AcceptRole)
            xbtn = dlg.addButton("Cancel", QMessageBox.RejectRole)
            dlg.exec()
            
            if dlg.clickedButton() == xbtn:
                self.status.setText("Canceled")
                return
            
            export_mode = "cluster" if dlg.clickedButton() == cbtn else "random"
        
        # Build windows
        if export_mode == "cluster":
            windows = build_cluster_windows(all_labels, pix.width(), pix.height())
        else:
            sel = SizeSelector(
                self, pix.width(), pix.height(),
                self.last_base_w, self.last_base_h, self.last_density
            )
            if sel.exec() != QDialog.Accepted:
                self.status.setText("Canceled")
                return
            
            self.last_base_w = sel.base_w
            self.last_base_h = sel.base_h
            self.last_density = sel.density
            
            windows = build_random_windows(
                pix.width(), pix.height(), sel.base_w, sel.base_h,
                variation=RANDOM_VAR, density=sel.density
            )
        
        # Draw everything
        out_scene = QGraphicsScene(self)
        out_scene.addPixmap(pix)
        draw_all_annotations(out_scene, all_labels, pix.width(), pix.height())
        
        pen = QPen(QColor("orange"), 3, Qt.DashLine)
        for w in windows:
            x0, y0, x1, y1 = w.bounds
            out_scene.addRect(QRectF(x0, y0, x1 - x0, y1 - y0), pen)
        
        self.view.setScene(out_scene)
        self.view.fitInView(out_scene.sceneRect(), Qt.KeepAspectRatio)
        self.status.setText(f"Exporting {len(windows)} windows…")
        QTimer.singleShot(
            0, lambda: export_windows(img_np, img_path, all_labels, windows, self.status, export_mode)
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    
    class _EscFilter(QObject):
        def eventFilter(self, o, e):
            if e.type() == QEvent.KeyPress and e.key() == Qt.Key_Escape:
                w.close()
                return True
            return False
    
    app.installEventFilter(_EscFilter())
    sys.exit(app.exec())
