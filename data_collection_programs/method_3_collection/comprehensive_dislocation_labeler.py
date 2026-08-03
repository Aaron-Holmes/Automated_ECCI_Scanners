#!/usr/bin/env python3
"""
Combined Multi-Model Sequential Workflow Dislocation Labeler - FIXED VERSION v2
- Fixed pose keypoint matching ✓
- Fixed segment/OBB with lower confidence and better debugging
- Fixed right-click delete to work through overlays
"""

import sys
import cv2
import numpy as np
import shutil
import random
from pathlib import Path
from enum import Enum
from copy import deepcopy
from scipy.optimize import curve_fit

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, QLineF
from PySide6.QtGui import QPen, QBrush, QColor, QPixmap, QPainter, QImage, QAction, QPolygonF

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

# Default configuration - edit these paths for your machine

TEST_IMAGE_FOLDER = Path("data/test_images")

DEFAULT_MODELS = {
    'detect': Path("data/YOLO_training_set/comprehensive_dataset/training/random_windows/detect/runs/random_windows_detect_20251010_143715/weights/best.pt"),
    'pose': Path("data/YOLO_training_set/comprehensive_dataset/training/random_windows/pose/runs/random_windows_pose_20251010_144749/weights/best.pt"),
    'segment': Path("data/YOLO_training_set/comprehensive_dataset/training/random_windows/segment/runs/random_windows_segment_20251010_143930/weights/best.pt"),
    'obb': Path("data/YOLO_training_set/comprehensive_dataset/training/random_windows/obb/runs/random_windows_obb_20251010_144020/weights/best.pt")
}

# Training image sizes
DETECT_TRAIN_SIZE = 512
OTHER_MODELS_TRAIN_SIZE = 64

# Confidence thresholds - LOWERED for segment/OBB
DETECT_CONF = 0.25
POSE_CONF = 0.25
SEGMENT_CONF = 0.10  # Lower for segment (was 0.25)
OBB_CONF = 0.10      # Lower for OBB (was 0.25)
IOU_THRESHOLD = 0.45

# Debug output
VERBOSE_PREDICTIONS = True  # Set to True to see what models detect

class WorkflowStage(Enum):
    POSE = "pose"
    SEGMENT = "segment"
    OBB = "obb"
    DONE = "done"

class Config:
    DATA_ROOT = Path("data/YOLO_training_set/comprehensive_dataset")
    CHARACTERS = ['edge', 'screw', 'mixed', 'unknown']
    CLASS_TO_ID = {c: i for i, c in enumerate(CHARACTERS)}
    
    @staticmethod
    def path_to_unique_name(img_path, base_folder):
        try:
            rel_path = img_path.relative_to(base_folder)
        except ValueError:
            rel_path = img_path
        parts = list(rel_path.parts[:-1])
        filename = rel_path.stem
        extension = rel_path.suffix
        if parts:
            unique_name = '-'.join(parts) + '-' + filename
        else:
            unique_name = filename
        unique_name = unique_name.replace(' ', '_')
        return unique_name + extension

def _sliding_positions(L: int, win: int, step: int):
    if win >= L:
        return [0]
    pos = list(range(0, L - win + 1, step))
    if pos[-1] != L - win:
        pos.append(L - win)
    return pos

def coarse_scan_estimate(preproc_img, model, device=0, initial_factor=10, 
                         init_window=200, step_frac=0.25, max_samples=8, verbose=True):
    STEP = int(init_window * step_frac)
    h, w = preproc_img.shape
    
    if len(preproc_img.shape) == 2:
        preproc_img = cv2.cvtColor(preproc_img, cv2.COLOR_GRAY2BGR)
    
    sizes_px, xs, ys = [], [], []
    for y in range(0, max(1, h - init_window + 1), STEP):
        for x in range(0, max(1, w - init_window + 1), STEP):
            crop = preproc_img[y:y+init_window, x:x+init_window]
            res = model.predict(crop, conf=0.05, iou=0.30, imgsz=DETECT_TRAIN_SIZE, device=device, verbose=False)[0]
            for bx0, by0, bx1, by1 in res.boxes.xyxy.cpu().numpy():
                bx0, by0, bx1, by1 = map(int, (bx0, by0, bx1, by1))
                sizes_px.append(max(bx1-bx0, by1-by0))
                xs.extend([bx0+x, bx1+x])
                ys.extend([by0+y, by1+y])
            if len(sizes_px) >= max_samples:
                break
        if len(sizes_px) >= max_samples:
            break
    if not sizes_px:
        if verbose: 
            print("[coarse] nothing found – defaulting to 50 px")
        return 50, (w//2, h//2)
    mean_sz = np.mean(sizes_px)
    size_out = int(round(max(10, mean_sz * initial_factor)))
    cx, cy = (min(xs)+max(xs))//2, (min(ys)+max(ys))//2
    if verbose:
        print(f"[coarse] mean defect ≈ {mean_sz:.1f}px  →  coarse scan_box_size = {size_out}px")
    return size_out, (cx, cy)

def refine_scan_box_size(preproc_crop, model, device=0, coarse_guess_px=120, 
                        verbose=True, conf_threshold=0.025, iou_threshold=0.30):
    if len(preproc_crop.shape) == 2:
        preproc_crop = cv2.cvtColor(preproc_crop, cv2.COLOR_GRAY2BGR)
    
    def _count_detections(img, box_size, edge_tol_px=5):
        WIN_SIZE = int(box_size)
        STEP = max(1, int(0.8 * WIN_SIZE))
        h, w = img.shape[:2]
        ys = _sliding_positions(h, WIN_SIZE, STEP)
        xs = _sliding_positions(w, WIN_SIZE, STEP)
        good = 0
        for y in ys:
            for x in xs:
                crop = img[y:y+WIN_SIZE, x:x+WIN_SIZE]
                crop_h, crop_w = crop.shape[:2]
                crop_at_left = (x == 0)
                crop_at_top = (y == 0)
                crop_at_right = (x + WIN_SIZE >= w)
                crop_at_bottom = (y + WIN_SIZE >= h)
                res = model.predict(crop, conf=conf_threshold, iou=iou_threshold, 
                                   imgsz=DETECT_TRAIN_SIZE, device=device, verbose=False)[0]
                for bx0, by0, bx1, by1 in res.boxes.xyxy.cpu().numpy():
                    bx0, by0, bx1, by1 = map(int, (bx0, by0, bx1, by1))
                    near_left = bx0 < edge_tol_px
                    near_top = by0 < edge_tol_px
                    near_right = bx1 > crop_w - edge_tol_px
                    near_bottom = by1 > crop_h - edge_tol_px
                    if (near_left and not crop_at_left) or \
                       (near_top and not crop_at_top) or \
                       (near_right and not crop_at_right) or \
                       (near_bottom and not crop_at_bottom):
                        continue
                    good += 1
        return good
    
    sizes = []
    jump = coarse_guess_px * 0.1
    s = float(coarse_guess_px * 0.8)
    H, W = preproc_crop.shape[:2]
    while s <= 2.0*coarse_guess_px + 0.5 and s <= max(W, H):
        sizes.append(int(round(s)))
        s += jump
    sizes = np.array(sizes, dtype=int)
    if verbose:
        print(f"[refine] Testing {len(sizes)} sizes from {sizes[0]} to {sizes[-1]}px...")
    counts = np.array([_count_detections(preproc_crop, sz, edge_tol_px=3) for sz in sizes], dtype=float)
    def _gauss(x, A, mu, sigma, C):
        return A * np.exp(-0.5*((x-mu)/sigma)**2) + C
    p0 = [counts.max() - counts.min(), sizes[counts.argmax()],
          max(1.0, 0.2*coarse_guess_px), counts.min()]
    bounds = ([0, sizes.min(), 1, 0],
              [np.inf, sizes.max(), sizes.max()-sizes.min(), np.inf])
    best_sz = sizes[counts.argmax()]
    try:
        popt, _ = curve_fit(_gauss, sizes, counts, p0=p0, bounds=bounds, maxfev=2000)
        A, mu, sigma, C = popt
        if sigma > 1:
            best_sz = int(round(mu))
            if verbose:
                print(f"[refine] Gaussian fit: μ={mu:.1f}px")
    except RuntimeError:
        if verbose:
            print("[refine] Using argmax")
    if verbose:
        print(f"[refine] Chosen: {best_sz}px")
    return best_sz

class DislocationBox(QGraphicsRectItem):
    def __init__(self, rect, character='unknown', is_prediction=False):
        super().__init__(rect)
        self.character = character
        self.is_prediction = is_prediction
        self.is_current = False
        
        self.pred_lobes = []
        self.pred_segment = None
        self.pred_obb_angle = 0.0
        
        self.lobes = []
        self.segment_polygon = None
        self.obb_angle = 0.0
        self.obb_center = None
        self.obb_visual = None
        self.obb_orientation_line = None
        self.segment_visual = None
        
        self.update_color()
    
    def update_color(self):
        colors = {'edge': QColor(255,50,50), 'screw': QColor(50,255,50),
                  'mixed': QColor(50,50,255), 'unknown': QColor(255,255,50)}
        if self.is_current:
            color = QColor(0, 255, 255)
            width = 3
        elif self.is_prediction:
            color = QColor(255, 165, 0)
            width = 2
        else:
            color = colors.get(self.character, QColor(255,255,50))
            width = 2
        pen = QPen(color, width)
        if self.is_prediction:
            pen.setStyle(Qt.DashLine)
        self.setPen(pen)
    
    def fit_to_segment_rotated(self):
        if not self.segment_polygon or self.segment_polygon.isEmpty():
            return
        points = []
        for i in range(self.segment_polygon.count()):
            p = self.segment_polygon.at(i)
            points.append([p.x(), p.y()])
        points = np.array(points, dtype=np.float32)
        center = np.mean(points, axis=0)
        angle_rad = np.radians(-self.obb_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        centered = points - center
        rotated = centered @ rotation_matrix.T
        min_x, min_y = rotated.min(axis=0)
        max_x, max_y = rotated.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        self.setRect(center[0] - width/2, center[1] - height/2, width, height)
    
    def to_yolo_detect(self, w, h):
        r = self.rect()
        xc = (r.x() + r.width()/2) / w
        yc = (r.y() + r.height()/2) / h
        bw = r.width() / w
        bh = r.height() / h
        cls = Config.CLASS_TO_ID[self.character]
        return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
    
    def to_yolo_pose(self, w, h):
        if not self.lobes:
            return None
        r = self.rect()
        xc = (r.x() + r.width()/2) / w
        yc = (r.y() + r.height()/2) / h
        bw = r.width() / w
        bh = r.height() / h
        cls = Config.CLASS_TO_ID[self.character]
        parts = [f"{cls}", f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for lobe_item, lobe_type in self.lobes:
            lobe_x = lobe_item.rect().center().x() / w
            lobe_y = lobe_item.rect().center().y() / h
            parts.extend([f"{lobe_x:.6f}", f"{lobe_y:.6f}", "2"])
        for _ in range(4 - len(self.lobes)):
            parts.extend(["0.0", "0.0", "0"])
        return " ".join(parts)
    
    def to_yolo_segment(self, w, h):
        if not self.segment_polygon or self.segment_polygon.isEmpty():
            return None
        pts = []
        for i in range(self.segment_polygon.count()):
            p = self.segment_polygon.at(i)
            pts.extend([p.x()/w, p.y()/h])
        cls = Config.CLASS_TO_ID[self.character]
        return f"{cls} " + ' '.join(f"{p:.6f}" for p in pts)
    
    def to_yolo_obb(self, w, h):
        if not self.segment_polygon or self.segment_polygon.isEmpty():
            return None
        r = self.rect()
        center = r.center()
        hw, hh = r.width()/2, r.height()/2
        corners = [
            QPointF(center.x() - hw, center.y() - hh),
            QPointF(center.x() + hw, center.y() - hh),
            QPointF(center.x() + hw, center.y() + hh),
            QPointF(center.x() - hw, center.y() + hh)
        ]
        angle_rad = np.radians(self.obb_angle)
        rotated = []
        for corner in corners:
            dx = corner.x() - center.x()
            dy = corner.y() - center.y()
            rx = dx * np.cos(angle_rad) - dy * np.sin(angle_rad) + center.x()
            ry = dx * np.sin(angle_rad) + dy * np.cos(angle_rad) + center.y()
            rotated.extend([rx/w, ry/h])
        cls = Config.CLASS_TO_ID[self.character]
        return f"{cls} " + ' '.join(f"{p:.6f}" for p in rotated)

class FullView(QGraphicsView):
    def __init__(self, parent):
        super().__init__()
        self.parent_window = parent
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMinimumWidth(400)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        self.drawing_box = False
        self.box_start = None
        self.temp_box = None
    
    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.parent_window.add_mode:
            self.setDragMode(QGraphicsView.NoDrag)
            pos = self.mapToScene(event.position().toPoint())
            self.drawing_box = True
            self.box_start = pos
            return
        elif event.button() == Qt.RightButton:
            pos = self.mapToScene(event.position().toPoint())
            
            # FIXED: Search through ALL items at this position to find DislocationBox
            items = self.scene.items(pos)
            dislocation_box = None
            for item in items:
                if isinstance(item, DislocationBox):
                    dislocation_box = item
                    break
            
            if dislocation_box:
                reply = QMessageBox.question(
                    self.parent_window, 
                    "Delete Dislocation",
                    "Delete this dislocation?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    try:
                        idx = self.parent_window.dislocations.index(dislocation_box)
                        self.scene.removeItem(dislocation_box)
                        for lobe_item, lobe_type in dislocation_box.lobes:
                            if lobe_item:
                                self.scene.removeItem(lobe_item)
                        if dislocation_box.obb_visual:
                            self.scene.removeItem(dislocation_box.obb_visual)
                        if dislocation_box.obb_orientation_line:
                            self.scene.removeItem(dislocation_box.obb_orientation_line)
                        if dislocation_box.segment_visual:
                            self.scene.removeItem(dislocation_box.segment_visual)
                        self.parent_window.dislocations.pop(idx)
                        if idx < self.parent_window.current_disloc_idx:
                            self.parent_window.current_disloc_idx -= 1
                        elif idx == self.parent_window.current_disloc_idx:
                            if self.parent_window.current_disloc_idx >= len(self.parent_window.dislocations):
                                self.parent_window.current_disloc_idx = max(0, len(self.parent_window.dislocations) - 1)
                            self.parent_window.current_stage = WorkflowStage.POSE
                            self.parent_window.undo_stack = []
                        self.parent_window.update_progress()
                        if self.parent_window.dislocations:
                            self.parent_window.show_current(reset_prediction_state=True)  # Reset for next dislocation
                        else:
                            self.parent_window.zoom_view.scene.clear()
                            self.parent_window.instruction_label.setText("All dislocations deleted. Add new ones (N) or move to next image.")
                        print(f"Deleted dislocation at index {idx}")
                    except ValueError:
                        pass
                return
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.drawing_box and self.box_start:
            pos = self.mapToScene(event.position().toPoint())
            if self.temp_box:
                self.scene.removeItem(self.temp_box)
            x = min(self.box_start.x(), pos.x())
            y = min(self.box_start.y(), pos.y())
            w = abs(pos.x() - self.box_start.x())
            h = abs(pos.y() - self.box_start.y())
            self.temp_box = self.scene.addRect(
                x, y, w, h,
                QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            )
            return
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing_box:
            pos = self.mapToScene(event.position().toPoint())
            if self.temp_box:
                self.scene.removeItem(self.temp_box)
                self.temp_box = None
            x = min(self.box_start.x(), pos.x())
            y = min(self.box_start.y(), pos.y())
            w = abs(pos.x() - self.box_start.x())
            h = abs(pos.y() - self.box_start.y())
            if w > 5 and h > 5:
                rect = QRectF(x, y, w, h)
                new_box = DislocationBox(rect, 'unknown', is_prediction=False)
                self.scene.addItem(new_box)
                self.parent_window.dislocations.append(new_box)
                print(f"Added new dislocation at ({x:.0f}, {y:.0f}) size {w:.0f}x{h:.0f}")
            self.drawing_box = False
            self.box_start = None
            self.parent_window.add_mode = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.parent_window.instruction_label.setText("✓ Dislocation added! Press 'N' to add more or continue labeling")
            QTimer.singleShot(2000, self.parent_window.update_instruction)
            self.parent_window.update_progress()
            return
        super().mouseReleaseEvent(event)

class ZoomView(QGraphicsView):
    def __init__(self, parent):
        super().__init__()
        self.parent_window = parent
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMinimumWidth(400)
        
        self.current_box = None
        self.offset_x = 0
        self.offset_y = 0
        
        self.waiting_for_lobe = False
        self.expected_lobe_index = 0
        self.prediction_cleared = False
        
        self.painting = False
        self.brush_size = 3  # FIXED: Smaller default brush for finer control
        self.mask = None
        self.mask_item = None
        
        self.rotating = False
        self.rotation_start = None
    
    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
    
    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        
        if event.button() == Qt.LeftButton:
            stage = self.parent_window.current_stage
            
            if stage == WorkflowStage.POSE:
                if not self.prediction_cleared and self.current_box and self.current_box.pred_lobes:
                    self.parent_window.save_undo_state('pose_clear_prediction')
                    for lobe_item, _ in self.current_box.lobes:
                        if lobe_item:
                            self.parent_window.full_view.scene.removeItem(lobe_item)
                    self.current_box.lobes = []
                    self.expected_lobe_index = 0
                    self.prediction_cleared = True
                    self.parent_window.show_current(reset_prediction_state=False)  # Don't reset, we just set it!
                
                if self.waiting_for_lobe and self.current_box:
                    lobe_type = 'dark' if self.expected_lobe_index % 2 == 0 else 'bright'
                    color = QColor(255,0,255) if lobe_type == 'dark' else QColor(0,255,255)
                    self.parent_window.save_undo_state('pose_lobe')
                    global_pos = QPointF(pos.x() + self.offset_x, pos.y() + self.offset_y)
                    item = self.parent_window.full_view.scene.addEllipse(
                        global_pos.x()-0.5, global_pos.y()-0.5, 1, 1,
                        QPen(color, 1), QBrush(color))
                    item.setZValue(10)
                    self.scene.addEllipse(
                        pos.x()-0.5, pos.y()-0.5, 1, 1,
                        QPen(color, 1), QBrush(color))
                    self.current_box.lobes.append((item, lobe_type))
                    self.expected_lobe_index += 1
                    if len(self.current_box.lobes) >= 4:
                        self.waiting_for_lobe = False
                    self.parent_window.update_instruction()
                    return
            
            elif stage == WorkflowStage.SEGMENT:
                if not self.prediction_cleared and self.current_box and self.current_box.pred_segment:
                    self.parent_window.save_undo_state('segment_clear_prediction')
                    if self.current_box.segment_visual:
                        self.parent_window.full_view.scene.removeItem(self.current_box.segment_visual)
                        self.current_box.segment_visual = None
                    self.current_box.segment_polygon = None
                    self.prediction_cleared = True
                    self.parent_window.show_current(reset_prediction_state=False)  # Don't reset, we just set it!
                
                if not self.painting:
                    self.parent_window.save_undo_state('segment_paint')
                self.painting = True
                if self.mask is None:
                    rect = self.scene.sceneRect()
                    self.mask = np.zeros((int(rect.height()), int(rect.width())), dtype=np.uint8)
                self.paint_at(pos)
                return
            
            elif stage == WorkflowStage.OBB:
                if not self.prediction_cleared and self.current_box:
                    self.parent_window.save_undo_state('obb_clear_prediction')
                    self.current_box.obb_angle = 0.0
                    self.prediction_cleared = True
                    self.parent_window.update_obb_visual(self.current_box)
                    self.parent_window.show_current(reset_prediction_state=False)  # Don't reset, we just set it!
                
                self.rotating = True
                self.rotation_start = pos
                self.parent_window.save_undo_state('obb_rotate')
                return
        
        elif event.button() == Qt.RightButton:
            stage = self.parent_window.current_stage
            if stage == WorkflowStage.POSE:
                if self.waiting_for_lobe and self.current_box and len(self.current_box.lobes) < 4:
                    self.parent_window.save_undo_state('pose_skip')
                    lobe_type = 'dark' if self.expected_lobe_index % 2 == 0 else 'bright'
                    print(f"Skipped {lobe_type} lobe at position {self.expected_lobe_index}")
                    self.expected_lobe_index += 1
                    if self.expected_lobe_index >= 4:
                        self.waiting_for_lobe = False
                    self.parent_window.update_instruction()
                    return
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        
        if self.painting and self.parent_window.current_stage == WorkflowStage.SEGMENT:
            self.paint_at(pos)
            return
        
        if self.rotating and self.rotation_start and self.current_box:
            r = self.current_box.rect()
            center = QPointF(r.center().x() - self.offset_x, r.center().y() - self.offset_y)
            dx1 = self.rotation_start.x() - center.x()
            dy1 = self.rotation_start.y() - center.y()
            dx2 = pos.x() - center.x()
            dy2 = pos.y() - center.y()
            angle1 = np.degrees(np.arctan2(dy1, dx1))
            angle2 = np.degrees(np.arctan2(dy2, dx2))
            delta = angle2 - angle1
            self.current_box.obb_angle += delta
            self.rotation_start = pos
            self.current_box.fit_to_segment_rotated()
            self.parent_window.update_obb_visual(self.current_box)
            self.parent_window.update_segment_visual(self.current_box)
            self.parent_window.show_current(reset_prediction_state=False)  # Just updating view
            return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.painting = False
            self.rotating = False
            self.rotation_start = None
        super().mouseReleaseEvent(event)
    
    def paint_at(self, pos):
        if self.mask is None:
            return
        x, y = int(pos.x()), int(pos.y())
        if 0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0]:
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            self.update_mask_visual()
    
    def update_mask_visual(self):
        if self.mask is None:
            return
        if self.mask_item:
            self.scene.removeItem(self.mask_item)
        mask_colored = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
        mask_colored[self.mask > 0] = [255, 100, 100, 120]
        h, w = self.mask.shape
        q_img = QImage(mask_colored.data, w, h, 4*w, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_img)
        self.mask_item = self.scene.addPixmap(pixmap)
        self.mask_item.setZValue(5)
    
    def finish_segment(self):
        if not self.current_box or self.mask is None or not np.any(self.mask):
            return
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        points = []
        for point in approx:
            x, y = point[0]
            global_pos = QPointF(x + self.offset_x, y + self.offset_y)
            points.append(global_pos)
        if len(points) >= 3:
            poly = QPolygonF(points)
            self.current_box.segment_polygon = poly
            self.current_box.obb_angle = 0.0
            self.current_box.fit_to_segment_rotated()
            self.parent_window.update_segment_visual(self.current_box)
        self.mask = None
        if self.mask_item:
            self.scene.removeItem(self.mask_item)
            self.mask_item = None
        self.parent_window.update_instruction()
        self.parent_window.show_current(reset_prediction_state=False)  # Just updating view

class Labeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Combined Multi-Model Sequential Labeler - FIXED v2")
        self.resize(1600, 900)
        
        self.image_files = []
        self.current_idx = 0
        self.current_image_path = None
        self.current_image = None
        self.current_image_rgb = None
        self.base_folder = None
        
        self.model_detect = None
        self.model_pose = None
        self.model_segment = None
        self.model_obb = None
        
        self.dislocations = []
        self.current_disloc_idx = 0
        self.current_stage = WorkflowStage.POSE
        self.undo_stack = []
        self.add_mode = False
        self.hide_others = False
        
        (Config.DATA_ROOT / 'images' / 'all').mkdir(parents=True, exist_ok=True)
        for task in ['detect', 'pose', 'segment', 'obb']:
            (Config.DATA_ROOT / 'labels' / task / 'all').mkdir(parents=True, exist_ok=True)
        
        self.setup_ui()
        self.load_default_models()
        
        if TEST_IMAGE_FOLDER and TEST_IMAGE_FOLDER.exists():
            QTimer.singleShot(500, lambda: self.auto_load_folder(TEST_IMAGE_FOLDER))
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        toolbar.addAction("Load Folder", self.load_folder)
        toolbar.addAction("Load Models", self.load_models)
        toolbar.addSeparator()
        
        self.stage_label = QLabel("Stage: POSE")
        self.stage_label.setStyleSheet("font-weight: bold; padding: 5px;")
        toolbar.addWidget(self.stage_label)
        
        self.instruction_label = QLabel("Load a folder to begin")
        self.instruction_label.setStyleSheet("""
            background: #2c3e50; color: #ecf0f1; padding: 10px; 
            font-size: 14px; border-radius: 5px; font-weight: bold;
        """)
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        main_layout.addWidget(self.instruction_label)
        
        panels = QHBoxLayout()
        
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_title = QLabel("Full Image")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setStyleSheet("background: #444; color: white; padding: 5px; font-weight: bold;")
        left_layout.addWidget(left_title)
        self.full_view = FullView(self)
        left_layout.addWidget(self.full_view)
        panels.addWidget(left, 1)
        
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.zoom_title = QLabel("Zoomed View")
        self.zoom_title.setAlignment(Qt.AlignCenter)
        self.zoom_title.setStyleSheet("background: #444; color: white; padding: 5px; font-weight: bold;")
        right_layout.addWidget(self.zoom_title)
        self.zoom_view = ZoomView(self)
        right_layout.addWidget(self.zoom_view)
        
        brush_control = QHBoxLayout()
        brush_control.addWidget(QLabel("Brush Size:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)  # Allow smaller brushes
        self.brush_slider.setMaximum(30)
        self.brush_slider.setValue(3)  # FIXED: Start with smaller brush
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        brush_control.addWidget(self.brush_slider)
        self.brush_label = QLabel("3")  # FIXED: Update label
        brush_control.addWidget(self.brush_label)
        right_layout.addLayout(brush_control)
        
        panels.addWidget(right, 1)
        main_layout.addLayout(panels, 1)
        
        controls = QHBoxLayout()
        controls.addWidget(QPushButton("← Prev", clicked=self.prev_image))
        controls.addWidget(QPushButton("Next →", clicked=self.next_image))
        controls.addWidget(QPushButton("Save & Next (S)", clicked=self.save_and_next_image))
        controls.addStretch()
        controls.addWidget(QPushButton("Add Dislocation (N)", clicked=self.toggle_add_mode))
        self.hide_button = QPushButton("Hide Others (H)")
        self.hide_button.clicked.connect(self.toggle_hide_others)
        controls.addWidget(self.hide_button)
        controls.addWidget(QPushButton("Undo (Ctrl+Z)", clicked=self.undo))
        controls.addWidget(QPushButton("Classify (R)", clicked=self.classify_current))
        controls.addWidget(QPushButton("Delete (Del)", clicked=self.delete_current))
        controls.addWidget(QPushButton("Accept All (A)", clicked=self.accept_all))
        controls.addWidget(QPushButton("Advance (SPACE)", clicked=self.advance))
        main_layout.addLayout(controls)
        
        self.status_label = QLabel("Load folder to begin")
        self.statusBar().addWidget(self.status_label)
        self.progress_label = QLabel("")
        self.statusBar().addPermanentWidget(self.progress_label)
        
        shortcuts = [
            ('Space', self.advance),
            ('R', self.classify_current),
            ('Delete', self.delete_current),
            ('A', self.accept_all),
            ('Ctrl+Z', self.undo),
            ('N', self.toggle_add_mode),
            ('S', self.save_and_next_image),
            ('H', self.toggle_hide_others)
        ]
        for key, func in shortcuts:
            action = QAction(self)
            action.setShortcut(key)
            action.triggered.connect(func)
            self.addAction(action)
    
    def update_brush_size(self, value):
        self.zoom_view.brush_size = value
        self.brush_label.setText(str(value))
    
    def toggle_add_mode(self):
        self.add_mode = not self.add_mode
        if self.add_mode:
            self.full_view.setDragMode(QGraphicsView.NoDrag)
            self.instruction_label.setText("➕ ADD MODE: Click and drag on the left image to draw a bounding box")
        else:
            self.full_view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.update_instruction()
    
    def toggle_hide_others(self):
        self.hide_others = not self.hide_others
        
        if self.hide_others:
            self.hide_button.setText("Show All (H)")
            for i, d in enumerate(self.dislocations):
                if i != self.current_disloc_idx:
                    d.setVisible(False)
                    for lobe_item, _ in d.lobes:
                        if lobe_item:
                            lobe_item.setVisible(False)
                    if d.segment_visual:
                        d.segment_visual.setVisible(False)
                    if d.obb_visual:
                        d.obb_visual.setVisible(False)
                    if d.obb_orientation_line:
                        d.obb_orientation_line.setVisible(False)
        else:
            self.hide_button.setText("Hide Others (H)")
            for d in self.dislocations:
                d.setVisible(True)
                for lobe_item, _ in d.lobes:
                    if lobe_item:
                        lobe_item.setVisible(True)
                if d.segment_visual:
                    d.segment_visual.setVisible(True)
                if d.obb_visual:
                    d.obb_visual.setVisible(True)
                if d.obb_orientation_line:
                    d.obb_orientation_line.setVisible(True)
    
    def save_undo_state(self, action_type):
        if not self.dislocations or self.current_disloc_idx >= len(self.dislocations):
            return
        current = self.dislocations[self.current_disloc_idx]
        state = {
            'action': action_type,
            'disloc_idx': self.current_disloc_idx,
            'stage': self.current_stage,
            'lobes': list(current.lobes),
            'expected_lobe_index': self.zoom_view.expected_lobe_index,
            'segment_polygon': QPolygonF(current.segment_polygon) if current.segment_polygon else None,
            'obb_angle': current.obb_angle,
            'character': current.character,
            'prediction_cleared': self.zoom_view.prediction_cleared
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
    
    def undo(self):
        if not self.undo_stack:
            self.instruction_label.setText("⚠️ Nothing to undo")
            QTimer.singleShot(2000, self.update_instruction)
            return
        if self.current_disloc_idx >= len(self.dislocations):
            return
        state = self.undo_stack.pop()
        current = self.dislocations[state['disloc_idx']]
        
        if state['action'] in ['pose_lobe', 'pose_clear_prediction']:
            if current.lobes and state['action'] == 'pose_lobe':
                lobe_item, _ = current.lobes.pop()
                if lobe_item:
                    self.full_view.scene.removeItem(lobe_item)
            elif state['action'] == 'pose_clear_prediction':
                current.lobes = state['lobes']
            self.zoom_view.expected_lobe_index = state.get('expected_lobe_index', max(0, len(current.lobes)))
        
        elif state['action'] == 'pose_skip':
            self.zoom_view.expected_lobe_index = state.get('expected_lobe_index', max(0, len(current.lobes)))
        
        elif state['action'] in ['segment_paint', 'segment_clear_prediction']:
            current.segment_polygon = state['segment_polygon']
            self.update_segment_visual(current)
        
        elif state['action'] in ['obb_rotate', 'obb_clear_prediction']:
            current.obb_angle = state['obb_angle']
            current.fit_to_segment_rotated()
            self.update_obb_visual(current)
            self.update_segment_visual(current)
        
        elif state['action'] == 'classify':
            current.character = state['character']
            current.update_color()
        
        self.zoom_view.prediction_cleared = state.get('prediction_cleared', False)
        self.current_stage = state['stage']
        self.show_current(reset_prediction_state=False)  # Don't reset when undoing
        self.instruction_label.setText("↶ Undo completed")
        QTimer.singleShot(2000, self.update_instruction)
    
    def load_default_models(self):
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO not available - predictions disabled")
            return
        
        loaded = []
        for task, path in DEFAULT_MODELS.items():
            if path and path.exists():
                try:
                    setattr(self, f'model_{task}', YOLO(str(path)))
                    loaded.append(task)
                    print(f"✓ Loaded {task} model: {path.name}")
                except Exception as e:
                    print(f"✗ Failed to load {task}: {e}")
            elif path:
                print(f"⚠️ {task} model not found: {path}")
        
        if loaded:
            self.status_label.setText(f"Models: {', '.join(loaded)}")
        else:
            self.status_label.setText("No models loaded")
    
    def load_models(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Load Models")
        layout = QVBoxLayout(dialog)
        paths = {}
        for task in ['detect', 'pose', 'segment', 'obb']:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{task.upper()}:"))
            line = QLineEdit()
            if DEFAULT_MODELS.get(task):
                line.setText(str(DEFAULT_MODELS[task]))
            paths[task] = line
            row.addWidget(line)
            btn = QPushButton("Browse")
            btn.clicked.connect(lambda checked, t=task: self.browse_model(paths[t]))
            row.addWidget(btn)
            layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.Accepted:
            if YOLO_AVAILABLE:
                for task, line in paths.items():
                    if line.text() and Path(line.text()).exists():
                        try:
                            setattr(self, f'model_{task}', YOLO(line.text()))
                            print(f"Loaded {task}")
                        except Exception as e:
                            QMessageBox.warning(self, "Error", f"Failed {task}: {e}")
    
    def browse_model(self, line_edit):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch (*.pt)")
        if path:
            line_edit.setText(path)
    
    def auto_load_folder(self, folder_path):
        self.base_folder = folder_path
        exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        all_images = sorted([p for p in folder_path.rglob("*") if p.suffix.lower() in exts])
        if not all_images:
            self.instruction_label.setText("⚠️ No images found in test folder")
            return
        self.image_files = []
        skipped = 0
        for img_path in all_images:
            unique_name = Config.path_to_unique_name(img_path, self.base_folder)
            label_name = Path(unique_name).stem + '.txt'
            label_path = Config.DATA_ROOT / 'labels' / 'detect' / 'all' / label_name
            if label_path.exists():
                skipped += 1
            else:
                self.image_files.append(img_path)
        if not self.image_files:
            self.instruction_label.setText(f"✓ All {len(all_images)} images already labeled!")
            return
        random.shuffle(self.image_files)
        print(f"\n✓ Auto-loaded {len(self.image_files)} unlabeled images (skipped {skipped} labeled)")
        self.current_idx = 0
        self.load_image(0)
    
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder")
        if not folder:
            return
        self.base_folder = Path(folder)
        exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        all_images = sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in exts])
        if not all_images:
            QMessageBox.warning(self, "No Images", "No images found")
            return
        self.image_files = []
        skipped = 0
        for img_path in all_images:
            unique_name = Config.path_to_unique_name(img_path, self.base_folder)
            label_name = Path(unique_name).stem + '.txt'
            label_path = Config.DATA_ROOT / 'labels' / 'detect' / 'all' / label_name
            if label_path.exists():
                skipped += 1
            else:
                self.image_files.append(img_path)
        if not self.image_files:
            QMessageBox.information(self, "All Done", f"All {len(all_images)} images are already labeled!")
            return
        random.shuffle(self.image_files)
        print(f"\nLoaded {len(self.image_files)} unlabeled images (skipped {skipped} already labeled)")
        self.current_idx = 0
        self.load_image(0)
    
    def load_image(self, idx):
        if idx < 0 or idx >= len(self.image_files):
            return
        self.current_idx = idx
        self.current_image_path = self.image_files[idx]
        self.current_image = cv2.imread(str(self.current_image_path), cv2.IMREAD_GRAYSCALE)
        if self.current_image is None:
            return
        self.current_image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
        
        print(f"\n→ Loading image {idx+1}/{len(self.image_files)}: {self.current_image_path.name}")
        
        h, w = self.current_image.shape
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
        q_img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        self.full_view.resetTransform()
        self.full_view.scene.clear()
        pixmap_item = self.full_view.scene.addPixmap(pixmap)
        self.full_view.scene.setSceneRect(pixmap_item.boundingRect())
        self.full_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.full_view.fitInView(self.full_view.scene.sceneRect(), Qt.KeepAspectRatio)
        
        self.dislocations = []
        self.current_disloc_idx = 0
        self.current_stage = WorkflowStage.POSE
        self.undo_stack = []
        self.hide_others = False
        self.hide_button.setText("Hide Others (H)")
        
        if self.model_detect:
            QTimer.singleShot(100, self.predict_all_models)
    
    def predict_all_models(self):
        """Run detection + pose/segment/obb with improved debugging"""
        if not self.model_detect or self.current_image_rgb is None:
            return
        
        print(f"\n{'='*60}")
        print(f"Running predictions with LOWER segment/OBB thresholds...")
        print(f"  Detect conf: {DETECT_CONF}")
        print(f"  Pose conf: {POSE_CONF}")
        print(f"  Segment conf: {SEGMENT_CONF} (lowered from 0.25)")
        print(f"  OBB conf: {OBB_CONF} (lowered from 0.25)")
        print(f"{'='*60}")
        
        h, w = self.current_image_rgb.shape[:2]
        
        # Step 1: Calibrate detection scan box size
        print("\n[1/3] Calibrating detection scan size...")
        coarse_sz, (cx, cy) = coarse_scan_estimate(
            self.current_image, self.model_detect, verbose=True)
        expand_factor = 10
        half_side = coarse_sz * expand_factor // 2
        x0 = max(0, cx - half_side)
        y0 = max(0, cy - half_side)
        x1 = min(w, cx + half_side)
        y1 = min(h, cy + half_side)
        crop = self.current_image[y0:y1, x0:x1]
        optimal_scan_size = refine_scan_box_size(
            crop, self.model_detect, coarse_guess_px=coarse_sz, verbose=True)
        
        # Step 2: Run detection sliding window
        print(f"\n[2/3] Running detection sliding window...")
        WIN_SIZE = optimal_scan_size
        STEP = max(1, int(0.8 * WIN_SIZE))
        ys = _sliding_positions(h, WIN_SIZE, STEP)
        xs = _sliding_positions(w, WIN_SIZE, STEP)
        EDGE_TOL = int(WIN_SIZE * 0.05)
        
        detect_boxes = []
        for y in ys:
            for x in xs:
                crop = self.current_image_rgb[y:y+WIN_SIZE, x:x+WIN_SIZE]
                crop_h, crop_w = crop.shape[:2]
                crop_at_left = (x == 0)
                crop_at_top = (y == 0)
                crop_at_right = (x + WIN_SIZE >= w)
                crop_at_bottom = (y + WIN_SIZE >= h)
                
                results = self.model_detect.predict(crop, conf=DETECT_CONF, iou=IOU_THRESHOLD, 
                                                    imgsz=DETECT_TRAIN_SIZE, device=0, verbose=False)[0]
                if results.boxes is not None:
                    for box in results.boxes.xyxy.cpu().numpy():
                        bx0, by0, bx1, by1 = map(int, box[:4])
                        near_left = bx0 < EDGE_TOL
                        near_top = by0 < EDGE_TOL
                        near_right = bx1 > crop_w - EDGE_TOL
                        near_bottom = by1 > crop_h - EDGE_TOL
                        if (near_left and not crop_at_left) or \
                           (near_top and not crop_at_top) or \
                           (near_right and not crop_at_right) or \
                           (near_bottom and not crop_at_bottom):
                            continue
                        gx0, gy0 = bx0 + x, by0 + y
                        gx1, gy1 = bx1 + x, by1 + y
                        detect_boxes.append((gx0, gy0, gx1, gy1))
        
        if not detect_boxes:
            print("No dislocations detected")
            self.instruction_label.setText("No detections. Add manually (N) or next image.")
            return
        
        final_boxes = self._nms_boxes(detect_boxes, iou_threshold=0.3)
        print(f"After NMS: {len(final_boxes)} unique dislocations\n")
        
        # Step 3: Run other models on individual detections
        print(f"[3/3] Running pose/segment/obb on {len(final_boxes)} detections...\n")
        for idx, (x0, y0, x1, y1) in enumerate(final_boxes):
            bw, bh = x1 - x0, y1 - y0
            box_cx, box_cy = (x0 + x1) / 2, (y0 + y1) / 2
            
            # Create square crop
            box_size = max(bw, bh)
            pad_factor = 0.3
            crop_size = int(box_size * (1 + pad_factor))
            
            crop_x0 = max(0, int(box_cx - crop_size/2))
            crop_y0 = max(0, int(box_cy - crop_size/2))
            crop_x1 = min(w, int(box_cx + crop_size/2))
            crop_y1 = min(h, int(box_cy + crop_size/2))
            
            crop_w = crop_x1 - crop_x0
            crop_h = crop_y1 - crop_y0
            actual_crop_size = min(crop_w, crop_h)
            crop_x1 = crop_x0 + actual_crop_size
            crop_y1 = crop_y0 + actual_crop_size
            
            crop = self.current_image_rgb[crop_y0:crop_y1, crop_x0:crop_x1]
            
            rect = QRectF(x0, y0, x1 - x0, y1 - y0)
            item = DislocationBox(rect, 'unknown', is_prediction=True)
            
            if VERBOSE_PREDICTIONS:
                print(f"  Detection {idx+1}/{len(final_boxes)}: box=({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}) crop={actual_crop_size}px")
            
            # Run pose model
            if self.model_pose:
                results = self.model_pose.predict(crop, conf=POSE_CONF, iou=IOU_THRESHOLD, 
                                                  imgsz=OTHER_MODELS_TRAIN_SIZE, device=0, verbose=False)[0]
                if results.boxes is not None and hasattr(results, 'keypoints') and results.keypoints is not None:
                    center_x, center_y = box_cx, box_cy
                    best_kpts = None
                    min_dist = float('inf')
                    
                    for box, kpts in zip(results.boxes.xyxy.cpu().numpy(), results.keypoints.xy.cpu().numpy()):
                        bx0, by0, bx1, by1 = box[:4]
                        pred_cx = (bx0 + bx1) / 2 + crop_x0
                        pred_cy = (by0 + by1) / 2 + crop_y0
                        dist = ((center_x - pred_cx)**2 + (center_y - pred_cy)**2)**0.5
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_kpts = []
                            for kx, ky in kpts:
                                if kx > 0 and ky > 0:
                                    global_kx = kx + crop_x0
                                    global_ky = ky + crop_y0
                                    best_kpts.append(QPointF(global_kx, global_ky))
                    
                    if best_kpts:
                        item.pred_lobes = best_kpts
                        if VERBOSE_PREDICTIONS:
                            print(f"    ✓ Pose: {len(best_kpts)} keypoints")
                        for i, pt in enumerate(best_kpts):
                            lobe_type = 'dark' if i % 2 == 0 else 'bright'
                            color = QColor(255,0,255) if lobe_type == 'dark' else QColor(0,255,255)
                            lobe_item = self.full_view.scene.addEllipse(
                                pt.x()-0.5, pt.y()-0.5, 1, 1,
                                QPen(color, 1), QBrush(color))
                            lobe_item.setZValue(10)
                            item.lobes.append((lobe_item, lobe_type))
                    elif VERBOSE_PREDICTIONS:
                        print(f"    ✗ Pose: No keypoints found")
            
            # Run segment model with LOWER confidence
            if self.model_segment:
                results = self.model_segment.predict(crop, conf=SEGMENT_CONF, iou=IOU_THRESHOLD, 
                                                     imgsz=OTHER_MODELS_TRAIN_SIZE, device=0, verbose=False)[0]
                
                if VERBOSE_PREDICTIONS:
                    if hasattr(results, 'masks') and results.masks is not None:
                        print(f"    → Segment: Found {len(results.masks)} masks")
                    else:
                        print(f"    ✗ Segment: No masks detected (try lowering SEGMENT_CONF further)")
                
                if hasattr(results, 'masks') and results.masks is not None:
                    center_x, center_y = box_cx, box_cy
                    best_poly = None
                    min_dist = float('inf')
                    
                    for mask_idx, mask in enumerate(results.masks.xy):
                        if len(mask) >= 3:
                            global_poly = [QPointF(px + crop_x0, py + crop_y0) for px, py in mask]
                            poly_cx = np.mean([p.x() for p in global_poly])
                            poly_cy = np.mean([p.y() for p in global_poly])
                            dist = ((center_x - poly_cx)**2 + (center_y - poly_cy)**2)**0.5
                            
                            if VERBOSE_PREDICTIONS and mask_idx < 3:  # Show first 3
                                print(f"      Mask {mask_idx}: {len(mask)} pts, dist={dist:.1f}px")
                            
                            if dist < min_dist:
                                min_dist = dist
                                best_poly = global_poly
                    
                    if best_poly:
                        qpoly = QPolygonF(best_poly)
                        item.pred_segment = qpoly
                        item.segment_polygon = qpoly
                        if VERBOSE_PREDICTIONS:
                            print(f"    ✓ Segment: Selected mask with {len(best_poly)} points (dist={min_dist:.1f})")
            
            # Run OBB model with LOWER confidence
            if self.model_obb:
                results = self.model_obb.predict(crop, conf=OBB_CONF, iou=IOU_THRESHOLD, 
                                                imgsz=OTHER_MODELS_TRAIN_SIZE, device=0, verbose=False)[0]
                
                if VERBOSE_PREDICTIONS:
                    if hasattr(results, 'obb') and results.obb is not None:
                        print(f"    → OBB: Found {len(results.obb)} oriented boxes")
                    else:
                        print(f"    ✗ OBB: No boxes detected (try lowering OBB_CONF further)")
                
                if hasattr(results, 'obb') and results.obb is not None:
                    center_x, center_y = box_cx, box_cy
                    best_angle = 0.0
                    min_dist = float('inf')
                    
                    for obb_idx, obb in enumerate(results.obb.xyxyxyxy.cpu().numpy()):
                        corners = obb.reshape(-1, 2)
                        global_corners = corners + np.array([crop_x0, crop_y0])
                        obb_cx = np.mean(global_corners[:, 0])
                        obb_cy = np.mean(global_corners[:, 1])
                        dist = ((center_x - obb_cx)**2 + (center_y - obb_cy)**2)**0.5
                        
                        if dist < min_dist:
                            min_dist = dist
                            vec = global_corners[1] - global_corners[0]
                            best_angle = np.degrees(np.arctan2(vec[1], vec[0]))
                    
                    if abs(best_angle) > 0.1:
                        item.pred_obb_angle = best_angle
                        item.obb_angle = best_angle
                        if VERBOSE_PREDICTIONS:
                            print(f"    ✓ OBB: angle={best_angle:.1f}° (dist={min_dist:.1f})")
            
            if VERBOSE_PREDICTIONS:
                print()  # Blank line between detections
            
            self.full_view.scene.addItem(item)
            self.dislocations.append(item)
            self.update_segment_visual(item)
            self.update_obb_visual(item)
        
        print(f"{'='*60}\n")
        
        if self.dislocations:
            self.show_current(reset_prediction_state=True)  # Reset for new predictions
        else:
            self.instruction_label.setText("No detections. Add manually (N) or next image.")
    
    def _nms_boxes(self, boxes, iou_threshold=0.3):
        if not boxes:
            return []
        boxes_array = np.array(boxes, dtype=np.float32)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return [tuple(map(int, boxes_array[i])) for i in keep]
    
    def update_segment_visual(self, box):
        if box.segment_visual:
            self.full_view.scene.removeItem(box.segment_visual)
            box.segment_visual = None
        if box.segment_polygon and not box.segment_polygon.isEmpty():
            box.segment_visual = self.full_view.scene.addPolygon(
                box.segment_polygon,
                QPen(QColor(255,100,100), 1),
                QBrush(QColor(255,100,100,50))
            )
            box.segment_visual.setZValue(4)
    
    def update_obb_visual(self, box):
        if box.obb_visual:
            self.full_view.scene.removeItem(box.obb_visual)
            box.obb_visual = None
        if box.obb_orientation_line:
            self.full_view.scene.removeItem(box.obb_orientation_line)
            box.obb_orientation_line = None
        
        r = box.rect()
        center = r.center()
        hw, hh = r.width()/2, r.height()/2
        
        if abs(box.obb_angle) < 0.01:
            box.obb_visual = self.full_view.scene.addRect(
                r.x(), r.y(), r.width(), r.height(),
                QPen(QColor(0,255,255,150), 2))
            box.obb_visual.setZValue(5)
            box.obb_orientation_line = self.full_view.scene.addLine(
                center.x(), r.y(), center.x(), r.y() + r.height(),
                QPen(QColor(0,255,255,100), 1))
            box.obb_orientation_line.setZValue(6)
        else:
            corners = [
                QPointF(-hw, -hh), QPointF(hw, -hh),
                QPointF(hw, hh), QPointF(-hw, hh)
            ]
            angle_rad = np.radians(box.obb_angle)
            poly_points = []
            for corner in corners:
                rx = corner.x() * np.cos(angle_rad) - corner.y() * np.sin(angle_rad) + center.x()
                ry = corner.x() * np.sin(angle_rad) + corner.y() * np.cos(angle_rad) + center.y()
                poly_points.append(QPointF(rx, ry))
            box.obb_visual = self.full_view.scene.addPolygon(
                QPolygonF(poly_points), QPen(QColor(0,255,255,150), 2))
            box.obb_visual.setZValue(5)
            line_top = QPointF(0, -hh)
            line_bottom = QPointF(0, hh)
            top_rx = line_top.x() * np.cos(angle_rad) - line_top.y() * np.sin(angle_rad) + center.x()
            top_ry = line_top.x() * np.sin(angle_rad) + line_top.y() * np.cos(angle_rad) + center.y()
            bot_rx = line_bottom.x() * np.cos(angle_rad) - line_bottom.y() * np.sin(angle_rad) + center.x()
            bot_ry = line_bottom.x() * np.sin(angle_rad) + line_bottom.y() * np.cos(angle_rad) + center.y()
            box.obb_orientation_line = self.full_view.scene.addLine(
                top_rx, top_ry, bot_rx, bot_ry,
                QPen(QColor(0,255,255,100), 1))
            box.obb_orientation_line.setZValue(6)
    
    def show_current(self, reset_prediction_state=False):
        if not self.dislocations:
            self.zoom_view.scene.clear()
            return
        if self.current_disloc_idx >= len(self.dislocations):
            self.save_and_next_image()
            return
        
        for i, d in enumerate(self.dislocations):
            d.is_current = (i == self.current_disloc_idx)
            d.update_color()
            
            if self.hide_others and i != self.current_disloc_idx:
                d.setVisible(False)
                for lobe_item, _ in d.lobes:
                    if lobe_item:
                        lobe_item.setVisible(False)
                if d.segment_visual:
                    d.segment_visual.setVisible(False)
                if d.obb_visual:
                    d.obb_visual.setVisible(False)
                if d.obb_orientation_line:
                    d.obb_orientation_line.setVisible(False)
        
        current = self.dislocations[self.current_disloc_idx]
        
        r = current.rect()
        pad = 20
        x0 = max(0, int(r.x() - pad))
        y0 = max(0, int(r.y() - pad))
        x1 = min(self.current_image.shape[1], int(r.x() + r.width() + pad))
        y1 = min(self.current_image.shape[0], int(r.y() + r.height() + pad))
        
        patch = self.current_image[y0:y1, x0:x1].copy()
        ph, pw = patch.shape
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        q_img = QImage(patch_rgb.data, pw, ph, 3*pw, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        self.zoom_view.resetTransform()
        self.zoom_view.scene.clear()
        self.zoom_view.mask = None
        self.zoom_view.mask_item = None
        pixmap_item = self.zoom_view.scene.addPixmap(pixmap)
        self.zoom_view.scene.setSceneRect(pixmap_item.boundingRect())
        
        local_rect = QRectF(r.x()-x0, r.y()-y0, r.width(), r.height())
        
        if self.current_stage == WorkflowStage.OBB and current.obb_angle != 0:
            center = local_rect.center()
            hw, hh = local_rect.width()/2, local_rect.height()/2
            corners = [
                QPointF(-hw, -hh), QPointF(hw, -hh),
                QPointF(hw, hh), QPointF(-hw, hh)
            ]
            angle_rad = np.radians(current.obb_angle)
            poly_points = []
            for corner in corners:
                rx = corner.x() * np.cos(angle_rad) - corner.y() * np.sin(angle_rad) + center.x()
                ry = corner.x() * np.sin(angle_rad) + corner.y() * np.cos(angle_rad) + center.y()
                poly_points.append(QPointF(rx, ry))
            self.zoom_view.scene.addPolygon(
                QPolygonF(poly_points), 
                QPen(QColor(0,255,255,120), 1))
            line_top = QPointF(0, -hh)
            line_bottom = QPointF(0, hh)
            top_rx = line_top.x() * np.cos(angle_rad) - line_top.y() * np.sin(angle_rad) + center.x()
            top_ry = line_top.x() * np.sin(angle_rad) + line_top.y() * np.cos(angle_rad) + center.y()
            bot_rx = line_bottom.x() * np.cos(angle_rad) - line_bottom.y() * np.sin(angle_rad) + center.x()
            bot_ry = line_bottom.x() * np.sin(angle_rad) + line_bottom.y() * np.cos(angle_rad) + center.y()
            self.zoom_view.scene.addLine(
                top_rx, top_ry, bot_rx, bot_ry,
                QPen(QColor(0,255,255,80), 1))
        else:
            self.zoom_view.scene.addRect(
                local_rect, 
                QPen(QColor(0,255,255,120), 1))
        
        self.zoom_view.offset_x = x0
        self.zoom_view.offset_y = y0
        self.zoom_view.current_box = current
        
        # FIXED: Only reset prediction_cleared when switching dislocations/stages
        if reset_prediction_state:
            self.zoom_view.prediction_cleared = False
        
        if self.current_stage == WorkflowStage.POSE:
            self.zoom_view.waiting_for_lobe = True
            self.zoom_view.expected_lobe_index = len(current.lobes)
        else:
            self.zoom_view.waiting_for_lobe = False
        
        if self.current_stage == WorkflowStage.POSE:
            for lobe_item, lobe_type in current.lobes:
                if lobe_item:
                    center = lobe_item.rect().center()
                    color = QColor(255,0,255) if lobe_type == 'dark' else QColor(0,255,255)
                    self.zoom_view.scene.addEllipse(
                        center.x()-x0-0.5, center.y()-y0-0.5, 1, 1,
                        QPen(color, 1), QBrush(color))
        
        if self.current_stage == WorkflowStage.SEGMENT:
            if current.segment_polygon and not current.segment_polygon.isEmpty():
                poly_local = QPolygonF()
                for i in range(current.segment_polygon.count()):
                    p = current.segment_polygon.at(i)
                    poly_local.append(QPointF(p.x() - x0, p.y() - y0))
                self.zoom_view.scene.addPolygon(poly_local, 
                    QPen(QColor(255,100,100), 2), QBrush(QColor(255,100,100,50)))
        
        self.zoom_view.fitInView(self.zoom_view.scene.sceneRect(), Qt.KeepAspectRatio)
        
        self.zoom_title.setText(
            f"Dislocation {self.current_disloc_idx+1}/{len(self.dislocations)} - {current.character}")
        self.stage_label.setText(f"Stage: {self.current_stage.value.upper()}")
        
        self.update_instruction()
        self.update_progress()
    
    def update_instruction(self):
        if not self.dislocations or self.current_disloc_idx >= len(self.dislocations):
            return
        
        current = self.dislocations[self.current_disloc_idx]
        stage = self.current_stage
        
        has_pose_pred = bool(current.pred_lobes)
        has_segment_pred = bool(current.pred_segment)
        has_obb_pred = abs(current.pred_obb_angle) > 0.1
        
        if stage == WorkflowStage.POSE:
            num_lobes = len(current.lobes)
            pred_text = " [prediction shown]" if has_pose_pred and not self.zoom_view.prediction_cleared else ""
            if num_lobes < 2:
                lobe_type = 'DARK' if num_lobes == 0 else 'BRIGHT'
                self.instruction_label.setText(
                    f"📍 POSE: Click {lobe_type} lobe ({num_lobes}/2 min){pred_text} | "
                    f"SPACE to accept | Right-click skip")
            elif num_lobes < 4:
                next_type = 'DARK' if self.zoom_view.expected_lobe_index % 2 == 0 else 'BRIGHT'
                self.instruction_label.setText(
                    f"📍 POSE: Add more ({num_lobes}/4) - Next: {next_type} | "
                    f"Right-click skip | SPACE continue")
            else:
                self.zoom_view.waiting_for_lobe = False
                self.instruction_label.setText("✓ POSE complete! Press SPACE")
        
        elif stage == WorkflowStage.SEGMENT:
            pred_text = " [prediction shown]" if has_segment_pred and not self.zoom_view.prediction_cleared else ""
            if not current.segment_polygon or current.segment_polygon.isEmpty():
                self.instruction_label.setText(
                    f"🖌️ SEGMENT: Paint area{pred_text} | SPACE to accept")
            else:
                self.instruction_label.setText("✓ SEGMENT complete! Press SPACE")
        
        elif stage == WorkflowStage.OBB:
            pred_text = " [prediction shown]" if has_obb_pred and not self.zoom_view.prediction_cleared else ""
            self.instruction_label.setText(
                f"🔄 OBB: Rotate ({current.obb_angle:.1f}°){pred_text} | SPACE to accept/continue")
    
    def advance(self):
        if not self.dislocations or self.current_disloc_idx >= len(self.dislocations):
            return
        
        current = self.dislocations[self.current_disloc_idx]
        
        if self.current_stage == WorkflowStage.POSE:
            if len(current.lobes) >= 2:
                self.zoom_view.waiting_for_lobe = False
                self.current_stage = WorkflowStage.SEGMENT
                self.show_current(reset_prediction_state=True)  # Reset for new stage
            else:
                self.instruction_label.setText("⚠️ Complete POSE first: mark at least 2 lobes")
                QTimer.singleShot(2000, self.update_instruction)
        
        elif self.current_stage == WorkflowStage.SEGMENT:
            if self.zoom_view.mask is not None and np.any(self.zoom_view.mask):
                self.zoom_view.finish_segment()
            if current.segment_polygon and not current.segment_polygon.isEmpty():
                self.current_stage = WorkflowStage.OBB
                current.setVisible(False)
                if not self.zoom_view.prediction_cleared and abs(current.pred_obb_angle) > 0.1:
                    current.obb_angle = current.pred_obb_angle
                else:
                    current.obb_angle = 0.0
                current.fit_to_segment_rotated()
                self.update_obb_visual(current)
                self.update_segment_visual(current)
                self.show_current(reset_prediction_state=True)  # Reset for new stage
            else:
                self.instruction_label.setText("⚠️ Complete SEGMENT first: paint the dislocation area")
                QTimer.singleShot(2000, self.update_instruction)
        
        elif self.current_stage == WorkflowStage.OBB:
            self.current_disloc_idx += 1
            self.current_stage = WorkflowStage.POSE
            self.undo_stack = []
            if self.current_disloc_idx >= len(self.dislocations):
                self.save_and_next_image()
            else:
                self.show_current(reset_prediction_state=True)  # Reset for new dislocation
    
    def classify_current(self):
        if not self.dislocations or self.current_disloc_idx >= len(self.dislocations):
            return
        current = self.dislocations[self.current_disloc_idx]
        self.save_undo_state('classify')
        char, ok = QInputDialog.getItem(self, "Classify", "Character:",
            Config.CHARACTERS, Config.CHARACTERS.index(current.character), False)
        if ok:
            current.character = char
            current.is_prediction = False
            current.update_color()
            self.zoom_title.setText(
                f"Dislocation {self.current_disloc_idx+1}/{len(self.dislocations)} - {char}")
    
    def delete_current(self):
        if not self.dislocations or self.current_disloc_idx >= len(self.dislocations):
            return
        current = self.dislocations[self.current_disloc_idx]
        self.full_view.scene.removeItem(current)
        for lobe_item, lobe_type in current.lobes:
            if lobe_item:
                self.full_view.scene.removeItem(lobe_item)
        if current.obb_visual:
            self.full_view.scene.removeItem(current.obb_visual)
        if current.obb_orientation_line:
            self.full_view.scene.removeItem(current.obb_orientation_line)
        if current.segment_visual:
            self.full_view.scene.removeItem(current.segment_visual)
        self.dislocations.pop(self.current_disloc_idx)
        if self.current_disloc_idx >= len(self.dislocations):
            self.current_disloc_idx = max(0, len(self.dislocations)-1)
        self.current_stage = WorkflowStage.POSE
        self.undo_stack = []
        if self.dislocations:
            self.show_current(reset_prediction_state=True)  # Reset for next dislocation
        else:
            self.save_and_next_image()
    
    def accept_all(self):
        for d in self.dislocations:
            d.is_prediction = False
            d.update_color()
    
    def save_and_next_image(self):
        if not self.validate_annotations():
            return
        self.save_current()
        if self.current_idx < len(self.image_files) - 1:
            self.load_image(self.current_idx + 1)
        else:
            self.instruction_label.setText("🎉 All images complete!")
            self.create_yaml_configs()
    
    def validate_annotations(self):
        if not self.dislocations:
            return True
        issues = []
        unknown_count = 0
        missing_pose_count = 0
        missing_segment_count = 0
        for i, d in enumerate(self.dislocations):
            if d.character == 'unknown':
                unknown_count += 1
            if len(d.lobes) < 2:
                missing_pose_count += 1
            if not d.segment_polygon or d.segment_polygon.isEmpty():
                missing_segment_count += 1
        if unknown_count > 0:
            issues.append(f"• {unknown_count} dislocation(s) still marked as 'unknown'")
        if missing_pose_count > 0:
            issues.append(f"• {missing_pose_count} dislocation(s) missing pose keypoints")
        if missing_segment_count > 0:
            issues.append(f"• {missing_segment_count} dislocation(s) missing segmentation")
        if issues and not (missing_pose_count == 0 and missing_segment_count == 0):
            message = "⚠️ Incomplete Annotations Detected:\n\n" + "\n".join(issues)
            message += "\n\nDo you want to save anyway?"
            reply = QMessageBox.question(
                self, "Validation Warning", message,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return reply == QMessageBox.Yes
        return True
    
    def save_current(self):
        if not self.current_image_path or self.current_image is None:
            return
        h, w = self.current_image.shape
        unique_name = Config.path_to_unique_name(self.current_image_path, self.base_folder)
        unique_stem = Path(unique_name).stem
        print(f"\nSaving: {unique_name}")
        img_dest = Config.DATA_ROOT / 'images' / 'all' / unique_name
        shutil.copy2(self.current_image_path, img_dest)
        
        lines = [d.to_yolo_detect(w, h) for d in self.dislocations]
        if lines:
            (Config.DATA_ROOT / 'labels' / 'detect' / 'all' / (unique_stem + '.txt')).write_text('\n'.join(lines))
        
        lines = [d.to_yolo_pose(w, h) for d in self.dislocations if d.to_yolo_pose(w, h)]
        if lines:
            (Config.DATA_ROOT / 'labels' / 'pose' / 'all' / (unique_stem + '.txt')).write_text('\n'.join(lines))
        
        lines = [d.to_yolo_segment(w, h) for d in self.dislocations if d.to_yolo_segment(w, h)]
        if lines:
            (Config.DATA_ROOT / 'labels' / 'segment' / 'all' / (unique_stem + '.txt')).write_text('\n'.join(lines))
        
        lines = [d.to_yolo_obb(w, h) for d in self.dislocations if d.to_yolo_obb(w, h)]
        if lines:
            (Config.DATA_ROOT / 'labels' / 'obb' / 'all' / (unique_stem + '.txt')).write_text('\n'.join(lines))
        
        print(f"Saved {len(self.dislocations)} dislocations\n")
    
    def create_yaml_configs(self):
        print(f"\n{'='*60}")
        print("Creating YAML configuration files...")
        print(f"{'='*60}")
        for task in ['detect', 'pose', 'segment', 'obb']:
            yaml_content = f"""# YOLOv11 {task} configuration
path: {task}
train: images/all
val: images/all
names:
  0: edge
  1: screw
  2: mixed
  3: unknown
"""
            if task == 'pose':
                yaml_content += """
kpt_shape: [4, 2]
flip_idx: [0, 1, 2, 3]
"""
            yaml_path = Config.DATA_ROOT / f'{task}_data.yaml'
            yaml_path.write_text(yaml_content)
            print(f"  {task.upper()} config → {yaml_path}")
        print(f"{'='*60}\n")
    
    def prev_image(self):
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)
    
    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.load_image(self.current_idx + 1)
    
    def update_progress(self):
        if self.image_files:
            self.progress_label.setText(
                f"Image {self.current_idx+1}/{len(self.image_files)} | "
                f"{len(self.dislocations)} dislocations")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Labeler()
    window.show()
    sys.exit(app.exec())
