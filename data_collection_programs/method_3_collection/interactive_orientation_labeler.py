#!/usr/bin/env python3
"""
interactive_orientation_labeler.py
===================================
PySide6 GUI for labeling dislocation orientations.

LEFT CLICK: Auto-adjust to nearby peak/valley
RIGHT CLICK: Use exact position (no adjustment)
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import random

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QProgressBar,
                               QMessageBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QKeySequence


def yolo_to_rect(line: str, w: int, h: int) -> Tuple[int, int, int, int]:
    """Convert YOLO format to pixel rectangle (x0, y0, x1, y1)"""
    parts = line.split()
    if len(parts) < 5:
        return None
    
    _cls, xc, yc, bw, bh = map(float, parts[:5])
    x0 = int((xc - bw / 2) * w)
    y0 = int((yc - bh / 2) * h)
    x1 = int((xc + bw / 2) * w)
    y1 = int((yc + bh / 2) * h)
    return (x0, y0, x1, y1)


def _atomic_json_write(path: Path, data) -> bool:
    """
    Crash-safe JSON write:
      - write to <file>.tmp
      - flush + fsync
      - os.replace to final path
    Returns True on success, False on failure.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"ERROR: atomic write failed for {path}: {e}")
        return False


class ImageLabel(QLabel):
    """Custom QLabel that emits click signals with button info"""
    clicked = Signal(int, int, bool)  # x, y, use_peak_finding
    
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setFocusPolicy(Qt.StrongFocus)  # Can receive keyboard focus
    
    def mousePressEvent(self, event):
        pos = event.pos()
        pixmap = self.pixmap()
        if pixmap:
            label_w = self.width()
            label_h = self.height()
            pixmap_w = pixmap.width()
            pixmap_h = pixmap.height()
            
            offset_x = (label_w - pixmap_w) // 2
            offset_y = (label_h - pixmap_h) // 2
            
            x = pos.x() - offset_x
            y = pos.y() - offset_y
            
            if 0 <= x < pixmap_w and 0 <= y < pixmap_h:
                # Left click = use peak finding, Right click = exact position
                use_peak_finding = (event.button() == Qt.LeftButton)
                self.clicked.emit(x, y, use_peak_finding)
                
                # Keep focus on this widget so Enter key works
                self.setFocus()


class OrientationLabelerGUI(QMainWindow):
    def __init__(self, root_path: Path, target_count: int = 200, marker_radius: int = 5):
        super().__init__()
        
        self.root_path = root_path
        self.target_count = target_count
        self.marker_radius = marker_radius
        self.peak_search_radius = 2  # pixels to search around click
        
        # Create orientation_data folder structure
        self.orientation_root = self.root_path / 'orientation_data'
        for subdir in ['train', 'val']:
            (self.orientation_root / subdir).mkdir(parents=True, exist_ok=True)
        
        # Find all image/label pairs using your logic
        print("Finding image/label pairs...")
        self.pairs = self.find_image_label_pairs()
        
        if not self.pairs:
            QMessageBox.critical(None, "Error", "No image/label pairs found!")
            sys.exit(1)
        
        print(f"Found {len(self.pairs)} image/label pairs")
        
        # Load existing labels and filter out fully-labeled images
        print("Loading existing orientation labels...")
        self.labels = self.load_existing_labels()
        
        initial_count = len(self.pairs)
        self.pairs = self.filter_unlabeled_pairs()
        skipped_count = initial_count - len(self.pairs)
        
        print(f"Already labeled: {self.get_labeled_count()} dislocations")
        print(f"Skipped {skipped_count} fully-labeled images")
        print(f"Remaining images to label: {len(self.pairs)}")
        
        if not self.pairs:
            print("All images are already fully labeled!")
            QMessageBox.information(None, "Done", 
                                   f"All images already labeled!\n"
                                   f"Total: {self.get_labeled_count()} dislocations")
            sys.exit(0)
        
        # Current state
        self.current_pair_idx = 0
        self.current_bbox_idx = 0
        self.current_image = None
        self.current_image_path = None
        self.current_label_lines = None
        self.current_bboxes = None
        self.current_bbox = None
        self.current_clicks = []
        self.patch_smoothed = None  # For peak finding
        self.patch_offset = (0, 0)
        self.scale_factor = 1.0
        
        # UI setup
        self.setup_ui()
        
        # Load first image
        self.load_next_image()
    
    def find_image_label_pairs(self) -> List[Tuple[Path, Path]]:
        """Find image/label pairs using the same logic as training script"""
        image_exts = (".tiff", ".tif", ".png", ".jpg", ".jpeg")
        pairs = []
        
        for img_path in self.root_path.rglob("*"):
            if img_path.suffix.lower() not in image_exts:
                continue
            
            # Swap "/images/" with "/labels/" and change suffix to .txt
            parts = list(img_path.parts)
            try:
                idx = parts.index("images")
            except ValueError:
                continue  # skip files outside the images tree
            
            parts[idx] = "labels"
            lbl_path = Path(*parts).with_suffix(".txt")
            
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
        
        # Shuffle for diversity
        random.shuffle(pairs)
        
        return pairs
    
    def get_orientation_path(self, img_path: Path) -> Path:
        """Get orientation file path for an image, mirroring the structure"""
        # Swap "images" with "orientation_data" and change to .json
        parts = list(img_path.parts)
        try:
            idx = parts.index("images")
        except ValueError:
            return None
        
        parts[idx] = "orientation_data"
        base_path = Path(*parts)
        
        # Create the new filename: original_stem + "_orientations.json"
        orientation_path = base_path.parent / (base_path.stem + "_orientations.json")
        
        return orientation_path
    
    def load_existing_labels(self) -> Dict:
        """Load all existing orientation labels"""
        labels = {}
        
        for img_path, lbl_path in self.pairs:
            orientation_path = self.get_orientation_path(img_path)
            if orientation_path and orientation_path.exists():
                try:
                    with open(orientation_path, 'r') as f:
                        data = json.load(f)
                        labels[str(img_path)] = data
                        print(f"  Loaded {len(data)} labels from {orientation_path.name}")
                except Exception as e:
                    print(f"  Error loading {orientation_path}: {e}")
                    labels[str(img_path)] = []
            else:
                labels[str(img_path)] = []
        
        return labels
    
    def filter_unlabeled_pairs(self) -> List[Tuple[Path, Path]]:
        """Filter out image pairs where all dislocations are already labeled"""
        unlabeled_pairs = []
        
        for img_path, lbl_path in self.pairs:
            # Load image to get dimensions
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Load bboxes from label file
            try:
                label_lines = lbl_path.read_text().splitlines()
            except:
                continue
            
            h, w = img.shape
            bboxes = []
            for line in label_lines:
                if line.strip():
                    bbox = yolo_to_rect(line, w, h)
                    if bbox:
                        bboxes.append(bbox)
            
            if not bboxes:
                continue
            
            # Check if all bboxes are labeled
            image_labels = self.labels.get(str(img_path), [])
            
            unlabeled_count = 0
            for bbox in bboxes:
                is_labeled = False
                for label in image_labels:
                    if 'bbox_pixels' in label:
                        existing_bbox = label['bbox_pixels']
                        if all(abs(a - b) < 5 for a, b in zip(existing_bbox, bbox)):
                            is_labeled = True
                            break
                
                if not is_labeled:
                    unlabeled_count += 1
            
            # Only include if there are unlabeled dislocations
            if unlabeled_count > 0:
                unlabeled_pairs.append((img_path, lbl_path))
        
        return unlabeled_pairs
    
    def get_labeled_count(self) -> int:
        """Get total number of labeled dislocations"""
        return sum(len(dislocations) for dislocations in self.labels.values())
    
    def setup_ui(self):
        """Setup the GUI"""
        self.setWindowTitle("Dislocation Orientation Labeler")
        self.setGeometry(100, 100, 1600, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top info panel
        info_layout = QHBoxLayout()
        self.info_label = QLabel("Loading...")
        self.info_label.setFont(QFont("Arial", 9))
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        
        self.progress_label = QLabel("0 / 0")
        self.progress_label.setFont(QFont("Arial", 9, QFont.Bold))
        info_layout.addWidget(self.progress_label)
        main_layout.addLayout(info_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.target_count)
        main_layout.addWidget(self.progress_bar)
        
        # Two-panel layout
        panels_layout = QHBoxLayout()
        
        # Left panel - Full image
        left_panel = QVBoxLayout()
        left_label = QLabel("Full Image - All Dislocations")
        left_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(left_label)
        
        self.full_image_label = QLabel()
        self.full_image_label.setMinimumSize(700, 700)
        self.full_image_label.setAlignment(Qt.AlignCenter)
        self.full_image_label.setStyleSheet("border: 2px solid #555; background-color: #222;")
        left_panel.addWidget(self.full_image_label)
        panels_layout.addLayout(left_panel, 1)
        
        # Right panel - Zoomed patch
        right_panel = QVBoxLayout()
        right_label = QLabel("Left Click: Dark Lobe | Right Click: Bright Lobe")
        right_label.setFont(QFont("Arial", 10, QFont.Bold))
        right_label.setAlignment(Qt.AlignCenter)
        self.instruction_label = right_label
        right_panel.addWidget(right_label)
        
        self.patch_label = ImageLabel()
        self.patch_label.setMinimumSize(700, 700)
        self.patch_label.setStyleSheet("border: 2px solid #555; background-color: #222;")
        self.patch_label.clicked.connect(self.on_patch_click)
        right_panel.addWidget(self.patch_label)
        panels_layout.addLayout(right_panel, 1)
        
        main_layout.addLayout(panels_layout)
        
        # Button panel
        button_layout = QHBoxLayout()
        
        self.skip_button = QPushButton("Skip (S)")
        self.skip_button.setFont(QFont("Arial", 10))
        self.skip_button.setMinimumHeight(40)
        self.skip_button.clicked.connect(self.skip_dislocation)
        self.skip_button.setShortcut("S")
        self.skip_button.setAutoDefault(False)  # CRITICAL: Prevent Enter from triggering
        self.skip_button.setDefault(False)
        button_layout.addWidget(self.skip_button)
        
        self.undo_button = QPushButton("Undo Last Click (U)")
        self.undo_button.setFont(QFont("Arial", 10))
        self.undo_button.setMinimumHeight(40)
        self.undo_button.clicked.connect(self.undo_click)
        self.undo_button.setShortcut("U")
        self.undo_button.setAutoDefault(False)  # CRITICAL: Prevent Enter from triggering
        self.undo_button.setDefault(False)
        self.undo_button.setEnabled(False)
        button_layout.addWidget(self.undo_button)
        
        button_layout.addStretch()
        
        self.quit_button = QPushButton("Save and Quit (Q)")
        self.quit_button.setFont(QFont("Arial", 10))
        self.quit_button.setMinimumHeight(40)
        self.quit_button.clicked.connect(self.quit_application)
        self.quit_button.setShortcut("Q")
        self.quit_button.setAutoDefault(False)  # CRITICAL: Prevent Enter from triggering
        self.quit_button.setDefault(False)
        button_layout.addWidget(self.quit_button)
        
        main_layout.addLayout(button_layout)
        
        self.update_progress()
    
    def load_next_image(self):
        """Load the next image that needs labeling"""
        while self.current_pair_idx < len(self.pairs):
            img_path, lbl_path = self.pairs[self.current_pair_idx]
            
            # Load image
            self.current_image_path = img_path
            self.current_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if self.current_image is None:
                print(f"Could not load image: {img_path}")
                self.current_pair_idx += 1
                continue
            
            # Load label file
            try:
                self.current_label_lines = lbl_path.read_text().splitlines()
            except Exception as e:
                print(f"Could not load labels: {lbl_path}: {e}")
                self.current_pair_idx += 1
                continue
            
            # Parse all bboxes
            h, w = self.current_image.shape
            self.current_bboxes = []
            for line in self.current_label_lines:
                if line.strip():
                    bbox = yolo_to_rect(line, w, h)
                    if bbox:
                        self.current_bboxes.append(bbox)
            
            if not self.current_bboxes:
                self.current_pair_idx += 1
                continue
            
            # Find first unlabeled dislocation
            self.current_bbox_idx = 0
            if self.find_next_unlabeled_dislocation():
                self.display_current_dislocation()
                return
            
            # All dislocations in this image are labeled
            self.current_pair_idx += 1
        
        # No more images
        self.show_completion_message()
    
    def find_next_unlabeled_dislocation(self) -> bool:
        """Find next unlabeled dislocation in current image"""
        image_labels = self.labels.get(str(self.current_image_path), [])
        
        while self.current_bbox_idx < len(self.current_bboxes):
            bbox = self.current_bboxes[self.current_bbox_idx]
            
            # Check if already labeled
            is_labeled = False
            for label in image_labels:
                if 'bbox_pixels' in label:
                    existing_bbox = label['bbox_pixels']
                    if all(abs(a - b) < 5 for a, b in zip(existing_bbox, bbox)):
                        is_labeled = True
                        break
            
            if not is_labeled:
                self.current_bbox = bbox
                return True
            
            self.current_bbox_idx += 1
        
        return False
    
    def display_current_dislocation(self):
        """Display current dislocation in both panels"""
        if self.current_image is None or self.current_bbox is None:
            return
        
        x0, y0, x1, y1 = self.current_bbox
        h, w = self.current_image.shape
        
        # Update info label
        self.info_label.setText(
            f"Image: {self.current_image_path.stem} | "
            f"Dislocation: {self.current_bbox_idx + 1} / {len(self.current_bboxes)}"
        )
        
        # Display full image with rectangles
        full_display = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
        
        # Draw current bbox in bright cyan
        cv2.rectangle(full_display, (x0, y0), (x1, y1), (0, 255, 255), 2)
        
        # Draw all other bboxes in dim gray
        for idx, other_bbox in enumerate(self.current_bboxes):
            if idx == self.current_bbox_idx:
                continue
            ox0, oy0, ox1, oy1 = other_bbox
            cv2.rectangle(full_display, (ox0, oy0), (ox1, oy1), (100, 100, 100), 1)
        
        # Convert to QPixmap
        height, width, channel = full_display.shape
        bytes_per_line = 3 * width
        q_image = QImage(full_display.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.full_image_label.size(), 
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.full_image_label.setPixmap(scaled_pixmap)
        
        # Extract and display zoomed patch
        pad = 20
        x0_pad = max(0, x0 - pad)
        y0_pad = max(0, y0 - pad)
        x1_pad = min(w, x1 + pad)
        y1_pad = min(h, y1 + pad)
        
        self.patch_offset = (x0_pad, y0_pad)
        self.patch = self.current_image[y0_pad:y1_pad, x0_pad:x1_pad].copy()
        self.patch_display = cv2.cvtColor(self.patch, cv2.COLOR_GRAY2RGB)
        
        # Create smoothed version for peak finding
        self.patch_smoothed = cv2.GaussianBlur(self.patch, (7, 7), 1.5)
        
        # Draw bbox within patch
        bbox_in_patch = (x0 - x0_pad, y0 - y0_pad, x1 - x0_pad, y1 - y0_pad)
        cv2.rectangle(self.patch_display, 
                     (bbox_in_patch[0], bbox_in_patch[1]),
                     (bbox_in_patch[2], bbox_in_patch[3]),
                     (0, 255, 255), 1)
        
        # Reset clicks
        self.current_clicks = []
        self.undo_button.setEnabled(False)
        self.instruction_label.setText("Left Click: Dark Lobe | Right Click: Bright Lobe")
        
        self.update_patch_display()
        
        # Set focus to patch label so Enter key works
        self.patch_label.setFocus()
    
    def find_peak(self, x: int, y: int, find_maximum: bool) -> Tuple[int, int]:
        """
        Find local peak (max or min) near the clicked position
        Uses the smoothed patch for peak finding
        """
        if self.patch_smoothed is None:
            return (x, y)
        
        h, w = self.patch_smoothed.shape
        search_rad = self.peak_search_radius
        
        # Define search window
        x_min = max(0, x - search_rad)
        x_max = min(w, x + search_rad + 1)
        y_min = max(0, y - search_rad)
        y_max = min(h, y + search_rad + 1)
        
        # Extract region
        region = self.patch_smoothed[y_min:y_max, x_min:x_max]
        
        if region.size == 0:
            return (x, y)
        
        # Find peak
        if find_maximum:
            # Find brightest point
            peak_idx = np.argmax(region)
        else:
            # Find darkest point
            peak_idx = np.argmin(region)
        
        # Convert back to patch coordinates
        peak_y, peak_x = np.unravel_index(peak_idx, region.shape)
        peak_x += x_min
        peak_y += y_min
        
        return (peak_x, peak_y)
    
    def update_patch_display(self):
        """Update the patch display with current clicks"""
        display = self.patch_display.copy()
        
        # Draw clicks
        for idx, (x, y) in enumerate(self.current_clicks):
            color = (255, 0, 0) if idx == 0 else (0, 255, 255)
            cv2.circle(display, (x, y), self.marker_radius, color, -1)
            cv2.circle(display, (x, y), self.marker_radius + 1, (255, 255, 255), 1)
        
        # If both clicks done, show ready message
        if len(self.current_clicks) == 2:
            self.instruction_label.setText("Press ENTER to Save & Continue")
            self.instruction_label.setStyleSheet("color: #00FF00;")  # Green to indicate ready
        else:
            self.instruction_label.setStyleSheet("")  # Reset color
        
        # Convert to QPixmap
        height, width, channel = display.shape
        bytes_per_line = 3 * width
        q_image = QImage(display.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.patch_label.size(),
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.patch_label.setPixmap(scaled_pixmap)
        
        self.scale_factor = scaled_pixmap.width() / width if width > 0 else 1.0
    
    def on_patch_click(self, x: int, y: int, use_peak_finding: bool):
        """Handle click on zoomed patch"""
        if len(self.current_clicks) >= 2:
            return
        
        x_patch = int(x / self.scale_factor)
        y_patch = int(y / self.scale_factor)
        
        if 0 <= x_patch < self.patch.shape[1] and 0 <= y_patch < self.patch.shape[0]:
            # Apply peak finding if left click
            if use_peak_finding:
                # First click: find dark (minimum), second click: find bright (maximum)
                find_maximum = (len(self.current_clicks) == 1)
                x_patch, y_patch = self.find_peak(x_patch, y_patch, find_maximum)
            
            self.current_clicks.append((x_patch, y_patch))
            
            if len(self.current_clicks) == 1:
                self.instruction_label.setText("Now Click Bright Lobe (Left=auto, Right=exact)")
            
            self.undo_button.setEnabled(True)
            self.update_patch_display()
    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        key = event.key()
        
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if len(self.current_clicks) == 2:
                print("[DEBUG] Enter pressed, calling accept_labels()")
                self.accept_labels()
                event.accept()
                return
        elif key == Qt.Key_U:
            self.undo_click()
            event.accept()
            return
        elif key == Qt.Key_S:
            self.skip_dislocation()
            event.accept()
            return
        elif key == Qt.Key_Q:
            self.quit_application()
            event.accept()
            return
        
        event.ignore()
    
    def undo_click(self):
        """Undo last click"""
        if self.current_clicks:
            self.current_clicks.pop()
            self.undo_button.setEnabled(len(self.current_clicks) > 0)
            
            if len(self.current_clicks) == 0:
                self.instruction_label.setText("Left Click: Dark Lobe | Right Click: Bright Lobe")
            elif len(self.current_clicks) == 1:
                self.instruction_label.setText("Now Click Bright Lobe (Left=auto, Right=exact)")
            
            self.update_patch_display()
    
    def accept_labels(self):
        """Accept current labels and save immediately (atomically)"""
        if len(self.current_clicks) != 2:
            print("[DEBUG] accept_labels called but don't have 2 clicks")
            return
        
        print("[DEBUG] accept_labels: processing 2 clicks...")
        
        # Convert patch coordinates to global
        dark_x = self.current_clicks[0][0] + self.patch_offset[0]
        dark_y = self.current_clicks[0][1] + self.patch_offset[1]
        bright_x = self.current_clicks[1][0] + self.patch_offset[0]
        bright_y = self.current_clicks[1][1] + self.patch_offset[1]
        
        # Calculate orientation
        dx = bright_x - dark_x
        dy = bright_y - dark_y
        orientation = (np.degrees(np.arctan2(dy, dx)) + 90) % 180
        
        # Create label
        h, w = self.current_image.shape
        x0, y0, x1, y1 = self.current_bbox
        
        label = {
            'bbox_pixels': list(self.current_bbox),
            'bbox_normalized': {
                'x_center': ((x0 + x1) / 2) / w,
                'y_center': ((y0 + y1) / 2) / h,
                'width': (x1 - x0) / w,
                'height': (y1 - y0) / h
            },
            'dark_lobe': {'x': float(dark_x), 'y': float(dark_y)},
            'bright_lobe': {'x': float(bright_x), 'y': float(bright_y)},
            'dark_lobe_normalized': {'x': float(dark_x) / w, 'y': float(dark_y) / h},
            'bright_lobe_normalized': {'x': float(bright_x) / w, 'y': float(bright_y) / h},
            'orientation_deg': float(orientation),
            'bbox_index': int(self.current_bbox_idx)
        }
        
        # Add to labels (in-memory)
        image_key = str(self.current_image_path)
        if image_key not in self.labels:
            self.labels[image_key] = []
        self.labels[image_key].append(label)
        
        print(f"[DEBUG] Added label to memory, now saving...")
        
        # Save to disk (atomic)
        if self.save_current_image_labels():
            print(f"✓ Saved: {self.current_image_path.name} - "
                  f"Dislocation {self.current_bbox_idx + 1} - "
                  f"Orientation: {orientation:.1f}°")
        else:
            print("✗ Save failed (see error above).")
        
        self.update_progress()
        
        # Check if target reached
        if self.get_labeled_count() >= self.target_count:
            self.show_completion_message()
            return
        
        # Load next dislocation
        self.current_bbox_idx += 1
        if self.find_next_unlabeled_dislocation():
            self.display_current_dislocation()
        else:
            self.current_pair_idx += 1
            self.load_next_image()
    
    def skip_dislocation(self):
        """Skip current dislocation (no label added)"""
        print(f"Skipped: {self.current_image_path.name} - Dislocation {self.current_bbox_idx + 1}")
        self.current_bbox_idx += 1
        if self.find_next_unlabeled_dislocation():
            self.display_current_dislocation()
        else:
            self.current_pair_idx += 1
            self.load_next_image()
    
    def save_current_image_labels(self) -> bool:
        """Save labels for current image atomically - returns True if successful"""
        orientation_path = self.get_orientation_path(self.current_image_path)
        if not orientation_path:
            print(f"ERROR: Could not create orientation path for {self.current_image_path}")
            return False
        
        # Get labels to save (only for current image)
        image_key = str(self.current_image_path)
        labels_to_save = self.labels.get(image_key, [])
        
        print(f"[DEBUG] Attempting to save {len(labels_to_save)} labels to {orientation_path}")
        
        return _atomic_json_write(orientation_path, labels_to_save)
    
    def update_progress(self):
        """Update progress display"""
        labeled = self.get_labeled_count()
        self.progress_label.setText(f"{labeled} / {self.target_count}")
        self.progress_bar.setValue(labeled)
    
    def show_completion_message(self):
        """Show completion message"""
        labeled = self.get_labeled_count()
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Labeling Complete!")
        msg.setInformativeText(
            f"You have labeled {labeled} dislocations.\n\n"
            f"Orientation data saved in '{self.orientation_root}' folder.\n\n"
            f"Ready to train your model!"
        )
        msg.setWindowTitle("Complete")
        msg.exec()
        self.close()
    
    def quit_application(self):
        """Save current image's labels again, then quit."""
        # Best-effort extra save for current image
        if self.current_image_path:
            ok = self.save_current_image_labels()
            if ok:
                print(f"[Exit] Saved current image labels to disk.")
            else:
                print(f"[Exit] WARNING: failed to save current image labels (see error above).")
        
        labeled = self.get_labeled_count()
        print(f"\nSession complete: {labeled} dislocations labeled.")
        self.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive PySide6 GUI for labeling dislocation orientations')
    parser.add_argument('--root_path', type=str, required=True,
                       help='Root path containing images/ and labels/ folders')
    parser.add_argument('--target_count', type=int, default=200,
                       help='Target number of dislocations to label')
    parser.add_argument('--marker_radius', type=int, default=1,
                       help='Radius of click markers in pixels')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
    app.setPalette(palette)
    
    window = OrientationLabelerGUI(Path(args.root_path), args.target_count, args.marker_radius)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
