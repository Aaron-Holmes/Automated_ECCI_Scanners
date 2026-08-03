#!/usr/bin/env python3

import os
import sys
import time
import gc
import csv
import logging
import ast
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, label
from skimage.restoration import denoise_wavelet, estimate_sigma
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# PySide6 / Qt imports
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QPainter, QMouseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QListWidget,
    QVBoxLayout, QWidget, QMessageBox, QLabel, QDialog, QHBoxLayout,
    QCheckBox, QProgressBar, QSlider, QFormLayout, QListWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QSpinBox, QComboBox, QDoubleSpinBox, QGroupBox, QDialogButtonBox,
    QGridLayout
)
from ultralytics import YOLO
import poisson_disc
from poisson_disc import Bridson_sampling
import math
from scipy.optimize import curve_fit

try:
    from rtree import index
    HAVE_RTREE = True
except ImportError:
    HAVE_RTREE = False

# Dislocation removal imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pandas as pd
    from PIL import Image
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print("Warning: PyTorch not available - dislocation removal will be disabled")

# Global configuration

# Model path for best dislocation locator model
YOLO_MODEL_PATH = "models/method_3/best.pt"  # Update this to your actual path if needed

# Default path for dislocation removal model
DISLOCATION_REMOVAL_MODEL_PATH = "autoencoder_dislocation_removal/best_model.pth" # Update this to your actual path

global_conf_threshold = 0.025
global_iou_threshold = 0.3

# Class Definitions
p3_pole_class = 0
p4_pole_class = 1
dislocation_class = 2
faint_dislocation_class = 3
multiple_dislocations_class = 4
no_dislocation_class = 5
overlapping_class = 8
ultra_faint_dislocation_class = 6
unknown_class = 7

scan_box_size = 50  # default – can be overridden
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
global_model_verbose = 0

classification_label_map = {
    p3_pole_class: 'p3_pole',
    p4_pole_class: 'p4_pole',
    dislocation_class: 'dislocation',
    faint_dislocation_class: 'faint',
    multiple_dislocations_class: 'multiple',
    no_dislocation_class: 'no_dislocation',
    overlapping_class: 'overlapping',
    ultra_faint_dislocation_class: 'ultra_faint',
    unknown_class: 'unknown'
}

classification_color_map = {
    'dislocation': 'red',
    'faint': 'purple',
    'multiple': 'blue',
    'p3_pole': 'magenta',
    'p4_pole': 'cyan',
    'no_dislocation': 'brown',
    'overlapping': 'yellow',
    'ultra_faint': 'gray',
    'unknown': 'white'
}

CLASS_INFO = {
    p3_pole_class: {'name': 'p3_pole_class', 'color': 'magenta', 'draw_patch': True},
    p4_pole_class: {'name': 'p4_pole_class', 'color': 'cyan', 'draw_patch': True},
    dislocation_class: {'name': 'dislocation_class', 'color': 'red', 'draw_patch': True},
    multiple_dislocations_class: {'name': 'multiple_dislocations_class', 'color': 'blue', 'draw_patch': True},
    no_dislocation_class: {'name': 'no_dislocation_class', 'color': 'brown', 'draw_patch': False},
    overlapping_class: {'name': 'overlapping_class', 'color': 'yellow', 'draw_patch': True},
    faint_dislocation_class: {'name': 'faint_dislocation_class', 'color': 'purple', 'draw_patch': True},
    ultra_faint_dislocation_class: {'name': 'ultra_faint_dislocation_class', 'color': 'gray', 'draw_patch': True},
    unknown_class: {'name': 'unknown_class', 'color': 'white', 'draw_patch': True}
}

# Headless preprocessing, used when "Apply to All" settings are saved
def apply_headless_preprocessing(base_array, settings):
    """
    Applies crop and denoise settings to an image array without a GUI.
    """
    h_orig, w_orig = base_array.shape
    t, b, l, r = settings['crop_t'], settings['crop_b'], settings['crop_l'], settings['crop_r']

    # Safety check: ensure crop doesn't exceed image dimensions
    if t + b >= h_orig or l + r >= w_orig:
        print("Warning: Crop settings exceed image dimensions. Skipping crop.")
        temp = base_array.copy()
    else:
        end_y = max(t + 1, h_orig - b)
        end_x = max(l + 1, w_orig - r)
        temp = base_array[t:end_y, l:end_x].copy()

    # Apply Denoise
    mode = settings['blur_mode']
    p1 = settings['kernel_size']
    p2 = settings['sigma']
    k_odd = p1 if p1 % 2 != 0 else p1 + 1

    if "Gaussian" in mode:
        temp = cv2.GaussianBlur(temp, (k_odd, k_odd), p2)
    elif "Median" in mode:
        t_8 = np.clip(temp, 0, 255).astype(np.uint8)
        t_8 = cv2.medianBlur(t_8, k_odd)
        temp = t_8.astype(np.float32)
    elif "Non-Local Means" in mode:
        t_8 = np.clip(temp, 0, 255).astype(np.uint8)
        t_8 = cv2.fastNlMeansDenoising(t_8, None, p2, 7, k_odd)
        temp = t_8.astype(np.float32)
    elif "Wavelet" in mode:
        try:
            temp = denoise_wavelet(temp, sigma=p2, rescale_sigma=True, channel_axis=None)
            if temp.max() <= 1.05: temp *= 255.0
        except: pass

    return temp

# Image editor dialog (crop/denoise preprocessing, apply-to-all option)
class ImageEditorDialog(QDialog):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Manual Preprocessing: {os.path.basename(file_path)}")
        self.resize(1200, 850)

        # Load image data
        img = load_img(file_path, color_mode='grayscale')
        raw_array = np.squeeze(img_to_array(img)).astype(np.float32)

        # Normalize to 0-255 immediately to fix visibility issues
        if raw_array.max() > raw_array.min():
            self.base_array = (raw_array - raw_array.min()) / (raw_array.max() - raw_array.min()) * 255.0
        else:
            self.base_array = raw_array

        self.h_orig, self.w_orig = self.base_array.shape

        # Internal state
        self.displayed_array = self.base_array.copy()

        self.setup_ui()
        self.apply_pipeline()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        # Left: graphics view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        layout.addWidget(self.view, stretch=3)

        # Right: controls
        controls_layout = QVBoxLayout()

        # Crop sliders
        crop_group = QGroupBox("Edge Cropping")
        crop_lay = QGridLayout()

        # Limit crop to 45% per side to prevent errors
        max_w = int(self.w_orig * 0.45)
        max_h = int(self.h_orig * 0.45)

        self.slider_top = self._create_slider("Top:", max_h, crop_lay, 0)
        self.slider_bottom = self._create_slider("Bottom:", max_h, crop_lay, 1)
        self.slider_left = self._create_slider("Left:", max_w, crop_lay, 2)
        self.slider_right = self._create_slider("Right:", max_w, crop_lay, 3)

        crop_group.setLayout(crop_lay)
        controls_layout.addWidget(crop_group)

        # Denoise controls
        blur_group = QGroupBox("Denoising")
        blur_lay = QFormLayout()

        self.combo_blur = QComboBox()
        self.combo_blur.addItems(["None", "Gaussian Blur", "Median Blur", "Non-Local Means", "Wavelet Denoising"])
        self.combo_blur.currentIndexChanged.connect(self.apply_pipeline)

        self.spin_p1 = QSpinBox()
        self.spin_p1.setRange(1, 50); self.spin_p1.setValue(3)
        self.spin_p1.valueChanged.connect(self.apply_pipeline)
        self.lbl_p1 = QLabel("Kernel Size:")

        self.spin_p2 = QDoubleSpinBox()
        self.spin_p2.setRange(0.1, 200.0); self.spin_p2.setValue(3.0); self.spin_p2.setSingleStep(0.5)
        self.spin_p2.valueChanged.connect(self.apply_pipeline)
        self.lbl_p2 = QLabel("Strength:")

        blur_lay.addRow("Method:", self.combo_blur)
        blur_lay.addRow(self.lbl_p1, self.spin_p1)
        blur_lay.addRow(self.lbl_p2, self.spin_p2)

        self.lbl_hint = QLabel("Select a method.")
        self.lbl_hint.setStyleSheet("color: gray; font-style: italic;")
        blur_lay.addRow(self.lbl_hint)

        blur_group.setLayout(blur_lay)
        controls_layout.addWidget(blur_group)

        controls_layout.addStretch()

        self.chk_apply_all = QCheckBox("Apply these settings to all remaining images")
        self.chk_apply_all.setStyleSheet("font-weight: bold; color: blue;")
        controls_layout.addWidget(self.chk_apply_all)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        controls_layout.addWidget(self.buttons)

        layout.addLayout(controls_layout, stretch=1)

    def _create_slider(self, label_text, max_val, layout, row):
        lbl = QLabel(label_text)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, max_val)
        spin = QSpinBox()
        spin.setRange(0, max_val); spin.setSuffix(" px")

        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(self.apply_pipeline)

        layout.addWidget(lbl, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(spin, row, 2)
        return slider

    def apply_pipeline(self):
        # Crop
        t, b = self.slider_top.value(), self.slider_bottom.value()
        l, r = self.slider_left.value(), self.slider_right.value()

        end_y = max(t + 1, self.h_orig - b)
        end_x = max(l + 1, self.w_orig - r)

        temp = self.base_array[t:end_y, l:end_x].copy()

        # Denoise
        mode = self.combo_blur.currentText()
        p1, p2 = int(self.spin_p1.value()), float(self.spin_p2.value())
        k_odd = p1 if p1 % 2 != 0 else p1 + 1

        if "Gaussian" in mode:
            self.lbl_p1.setVisible(True); self.lbl_p2.setVisible(True)
            self.lbl_hint.setText("Standard blur.")
            temp = cv2.GaussianBlur(temp, (k_odd, k_odd), p2)
        elif "Median" in mode:
            self.lbl_p1.setVisible(True); self.lbl_p2.setVisible(False)
            self.lbl_hint.setText("Removes salt & pepper noise.")
            t_8 = np.clip(temp, 0, 255).astype(np.uint8)
            t_8 = cv2.medianBlur(t_8, k_odd)
            temp = t_8.astype(np.float32)
        elif "Non-Local Means" in mode:
            self.lbl_p1.setVisible(True); self.lbl_p2.setVisible(True)
            self.lbl_hint.setText("Best quality. Slow.")
            t_8 = np.clip(temp, 0, 255).astype(np.uint8)
            t_8 = cv2.fastNlMeansDenoising(t_8, None, p2, 7, k_odd)
            temp = t_8.astype(np.float32)
        elif "Wavelet" in mode:
            self.lbl_p1.setVisible(False); self.lbl_p2.setVisible(True)
            self.lbl_hint.setText("Fine detail.")
            try:
                temp = denoise_wavelet(temp, sigma=p2, rescale_sigma=True, channel_axis=None)
                if temp.max() <= 1.05: temp *= 255.0
            except: pass
        else:
            self.lbl_p1.setVisible(False); self.lbl_p2.setVisible(False)
            self.lbl_hint.setText("No filter.")

        self.displayed_array = temp
        self.update_display()

    def update_display(self):
        disp = np.clip(self.displayed_array, 0, 255).astype(np.uint8)
        disp = np.require(disp, np.uint8, 'C') # Force contiguous memory
        h, w = disp.shape
        qimg = QImage(disp.data, w, h, w, QImage.Format_Grayscale8)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def get_final_image(self):
        return self.displayed_array

    def get_settings(self):
        return {
            'crop_t': self.slider_top.value(),
            'crop_b': self.slider_bottom.value(),
            'crop_l': self.slider_left.value(),
            'crop_r': self.slider_right.value(),
            'blur_mode': self.combo_blur.currentText(),
            'kernel_size': int(self.spin_p1.value()),
            'sigma': float(self.spin_p2.value()),
            'apply_all': self.chk_apply_all.isChecked()
        }

# Result filter dialog: prune detection artifacts by brightness/contrast
class ResultFilterDialog(QDialog):
    def __init__(self, image_data, locations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Detection Results")
        self.resize(1200, 800)

        self.image_data = image_data
        self.locations = locations # list of ([xs], [ys])
        self.valid_indices = set(range(len(locations))) # All valid by default

        self.box_metrics = [] # Stores (min, max, contrast) for each box
        self.box_items = []

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        self.setup_ui()

        # Calculate initial metrics (using default 1x1 kernel)
        self._calculate_metrics(kernel_size=1)
        self.draw_initial_scene()

    def _calculate_metrics(self, kernel_size=1):
        # Reset metrics
        self.box_metrics = []

        # Ensure image is 0-255
        img = self.image_data.astype(np.float32)
        if img.max() <= 1.5: img *= 255.0

        h, w = img.shape

        for i, (xs, ys) in enumerate(self.locations):
            # Get bounding box
            x0, x1 = int(min(xs)), int(max(xs))
            y0, y1 = int(min(ys)), int(max(ys))

            # Clip to image bounds
            x0, x1 = max(0, x0), min(w, x1)
            y0, y1 = max(0, y0), min(h, y1)

            if x1 > x0 and y1 > y0:
                roi = img[y0:y1, x0:x1]

                # Apply smoothing if kernel > 1 (avoids single pixel noise)
                if kernel_size > 1:
                    roi = cv2.blur(roi, (kernel_size, kernel_size))

                mx = np.max(roi)
                mn = np.min(roi)
                contrast = mx - mn
                self.box_metrics.append((mn, mx, contrast))
            else:
                self.box_metrics.append((0, 0, 0))

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.addWidget(self.view, stretch=3)

        controls = QVBoxLayout()
        grp = QGroupBox("Artifact Rejection Filters")
        form = QFormLayout()

        # Kernel dropdown
        self.combo_kernel = QComboBox()
        self.combo_kernel.addItems(["1x1 (None)", "2x2 Average", "3x3 Average", "5x5 Average"])
        self.combo_kernel.currentIndexChanged.connect(self.on_kernel_changed)
        form.addRow("Metric Smoothing:", self.combo_kernel)
        form.addRow(QLabel("<i>(Averages pixels before checking min/max)</i>"), QLabel(""))

        # Min brightness slider
        self.slider_min_bright = QSlider(Qt.Horizontal)
        self.slider_min_bright.setRange(0, 255)
        self.slider_min_bright.setValue(0) # Default: Allow dark
        self.spin_min_bright = QSpinBox()
        self.spin_min_bright.setRange(0, 255); self.spin_min_bright.setValue(0)

        self.slider_min_bright.valueChanged.connect(self.spin_min_bright.setValue)
        self.spin_min_bright.valueChanged.connect(self.slider_min_bright.setValue)
        self.slider_min_bright.valueChanged.connect(self.update_visibility)

        # Max brightness slider
        self.slider_max_bright = QSlider(Qt.Horizontal)
        self.slider_max_bright.setRange(0, 255)
        self.slider_max_bright.setValue(255) # Default: Allow bright
        self.spin_max_bright = QSpinBox()
        self.spin_max_bright.setRange(0, 255); self.spin_max_bright.setValue(255)

        self.slider_max_bright.valueChanged.connect(self.spin_max_bright.setValue)
        self.spin_max_bright.valueChanged.connect(self.slider_max_bright.setValue)
        self.slider_max_bright.valueChanged.connect(self.update_visibility)

        # Min contrast slider
        self.slider_min_cont = QSlider(Qt.Horizontal)
        self.slider_min_cont.setRange(0, 255)
        self.slider_min_cont.setValue(0) # Default: Allow low contrast
        self.spin_min_cont = QSpinBox()
        self.spin_min_cont.setRange(0, 255); self.spin_min_cont.setValue(0)

        self.slider_min_cont.valueChanged.connect(self.spin_min_cont.setValue)
        self.spin_min_cont.valueChanged.connect(self.slider_min_cont.setValue)
        self.slider_min_cont.valueChanged.connect(self.update_visibility)

        # Max contrast slider
        self.slider_max_cont = QSlider(Qt.Horizontal)
        self.slider_max_cont.setRange(0, 255)
        self.slider_max_cont.setValue(255) # Default: Allow high contrast
        self.spin_max_cont = QSpinBox()
        self.spin_max_cont.setRange(0, 255); self.spin_max_cont.setValue(255)

        self.slider_max_cont.valueChanged.connect(self.spin_max_cont.setValue)
        self.spin_max_cont.valueChanged.connect(self.slider_max_cont.setValue)
        self.slider_max_cont.valueChanged.connect(self.update_visibility)

        form.addRow(QLabel("<b>Min Brightness:</b><br>(Removes deep black artifacts)"), self.slider_min_bright)
        form.addRow("", self.spin_min_bright)
        form.addRow(QLabel("<b>Max Brightness:</b><br>(Removes white spots)"), self.slider_max_bright)
        form.addRow("", self.spin_max_bright)

        form.addRow(QLabel("<b>Min Contrast:</b><br>(Removes faint objects)"), self.slider_min_cont)
        form.addRow("", self.spin_min_cont)
        form.addRow(QLabel("<b>Max Contrast:</b><br>(Removes harsh shadows)"), self.slider_max_cont)
        form.addRow("", self.spin_max_cont)

        grp.setLayout(form)
        controls.addWidget(grp)

        # Info Label
        self.lbl_stats = QLabel(f"Showing: {len(self.locations)} / {len(self.locations)}")
        controls.addWidget(self.lbl_stats)

        controls.addStretch()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        controls.addWidget(btns)

        layout.addLayout(controls, stretch=1)

    def draw_initial_scene(self):
        # Draw Image
        disp = np.clip(self.image_data, 0, 255).astype(np.uint8)
        # Fix memory alignment for QImage
        disp = np.require(disp, np.uint8, 'C')
        h, w = disp.shape
        qimg = QImage(disp.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.scene.addPixmap(pix)
        self.scene.setSceneRect(0, 0, w, h)

        # Draw Boxes
        pen = QPen(Qt.green, 2)
        pen.setCosmetic(True) # Keeps line width constant when zooming

        for i, (xs, ys) in enumerate(self.locations):
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            rect = QGraphicsRectItem(QRectF(x0, y0, x1-x0, y1-y0))
            rect.setPen(pen)
            self.scene.addItem(rect)
            self.box_items.append(rect)

        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def on_kernel_changed(self):
        txt = self.combo_kernel.currentText()
        if "2x2" in txt: k = 2
        elif "3x3" in txt: k = 3
        elif "5x5" in txt: k = 5
        else: k = 1

        # Recalculate metrics with new kernel
        self._calculate_metrics(kernel_size=k)
        # Update visibility based on new metrics
        self.update_visibility()

    def update_visibility(self):
        min_b = self.slider_min_bright.value()
        max_b = self.slider_max_bright.value()
        min_c = self.slider_min_cont.value()
        max_c = self.slider_max_cont.value()

        visible_count = 0
        self.valid_indices = set()

        for i, (mn_metric, mx_metric, c_metric) in enumerate(self.box_metrics):
            # Check Brightness Range
            # "Min Brightness" checks the MIN pixel value of the box
            if mn_metric < min_b:
                self.box_items[i].setVisible(False)
                continue

            # "Max Brightness" checks the MAX pixel value of the box
            if mx_metric > max_b:
                self.box_items[i].setVisible(False)
                continue

            # Check Contrast Range
            if not (min_c <= c_metric <= max_c):
                self.box_items[i].setVisible(False)
                continue

            self.box_items[i].setVisible(True)
            self.valid_indices.add(i)
            visible_count += 1

        self.lbl_stats.setText(f"Showing: {visible_count} / {len(self.locations)}")

    def get_filtered_indices(self):
        # Return a sorted list of indices that survived the filter
        return sorted(list(self.valid_indices))

# Dislocation removal (autoencoder-based)

if HAVE_TORCH:
    class DoubleConv(nn.Module):
        """Double convolution block used in U-Net"""
        def __init__(self, in_channels, out_channels):
            super(DoubleConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.conv(x)

    class UNetInpainter(nn.Module):
        def __init__(self, in_channels=2, out_channels=1, features=[64, 128, 256, 512]):
            super(UNetInpainter, self).__init__()
            self.ups = nn.ModuleList(); self.downs = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            for feature in features:
                self.downs.append(DoubleConv(in_channels, feature))
                in_channels = feature
            for feature in reversed(features):
                self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
                self.ups.append(DoubleConv(feature*2, feature))
            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        def forward(self, x):
            skip_connections = []
            for down in self.downs:
                x = down(x); skip_connections.append(x); x = self.pool(x)
            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]
                if x.shape != skip_connection.shape:
                    x = F.interpolate(x, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_skip)
            return torch.sigmoid(self.final_conv(x))

    class BasicDislocationProcessor:
        def __init__(self, model, device, tile_size: int = 512, overlap: int = 128, context_expand_pct: float = 0.75):
            self.model = model; self.device = device; self.tile_size = tile_size; self.overlap = overlap
            self.context_expand_pct = context_expand_pct

        def _parse_coordinates(self, rect_str: str):
            try:
                coords = ast.literal_eval(rect_str)
                if (len(coords) == 2 and isinstance(coords[0], (list, tuple))):
                    xs, ys = coords; return list(zip(xs, ys))
                elif all(isinstance(coord, (list, tuple)) and len(coord) == 2 for coord in coords):
                    return [(float(x), float(y)) for x, y in coords]
            except Exception: pass
            return []

        def _bbox_from_coords(self, coords):
            if not coords: return 0, 0, 0, 0
            xs, ys = zip(*coords)
            return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

        def _create_mask_from_csv(self, csv_path, img_shape):
            mask = np.zeros(img_shape, dtype=np.uint8)
            try: df = pd.read_csv(csv_path)
            except: return mask
            if df.empty or 'rect' not in df.columns: return mask
            box_sizes = []
            for idx, row in df.iterrows():
                coords = self._parse_coordinates(str(row['rect']))
                if coords:
                    x0, y0, x1, y1 = self._bbox_from_coords(coords)
                    box_sizes.append(max(x1 - x0, y1 - y0))
            if not box_sizes: return mask
            avg_box_size = np.mean(box_sizes)
            expand_pixels = int(avg_box_size * self.context_expand_pct)
            h, w = img_shape
            for idx, row in df.iterrows():
                coords = self._parse_coordinates(str(row['rect']))
                if coords:
                    x0, y0, x1, y1 = self._bbox_from_coords(coords)
                    x0_exp = max(0, x0 - expand_pixels)
                    y0_exp = max(0, y0 - expand_pixels)
                    x1_exp = min(w - 1, x1 + expand_pixels)
                    y1_exp = min(h - 1, y1 + expand_pixels)
                    mask[y0_exp:y1_exp+1, x0_exp:x1_exp+1] = 255
            return mask

        def _apply_model_to_tile(self, tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
            tile_norm = (tile - tile.min()) / (np.ptp(tile) + 1e-12)
            mask_norm = mask / 255.0
            tile_tensor = torch.from_numpy(tile_norm).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask_norm).float().unsqueeze(0).unsqueeze(0)
            model_input = torch.cat([tile_tensor, mask_tensor], dim=1).to(self.device)
            with torch.no_grad(): output = self.model(model_input)
            result = output[0, 0].cpu().numpy()
            result = result * (tile.max() - tile.min()) + tile.min()
            return result.astype(tile.dtype)

        def _calculate_global_noise_sigma(self, image: np.ndarray, mask: np.ndarray) -> float:
            smoothed = cv2.GaussianBlur((image * 255).astype(np.uint8), (7, 7), 0).astype(np.float32) / 255.0
            non_mask_regions = mask == 0
            if non_mask_regions.sum() > 100:
                noise_samples = (image - smoothed)[non_mask_regions]
                sigma = max(np.std(noise_samples), 0.001)
            else: sigma = 0.001
            return sigma

        def _add_consistent_noise(self, inpainted_image: np.ndarray, original_image: np.ndarray,
                                mask: np.ndarray, global_sigma: float) -> np.ndarray:
            noise_strength = 0.75
            noise = np.random.normal(0, noise_strength * global_sigma, original_image.shape).astype(np.float32)
            result = inpainted_image.copy()
            masked_regions = mask > 0
            result[masked_regions] = np.clip(result[masked_regions] + noise[masked_regions], 0, 1)
            return result

        def _create_tile_weight_mask(self, tile_h: int, tile_w: int, fade_width: int) -> np.ndarray:
            weight_mask = np.ones((tile_h, tile_w), dtype=np.float32)
            if fade_width > 0:
                fade_width = min(fade_width, min(tile_h, tile_w) // 4)
                for i in range(min(fade_width, tile_h)):
                    weight = (i + 1) / (fade_width + 1)
                    weight_mask[i, :] *= weight; weight_mask[tile_h - 1 - i, :] *= weight
                for j in range(min(fade_width, tile_w)):
                    weight = (j + 1) / (fade_width + 1)
                    weight_mask[:, j] *= weight; weight_mask[:, tile_w - 1 - j] *= weight
            return weight_mask

        def process_image(self, image_path, csv_path, override_image=None):
            if override_image is not None: image = override_image
            else: image = np.asarray(Image.open(image_path).convert("F"), dtype=np.float32)
            image = (image - image.min()) / (np.ptp(image) + 1e-12)
            mask = self._create_mask_from_csv(csv_path, image.shape)
            if mask.sum() == 0: return image
            return self._process_with_tiles(image, mask)

        def _process_with_tiles(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            h, w = image.shape
            global_noise_sigma = self._calculate_global_noise_sigma(image, mask)
            stride = self.tile_size - self.overlap
            y_positions = list(range(0, max(1, h - self.tile_size + 1), stride))
            if y_positions[-1] + self.tile_size < h: y_positions.append(max(0, h - self.tile_size))
            x_positions = list(range(0, max(1, w - self.tile_size + 1), stride))
            if x_positions[-1] + self.tile_size < w: x_positions.append(max(0, w - self.tile_size))
            tile_weights = np.zeros((h, w), dtype=np.float32)
            blended_result = np.zeros((h, w), dtype=np.float32)

            for y in y_positions:
                for x in x_positions:
                    y_end = min(y + self.tile_size, h)
                    x_end = min(x + self.tile_size, w)
                    tile = image[y:y_end, x:x_end]; tile_mask = mask[y:y_end, x:x_end]
                    if tile_mask.sum() == 0: continue

                    inpainted_tile = self._apply_model_to_tile(tile, tile_mask)
                    clean_tile = tile.copy()
                    clean_tile[tile_mask > 0] = inpainted_tile[tile_mask > 0]
                    final_tile = self._add_consistent_noise(clean_tile, tile, tile_mask, global_noise_sigma)

                    th, tw = final_tile.shape
                    weight_mask = self._create_tile_weight_mask(th, tw, self.overlap // 4)
                    blended_result[y:y_end, x:x_end] += final_tile * weight_mask
                    tile_weights[y:y_end, x:x_end] += weight_mask

            processed_mask = tile_weights > 1e-6
            blended_result[processed_mask] /= tile_weights[processed_mask]
            blended_result[~processed_mask] = image[~processed_mask]

            inpainted_regions = mask > 0
            if inpainted_regions.sum() > 0:
                smoothed = cv2.GaussianBlur(blended_result, (5, 5), 0)
                noise = np.random.normal(0, global_noise_sigma, blended_result.shape).astype(np.float32)
                final_result = np.clip(smoothed + noise, 0, 1)
                blended_result[inpainted_regions] = final_result[inpainted_regions]
            return blended_result

    def load_dislocation_removal_model(model_path, device):
        print(f"Loading dislocation removal model from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = UNetInpainter(in_channels=2, out_channels=1)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device); model.eval()
            return model
        except Exception as e:
            print(f"Error loading dislocation removal model: {e}"); return None
else:
    class BasicDislocationProcessor:
        def __init__(self, *args, **kwargs): pass
        def process_image(self, *args, **kwargs): pass
    def load_dislocation_removal_model(*args, **kwargs): return None

# Core functions: YOLO detection, calibration, processing

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxA_area < 1e-9 or boxB_area < 1e-9: return 0.0
    return interArea / (boxA_area + boxB_area - interArea)

_yolo_model_cache = {}
def _get_yolo(model_path, device):
    key = (str(model_path), device)
    if key not in _yolo_model_cache: _yolo_model_cache[key] = YOLO(str(model_path))
    return _yolo_model_cache[key]

def _sliding_positions(L: int, win: int, step: int):
    if win >= L: return [0]
    pos = list(range(0, L - win + 1, step))
    if pos[-1] != L - win: pos.append(L - win)
    return pos

def coarse_scan_estimate(preproc_img, model_path, device=0, initial_factor=10,
                          init_window=200, step_frac=0.25, max_samples=8, verbose=True):
    model = YOLO(str(model_path))
    STEP = int(init_window * step_frac)
    h, w = preproc_img.shape
    sizes_px, xs, ys = [], [], []
    for y in range(0, max(1, h - init_window + 1), STEP):
        for x in range(0, max(1, w - init_window + 1), STEP):
            crop = preproc_img[y:y+init_window, x:x+init_window]
            res = model.predict(crop, conf=0.05, iou=0.30, device=device, verbose=False)[0]
            for bx0, by0, bx1, by1 in res.boxes.xyxy.cpu().numpy():
                sizes_px.append(max(bx1-bx0, by1-by0))
                xs.extend([bx0+x, bx1+x]); ys.extend([by0+y, by1+y])
            if len(sizes_px) >= max_samples: break
        if len(sizes_px) >= max_samples: break

    if not sizes_px: return 50, (w//2, h//2)
    mean_sz = np.mean(sizes_px)
    size_out = int(round(max(10, mean_sz * initial_factor)))
    cx, cy = (min(xs)+max(xs))//2, (min(ys)+max(ys))//2
    return size_out, (cx, cy)

def refine_scan_box_size(preproc_crop, model_path, device=0, coarse_guess_px=120,
                          verbose=True, show_tuning_plot=False,
                          conf_threshold=global_conf_threshold, iou_threshold=global_iou_threshold):
    def _count(img, box, edge_tol_px=5):
        (_1, _2, _3, _4, locs, *_rest) = process_image_yolo(
            raw_array=img, image_array=img, base_filename="GAUSS",
            scan_box_size=int(box), conf_threshold=global_conf_threshold,
            iou_threshold=global_iou_threshold, model_path=model_path,
            device=device, full_shape=img.shape, offset_xy=(0, 0), progress_bar=None)
        H, W = img.shape
        seen_boxes = []; unique_count = 0
        for (xs, ys) in locs:
            x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
            if (x_min <= edge_tol_px or y_min <= edge_tol_px or
                x_max >= W - edge_tol_px or y_max >= H - edge_tol_px): continue
            bbox = (x_min, y_min, x_max, y_max)
            is_duplicate = False
            for b in seen_boxes:
                if compute_iou(bbox, b) >= 0.5: is_duplicate = True; break
            if not is_duplicate: seen_boxes.append(bbox); unique_count += 1
        return unique_count

    sizes = []
    jump = coarse_guess_px * 0.3
    s = float(coarse_guess_px ** 0.8)
    H, W = preproc_crop.shape[:2]
    while s <= 3.0*coarse_guess_px + 0.5 and s <= max(W, H):
        sizes.append(int(round(s))); s += jump
    sizes = np.array(sizes, dtype=int)
    counts = np.array([_count(preproc_crop, sz, edge_tol_px=3) for sz in sizes], dtype=float)

    def _gauss(x, A, mu, sigma, C): return A * np.exp(-0.5*((x-mu)/sigma)**2) + C
    p0 = [counts.max() - counts.min(), sizes[counts.argmax()], max(1.0, 0.2*coarse_guess_px), counts.min()]
    bounds = ([0, sizes.min(), 1, 0], [np.inf, sizes.max(), sizes.max()-sizes.min(), np.inf])
    best_sz = sizes[counts.argmax()]
    try:
        popt, _ = curve_fit(_gauss, sizes, counts, p0=p0, bounds=bounds, maxfev=2000)
        A, mu, sigma, C = popt
        if sigma > 1: best_sz = int(round(mu))
    except RuntimeError: pass
    if verbose: print(f"[refine-gauss] sweep {sizes[0]}–{sizes[-1]} px -> chosen {best_sz} px")
    return best_sz

def process_image_yolo(*, raw_array, image_array, base_filename,
                        scan_box_size, conf_threshold, iou_threshold,
                        model_path, device, full_shape=None, offset_xy=(0, 0), progress_bar=None):
    H_full, W_full = full_shape if full_shape else raw_array.shape
    off_x, off_y = offset_xy
    WIN_SIZE = scan_box_size
    STEP = max(1, int(0.8 * WIN_SIZE))
    h, w = raw_array.shape
    ys = _sliding_positions(h, WIN_SIZE, STEP)
    xs = _sliding_positions(w, WIN_SIZE, STEP)
    total = len(ys) * len(xs)
    model = _get_yolo(model_path, device)

    dislocation_locations = []
    dislocation_classification_labels = []
    dislocation_location_status = []
    class_rectangles = {dislocation_class: []}
    patch_locations, popt_list, pixel_size_list = [], [], []

    copy_to_draw_on = image_array.copy()
    dislocation_image = np.zeros_like(raw_array, dtype=np.float32)
    background_image = image_array.copy()

    done = 0
    EDGE_TOL = int(WIN_SIZE * 0.05)

    for y in ys:
        for x in xs:
            crop = raw_array[y:y+WIN_SIZE, x:x+WIN_SIZE]
            res = model.predict(crop, conf=conf_threshold, iou=iou_threshold, device=device, verbose=False)[0]

            for bx0, by0, bx1, by1 in res.boxes.xyxy.cpu().numpy():
                bx0, by0, bx1, by1 = map(int, (bx0, by0, bx1, by1))
                crop_at_left = (x == 0); crop_at_top = (y == 0)
                crop_at_right = (x + WIN_SIZE >= w); crop_at_bottom = (y + WIN_SIZE >= h)
                near_left = bx0 < EDGE_TOL; near_top = by0 < EDGE_TOL
                near_right = bx1 > WIN_SIZE - EDGE_TOL; near_bottom = by1 > WIN_SIZE - EDGE_TOL

                if ((near_left and not crop_at_left) or (near_top and not crop_at_top) or
                    (near_right and not crop_at_right) or (near_bottom and not crop_at_bottom)):
                    continue

                gx0, gy0, gx1, gy1 = bx0 + x, by0 + y, bx1 + x, by1 + y

                dislocation_locations.append(([gx0, gx1, gx1, gx0, gx0], [gy0, gy0, gy1, gy1, gy0]))
                dislocation_classification_labels.append("dislocation")
                dislocation_location_status.append("Successful_fit")
                patch_locations.append(None); popt_list.append(None); pixel_size_list.append(None)
                cv2.rectangle(copy_to_draw_on, (gx0, gy0), (gx1, gy1), (0, 255, 0), 2)

            done += 1
            if progress_bar and total:
                progress_bar.setValue(int(done / total * 100))
                QApplication.processEvents()

    return (image_array, copy_to_draw_on, dislocation_image, background_image,
            dislocation_locations, dislocation_location_status, dislocation_classification_labels,
            [], patch_locations, popt_list, pixel_size_list, class_rectangles)

def plot_results(original_image, image_with_rectangles, cleaned_image,
                 dislocation_locations, classification_labels, file_path,
                 total_dislocations_found):
    def normalize_image_for_display(img):
        if img is None: return None
        img_float = img.astype(np.float64)
        img_min, img_max = img_float.min(), img_float.max()
        if img_max - img_min < 1e-10: return np.zeros_like(img_float)
        return (img_float - img_min) / (img_max - img_min)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    if cleaned_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        titles = ['Original Image', 'Detected Dislocations', 'Cleaned Image']
        images = [original_image, image_with_rectangles, cleaned_image]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        titles = ['Original Image', 'Detected Dislocations']
        images = [original_image, image_with_rectangles]

    if len(images) == 1: axes = [axes]
    elif len(images) == 2 and not isinstance(axes, list): axes = list(axes)

    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        if img is not None:
            display_img = normalize_image_for_display(img)
            if display_img is not None:
                im = ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
                ax.set_title(title); ax.axis('off')
                if i == 1 and dislocation_locations:
                    from matplotlib.patches import Rectangle
                    for j, (rect_coords, label) in enumerate(zip(dislocation_locations, classification_labels)):
                        xs, ys = rect_coords
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        color = classification_color_map.get(label, 'red')
                        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
    plt.suptitle(f'{base_filename} - {total_dislocations_found} dislocations detected', fontsize=14)
    plt.tight_layout()
    plt.show()

def process_specific_tiff(file_path, *, fit=True, scan_box_size=50,
                          clip_top=0, clip_bottom=0, clip_left=0, clip_right=0,
                          max_sigma=50, increment=1, max_gradient_ratio=0.2,
                          max_allowed_percentage=0.8, image_scan_progress_bar=None,
                          max_tile_area=2_500_000, overlap_margin=None,
                          test_print=False, show_results=False, override_image=None):

    def choose_grid(w, h, max_area=max_tile_area):
        min_tiles = math.ceil((w * h) / max_area)
        best = None
        search = int(math.ceil(math.sqrt(min_tiles))) + 4
        for r in range(1, search):
            c = math.ceil(min_tiles / r)
            tw, th = w / c, h / r
            if tw * th > max_area: continue
            score = (r*c, abs(tw-th), abs(r-c), r, c)
            if best is None or score < best: best = score
        if best is None: raise RuntimeError("max_tile_area too small for this image")
        return best[3], best[4]

    def make_tiles(w, h, rows, cols, margin):
        tw, th = w / cols, h / rows
        for r in range(rows):
            for c in range(cols):
                x0 = int(max(0, c*tw - margin))
                y0 = int(max(0, r*th - margin))
                x1 = int(min(w, (c+1)*tw + margin))
                y1 = int(min(h, (r+1)*th + margin))
                yield (x0, y0, x1, y1)

    print(f"Processing {file_path} with scan_box_size={scan_box_size} ...")
    start_time = time.time()
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    if override_image is not None:
        print("Using manually pre-processed image data.")
        image_np = override_image
    else:
        img = load_img(file_path, color_mode='grayscale')
        image_np = np.squeeze(img_to_array(img))

    H_raw, W_raw = image_np.shape
    H = max(1, H_raw - clip_top - clip_bottom)
    W = max(1, W_raw - clip_left - clip_right)

    if overlap_margin is None: overlap_margin = scan_box_size
    rows, cols = choose_grid(W, H, max_area=max_tile_area)

    dislocation_locations = []
    dislocation_location_status = []
    dislocation_classification_labels = []
    patch_locations, popt_list, pixel_size_list = [], [], []
    class_rectangles = {c: [] for c in CLASS_INFO.keys()}

    copy_to_draw_on = image_np.copy()

    if HAVE_RTREE: seen = index.Index()
    else: seen_boxes = []

    tile_id = 0
    for x0, y0, x1, y1 in make_tiles(W, H, rows, cols, overlap_margin):
        raw_tile = image_np[y0:y1, x0:x1]
        (_img_arr, _copy, disloc_tile, back_tile, locs, status, labels, centres,
         patches, popts, pixsizes, cls_rects) = process_image_yolo(
            raw_array=raw_tile, image_array=raw_tile, base_filename=f"{base_filename}_T{tile_id}",
            scan_box_size=scan_box_size, conf_threshold=global_conf_threshold,
            iou_threshold=global_iou_threshold, model_path=YOLO_MODEL_PATH, device=0,
            full_shape=(H, W), offset_xy=(x0, y0), progress_bar=image_scan_progress_bar)

        for (rect, st, lab) in zip(locs, status, labels):
            gxs = [x + x0 for x in rect[0]]
            gys = [y + y0 for y in rect[1]]
            bbox = (min(gxs), min(gys), max(gxs), max(gys))
            duplicate = False
            if HAVE_RTREE: duplicate = bool(list(seen.intersection(bbox)))
            else:
                for b in seen_boxes:
                    if compute_iou(bbox, b) >= 0.25: duplicate = True; break
            if duplicate: continue

            if HAVE_RTREE: seen.insert(tile_id, bbox)
            else: seen_boxes.append(bbox)

            dislocation_locations.append((gxs, gys))
            dislocation_location_status.append(st)
            dislocation_classification_labels.append(lab)

        patch_locations.extend([(pl and (pl[0]+x0, pl[1]+y0, pl[2]+x0, pl[3]+y0)) if pl else None for pl in patches])
        popt_list.extend(popts); pixel_size_list.extend(pixsizes)
        for cls, rects in cls_rects.items():
            class_rectangles[cls].extend([([x+x0 for x in r[0]], [y+y0 for y in r[1]]) for r in rects])

        tf.keras.backend.clear_session(); gc.collect()
        tile_id += 1

    elapsed_time = time.time() - start_time
    print(f"Processing time (all tiles): {elapsed_time:.2f} seconds")

    total_dislocations_found = len([i for i, lbl in enumerate(dislocation_classification_labels) if lbl != 'unknown'])
    save_file = os.path.join(os.path.dirname(file_path), f"{base_filename}_data_{total_dislocations_found}_dislocations_data_unfiltered.csv")

    with open(save_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['rect', 'classification_label', 'status', 'patch_location', 'popt', 'pixel_size'])
        for i in range(len(dislocation_locations)):
            csvwriter.writerow([dislocation_locations[i], dislocation_classification_labels[i],
                                dislocation_location_status[i], patch_locations[i], popt_list[i], pixel_size_list[i]])

    results_data = {
        'original_image': image_np, 'image_with_rectangles': copy_to_draw_on,
        'cleaned_image': None, 'dislocation_locations': dislocation_locations,
        'classification_labels': dislocation_classification_labels,
        'file_path': file_path, 'total_dislocations_found': total_dislocations_found,
        'show_results': show_results,
        'patch_locations': patch_locations, 'popt_list': popt_list, 'pixel_size_list': pixel_size_list
    }

    tf.keras.backend.clear_session(); gc.collect()
    return results_data

# Main GUI

class RemovableFileItem(QWidget):
    def __init__(self, file_path, remove_callback):
        super().__init__()
        self._file_path = file_path
        self._remove_callback = remove_callback
        lay = QHBoxLayout(self)
        label = QLabel(os.path.basename(file_path))
        btn = QPushButton("X", clicked=self._on_remove); btn.setFixedWidth(30)
        lay.addWidget(label); lay.addWidget(btn); lay.setContentsMargins(0,0,0,0)
    def _on_remove(self): self._remove_callback(self._file_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF File Processor")
        self._last_dir = os.path.expanduser("~")
        self._selected_files = []
        self._file_params = {}
        self._current_scan_box_size = 50

        self._btn_files = QPushButton("Select TIFF Files")
        self._btn_dirs = QPushButton("Select Directories")
        self._btn_clear = QPushButton("Clear Images")
        self._chk_remove_dislocations = QCheckBox("Remove Dislocations")
        if not HAVE_TORCH:
            self._chk_remove_dislocations.setEnabled(False)
            self._chk_remove_dislocations.setToolTip("PyTorch not available")

        self._chk_show_results = QCheckBox("Show Results")

        self._chk_manual_preprocess = QCheckBox("Manual Preprocessing (Crop/Smooth)")
        self._chk_manual_preprocess.setToolTip("Opens an editor for each image before processing")

        self._chk_interactive_filter = QCheckBox("Interactive Result Filtering")
        self._chk_interactive_filter.setToolTip("Manually remove artifacts using sliders after detection")

        self._chk_force_scanbox = QCheckBox("Force Scan Box Size")
        self._chk_force_scanbox.toggled.connect(self._on_force_scanbox_toggled)

        scanbox_layout = QHBoxLayout()
        self._lbl_scanbox = QLabel("Scan Box Size (px):")
        self._spin_scanbox = QSpinBox()
        self._spin_scanbox.setRange(10, 1000); self._spin_scanbox.setValue(120); self._spin_scanbox.setSingleStep(10)
        self._spin_scanbox.setEnabled(False); self._lbl_scanbox.setEnabled(False)
        scanbox_layout.addWidget(self._lbl_scanbox); scanbox_layout.addWidget(self._spin_scanbox); scanbox_layout.addStretch()
        self._scanbox_widget = QWidget(); self._scanbox_widget.setLayout(scanbox_layout)

        self._btn_process = QPushButton("Start Processing")
        self._list_items = QListWidget()
        self._status = QLabel("Select files and directories to process.")
        self._bar_total = QProgressBar(format="Overall progress: %p%")
        self._bar_scan = QProgressBar(format="Scanning current image: %p%")

        controls = QVBoxLayout()
        for w in (self._btn_files, self._btn_dirs, self._btn_clear, self._list_items,
                  self._chk_remove_dislocations, self._chk_manual_preprocess,
                  self._chk_interactive_filter,
                  self._chk_show_results, self._chk_force_scanbox,
                  self._scanbox_widget, self._btn_process, self._status,
                  self._bar_total, self._bar_scan):
            controls.addWidget(w)
        controls.addStretch()

        base = QWidget(); base.setLayout(controls)
        self.setCentralWidget(base)

        self._btn_files.clicked.connect(self._choose_files)
        self._btn_dirs.clicked.connect(self._choose_dirs)
        self._btn_clear.clicked.connect(self._clear_images)
        self._btn_process.clicked.connect(self._start_processing)

    def _on_force_scanbox_toggled(self, checked):
        self._spin_scanbox.setEnabled(checked)
        self._lbl_scanbox.setEnabled(checked)

    def _clear_images(self):
        self._selected_files.clear(); self._refresh_list()

    def _remove_fp(self, fp):
        if fp in self._selected_files:
            self._selected_files.remove(fp)
            self._refresh_list()

    def _refresh_list(self):
        self._list_items.clear()
        for fp in self._selected_files:
            it = QListWidgetItem(self._list_items)
            w = RemovableFileItem(fp, self._remove_fp)
            self._list_items.setItemWidget(it, w)
            it.setSizeHint(w.sizeHint())

    def _choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select TIFF files", self._last_dir, "TIFF files (*.tif *.tiff)")
        if not files: return
        self._last_dir = os.path.dirname(files[0])
        for fp in files:
            self._file_params[fp] = dict(clip_top=0, clip_bottom=0, clip_left=0, clip_right=0,
                                         max_sigma=0, max_gradient_ratio=0.2, max_allowed_percentage=0.8,
                                         scan_box_size=50, increment=1)
            self._selected_files.append(fp)
        self._selected_files = list(dict.fromkeys(self._selected_files))
        self._refresh_list()

    def _choose_dirs(self):
        d = QFileDialog.getExistingDirectory(self, "Select a directory", self._last_dir)
        if not d: return
        self._last_dir = d
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.lower().endswith((".tif", ".tiff")):
                    fp = os.path.join(root, fname)
                    if fp not in self._file_params:
                        self._file_params[fp] = dict(clip_top=0, clip_bottom=0, clip_left=0, clip_right=0,
                                                     max_sigma=0, max_gradient_ratio=0.2, max_allowed_percentage=0.8,
                                                     scan_box_size=50, increment=1)
                    self._selected_files.append(fp)
        self._selected_files = list(dict.fromkeys(self._selected_files))
        self._refresh_list()

    def _start_processing(self):
        paths = list(dict.fromkeys(self._selected_files))
        if not paths:
            QMessageBox.warning(self, "No files", "No TIFF files selected.")
            return

        remove_dislocations = self._chk_remove_dislocations.isChecked()
        show_results = self._chk_show_results.isChecked()
        do_interactive_filter = self._chk_interactive_filter.isChecked()

        # Variable to store "Apply to all" settings
        saved_preprocess_settings = None

        dislocation_processor = None
        if remove_dislocations:
            if not HAVE_TORCH: QMessageBox.warning(self, "Error", "PyTorch not available."); return
            if not os.path.exists(DISLOCATION_REMOVAL_MODEL_PATH): QMessageBox.warning(self, "Error", f"Model not found at {DISLOCATION_REMOVAL_MODEL_PATH}"); return
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = load_dislocation_removal_model(DISLOCATION_REMOVAL_MODEL_PATH, device)
                if not model: return
                dislocation_processor = BasicDislocationProcessor(model=model, device=device, tile_size=512, overlap=128)
            except Exception as e: QMessageBox.critical(self, "Error", str(e)); return

        for w in (self._btn_files, self._btn_dirs, self._btn_clear, self._btn_process,
                  self._chk_remove_dislocations, self._chk_manual_preprocess,
                  self._chk_interactive_filter,
                  self._chk_show_results, self._chk_force_scanbox, self._spin_scanbox):
            w.setEnabled(False)

        self._bar_total.setMaximum(len(paths)); self._bar_total.setValue(0)
        self._bar_scan.setValue(0)

        for idx, fp in enumerate(paths, 1):
            self._status.setText(f"Processing ({idx}/{len(paths)}): {fp}")
            QApplication.processEvents()

            try:
                final_image_data = None

                # Manual preprocessing
                if self._chk_manual_preprocess.isChecked():
                    if saved_preprocess_settings is not None:
                        # Load raw image directly
                        img = load_img(fp, color_mode='grayscale')
                        raw_array = np.squeeze(img_to_array(img)).astype(np.float32)
                        if raw_array.max() > raw_array.min():
                            base_array = (raw_array - raw_array.min()) / (raw_array.max() - raw_array.min()) * 255.0
                        else:
                            base_array = raw_array

                        # Apply saved settings without opening dialog
                        final_image_data = apply_headless_preprocessing(base_array, saved_preprocess_settings)
                    else:
                        # Open Dialog
                        dialog = ImageEditorDialog(fp, self)
                        result = dialog.exec()

                        if result == QDialog.Accepted:
                            final_image_data = dialog.get_final_image()
                            settings = dialog.get_settings()
                            if settings['apply_all']:
                                saved_preprocess_settings = settings
                        else:
                            continue # Skip image if canceled

                    if fp in self._file_params:
                         for k in ['clip_top','clip_bottom','clip_left','clip_right']: self._file_params[fp][k] = 0

                p = self._file_params.get(fp, dict(clip_top=0, clip_bottom=0, clip_left=0, clip_right=0,
                                                 max_sigma=50, max_gradient_ratio=0.2, max_allowed_percentage=0.8,
                                                 scan_box_size=50, increment=1))

                if self._chk_force_scanbox.isChecked():
                    scan_box_size = self._spin_scanbox.value()
                else:
                    if final_image_data is not None: image_np = final_image_data
                    else:
                        img = load_img(fp, color_mode='grayscale')
                        image_np = np.squeeze(img_to_array(img))

                    coarse_sz, (cx, cy) = coarse_scan_estimate(image_np, YOLO_MODEL_PATH, device=0, initial_factor=11)

                    # --- FIX: Ensure these are integers ---
                    cx, cy = int(cx), int(cy)
                    expand_factor = 10
                    half_side = int(coarse_sz * expand_factor // 2)

                    x0 = int(max(0, cx - half_side))
                    y0 = int(max(0, cy - half_side))
                    x1 = int(min(image_np.shape[1], cx + half_side))
                    y1 = int(min(image_np.shape[0], cy + half_side))
                    # --------------------------------------

                    crop = image_np[y0:y1, x0:x1]
                    scan_box_size = refine_scan_box_size(crop, YOLO_MODEL_PATH, device=0, coarse_guess_px=coarse_sz,
                                                         verbose=True, show_tuning_plot=False)
                p['scan_box_size'] = scan_box_size

                results_data = process_specific_tiff(
                    file_path=fp, fit=True, scan_box_size=p['scan_box_size'],
                    clip_top=p['clip_top'], clip_bottom=p['clip_bottom'],
                    clip_left=p['clip_left'], clip_right=p['clip_right'],
                    max_sigma=p['max_sigma'], increment=p['increment'],
                    max_gradient_ratio=p['max_gradient_ratio'],
                    max_allowed_percentage=p['max_allowed_percentage'],
                    image_scan_progress_bar=self._bar_scan,
                    overlap_margin=10, show_results=False,
                    override_image=final_image_data
                )

                if do_interactive_filter and results_data['dislocation_locations']:
                    img_for_filter = results_data['original_image']
                    locs_for_filter = results_data['dislocation_locations']
                    filter_dlg = ResultFilterDialog(img_for_filter, locs_for_filter, self)
                    if filter_dlg.exec() == QDialog.Accepted:
                        valid_indices = filter_dlg.get_filtered_indices()
                        filt_locs = [results_data['dislocation_locations'][i] for i in valid_indices]
                        results_data['dislocation_locations'] = filt_locs
                        results_data['classification_labels'] = [results_data['classification_labels'][i] for i in valid_indices]
                        results_data['total_dislocations_found'] = len(filt_locs)

                        base_filename = os.path.splitext(os.path.basename(fp))[0]
                        csv_dir = os.path.dirname(fp)
                        csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
                                     if f.startswith(f"{base_filename}_data_") and f.endswith("_dislocations.csv")]

                        if csv_files:
                            old_csv_path = max(csv_files, key=os.path.getmtime)
                            df = pd.read_csv(old_csv_path)
                            df_clean = df.iloc[valid_indices]
                            df_clean.to_csv(old_csv_path, index=False)
                            print(f"Updated CSV file: {old_csv_path}")

                if remove_dislocations and dislocation_processor:
                    self._status.setText(f"Removing dislocations ({idx}/{len(paths)}): {os.path.basename(fp)}")
                    QApplication.processEvents()
                    base_filename = os.path.splitext(os.path.basename(fp))[0]
                    csv_dir = os.path.dirname(fp)
                    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
                                 if f.startswith(f"{base_filename}_data_") and f.endswith("_dislocations.csv")]

                    if csv_files:
                        csv_path = max(csv_files, key=os.path.getmtime)
                        result_image = dislocation_processor.process_image(fp, csv_path, override_image=final_image_data)
                        output_filename = f"{base_filename}_dislocations_removed.tiff"
                        output_path = os.path.join(csv_dir, output_filename)
                        result_16bit = (result_image * 65535).astype(np.uint16)
                        if not cv2.imwrite(output_path, result_16bit):
                             Image.fromarray(result_16bit, mode='I;16').save(output_path, format='TIFF', compression='lzw')
                        results_data['cleaned_image'] = result_image

                if show_results and results_data:
                    plot_results(results_data['original_image'], results_data['image_with_rectangles'],
                                 results_data.get('cleaned_image'), results_data['dislocation_locations'],
                                 results_data['classification_labels'], results_data['file_path'],
                                 results_data['total_dislocations_found'])
                    QApplication.processEvents()

            except Exception as e:
                print(f"Error processing {fp}: {e}")
                import traceback; traceback.print_exc()

            self._bar_total.setValue(idx)

        QMessageBox.information(self, "Done", "Processing complete.")
        self._status.setText("Processing complete.")
        for w in (self._btn_files, self._btn_dirs, self._btn_clear, self._btn_process,
                  self._chk_remove_dislocations, self._chk_manual_preprocess,
                  self._chk_interactive_filter,
                  self._chk_show_results, self._chk_force_scanbox, self._spin_scanbox):
            w.setEnabled(True)
        self._bar_scan.setValue(0)

    def closeEvent(self, ev):
        cv2.destroyAllWindows(); QApplication.quit(); ev.accept()

def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
