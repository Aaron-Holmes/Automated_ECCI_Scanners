import os
import sys
import time
import gc
import csv
import math
import logging
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from skimage.restoration import denoise_wavelet, estimate_sigma
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# PySide6 / Qt Imports
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QImage, QPixmap, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QListWidget,
    QVBoxLayout, QWidget, QMessageBox, QLabel, QDialog, QHBoxLayout,
    QCheckBox, QProgressBar, QSlider, QFormLayout, QListWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QComboBox
)

from ultralytics import YOLO
import poisson_disc
from poisson_disc import Bridson_sampling

# Local import for the fitting function
import fitting_function as ff

try:
    from rtree import index
    HAVE_RTREE = True
except ImportError:
    HAVE_RTREE = False

# Scanning configuration and globals
# Scanning engine selector ("legacy" or "yolo")
SCAN_ENGINE = "yolo"  

YOLO_MODEL_PATH = "models/method_3/best.pt"

# Model Paths
binary_classification_model_path = 'models/method_2/binary.h5'
classification_model_double_check_path = 'models/method_2/binary.h5'
binary_classification_model = load_model(binary_classification_model_path)
classification_model_double_check = load_model(classification_model_double_check_path)
dislocation_locator_model_1 = load_model('models/method_2/locator.h5')
closeup_classification_model = load_model('models/method_2/multi_class.h5')

# Class Enums
p3_pole_class = 0
p4_pole_class = 1
dislocation_class = 2
faint_dislocation_class = 3
multiple_dislocations_class = 4
no_dislocation_class = 5
overlapping_class = 8
ultra_faint_dislocation_class = 6
unknown_class = 7

# Sizing and Overlap Parameters
classification_model_box_size = 60
bounds_locator_model_box_size = 60
scan_box_size = 50
step_size = scan_box_size // 8
allowed_overlap_percentage = 25

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

# Classification & feature detection utilities

def classify_patch(patch, model='model2'):
    """
    Evaluates a patch using either the double-check classification model or the 
    binary verification model. Interprets the last index as 'unknown' if utilizing an 
    OpenMax compiled model architecture.
    """
    global classification_model_path
    global classification_model_double_check_path
    global binary_classification_model_path

    if patch.shape[0] == 0 or patch.shape[1] == 0:
        logging.warning("Empty patch for classification.")
        return no_dislocation_class, 1.0

    patch_resized = cv2.resize(patch, (classification_model_box_size, classification_model_box_size))
    patch_resized = np.expand_dims(patch_resized, axis=0)
    patch_resized = np.expand_dims(patch_resized, axis=-1)

    if model == 'model2':
        prediction = classification_model_double_check.predict(patch_resized, verbose=global_model_verbose)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        if 'openmax' in classification_model_double_check_path.lower():
            if predicted_class == prediction.shape[1] - 1:
                predicted_class = unknown_class

    elif model == 'binary':
        prediction = binary_classification_model.predict(patch_resized, verbose=global_model_verbose)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

    else:
        raise ValueError(f"Invalid model: {model}")

    return predicted_class, confidence

def find_hill_like_peaks_2d(
    grad_mag,
    min_distance=6,
    threshold_rel=0.1,
    footprint_size=5,
    gaussian_sigma=1.5,
    min_hill_radius=2
):
    """
    Identifies robust 2D gradient peaks while filtering out single-pixel artifacts.
    """
    smoothed = gaussian_filter(grad_mag, sigma=gaussian_sigma)
    footprint = np.ones((footprint_size, footprint_size), dtype=bool)

    candidate_peaks = peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        footprint=footprint,
        exclude_border=False
    )

    final_peaks = []
    H, W = smoothed.shape
    for (r, c) in candidate_peaks:
        r0 = max(0, r - min_hill_radius)
        r1 = min(H, r + min_hill_radius + 1)
        c0 = max(0, c - min_hill_radius)
        c1 = min(W, c + min_hill_radius + 1)

        patch = smoothed[r0:r1, c0:c1]
        peak_val = smoothed[r, c]

        fraction_over_70pct = np.mean(patch >= 0.70 * peak_val)

        if fraction_over_70pct > 0.3:
            final_peaks.append((r, c))

    return np.array(final_peaks)

def locate_dislocation_bounds(patch, patch_location):
    """
    Leverages a CNN to predict bounding boxes within a given patch, scaling the 
    resulting coordinates to the global image perspective.
    """
    loc_x_min, loc_y_min = patch_location
    width = patch.shape[1]
    height = patch.shape[0]

    if patch.shape[0] == 0 or patch.shape[1] == 0:
        logging.warning("Empty patch for bounding box prediction.")
        return None

    patch_resized = cv2.resize(patch, (60, 60))
    patch_resized = np.expand_dims(patch_resized, axis=0)
    patch_resized = np.expand_dims(patch_resized, axis=-1)

    prediction = dislocation_locator_model_1.predict(patch_resized, verbose=global_model_verbose)
    x_min, y_min, x_max, y_max = prediction[0]

    x_min = int(x_min/60 * width + loc_x_min)
    y_min = int(y_min/60 * height + loc_y_min)
    x_max = int(x_max/60 * width + loc_x_min)
    y_max = int(y_max/60 * height + loc_y_min)

    return [x_min, y_min, x_max, y_max]

# Preprocessing / image smoothing

def calculate_gradient(image):
    gx, gy = np.gradient(image)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return grad_mag, gx, gy

def adaptively_smooth_image(image, max_sigma=50, increment=1,
                            max_gradient_ratio=0.2, max_allowed_percentage=0.8):
    original_gradient, _, _ = calculate_gradient(image)
    max_allowed_gradient = max_gradient_ratio * np.mean(original_gradient)
    num_exceeding_pixels = np.sum(original_gradient > max_allowed_gradient)

    for sigma in range(1, max_sigma + 1, increment):
        smoothed_image = gaussian_filter(image, sigma=sigma)
        current_gradient, _, _ = calculate_gradient(smoothed_image)
        num_exceeding_pixels_after = np.sum(current_gradient > max_allowed_gradient)
        reduction_percentage = (num_exceeding_pixels - num_exceeding_pixels_after) / num_exceeding_pixels

        logging.info(f"Sigma: {sigma}, Reduction Percentage: {reduction_percentage:.3f}")
        if reduction_percentage >= max_allowed_percentage:
            return smoothed_image, sigma

    return gaussian_filter(image, sigma=max_sigma), max_sigma

def preprocess_image(image_array, preprocess_steps,
                     clip_top=0, clip_bottom=0, clip_left=0, clip_right=0,
                     max_sigma=50, increment=1,
                     max_gradient_ratio=0.2, max_allowed_percentage=0.8, final_blur_sigma=1.0):
    """Executes normalization, noise reduction, and smoothing cascades."""
    noise_estimation = 0

    if 'clip' in preprocess_steps:
        orig_h, orig_w = image_array.shape
        image_array = image_array[clip_top:orig_h - clip_bottom,
                                  clip_left:orig_w - clip_right]

    for step in preprocess_steps:
        if step == 'normalize':
            arr_min = np.min(image_array)
            arr_max = np.max(image_array)
            if arr_max > arr_min:
                image_array = (image_array - arr_min) / (arr_max - arr_min)
            else:
                image_array = np.zeros_like(image_array)
        elif step == 'denoise':
            noise_estimation = estimate_sigma(image_array, average_sigmas=True)
            image_array = denoise_wavelet(image_array,
                                          sigma=0.5 * noise_estimation,
                                          wavelet='sym4',
                                          method='BayesShrink',
                                          mode='soft')
        elif step == 'adaptive_smoothing':
            image_array, smoothing_sigma = adaptively_smooth_image(
                image_array, max_sigma, increment,
                max_gradient_ratio, max_allowed_percentage
            )
            logging.info(f"Adaptive smoothing applied with sigma: {smoothing_sigma}")

    if final_blur_sigma > 0:
        image_array = gaussian_filter(image_array, sigma=final_blur_sigma)
    image_array = (image_array * 255).astype(np.uint8)
    return image_array, noise_estimation

# Dislocation fitting logic (multiple dislocations)

def post_fit_checks(popt, pcov, xy, patch, z):
    keep_fit = True
    (translation_factor_x, translation_factor_y, stretch_factor_x, stretch_factor_y, theta_x,
     theta_y, theta_z, amplitude, height_factor, baseline, shear_factor_x, shear_factor_y) = popt

    mask_threshold = 0.08
    mask = np.logical_and(z >= baseline - mask_threshold, z <= baseline + mask_threshold)

    grad_y, grad_x = np.gradient(z)
    grad_mag = np.hypot(grad_x, grad_y)
    gradient_threshold = 0.03
    mask_gradient = grad_mag >= gradient_threshold
    mask_combined = np.logical_and(mask, np.logical_not(mask_gradient))

    s = [[1,1,1],[1,1,1],[1,1,1]]
    labeled_mask, num_features = label(mask_combined, structure=s)
    if num_features > 1:
        logging.warning(f"Mask has {num_features} separate features => suspicious.")
        keep_fit = False

    z_masked = np.ma.masked_array(z, mask=mask_combined)
    patch_masked = np.ma.masked_array(patch, mask=mask_combined)

    z_min = z_masked.min()
    z_max = z_masked.max()
    if (z_max - z_min) == 0:
        return False, mask_combined

    z_norm = (z_masked - z_min) / (z_max - z_min)

    patch_min = patch_masked.min()
    patch_max = patch_masked.max()
    if (patch_max - patch_min) == 0:
        return False, mask_combined

    patch_norm = (patch_masked - patch_min) / (patch_max - patch_min)
    mse = np.ma.mean((z_norm - patch_norm)**2)
    logging.info(f"Mean squared error for fit: {mse:.4f}")
    if mse > 0.06:
        logging.warning("Mean squared error is too high.")
        keep_fit = False

    return keep_fit, mask_combined

def fit_process_multiple_dislocations(
    raw_array, image, dislocation_image, centers, bounds, centers_of_fits,
    dislocation_locations, dislocation_location_status, dislocation_classification_labels,
    patch_locations, popt_list, pixel_size_list, candidate_confidences,
    approved_dislocation_confidence_scores, fit, classification_label
):
    """
    Compiles adjacent overlapping structures and attempts multi-variable fitting 
    across the entire combined footprint. Updates status tracking arrays.
    """
    image_copy = raw_array.copy()
    dislocation_image_copy = dislocation_image.copy()

    for c in centers:
        centers_of_fits.append(c)

    logging.info("Dislocations found, attempting fitting...")

    width = image_copy.shape[1]
    height = image_copy.shape[0]

    increased_bounds = []
    for i, bound in enumerate(bounds):
        x_min, y_min, x_max, y_max = bound
        center = centers[i]
        largest_side = max(x_max - x_min, y_max - y_min)
        largest_side_ext = largest_side * 1.25
        x_start = int(max(0, center[0] - largest_side_ext))
        y_start = int(max(0, center[1] - largest_side_ext))

        x_end = (int(min(width, center[0] + largest_side_ext))
                 if x_start != 0 else 1 + 2*int(largest_side_ext))
        y_end = (int(min(height, center[1] + largest_side_ext))
                 if y_start != 0 else 1 + 2*int(largest_side_ext))

        if x_end >= width:
            x_start = x_end - 1 - 2*int(largest_side_ext)
        if y_end >= height:
            y_start = y_end - 1 - 2*int(largest_side_ext)

        increased_bounds.append([x_start, y_start, x_end, y_end])

    x_min_crop = min(b[0] for b in increased_bounds)
    y_min_crop = min(b[1] for b in increased_bounds)
    x_max_crop = max(b[2] for b in increased_bounds)
    y_max_crop = max(b[3] for b in increased_bounds)
    super_bounds = [x_min_crop, y_min_crop, x_max_crop, y_max_crop]

    patch = image_copy[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
    raw_patch = raw_array[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

    result = ff.fit_multiple_dislocations(raw_patch, centers, bounds, super_bounds, fit=fit)
    if result is None:
        logging.error("Fitting failed for all dislocations.")
        for i in range(len(bounds)):
            dislocation_locations.append((
                [
                    bounds[i][0] + super_bounds[0],
                    bounds[i][2] + super_bounds[0],
                    bounds[i][2] + super_bounds[0],
                    bounds[i][0] + super_bounds[0],
                    bounds[i][0] + super_bounds[0]
                ],
                [
                    bounds[i][1] + super_bounds[1],
                    bounds[i][1] + super_bounds[1],
                    bounds[i][3] + super_bounds[1],
                    bounds[i][3] + super_bounds[1],
                    bounds[i][1] + super_bounds[1]
                ]
            ))
            dislocation_location_status.append('Failed_fit')
            dislocation_classification_labels.append(classification_label)
            patch_locations.append(None)
            popt_list.append(None)
            pixel_size_list.append(None)
            if i < len(candidate_confidences):
                approved_dislocation_confidence_scores.append(candidate_confidences[i])
        return image_copy, dislocation_image_copy, centers_of_fits

    try:
        popt, pcov, xy, centers_from_multi_fit, pixel_size = result
    except ValueError as e:
        logging.error(f"Error unpacking fit result: {e}")
        popt = None

    if popt is None:
        logging.error("No popt from fitting result.")
        for i in range(len(bounds)):
            dislocation_locations.append((
                [
                    bounds[i][0] + super_bounds[0],
                    bounds[i][2] + super_bounds[0],
                    bounds[i][2] + super_bounds[0],
                    bounds[i][0] + super_bounds[0],
                    bounds[i][0] + super_bounds[0]
                ],
                [
                    bounds[i][1] + super_bounds[1],
                    bounds[i][1] + super_bounds[1],
                    bounds[i][3] + super_bounds[1],
                    bounds[i][3] + super_bounds[1],
                    bounds[i][1] + super_bounds[1]
                ]
            ))
            dislocation_location_status.append('Failed_fit')
            dislocation_classification_labels.append(classification_label)
            patch_locations.append(None)
            popt_list.append(None)
            pixel_size_list.append(None)
            if i < len(candidate_confidences):
                approved_dislocation_confidence_scores.append(candidate_confidences[i])
        return image_copy, dislocation_image_copy, centers_of_fits

    num_params = 12
    num_dislocations = len(popt) // num_params

    for i in range(num_dislocations):
        popt_single = popt[i*num_params:(i+1)*num_params]
        fit_status = 'Failed_fit'
        patch_loc = None
        popt_list.append(None)
        pixel_size_list.append(None)

        if np.isnan(popt_single).any():
            logging.warning(f"Dislocation {i}: NaNs in fit parameters.")
        else:
            z_single = ff.model(xy, *popt_single).reshape(patch.shape)
            min_z, max_z = np.min(z_single), np.max(z_single)
            if (max_z - min_z) == 0:
                logging.warning(f"Dislocation {i}: zero variance in fitted surface.")
            else:
                if fit:
                    normalised_z = (z_single - min_z) / (max_z - min_z)
                    keep_fit, mask = post_fit_checks(popt_single, pcov, xy, raw_patch, z_single)

                    if keep_fit:
                        fit_status = 'Successful_fit'
                        dislocation_image_copy[y_min_crop:y_max_crop, x_min_crop:x_max_crop][~mask] += normalised_z[~mask]

                        smoothed_patch = ff.smooth_and_add_noise(
                            raw_patch, z_single, centers[i][0], centers[i][1],
                            bounds[i][0], bounds[i][1], bounds[i][2], bounds[i][3],
                            gradient_threshold=0.01,
                            laplacian_threshold=0.01,
                            smoothing_sigma=100,
                            boundary_blur_sigma=0.4,
                            noise_sigma_estimation=0.3,
                            noise_type='gaussian',
                            structuring_element_size=5,
                            dilation_iterations=5,
                            grey_dilation_iterations=15,
                            beta=1.0,
                            gamma=3.0,
                            min_smoothing_value=0.3,
                            show_plots=False
                        )
                        image_copy[y_min_crop:y_max_crop, x_min_crop:x_max_crop] = smoothed_patch

                        patch_loc = (x_min_crop, y_min_crop, x_max_crop, y_max_crop)
                        popt_list[-1] = popt_single
                        pixel_size_list[-1] = pixel_size
                        if centers_from_multi_fit is not None and i < len(centers_from_multi_fit):
                            centers_of_fits.append((centers_from_multi_fit[i][1], centers_from_multi_fit[i][0]))
                else:
                    fit_status = 'Successful_fit'
                    patch_loc = (x_min_crop, y_min_crop, x_max_crop, y_max_crop)
                    popt_list[-1] = popt_single
                    pixel_size_list[-1] = pixel_size
                    if centers_from_multi_fit is not None and i < len(centers_from_multi_fit):
                        centers_of_fits.append((centers_from_multi_fit[i][1], centers_from_multi_fit[i][0]))

        dislocation_locations.append((
            [
                bounds[i][0] + super_bounds[0],
                bounds[i][2] + super_bounds[0],
                bounds[i][2] + super_bounds[0],
                bounds[i][0] + super_bounds[0],
                bounds[i][0] + super_bounds[0]
            ],
            [
                bounds[i][1] + super_bounds[1],
                bounds[i][1] + super_bounds[1],
                bounds[i][3] + super_bounds[1],
                bounds[i][3] + super_bounds[1],
                bounds[i][1] + super_bounds[1]
            ]
        ))
        dislocation_location_status.append(fit_status)
        dislocation_classification_labels.append(classification_label)
        patch_locations.append(patch_loc)

        if i < len(candidate_confidences):
            approved_dislocation_confidence_scores.append(candidate_confidences[i])

    return image_copy, dislocation_image_copy, centers_of_fits

# Spatial search and localized validation

def _local_perimeter_search(
    image_array,
    bounds,
    scan_box_size,
    perimeter_patches_per_side=6,
    offset_percent=0.05,
    min_positive=5
):
    """
    Validates boundary regions by sliding a secondary scanner to determine if 
    defects extend beyond standard footprint constraints.
    """
    x_min, y_min, x_max, y_max = bounds
    if x_min >= x_max or y_min >= y_max:
        return False

    width = x_max - x_min
    height = y_max - y_min
    if width < 5 or height < 5:
        return False

    N = perimeter_patches_per_side
    if N < 1:
        return False

    off_x = int(width * offset_percent)
    off_y = int(height * offset_percent)

    left_x = x_min + off_x
    right_x = x_max - off_x
    top_y = y_min + off_y
    bottom_y = y_max - off_y

    if left_x >= right_x or top_y >= bottom_y:
        return False

    points = []
    for i in range(N):
        px = left_x + i * (right_x - left_x) // max(1, (N-1))
        py = top_y
        points.append((px, py))
    for i in range(N):
        px = right_x
        py = top_y + i * (bottom_y - top_y) // max(1, (N-1))
        points.append((px, py))
    for i in range(N):
        px = right_x - i * (right_x - left_x) // max(1, (N-1))
        py = bottom_y
        points.append((px, py))
    for i in range(N):
        px = left_x
        py = bottom_y - i * (bottom_y - top_y) // max(1, (N-1))
        points.append((px, py))

    dislocation_count = 0
    half_s = scan_box_size // 2

    for (px, py) in points:
        x0 = max(0, px - half_s)
        y0 = max(0, py - half_s)
        x1 = min(image_array.shape[1], x0 + scan_box_size)
        y1 = min(image_array.shape[0], y0 + scan_box_size)

        if (x1 - x0) < 5 or (y1 - y0) < 5:
            continue

        patch = image_array[y0:y1, x0:x1]

        pred_class, conf = classify_patch(patch, model='binary')
        if pred_class == 1: 
            dislocation_count += 1

    return (dislocation_count >= min_positive)


def generate_poisson_points(width, height, r, k, total_samples):
    try:
        points = Bridson_sampling(width, height, r=r, k=k)
    except Exception:
        points = []

    if len(points) < total_samples:
        needed = total_samples - len(points)
        random_points = np.random.rand(needed, 2)
        random_points = [tuple(pt) for pt in random_points]
        points.extend(random_points)

    return points[:total_samples]


def local_randomized_search(
    image_array,
    bounds,
    scan_box_size,
    total_samples=30,
    offset_percent=0.1,
    min_positive_percent=25,
    min_distance_ratio=0.2,
    test_mode=False
):
    x_min, y_min, x_max, y_max = bounds
    if x_min >= x_max or y_min >= y_max:
        return False

    width = x_max - x_min
    height = y_max - y_min

    if width < 5 or height < 5:
        return False

    off_x = int(width * offset_percent)
    off_y = int(height * offset_percent)

    left_x = x_min + off_x
    right_x = x_max - off_x
    top_y = y_min + off_y
    bottom_y = y_max - off_y

    if left_x >= right_x or top_y >= bottom_y:
        return False

    region_width = right_x - left_x
    region_height = bottom_y - top_y

    points = generate_poisson_points(
        width=1.0,
        height=1.0,
        r=min_distance_ratio,
        k=30,
        total_samples=total_samples
    )

    scaled_points = [
        (
            int(left_x + px * region_width),
            int(top_y + py * region_height)
        )
        for px, py in points
    ]

    dislocation_count = 0
    half_s = scan_box_size // 2

    if test_mode:
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap='gray')
        ax.add_patch(plt.Rectangle((x_min, y_min), width, height, edgecolor='yellow', facecolor='none', lw=2, label='Bounds'))
        ax.add_patch(plt.Rectangle((left_x, top_y), region_width, region_height, edgecolor='blue', facecolor='none', lw=1, label='Inset region'))

    for (px, py) in scaled_points:
        x0 = max(0, px - half_s)
        y0 = max(0, py - half_s)
        x1 = min(image_array.shape[1], x0 + scan_box_size)
        y1 = min(image_array.shape[0], y0 + scan_box_size)

        if (x1 - x0) < 5 or (y1 - y0) < 5:
            continue

        patch = image_array[y0:y1, x0:x1]
        pred_class, conf = classify_patch(patch, model='binary')

        if pred_class == 1:
            dislocation_count += 1

        if test_mode:
            color = 'green' if pred_class == 1 else 'red'
            rect = plt.Rectangle((x0, y0), scan_box_size, scan_box_size, edgecolor=color, facecolor='none', lw=1)
            ax.add_patch(rect)
            ax.text(x0, y0, f"{conf:.2f}", color=color, fontsize=6)

    if test_mode:
        ax.set_title(f"Dislocation count = {dislocation_count}/{total_samples}")
        ax.legend()
        plt.show()

    min_required = int((min_positive_percent / 100.0) * total_samples)
    return dislocation_count >= min_required


def initial_dislocation_checks(
    image_array,
    bounds,
    scan_box_area,
    scan_box_size,
    dislocation_locations,
    dislocation_location_status,
    avg_grad_mag,
    base_filename,
    grad_factor=3,
    perimeter_patches_per_side=6,
    offset_percent=0.05,
    min_positive=5
):
    def _fail_local_search():
        ok = local_randomized_search(
            image_array=image_array,
            bounds=bounds,
            scan_box_size=scan_box_size,
            total_samples=30,
            offset_percent=0.1,
            min_positive_percent=25,
            min_distance_ratio=0.2, 
            test_mode=False
        )
        return ok

    x_min, y_min, x_max, y_max = bounds
    width = x_max - x_min
    height = y_max - y_min

    if x_min >= x_max or y_min >= y_max:
        print(f"Invalid bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
        return True if _fail_local_search() else False

    if width <= 0 or height <= 0:
        print("Bounding box has zero width or height.")
        return True if _fail_local_search() else False

    patch = image_array[y_min:y_max, x_min:x_max]
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        print("Patch is empty after applying bounds.")
        return True if _fail_local_search() else False

    bounding_box_area = width * height
    if bounding_box_area > 0.8 * scan_box_area:
        return True if _fail_local_search() else False
    if bounding_box_area < 0.05 * scan_box_area:
        return True if _fail_local_search() else False

    if width < 0.05 * scan_box_size or height < 0.05 * scan_box_size:
        return True if _fail_local_search() else False

    min_val = np.min(patch)
    max_val = np.max(patch)
    if max_val == min_val:
        print("Patch has zero variance; cannot normalize.")
        return True if _fail_local_search() else False

    patch_normalised = (patch - min_val) / (max_val - min_val)
    patch_resized = cv2.resize(patch_normalised, (60, 60))
    if patch_resized.max() == patch_resized.min():
        print("Patch has zero variance (2nd pass).")
        return True if _fail_local_search() else False

    patch_input = np.expand_dims(patch_resized, axis=0)
    patch_input = np.expand_dims(patch_input, axis=-1)

    prediction = closeup_classification_model.predict(patch_input, verbose=0)
    pred_cls = np.argmax(prediction)
    conf_closeup = prediction[0][pred_cls]

    if pred_cls == no_dislocation_class:
        print("Patch is not classified as a dislocation (closeup model).")
        return True if _fail_local_search() else False

    gy, gx = np.gradient(patch)
    gmag = np.hypot(gx, gy)
    grad_threshold = avg_grad_mag * grad_factor
    if not np.any(gmag >= grad_threshold):
        print("Patch has no high gradient magnitude.")
        return True if _fail_local_search() else False

    pred_bin_class, bin_conf = classify_patch(patch, model='binary')
    if pred_bin_class == 0:
        print("Patch is not classified as a dislocation (binary model).")
        return True if _fail_local_search() else False

    print("Initial dislocation checks passed.")
    return True


def check_for_closeby_dislocations(
    image_array, x, y, largest_side,
    dislocation_locations, dislocation_location_status,
    avg_grad_mag, centers_of_fits,
    base_filename,
    approved_dislocation_confidence_scores,
    rejected_dislocation_confidence_scores,
    non_dislocation_confidence_scores,
    scan_box_size=50
):
    """
    Scans around coordinates to identify adjacent dislocation candidates 
    for multi-patch fitting aggregation.
    """
    buffer = scan_box_size//6
    x_0 = max(0, x - scan_box_size//2 - buffer)
    y_0 = max(0, y - scan_box_size//2 - buffer)
    x_1 = min(image_array.shape[1], x + scan_box_size//2 + buffer)
    y_1 = min(image_array.shape[0], y + scan_box_size//2 + buffer)

    centers = [(x, y)]
    bounds = []
    closeby_confidences = []

    num_steps = 5
    path = []

    for i in range(num_steps+1):
        path.append((x_0 + i*(x_1 - x_0)//num_steps, y_0))
    for i in range(num_steps+1):
        path.append((x_1, y_0 + i*(y_1 - y_0)//num_steps))
    for i in range(num_steps+1):
        path.append((x_1 - i*(x_1 - x_0)//num_steps, y_1))
    for i in range(num_steps+1):
        path.append((x_0, y_1 - i*(y_1 - y_0)//num_steps))

    for point in path:
        px, py = point
        too_close_flag = any(
            np.sqrt((c[0] - px)**2 + (c[1] - py)**2) < scan_box_size//4
            for c in centers
        )
        if too_close_flag:
            continue
        too_close_flag = any(
            np.sqrt((fc[0] - px)**2 + (fc[1] - py)**2) < scan_box_size//3
            for fc in centers_of_fits
        )
        if too_close_flag:
            continue

        min_x = max(0, px - scan_box_size//2)
        min_y = max(0, py - scan_box_size//2)
        max_x = min(image_array.shape[1], px + scan_box_size//2)
        max_y = min(image_array.shape[0], py + scan_box_size//2)

        patch = image_array[min_y:max_y, min_x:max_x]
        predicted_class, confidence = classify_patch(patch, model='model2')

        if predicted_class == dislocation_class or predicted_class == faint_dislocation_class:
            new_bounds = locate_dislocation_bounds(patch, (min_x, min_y))
            if new_bounds is None:
                continue
            bx_min, by_min, bx_max, by_max = new_bounds
            center_x, center_y = (bx_min + bx_max)//2, (by_min + by_max)//2

            if any(np.sqrt((c[0]-center_x)**2 + (c[1]-center_y)**2) < scan_box_size//3
                   for c in centers):
                continue
            if any(np.sqrt((fc[0]-center_x)**2 + (fc[1]-center_y)**2) < scan_box_size//3
                   for fc in centers_of_fits):
                continue

            if not initial_dislocation_checks(
                image_array, new_bounds, scan_box_size*scan_box_size, scan_box_size,
                dislocation_locations, dislocation_location_status,
                avg_grad_mag, base_filename, grad_factor = 3
            ):
                rejected_dislocation_confidence_scores.append(confidence)
                continue

            centers.append((center_x, center_y))
            bounds.append(new_bounds)
            closeby_confidences.append(confidence)

        elif predicted_class == no_dislocation_class:
            if confidence < 0.99:
                non_dislocation_confidence_scores.append(confidence)
            continue

    indices_to_remove = []
    for i in range(1, len(centers)):
        cx, cy = centers[i]
        distance = np.sqrt((cx - x)**2 + (cy - y)**2)
        if distance > largest_side * 1.5:
            indices_to_remove.append(i)

    for idx in reversed(indices_to_remove):
        del centers[idx]
        del bounds[idx-1]
        del closeby_confidences[idx-1]

    if len(centers) == 1:
        return centers, None, []
    else:
        return centers, bounds, closeby_confidences


# Class routing & per-class processing

def classification_switch(args, new_class):
    max_switch = 5
    old_class = args['current_class']
    
    if new_class != old_class:
        args['switch_count'] += 1
        print(f"[Switch] from class {old_class} to {new_class}, "
              f"switch_count={args['switch_count']}")
    else:
        print(f"[Switch] classification remains {new_class}, "
              f"switch_count={args['switch_count']}")

    if args['switch_count'] > max_switch:
        print("Switch count exceeded threshold! Forcing unknown_class.")
        return unknown_class, args['switch_count']
    else:
        args['current_class'] = new_class
        return new_class, args['switch_count']


def process_unknown_class(xx, yy, patch, confidence,
                          background_image, dislocation_image,
                          image_array, dislocation_locations, dislocation_location_status,
                          dislocation_classification_labels,
                          centers_of_fits, patch_locations, popt_list, pixel_size_list,
                          rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                          non_dislocation_confidence_scores,
                          base_filename, avg_grad_mag, scan_box_size,
                          step_size, too_close, scan_box_area,
                          fit=True, raw_array=None,
                          classification_label='unknown',
                          switch_count=0,
                          max_switch=5):
    print(f"[Unknown] Starting patch=({xx},{yy}), conf={confidence}, switch_count={switch_count}")
    
    max_iter = 3
    args = {
        'current_class': unknown_class,
        'switch_count': switch_count
    }

    patch_for_this_class = patch
    current_confidence = confidence
    current_x, current_y = xx, yy

    final_bounds = None
    final_patch_centered = False

    for iteration in range(max_iter):
        bounds = locate_dislocation_bounds(patch_for_this_class, (current_x, current_y))
        if bounds is None:
            print("[Unknown] locate_dislocation_bounds failed => cannot recenter.")
            break

        x_min, y_min, x_max, y_max = bounds

        bx_center = (x_min + x_max) // 2
        by_center = (y_min + y_max) // 2
        side_w = x_max - x_min
        side_h = y_max - y_min
        largest_side = max(side_w, side_h, scan_box_size // 3)
        largest_side_ext = int(largest_side * 1.25)

        x_start = bx_center - largest_side_ext
        y_start = by_center - largest_side_ext
        x_end   = bx_center + largest_side_ext
        y_end   = by_center + largest_side_ext

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end   = min(background_image.shape[1], x_end)
        y_end   = min(background_image.shape[0], y_end)

        patch_for_this_class = background_image[y_start:y_end, x_start:x_end]
        final_bounds = [x_min, y_min, x_max, y_max] 
        final_patch_centered = True

        new_pred_class, new_conf = classify_patch(patch_for_this_class, model='model2')
        new_pred_class, args['switch_count'] = classification_switch(args, new_pred_class)
        current_confidence = new_conf

        if args['switch_count'] > max_switch:
            print(f"[Unknown] Switch count {args['switch_count']} > {max_switch} => finalize unknown.")
            break

        if new_pred_class != unknown_class:
            print(f"[Unknown] classification changed to {new_pred_class} => switching out.")
            new_class_info = CLASS_INFO.get(new_pred_class)
            if new_class_info and new_class_info['process_func']:
                return new_class_info['process_func'](
                    xx=x_start, yy=y_start, patch=patch_for_this_class, confidence=new_conf,
                    background_image=background_image, dislocation_image=dislocation_image,
                    image_array=image_array, dislocation_locations=dislocation_locations,
                    dislocation_location_status=dislocation_location_status,
                    dislocation_classification_labels=dislocation_classification_labels,
                    centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                    popt_list=popt_list, pixel_size_list=pixel_size_list,
                    rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                    approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                    non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                    base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                    scan_box_size=scan_box_size, step_size=step_size,
                    too_close=too_close, scan_box_area=scan_box_area,
                    fit=fit, raw_array=raw_array,
                    classification_label=classification_label_map.get(new_pred_class, 'unknown'),
                    switch_count=args['switch_count']
                )
            else:
                break

        if iteration == (max_iter - 1):
            break

    if final_patch_centered and final_bounds is not None:
        x_min, y_min, x_max, y_max = final_bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(x_max, background_image.shape[1])
        y_max = min(y_max, background_image.shape[0])
        print(f"[Unknown] Final bounding box: ({x_min},{y_min})..({x_max},{y_max})")
    else:
        x_min = xx
        y_min = yy
        x_max = xx + scan_box_size
        y_max = yy + scan_box_size
        x_max = min(x_max, background_image.shape[1])
        y_max = min(y_max, background_image.shape[0])
        print(f"[Unknown] Could not do bounding-box center => storing the original patch {x_min},{y_min},{x_max},{y_max}")

    box_x_coords = [x_min, x_max, x_max, x_min, x_min]
    box_y_coords = [y_min, y_min, y_max, y_max, y_min]

    dislocation_locations.append((box_x_coords, box_y_coords))
    dislocation_location_status.append("Failed_fit")
    dislocation_classification_labels.append('unknown')

    patch_locations.append(None)
    popt_list.append(None)
    pixel_size_list.append(None)

    rejected_dislocation_confidence_scores.append(current_confidence)

    return background_image, dislocation_image, centers_of_fits, True


def process_p3_pole_class(xx, yy, patch, confidence,
                          background_image, dislocation_image,
                          image_array, dislocation_locations, dislocation_location_status,
                          dislocation_classification_labels,
                          centers_of_fits, patch_locations, popt_list, pixel_size_list,
                          rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                          non_dislocation_confidence_scores,
                          base_filename, avg_grad_mag, scan_box_size,
                          step_size, too_close, scan_box_area,
                          fit=True, raw_array=None,
                          classification_label='p3_pole',
                          switch_count=0):
    print(f"[p3_pole] Starting patch ({xx}, {yy}), switch_count={switch_count}")

    args = {
        'current_class': p3_pole_class,
        'switch_count': switch_count
    }

    max_iter = 1
    p3_confirmed = False
    patch_for_this_class = patch
    current_x, current_y = xx, yy
    current_confidence = confidence
    dist_centers = 999999

    for iteration in range(max_iter):
        bounds = locate_dislocation_bounds(
            patch_for_this_class, (current_x, current_y)
        )
        if bounds is None:
            return background_image, dislocation_image, centers_of_fits, True
        
        x_min, y_min, x_max, y_max = bounds

        bx_center = (x_min + x_max) // 2
        by_center = (y_min + y_max) // 2
        side1 = x_max - x_min
        side2 = y_max - y_min
        largest_side = max(side1, side2, scan_box_size // 3)
        largest_side_ext = int(largest_side * 1.25)

        x_start = bx_center - largest_side_ext
        y_start = by_center - largest_side_ext
        x_end   = bx_center + largest_side_ext
        y_end   = by_center + largest_side_ext

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end   = min(background_image.shape[1], x_end)
        y_end   = min(background_image.shape[0], y_end)

        patch_for_this_class = background_image[y_start:y_end, x_start:x_end]

        new_pred_class, new_conf = classify_patch(patch_for_this_class, model='model2')
        new_pred_class, args['switch_count'] = classification_switch(args, new_pred_class)

        patch_cx = (x_start + x_end)//2
        patch_cy = (y_start + y_end)//2
        new_bounds = locate_dislocation_bounds(
            patch_for_this_class, (x_start, y_start)
        )
        if new_bounds is None:
            return background_image, dislocation_image, centers_of_fits, True
        nx_min, ny_min, nx_max, ny_max = new_bounds

        nbx_center = (nx_min + nx_max)//2
        nby_center = (ny_min + ny_max)//2
        dist_centers = np.sqrt((patch_cx - nbx_center)**2 + (patch_cy - nby_center)**2)

        if new_pred_class == unknown_class:
            return process_unknown_class(
                x_start, y_start, patch_for_this_class, new_conf,
                background_image, dislocation_image,
                image_array, dislocation_locations, dislocation_location_status,
                dislocation_classification_labels,
                centers_of_fits, patch_locations, popt_list, pixel_size_list,
                rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores,
                base_filename, avg_grad_mag, scan_box_size,
                step_size, too_close, scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label='unknown',
                switch_count=args['switch_count']
            )

        if new_pred_class != p3_pole_class:
            print(f"[p3_pole] Changed to {new_pred_class}, switching.")
            new_class_info = CLASS_INFO.get(new_pred_class)
            if new_class_info and new_class_info['process_func']:
                return new_class_info['process_func'](
                    xx=x_start, yy=y_start, patch=patch_for_this_class, confidence=new_conf,
                    background_image=background_image, dislocation_image=dislocation_image,
                    image_array=image_array, dislocation_locations=dislocation_locations,
                    dislocation_location_status=dislocation_location_status,
                    dislocation_classification_labels=dislocation_classification_labels,
                    centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                    popt_list=popt_list, pixel_size_list=pixel_size_list,
                    rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                    approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                    non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                    base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                    scan_box_size=scan_box_size, step_size=step_size,
                    too_close=too_close, scan_box_area=scan_box_area,
                    fit=fit, raw_array=raw_array,
                    classification_label=classification_label_map[new_pred_class],
                    switch_count=args['switch_count']
                )
            else:
                return background_image, dislocation_image, centers_of_fits, True

        p3_confirmed = True
        current_confidence = new_conf
        current_x, current_y = x_start, y_start

        if dist_centers < 1:
            break

    if not p3_confirmed:
        return background_image, dislocation_image, centers_of_fits, True

    final_pred_class, args['switch_count'] = classification_switch(args, p3_pole_class)
    if final_pred_class == unknown_class:
        return process_unknown_class(
            current_x, current_y, patch_for_this_class, current_confidence,
            background_image, dislocation_image,
            image_array, dislocation_locations, dislocation_location_status,
            dislocation_classification_labels,
            centers_of_fits, patch_locations, popt_list, pixel_size_list,
            rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
            non_dislocation_confidence_scores,
            base_filename, avg_grad_mag, scan_box_size,
            step_size, too_close, scan_box_area,
            fit=fit, raw_array=raw_array,
            classification_label='unknown',
            switch_count=args['switch_count']
        )

    if final_pred_class != p3_pole_class:
        new_class_info = CLASS_INFO.get(final_pred_class)
        if new_class_info and new_class_info['process_func']:
            return new_class_info['process_func'](
                xx=current_x, yy=current_y, patch=patch_for_this_class, confidence=current_confidence,
                background_image=background_image, dislocation_image=dislocation_image,
                image_array=image_array, dislocation_locations=dislocation_locations,
                dislocation_location_status=dislocation_location_status,
                dislocation_classification_labels=dislocation_classification_labels,
                centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                popt_list=popt_list, pixel_size_list=pixel_size_list,
                rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                scan_box_size=scan_box_size, step_size=step_size,
                too_close=too_close, scan_box_area=scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label=classification_label_map[new_pred_class],
                switch_count=args['switch_count']
            )
        else:
            return background_image, dislocation_image, centers_of_fits, True

    patch_gray = patch_for_this_class.astype(float)
    gx, gy = np.gradient(patch_gray)
    grad_mag = np.hypot(gx, gy)

    peaks = peak_local_max(
        grad_mag,
        min_distance=4,
        threshold_abs=0.6 * grad_mag.max()
    )

    sub_patch_size = max(5, int(0.5 * scan_box_size))
    half_sub = sub_patch_size // 2

    all_sub_bounds = []
    for (py, px) in peaks:
        global_x = current_x + px
        global_y = current_y + py

        sub_x0 = max(0, global_x - half_sub)
        sub_y0 = max(0, global_y - half_sub)
        sub_x1 = min(background_image.shape[1], global_x + half_sub)
        sub_y1 = min(background_image.shape[0], global_y + half_sub)

        if (sub_x1 - sub_x0) < 5 or (sub_y1 - sub_y0) < 5:
            continue

        sub_patch = background_image[sub_y0:sub_y1, sub_x0:sub_x1]
        sub_bounds = locate_dislocation_bounds(sub_patch, (sub_x0, sub_y0))
        if sub_bounds is None:
            continue

        sbx_min, sby_min, sbx_max, sby_max = sub_bounds

        sub_center_x = (sbx_min + sbx_max) // 2
        sub_center_y = (sby_min + sby_max) // 2
        if any(
            np.sqrt((cx - sub_center_x)**2 + (cy - sub_center_y)**2) < too_close
            for (cx, cy) in centers_of_fits
        ):
            continue

        all_sub_bounds.append((sbx_min, sby_min, sbx_max, sby_max))

    if len(all_sub_bounds) == 0:
        print("[p3_pole] No local peaks => skipping.")
        return background_image, dislocation_image, centers_of_fits, True

    if len(all_sub_bounds) == 1:
        (bxmin, bymin, bxmax, bymax) = all_sub_bounds[0]
        box_x_coords = [bxmin, bxmax, bxmax, bxmin, bxmin]
        box_y_coords = [bymin, bymin, bymax, bymax, bymin]

        dislocation_locations.append((box_x_coords, box_y_coords))
        dislocation_location_status.append("Successful_fit")
        dislocation_classification_labels.append(classification_label)

        cx_final = (bxmin + bxmax)//2
        cy_final = (bymin + bymax)//2
        centers_of_fits.append((cx_final, cy_final))

        patch_locations.append(None)
        popt_list.append(None)
        pixel_size_list.append(None)

        return background_image, dislocation_image, centers_of_fits, False

    dist_threshold = scan_box_size // 2 

    def box_center(bx):
        xmin, ymin, xmax, ymax = bx
        return ((xmin + xmax)//2, (ymin + ymax)//2)

    adjacency_list = [[] for _ in range(len(all_sub_bounds))]

    for i in range(len(all_sub_bounds)):
        cxi, cyi = box_center(all_sub_bounds[i])
        for j in range(i+1, len(all_sub_bounds)):
            cxj, cyj = box_center(all_sub_bounds[j])
            dist_ij = np.hypot(cxi - cxj, cyi - cyj)
            if dist_ij < dist_threshold:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    visited = [False]*len(all_sub_bounds)
    clusters = []

    for i in range(len(all_sub_bounds)):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            cluster = [i]

            while queue:
                current_idx = queue.pop(0)
                for neighbor in adjacency_list[current_idx]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        cluster.append(neighbor)
            clusters.append(cluster)

    max_size = max(len(c) for c in clusters)
    largest_clusters = [c for c in clusters if len(c) == max_size]

    for cluster_indices in largest_clusters:
        cluster_boxes = [all_sub_bounds[idx] for idx in cluster_indices]
        merged_xmin = min(b[0] for b in cluster_boxes)
        merged_ymin = min(b[1] for b in cluster_boxes)
        merged_xmax = max(b[2] for b in cluster_boxes)
        merged_ymax = max(b[3] for b in cluster_boxes)

        box_x_coords = [merged_xmin, merged_xmax, merged_xmax, merged_xmin, merged_xmin]
        box_y_coords = [merged_ymin, merged_ymin, merged_ymax, merged_ymax, merged_ymin]

        dislocation_locations.append((box_x_coords, box_y_coords))
        dislocation_location_status.append("Successful_fit")
        dislocation_classification_labels.append(classification_label)

        cx_merged = (merged_xmin + merged_xmax)//2
        cy_merged = (merged_ymin + merged_ymax)//2
        centers_of_fits.append((cx_merged, cy_merged))

        patch_locations.append(None)
        popt_list.append(None)
        pixel_size_list.append(None)

    return background_image, dislocation_image, centers_of_fits, False

def process_p4_pole_class(xx, yy, patch, confidence,
                          background_image, dislocation_image,
                          image_array, dislocation_locations, dislocation_location_status,
                          dislocation_classification_labels,
                          centers_of_fits, patch_locations, popt_list, pixel_size_list,
                          rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                          non_dislocation_confidence_scores,
                          base_filename, avg_grad_mag, scan_box_size,
                          step_size, too_close, scan_box_area,
                          fit=True, raw_array=None,
                          classification_label='p4_pole',
                          switch_count=0):
    print(f"[p4_pole] Starting patch ({xx}, {yy}), switch_count={switch_count}")

    args = {
        'current_class': p4_pole_class,
        'switch_count': switch_count
    }

    max_iter = 1
    p4_confirmed = False
    patch_for_this_class = patch
    current_x, current_y = xx, yy
    current_confidence = confidence
    dist_centers = 999999

    for iteration in range(max_iter):
        bounds = locate_dislocation_bounds(
            patch_for_this_class, (current_x, current_y)
        )
        
        if bounds is None:
            return background_image, dislocation_image, centers_of_fits, True

        x_min, y_min, x_max, y_max = bounds

        bx_center = (x_min + x_max)//2
        by_center = (y_min + y_max)//2
        side1 = x_max - x_min
        side2 = y_max - y_min
        largest_side = max(side1, side2, scan_box_size // 3)
        largest_side_ext = int(largest_side * 1.25)

        x_start = bx_center - largest_side_ext
        y_start = by_center - largest_side_ext
        x_end   = bx_center + largest_side_ext
        y_end   = by_center + largest_side_ext

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end   = min(background_image.shape[1], x_end)
        y_end   = min(background_image.shape[0], y_end)

        patch_for_this_class = background_image[y_start:y_end, x_start:x_end]

        new_pred_class, new_conf = classify_patch(patch_for_this_class, model='model2')
        new_pred_class, args['switch_count'] = classification_switch(args, new_pred_class)

        patch_cx = (x_start + x_end)//2
        patch_cy = (y_start + y_end)//2
        new_bounds = locate_dislocation_bounds(
            patch_for_this_class, (x_start, y_start)
        )
        if new_bounds is None:
            return background_image, dislocation_image, centers_of_fits, True
        nx_min, ny_min, nx_max, ny_max = new_bounds

        nbx_center = (nx_min + nx_max)//2
        nby_center = (ny_min + ny_max)//2
        dist_centers = np.sqrt((patch_cx - nbx_center)**2 + (patch_cy - nby_center)**2)

        if new_pred_class == unknown_class:
            return process_unknown_class(
                x_start, y_start, patch_for_this_class, new_conf,
                background_image, dislocation_image,
                image_array, dislocation_locations, dislocation_location_status,
                dislocation_classification_labels,
                centers_of_fits, patch_locations, popt_list, pixel_size_list,
                rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores,
                base_filename, avg_grad_mag, scan_box_size,
                step_size, too_close, scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label='unknown',
                switch_count=args['switch_count']
            )

        if new_pred_class != p4_pole_class:
            print(f"[p4_pole] Changed to {new_pred_class}, switching.")
            new_class_info = CLASS_INFO.get(new_pred_class)
            if new_class_info and new_class_info['process_func']:
                return new_class_info['process_func'](
                    xx=x_start, yy=y_start, patch=patch_for_this_class, confidence=new_conf,
                    background_image=background_image, dislocation_image=dislocation_image,
                    image_array=image_array, dislocation_locations=dislocation_locations,
                    dislocation_location_status=dislocation_location_status,
                    dislocation_classification_labels=dislocation_classification_labels,
                    centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                    popt_list=popt_list, pixel_size_list=pixel_size_list,
                    rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                    approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                    non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                    base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                    scan_box_size=scan_box_size, step_size=step_size,
                    too_close=too_close, scan_box_area=scan_box_area,
                    fit=fit, raw_array=raw_array,
                    classification_label=classification_label_map[new_pred_class],
                    switch_count=args['switch_count']
                )
            else:
                return background_image, dislocation_image, centers_of_fits, True

        p4_confirmed = True
        current_confidence = new_conf
        current_x, current_y = x_start, y_start

        if dist_centers < 1:
            break

    if not p4_confirmed:
        return background_image, dislocation_image, centers_of_fits, True

    final_pred_class, args['switch_count'] = classification_switch(args, p4_pole_class)
    if final_pred_class == unknown_class:
        return process_unknown_class(
            current_x, current_y, patch_for_this_class, current_confidence,
            background_image, dislocation_image,
            image_array, dislocation_locations, dislocation_location_status,
            dislocation_classification_labels,
            centers_of_fits, patch_locations, popt_list, pixel_size_list,
            rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
            non_dislocation_confidence_scores,
            base_filename, avg_grad_mag, scan_box_size,
            step_size, too_close, scan_box_area,
            fit=fit, raw_array=raw_array,
            classification_label='unknown',
            switch_count=args['switch_count']
        )

    if final_pred_class != p4_pole_class:
        new_class_info = CLASS_INFO.get(final_pred_class)
        if new_class_info and new_class_info['process_func']:
            return new_class_info['process_func'](
                xx=current_x, yy=current_y, patch=patch_for_this_class, confidence=current_confidence,
                background_image=background_image, dislocation_image=dislocation_image,
                image_array=image_array, dislocation_locations=dislocation_locations,
                dislocation_location_status=dislocation_location_status,
                dislocation_classification_labels=dislocation_classification_labels,
                centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                popt_list=popt_list, pixel_size_list=pixel_size_list,
                rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                scan_box_size=scan_box_size, step_size=step_size,
                too_close=too_close, scan_box_area=scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label=classification_label_map[final_pred_class],
                switch_count=args['switch_count']
            )
        else:
            return background_image, dislocation_image, centers_of_fits, True

    patch_gray = patch_for_this_class.astype(float)
    gx, gy = np.gradient(patch_gray)
    grad_mag = np.hypot(gx, gy)

    peaks = peak_local_max(
        grad_mag,
        min_distance=4,
        threshold_abs=0.6 * grad_mag.max()
    )

    sub_patch_size = max(5, int(0.5 * scan_box_size))
    half_sub = sub_patch_size // 2

    all_sub_bounds = []
    for (py, px) in peaks:
        global_x = current_x + px
        global_y = current_y + py

        sub_x0 = max(0, global_x - half_sub)
        sub_y0 = max(0, global_y - half_sub)
        sub_x1 = min(background_image.shape[1], global_x + half_sub)
        sub_y1 = min(background_image.shape[0], global_y + half_sub)

        if (sub_x1 - sub_x0) < 5 or (sub_y1 - sub_y0) < 5:
            continue

        sub_patch = background_image[sub_y0:sub_y1, sub_x0:sub_x1]
        sub_bounds = locate_dislocation_bounds(sub_patch, (sub_x0, sub_y0))
        if sub_bounds is None:
            continue

        sbx_min, sby_min, sbx_max, sby_max = sub_bounds

        sub_center_x = (sbx_min + sbx_max) // 2
        sub_center_y = (sby_min + sby_max) // 2
        if any(
            np.sqrt((cx - sub_center_x)**2 + (cy - sub_center_y)**2) < too_close
            for (cx, cy) in centers_of_fits
        ):
            continue

        all_sub_bounds.append((sbx_min, sby_min, sbx_max, sby_max))

    if len(all_sub_bounds) == 0:
        print("[p4_pole] No local peaks => skipping.")
        return background_image, dislocation_image, centers_of_fits, True

    if len(all_sub_bounds) == 1:
        (bxmin, bymin, bxmax, bymax) = all_sub_bounds[0]
        box_x_coords = [bxmin, bxmax, bxmax, bxmin, bxmin]
        box_y_coords = [bymin, bymin, bymax, bymax, bymin]

        dislocation_locations.append((box_x_coords, box_y_coords))
        dislocation_location_status.append("Successful_fit")
        dislocation_classification_labels.append(classification_label)

        cx_final = (bxmin + bxmax)//2
        cy_final = (bymin + bymax)//2
        centers_of_fits.append((cx_final, cy_final))

        patch_locations.append(None)
        popt_list.append(None)
        pixel_size_list.append(None)

        return background_image, dislocation_image, centers_of_fits, False

    dist_threshold = scan_box_size // 2

    def box_center(bx):
        xmin, ymin, xmax, ymax = bx
        return ((xmin + xmax)//2, (ymin + ymax)//2)

    adjacency_list = [[] for _ in range(len(all_sub_bounds))]

    for i in range(len(all_sub_bounds)):
        cxi, cyi = box_center(all_sub_bounds[i])
        for j in range(i+1, len(all_sub_bounds)):
            cxj, cyj = box_center(all_sub_bounds[j])
            dist_ij = np.hypot(cxi - cxj, cyi - cyj)
            if dist_ij < dist_threshold:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    visited = [False]*len(all_sub_bounds)
    clusters = []

    for i in range(len(all_sub_bounds)):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            cluster = [i]

            while queue:
                cur = queue.pop(0)
                for nbr in adjacency_list[cur]:
                    if not visited[nbr]:
                        visited[nbr] = True
                        queue.append(nbr)
                        cluster.append(nbr)
            clusters.append(cluster)

    max_size = max(len(c) for c in clusters)
    largest_clusters = [c for c in clusters if len(c) == max_size]

    for cluster_indices in largest_clusters:
        cluster_boxes = [all_sub_bounds[idx] for idx in cluster_indices]
        merged_xmin = min(b[0] for b in cluster_boxes)
        merged_ymin = min(b[1] for b in cluster_boxes)
        merged_xmax = max(b[2] for b in cluster_boxes)
        merged_ymax = max(b[3] for b in cluster_boxes)

        box_x_coords = [merged_xmin, merged_xmax, merged_xmax, merged_xmin, merged_xmin]
        box_y_coords = [merged_ymin, merged_ymin, merged_ymax, merged_ymax, merged_ymin]

        dislocation_locations.append((box_x_coords, box_y_coords))
        dislocation_location_status.append("Successful_fit")
        dislocation_classification_labels.append(classification_label)

        cx_merged = (merged_xmin + merged_xmax)//2
        cy_merged = (merged_ymin + merged_ymax)//2
        centers_of_fits.append((cx_merged, cy_merged))

        patch_locations.append(None)
        popt_list.append(None)
        pixel_size_list.append(None)

    return background_image, dislocation_image, centers_of_fits, False

def process_no_dislocation(xx, yy, patch, confidence,
                           background_image, dislocation_image,
                           image_array, dislocation_locations, dislocation_location_status,
                           dislocation_classification_labels,
                           centers_of_fits, patch_locations, popt_list, pixel_size_list,
                           rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                           non_dislocation_confidence_scores,
                           base_filename, avg_grad_mag, scan_box_size,
                           step_size, too_close, scan_box_area,
                           fit=True, raw_array=None,
                           classification_label='no_dislocation',
                           switch_count=0):
    return background_image, dislocation_image, centers_of_fits, False


def process_overlapping_dislocation(xx, yy, patch, confidence,
                                    background_image, dislocation_image,
                                    image_array, dislocation_locations, dislocation_location_status,
                                    dislocation_classification_labels,
                                    centers_of_fits, patch_locations, popt_list, pixel_size_list,
                                    rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                                    non_dislocation_confidence_scores,
                                    base_filename, avg_grad_mag, scan_box_size,
                                    step_size, too_close, scan_box_area,
                                    fit=True, raw_array=None,
                                    classification_label='overlapping',
                                    switch_count=0):
    print(f"[Overlapping] Starting at patch ({xx}, {yy}), switch_count={switch_count}")

    args = {
        'current_class': overlapping_class,
        'switch_count': switch_count
    }

    max_iter = 2
    overlapping_confirmed = False

    patch_for_this_class = patch
    current_x, current_y = xx, yy
    current_confidence = confidence
    dist_centers = 999999

    for iteration in range(max_iter):
        bounds = locate_dislocation_bounds(
            patch_for_this_class, (current_x, current_y)
        )
        if bounds is None:
            return background_image, dislocation_image, centers_of_fits, True

        x_min, y_min, x_max, y_max = bounds

        bx_center = (x_min + x_max) // 2
        by_center = (y_min + y_max) // 2

        side1 = x_max - x_min
        side2 = y_max - y_min
        largest_side = max(side1, side2, scan_box_size // 3)
        largest_side_ext = int(largest_side * 1.25)

        x_start = bx_center - largest_side_ext
        y_start = by_center - largest_side_ext
        x_end   = bx_center + largest_side_ext
        y_end   = by_center + largest_side_ext

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end   = min(background_image.shape[1], x_end)
        y_end   = min(background_image.shape[0], y_end)

        patch_for_this_class = background_image[y_start:y_end, x_start:x_end]

        new_pred_class, new_conf = classify_patch(patch_for_this_class, model='model2')
        new_pred_class, args['switch_count'] = classification_switch(args, new_pred_class)

        patch_cx = (x_start + x_end) // 2
        patch_cy = (y_start + y_end) // 2
        new_bounds = locate_dislocation_bounds(
            patch_for_this_class, (x_start, y_start)
        )
        
        if new_bounds is None:
            return background_image, dislocation_image, centers_of_fits, True
        nx_min, ny_min, nx_max, ny_max = new_bounds
        nbx_center = (nx_min + nx_max) // 2
        nby_center = (ny_min + ny_max) // 2
        dist_centers = np.sqrt((patch_cx - nbx_center)**2 + (patch_cy - nby_center)**2)

        if new_pred_class == unknown_class:
            return process_unknown_class(
                x_start, y_start, patch_for_this_class, new_conf,
                background_image, dislocation_image,
                image_array, dislocation_locations, dislocation_location_status,
                dislocation_classification_labels,
                centers_of_fits, patch_locations, popt_list, pixel_size_list,
                rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores,
                base_filename, avg_grad_mag, scan_box_size,
                step_size, too_close, scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label='unknown',
                switch_count=args['switch_count']
            )

        if new_pred_class != overlapping_class:
            print(f"[Overlapping] classification changed => {new_pred_class}, switching out.")
            new_class_info = CLASS_INFO.get(new_pred_class)
            if new_class_info and new_class_info['process_func']:
                return new_class_info['process_func'](
                    xx=x_start, yy=y_start, patch=patch_for_this_class, confidence=new_conf,
                    background_image=background_image, dislocation_image=dislocation_image,
                    image_array=image_array, dislocation_locations=dislocation_locations,
                    dislocation_location_status=dislocation_location_status,
                    dislocation_classification_labels=dislocation_classification_labels,
                    centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                    popt_list=popt_list, pixel_size_list=pixel_size_list,
                    rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                    approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                    non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                    base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                    scan_box_size=scan_box_size, step_size=step_size,
                    too_close=too_close, scan_box_area=scan_box_area,
                    fit=fit, raw_array=raw_array,
                    classification_label=classification_label_map[new_pred_class],
                    switch_count=args['switch_count']
                )
            else:
                return background_image, dislocation_image, centers_of_fits, True

        if not initial_dislocation_checks(
            image_array,
            (x_min, y_min, x_max, y_max),
            scan_box_area,
            scan_box_size,
            dislocation_locations,
            dislocation_location_status,
            avg_grad_mag,
            base_filename,
            grad_factor=3
        ):
            overlapping_confirmed = False
            rejected_dislocation_confidence_scores.append(new_conf)
            return background_image, dislocation_image, centers_of_fits, True

        overlapping_confirmed = True
        current_confidence = new_conf
        current_x, current_y = x_start, y_start

        if dist_centers < 1:
            break

    if not overlapping_confirmed:
        return background_image, dislocation_image, centers_of_fits, True

    final_pred_class, args['switch_count'] = classification_switch(args, overlapping_class)

    if final_pred_class != overlapping_class:
        new_class_info = CLASS_INFO.get(final_pred_class)
        if new_class_info and new_class_info['process_func']:
            return new_class_info['process_func'](
                xx=current_x, yy=current_y, patch=patch_for_this_class, confidence=current_confidence,
                background_image=background_image, dislocation_image=dislocation_image,
                image_array=image_array, dislocation_locations=dislocation_locations,
                dislocation_location_status=dislocation_location_status,
                dislocation_classification_labels=dislocation_classification_labels,
                centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                popt_list=popt_list, pixel_size_list=pixel_size_list,
                rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                scan_box_size=scan_box_size, step_size=step_size,
                too_close=too_close, scan_box_area=scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label=classification_label_map[final_pred_class],
                switch_count=args['switch_count']
            )
        else:
            return background_image, dislocation_image, centers_of_fits, True

    overlap_amount = scan_box_size // 8
    effective_box = scan_box_size - overlap_amount
    half_eff = effective_box // 2

    original_center_x = current_x + scan_box_size // 2
    original_center_y = current_y + scan_box_size // 2

    sx0 = max(0, original_center_x - half_eff)
    sy0 = max(0, original_center_y - half_eff)
    sx1 = min(background_image.shape[1], original_center_x + half_eff)
    sy1 = min(background_image.shape[0], original_center_y + half_eff)

    num_steps = max(4, (sx1 - sx0)//(step_size//2))
    perimeter_points = []
    for i in range(num_steps+1):
        px = sx0 + i*(sx1 - sx0)//num_steps
        py = sy0
        perimeter_points.append((px, py))
    for i in range(num_steps+1):
        px = sx1
        py = sy0 + i*(sy1 - sy0)//num_steps
        perimeter_points.append((px, py))
    for i in range(num_steps+1):
        px = sx1 - i*(sx1 - sx0)//num_steps
        py = sy1
        perimeter_points.append((px, py))
    for i in range(num_steps+1):
        px = sx0
        py = sy1 - i*(sy1 - sy0)//num_steps
        perimeter_points.append((px, py))

    for (px, py) in perimeter_points:
        sub_x0 = max(0, px - scan_box_size//2)
        sub_y0 = max(0, py - scan_box_size//2)
        sub_x1 = min(background_image.shape[1], px + scan_box_size//2)
        sub_y1 = min(background_image.shape[0], py + scan_box_size//2)

        if (sub_x1 - sub_x0) < 5 or (sub_y1 - sub_y0) < 5:
            continue

        test_center_x = sub_x0 + (sub_x1 - sub_x0)//2
        test_center_y = sub_y0 + (sub_y1 - sub_y0)//2
        if any(np.sqrt((cx - test_center_x)**2 + (cy - test_center_y)**2) < too_close for (cx, cy) in centers_of_fits):
            continue

        sub_patch = background_image[sub_y0:sub_y1, sub_x0:sub_x1]

        cclass, cconf = classify_patch(sub_patch, model='model2')
        if cclass == overlapping_class or cclass == no_dislocation_class:
            continue

        if cclass == unknown_class:
            continue

        b_bounds = locate_dislocation_bounds(sub_patch, (sub_x0, sub_y0))
        if b_bounds is None:
            continue
        bx_min, by_min, bx_max, by_max = b_bounds
        
        center_bx = (bx_min + bx_max)//2
        center_by = (by_min + by_max)//2
        too_close_flag = any(
            np.sqrt((fc[0] - center_bx)**2 + (fc[1] - center_by)**2) < too_close
            for fc in centers_of_fits
        )
        if too_close_flag:
            continue

        new_class_info = CLASS_INFO.get(cclass)
        if not new_class_info or not new_class_info['process_func']:
            continue
        
        print(f"[Overlapping->SurroundingsScan] Found distinct class {cclass} at perimeter => calling process function.")
        subpatch_for_class = background_image[by_min:by_max, bx_min:bx_max]

        (background_image, dislocation_image, centers_of_fits, skip_this) = new_class_info['process_func'](
             xx=bx_min, yy=by_min, patch=subpatch_for_class, confidence=cconf,
             background_image=background_image, dislocation_image=dislocation_image,
             image_array=image_array, dislocation_locations=dislocation_locations,
             dislocation_location_status=dislocation_location_status,
             dislocation_classification_labels=dislocation_classification_labels,
             centers_of_fits=centers_of_fits, patch_locations=patch_locations,
             popt_list=popt_list, pixel_size_list=pixel_size_list,
             rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
             approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
             non_dislocation_confidence_scores=non_dislocation_confidence_scores,
             base_filename=base_filename, avg_grad_mag=avg_grad_mag,
             scan_box_size=scan_box_size, step_size=step_size,
             too_close=too_close, scan_box_area=scan_box_area, fit=fit, raw_array=raw_array,
             classification_label=classification_label_map.get(cclass,'unknown'),
             switch_count=args['switch_count']
         )

    patch_gray = patch_for_this_class.astype(float)
    gx, gy = np.gradient(patch_gray)
    grad_mag = np.hypot(gx, gy)

    peaks = peak_local_max(
        grad_mag,
        min_distance=4,
        threshold_abs=0.6 * grad_mag.max()
    )

    sub_patch_size = max(5, int(0.5 * scan_box_size))
    half_sub = sub_patch_size // 2
    overlapping_min_distance = too_close // 4

    new_subpatch_centers = []

    for (py, px) in peaks:
        global_x = current_x + px
        global_y = current_y + py

        sub_x0 = max(0, global_x - half_sub)
        sub_y0 = max(0, global_y - half_sub)
        sub_x1 = min(background_image.shape[1], global_x + half_sub)
        sub_y1 = min(background_image.shape[0], global_y + half_sub)

        if (sub_x1 - sub_x0) < 5 or (sub_y1 - sub_y0) < 5:
            continue

        sub_patch = background_image[sub_y0:sub_y1, sub_x0:sub_x1]
        sub_bounds = locate_dislocation_bounds(sub_patch, (sub_x0, sub_y0))
        if sub_bounds is None:
            continue

        sub_xmin, sub_ymin, sub_xmax, sub_ymax = sub_bounds
        center_x = (sub_xmin + sub_xmax) // 2
        center_y = (sub_ymin + sub_ymax) // 2

        too_close_flag = False
        for (cx, cy) in centers_of_fits:
            dist = np.sqrt((center_x - cx)**2 + (center_y - cy)**2)
            if dist < too_close:
                print(f"[Overlapping] Skipping new sub-patch center ({center_x},{center_y}); dist={dist:.1f}px from existing center ({cx},{cy}).")
                too_close_flag = True
                break

        if not too_close_flag:
            for (cx, cy) in new_subpatch_centers:
                dist = np.sqrt((center_x - cx)**2 + (center_y - cy)**2)
                if dist < overlapping_min_distance:
                    print(f"[Overlapping] Skipping new sub-patch center ({center_x},{center_y}); dist={dist:.1f}px from newly accepted center ({cx},{cy}).")
                    too_close_flag = True
                    break

        if too_close_flag:
            continue

        box_x_coords = [sub_xmin, sub_xmax, sub_xmax, sub_xmin, sub_xmin]
        box_y_coords = [sub_ymin, sub_ymin, sub_ymax, sub_ymax, sub_ymin]

        dislocation_locations.append((box_x_coords, box_y_coords))
        dislocation_location_status.append("Successful_fit")
        dislocation_classification_labels.append(classification_label)

        patch_locations.append(None)
        popt_list.append(None)
        pixel_size_list.append(None)

        centers_of_fits.append((center_x, center_y))
        new_subpatch_centers.append((center_x, center_y))

    return background_image, dislocation_image, centers_of_fits, False

def process_dislocation(xx, yy, patch, confidence,
                        background_image, dislocation_image,
                        image_array, dislocation_locations, dislocation_location_status,
                        dislocation_classification_labels,
                        centers_of_fits, patch_locations, popt_list, pixel_size_list,
                        rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                        non_dislocation_confidence_scores,
                        base_filename, avg_grad_mag, scan_box_size,
                        step_size, too_close, scan_box_area,
                        fit=True, raw_array=None,
                        classification_label='dislocation',
                        switch_count=0):
    print(f"[Dislocation] Starting at patch ({xx}, {yy}), switch_count={switch_count}")

    max_iter = 1
    dislocation_confirmed = False

    args = {
        'current_class': dislocation_class,
        'switch_count': switch_count
    }

    patch_for_this_class = patch
    current_x, current_y = xx, yy
    current_confidence = confidence
    dist_centers = 999999

    for iteration in range(max_iter):
        bounds = locate_dislocation_bounds(
            patch_for_this_class, (current_x, current_y)
        )
        if bounds is None:
            return background_image, dislocation_image, centers_of_fits, True

        x_min, y_min, x_max, y_max = bounds
        bx_center = (x_min + x_max) // 2
        by_center = (y_min + y_max) // 2

        side1 = x_max - x_min
        side2 = y_max - y_min
        largest_side = max(side1, side2, scan_box_size // 3)
        largest_side_ext = int(largest_side * 1.25)

        x_start = bx_center - largest_side_ext
        y_start = by_center - largest_side_ext
        x_end   = bx_center + largest_side_ext
        y_end   = by_center + largest_side_ext

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end   = min(background_image.shape[1], x_end)
        y_end   = min(background_image.shape[0], y_end)

        patch_for_this_class = background_image[y_start:y_end, x_start:x_end]

        new_pred_class, new_conf = classify_patch(patch_for_this_class, model='model2')
        patch_cx = (x_start + x_end) // 2
        patch_cy = (y_start + y_end) // 2
        
        new_bounds = locate_dislocation_bounds(
            patch_for_this_class, (x_start, y_start)
        )
        if new_bounds is None:
            return background_image, dislocation_image, centers_of_fits, True
        nx_min, ny_min, nx_max, ny_max = new_bounds
        nbx_center = (nx_min + nx_max) // 2
        nby_center = (ny_min + ny_max) // 2
        dist_centers = np.sqrt((patch_cx - nbx_center)**2 + (patch_cy - nby_center)**2)

        new_pred_class, args['switch_count'] = classification_switch(args, new_pred_class)

        if new_pred_class == unknown_class:
            return process_unknown_class(
                x_start, y_start, patch_for_this_class, new_conf,
                background_image, dislocation_image,
                image_array, dislocation_locations, dislocation_location_status,
                dislocation_classification_labels,
                centers_of_fits, patch_locations, popt_list, pixel_size_list,
                rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores,
                base_filename, avg_grad_mag, scan_box_size,
                step_size, too_close, scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label='unknown',
                switch_count=args['switch_count']
            )

        if new_pred_class != dislocation_class:
            print(f"[Dislocation] Changed to {new_pred_class} => switching out.")
            new_class_info = CLASS_INFO.get(new_pred_class)
            if new_class_info and new_class_info['process_func']:
                return new_class_info['process_func'](
                    xx=x_start, yy=y_start, patch=patch_for_this_class, confidence=new_conf,
                    background_image=background_image, dislocation_image=dislocation_image,
                    image_array=image_array, dislocation_locations=dislocation_locations,
                    dislocation_location_status=dislocation_location_status,
                    dislocation_classification_labels=dislocation_classification_labels,
                    centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                    popt_list=popt_list, pixel_size_list=pixel_size_list,
                    rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                    approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                    non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                    base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                    scan_box_size=scan_box_size, step_size=step_size,
                    too_close=too_close, scan_box_area=scan_box_area,
                    fit=fit, raw_array=raw_array,
                    classification_label=classification_label_map[new_pred_class],
                    switch_count=args['switch_count']
                )
            return background_image, dislocation_image, centers_of_fits, True

        if not initial_dislocation_checks(
            image_array,
            (x_min, y_min, x_max, y_max),
            scan_box_area,
            scan_box_size,
            dislocation_locations,
            dislocation_location_status,
            avg_grad_mag,
            base_filename,
            grad_factor=3
        ):
            print("[Dislocation] Initial check failed => skipping this patch.")
            dislocation_confirmed = False
            rejected_dislocation_confidence_scores.append(new_conf)
            return background_image, dislocation_image, centers_of_fits, True

        dislocation_confirmed = True
        current_confidence = new_conf
        current_x, current_y = x_start, y_start

        if dist_centers < 1:
            break

    if not dislocation_confirmed:
        return background_image, dislocation_image, centers_of_fits, True

    final_pred_class, args['switch_count'] = classification_switch(args, dislocation_class)
    if final_pred_class == unknown_class:
        return process_unknown_class(
            current_x, current_y, patch_for_this_class, current_confidence,
            background_image, dislocation_image,
            image_array, dislocation_locations, dislocation_location_status,
            dislocation_classification_labels,
            centers_of_fits, patch_locations, popt_list, pixel_size_list,
            rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
            non_dislocation_confidence_scores,
            base_filename, avg_grad_mag, scan_box_size,
            step_size, too_close, scan_box_area,
            fit=fit, raw_array=raw_array,
            classification_label='unknown',
            switch_count=args['switch_count']
        )

    if final_pred_class != dislocation_class:
        new_class_info = CLASS_INFO.get(final_pred_class)
        if new_class_info and new_class_info['process_func']:
            return new_class_info['process_func'](
                xx=current_x, yy=current_y, patch=patch_for_this_class, confidence=current_confidence,
                background_image=background_image, dislocation_image=dislocation_image,
                image_array=image_array, dislocation_locations=dislocation_locations,
                dislocation_location_status=dislocation_location_status,
                dislocation_classification_labels=dislocation_classification_labels,
                centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                popt_list=popt_list, pixel_size_list=pixel_size_list,
                rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                scan_box_size=scan_box_size, step_size=step_size,
                too_close=too_close, scan_box_area=scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label=classification_label_map[final_pred_class],
                switch_count=args['switch_count']
            )
        else:
            return background_image, dislocation_image, centers_of_fits, True

    bx_center = (x_min + x_max) // 2
    by_center = (y_min + y_max) // 2
    if any(np.sqrt((cx - bx_center)**2 + (cy - by_center)**2) < too_close
           for (cx, cy) in centers_of_fits):
        return background_image, dislocation_image, centers_of_fits, True

    main_conf = current_confidence
    closeby_centers, closeby_bounds, closeby_confidences = check_for_closeby_dislocations(
        background_image,
        bx_center, by_center,
        largest_side, 
        dislocation_locations, dislocation_location_status,
        avg_grad_mag, centers_of_fits,
        base_filename,
        approved_dislocation_confidence_scores,
        rejected_dislocation_confidence_scores,
        non_dislocation_confidence_scores,
        scan_box_size=scan_box_size
    )

    multi_bounds = [(x_min, y_min, x_max, y_max)]
    candidate_confidences = [main_conf]
    if closeby_bounds is not None:
        multi_bounds.extend(closeby_bounds)
        candidate_confidences.extend(closeby_confidences)

    result = fit_process_multiple_dislocations(
        raw_array, background_image, dislocation_image,
        closeby_centers, multi_bounds, centers_of_fits,
        dislocation_locations, dislocation_location_status,
        dislocation_classification_labels,
        patch_locations, popt_list, pixel_size_list,
        candidate_confidences, approved_dislocation_confidence_scores,
        fit=fit, classification_label=classification_label
    )
    if result is not None:
        new_bg, new_disloc_img, new_centers = result
        if new_bg is not None and new_disloc_img is not None:
            background_image = new_bg
            dislocation_image = new_disloc_img
            centers_of_fits = new_centers

    return background_image, dislocation_image, centers_of_fits, False

def process_multiple_dislocation(xx, yy, patch, confidence,
                                 background_image, dislocation_image,
                                 image_array, dislocation_locations, dislocation_location_status,
                                 dislocation_classification_labels,
                                 centers_of_fits, patch_locations, popt_list, pixel_size_list,
                                 rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                                 non_dislocation_confidence_scores,
                                 base_filename, avg_grad_mag, scan_box_size,
                                 step_size, too_close, scan_box_area,
                                 fit=True, raw_array=None,
                                 classification_label='multiple',
                                 switch_count=0):
    print(f"[Multiple] Starting patch ({xx}, {yy}), switch_count={switch_count}")

    args = {
        'current_class': multiple_dislocations_class,
        'switch_count': switch_count
    }

    max_iter = 1
    multiple_confirmed = False
    patch_for_this_class = patch
    current_x, current_y = xx, yy
    current_confidence = confidence
    dist_centers = 999999

    for iteration in range(max_iter):
        bounds = locate_dislocation_bounds(
            patch_for_this_class, (current_x, current_y)
        )
        if bounds is None:
            return background_image, dislocation_image, centers_of_fits, True

        x_min, y_min, x_max, y_max = bounds
        bx_center = (x_min + x_max) // 2
        by_center = (y_min + y_max) // 2
        side1 = x_max - x_min
        side2 = y_max - y_min
        largest_side = max(side1, side2, scan_box_size // 3)
        largest_side_ext = int(largest_side * 1.25)

        x_start = bx_center - largest_side_ext
        y_start = by_center - largest_side_ext
        x_end   = bx_center + largest_side_ext
        y_end   = by_center + largest_side_ext

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end   = min(background_image.shape[1], x_end)
        y_end   = min(background_image.shape[0], y_end)

        patch_for_this_class = background_image[y_start:y_end, x_start:x_end]

        new_pred_class, new_conf = classify_patch(patch_for_this_class, model='model2')
        new_pred_class, args['switch_count'] = classification_switch(args, new_pred_class)        

        patch_cx = (x_start + x_end)//2
        patch_cy = (y_start + y_end)//2
        new_bounds = locate_dislocation_bounds(
            patch_for_this_class, (x_start, y_start)
        )
        if new_bounds is None:
            return background_image, dislocation_image, centers_of_fits, True

        nx_min, ny_min, nx_max, ny_max = new_bounds
        nbx_center = (nx_min + nx_max)//2
        nby_center = (ny_min + ny_max)//2
        dist_centers = np.sqrt((patch_cx - nbx_center)**2 + (patch_cy - nby_center)**2)

        if new_pred_class == unknown_class:
            return process_unknown_class(
                x_start, y_start, patch_for_this_class, new_conf,
                background_image, dislocation_image,
                image_array, dislocation_locations, dislocation_location_status,
                dislocation_classification_labels,
                centers_of_fits, patch_locations, popt_list, pixel_size_list,
                rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores,
                base_filename, avg_grad_mag, scan_box_size,
                step_size, too_close, scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label='unknown',
                switch_count=args['switch_count']
            )

        if new_pred_class != multiple_dislocations_class:
            print(f"[Multiple] Changed to {new_pred_class}, switching out.")
            new_class_info = CLASS_INFO.get(new_pred_class)
            if new_class_info and new_class_info['process_func']:
                return new_class_info['process_func'](
                    xx=x_start, yy=y_start, patch=patch_for_this_class, confidence=new_conf,
                    background_image=background_image, dislocation_image=dislocation_image,
                    image_array=image_array, dislocation_locations=dislocation_locations,
                    dislocation_location_status=dislocation_location_status,
                    dislocation_classification_labels=dislocation_classification_labels,
                    centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                    popt_list=popt_list, pixel_size_list=pixel_size_list,
                    rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                    approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                    non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                    base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                    scan_box_size=scan_box_size, step_size=step_size,
                    too_close=too_close, scan_box_area=scan_box_area,
                    fit=fit, raw_array=raw_array,
                    classification_label=classification_label_map[new_pred_class],
                    switch_count=args['switch_count']
                )
            return background_image, dislocation_image, centers_of_fits, True

        if not initial_dislocation_checks(
            image_array,
            (x_min, y_min, x_max, y_max),
            scan_box_area, scan_box_size,
            dislocation_locations, dislocation_location_status,
            avg_grad_mag, base_filename,
            grad_factor=3
        ):
            multiple_confirmed = False
            rejected_dislocation_confidence_scores.append(new_conf)
            return background_image, dislocation_image, centers_of_fits, True

        multiple_confirmed = True
        current_confidence = new_conf
        current_x, current_y = x_start, y_start

        if dist_centers < 1:
            break

    if not multiple_confirmed:
        return background_image, dislocation_image, centers_of_fits, True

    final_pred_class, args['switch_count'] = classification_switch(args, multiple_dislocations_class)
    if final_pred_class == unknown_class:
        return process_unknown_class(
            current_x, current_y, patch_for_this_class, current_confidence,
            background_image, dislocation_image,
            image_array, dislocation_locations, dislocation_location_status,
            dislocation_classification_labels,
            centers_of_fits, patch_locations, popt_list, pixel_size_list,
            rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
            non_dislocation_confidence_scores,
            base_filename, avg_grad_mag, scan_box_size,
            step_size, too_close, scan_box_area,
            fit=fit, raw_array=raw_array,
            classification_label='unknown',
            switch_count=args['switch_count']
        )

    if final_pred_class != multiple_dislocations_class:
        new_class_info = CLASS_INFO.get(final_pred_class)
        if new_class_info and new_class_info['process_func']:
            return new_class_info['process_func'](
                xx=current_x, yy=current_y, patch=patch_for_this_class, confidence=current_confidence,
                background_image=background_image, dislocation_image=dislocation_image,
                image_array=image_array, dislocation_locations=dislocation_locations,
                dislocation_location_status=dislocation_location_status,
                dislocation_classification_labels=dislocation_classification_labels,
                centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                popt_list=popt_list, pixel_size_list=pixel_size_list,
                rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                scan_box_size=scan_box_size, step_size=step_size,
                too_close=too_close, scan_box_area=scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label=classification_label_map[final_pred_class],
                switch_count=args['switch_count']
            )
        else:
            return background_image, dislocation_image, centers_of_fits, True

    bx_center = (x_min + x_max)//2
    by_center = (y_min + y_max)//2

    if any(np.sqrt((cx - bx_center)**2 + (cy - by_center)**2) < too_close
           for (cx, cy) in centers_of_fits):
        return background_image, dislocation_image, centers_of_fits, True

    main_conf = current_confidence
    closeby_dis_centers, closeby_dis_bounds, closeby_confidences = check_for_closeby_dislocations(
        background_image,
        bx_center, by_center,
        largest_side,
        dislocation_locations, dislocation_location_status,
        avg_grad_mag, centers_of_fits,
        base_filename,
        approved_dislocation_confidence_scores,
        rejected_dislocation_confidence_scores,
        non_dislocation_confidence_scores,
        scan_box_size=scan_box_size
    )

    multi_bounds = [(x_min, y_min, x_max, y_max)]
    candidate_confidences = [main_conf]
    if closeby_dis_bounds:
        multi_bounds.extend(closeby_dis_bounds)
        candidate_confidences.extend(closeby_confidences)

    result = fit_process_multiple_dislocations(
        raw_array, background_image, dislocation_image,
        closeby_dis_centers, multi_bounds, centers_of_fits,
        dislocation_locations, dislocation_location_status,
        dislocation_classification_labels,
        patch_locations, popt_list, pixel_size_list,
        candidate_confidences, approved_dislocation_confidence_scores,
        fit=fit, classification_label=classification_label
    )
    if result is not None:
        new_bg, new_disloc_img, new_centers = result
        if new_bg is not None and new_disloc_img is not None:
            background_image = new_bg
            dislocation_image = new_disloc_img
            centers_of_fits = new_centers

    return background_image, dislocation_image, centers_of_fits, False

def process_faint_dislocation(xx, yy, patch, confidence,
                              background_image, dislocation_image,
                              image_array, dislocation_locations, dislocation_location_status,
                              dislocation_classification_labels,
                              centers_of_fits, patch_locations, popt_list, pixel_size_list,
                              rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                              non_dislocation_confidence_scores,
                              base_filename, avg_grad_mag, scan_box_size,
                              step_size, too_close, scan_box_area,
                              fit=True, raw_array=None,
                              classification_label='faint',
                              switch_count=0):
    print(f"[Faint] Starting patch ({xx}, {yy}), switch_count={switch_count}")

    args = {
        'current_class': faint_dislocation_class,
        'switch_count': switch_count
    }

    max_iter = 1
    faint_confirmed = False
    patch_for_this_class = patch
    current_x, current_y = xx, yy
    current_confidence = confidence
    dist_centers = 999999

    for iteration in range(max_iter):
        bounds = locate_dislocation_bounds(
            patch_for_this_class, (current_x, current_y)
        )
        if bounds is None:
            return background_image, dislocation_image, centers_of_fits, True

        x_min, y_min, x_max, y_max = bounds
        bx_center = (x_min + x_max)//2
        by_center = (y_min + y_max)//2
        side1 = x_max - x_min
        side2 = y_max - y_min
        largest_side = max(side1, side2, scan_box_size // 3)
        largest_side_ext = int(largest_side * 1.25)

        x_start = bx_center - largest_side_ext
        y_start = by_center - largest_side_ext
        x_end   = bx_center + largest_side_ext
        y_end   = by_center + largest_side_ext

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end   = min(background_image.shape[1], x_end)
        y_end   = min(background_image.shape[0], y_end)

        patch_for_this_class = background_image[y_start:y_end, x_start:x_end]

        new_pred_class, new_conf = classify_patch(patch_for_this_class, model='model2')
        new_pred_class, args['switch_count'] = classification_switch(args, new_pred_class)

        patch_cx = (x_start + x_end)//2
        patch_cy = (y_start + y_end)//2
        new_bounds = locate_dislocation_bounds(
            patch_for_this_class, (x_start, y_start)
        )
        if new_bounds is None:
            return background_image, dislocation_image, centers_of_fits, True

        nx_min, ny_min, nx_max, ny_max = new_bounds
        nbx_center = (nx_min + nx_max)//2
        nby_center = (ny_min + ny_max)//2
        dist_centers = np.sqrt((patch_cx - nbx_center)**2 + (patch_cy - nby_center)**2)

        if new_pred_class == unknown_class:
            return process_unknown_class(
                x_start, y_start, patch_for_this_class, new_conf,
                background_image, dislocation_image,
                image_array, dislocation_locations, dislocation_location_status,
                dislocation_classification_labels,
                centers_of_fits, patch_locations, popt_list, pixel_size_list,
                rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores,
                base_filename, avg_grad_mag, scan_box_size,
                step_size, too_close, scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label='unknown',
                switch_count=args['switch_count']
            )

        if new_pred_class not in [faint_dislocation_class]:
            print(f"[Faint] Changed to {new_pred_class}, switching out.")
            new_class_info = CLASS_INFO.get(new_pred_class)
            if new_class_info and new_class_info['process_func']:
                return new_class_info['process_func'](
                    xx=x_start, yy=y_start, patch=patch_for_this_class, confidence=new_conf,
                    background_image=background_image, dislocation_image=dislocation_image,
                    image_array=image_array, dislocation_locations=dislocation_locations,
                    dislocation_location_status=dislocation_location_status,
                    dislocation_classification_labels=dislocation_classification_labels,
                    centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                    popt_list=popt_list, pixel_size_list=pixel_size_list,
                    rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                    approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                    non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                    base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                    scan_box_size=scan_box_size, step_size=step_size,
                    too_close=too_close, scan_box_area=scan_box_area,
                    fit=fit, raw_array=raw_array,
                    classification_label=classification_label_map[new_pred_class],
                    switch_count=args['switch_count']
                )
            return background_image, dislocation_image, centers_of_fits, True

        if not initial_dislocation_checks(
            image_array,
            (x_min, y_min, x_max, y_max),
            scan_box_area, scan_box_size,
            dislocation_locations, dislocation_location_status,
            avg_grad_mag, base_filename,
            grad_factor=1.7
        ):
            faint_confirmed = False
            rejected_dislocation_confidence_scores.append(new_conf)
            return background_image, dislocation_image, centers_of_fits, True

        faint_confirmed = True
        current_confidence = new_conf
        current_x, current_y = x_start, y_start

        if dist_centers < 1:
            break

    if not faint_confirmed:
        return background_image, dislocation_image, centers_of_fits, True

    final_pred_class, args['switch_count'] = classification_switch(args, faint_dislocation_class)
    if final_pred_class == unknown_class:
        return process_unknown_class(
            current_x, current_y, patch_for_this_class, current_confidence,
            background_image, dislocation_image,
            image_array, dislocation_locations, dislocation_location_status,
            dislocation_classification_labels,
            centers_of_fits, patch_locations, popt_list, pixel_size_list,
            rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
            non_dislocation_confidence_scores,
            base_filename, avg_grad_mag, scan_box_size,
            step_size, too_close, scan_box_area,
            fit=fit, raw_array=raw_array,
            classification_label='unknown',
            switch_count=args['switch_count']
        )

    if final_pred_class != faint_dislocation_class:
        new_class_info = CLASS_INFO.get(final_pred_class)
        if new_class_info and new_class_info['process_func']:
            return new_class_info['process_func'](
                xx=current_x, yy=current_y, patch=patch_for_this_class, confidence=current_confidence,
                background_image=background_image, dislocation_image=dislocation_image,
                image_array=image_array, dislocation_locations=dislocation_locations,
                dislocation_location_status=dislocation_location_status,
                dislocation_classification_labels=dislocation_classification_labels,
                centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                popt_list=popt_list, pixel_size_list=pixel_size_list,
                rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                scan_box_size=scan_box_size, step_size=step_size,
                too_close=too_close, scan_box_area=scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label=classification_label_map[final_pred_class],
                switch_count=args['switch_count']
            )
        else:
            return background_image, dislocation_image, centers_of_fits, True

    bx_center = (x_min + x_max)//2
    by_center = (y_min + y_max)//2

    if any(np.sqrt((cx - bx_center)**2 + (cy - by_center)**2) < too_close
           for (cx, cy) in centers_of_fits):
        return background_image, dislocation_image, centers_of_fits, True

    main_conf = current_confidence
    closeby_dis_centers, closeby_dis_bounds, closeby_confidences = check_for_closeby_dislocations(
        background_image,
        bx_center, by_center,
        largest_side,
        dislocation_locations, dislocation_location_status,
        avg_grad_mag, centers_of_fits,
        base_filename,
        approved_dislocation_confidence_scores,
        rejected_dislocation_confidence_scores,
        non_dislocation_confidence_scores,
        scan_box_size=scan_box_size
    )

    multi_bounds = [(x_min, y_min, x_max, y_max)]
    candidate_confidences = [main_conf]
    if closeby_dis_bounds:
        multi_bounds.extend(closeby_dis_bounds)
        candidate_confidences.extend(closeby_confidences)

    result = fit_process_multiple_dislocations(
        raw_array, background_image, dislocation_image,
        closeby_dis_centers, multi_bounds, centers_of_fits,
        dislocation_locations, dislocation_location_status,
        dislocation_classification_labels,
        patch_locations, popt_list, pixel_size_list,
        candidate_confidences, approved_dislocation_confidence_scores,
        fit=fit, classification_label=classification_label
    )
    if result:
        new_bg, new_disloc_img, new_centers = result
        if new_bg is not None and new_disloc_img is not None:
            background_image = new_bg
            dislocation_image = new_disloc_img
            centers_of_fits = new_centers

    return background_image, dislocation_image, centers_of_fits, False

def process_ultra_faint_dislocation(xx, yy, patch, confidence,
                                    background_image, dislocation_image,
                                    image_array, dislocation_locations, dislocation_location_status,
                                    dislocation_classification_labels,
                                    centers_of_fits, patch_locations, popt_list, pixel_size_list,
                                    rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                                    non_dislocation_confidence_scores,
                                    base_filename, avg_grad_mag, scan_box_size,
                                    step_size, too_close, scan_box_area,
                                    fit=True, raw_array=None,
                                    classification_label='ultra_faint',
                                    switch_count=0):
    print(f"[Ultra Faint] Starting patch ({xx}, {yy}), switch_count={switch_count}")

    args = {
        'current_class': ultra_faint_dislocation_class,
        'switch_count': switch_count
    }

    max_iter = 1
    faint_confirmed = False
    patch_for_this_class = patch
    current_x, current_y = xx, yy
    current_confidence = confidence
    dist_centers = 999999

    for iteration in range(max_iter):
        bounds = locate_dislocation_bounds(
            patch_for_this_class, (current_x, current_y)
        )
        if bounds is None:
            return background_image, dislocation_image, centers_of_fits, True

        x_min, y_min, x_max, y_max = bounds
        bx_center = (x_min + x_max)//2
        by_center = (y_min + y_max)//2
        side1 = x_max - x_min
        side2 = y_max - y_min
        largest_side = max(side1, side2, scan_box_size // 3)
        largest_side_ext = int(largest_side * 1.25)

        x_start = bx_center - largest_side_ext
        y_start = by_center - largest_side_ext
        x_end   = bx_center + largest_side_ext
        y_end   = by_center + largest_side_ext

        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end   = min(background_image.shape[1], x_end)
        y_end   = min(background_image.shape[0], y_end)

        patch_for_this_class = background_image[y_start:y_end, x_start:x_end]

        new_pred_class, new_conf = classify_patch(patch_for_this_class, model='model2')
        new_pred_class, args['switch_count'] = classification_switch(args, new_pred_class)

        patch_cx = (x_start + x_end)//2
        patch_cy = (y_start + y_end)//2
        new_bounds = locate_dislocation_bounds(
            patch_for_this_class, (x_start, y_start)
        )
        if new_bounds is None:
            return background_image, dislocation_image, centers_of_fits, True

        nx_min, ny_min, nx_max, ny_max = new_bounds
        nbx_center = (nx_min + nx_max)//2
        nby_center = (ny_min + ny_max)//2
        dist_centers = np.sqrt((patch_cx - nbx_center)**2 + (patch_cy - nby_center)**2)

        if new_pred_class == unknown_class:
            return process_unknown_class(
                x_start, y_start, patch_for_this_class, new_conf,
                background_image, dislocation_image,
                image_array, dislocation_locations, dislocation_location_status,
                dislocation_classification_labels,
                centers_of_fits, patch_locations, popt_list, pixel_size_list,
                rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores,
                base_filename, avg_grad_mag, scan_box_size,
                step_size, too_close, scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label='unknown',
                switch_count=args['switch_count']
            )

        if new_pred_class not in [ultra_faint_dislocation_class]:
            print(f"[Ultra Faint] Changed to {new_pred_class}, switching out.")
            new_class_info = CLASS_INFO.get(new_pred_class)
            if new_class_info and new_class_info['process_func']:
                return new_class_info['process_func'](
                    xx=x_start, yy=y_start, patch=patch_for_this_class, confidence=new_conf,
                    background_image=background_image, dislocation_image=dislocation_image,
                    image_array=image_array, dislocation_locations=dislocation_locations,
                    dislocation_location_status=dislocation_location_status,
                    dislocation_classification_labels=dislocation_classification_labels,
                    centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                    popt_list=popt_list, pixel_size_list=pixel_size_list,
                    rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                    approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                    non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                    base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                    scan_box_size=scan_box_size, step_size=step_size,
                    too_close=too_close, scan_box_area=scan_box_area,
                    fit=fit, raw_array=raw_array,
                    classification_label=classification_label_map[new_pred_class],
                    switch_count=args['switch_count']
                )
            return background_image, dislocation_image, centers_of_fits, True

        if not initial_dislocation_checks(
            image_array,
            (x_min, y_min, x_max, y_max),
            scan_box_area, scan_box_size,
            dislocation_locations, dislocation_location_status,
            avg_grad_mag, base_filename,
            grad_factor=1.7
        ):
            faint_confirmed = False
            rejected_dislocation_confidence_scores.append(new_conf)
            return background_image, dislocation_image, centers_of_fits, True

        faint_confirmed = True
        current_confidence = new_conf
        current_x, current_y = x_start, y_start

        if dist_centers < 1:
            break

    if not faint_confirmed:
        return background_image, dislocation_image, centers_of_fits, True

    final_pred_class, args['switch_count'] = classification_switch(args, ultra_faint_dislocation_class)
    if final_pred_class == unknown_class:
        return process_unknown_class(
            current_x, current_y, patch_for_this_class, current_confidence,
            background_image, dislocation_image,
            image_array, dislocation_locations, dislocation_location_status,
            dislocation_classification_labels,
            centers_of_fits, patch_locations, popt_list, pixel_size_list,
            rejected_dislocation_confidence_scores, approved_dislocation_confidence_scores,
            non_dislocation_confidence_scores,
            base_filename, avg_grad_mag, scan_box_size,
            step_size, too_close, scan_box_area,
            fit=fit, raw_array=raw_array,
            classification_label='unknown',
            switch_count=args['switch_count']
        )

    if final_pred_class != ultra_faint_dislocation_class:
        new_class_info = CLASS_INFO.get(final_pred_class)
        if new_class_info and new_class_info['process_func']:
            return new_class_info['process_func'](
                xx=current_x, yy=current_y, patch=patch_for_this_class, confidence=current_confidence,
                background_image=background_image, dislocation_image=dislocation_image,
                image_array=image_array, dislocation_locations=dislocation_locations,
                dislocation_location_status=dislocation_location_status,
                dislocation_classification_labels=dislocation_classification_labels,
                centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                popt_list=popt_list, pixel_size_list=pixel_size_list,
                rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                scan_box_size=scan_box_size, step_size=step_size,
                too_close=too_close, scan_box_area=scan_box_area,
                fit=fit, raw_array=raw_array,
                classification_label=classification_label_map[final_pred_class],
                switch_count=args['switch_count']
            )
        else:
            return background_image, dislocation_image, centers_of_fits, True

    bx_center = (x_min + x_max)//2
    by_center = (y_min + y_max)//2

    if any(np.sqrt((cx - bx_center)**2 + (cy - by_center)**2) < too_close
           for (cx, cy) in centers_of_fits):
        return background_image, dislocation_image, centers_of_fits, True

    main_conf = current_confidence
    closeby_dis_centers, closeby_dis_bounds, closeby_confidences = check_for_closeby_dislocations(
        background_image,
        bx_center, by_center,
        largest_side,
        dislocation_locations, dislocation_location_status,
        avg_grad_mag, centers_of_fits,
        base_filename,
        approved_dislocation_confidence_scores,
        rejected_dislocation_confidence_scores,
        non_dislocation_confidence_scores,
        scan_box_size=scan_box_size
    )

    multi_bounds = [(x_min, y_min, x_max, y_max)]
    candidate_confidences = [main_conf]
    if closeby_dis_bounds:
        multi_bounds.extend(closeby_dis_bounds)
        candidate_confidences.extend(closeby_confidences)

    result = fit_process_multiple_dislocations(
        raw_array, background_image, dislocation_image,
        closeby_dis_centers, multi_bounds, centers_of_fits,
        dislocation_locations, dislocation_location_status,
        dislocation_classification_labels,
        patch_locations, popt_list, pixel_size_list,
        candidate_confidences, approved_dislocation_confidence_scores,
        fit=fit, classification_label=classification_label
    )
    if result:
        new_bg, new_disloc_img, new_centers = result
        if new_bg is not None and new_disloc_img is not None:
            background_image = new_bg
            dislocation_image = new_disloc_img
            centers_of_fits = new_centers

    return background_image, dislocation_image, centers_of_fits, False


# Class routing dictionary

CLASS_INFO = {
    p3_pole_class: {
        'name': 'p3_pole_class',
        'color': 'magenta',
        'draw_patch': True,
        'process_func': process_p3_pole_class
    },
    p4_pole_class: {
        'name': 'p4_pole_class',
        'color': 'cyan',
        'draw_patch': True,
        'process_func': process_p4_pole_class
    },
    dislocation_class: {
        'name': 'dislocation_class',
        'color': 'red',
        'draw_patch': True,
        'process_func': process_dislocation
    },
    multiple_dislocations_class: {
        'name': 'multiple_dislocations_class',
        'color': 'blue',
        'draw_patch': True,
        'process_func': process_multiple_dislocation
    },
    no_dislocation_class: {
        'name': 'no_dislocation_class',
        'color': 'brown',
        'draw_patch': False,
        'process_func': process_no_dislocation
    },
    overlapping_class: {
        'name': 'overlapping_class',
        'color': 'yellow',
        'draw_patch': True,
        'process_func': process_overlapping_dislocation
    },
    faint_dislocation_class: {
        'name': 'faint_dislocation_class',
        'color': 'purple',
        'draw_patch': True,
        'process_func': process_faint_dislocation
    },
    ultra_faint_dislocation_class: {
        'name': 'ultra_faint_dislocation_class',
        'color': 'gray',
        'draw_patch': True,
        'process_func': process_ultra_faint_dislocation
    },
    unknown_class: {
        'name': 'unknown_class',
        'color': 'white',
        'draw_patch': True,
        'process_func': process_unknown_class
    }
}

# Bounding box merging

def compute_iou(boxA, boxB):
    """Computes intersection-over-union metric for resolving overlap duplicates."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxA_area < 1e-9 or boxB_area < 1e-9:
        return 0.0

    union_area = boxA_area + boxB_area - interArea
    return interArea / union_area

def merge_overlapping_unknowns(
    disloc_locs,
    disloc_labels,
    disloc_status,
    patch_locs,
    popt_list,
    pix_size_list,
    overlap_threshold=0.5
):
    """
    Merges all 'unknown' bounding boxes that overlap (IoU >= overlap_threshold)
    transitively. Known boxes are returned unchanged.
    """
    unknown_indices = [
        i for i, lbl in enumerate(disloc_labels)
        if lbl == 'unknown'
    ]
    known_indices = [
        i for i, lbl in enumerate(disloc_labels)
        if lbl != 'unknown'
    ]

    if not unknown_indices:
        return (
            disloc_locs,
            disloc_labels,
            disloc_status,
            patch_locs,
            popt_list,
            pix_size_list
        )

    unknown_boxes_xyxy = []
    for idx in unknown_indices:
        xs, ys = disloc_locs[idx]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        unknown_boxes_xyxy.append([x_min, y_min, x_max, y_max])

    n = len(unknown_boxes_xyxy)
    adjacency_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            iou_val = compute_iou(unknown_boxes_xyxy[i], unknown_boxes_xyxy[j])
            if iou_val >= overlap_threshold:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    visited = [False]*n
    merged_xyxy = []

    for start in range(n):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        component = [start]

        while queue:
            curr = queue.pop(0)
            for nbr in adjacency_list[curr]:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
                    component.append(nbr)

        c_xmin = min(unknown_boxes_xyxy[k][0] for k in component)
        c_ymin = min(unknown_boxes_xyxy[k][1] for k in component)
        c_xmax = max(unknown_boxes_xyxy[k][2] for k in component)
        c_ymax = max(unknown_boxes_xyxy[k][3] for k in component)
        merged_xyxy.append([c_xmin, c_ymin, c_xmax, c_ymax])

    merged_unknown_locs = []
    for (xmin, ymin, xmax, ymax) in merged_xyxy:
        merged_unknown_locs.append((
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin]
        ))

    new_disloc_locs = []
    new_disloc_labels = []
    new_disloc_status = []
    new_patch_locs = []
    new_popt_list = []
    new_pix_size_list = []

    for ki in known_indices:
        new_disloc_locs.append(disloc_locs[ki])
        new_disloc_labels.append(disloc_labels[ki])
        new_disloc_status.append(disloc_status[ki])
        new_patch_locs.append(patch_locs[ki])
        new_popt_list.append(popt_list[ki])
        new_pix_size_list.append(pix_size_list[ki])

    for loc in merged_unknown_locs:
        new_disloc_locs.append(loc)
        new_disloc_labels.append('unknown')
        new_disloc_status.append('Failed_fit')
        new_patch_locs.append(None)
        new_popt_list.append(None)
        new_pix_size_list.append(None)

    return (
        new_disloc_locs,
        new_disloc_labels,
        new_disloc_status,
        new_patch_locs,
        new_popt_list,
        new_pix_size_list
    )


# YOLO & tile scanning engines

def process_image(raw_array, image_array, base_filename, image_counters,
                  save_patches=False, fit=True, scan_box_size=50,
                  image_scan_progress_bar=None):
    """
    Main CNN sliding window scanning routine. Each patch is classified, checked 
    against CLASS_INFO, and processed by its designated bounding/fitting function.
    """
    copy_to_draw_on = image_array.copy()
    dislocation_image = np.zeros_like(image_array, dtype=np.float32)
    background_image = image_array.copy()

    class_rectangles = {c: [] for c in CLASS_INFO.keys()}

    dislocation_locations = []
    dislocation_location_status = []
    dislocation_classification_labels = []
    centers_of_fits = []
    patch_locations = []
    popt_list = []
    pixel_size_list = []

    rejected_dislocation_confidence_scores = []
    approved_dislocation_confidence_scores = []
    non_dislocation_confidence_scores = []

    grad_mag, _, _ = calculate_gradient(image_array)
    avg_grad_mag = np.mean(grad_mag)
    print(f"Average gradient magnitude: {avg_grad_mag}")

    too_close = scan_box_size // 3
    scan_box_area = scan_box_size * scan_box_size
    step_size = scan_box_size // 8

    edge_buffer_factor = scan_box_size
    y_start = int(scan_box_size // edge_buffer_factor)
    y_end   = background_image.shape[0] - scan_box_size - int(scan_box_size // edge_buffer_factor)
    if y_end < y_start:
        y_end = y_start

    x_start = int(scan_box_size // edge_buffer_factor)
    x_end   = background_image.shape[1] - scan_box_size - int(scan_box_size // edge_buffer_factor)
    if x_end < x_start:
        x_end = x_start

    total_steps_y = 1 + max(0, (y_end - y_start)//step_size)
    total_steps_x = 1 + max(0, (x_end - x_start)//step_size)
    total_patches = total_steps_y * total_steps_x
    patches_done = 0

    for yy in range(y_start, y_end+1, step_size):
        for xx in range(x_start, x_end+1, step_size):
            patches_done += 1

            if image_scan_progress_bar is not None and total_patches > 0:
                pct = int((patches_done / total_patches)*100)
                image_scan_progress_bar.setValue(pct)
                QApplication.processEvents()

            if any(np.sqrt((cx - (xx + scan_box_size//2))**2 + (cy - (yy + scan_box_size//2))**2) < too_close
                   for (cx, cy) in centers_of_fits):
                continue

            patch = background_image[yy:yy+scan_box_size, xx:xx+scan_box_size]
            predicted_class, confidence = classify_patch(patch, model='model2')

            class_info = CLASS_INFO.get(predicted_class)
            if not class_info:
                predicted_class = unknown_class
                class_info = CLASS_INFO[unknown_class]

            classification_label = classification_label_map.get(predicted_class, 'unknown')

            if class_info['draw_patch']:
                rect = ([xx, xx+scan_box_size, xx+scan_box_size, xx, xx],
                        [yy, yy, yy+scan_box_size, yy+scan_box_size, yy])
                class_rectangles[predicted_class].append(rect)

            skip_this_patch = False
            if class_info['process_func'] is not None:
                (background_image,
                 dislocation_image,
                 centers_of_fits,
                 skip_this_patch) = class_info['process_func'](
                    xx=xx, yy=yy, patch=patch, confidence=confidence,
                    background_image=background_image, dislocation_image=dislocation_image,
                    image_array=image_array, dislocation_locations=dislocation_locations,
                    dislocation_location_status=dislocation_location_status,
                    dislocation_classification_labels=dislocation_classification_labels,
                    centers_of_fits=centers_of_fits, patch_locations=patch_locations,
                    popt_list=popt_list, pixel_size_list=pixel_size_list,
                    rejected_dislocation_confidence_scores=rejected_dislocation_confidence_scores,
                    approved_dislocation_confidence_scores=approved_dislocation_confidence_scores,
                    non_dislocation_confidence_scores=non_dislocation_confidence_scores,
                    base_filename=base_filename, avg_grad_mag=avg_grad_mag,
                    scan_box_size=scan_box_size, step_size=step_size,
                    too_close=too_close, scan_box_area=scan_box_area,
                    fit=fit, raw_array=raw_array,
                    classification_label=classification_label,
                    switch_count=0
                )

            if skip_this_patch:
                continue

    return (
        image_array,
        copy_to_draw_on,
        dislocation_image,
        background_image,
        dislocation_locations,
        dislocation_location_status,
        dislocation_classification_labels,
        centers_of_fits,
        patch_locations,
        popt_list,
        pixel_size_list,
        class_rectangles
    )


_yolo_model_cache = {}

def _get_yolo(model_path: str, device):
    """Caches YOLO model initialization into memory to prevent reloading across tiles."""
    key = (str(model_path), device)
    if key not in _yolo_model_cache:
        _yolo_model_cache[key] = YOLO(str(model_path))
    return _yolo_model_cache[key]

def _sliding_positions(L: int, win: int, step: int):
    """Calculates sliding window indices to ensure full tile coverage."""
    if win >= L:
        return [0]
    pos = list(range(0, L - win + 1, step))
    if pos[-1] != L - win:
        pos.append(L - win)
    return pos

def coarse_scan_estimate(preproc_img,
                         model_path,
                         device         = 0,
                         initial_factor = 10,
                         init_window    = 200,
                         step_frac      = 0.25,
                         max_samples    = 8,
                         verbose        = True):
    """
    Rapid, wide-net YOLO probe over the frame to estimate appropriate scan-box 
    parameters dynamically before committing to a full pass.
    """
    model = YOLO(str(model_path))

    STEP = int(init_window * step_frac)
    h, w = preproc_img.shape
    sizes_px, xs, ys = [], [], []

    for y in range(0, max(1, h - init_window + 1), STEP):
        for x in range(0, max(1, w - init_window + 1), STEP):
            crop = preproc_img[y:y+init_window, x:x+init_window]
            res  = model.predict(crop,
                                 conf=0.05, iou=0.30,
                                 device=device, verbose=False)[0]
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
        if verbose: print("[coarse] nothing found – defaulting to 50 px")
        return 50, (w//2, h//2)

    mean_sz   = np.mean(sizes_px)
    size_out  = int(round(max(10, mean_sz * initial_factor)))
    cx, cy    = (min(xs)+max(xs))//2, (min(ys)+max(ys))//2

    if verbose:
        print(f"[coarse] mean defect ≈ {mean_sz:.1f}px  →  "
              f"coarse scan_box_size = {size_out}px,  centre = ({cx},{cy})")

    return size_out, (cx, cy)

def refine_scan_box_size(preproc_crop,
                         model_path,
                         device            = 0,
                         coarse_guess_px   = 120,
                         verbose           = True,
                         show_tuning_plot  = False,
                         conf_threshold    = 0.025,
                         iou_threshold     = 0.30):
    """
    Evaluates incremental YOLO passes via a Gaussian curve fit over detection counts
    to define the optimal feature bounding box scale.
    """
    def _count(img, box, edge_tol_px=5):
        (_1, _2, _3, _4, locs, *_rest) = process_image_yolo(
            raw_array      = img,
            image_array    = img,
            base_filename  = "GAUSS",
            scan_box_size  = int(box),
            conf_threshold = 0.025,
            iou_threshold  = 0.30,
            model_path     = model_path,
            device         = device,
            full_shape     = img.shape,
            offset_xy      = (0, 0),
            progress_bar   = None)

        H, W = img.shape
        good = 0
        for (xs, ys) in locs:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            if (x_min <= edge_tol_px          or
                y_min <= edge_tol_px          or
                x_max >= W - edge_tol_px      or
                y_max >= H - edge_tol_px):
                continue
            good += 1
        return good

    sizes   = []
    jump = coarse_guess_px * 0.1
    s       = float(coarse_guess_px ** 0.8)
    H, W = preproc_crop.shape[:2]
    while s <= 2.0*coarse_guess_px + 0.5 and s <= max(W, H):
        sizes.append(int(round(s)))
        s += jump
    sizes = np.array(sizes, dtype=int)

    counts  = np.array([_count(preproc_crop, sz, edge_tol_px=3) for sz in sizes], dtype=float)

    def _gauss(x, A, mu, sigma, C):
        return A * np.exp(-0.5*((x-mu)/sigma)**2) + C

    p0   = [counts.max() - counts.min(), sizes[counts.argmax()],
            max(1.0, 0.2*coarse_guess_px), counts.min()]
    bounds = ([0,   sizes.min(),  1,    0],
              [np.inf, sizes.max(), sizes.max()-sizes.min(), np.inf])
    best_sz = sizes[counts.argmax()]

    try:
        popt, _ = curve_fit(_gauss, sizes, counts, p0=p0, bounds=bounds, maxfev=2000)
        A, mu, sigma, C = popt
        if sigma > 1:
            best_sz = int(round(mu))
    except RuntimeError:
        if verbose:
            print("[refine-gauss] curve_fit did not converge – using arg-max")

    if verbose:
        rng = f"{sizes[0]}–{sizes[-1]} px ({len(sizes)} samples)"
        print(f"[refine-gauss] sweep {rng}  →  chosen {best_sz} px")

    if show_tuning_plot:
        xs = np.linspace(sizes.min(), sizes.max(), 300)
        try:
            ys = _gauss(xs, *popt)
        except Exception:
            ys = np.interp(xs, sizes, counts)
        plt.figure(figsize=(6,4))
        plt.plot(sizes, counts, 'o', label="#detections")
        plt.plot(xs, ys,  '-', label="Gaussian fit")
        plt.axvline(coarse_guess_px, ls=':', label=f"coarse {coarse_guess_px}px")
        plt.axvline(best_sz, ls='--', label=f"chosen {best_sz}px")
        plt.xlabel("scan_box_size [px]"); plt.ylabel("detections")
        plt.title("Gaussian refinement"); plt.legend(); plt.show()

    return best_sz

def process_image_yolo(*, raw_array, image_array, base_filename,
                       scan_box_size, conf_threshold, iou_threshold,
                       model_path, device,
                       full_shape=None,
                       offset_xy=(0, 0),
                       progress_bar=None):
    """
    Replaces standard CNN processing by executing an Ultralytics YOLO inference 
    pass over the designated image tile, yielding localized bounding box coordinates.
    """
    H_full, W_full = full_shape if full_shape else raw_array.shape
    off_x, off_y   = offset_xy

    WIN_SIZE = scan_box_size
    STEP     = max(1, int(0.8 * WIN_SIZE))
    h, w     = raw_array.shape
    ys       = _sliding_positions(h, WIN_SIZE, STEP)
    xs       = _sliding_positions(w, WIN_SIZE, STEP)
    total  = len(ys) * len(xs)
    model  = _get_yolo(model_path, device)

    dislocation_locations             = []
    dislocation_classification_labels = []
    dislocation_location_status      = []
    class_rectangles                 = {dislocation_class: []}

    patch_locations, popt_list, pixel_size_list = [], [], []

    copy_to_draw_on  = image_array.copy()
    dislocation_image = np.zeros_like(raw_array, dtype=np.float32)
    background_image  = image_array.copy()

    done = 0
    EDGE_TOL = int(WIN_SIZE * 0.05)
    for y in ys:
        for x in xs:
            crop = raw_array[y:y+WIN_SIZE, x:x+WIN_SIZE]
            res  = model.predict(crop,
                                 conf   = conf_threshold,
                                 iou    = iou_threshold,
                                 device = device,
                                 verbose = False)[0]

            for bx0, by0, bx1, by1 in res.boxes.xyxy.cpu().numpy():
                bx0, by0, bx1, by1 = map(int, (bx0, by0, bx1, by1))

                crop_at_left   = (x == 0)
                crop_at_top    = (y == 0)
                crop_at_right  = (x + WIN_SIZE >= w)
                crop_at_bottom = (y + WIN_SIZE >= h)

                near_left   = bx0 <  EDGE_TOL
                near_top    = by0 <  EDGE_TOL
                near_right  = bx1 >  WIN_SIZE - EDGE_TOL
                near_bottom = by1 >  WIN_SIZE - EDGE_TOL

                if (near_left   and not crop_at_left  ) or \
                   (near_top    and not crop_at_top   ) or \
                   (near_right  and not crop_at_right ) or \
                   (near_bottom and not crop_at_bottom):
                    continue

                gx0, gy0, gx1, gy1 = bx0 + x, by0 + y, bx1 + x, by1 + y
                dislocation_locations.append(([gx0, gx1, gx1, gx0, gx0],
                                            [gy0, gy0, gy1, gy1, gy0]))
                dislocation_classification_labels.append("dislocation")
                dislocation_location_status.append("Successful_fit")

                patch_locations.append(None)
                popt_list.append(None)
                pixel_size_list.append(None)

                cv2.rectangle(copy_to_draw_on, (gx0, gy0), (gx1, gy1),
                            (0, 255, 0), 2)

            done += 1
            if progress_bar and total:
                progress_bar.setValue(int(done / total * 100))
                QApplication.processEvents()

    return (image_array, copy_to_draw_on,
            dislocation_image, background_image,
            dislocation_locations,
            dislocation_location_status,
            dislocation_classification_labels,
            [],
            patch_locations,
            popt_list,
            pixel_size_list,
            class_rectangles)

# Final execution wrappers & tiling

def plot_results(image_array, copy_to_draw_on,
                 dislocation_image, background_image,
                 dislocation_locations, dislocation_location_status,
                 dislocation_classification_labels,
                 centers_of_fits, class_rectangles, file_path):
    """Generates the four-pane Matplotlib diagnostic output figure."""
    fig = plt.figure(figsize=(10,10))

    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(2,2,3, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(2,2,4, sharex=ax1, sharey=ax1)

    ax1.imshow(image_array, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(copy_to_draw_on, cmap='gray')
    ax2.set_title('Full Image (patch annotations)')
    ax2.axis('off')

    for (rect, status) in zip(dislocation_locations, dislocation_location_status):
        x_coords, y_coords = rect
        if status == 'Successful_fit':
            ax2.plot(x_coords, y_coords, 'g--', linewidth=2, alpha=0.3)
        elif status == 'Failed_fit':
            ax2.plot(x_coords, y_coords, 'y--', linewidth=2, alpha=0.3)
        else:
            ax2.plot(x_coords, y_coords, 'b--', linewidth=2, alpha=0.3)

    for center in centers_of_fits:
        ax2.plot(center[0], center[1], 'ro', markersize=2)

    for cclass, rectangles in class_rectangles.items():
        class_info = CLASS_INFO.get(cclass, None)
        if class_info:
            color = class_info['color']
        else:
            color = 'white'
        for rect in rectangles:
            ax2.plot(rect[0], rect[1], '--', color=color, linewidth=2)

    ax3.imshow(dislocation_image, cmap='gray')
    ax3.set_title('Dislocation Image')
    ax3.axis('off')

    ax4.imshow(background_image, cmap='gray')
    ax4.set_title('Background Image')
    ax4.axis('off')

    for (rect, status) in zip(dislocation_locations, dislocation_location_status):
        x_coords, y_coords = rect
        if status == 'Successful_fit':
            ax2.plot(x_coords, y_coords, 'g--', linewidth=2, alpha=0.3)
        elif status == 'Failed_fit':
            ax2.plot(x_coords, y_coords, 'y--', linewidth=2, alpha=0.3)
        else:
            ax2.plot(x_coords, y_coords, 'b--', linewidth=2, alpha=0.3)

    folder_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    plt.suptitle(f"{folder_name}/{file_name}")

    for (rect, classification_label) in zip(dislocation_locations, dislocation_classification_labels):
        x_coords, y_coords = rect
        color = classification_color_map.get(classification_label, 'white')
        ax4.plot(x_coords, y_coords, '-', color=color, linewidth=2)

    plt.show()


def process_specific_tiff(
        file_path, *,
        fit=True, scan_box_size=50,
        clip_top=0, clip_bottom=0, clip_left=0, clip_right=0,
        max_sigma=50, increment=1, max_gradient_ratio=0.2,
        max_allowed_percentage=0.8,
        image_scan_progress_bar=None,
        max_tile_area=2_500_000,
        overlap_margin=None,
        test_print=False):
    """
    Main file loader and grid tiling orchestrator. Segments huge TIFFs into 
    digestible grids for inference memory allocation before stitching them back.
    """
    def choose_grid(w, h, max_area=max_tile_area):
        min_tiles = math.ceil((w * h) / max_area)
        best = None
        search = int(math.ceil(math.sqrt(min_tiles))) + 4
        for r in range(1, search):
            c = math.ceil(min_tiles / r)
            tw, th = w / c, h / r
            if tw * th > max_area:
                continue
            score = (r*c, abs(tw-th), abs(r-c), r, c)
            if best is None or score < best:
                best = score
        if best is None:
            raise RuntimeError("max_tile_area too small for this image")
        return best[3], best[4]

    def make_tiles(w, h, rows, cols, margin):
        tw, th = w / cols, h / rows
        for r in range(rows):
            for c in range(cols):
                x0 = int(max(0,         c*tw      - margin))
                y0 = int(max(0,         r*th      - margin))
                x1 = int(min(w,      (c+1)*tw     + margin))
                y1 = int(min(h,      (r+1)*th     + margin))
                yield (x0, y0, x1, y1)

    print(f"Processing {file_path} with scan_box_size={scan_box_size} ...")
    start_time = time.time()
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    image_counters = {'dislocation': 1,
                      'multiple_dislocations': 1,
                      'no_dislocations': 1}

    img = load_img(file_path, color_mode='grayscale')
    image_np = np.squeeze(img_to_array(img))

    if SCAN_ENGINE == "yolo":
        coarse_sz, (cx, cy) = coarse_scan_estimate(
            preproc_img  = image_np,
            model_path   = YOLO_MODEL_PATH,
            device       = 0,
            initial_factor = 11)

        expand_factor = 10
        half_side     = coarse_sz * expand_factor // 2
        x0 = max(0, cx - half_side); y0 = max(0, cy - half_side)
        x1 = min(image_np.shape[1], cx + half_side)
        y1 = min(image_np.shape[0], cy + half_side)
        crop = image_np[y0:y1, x0:x1]

        scan_box_size = refine_scan_box_size(crop,
                         YOLO_MODEL_PATH,
                         device            = 0,
                         coarse_guess_px   = coarse_sz,
                         verbose           = True,
                         show_tuning_plot  = True,
                         conf_threshold    = 0.025,
                         iou_threshold     = 0.30)

        print(f"→ final scan_box_size to feed into process_image = {scan_box_size}px")

    H_raw, W_raw = image_np.shape
    H = max(1, H_raw - clip_top    - clip_bottom)
    W = max(1, W_raw - clip_left   - clip_right)

    if overlap_margin is None:
        overlap_margin = scan_box_size

    rows, cols = choose_grid(W, H, max_area=max_tile_area)
    print(f"   tiling → {rows}×{cols}  "
          f"(tile ≈ {int(W/cols)}×{int(H/rows)} px, "
          f"overlap {overlap_margin}px)")

    if test_print:
        plt.figure(figsize=(6, 6))
        plt.imshow(image_np, cmap='gray')
        plt.title(f"{base_filename} – tile overlay (before preprocessing)")

        for (x0, y0, x1, y1) in make_tiles(W, H, rows, cols, overlap_margin):
            x0_r = x0 + clip_left
            x1_r = x1 + clip_left
            y0_r = y0 + clip_top
            y1_r = y1 + clip_top
            plt.gca().add_patch(
                plt.Rectangle((x0_r, y0_r), x1_r - x0_r, y1_r - y0_r,
                              edgecolor='red', facecolor='none', lw=1))
        plt.show()

    steps = ['normalize', 'clip']
    if max_sigma > 0:
        steps.append('adaptive_smoothing')

    preprocessed_image, _ = preprocess_image(
        image_np.copy(),
        steps,
        clip_top=clip_top, clip_bottom=clip_bottom,
        clip_left=clip_left, clip_right=clip_right,
        max_sigma=max_sigma, increment=increment,
        max_gradient_ratio=max_gradient_ratio,
        max_allowed_percentage=max_allowed_percentage,
        final_blur_sigma=1 if max_sigma > 0 else 0
    )

    dislocation_locations = []
    dislocation_location_status = []
    dislocation_classification_labels = []
    centers_of_fits = []
    patch_locations, popt_list, pixel_size_list = [], [], []
    class_rectangles = {c: [] for c in CLASS_INFO.keys()}

    copy_to_draw_on = preprocessed_image.copy()
    dislocation_image = np.zeros_like(preprocessed_image, dtype=np.float32)
    background_image  = preprocessed_image.copy()

    if HAVE_RTREE:
        seen = index.Index()
    else:
        seen_boxes = []

    tile_id = 0
    for x0, y0, x1, y1 in make_tiles(W, H, rows, cols, overlap_margin):
        raw_tile = image_np[y0:y1, x0:x1]
        pre_tile = preprocessed_image[y0:y1, x0:x1]
        if SCAN_ENGINE == "legacy":
            ( _img_arr, _copy, disloc_tile, back_tile,
            locs, status, labels, centres,
            patches, popts, pixsizes, cls_rects) = process_image(
                    raw_array=raw_tile,
                    image_array=pre_tile,
                    base_filename=f"{base_filename}_T{tile_id}",
                    image_counters=image_counters,
                    save_patches=False,
                    fit=fit,
                    scan_box_size=scan_box_size,
                    image_scan_progress_bar=image_scan_progress_bar)

        else:
            (_img_arr, _copy, disloc_tile, back_tile,
            locs, status, labels, centres,
            patches, popts, pixsizes, cls_rects) = process_image_yolo(
                raw_array=raw_tile,
                image_array=raw_tile,
                base_filename=f"{base_filename}_T{tile_id}",
                scan_box_size=scan_box_size,
                conf_threshold=0.025,
                iou_threshold=0.30,
                model_path=YOLO_MODEL_PATH,
                device=0,
                full_shape    = (H, W),
                offset_xy     = (x0, y0),
                progress_bar=image_scan_progress_bar,
            )

        dislocation_image[y0:y1, x0:x1] = disloc_tile
        background_image [y0:y1, x0:x1] = back_tile

        for (rect, st, lab) in zip(locs, status, labels):
            gxs = [x + x0 for x in rect[0]]
            gys = [y + y0 for y in rect[1]]
            bbox = (min(gxs), min(gys), max(gxs), max(gys))

            duplicate = False
            if HAVE_RTREE:
                duplicate = bool(list(seen.intersection(bbox)))
            else:
                for b in seen_boxes:
                    ix0 = max(b[0], bbox[0]); iy0 = max(b[1], bbox[1])
                    ix1 = min(b[2], bbox[2]); iy1 = min(b[3], bbox[3])
                    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
                    union = ((b[2]-b[0])*(b[3]-b[1]) +
                             (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) - inter)
                    if union and inter / union >= 0.25:
                        duplicate = True; break

            if duplicate:
                continue
            if HAVE_RTREE:
                seen.insert(tile_id, bbox)
            else:
                seen_boxes.append(bbox)

            dislocation_locations.append((gxs, gys))
            dislocation_location_status.append(st)
            dislocation_classification_labels.append(lab)

        centers_of_fits.extend([(cx+x0, cy+y0) for (cx,cy) in centres])

        patch_locations.extend([
            (pl and (pl[0]+x0, pl[1]+y0, pl[2]+x0, pl[3]+y0)) if pl else None
            for pl in patches])
        popt_list   .extend(popts)
        pixel_size_list.extend(pixsizes)

        for cls, rects in cls_rects.items():
            class_rectangles[cls].extend([
                ([x+x0 for x in r[0]], [y+y0 for y in r[1]]) for r in rects])

        if test_print:
            plt.gca().add_patch(
                plt.Rectangle((x0,y0), x1-x0, y1-y0,
                              edgecolor='red', facecolor='none', lw=1))
        tf.keras.backend.clear_session(); gc.collect()
        tile_id += 1

    if test_print:
        plt.show()

    elapsed_time = time.time() - start_time
    print(f"Processing time (all tiles): {elapsed_time:.2f} seconds")

    (final_disloc_locs,
     final_disloc_labels,
     final_disloc_status,
     final_patch_locs,
     final_popt_list,
     final_pix_size_list) = merge_overlapping_unknowns(
        dislocation_locations,
        dislocation_classification_labels,
        dislocation_location_status,
        patch_locations,
        popt_list,
        pixel_size_list,
        overlap_threshold=0.25)

    real_dislocation_indices = [
        i for i, lbl in enumerate(final_disloc_labels) if lbl != 'unknown']
    total_dislocations_found = len(real_dislocation_indices)

    save_file = os.path.join(
        os.path.dirname(file_path),
        f"{base_filename}_data_{total_dislocations_found}_dislocations.csv")
    with open(save_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['rect', 'classification_label', 'status',
                            'patch_location', 'popt', 'pixel_size'])
        for i in range(len(final_disloc_locs)):
            csvwriter.writerow([final_disloc_locs[i],
                                final_disloc_labels[i],
                                final_disloc_status[i],
                                final_patch_locs[i],
                                final_popt_list[i],
                                final_pix_size_list[i]])

    plot_results(
        preprocessed_image,
        copy_to_draw_on,
        dislocation_image,
        background_image,
        final_disloc_locs,
        final_disloc_status,
        final_disloc_labels,
        centers_of_fits,
        class_rectangles,
        file_path)

    tf.keras.backend.clear_session()
    del image_np, preprocessed_image, copy_to_draw_on
    del dislocation_image, background_image
    del patch_locations, popt_list, pixel_size_list
    gc.collect()
    print("Cleared TensorFlow session and garbage collected.")


# PySide6 GUI interface

class RemovableFileItem(QWidget):
    """Custom QWidget representing a loaded file in the GUI list, with an inline remove button."""
    def __init__(self, file_path, remove_callback):
        super().__init__()
        self._file_path       = file_path
        self._remove_callback = remove_callback

        lay   = QHBoxLayout(self)
        label = QLabel(os.path.basename(file_path))
        btn   = QPushButton("X", clicked=self._on_remove); btn.setFixedWidth(30)
        lay.addWidget(label); lay.addWidget(btn); lay.setContentsMargins(0,0,0,0)

    def _on_remove(self):
        self._remove_callback(self._file_path)


class ImageViewer(QGraphicsView):
    """
    Custom QGraphicsView subclass enabling interactive preview capabilities:
    Provides wheel‑zoom, drag‑pan, and click-to-centre functionality for the green scan‑box.
    """
    clicked = Signal(QPointF)
    _STEP   = 1.25

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self); self.setScene(self._scene)
        self._pix   = QGraphicsPixmapItem(); self._scene.addItem(self._pix)

        self._box   = QGraphicsRectItem(); self._scene.addItem(self._box)
        pen = QPen(QColor(0,255,0)); pen.setWidth(2); self._box.setPen(pen)
        self._box_size = 50

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def set_pixmap_first(self, pm: QPixmap, box_size: int, centre: QPointF):
        """Initializes the view, calculating initial aspect ratios and zoom parameters."""
        self._pix.setPixmap(pm)
        self._scene.setSceneRect(self._pix.boundingRect())
        self.resetTransform()
        self.fitInView(self._pix, Qt.AspectRatioMode.KeepAspectRatio)
        self._box_size = box_size
        self._move_box(centre)

    def update_pixmap(self, pm: QPixmap):
        """Swaps active pixels without resetting the current user-defined pan/zoom vectors."""
        self._pix.setPixmap(pm)
        self._scene.setSceneRect(self._pix.boundingRect())

    def _move_box(self, centre: QPointF):
        h = self._box_size / 2
        self._box.setRect(QRectF(centre.x()-h, centre.y()-h,
                                 self._box_size, self._box_size))

    def set_box_centre(self, centre: QPointF, size: int):
        self._box_size = size
        self._move_box(centre)

    def wheelEvent(self, ev):
        self.scale(self._STEP if ev.angleDelta().y()>0 else 1/self._STEP,
                   self._STEP if ev.angleDelta().y()>0 else 1/self._STEP)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.mapToScene(ev.pos()))
        super().mousePressEvent(ev)


class MainWindow(QMainWindow):
    """Primary application controller mapping UI inputs to the file processing engine."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF File Processor (Dictionary-based, multi-return)")

        self._last_dir              = os.path.expanduser("~")
        self._selected_files        = []
        self._file_params           = {}
        self._current_scan_box_size = 50

        self._btn_files   = QPushButton("Select TIFF Files")
        self._btn_dirs    = QPushButton("Select Directories")
        self._btn_clear   = QPushButton("Clear Images")

        self._chk_fit     = QCheckBox("Enable Fitting")
        self._cmb_engine  = QComboBox()
        self._cmb_engine.addItems(["legacy", "yolo"])
        self._cmb_engine.setCurrentText("yolo") 

        eng_row = QWidget()
        eng_lay = QHBoxLayout(eng_row)
        eng_lay.setContentsMargins(0, 0, 0, 0)
        eng_lay.addWidget(QLabel("Scanning engine:"))
        eng_lay.addWidget(self._cmb_engine)

        self._btn_process = QPushButton("Start Processing")
        self._list_items  = QListWidget()
        self._status      = QLabel("Select files and directories to process.")
        self._bar_total   = QProgressBar(format="Overall progress: %p%")
        self._bar_scan    = QProgressBar(format="Scanning current image: %p%")

        controls = QVBoxLayout()
        for w in (self._btn_files, self._btn_dirs, self._btn_clear, self._list_items,
                  self._chk_fit, eng_row, self._btn_process, self._status,
                  self._bar_total, self._bar_scan):
            controls.addWidget(w)
        controls.addStretch()

        help_lbl = QLabel(
            "<h2>TIFF File Processor Instructions</h2>"
            "<ol>"
            "<li>Select TIFF files or entire directories.</li>"
            "<li>Preview each if you want to tweak clipping/smoothing.</li>"
            "<li>Enable Fitting to remove dislocations (slower).</li>"
            "<li>Start Processing to scan &amp; save results.</li></ol>",
            wordWrap=True, alignment=Qt.AlignmentFlag.AlignTop
        )
        help_box = QVBoxLayout(); help_box.addWidget(help_lbl); help_box.addStretch()
        help_w   = QWidget(); help_w.setLayout(help_box)

        centre = QHBoxLayout(); centre.addLayout(controls, 3); centre.addWidget(help_w, 1)
        base   = QWidget(); base.setLayout(centre)
        self.setCentralWidget(base)

        self._btn_files.clicked.connect(self._choose_files)
        self._btn_dirs .clicked.connect(self._choose_dirs)
        self._btn_clear.clicked.connect(self._clear_images)
        self._btn_process.clicked.connect(self._start_processing)

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
            w  = RemovableFileItem(fp, self._remove_fp)
            self._list_items.setItemWidget(it, w)
            it.setSizeHint(w.sizeHint())

    def _choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select TIFF files",
                                                self._last_dir, "TIFF files (*.tif *.tiff)")
        if not files:
            return
        self._last_dir = os.path.dirname(files[0])

        use_preview = (self._cmb_engine.currentText() != "yolo")

        for fp in files:
            if use_preview:
                if not self._preview_dialog(fp):
                    continue
            else:
                self._file_params[fp] = dict(
                    clip_top=0, clip_bottom=0, clip_left=0, clip_right=0,
                    max_sigma=0, max_gradient_ratio=0.2, max_allowed_percentage=0.8,
                    scan_box_size=50, increment=1
                )
            self._selected_files.append(fp)

        self._refresh_list()

    def _choose_dirs(self):
        """
        Add every .tif/.tiff file found in a directory (recursively).
        """
        d = QFileDialog.getExistingDirectory(self,
                                            "Select a directory",
                                            self._last_dir)
        if not d:
            return
        self._last_dir = d

        use_preview = (self._cmb_engine.currentText() != "yolo")

        for root, _, files in os.walk(d):
            for fname in files:
                if not fname.lower().endswith((".tif", ".tiff")):
                    continue

                fp = os.path.join(root, fname)

                if use_preview:
                    if not self._preview_dialog(fp):
                        continue
                else:
                    if fp not in self._file_params:
                        self._file_params[fp] = dict(
                            clip_top=0, clip_bottom=0,
                            clip_left=0, clip_right=0,
                            max_sigma=0,
                            max_gradient_ratio=0.2,
                            max_allowed_percentage=0.8,
                            scan_box_size=50,
                            increment=1
                        )

                self._selected_files.append(fp)

        self._selected_files = list(dict.fromkeys(self._selected_files))
        self._refresh_list()


    def _preview_dialog(self, fp) -> bool:
        """
        Show an interactive preview of a TIFF image.
        Lets the user tweak clipping, smoothing and scan‑box size.
        Returns True if user accepted the file, False if user cancelled.
        """
        img    = load_img(fp, color_mode='grayscale')
        img_np = np.squeeze(img_to_array(img))

        dlg = QDialog(self)
        dlg.setWindowTitle("Preview TIFF File")
        dlg.resize(1200, 900)
        dlg.setSizeGripEnabled(True)

        view = ImageViewer()
        view.setMinimumSize(800, 600)

        def slider(lo, hi, val):
            s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val); return s

        sliders = {
            'clip_top'   : slider(0, 200, 0),
            'clip_bottom': slider(0, 200, 0),
            'clip_left'  : slider(0, 200, 0),
            'clip_right' : slider(0, 200, 0),
            'max_sigma'  : slider(0, 100, 0),
            'scan_box'   : slider(20, 500, self._current_scan_box_size)
        }

        smooth_chk = QCheckBox("Apply smoothing")
        smooth_chk.setChecked(True)

        form = QFormLayout()
        form.addRow("Clip Top:",      sliders['clip_top'])
        form.addRow("Clip Bottom:",   sliders['clip_bottom'])
        form.addRow("Clip Left:",     sliders['clip_left'])
        form.addRow("Clip Right:",    sliders['clip_right'])
        form.addRow(smooth_chk)
        form.addRow("Max Sigma:",     sliders['max_sigma'])
        form.addRow("Scan Box Size:", sliders['scan_box'])

        left = QVBoxLayout(); left.addLayout(form); left.addWidget(view)

        cx, cy = img_np.shape[1] // 2, img_np.shape[0] // 2
        first  = True

        def dotted(img, p1, p2, col=(0, 0, 255), t=2, gap=10):
            (x1, y1), (x2, y2) = p1, p2
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            for x in range(x1, x2, gap * 2):
                cv2.line(img, (x, y1), (min(x + gap, x2), y1), col, t)
                cv2.line(img, (x, y2), (min(x + gap, x2), y2), col, t)
            for y in range(y1, y2, gap * 2):
                cv2.line(img, (x1, y), (x1, min(y + gap, y2)), col, t)
                cv2.line(img, (x2, y), (x2, min(y + gap, y2)), col, t)

        def refresh():
            vals = {k: s.value() for k, s in sliders.items()}
            apply_smooth = smooth_chk.isChecked()
            max_sigma    = vals['max_sigma'] if apply_smooth else 0

            steps = ['normalize', 'clip']
            if max_sigma > 0:
                steps.append('adaptive_smoothing')

            processed, _ = preprocess_image(
                img_np.copy(), steps,
                clip_top=vals['clip_top'], clip_bottom=vals['clip_bottom'],
                clip_left=vals['clip_left'], clip_right=vals['clip_right'],
                max_sigma=max_sigma, increment=1,
                max_gradient_ratio=0.2, max_allowed_percentage=0.8,
                final_blur_sigma=1 if max_sigma > 0 else 0
            )

            h, w = processed.shape
            rgb  = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

            dotted(rgb, (vals['clip_left'], vals['clip_top']),
                        (w - vals['clip_right'], h - vals['clip_bottom']))

            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qimg)

            if first:
                view.set_pixmap_first(pix, vals['scan_box'], QPointF(cx, cy))
            else:
                view.update_pixmap(pix)
                view.set_box_centre(QPointF(cx, cy), vals['scan_box'])

            sliders['max_sigma'].setEnabled(apply_smooth)

        for s in sliders.values():
            s.valueChanged.connect(refresh)
        smooth_chk.toggled.connect(refresh)

        def relocate(pt: QPointF):
            nonlocal cx, cy, first
            cx, cy = int(pt.x()), int(pt.y())
            first  = False
            refresh()
            
        view.clicked.connect(relocate)
        refresh()

        ok_btn     = QPushButton("Select TIFF file", clicked=dlg.accept)
        cancel_btn = QPushButton("Cancel",           clicked=dlg.reject)
        row        = QHBoxLayout(); row.addWidget(ok_btn); row.addWidget(cancel_btn)
        left.addLayout(row)

        dlg.setLayout(left)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            vals = {k: s.value() for k, s in sliders.items()}
            apply_smooth = smooth_chk.isChecked()
            max_sigma    = vals['max_sigma'] if apply_smooth else 0

            self._file_params[fp] = {
                'clip_top'             : vals['clip_top'],
                'clip_bottom'          : vals['clip_bottom'],
                'clip_left'            : vals['clip_left'],
                'clip_right'           : vals['clip_right'],
                'max_sigma'            : max_sigma,
                'max_gradient_ratio'   : 0.2,
                'max_allowed_percentage': 0.8,
                'scan_box_size'        : vals['scan_box'],
                'increment'            : 1
            }
            self._current_scan_box_size = vals['scan_box']
            return True

        return False

    def _start_processing(self):
        global SCAN_ENGINE
        SCAN_ENGINE = self._cmb_engine.currentText() 
        paths = list(dict.fromkeys(self._selected_files))
        
        if not paths:
            QMessageBox.warning(self, "No TIFF files", "No TIFF files selected for processing.")
            return

        for w in (self._btn_files, self._btn_dirs, self._btn_clear,
                  self._btn_process, self._chk_fit):
            w.setEnabled(False)

        self._bar_total.setMaximum(len(paths)); self._bar_total.setValue(0)
        self._bar_scan .setValue(0)
        fit = self._chk_fit.isChecked()

        for idx, fp in enumerate(paths, 1):
            self._status.setText(f"Processing ({idx}/{len(paths)}): {fp}")
            QApplication.processEvents()
            try:
                p = self._file_params.get(fp, dict(
                    clip_top=0, clip_bottom=0, clip_left=0, clip_right=0,
                    max_sigma=50, max_gradient_ratio=0.2, max_allowed_percentage=0.8,
                    scan_box_size=self._current_scan_box_size, increment=1
                ))
                process_specific_tiff(
                    file_path=fp, fit=fit, scan_box_size=p['scan_box_size'],
                    clip_top=p['clip_top'], clip_bottom=p['clip_bottom'],
                    clip_left=p['clip_left'], clip_right=p['clip_right'],
                    max_sigma=p['max_sigma'], increment=p['increment'],
                    max_gradient_ratio=p['max_gradient_ratio'],
                    max_allowed_percentage=p['max_allowed_percentage'],
                    image_scan_progress_bar=self._bar_scan,
                    max_tile_area= 15_000_000,
                    overlap_margin=10,
                    test_print=False
                )
            except Exception as e:
                print(f"Error processing {fp}: {e}")

            self._bar_total.setValue(idx)

        QMessageBox.information(self, "Processing Complete", "All files have been processed.")
        self._status.setText("Processing complete.")
        for w in (self._btn_files, self._btn_dirs, self._btn_clear,
                  self._btn_process, self._chk_fit):
            w.setEnabled(True)
        self._bar_scan.setValue(0)

    def closeEvent(self, ev):
        cv2.destroyAllWindows()
        QApplication.quit(); ev.accept()

# Entry point

def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()