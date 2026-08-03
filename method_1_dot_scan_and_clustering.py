import os
import numpy as np
import pandas as pd
import h5py
import shutil
import sys
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from skimage.measure import shannon_entropy, regionprops, label
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.morphology import remove_small_objects, disk
import random
from matplotlib.patches import Rectangle, Polygon as MplPolygon, Circle
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import MeanShift
import json
from shapely.geometry import mapping

# =============================================================================
# USER CONFIGURATION & PLOTTING FLAGS
# =============================================================================
# Configuration for which plots to display
PLOT_IMAGE_PREPROCESSED_SMOOTHED = False
PLOT_BEFORE_AND_AFTER_CLEANUP = False
PLOT_HISTOGRAM = False
PLOT_REGIONS_WITH_BORDERS = False
PLOT_ENTROPY = False
PLOT_HIGH_CONTRAST_AREAS = False
PLOT_COLORED_REGIONS_AND_CONTRAST_AREAS = True
PLOT_REGION_MEANS_VS_CONTRAST_THRESHOLDS = False
PLOT_INITIAL_GRADIENT_POINTS = False
PLOT_SCALE_FINDING = False

# Publication-figure export: saves standalone, high-DPI images (blobs only,
# dots only, and a high-contrast Otsu regions view) instead of just the
# on-screen debug plots above.
SAVE_PLOTS_FOR_PAPER = True
PAPER_PLOT_DPI = 300
PAPER_PLOT_FORMATS = ['png', 'pdf']  # pdf keeps markers/outlines vector (crisp when zoomed in a paper)
PAPER_DOT_COLOR = (0.835, 0.369, 0.000)  # vermillion, reads clearly against grayscale SEM backgrounds
PAPER_DOT_SIZE = 18
PAPER_BLOB_ALPHA = 0.4

# File I/O Configuration
base_directory = 'data/paper_images'
output_directory = 'method_1_output'
# Path to the file where data will be saved
data_file_path = os.path.join(output_directory, "contrast_threshold_data.csv")

use_sister_folders = True
cleanup = False
save_format = 'csv'

# Processing Pipeline Configuration
num_files_to_process = 1
skip_files = 3
selection_method = 'filename'  # Options are 'first', 'skip', 'random', 'filename'
target_filename = 'Lower Centre 9.tiff'  # Used when selection_method == 'filename'

preprocess_steps = ['clip', 'normalize', 'denoise']

clip_top = 20
clip_bottom = 50
clip_left = 5
clip_right = 5

max_iterations = 10
max_gradient_ratio = 0.2
max_allowed_percentage = 0.8

manual_region_adjustment = False
num_peaks = 20

skip_files = 2

skip_files = skip_files-1
# save = True
save = False
save_data_to_file = save
save_blobs = save
shrink_width = 8

# search_area = 4
### The fastest way to dial in the parameters is to work through the false save values 1 by 1 leaving the rest of the regions at 255 then once happy, move the value you were working on dwn to the save parameters and return the false to 255 and start lowering the next parameter. The code will ignore parameters if there are more than regions.
if save == False:
    manual_contrast_thresholds = [255, 255, 55, 255, 255, 255]
else:
    manual_contrast_thresholds = [85, 70, 53, 50, 255, 255]
CLUSTER = True

# =============================================================================
# VISUALIZATION & FILE MANAGEMENT FUNCTIONS
# =============================================================================

# Okabe & Ito (2008) colorblind-safe palette. Stays distinguishable under the
# common forms of color vision deficiency and at small print size, unlike
# uniform random RGB.
COLORBLIND_PALETTE = [
    (0.902, 0.624, 0.000),  # orange
    (0.337, 0.706, 0.914),  # sky blue
    (0.000, 0.620, 0.451),  # bluish green
    (0.941, 0.894, 0.259),  # yellow
    (0.000, 0.447, 0.698),  # blue
    (0.835, 0.369, 0.000),  # vermillion
    (0.800, 0.475, 0.655),  # reddish purple
]

def get_colorblind_colors(num_colors, step=3):
    """
    Cycles the Okabe-Ito palette, striding by `step` (coprime with the palette
    length) so that once colors start repeating, immediate neighbors in the
    sequence don't repeat the same hue back-to-back.
    """
    n = len(COLORBLIND_PALETTE)
    return [COLORBLIND_PALETTE[(i * step) % n] for i in range(num_colors)]

def plot_blob(ax, blob, facecolor=None, edgecolor='black', alpha=0.4, linewidth=1.0, offset=(0, 0), point_radius=3):
    """
    Draws a blob of any Shapely geometry type (Polygon, MultiPolygon, or the
    Point/LineString/GeometryCollection produced when tangent circles fail to
    merge into a clean polygon during unary_union). Always draws a solid edge
    around the fill so blobs stay distinguishable by outline alone, not just
    by hue, for colorblind readers.
    """
    offset_x, offset_y = offset

    if blob.geom_type == 'Polygon':
        coords = np.array(blob.exterior.coords)
        xy = np.column_stack((coords[:, 1] - offset_y, coords[:, 0] - offset_x))
        patch = MplPolygon(xy, closed=True, facecolor=facecolor if facecolor else 'none',
                            edgecolor=edgecolor, alpha=alpha if facecolor else 1.0, linewidth=linewidth)
        ax.add_patch(patch)
        if facecolor:
            # Redraw a fully opaque outline on top so the edge stays crisp even under a transparent fill.
            edge_patch = MplPolygon(xy, closed=True, facecolor='none', edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(edge_patch)
    elif blob.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for geom in blob.geoms:
            plot_blob(ax, geom, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, offset=offset, point_radius=point_radius)
    elif blob.geom_type == 'Point':
        circle = Circle((blob.y - offset_y, blob.x - offset_x), radius=point_radius,
                         facecolor=facecolor if facecolor else 'none', edgecolor=edgecolor,
                         alpha=alpha if facecolor else 1.0, linewidth=linewidth)
        ax.add_patch(circle)
    elif blob.geom_type == 'LineString':
        x, y = blob.xy
        ax.plot(np.array(y) - offset_y, np.array(x) - offset_x, color=edgecolor, linewidth=linewidth)

def save_paper_figure(fig, output_folder, file_name_base, suffix):
    """Saves a publication-ready figure at high DPI in every format listed in PAPER_PLOT_FORMATS."""
    paper_dir = os.path.join(output_folder, 'paper_figures')
    os.makedirs(paper_dir, exist_ok=True)
    for fmt in PAPER_PLOT_FORMATS:
        path = os.path.join(paper_dir, f"{file_name_base}_{suffix}.{fmt}")
        fig.savefig(path, dpi=PAPER_PLOT_DPI, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved paper figure: {path}")

def generate_paper_figures(image_array, preprocessed_array, smoothed_array, blobs, all_high_contrast_areas,
                            colored_region_image_before, colors_before,
                            colored_region_image_after, colors_after,
                            regions_with_borders, thresholds, entropies,
                            region_means, contrast_thresholds_for_plot,
                            output_folder, file_name_base):
    """
    Saves every diagnostic view as its own standalone, high-DPI file so any
    of them can be dropped straight into a paper: each processing stage,
    blobs only, raw dots only (no region overlay), Otsu regions before/after
    cleanup (regions alone, no dot overlay), each individually bordered
    region, and the histogram/entropy/region-mean tuning plots. Kept
    separate from the interactive debug plots above so each panel can be
    sized/cropped independently and stays legible both at full-page scale
    and zoomed in (pdf export keeps markers/outlines vector).
    """
    image_height, image_width = preprocessed_array.shape
    fig_w = max(6, image_width / 200)
    fig_h = max(6, image_height / 200)

    def save_image_panel(array, cmap, suffix):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(array, cmap=cmap)
        ax.set_axis_off()
        save_paper_figure(fig, output_folder, file_name_base, suffix)
        plt.close(fig)

    def save_region_panel(colored_image, colors, suffix):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(colored_image)
        ax.set_axis_off()
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Region {i+1}',
                                       markerfacecolor=tuple(color), markersize=10) for i, color in enumerate(colors)]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        save_paper_figure(fig, output_folder, file_name_base, suffix)
        plt.close(fig)

    # Raw processing stages
    save_image_panel(image_array, 'gray', 'original_image')
    save_image_panel(preprocessed_array, 'gray', 'preprocessed_image')
    save_image_panel(smoothed_array, 'gray', 'smoothed_image')

    # Blobs only (filled, transparent, colorblind-safe, no dots)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(preprocessed_array, cmap='gray')
    ax.set_axis_off()
    blob_colors = get_colorblind_colors(len(blobs))
    for blob, color in zip(blobs, blob_colors):
        plot_blob(ax, blob, facecolor=color, edgecolor='black', alpha=PAPER_BLOB_ALPHA, linewidth=1.2)
    save_paper_figure(fig, output_folder, file_name_base, 'blobs_only')
    plt.close(fig)

    # Raw dots only - no blobs, no region overlay, bold white-edged marker so it reads at a distance
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(preprocessed_array, cmap='gray')
    ax.set_axis_off()
    if len(all_high_contrast_areas) > 0:
        xs = [p[1] for p in all_high_contrast_areas]
        ys = [p[0] for p in all_high_contrast_areas]
        ax.scatter(xs, ys, s=PAPER_DOT_SIZE, facecolors=PAPER_DOT_COLOR, edgecolors='white', linewidths=0.6)
    save_paper_figure(fig, output_folder, file_name_base, 'dots_only')
    plt.close(fig)

    # Otsu regions - just the regions, no dot/image overlay
    save_region_panel(colored_region_image_before, colors_before, 'otsu_regions_before_cleanup')
    save_region_panel(colored_region_image_after, colors_after, 'otsu_regions')

    # Each individually bordered region
    for idx, (region_with_border, color) in enumerate(zip(regions_with_borders, colors_after), start=1):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(region_with_border, cmap='gray')
        ax.imshow(np.dstack((region_with_border * color[0], region_with_border * color[1], region_with_border * color[2])), alpha=0.5)
        ax.set_axis_off()
        save_paper_figure(fig, output_folder, file_name_base, f'region_{idx}_with_border')
        plt.close(fig)

    # Histogram of smoothed image with Otsu thresholds marked
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(smoothed_array.flatten(), bins=256, color='gray', alpha=0.7)
    for threshold in thresholds:
        ax.axvline(x=threshold, color=PAPER_DOT_COLOR, linestyle='--')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    save_paper_figure(fig, output_folder, file_name_base, 'histogram')
    plt.close(fig)

    # Entropy vs recursion depth (Otsu region count selection)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(entropies) + 1), entropies, marker='o', color=PAPER_DOT_COLOR)
    ax.set_xlabel('Number of Regions')
    ax.set_ylabel('Entropy')
    ax.grid()
    save_paper_figure(fig, output_folder, file_name_base, 'entropy')
    plt.close(fig)

    # Region mean value vs contrast threshold (parameter-tuning plot)
    if region_means and contrast_thresholds_for_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(region_means, contrast_thresholds_for_plot, color=PAPER_DOT_COLOR)
        ax.set_xlabel('Mean Value of Region')
        ax.set_ylabel('Contrast Threshold')
        ax.grid()
        save_paper_figure(fig, output_folder, file_name_base, 'region_means_vs_contrast_thresholds')
        plt.close(fig)

def visualize_blobs(blobs, image_array, high_contrast_areas, num_blobs=5):
    """
    Visualizes extracted features (blobs) overlaying them on their local image regions.
    Used for debugging detection bounds and verifying local coordinate translations.
    """
    for i, blob in enumerate(blobs[:num_blobs]):
        minx, miny, maxx, maxy = map(int, blob.bounds)
        
        minx = max(minx, 0)
        miny = max(miny, 0)
        maxx = min(maxx, image_array.shape[1] - 1)
        maxy = min(maxy, image_array.shape[0] - 1)
        
        print(f"Visualizing Blob {i+1} bounds: {(minx, miny, maxx, maxy)}")
        
        blob_region = image_array[minx:maxx+1, miny:maxy+1]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(blob_region, cmap='gray')

        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        plot_blob(ax, blob, facecolor=color, edgecolor='black', alpha=0.4, linewidth=1.2, offset=(minx, miny))

        for point in high_contrast_areas:
            px, py = point
            if minx <= px <= maxx and miny <= py <= maxy:
                plt.scatter(px - minx, py - miny, c='red', s=10)

        plt.title(f'Blob {i+1}')
        plt.show()

def find_tiff_files(directory):
    """Recursively search for and return a sorted list of all TIFF files in the target directory."""
    tiff_files = []
    for root, _, files in os.walk(directory):
        print(f"Searching in folder: {root}")
        for file in files:
            if file.lower().endswith('.tiff'):
                full_path = os.path.join(root, file)
                tiff_files.append(full_path)
    print(f"Total TIFF files found: {len(tiff_files)}")
    tiff_files.sort()

    return sorted(tiff_files)

def create_sister_folders(tiff_files, base_dir, out_dir):
    """
    Creates isolated mirror directories ('sister folders') for storing processed data 
    inside the unified output directory.
    """
    sister_folders = []
    for tiff_file in tiff_files:
        relative_path = os.path.relpath(tiff_file, base_dir)
        sister_folder = os.path.join(out_dir, 'processed_data', os.path.dirname(relative_path) + '-data')
        os.makedirs(sister_folder, exist_ok=True)
        sister_folders.append(sister_folder)
        print(f'Created sister folder: {sister_folder}')
    return sister_folders

# =============================================================================
# IMAGE PREPROCESSING & SMOOTHING
# =============================================================================

def preprocess_image(image_array, preprocess_steps, clip_top=0, clip_bottom=10, clip_left=0, clip_right=0):
    """
    Applies initial preprocessing pipeline: edge clipping, min-max normalization, 
    and wavelet-based denoising (BayesShrink) to reduce characteristic background noise.
    """
    noise_estimation = 0
    
    if 'clip' in preprocess_steps:
        image_array = image_array[clip_top:-clip_bottom if clip_bottom != 0 else None, clip_left:-clip_right if clip_right != 0 else None]
        preprocess_steps.remove('clip')
        if not preprocess_steps:
            return image_array, noise_estimation
    
    for step in preprocess_steps:
        if step == 'normalize':
            arr_min = np.min(image_array)
            arr_max = np.max(image_array)
            image_array = (image_array - arr_min) / (arr_max - arr_min)
        elif step == 'denoise':
            noise_estimation = estimate_sigma(image_array, average_sigmas=True, channel_axis=None)
            weight = 0.5 * noise_estimation
            image_array = denoise_wavelet(image_array, sigma=noise_estimation, wavelet='sym4', method='BayesShrink', mode='hard')
    
    return image_array, noise_estimation

def calculate_gradient(image):
    """Computes the numerical gradient (magnitude and direction) of the image."""
    gx, gy = np.gradient(image)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    return gradient_magnitude, gx, gy

def adaptively_smooth_image(image, max_sigma=50, increment=1, max_gradient_ratio=0.2, max_allowed_percentage=0.8):
    """
    Iteratively increases Gaussian smoothing (sigma) until the gradient distribution 
    stabilizes, removing high-frequency artifacts while retaining core macroscopic structures.
    """
    original_gradient = calculate_gradient(image)
    max_allowed_gradient = max_gradient_ratio * np.mean(original_gradient)
    num_exceeding_pixels = np.sum(original_gradient > max_allowed_gradient)

    for sigma in np.arange(1, max_sigma + 1, increment):
        smoothed_image = gaussian_filter(image, sigma=sigma)
        current_gradient = calculate_gradient(smoothed_image)
        
        num_exceeding_pixels_after = np.sum(current_gradient > max_allowed_gradient)
        reduction_percentage = (num_exceeding_pixels - num_exceeding_pixels_after) / num_exceeding_pixels

        print(f"Sigma: {sigma}, Reduction Percentage: {reduction_percentage}")

        if reduction_percentage >= max_allowed_percentage:
            return smoothed_image, sigma

    return gaussian_filter(image, sigma=max_sigma), max_sigma

# =============================================================================
# SEGMENTATION (MULTI-OTSU) & FEATURE DETECTION
# =============================================================================

def recursive_bi_otsu(image, max_depth=10, current_depth=0, entropies=None, min_threshold=0.005):
    """
    Recursively applies Otsu's thresholding to isolate sub-regions of interest.
    Monitors Shannon entropy to prevent over-segmentation.
    """
    if entropies is None:
        entropies = []

    if current_depth >= max_depth:
        return [], entropies

    unique_values = np.unique(image)
    if len(unique_values) <= 1:
        return [], entropies

    thresholds = threshold_multiotsu(image, classes=2)
    regions = np.digitize(image, bins=thresholds)

    lower_half = image[regions == 0]

    if len(lower_half) < 2:
        return [thresholds[0]], entropies

    initial_entropy = shannon_entropy(lower_half)
    entropies.append(initial_entropy)

    thresholds_lower = threshold_multiotsu(lower_half, classes=2)
    regions_lower = np.digitize(lower_half, bins=thresholds_lower)
    lower_half_split = lower_half[regions_lower == 0]

    split_entropy = shannon_entropy(lower_half_split)

    if thresholds_lower[0] < min_threshold:
        return [thresholds[0], thresholds_lower[0]], entropies

    if split_entropy < initial_entropy:
        further_thresholds, entropies = recursive_bi_otsu(lower_half_split, max_depth, current_depth + 1, entropies, min_threshold)
        return [thresholds[0]] + further_thresholds, entropies
    else:
        return [thresholds[0]], entropies

def find_elbow_point(entropies):
    """Identifies the optimal recursion depth for Otsu by finding the maximum curvature in the entropy plot."""
    second_derivative = np.diff(entropies, n=2)
    elbow_index = np.argmax(second_derivative) + 2
    return elbow_index

def add_border_to_region(regions, region_number, border_thickness):
    """Creates a buffered border around segmented morphological regions to capture edge defects."""
    region_pixels = regions == region_number
    structure = np.ones((3, 3), dtype=bool)
    expanded_region = binary_dilation(region_pixels, structure, iterations=border_thickness)
    border = expanded_region & ~region_pixels
    return border

def create_colored_region_image(regions):
    """Maps segmented region labels to colorblind-safe, high-contrast colors for debugging/paper plots."""
    unique_regions = np.unique(regions)
    colors = get_colorblind_colors(len(unique_regions))
    colored_image = np.zeros((*regions.shape, 3), dtype=np.uint8)
    for region_value, color in zip(unique_regions, colors):
        colored_image[regions == region_value] = np.array(color) * 255
    return colored_image, colors

def is_close(coord, coords_list, closeness_threshold):
    """Evaluates Euclidean distance to ensure spatially distinct feature detection."""
    for x, y in coords_list:
        distance = np.sqrt((x - coord[0]) ** 2 + (y - coord[1]) ** 2)
        if distance <= closeness_threshold:
            return True
    return False

def has_nearby_peaks(gradient_image, coord, closeness_threshold):
    """Checks a local neighborhood to determine if a point is a true local maximum."""
    x, y = coord
    min_x, max_x = max(0, x - closeness_threshold), min(gradient_image.shape[0], x + closeness_threshold)
    min_y, max_y = max(0, y - closeness_threshold), min(gradient_image.shape[1], y + closeness_threshold)
    
    region = gradient_image[min_x:max_x, min_y:max_y]
    if np.any(region > gradient_image[x, y]):
        return True
    return False

def find_high_points(gradient_image, num_points=10, min_distance_percentage=0.1):
    """Identifies top percentile gradient peaks to locate areas of maximum localized strain/contrast."""
    high_points = []
    min_distance = int(min_distance_percentage * gradient_image.shape[0])
    threshold = np.percentile(gradient_image, 99)
    candidate_points = np.argwhere(gradient_image >= threshold)
    
    while len(high_points) < num_points and len(candidate_points) > 0:
        idx = random.choice(range(len(candidate_points)))
        point = candidate_points[idx]
        if not is_close(point, high_points, min_distance) and not has_nearby_peaks(gradient_image, point, int(min_distance/2)):
            high_points.append(point)
        candidate_points = np.delete(candidate_points, idx, 0)
    
    return high_points

def initial_find_high_contrast_areas(image_array, gradient_image, gx, gy, search_area=10, num_peaks=20):
    """
    Primary feature detection loop. Scans image using a sliding window approach 
    to extract preliminary candidate positions with the highest local contrast variations.
    """
    height, width = image_array.shape
    contrast_values = np.zeros((height, width))
    min_distance = int(0.05 * width)

    for y in range(search_area, height - search_area, search_area):
        for x in range(search_area, width - search_area, search_area):
            window = image_array[y-search_area:y+search_area+1, x-search_area:x+search_area+1]
            contrast_values[y, x] = window.max() - window.min()

    selected_peaks = []
    adjusted_peaks = []
    peak_trough_pairs = []
    directions = []
    percentile = 99.99

    while len(selected_peaks) < num_peaks and percentile > 0:
        contrast_peaks = np.argwhere(contrast_values >= np.percentile(contrast_values, percentile))
        sorted_peaks = sorted(contrast_peaks, key=lambda p: contrast_values[p[0], p[1]], reverse=True)
        
        for peak in sorted_peaks:
            if all(np.linalg.norm(peak - np.array(p)) >= min_distance for p in selected_peaks):
                y, x = peak
                window = gradient_image[max(y-search_area, 0):min(y+search_area+1, height),
                                        max(x-search_area, 0):min(x+search_area+1, width)]
                local_max = np.unravel_index(np.argmax(window), window.shape)
                adjusted_y = max(y-search_area, 0) + local_max[0]
                adjusted_x = max(x-search_area, 0) + local_max[1]
                
                high_peak_pos, low_peak_pos, direction = find_peak_and_trough_rotating_line(image_array, adjusted_y, adjusted_x, gx, gy, max_distance=30)
                
                if high_peak_pos and low_peak_pos:
                    adjusted_peaks.append([adjusted_y, adjusted_x])
                    selected_peaks.append(peak)
                    peak_trough_pairs.append(([adjusted_y, adjusted_x], high_peak_pos, low_peak_pos))
                    directions.append(direction)
                    print(f"Selected peak at {peak}")
                
                if len(selected_peaks) >= num_peaks:
                    break
        
        if len(selected_peaks) < num_peaks:
            percentile -= 0.01
            print(f"Decreasing percentile to {percentile}")
        else:
            break

    return selected_peaks, adjusted_peaks, peak_trough_pairs, directions

def find_peak_and_trough_rotating_line(image_array, y, x, gx, gy, max_distance, num_angles=360, angle_step=1):
    """
    Evaluates rotational profiles around a candidate pixel to find the axis of maximum contrast.
    This effectively captures the characteristic dipole contrast (peak-to-trough) of a specific feature.
    """
    best_angle = None
    max_contrast = 0
    high_peak_pos = None
    low_peak_pos = None
    
    for angle in np.linspace(0, 2*np.pi, num_angles):
        for dist in range(1, max_distance + 1):
            ny1 = int(y + dist * np.cos(angle))
            nx1 = int(x + dist * np.sin(angle))
            ny2 = int(y - dist * np.cos(angle))
            nx2 = int(x - dist * np.sin(angle))
            
            if 0 <= ny1 < image_array.shape[0] and 0 <= nx1 < image_array.shape[1] and 0 <= ny2 < image_array.shape[0] and 0 <= nx2 < image_array.shape[1]:
                contrast = abs(image_array[ny1, nx1] - image_array[ny2, nx2])
                if contrast > max_contrast:
                    max_contrast = contrast
                    best_angle = angle
                    high_peak_pos = (ny1, nx1)
                    low_peak_pos = (ny2, nx2)
    
    return high_peak_pos, low_peak_pos, best_angle

def remove_outliers(data, threshold=1.5):
    """Statistically filters extreme geometric measurements (IQR method) to establish baseline feature width."""
    data = np.array(data)
    
    small_data = data[data <= 2]
    data = data[data > 2]
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    mean_filtered = np.mean(filtered_data)
    large_outliers = data[data > mean_filtered * 1.5]
    filtered_data = filtered_data[filtered_data <= mean_filtered * 1.5]
    outliers = np.concatenate([outliers, large_outliers])
    
    outliers = np.concatenate([outliers, small_data])
    
    return filtered_data.tolist(), outliers.tolist()

def find_high_contrast_areas(image_array, region, extended_region, smoothed_image, search_area, closeness_threshold, high_contrast_areas=None, contrast_threshold=None):
    """
    Iterates through designated regions to find pixel clusters that exceed local contrast thresholds.
    Adjusts local maxima iteratively within a bounded constraint (`max_step`).
    """
    mean_region_value = smoothed_image[region].mean()
    
    pixels = image_array
    height, width = pixels.shape

    if high_contrast_areas is None:
        high_contrast_areas = []

    max_step = search_area / 2
    b = 4 
    print(f'Searching for high contrast areas with contrast threshold: {contrast_threshold}')
    for x in range(b*search_area, width - b*search_area, search_area):
        for y in range(b*search_area, height - b*search_area, search_area):
            if region[y, x]:
                coord = (y, x)
                if is_close(coord, high_contrast_areas, closeness_threshold):
                    continue
                else:
                    neighbors = []
                    for i in range(max(0, x - search_area), min(width, x + search_area)):
                        for j in range(max(0, y - search_area), min(height, y + search_area)):
                            if 0 <= i < width and 0 <= j < height:
                                neighbors.append(pixels[j, i]) 
                if neighbors:
                    contrast = np.max(neighbors) - np.min(neighbors)
                    if contrast > contrast_threshold / 255:
                        window = gradient_image[max(y-search_area, 0):min(y+search_area+1, height),
                                                max(x-search_area, 0):min(x+search_area+1, width)]

                        if window.size == 0:
                            continue

                        local_max = np.unravel_index(np.argmax(window), window.shape)
                        local_y = max(y-search_area, 0) + local_max[0]
                        local_x = max(x-search_area, 0) + local_max[1]

                        step_y = min(max_step, max(-max_step, local_y - y))
                        step_x = min(max_step, max(-max_step, local_x - x))

                        adjusted_y = y + step_y
                        adjusted_x = x + step_x


                        high_contrast_areas.append((adjusted_y, adjusted_x))

                        print(f'High contrast area found at ({x}, {y}) with contrast {contrast}')
    print('Found high contrast areas')
    return high_contrast_areas, contrast_threshold


# =============================================================================
# DATA RECORDING, CLEANUP, & MORPHOLOGY
# =============================================================================

def save_data(df, output_path, save_format):
    """Outputs DataFrame logs into requested format (CSV/HDF5)."""
    if save_format == 'hdf5':
        df.to_hdf(output_path, key='df', mode='w')
    elif save_format == 'csv':
        df.to_csv(output_path, index=False)
    print(f'Saved {save_format.upper()} with DataFrame: {output_path}')

def cleanup_files(out_dir):
    """Removes the entire generated output directory if a clean re-run is forced."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f'Deleted all contents in output folder: {out_dir}')

def check_and_cleanup(cleanup, out_dir):
    """Checks the cleanup flag and triggers removal prior to execution termination."""
    if cleanup:
        cleanup_files(out_dir)
        sys.exit("Cleanup complete. Exiting program.")

def select_files(tiff_files, method, number, skip, target_filename=None):
    """Selects target dataset subset for partial batch processing."""
    if method == 'random':
        selected_indices = random.sample(range(len(tiff_files)), number)
        print(f"Randomly selected file indices: {selected_indices}")
        return [tiff_files[i] for i in selected_indices], selected_indices
    elif method == 'first':
        return tiff_files[:number], list(range(number))
    elif method == 'skip':
        return tiff_files[skip:skip+number], list(range(skip, skip+number))
    elif method == 'filename':
        matches = [i for i, f in enumerate(tiff_files) if os.path.basename(f) == target_filename]
        if not matches:
            raise ValueError(f"No TIFF file named '{target_filename}' found among: {[os.path.basename(f) for f in tiff_files]}")
        return [tiff_files[i] for i in matches], matches
    else:
        raise ValueError("Invalid file selection method")

def calculate_adaptive_area_threshold(regions):
    """Determines baseline area metrics dynamically based on morphological region properties."""
    labeled_array = label(regions)
    region_props = regionprops(labeled_array)
    
    region_areas = [prop.area for prop in region_props if prop.area > 1]
    
    if region_areas:
        smallest_blob_area = min(region_areas)
    else:
        smallest_blob_area = 1
    
    print(f"Region areas: {region_areas}")
    print(f"Smallest blob area: {smallest_blob_area}")
    
    return int(np.ceil(smallest_blob_area))

def apply_morphological_operations(segmented_image, area_threshold):
    """Removes microscopic granular noise that fell below the established adaptive area threshold."""
    cleaned_image = np.zeros_like(segmented_image)
    unique_regions = np.unique(segmented_image)

    print(f"Unique regions before cleaning: {unique_regions}")

    for region_value in unique_regions:
        if region_value == 0:
            continue
        region_mask = segmented_image == region_value
        print(f"Cleaning region: {region_value}, initial area: {np.sum(region_mask)}")
        cleaned_region = remove_small_objects(region_mask, min_size=area_threshold)
        cleaned_image[cleaned_region] = region_value
        print(f"Region: {region_value}, area after cleaning: {np.sum(cleaned_region)}")

    unique_regions_after_cleaning = np.unique(cleaned_image)
    print(f"Unique regions after cleaning: {unique_regions_after_cleaning}")

    return cleaned_image

def merge_nearby_regions(cleaned_image, merge_distance=5):
    """Applies dilation and erosion to bridge small gaps, synthesizing fragmented regions."""
    def pad_with_reflect(array, pad_width):
        return np.pad(array, pad_width, mode='reflect')

    merged_image = np.zeros_like(cleaned_image)
    unique_regions = np.unique(cleaned_image)

    print(f"Unique regions before merging: {unique_regions}")

    for region_value in unique_regions:
        if region_value == 0:
            continue
        region_mask = cleaned_image == region_value

        padded_region_mask = pad_with_reflect(region_mask, merge_distance)

        dilated_region = binary_dilation(padded_region_mask, structure=disk(merge_distance))
        eroded_region = binary_erosion(dilated_region, structure=disk(merge_distance))

        eroded_region = eroded_region[merge_distance:-merge_distance, merge_distance:-merge_distance]

        merged_image[eroded_region] = region_value
        print(f"Region: {region_value}, area after merging: {np.sum(eroded_region)}")

    unique_regions_after_merging = np.unique(merged_image)
    print(f"Unique regions after merging: {unique_regions_after_merging}")

    return merged_image

def compare_original_and_gradients(preprocessed_array, gradient_image, selected_peaks, adjusted_peaks, peak_trough_pairs, directions, peak_to_peak_distances, peak_to_trough_distances, peak_to_peak_outliers, peak_to_trough_outliers, avg_peak_to_peak_distance, avg_peak_to_trough_distance):
    """Renders debug plots mapping peaks, troughs, and distances to validate orientation assumptions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes.ravel()

    ax1.imshow(preprocessed_array, cmap='gray')
    ax1.set_title('Preprocessed Image')

    ax2.imshow(preprocessed_array, cmap='gray')
    for peak, high_peak, low_trough, direction in zip(adjusted_peaks, [pt[1] for pt in peak_trough_pairs], [pt[2] for pt in peak_trough_pairs], directions):
        color = 'blue' if np.linalg.norm(np.array(peak) - np.array(high_peak)) in peak_to_peak_outliers or np.linalg.norm(np.array(peak) - np.array(low_trough)) in peak_to_trough_outliers else None
        ax2.scatter(peak[1], peak[0], c='yellow' if not color else color, s=50, marker='o')
        if high_peak:
            ax2.scatter(high_peak[1], high_peak[0], c='red' if not color else color, s=50, marker='x')
        if low_trough:
            ax2.scatter(low_trough[1], low_trough[0], c='green' if not color else color, s=50, marker='x')
        line_x = [peak[1] - 30 * np.cos(direction), peak[1] + 30 * np.cos(direction)]
        line_y = [peak[0] - 30 * np.sin(direction), peak[0] + 30 * np.sin(direction)]
        ax2.plot(line_x, line_y, 'cyan', linestyle='--')

    ax2.set_title('Peaks and Troughs in High Contrast Areas')

    ax3.bar(range(len(peak_to_peak_distances)), peak_to_peak_distances, color='r', alpha=0.3, label='Peak to Peak Distances')
    ax3.bar(range(len(peak_to_trough_distances)), peak_to_trough_distances, color='g', alpha=0.3, label='Peak to Trough Distances')
    for i, dist in enumerate(peak_to_peak_distances):
        color = 'r' if dist not in peak_to_peak_outliers else 'b'
        ax3.bar(i, dist, color=color, alpha=0.5)
        if dist in peak_to_peak_outliers:
            ax3.text(i, dist, 'x', color='blue', ha='center', va='bottom')
    for i, dist in enumerate(peak_to_trough_distances):
        color = 'g' if dist not in peak_to_trough_outliers else 'b'
        ax3.bar(i, dist, color=color, alpha=0.5)
        if dist in peak_to_trough_outliers:
            ax3.text(i, dist, 'x', color='blue', ha='center', va='bottom')

    if avg_peak_to_peak_distance:
        ax3.axhline(avg_peak_to_peak_distance, color='red', linestyle='--', label=f'Avg Peak to Peak: {avg_peak_to_peak_distance:.2f}')
    if avg_peak_to_trough_distance:
        ax3.axhline(avg_peak_to_trough_distance, color='green', linestyle='--', label=f'Avg Peak to Trough: {avg_peak_to_trough_distance:.2f}')
    ax3.set_title('Distances')
    ax3.set_xlabel('Peak Index')
    ax3.set_ylabel('Distance')
    ax3.legend()

    def on_zoom(event):
        if event.inaxes in [ax1, ax2, ax3]:
            for ax in [ax1, ax2, ax3]:
                if ax != event.inaxes:
                    ax.set_xlim(event.inaxes.get_xlim())
                    ax.set_ylim(event.inaxes.get_ylim())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_release_event', on_zoom)
    fig.canvas.mpl_connect('scroll_event', on_zoom)

    plt.tight_layout()
    plt.show()

def create_contrast_threshold_data_file(file_path):
    """Initializes CSV payload for open data publication compliance."""
    headers = ["image_index", "region_index", "mean_region_value", "region_variance", "region_std", "search_area", "smoothing_amount", "contrast_threshold", "noise_estimation", "avg_width_of_feature"]
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=headers)
        df.to_csv(file_path, index=False)
        print(f"Created file with headers: {headers}")

def append_data_tocontrast_threshold_file(file_path, data):
    """Safely appends row metadata during batch iteration."""
    df = pd.DataFrame([data])
    df.to_csv(file_path, mode='a', header=False, index=False)
    print(f"Appended data: {data} to {file_path}")

# =============================================================================
# MACHINE LEARNING / CNN DATA PREPARATION
# =============================================================================

def group_and_define_cluster_regions(data, bandwidth, avg_width_of_feature):
    """
    Uses MeanShift to cluster proximate, high-contrast target pixels into singular physical features.
    Transforms pixel coordinates into distinct geometric blobs for model training bounding boxes.
    """
    print('Grouping and defining cluster regions')
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    clusters = meanshift.fit_predict(data)

    n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
    print('Number of clusters:', n_clusters_)

    blobs = []
    radius = 0.625 * avg_width_of_feature
    for i in range(n_clusters_):
        cluster_points = data[clusters == i]
        circles = [Point(p[0], p[1]).buffer(radius) for p in cluster_points]
        blob = unary_union(circles)
        blobs.append(blob)
    print('Grouping and defining cluster blobs complete')
    return clusters, blobs, n_clusters_

def extract_base_folder_name(output_folder):
    """
    Extracts the last part of the folder name after the last '/'.
    Removes the '-data' suffix if present.
    """
    base_folder_name = os.path.basename(output_folder)
    return base_folder_name.replace('-data', '')

def save_blobs_data(file_name_base, base_folder_name, blobs, image_array, out_dir):
    """
    Strips out specific 60x60 square coordinate tensors of identified defects 
    and exports them alongside bounding geometry (JSON/txt) as ML training data inputs
    into the designated output directory.
    """
    image_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)
    
    output_folder = os.path.join(out_dir, 'training_data')
    base_folder_name = extract_base_folder_name(base_folder_name)
    
    blob_folder_name = f"{base_folder_name}_{file_name_base}_blob_training_set"

    print(f"Blob folder name: {blob_folder_name}")
    blob_folder = os.path.join(output_folder, blob_folder_name)
    
    if os.path.exists(blob_folder):
        shutil.rmtree(blob_folder)
    os.makedirs(blob_folder, exist_ok=True)
    
    raw_feature_folder = os.path.join(blob_folder, "raw_feature")
    os.makedirs(raw_feature_folder, exist_ok=True)
    
    blob_info_folder = os.path.join(blob_folder, "blob_info")
    os.makedirs(blob_info_folder, exist_ok=True)
    
    blob_info_path = os.path.join(blob_info_folder, "blob_info.txt")
    blob_shapes_path = os.path.join(blob_info_folder, "blob_shapes.json")
    
    blob_info_data = []
    blob_shapes_data = []
    
    height, width = image_array.shape

    for i, blob in enumerate(blobs):
        minx, miny, maxx, maxy = map(int, blob.bounds)

        minx = max(minx, 0)
        miny = max(miny, 0)
        maxx = min(maxx, width - 1)
        maxy = min(maxy, height - 1)

        center_x = (minx + maxx) // 2
        center_y = (miny + maxy) // 2

        image_size = 60
        half_size = 30

        start_x = max(center_x - half_size, 0)
        end_x = min(center_x + half_size, width)
        start_y = max(center_y - half_size, 0)
        end_y = min(center_y + half_size, height)

        if end_x - start_x < image_size:
            if start_x == 0:
                end_x = image_size
            else:
                start_x = end_x - image_size

        if end_y - start_y < image_size:
            if start_y == 0:
                end_y = image_size
            else:
                start_y = end_y - image_size

        blob_image = image_array[start_x:end_x, start_y:end_y]

        if blob_image.size == 0:
            print(f"Blob {i+1} image extraction failed due to size 0.")
            continue
        
        raw_feature_path = os.path.join(raw_feature_folder, f"blob_{i+1}.tiff")
        
        try:
            Image.fromarray(blob_image.astype(np.uint8)).save(raw_feature_path, format='TIFF')
        except Exception as e:
            print(f"Failed to save Blob {i+1} image: {e}")
            continue
        
        blob_info_data.append({
            "file_name": file_name_base,
            "blob_id": i+1,
            "bounds": (minx, miny, maxx, maxy),
            "bounds_buffered": (start_x, start_y, end_x, end_y)
        })

        blob_shapes_data.append({
            "blob_id": i+1,
            "shape": mapping(blob)
        })

    with open(blob_info_path, "w") as f:
        for info in blob_info_data:
            f.write(f"File Name: {info['file_name']}\n")
            f.write(f"Blob ID: {info['blob_id']}\n")
            f.write(f"Bounds minx, miny, maxx, maxy: {info['bounds']}\n")
            f.write(f"Bounds Buffered minx, miny, maxx, maxy: {info['bounds_buffered']}\n")
            f.write("\n")

    with open(blob_shapes_path, "w") as f:
        json.dump(blob_shapes_data, f, indent=4)
    
    print(f"Saved blobs data to {blob_folder}")

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

# Ensure the main output directory exists
os.makedirs(output_directory, exist_ok=True)

# Create the file with headers if it doesn't already exist
create_contrast_threshold_data_file(data_file_path)

tiff_files = find_tiff_files(base_directory)

selected_files, selected_indices = select_files(tiff_files, selection_method, num_files_to_process, skip_files, target_filename)
print(f"Selected file indices for processing: {selected_indices}")

if use_sister_folders:
    sister_folders = create_sister_folders(selected_files, base_directory, output_directory)
else:
    # If not using sister folders, still point to the main output directory
    sister_folders = [output_directory for _ in selected_files]

region_means = []
contrast_thresholds = []

for image_index, (tiff_file, output_folder) in enumerate(zip(selected_files, sister_folders)):
    print(f'Processing file: {output_folder}')
    file_name_base = os.path.splitext(os.path.basename(tiff_file))[0]

    image = Image.open(tiff_file).convert('L')
    image_array = np.array(image)
    image.close()
    image_array_clipped, _ = preprocess_image(image_array, ['clip'], clip_top=clip_top, clip_bottom=clip_bottom, clip_left=clip_left, clip_right=clip_right)
    
    preprocessed_array, noise_estimation = preprocess_image(image_array, preprocess_steps, clip_top=clip_top, clip_bottom=clip_bottom, clip_left=clip_left, clip_right=clip_right)
    gradient_image, gx, gy = calculate_gradient(preprocessed_array)
    
    selected_peaks, adjusted_peaks, peak_trough_pairs, directions = initial_find_high_contrast_areas(preprocessed_array, gradient_image, gx, gy, search_area=3, num_peaks=num_peaks)

    print(f"Number of peak-trough pairs found: {len(peak_trough_pairs)}")

    peak_to_peak_distances = []
    peak_to_trough_distances = []

    for peak, high_peak, low_trough in peak_trough_pairs:
        if high_peak:
            peak_to_peak_distances.append(np.linalg.norm(np.array(peak) - np.array(high_peak)))
        if low_trough:
            peak_to_trough_distances.append(np.linalg.norm(np.array(peak) - np.array(low_trough)))

    peak_to_peak_distances_filtered, peak_to_peak_outliers = remove_outliers(peak_to_peak_distances)
    peak_to_trough_distances_filtered, peak_to_trough_outliers = remove_outliers(peak_to_trough_distances)

    avg_peak_to_peak_distance = np.mean(peak_to_peak_distances_filtered) if peak_to_peak_distances_filtered else None
    avg_peak_to_trough_distance = np.mean(peak_to_trough_distances_filtered) if peak_to_trough_distances_filtered else None
    
    avg_width_of_feature = 2*(avg_peak_to_peak_distance+avg_peak_to_trough_distance) - shrink_width

    print(f"Average peak to peak distance (filtered): {avg_peak_to_peak_distance}")
    print(f"Average peak to trough distance (filtered): {avg_peak_to_trough_distance}")

    if PLOT_SCALE_FINDING:
        compare_original_and_gradients(preprocessed_array, gradient_image, selected_peaks, adjusted_peaks, peak_trough_pairs, directions, peak_to_peak_distances, peak_to_trough_distances, peak_to_peak_outliers, peak_to_trough_outliers, avg_peak_to_peak_distance, avg_peak_to_trough_distance)

    smoothed_array, smoothing_amount = adaptively_smooth_image(preprocessed_array, max_gradient_ratio=max_gradient_ratio, max_allowed_percentage=max_allowed_percentage)
    print(f"Chosen sigma for smoothing: {smoothing_amount}")
    
    thresholds, entropies = recursive_bi_otsu(smoothed_array, max_depth=max_iterations)
    print(f'Determined thresholds using recursive bi-Otsu: {thresholds}')
    print(f'Entropy values during recursion: {entropies}')
    
    optimal_iterations = find_elbow_point(entropies)
    print(f'Optimal number of iterations: {optimal_iterations}')
    
    thresholds, _ = recursive_bi_otsu(smoothed_array, max_depth=optimal_iterations)
    
    if manual_region_adjustment and len(thresholds) > 1:
        thresholds = thresholds[:-1]
    
    print(f'Final Otsu thresholds: {thresholds}')
    
    regions = np.digitize(smoothed_array, bins=thresholds)

    area_threshold = int(np.pi*(avg_width_of_feature/2)**2)

    cleaned_image = apply_morphological_operations(regions, area_threshold=area_threshold)
    cleaned_image = merge_nearby_regions(cleaned_image, merge_distance=5)
    colored_region_image_before, colors_before = create_colored_region_image(regions)
    colored_region_image_after, colors_after = create_colored_region_image(cleaned_image)
    unique_regions = np.unique(cleaned_image)
    regions_list = [(cleaned_image == region) for region in unique_regions]
    border_thickness = int(avg_width_of_feature/2)
    print('border thickness:', border_thickness)
    
    borders = []
    regions_with_borders = []

    for region_index in range(len(unique_regions)):
        region_mean = preprocessed_array[cleaned_image == unique_regions[region_index]].mean()
        region_variance = preprocessed_array[cleaned_image == unique_regions[region_index]].var()
        region_std = preprocessed_array[cleaned_image == unique_regions[region_index]].std()
        region_means.append(region_mean)
        print(f'Mean value of region {region_index + 1}: {region_mean}')
        
        border = add_border_to_region(cleaned_image, unique_regions[region_index], border_thickness)
        borders.append(border)
        region_with_border = cleaned_image == unique_regions[region_index]
        region_with_border = np.logical_or(region_with_border, border)
        regions_with_borders.append(region_with_border)
        
    search_area = int(avg_width_of_feature * 1/4)
    if search_area < 2:
        search_area = 2
    closeness_threshold = 2
    
    all_high_contrast_areas = []
    contrast_thresholds_for_plot = []
    
    for region_index, (region, extended_region, contrast_threshold) in enumerate(zip(regions_list, regions_with_borders, manual_contrast_thresholds)):
        print(f'Processing region {region_index + 1}')
        high_contrast_areas, _ = find_high_contrast_areas(preprocessed_array, region, extended_region, smoothed_array, search_area, closeness_threshold, high_contrast_areas=[], contrast_threshold=contrast_threshold)
        all_high_contrast_areas.extend(high_contrast_areas)
        contrast_thresholds_for_plot.append(contrast_threshold)
    
        data = {
            "image_index": selected_indices[image_index] + 1,
            "region_index": region_index + 1,
            "mean_region_value": region_mean,
            "region_variance": region_variance,
            "region_std": region_std,
            "search_area": search_area,
            "smoothing_amount": smoothing_amount,
            "contrast_threshold": manual_contrast_thresholds[region_index],
            "noise_estimation": noise_estimation,
            "avg_width_of_feature": avg_width_of_feature
        }
        
        if save_data_to_file:
            append_data_tocontrast_threshold_file(data_file_path, data)

    if CLUSTER:
        if len(all_high_contrast_areas) > 0:
            all_high_contrast_areas = np.array(all_high_contrast_areas)
            
            print(f"High contrast areas (shape: {all_high_contrast_areas.shape}): {all_high_contrast_areas}")

            if all_high_contrast_areas.ndim == 1:
                all_high_contrast_areas = all_high_contrast_areas.reshape(-1, 2)
            
            clusters, blobs, n_clusters_ = group_and_define_cluster_regions(all_high_contrast_areas, avg_width_of_feature/2, avg_width_of_feature)
        else:
            clusters, blobs, n_clusters_ = [], [], 0

    if save_blobs:
        save_blobs_data(file_name_base, output_folder, blobs, preprocessed_array, output_directory)

    if PLOT_IMAGE_PREPROCESSED_SMOOTHED:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        ax1.imshow(image_array, cmap='gray')
        ax1.set_title('Original Image')
        ax2.imshow(preprocessed_array, cmap='gray')
        ax2.set_title('Preprocessed Image')
        ax3.imshow(smoothed_array, cmap='gray')
        ax3.set_title('Smoothed Image')
        plt.show()

    if PLOT_BEFORE_AND_AFTER_CLEANUP:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(colored_region_image_before)
        ax1.set_title('Regions Before Cleanup')
        ax2.imshow(colored_region_image_after)
        ax2.set_title('Regions After Cleanup')
        legend_elements_before = [plt.Line2D([0], [0], marker='o', color='w', label=f'Region {i+1}', 
                                            markerfacecolor=tuple(color), markersize=10) for i, color in enumerate(colors_before)]
        ax1.legend(handles=legend_elements_before, loc='upper right')
        legend_elements_after = [plt.Line2D([0], [0], marker='o', color='w', label=f'Region {i+1}', 
                                            markerfacecolor=tuple(color), markersize=10) for i, color in enumerate(colors_after)]
        ax2.legend(handles=legend_elements_after, loc='upper right')
        plt.tight_layout()
        plt.show()

    if PLOT_HISTOGRAM:
        plt.figure(figsize=(8, 6))
        plt.hist(smoothed_array.flatten(), bins=256, color='gray', alpha=0.7)
        plt.title('Histogram of Smoothed Image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        for threshold in thresholds:
            plt.axvline(x=threshold, color='red', linestyle='--')
        plt.show()

    if PLOT_REGIONS_WITH_BORDERS:
        fig, axs = plt.subplots(1, len(regions_with_borders) + 1, figsize=(15, 5))
        
        axs[0].imshow(colored_region_image_after)
        axs[0].set_title('All Regions with Colors')
        
        for idx, (region_with_border, color) in enumerate(zip(regions_with_borders, colors_after), start=1):
            axs[idx].imshow(region_with_border, cmap='gray')
            axs[idx].imshow(np.dstack((region_with_border * color[0], region_with_border * color[1], region_with_border * color[2])), alpha=0.5)
            axs[idx].set_title(f'Region {idx} with Border')
        
        plt.tight_layout()
        plt.show()

    if PLOT_ENTROPY:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(entropies) + 1), entropies, marker='o')
        plt.title('Entropy Values for Number of Regions')
        plt.xlabel('Number of Regions')
        plt.ylabel('Entropy')
        plt.grid()
        plt.show()
    
    if PLOT_HIGH_CONTRAST_AREAS:
        plt.figure(figsize=(8, 8))
        plt.imshow(preprocessed_array, cmap='gray')
        plt.scatter([p[1] for p in all_high_contrast_areas], [p[0] for p in all_high_contrast_areas], c='red', s=1)
        plt.title('High Contrast Areas on Preprocessed Image')
        plt.show()
    
    if PLOT_COLORED_REGIONS_AND_CONTRAST_AREAS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(preprocessed_array, cmap='gray')
        ax1.scatter([p[1] for p in all_high_contrast_areas], [p[0] for p in all_high_contrast_areas], c='red', s=1, alpha=0.5)
        if CLUSTER:
            blob_colors = get_colorblind_colors(len(blobs))
            for blob, color in zip(blobs, blob_colors):
                plot_blob(ax1, blob, facecolor=color, edgecolor='black', alpha=PAPER_BLOB_ALPHA, linewidth=1.0)

        ax1.set_title('High Contrast Areas')    

        ax2.imshow(preprocessed_array, cmap='gray')
        ax2.scatter([p[1] for p in all_high_contrast_areas], [p[0] for p in all_high_contrast_areas], c='red', s=1, alpha=0.5)
        ax2.imshow(colored_region_image_after, alpha=0.3)
        ax2.set_title('Colored Regions')
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Region {i+1}', 
                                      markerfacecolor=tuple(color), markersize=10) for i, color in enumerate(colors_after)]
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        rect_size = 2 * search_area + 1
        rect = Rectangle((0.9, 0.1), rect_size, rect_size, linewidth=1, edgecolor='r', facecolor='none', transform=ax1.transData)
        ax1.add_patch(rect)
        ax1.text(0.95, 0.05, f'Search Area: {search_area}px', transform=ax1.transAxes, color='red', fontsize=12, ha='center')

        def on_zoom(event):
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
            fig.canvas.draw_idle()

        ax1.callbacks.connect('xlim_changed', on_zoom)
        ax1.callbacks.connect('ylim_changed', on_zoom)

        plt.tight_layout()
        plt.show()

    if SAVE_PLOTS_FOR_PAPER:
        generate_paper_figures(image_array, preprocessed_array, smoothed_array, blobs, all_high_contrast_areas,
                                colored_region_image_before, colors_before,
                                colored_region_image_after, colors_after,
                                regions_with_borders, thresholds, entropies,
                                region_means, contrast_thresholds_for_plot,
                                output_folder, file_name_base)

    if CLUSTER:
        if len(all_high_contrast_areas) > 0:
            all_high_contrast_areas = np.array(all_high_contrast_areas)

            print(f"High contrast areas (shape: {all_high_contrast_areas.shape}): {all_high_contrast_areas}")

            if all_high_contrast_areas.ndim == 1:
                all_high_contrast_areas = all_high_contrast_areas.reshape(-1, 2)

            clusters, blobs, n_clusters_ = group_and_define_cluster_regions(all_high_contrast_areas, avg_width_of_feature/2, avg_width_of_feature)

        else:
            clusters, blobs, n_clusters_ = [], [], 0

    if PLOT_REGION_MEANS_VS_CONTRAST_THRESHOLDS:
        plt.figure(figsize=(8, 6))
        plt.scatter(region_means, contrast_thresholds_for_plot)
        plt.title('Region Mean Values vs Contrast Thresholds')
        plt.xlabel('Mean Value of Region')
        plt.ylabel('Contrast Threshold')
        plt.grid()
        plt.show()

    df = pd.DataFrame(preprocessed_array.flatten(), columns=['Values'])

    save_path = os.path.join(output_folder, f"{file_name_base}.{save_format}")
    save_data(df, save_path, save_format)

print("\nRegion Means and Corresponding Contrast Thresholds:")
for mean, threshold in zip(region_means, contrast_thresholds_for_plot):
    print(f"Mean: {mean}, Threshold: {threshold}")

print(f"Processed file indices: {[i + 1 for i in selected_indices]} / {len(tiff_files)} images")

check_and_cleanup(cleanup, output_directory)