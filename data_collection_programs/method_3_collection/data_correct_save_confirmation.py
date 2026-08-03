#!/usr/bin/env python3
"""
Simple YOLO label visualizer - shows all annotation types on random images
"""

import cv2
import numpy as np
import random
from pathlib import Path

class Config:
    DATA_ROOT = Path("data/YOLO_training_set/comprehensive_dataset")
    CHARACTERS = ['edge', 'screw', 'mixed', 'unknown']
    COLORS = {
        'edge': (255, 50, 50),      # Red
        'screw': (50, 255, 50),     # Green
        'mixed': (50, 50, 255),     # Blue
        'unknown': (255, 255, 50)   # Yellow
    }

def load_labels(label_path):
    """Load labels from a text file"""
    if not label_path.exists():
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                labels.append([float(x) for x in parts])
    return labels

def draw_detect_boxes(img, labels):
    """Draw detection boxes (class xc yc w h)"""
    h, w = img.shape[:2]
    
    for label in labels:
        cls = int(label[0])
        xc, yc, bw, bh = label[1:5]
        
        # Convert normalized to pixel coordinates
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        
        color = Config.COLORS[Config.CHARACTERS[cls]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

def draw_pose_keypoints(img, labels):
    """Draw pose keypoints (class xc yc w h x1 y1 v1 x2 y2 v2 ...)"""
    h, w = img.shape[:2]
    
    for label in labels:
        cls = int(label[0])
        xc, yc, bw, bh = label[1:5]
        
        # Draw bbox
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        
        color = Config.COLORS[Config.CHARACTERS[cls]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        # Draw keypoints (up to 4 keypoints, 3 values each: x, y, visibility)
        keypoints = label[5:]
        for i in range(0, len(keypoints), 3):
            if i+2 < len(keypoints):
                kx, ky, vis = keypoints[i:i+3]
                if vis > 0:  # Only draw visible keypoints
                    px = int(kx * w)
                    py = int(ky * h)
                    # Alternate colors for dark/bright lobes
                    kpt_color = (255, 0, 255) if (i//3) % 2 == 0 else (0, 255, 255)
                    cv2.circle(img, (px, py), 3, kpt_color, -1)
                    cv2.circle(img, (px, py), 4, (255, 255, 255), 1)

def draw_segment_polygons(img, labels):
    """Draw segmentation polygons (class x1 y1 x2 y2 x3 y3 ...)"""
    h, w = img.shape[:2]
    
    for label in labels:
        cls = int(label[0])
        points = label[1:]
        
        # Convert to pixel coordinates
        poly_points = []
        for i in range(0, len(points), 2):
            if i+1 < len(points):
                px = int(points[i] * w)
                py = int(points[i+1] * h)
                poly_points.append([px, py])
        
        if len(poly_points) >= 3:
            poly_points = np.array(poly_points, dtype=np.int32)
            color = Config.COLORS[Config.CHARACTERS[cls]]
            
            # Draw filled polygon with transparency
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly_points], color)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            # Draw outline
            cv2.polylines(img, [poly_points], True, color, 1)

def draw_obb_boxes(img, labels):
    """Draw oriented bounding boxes (class x1 y1 x2 y2 x3 y3 x4 y4)"""
    h, w = img.shape[:2]
    
    for label in labels:
        cls = int(label[0])
        points = label[1:]
        
        if len(points) >= 8:
            # Convert to pixel coordinates
            corners = []
            for i in range(0, 8, 2):
                px = int(points[i] * w)
                py = int(points[i+1] * h)
                corners.append([px, py])
            
            corners = np.array(corners, dtype=np.int32)
            color = Config.COLORS[Config.CHARACTERS[cls]]
            
            # Draw rotated box
            cv2.polylines(img, [corners], True, color, 1)
            
            # Draw orientation line (from corner 0 to corner 1)
            cv2.line(img, tuple(corners[0]), tuple(corners[1]), (0, 255, 255), 1)

def visualize_random_image():
    """Load and visualize a random image with all its labels"""
    
    # Get all images
    image_dir = Config.DATA_ROOT / 'images' / 'all'
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    image_files = list(image_dir.glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']]
    
    if not image_files:
        print(f"Error: No images found in {image_dir}")
        return
    
    # Pick random image
    img_path = random.choice(image_files)
    print(f"\nLoading: {img_path.name}")
    
    # Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return
    
    # Convert to color for visualization
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Create separate images for each annotation type
    img_detect = img.copy()
    img_pose = img.copy()
    img_segment = img.copy()
    img_obb = img.copy()
    
    stem = img_path.stem
    
    # Load and draw each type
    print("\nAnnotations found:")
    
    # Detection
    detect_labels = load_labels(Config.DATA_ROOT / 'labels' / 'detect' / 'all' / f'{stem}.txt')
    if detect_labels:
        print(f"  ✓ Detect: {len(detect_labels)} boxes")
        draw_detect_boxes(img_detect, detect_labels)
    else:
        print(f"  ✗ Detect: No labels")
    
    # Pose
    pose_labels = load_labels(Config.DATA_ROOT / 'labels' / 'pose' / 'all' / f'{stem}.txt')
    if pose_labels:
        print(f"  ✓ Pose: {len(pose_labels)} boxes with keypoints")
        draw_pose_keypoints(img_pose, pose_labels)
    else:
        print(f"  ✗ Pose: No labels")
    
    # Segment
    segment_labels = load_labels(Config.DATA_ROOT / 'labels' / 'segment' / 'all' / f'{stem}.txt')
    if segment_labels:
        print(f"  ✓ Segment: {len(segment_labels)} polygons")
        draw_segment_polygons(img_segment, segment_labels)
    else:
        print(f"  ✗ Segment: No labels")
    
    # OBB
    obb_labels = load_labels(Config.DATA_ROOT / 'labels' / 'obb' / 'all' / f'{stem}.txt')
    if obb_labels:
        print(f"  ✓ OBB: {len(obb_labels)} oriented boxes")
        draw_obb_boxes(img_obb, obb_labels)
    else:
        print(f"  ✗ OBB: No labels")
    
    # Display all views
    print("\nDisplaying windows:")
    print("  - Press any key to load another random image")
    print("  - Press 'q' or ESC to quit")
    
    # Scale UP images by a factor
    scale_factor = 0.5  # 1.5x bigger, adjust this value to make images larger/smaller
    h, w = img.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    img_detect = cv2.resize(img_detect, (new_w, new_h))
    img_pose = cv2.resize(img_pose, (new_w, new_h))
    img_segment = cv2.resize(img_segment, (new_w, new_h))
    img_obb = cv2.resize(img_obb, (new_w, new_h))
    
    # Stack images in a 2x2 grid
    top_row = np.hstack([img_detect, img_pose])
    bottom_row = np.hstack([img_segment, img_obb])
    combined = np.vstack([top_row, bottom_row])
    
    cv2.imshow('YOLO Labels Visualizer', combined)
    
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return True to continue, False to quit
    return key not in [ord('q'), 27]  # 27 is ESC

def main():
    print("="*60)
    print("YOLO Label Visualizer")
    print("="*60)
    print(f"Dataset: {Config.DATA_ROOT}")
    print("="*60)
    
    # Keep showing random images until user quits
    while True:
        if not visualize_random_image():
            break
    
    print("\nDone!")

if __name__ == "__main__":
    main()
