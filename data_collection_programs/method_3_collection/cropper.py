#!/usr/bin/env python3
"""
Two-Stage Training Data Preparation (fixed cropping + preview)
- Tight crops around detection boxes and/or annotations (whichever is larger)
- Proper padding in pixels
- Saves preview images with crop box + detection box + annotation overlay
"""

from pathlib import Path
import shutil
import cv2
import numpy as np

# Configuration

DATA_ROOT = Path("data/YOLO_training_set/comprehensive_dataset")
WINDOW_TYPE = 'random_windows'  # or 'cluster_windows'
WINDOW_SOURCE = DATA_ROOT / "exported_windows" / WINDOW_TYPE

# Output for centered crops
EXPORT_ROOT = DATA_ROOT / "exported_windows_two_stage" / WINDOW_TYPE

# Crop parameters for pose/segment/obb
CROP_PADDING_FACTOR = 1.5  # 50% padding around bbox (1.5 => 50% extra on both axes)
MIN_CROP_SIZE = 10
MAX_CROP_SIZE = 256

# Stage 1: detect (windows copied as-is)

def copy_detect_data():
    """For detect task, just copy existing windows"""
    print(f"\n{'='*60}")
    print(f"STAGE 1: Detect - copying existing windows")
    print(f"{'='*60}\n")
    
    src_img = WINDOW_SOURCE / 'images' / 'train_val'
    src_lbl = WINDOW_SOURCE / 'labels' / 'detect' / 'train_val'
    
    dst_img = EXPORT_ROOT / 'detect' / 'images'
    dst_lbl = EXPORT_ROOT / 'detect' / 'labels'
    
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_path in sorted(src_img.iterdir()):
        if img_path.suffix.lower() not in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            continue
        
        lbl_path = src_lbl / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        
        shutil.copy2(img_path, dst_img / img_path.name)
        shutil.copy2(lbl_path, dst_lbl / f"{img_path.stem}.txt")
        count += 1
    
    print(f"Copied {count} detection windows")
    print(f"Output: {EXPORT_ROOT / 'detect'}\n")
    return count

# Stage 2: pose/segment/obb (centered crops)

def create_centered_crops(task_type):
    """Create crops around detection boxes for pose/segment/obb tasks and save preview overlays"""
    print(f"\n{'='*60}")
    print(f"STAGE 2: {task_type.upper()} - creating centered crops")
    print(f"{'='*60}\n")
    
    src_img = WINDOW_SOURCE / 'images' / 'train_val'
    src_detect = WINDOW_SOURCE / 'labels' / 'detect' / 'train_val'
    src_task = WINDOW_SOURCE / 'labels' / task_type / 'train_val'
    
    dst_img = EXPORT_ROOT / task_type / 'images'
    dst_lbl = EXPORT_ROOT / task_type / 'labels'
    dst_preview = EXPORT_ROOT / task_type / 'previews'
    
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    dst_preview.mkdir(parents=True, exist_ok=True)
    
    total_crops = 0
    
    for img_path in sorted(src_img.iterdir()):
        if img_path.suffix.lower() not in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            continue
        
        detect_lbl = src_detect / f"{img_path.stem}.txt"
        task_lbl = src_task / f"{img_path.stem}.txt"
        
        if not detect_lbl.exists() or not task_lbl.exists():
            continue
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape
        
        # parse detection boxes (YOLO normalized -> pixel)
        detect_boxes = []
        for line in detect_lbl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                parts = [float(x) for x in parts]
            except:
                continue
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = parts[1:5]
            x0 = int(round((xc - bw/2) * w))
            y0 = int(round((yc - bh/2) * h))
            x1 = int(round((xc + bw/2) * w))
            y1 = int(round((yc + bh/2) * h))
            # clamp
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            detect_boxes.append({
                'class': cls,
                'box_px': (x0, y0, x1, y1),
                'box_norm': (xc, yc, bw, bh)
            })
        
        # load task annotations in pixel coordinates
        task_annotations = load_task_annotations(task_lbl, task_type, w, h)
        
        # for preview overlay we will draw everything
        preview_overlay = img_rgb.copy()
        
        for crop_idx, det in enumerate(detect_boxes):
            x0, y0, x1, y1 = det['box_px']
            det_w = x1 - x0
            det_h = y1 - y0
            det_cx = (x0 + x1) / 2
            det_cy = (y0 + y1) / 2
            
            # draw detection box on preview (red)
            cv2.rectangle(preview_overlay, (x0, y0), (x1, y1), (0,0,255), 1)
            
            # find a matching annotation (closest center within reasonable distance)
            task_annot = find_matching_annotation(task_annotations, det_cx, det_cy, det_w, det_h)
            # compute annotation bbox in pixels (if available) to make crop tight to annotation
            annot_bbox = None
            if task_annot is not None:
                if task_type == 'pose' and 'keypoints' in task_annot:
                    kpts = [(kx, ky) for (kx, ky, vis) in task_annot['keypoints'] if vis > 0]
                    if kpts:
                        xs = [p[0] for p in kpts]; ys = [p[1] for p in kpts]
                        ax0, ay0 = int(min(xs)), int(min(ys))
                        ax1, ay1 = int(max(xs)), int(max(ys))
                        annot_bbox = (ax0, ay0, ax1, ay1)
                        # draw keypoints on preview
                        for (kx, ky, vis) in task_annot['keypoints']:
                            if vis > 0:
                                cv2.circle(preview_overlay, (int(round(kx)), int(round(ky))), 2, (255,0,0), -1)
                elif task_type == 'segment' and 'polygon' in task_annot:
                    poly = [(int(round(px)), int(round(py))) for (px, py) in task_annot['polygon']]
                    if poly:
                        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                        ax0, ay0 = int(min(xs)), int(min(ys))
                        ax1, ay1 = int(max(xs)), int(max(ys))
                        annot_bbox = (ax0, ay0, ax1, ay1)
                        # draw polygon on preview
                        cv2.polylines(preview_overlay, [np.array(poly, dtype=np.int32)], isClosed=True, color=(255,0,0), thickness=1)
                elif task_type == 'obb' and 'corners' in task_annot:
                    corners = [(int(round(px)), int(round(py))) for (px, py) in task_annot['corners']]
                    if corners:
                        xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
                        ax0, ay0 = int(min(xs)), int(min(ys))
                        ax1, ay1 = int(max(xs)), int(max(ys))
                        annot_bbox = (ax0, ay0, ax1, ay1)
                        # draw corners
                        for c in corners:
                            cv2.circle(preview_overlay, c, 2, (255,0,0), -1)
            
            # Determine base bbox to use for crop: prefer annotation bbox if available & larger
            base_w = det_w
            base_h = det_h
            base_cx = det_cx
            base_cy = det_cy
            if annot_bbox is not None:
                ax0, ay0, ax1, ay1 = annot_bbox
                aw = ax1 - ax0
                ah = ay1 - ay0
                a_cx = (ax0 + ax1) / 2
                a_cy = (ay0 + ay1) / 2
                # Use annotation bbox if it's larger than detection bbox (gives tightness to annotation)
                if max(aw, ah) > max(base_w, base_h):
                    base_w, base_h, base_cx, base_cy = aw, ah, a_cx, a_cy
            
            # compute crop size from base bbox + padding
            crop_size = int(round(max(base_w, base_h) * CROP_PADDING_FACTOR))
            crop_size = max(MIN_CROP_SIZE, min(MAX_CROP_SIZE, crop_size))
            
            crop_cx = int(round(base_cx))
            crop_cy = int(round(base_cy))
            crop_x0 = max(0, crop_cx - crop_size // 2)
            crop_y0 = max(0, crop_cy - crop_size // 2)
            crop_x1 = min(w, crop_x0 + crop_size)
            crop_y1 = min(h, crop_y0 + crop_size)
            
            # Adjust if we hit boundaries
            if (crop_x1 - crop_x0) < crop_size:
                crop_x0 = max(0, crop_x1 - crop_size)
            if (crop_y1 - crop_y0) < crop_size:
                crop_y0 = max(0, crop_y1 - crop_size)
            
            # draw crop rectangle on preview (green)
            cv2.rectangle(preview_overlay, (crop_x0, crop_y0), (crop_x1, crop_y1), (0,255,0), 1)
            
            # Extract and save crop
            crop_img = img_rgb[crop_y0:crop_y1, crop_x0:crop_x1]
            crop_w = crop_x1 - crop_x0
            crop_h = crop_y1 - crop_y0
            crop_name = f"{img_path.stem}_crop{crop_idx:03d}.png"
            cv2.imwrite(str(dst_img / crop_name), crop_img)
            
            # Convert task annotation to crop coordinates (normalized)
            # Pass the detection bbox as well so we can compute a consistent centered bbox in crop coords
            crop_label = convert_to_crop_coords(
                task_annot, task_type, det['class'],
                (crop_x0, crop_y0, crop_x1, crop_y1),
                crop_w, crop_h,
                det_bbox_px=(x0, y0, x1, y1)
            )
            
            if crop_label:
                (dst_lbl / f"{Path(crop_name).stem}.txt").write_text(crop_label)
                total_crops += 1
        
        # Save preview overlay for this image
        preview_name = f"preview_{img_path.stem}.png"
        cv2.imwrite(str(dst_preview / preview_name), preview_overlay)
    
    print(f"Created {total_crops} crops for {task_type}")
    print(f"Output: {EXPORT_ROOT / task_type}")
    print(f"Previews: {EXPORT_ROOT / task_type / 'previews'}\n")
    return total_crops

def load_task_annotations(label_path, task_type, img_w, img_h):
    """Load annotations and convert to pixel coordinates"""
    annotations = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            parts = [float(x) for x in parts]
        except:
            continue
        cls = int(parts[0])
        if task_type == 'pose':
            if len(parts) < 5:
                continue
            xc, yc, bw, bh = parts[1:5]
            keypoints = parts[5:]
            bbox_cx = xc * img_w
            bbox_cy = yc * img_h
            kpts_px = []
            for i in range(0, len(keypoints), 3):
                if i+2 < len(keypoints):
                    kx, ky, vis = keypoints[i:i+3]
                    kpts_px.append((kx * img_w, ky * img_h, vis))
            annotations.append({
                'class': cls,
                'center': (bbox_cx, bbox_cy),
                'keypoints': kpts_px,
                'raw': parts[1:]
            })
        elif task_type == 'segment':
            points = parts[1:]
            poly_px = []
            for i in range(0, len(points), 2):
                if i+1 < len(points):
                    poly_px.append((points[i] * img_w, points[i+1] * img_h))
            if poly_px:
                poly_cx = np.mean([p[0] for p in poly_px])
                poly_cy = np.mean([p[1] for p in poly_px])
                annotations.append({
                    'class': cls,
                    'center': (poly_cx, poly_cy),
                    'polygon': poly_px,
                    'raw': parts[1:]
                })
        elif task_type == 'obb':
            corners = parts[1:9]
            if len(corners) == 8:
                corners_px = []
                for i in range(0, 8, 2):
                    corners_px.append((corners[i] * img_w, corners[i+1] * img_h))
                obb_cx = np.mean([c[0] for c in corners_px])
                obb_cy = np.mean([c[1] for c in corners_px])
                annotations.append({
                    'class': cls,
                    'center': (obb_cx, obb_cy),
                    'corners': corners_px,
                    'raw': parts[1:9]
                })
    return annotations

def find_matching_annotation(annotations, det_cx, det_cy, det_w, det_h):
    """Find annotation closest to detection center (distance threshold relative to detection size)"""
    min_dist = float('inf')
    best_annot = None
    threshold = max(det_w, det_h) * 0.75  # tighter threshold
    for annot in annotations:
        annot_cx, annot_cy = annot['center']
        dist = np.hypot(det_cx - annot_cx, det_cy - annot_cy)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            best_annot = annot
    return best_annot

def convert_to_crop_coords(annot, task_type, cls, crop_bbox, crop_w, crop_h, det_bbox_px):
    """
    Convert annotation from image coords to crop-normalized coords suitable for training labels.
    det_bbox_px: the original detection box in pixel coords (x0,y0,x1,y1) - used to create a bbox label centered in crop
    """
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_bbox
    if annot is None:
        return None
    
    if task_type == 'pose':
        # Use detection bbox as the object's bbox in normalized crop coordinates (centered)
        dx0, dy0, dx1, dy1 = det_bbox_px
        db_w = dx1 - dx0
        db_h = dy1 - dy0
        # bbox center relative to crop
        bbox_xc = ((dx0 + dx1) / 2 - crop_x0) / crop_w
        bbox_yc = ((dy0 + dy1) / 2 - crop_y0) / crop_h
        bbox_w = db_w / crop_w
        bbox_h = db_h / crop_h
        # clamp to (0,1)
        bbox_xc = float(np.clip(bbox_xc, 0.0, 1.0))
        bbox_yc = float(np.clip(bbox_yc, 0.0, 1.0))
        bbox_w = float(np.clip(bbox_w, 1e-6, 1.0))
        bbox_h = float(np.clip(bbox_h, 1e-6, 1.0))
        
        parts = [str(cls), f"{bbox_xc:.6f}", f"{bbox_yc:.6f}", f"{bbox_w:.6f}", f"{bbox_h:.6f}"]
        
        # keypoints -> normalized crop coords
        for kx_px, ky_px, vis in annot['keypoints']:
            if vis > 0:
                if crop_x0 <= kx_px <= crop_x1 and crop_y0 <= ky_px <= crop_y1:
                    kx_crop = (kx_px - crop_x0) / crop_w
                    ky_crop = (ky_px - crop_y0) / crop_h
                    parts.extend([f"{kx_crop:.6f}", f"{ky_crop:.6f}", "2"])
                else:
                    parts.extend(["0.0", "0.0", "0"])
            else:
                parts.extend(["0.0", "0.0", "0"])
        # pad to 4 keypoints
        num_kpts = len(annot['keypoints'])
        for _ in range(4 - num_kpts):
            parts.extend(["0.0", "0.0", "0"])
        return " ".join(parts)
    
    elif task_type == 'segment':
        crop_points = []
        for px, py in annot['polygon']:
            crop_px = np.clip((px - crop_x0) / crop_w, 0.0, 1.0)
            crop_py = np.clip((py - crop_y0) / crop_h, 0.0, 1.0)
            crop_points.extend([crop_px, crop_py])
        if len(crop_points) >= 6:
            parts = [str(cls)] + [f"{v:.6f}" for v in crop_points]
            return " ".join(parts)
    
    elif task_type == 'obb':
        crop_corners = []
        for cx, cy in annot['corners']:
            crop_cx = np.clip((cx - crop_x0) / crop_w, 0.0, 1.0)
            crop_cy = np.clip((cy - crop_y0) / crop_h, 0.0, 1.0)
            crop_corners.extend([crop_cx, crop_cy])
        parts = [str(cls)] + [f"{v:.6f}" for v in crop_corners]
        return " ".join(parts)
    
    return None

# Main

def main():
    print(f"\n{'='*60}")
    print(f"Two-Stage Training Data Preparation (fixed cropping + previews)")
    print(f"{'='*60}\n")
    print(f"Source: {WINDOW_SOURCE}")
    print(f"Output: {EXPORT_ROOT}\n")
    
    detect_count = copy_detect_data()
    
    stats = {}
    for task in ['pose', 'segment', 'obb']:
        count = create_centered_crops(task)
        stats[task] = count
    
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Detect: {detect_count} windows (as-is)")
    for task, count in stats.items():
        print(f"{task.capitalize()}: {count} centered crops")
    print(f"\nOutput: {EXPORT_ROOT}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

