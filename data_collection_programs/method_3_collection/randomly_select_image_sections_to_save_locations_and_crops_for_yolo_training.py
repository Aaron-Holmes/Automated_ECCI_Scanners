#!/usr/bin/env python3
# random_cluster_window_exporter.py  (show + save + random & cluster window export)
# -------------------------------------------------------------------
#  – Prompts for TIFF/PNG/JPG and matching CSV
#  – Shows rectangles with TIFFViewer colours
#  – Builds cluster windows and/or random windows (variable size ±20%)
#  – For random: preview a resizable, fixed‐center window, density & preview‐count via sliders before export
#    * Full image is fit to dialog; sliders adjust width, height, density (# windows), and preview count
#    * Live preview of sample window placements up to the chosen preview count
#  – After drawing, exports each window as TIFF (+YOLO label) into organized folders
#  – Optionally, can export the full image with all "dislocation" boxes as class 0
#  – Progress printed and shown in status-bar
#  – Esc / Ctrl+Q / Quit button closes the app
#  – ENHANCED: Recursive folder search, optional *dislocations_*removed filter, organized output folders
# -------------------------------------------------------------------
import ast, sys, random, cv2
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtCore import QObject, QEvent
from PySide6.QtGui import QColor, QPen, QPixmap, QAction, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGraphicsScene, QGraphicsView,
    QLabel, QMainWindow, QMessageBox, QToolBar, QDialog,
    QSlider, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSizePolicy
)
from PySide6.QtWidgets import QCheckBox


# Adjustable parameters
TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN = 0, 20, 0, 0
MULTIPLIER   = 2.0
MIN_INSIDE   = 0.75
EXPORT_DIR   = Path("data/cleaned_YOLO_training_set/dataset_large_images_randomwindows")
CLASS_ID = {"dislocation":0,"faint":1,"multiple":2,"p3_pole":3,
            "p4_pole":4,"no_dislocation":5,"overlapping":6,"unknown":7}
RANDOM_VAR = 0.2  # ±20% variability in random windows size

@dataclass(frozen=True)
class Rect:
    xmin: float; ymin: float; xmax: float; ymax: float
    status: str; classification: str | None
    @property
    def width(self):  return self.xmax - self.xmin
    @property
    def height(self): return self.ymax - self.ymin
    @property
    def cx(self):     return (self.xmin + self.xmax) / 2
    @property
    def cy(self):     return (self.ymin + self.ymax) / 2

# CSV load
def _bbox(rec)->Tuple[float,float,float,float]:
    d = ast.literal_eval(rec) if isinstance(rec,str) else rec
    if len(d)==2 and all(isinstance(l,(list,tuple)) for l in d): xs,ys=d
    else: xs,ys=zip(*d)
    return min(xs),min(ys),max(xs),max(ys)

def read_rectangles(csv: Path)->List[Rect]:
    df=pd.read_csv(csv)
    return [Rect(*_bbox(r["rect"]),str(r["status"]),r.get("classification_label"))
            for _,r in df.iterrows()]

# Cluster windows
def build_cluster_windows(rects,w:int,h:int):
    usable=box(LEFT_MARGIN,TOP_MARGIN,w-RIGHT_MARGIN,h-BOTTOM_MARGIN)
    user=[r for r in rects if r.status=="User_Added" and
          r.xmin>=LEFT_MARGIN and r.ymin>=TOP_MARGIN and
          r.xmax<=w-RIGHT_MARGIN and r.ymax<=h-BOTTOM_MARGIN]
    if not user: return []
    avg=(sum(r.width for r in user)+sum(r.height for r in user))/(2*len(user))
    dmax=MULTIPLIER*avg
    wins=[box(r.cx-1.5*r.width,r.cy-1.5*r.height,r.cx+1.5*r.width,r.cy+1.5*r.height)
          for r in user]
    clusters,seen=[],set()
    for i in range(len(user)):
        if i in seen: continue
        grp,q=[i],[i]; seen.add(i)
        while q:
            cur=q.pop(); cx,cy=user[cur].cx,user[cur].cy
            for j,o in enumerate(user):
                if j not in seen and hypot(o.cx-cx,o.cy-cy)<=dmax:
                    seen.add(j); q.append(j); grp.append(j)
        clusters.append(unary_union([wins[k] for k in grp]).intersection(usable))
    return clusters

# Random windows
import random
import numpy as np
from shapely.geometry import box
from poisson_disc import Bridson_sampling   # pip install poisson-disc

def build_random_windows(w: int, h: int,
                         base_w: int, base_h: int,
                         variation: float = RANDOM_VAR,
                         density:   float = 1.0):
    """
    Same contract as before, but robust to tiny images:
    * If the nominal window is wider/taller than the image, we shrink it.
    * If the drawable area (after margins) is zero, we return [].
    """
    inner_w = max(0, w - LEFT_MARGIN  - RIGHT_MARGIN)
    inner_h = max(0, h - TOP_MARGIN   - BOTTOM_MARGIN)
    if inner_w == 0 or inner_h == 0:
        return []                      # image entirely eaten by margins

    # --- clamp the nominal size to the drawable area --------------------
    base_w = min(base_w, inner_w)
    base_h = min(base_h, inner_h)

    area_ratio       = (inner_w * inner_h) / (base_w * base_h)
    expected_windows = area_ratio * density
    total            = max(1, np.random.poisson(lam=expected_windows))

    r = max(1, min(base_w, base_h))    # Bridson minimum spacing
    try:
        centres = Bridson_sampling(inner_w, inner_h, r=r, k=30)
    except Exception:
        centres = []

    if len(centres) < total:           # optional top-up with uniform pts
        need   = total - len(centres)
        extra  = np.column_stack([np.random.uniform(0, inner_w, need),
                                  np.random.uniform(0, inner_h, need)])
        centres.extend(map(tuple, extra))
    centres = centres[:total]

    windows = []
    for cx, cy in centres:
        rw = int(base_w * random.uniform(1 - variation, 1 + variation))
        rh = int(base_h * random.uniform(1 - variation, 1 + variation))

        # final safety clamp ­– makes sure the actual rectangle fits
        rw = min(rw, inner_w)
        rh = min(rh, inner_h)

        x0 = int(cx - rw / 2) + LEFT_MARGIN
        y0 = int(cy - rh / 2) + TOP_MARGIN
        x0 = min(max(x0, LEFT_MARGIN),  w - RIGHT_MARGIN  - rw)
        y0 = min(max(y0, TOP_MARGIN),   h - BOTTOM_MARGIN - rh)

        windows.append(box(x0, y0, x0 + rw, y0 + rh))
    return windows


# Drawing pens
COL={"dislocation":"red","faint":"purple","multiple":"blue",
     "p3_pole":"magenta","p4_pole":"cyan","no_dislocation":"brown",
     "overlapping":"yellow","unknown":"white"}
def pen_for(r:Rect):
    c=QColor("orange") if r.status=="User_Added" else QColor(COL.get(str(r.classification).strip(),"black"))
    pen=QPen(c,2)
    if r.status=="Failed_fit": pen.setStyle(Qt.DotLine)
    return pen

# Export
def _next_index(folder:Path,stem:str):
    taken={int(p.stem.split("_")[-1][3:]) for p in folder.glob(f"{stem}_win*.tiff")}
    i=0
    while i in taken: i+=1
    return i

def export_windows(img:np.ndarray,img_path:Path,rects,windows,status:QLabel,mode:str="random"):
    """Export windows with organized folder structure based on mode (cluster/random)"""
    folder_suffix = "cluster_windows" if mode == "cluster" else "random_windows"
    base_dir = EXPORT_DIR / folder_suffix
    img_dir = base_dir/"images"/"train_val"
    lab_dir = base_dir/"labels"/"train_val"
    img_dir.mkdir(parents=True,exist_ok=True); lab_dir.mkdir(parents=True,exist_ok=True)
    base=f"{img_path.parent.stem}_{img_path.stem}"
    nxt=_next_index(img_dir,base)

    saved,total=0,len(windows)
    for w_box in windows:
        x0,y0,x1,y1=map(int,w_box.bounds)
        if x1<=x0 or y1<=y0: continue
        tile=img[y0:y1,x0:x1]
        th,tw=tile.shape
        lbl=[]
        for r in rects:
            inter=box(r.xmin,r.ymin,r.xmax,r.ymax).intersection(w_box)
            if not inter or inter.area/(r.width*r.height)<MIN_INSIDE: continue
            xmin,xmax=(r.xmin-x0)/tw,(r.xmax-x0)/tw
            ymin,ymax=(r.ymin-y0)/th,(r.ymax-y0)/th
            xc,yc,bw,bh=(xmin+xmax)/2,(ymin+ymax)/2,xmax-xmin,ymax-ymin
            cls=CLASS_ID.get(r.classification,0)
            lbl.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        if not lbl: continue
        name=f"{base}_win{nxt+saved}"
        cv2.imwrite(str(img_dir/f"{name}.tiff"),tile)
        (lab_dir/f"{name}.txt").write_text("\n".join(lbl))
        saved+=1
        msg=f"saved {saved}/{total} → {folder_suffix}/images/train_val/{name}.tiff"
        print(msg); status.setText(msg); QApplication.processEvents()

def export_full_image(img:np.ndarray,img_path:Path,rects, status:QLabel):
    """Export full image with organized folder structure"""
    base_dir = EXPORT_DIR / "full_images"
    img_dir = base_dir/"images"/"train_val"
    lab_dir = base_dir/"labels"/"train_val"
    img_dir.mkdir(parents=True,exist_ok=True); lab_dir.mkdir(parents=True,exist_ok=True)
    base=f"{img_path.parent.stem}_{img_path.stem}_full"
    cv2.imwrite(str(img_dir/f"{base}.tiff"), img)
    h, w = img.shape[:2]
    labels=[]
    for r in rects:
        if str(r.classification).strip() != "dislocation":
            continue
        xmin, xmax = r.xmin/w, r.xmax/w
        ymin, ymax = r.ymin/h, r.ymax/h
        xc, yc = (xmin+xmax)/2, (ymin+ymax)/2
        bw, bh = xmax-xmin, ymax-ymin
        labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    if labels:
        (lab_dir/f"{base}.txt").write_text("\n".join(labels))
    msg=f"saved full image + labels → full_images/images/train_val/{base}.tiff"
    print(msg); status.setText(msg); QApplication.processEvents()

# Custom view
class View(QGraphicsView):
    def __init__(self):
        super().__init__(); self.setRenderHints(self.renderHints()|QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
    def wheelEvent(self,e):
        f=1.25 if e.angleDelta().y()>0 else 0.8; self.scale(f,f)

# SizeSelector with preview count slider
class SizeSelector(QDialog):
    """
    Interactive dialog that lets the user pick:
      • window width & height
      • density multiplier
      • how many sample rectangles to preview

    Extras:
      • "Square (W = H)" checkbox – keeps width and height equal
      • live label ≈ expected number of windows with current settings
    """
    def __init__(self, parent, img_w: int, img_h: int, init_w: int = None, init_h: int = None, init_density: float = None):
        super().__init__(parent)
        self.setWindowTitle("Select Window Size, Density & Preview")
        self.resize(parent.width() * 0.8, parent.height() * 0.8)

        # store image size & defaults
        self.img_w, self.img_h = img_w, img_h
        # Use provided values or sensible defaults
        self.base_w = min(init_w if init_w else 100, img_w)
        self.base_h = min(init_h if init_h else 100, img_h)
        self.density = init_density if init_density else 3.0
        self.preview_count = 500

        # UI skeleton
        main = QVBoxLayout(self)

        # preview canvas ---------------------------------------------------
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setRenderHints(self.view.renderHints() | QPainter.Antialiasing)
        self.scene.addPixmap(parent.pixmap_item.pixmap())
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        main.addWidget(self.view)

        # reference rectangle (orange) ------------------------------------
        self.window_rect = self.scene.addRect(
            0, 0, self.base_w, self.base_h, QPen(QColor("orange"), 2)
        )

        # where we keep the preview rectangles
        self.random_items: list = []

        # sliders ----------------------------------------------------------
        self.w_slide = None
        self.h_slide = None
        self._build_sliders(main)

        # lock checkbox ----------------------------------------------------
        self.lock_cb = QCheckBox("Square (W = H)")
        self.lock_cb.setChecked(self.base_w == self.base_h)  # Check if initial values are equal
        main.addWidget(self.lock_cb, alignment=Qt.AlignRight)

        # windows‐estimate label ------------------------------------------
        self.win_lbl = QLabel()
        font = self.win_lbl.font(); font.setBold(True)
        self.win_lbl.setFont(font)
        main.addWidget(self.win_lbl, alignment=Qt.AlignCenter)

        # buttons ----------------------------------------------------------
        bh = QHBoxLayout()
        ok = QPushButton("OK"); ok.clicked.connect(self.accept)
        cn = QPushButton("Cancel"); cn.clicked.connect(self.reject)
        bh.addWidget(ok); bh.addWidget(cn)
        main.addLayout(bh)

        # first refresh
        self._refresh_all()

    # ------------------------------------------------------------------ UI helpers
    def _build_sliders(self, layout: QVBoxLayout):
        """Create four parameter rows and save width/height slider refs."""
        for label, attr, minv, maxv, step in [
            ("Width",   "base_w",         1, self.img_w,       1),
            ("Height",  "base_h",         1, self.img_h,       1),
            ("Density", "density",      0.1, 10.0,           0.1),
            ("Preview", "preview_count",  1, 500,              1),
        ]:
            row  = QWidget(); hbox = QHBoxLayout(row)
            txt  = QLabel(f"{label}:")
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
            hbox.addWidget(txt); hbox.addWidget(slid); hbox.addWidget(val)
            layout.addWidget(row)

            if attr == "base_w":
                self.w_slide = slid
            elif attr == "base_h":
                self.h_slide = slid

    # ------------------------------------------------------------------ slot
    def _on_slide(self, attr: str, value: int | float, lbl: QLabel):
        """Called whenever any slider moves."""
        if attr == "density":
            self.density = value / 10.0
            lbl.setText(f"{self.density:.1f}")
        else:
            # if the square lock is on, keep the two sliders equal
            if self.lock_cb.isChecked() and attr in ("base_w", "base_h"):
                partner = self.h_slide if attr == "base_w" else self.w_slide
                # block recursion
                partner.blockSignals(True)
                partner.setValue(value)
                partner.blockSignals(False)
                self.base_w = self.base_h = value
                # both numeric readouts
                lbl.setText(str(value))
                other_lbl = partner.parent().findChildren(QLabel)[-1]
                other_lbl.setText(str(value))
            else:
                setattr(self, attr, value)
                lbl.setText(str(value))

        self._refresh_all()

    # ------------------------------------------------------------------ helpers
    def _refresh_all(self):
        """Update rectangle position, window estimate & preview graphics."""
        # reposition reference rectangle (keep centred)
        self.window_rect.setRect(
            (self.img_w - self.base_w) / 2,
            (self.img_h - self.base_h) / 2,
            self.base_w,
            self.base_h,
        )

        # expected number of windows
        area_ratio = (self.img_w * self.img_h) / (self.base_w * self.base_h)
        est = int(area_ratio * self.density)
        self.win_lbl.setText(f"≈ {est} windows at export")

        # redraw preview rectangles
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

    def update_preview(self):
        for item in self.random_items: self.scene.removeItem(item)
        self.random_items.clear()
        sample=build_random_windows(self.img_w,self.img_h,self.base_w,self.base_h,
                                     variation=RANDOM_VAR,density=self.density)
        pen=QPen(QColor("cyan"),2,Qt.DashLine)
        for w in sample[:min(self.preview_count,len(sample))]:
            x0,y0,x1,y1=w.bounds
            rect=self.scene.addRect(x0,y0,x1-x0,y1-y0,pen)
            self.random_items.append(rect)

# Main window
class Window(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Cluster & Random Window Exporter")
        self.view=View(); self.setCentralWidget(self.view)
        self.status=QLabel("Choose image + CSV (Ctrl+O)"); self.statusBar().addPermanentWidget(self.status)
        
        # Remember last settings
        self.last_folder = Path.home()
        self.last_base_w = 100
        self.last_base_h = 100
        self.last_density = 3.0

        tb = QToolBar(); self.addToolBar(tb)

        load_one = QAction("Load Image && CSV", self)
        load_one.setShortcut(QKeySequence("Ctrl+O"))
        load_one.triggered.connect(self.open_files)
        tb.addAction(load_one)

        load_dir = QAction("Load Folder (Recursive)", self)
        load_dir.setShortcut(QKeySequence("Ctrl+Shift+O"))
        load_dir.triggered.connect(self.open_folder)
        tb.addAction(load_dir)

        quit_ = QAction("Quit", self); quit_.setShortcut(QKeySequence("Ctrl+Q"))
        quit_.triggered.connect(self.close); tb.addAction(quit_)
        
        self.resize(1280,800); QTimer.singleShot(0,self.open_files)
        
    def open_folder(self):
        """Open folder and recursively search for images with optional filtering"""
        folder = QFileDialog.getExistingDirectory(self, "Choose folder (will search recursively)", str(self.last_folder))
        if not folder:
            return
        folder = Path(folder)
        self.last_folder = folder

        # Ask user which images to process
        filter_dlg = QMessageBox(self)
        filter_dlg.setWindowTitle("Image Filter")
        filter_dlg.setText("Which images do you want to process?")
        filter_all = filter_dlg.addButton("All images", QMessageBox.AcceptRole)
        filter_rem = filter_dlg.addButton("Only *dislocations_*removed", QMessageBox.AcceptRole)
        filter_exc = filter_dlg.addButton("Exclude *dislocations_*removed", QMessageBox.AcceptRole)
        filter_dlg.exec()
        
        clicked = filter_dlg.clickedButton()
        filter_mode = "all" if clicked == filter_all else "only_removed" if clicked == filter_rem else "exclude_removed"

        # Ask for export mode ONCE at the start
        mode_dlg = QMessageBox(self)
        mode_dlg.setWindowTitle("Export Mode")
        mode_dlg.setText("Choose export mode for ALL images:")
        cbtn = mode_dlg.addButton("Cluster windows", QMessageBox.AcceptRole)
        rbtn = mode_dlg.addButton("Random windows", QMessageBox.AcceptRole)
        fbtn = mode_dlg.addButton("Full image labels", QMessageBox.AcceptRole)
        xbtn = mode_dlg.addButton("Cancel", QMessageBox.RejectRole)
        mode_dlg.exec()
        
        if mode_dlg.clickedButton() == xbtn:
            self.status.setText("Canceled")
            return
        
        export_mode = None
        if mode_dlg.clickedButton() == cbtn:
            export_mode = "cluster"
        elif mode_dlg.clickedButton() == rbtn:
            export_mode = "random"
        else:
            export_mode = "full"

        img_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        
        # Recursive search using rglob
        self.status.setText("Searching for images recursively...")
        QApplication.processEvents()
        
        all_images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in img_exts and p.is_file())
        
        # Apply filter based on mode
        if filter_mode == "all":
            images = all_images
        elif filter_mode == "only_removed":
            images = []
            for p in all_images:
                stem = p.stem
                if "dislocations_" in stem and stem.endswith("_removed"):
                    images.append(p)
        else:  # exclude_removed
            images = []
            for p in all_images:
                stem = p.stem
                # Exclude images that match *dislocations_*removed pattern
                if not ("dislocations_" in stem and stem.endswith("_removed")):
                    images.append(p)

        if not images:
            if filter_mode == "all":
                msg = "No supported images found."
            elif filter_mode == "only_removed":
                msg = "No images matching pattern *dislocations_*removed found."
            else:
                msg = "No images found (excluding *dislocations_*removed pattern)."
            QMessageBox.information(self, "No images", msg)
            return

        self.status.setText(f"Found {len(images)} images. Processing...")
        QApplication.processEvents()

        for idx, img_path in enumerate(images, 1):
            self.status.setText(f"Processing image {idx}/{len(images)}: {img_path.name}")
            QApplication.processEvents()
            
            # Extract base name for CSV matching
            stem = img_path.stem
            if "_dislocations_" in stem and stem.endswith("_removed"):
                # Extract base name before _dislocations_
                base_name = stem.split("_dislocations_")[0]
            else:
                base_name = stem
                
            # Look for matching CSV in the same folder
            pattern = base_name + "_data_*_dislocations.csv"
            matches = list(img_path.parent.glob(pattern))
            
            if matches:
                csv_path = matches[0]
            else:
                msg = QMessageBox(self)
                msg.setWindowTitle("CSV not found")
                msg.setText(f"No CSV matching pattern:\n{pattern}\n\nfor image: {img_path.name}")
                browse = msg.addButton("Browse…", QMessageBox.AcceptRole)
                skip = msg.addButton("Skip", QMessageBox.RejectRole)
                msg.exec()
                if msg.clickedButton() is skip:
                    continue
                csv_file, _ = QFileDialog.getOpenFileName(
                    self, f"Select CSV for {img_path.name}", str(img_path.parent), "CSV (*.csv)"
                )
                if not csv_file:
                    continue
                csv_path = Path(csv_file)

            self.process(img_path, csv_path, export_mode)

        self.status.setText(f"Completed processing {len(images)} images")

    def open_files(self):
        img_filters="Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"
        img_path,_=QFileDialog.getOpenFileName(self,"Choose image",str(self.last_folder),img_filters)
        if not img_path: return
        self.last_folder = Path(img_path).parent
        csv_path,_=QFileDialog.getOpenFileName(self,"Choose CSV",str(Path(img_path).parent),"CSV (*.csv)")
        if not csv_path: return
        self.process(Path(img_path),Path(csv_path), None)  # None means ask for mode

    def process(self,img_path,csv_path,export_mode=None):
        rects=read_rectangles(csv_path)
        pix=QPixmap(str(img_path));
        if pix.isNull(): QMessageBox.critical(self,"Error",f"Cannot load {img_path}"); return
        scene=QGraphicsScene(self); scene.addPixmap(pix)
        self.view.setScene(scene); self.pixmap_item=scene.items()[0]
        img_np=cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            QMessageBox.critical(self, "Error", f"Cannot read {img_path} with OpenCV")
            return
        
        # If export_mode not provided (single file mode), ask the user
        if export_mode is None:
            dlg=QMessageBox(self); dlg.setWindowTitle("Export Options")
            dlg.setText("Choose export mode:")
            cbtn=dlg.addButton("Cluster windows",QMessageBox.AcceptRole)
            rbtn=dlg.addButton("Random windows",QMessageBox.AcceptRole)
            fbtn=dlg.addButton("Full image labels",QMessageBox.AcceptRole)
            xbtn=dlg.addButton("Cancel",QMessageBox.RejectRole)
            dlg.exec()
            if dlg.clickedButton()==xbtn: self.status.setText("Canceled"); return
            
            if dlg.clickedButton()==fbtn:
                export_mode = "full"
            elif dlg.clickedButton()==cbtn:
                export_mode = "cluster"
            else:
                export_mode = "random"
        
        # Handle full image export
        if export_mode == "full":
            full_scene=QGraphicsScene(self); full_scene.addPixmap(pix)
            for r in rects: full_scene.addRect(QRectF(r.xmin,r.ymin,r.width,r.height), pen_for(r))
            self.view.setScene(full_scene); self.view.fitInView(full_scene.sceneRect(),Qt.KeepAspectRatio)
            self.status.setText("Exporting full image + labels…")
            QTimer.singleShot(0,lambda:export_full_image(img_np,img_path,rects,self.status))
            return
        
        # Handle cluster or random windows
        if export_mode == "cluster":
            windows=build_cluster_windows(rects,pix.width(),pix.height())
        else:  # random
            sel=SizeSelector(self,pix.width(),pix.height(),self.last_base_w,self.last_base_h,self.last_density)
            if sel.exec()!=QDialog.Accepted: self.status.setText("Canceled"); return
            # Save the slider values for next time
            self.last_base_w = sel.base_w
            self.last_base_h = sel.base_h
            self.last_density = sel.density
            windows=build_random_windows(pix.width(),pix.height(),sel.base_w,sel.base_h,
                                          variation=RANDOM_VAR,density=sel.density)
            
        out_scene=QGraphicsScene(self); out_scene.addPixmap(pix)
        for r in rects: out_scene.addRect(QRectF(r.xmin,r.ymin,r.width,r.height), pen_for(r))
        pen=QPen(QColor("orange"),3,Qt.DashLine)
        for w in windows: x0,y0,x1,y1=w.bounds; out_scene.addRect(QRectF(x0,y0,x1-x0,y1-y0),pen)
        self.view.setScene(out_scene); self.view.fitInView(out_scene.sceneRect(),Qt.KeepAspectRatio)
        self.status.setText(f"Exporting {len(windows)} windows…")
        QTimer.singleShot(0,lambda:export_windows(img_np,img_path,rects,windows,self.status,export_mode))

if __name__=="__main__":
    app=QApplication(sys.argv); w=Window(); w.show()
    class _EscFilter(QObject):
        def eventFilter(self,o,e):
            if e.type()==QEvent.KeyPress and e.key()==Qt.Key_Escape:
                w.close(); return True
            return False
    app.installEventFilter(_EscFilter()); sys.exit(app.exec())
