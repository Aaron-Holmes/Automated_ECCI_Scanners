import sys
import ast
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene,
    QVBoxLayout, QWidget, QPushButton, QGridLayout, QLabel, QHBoxLayout,
    QCheckBox, QFrame, QSlider, QGraphicsPolygonItem, QGraphicsItem, QMessageBox
)
from PySide6.QtGui import (
    QPixmap, QPen, QColor, QBrush, QImage, QPolygonF, QCursor, QPainter, QIcon
)
from PySide6.QtCore import Qt, QPointF, Signal, QRectF, QDir, QSize
import os
from PIL import Image
from shapely.geometry import Polygon
from skimage.restoration import denoise_wavelet, estimate_sigma
from scipy.ndimage import gaussian_filter
from fitting_function import model, smooth_and_add_noise
from PySide6.QtWidgets import QCheckBox
from PySide6.QtCore    import QEasingCurve, QPropertyAnimation, QPointF, Property
from PySide6.QtGui     import QColor, QPainter, QBrush, QPen

SVG_SWITCH_OFF = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='40' height='20' viewBox='0 0 40 20'>"
    "<rect width='40' height='20' rx='10' fill='%23777'/>"
    "<circle cx='10' cy='10' r='8' fill='%23ffffff'/>"
    "</svg>"
)

SVG_SWITCH_ON = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='40' height='20' viewBox='0 0 40 20'>"
    "<rect width='40' height='20' rx='10' fill='%234caf50'/>"
    "<circle cx='30' cy='10' r='8' fill='%23ffffff'/>"
    "</svg>"
)

ENABLE_OVERLAP_ON_OPEN = False

class ToggleSwitch(QCheckBox):
    """A flat on/off switch (40×20 px)."""
    def __init__(self, parent=None,
                 thumb_radius=8, track_radius=10,
                 off_color="#777", on_color="#4caf50",
                 animation_duration=120):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(track_radius*4, track_radius*2)
        self._off_color = QColor(off_color)
        self._on_color  = QColor(on_color)
        self._thumb_r   = thumb_radius
        self._track_r   = track_radius
        self._offset    = self._thumb_r + 2   # set before QPropertyAnimation reads it below
        self._x_anim    = QPropertyAnimation(self, b"offset", self)
        self._x_anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._x_anim.setDuration(animation_duration)

        # start/stop the animation when the checkbox is toggled
        self.toggled.connect(self._start_anim)

    # ------------------------------------------------------------------  Qt property
    def get_offset(self):         return self._offset
    def set_offset(self, val):    self._offset = val; self.update()
    offset = Property(float, get_offset, set_offset)     # noqa: E305

    # ------------------------------------------------------------------  paint
    def paintEvent(self, _event):
        """
        Draws the track, the two glyphs (🖍 and ■) and the thumb.
        The thumb is kept perfectly centred on both end-positions.
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # --- track -----------------------------------------------------------
        track_rect = self.rect().adjusted(0, 0, -1, -1)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._on_color if self.isChecked()
                        else self._off_color))
        p.drawRoundedRect(track_rect, self._track_r, self._track_r)

        # --- glyphs (behind the thumb) ---------------------------------------
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.NoBrush)
        p.drawText(track_rect.adjusted(4, 0, 0, -2),                 # left
                Qt.AlignVCenter, "🖍")
        p.drawText(track_rect.adjusted(-8, 0, -4, -2),               # right
                Qt.AlignRight | Qt.AlignVCenter, "■")

        # --- thumb -----------------------------------------------------------
        thumb_x = self._offset
        thumb_y = track_rect.center().y()
        p.setBrush(QBrush(Qt.white))
        p.setPen(QPen(Qt.black, .5))
        p.drawEllipse(QPointF(thumb_x, thumb_y),
                    self._thumb_r, self._thumb_r)


    # ------------------------------------------------------------------  animation
    def _start_anim(self, checked: bool):
        """
        Starts the little slide animation whenever the switch toggles.
        """
        left_pos  = self._thumb_r + 2
        right_pos = self.width() - self._thumb_r - 2

        self._x_anim.stop()
        self._x_anim.setStartValue(self._offset)          # current position
        self._x_anim.setEndValue(right_pos if checked     # target position
                                else left_pos)
        self._x_anim.start()




class SynchronizedGraphicsView(QGraphicsView):
    selection_changed_signal = Signal()

    def __init__(self, main_window, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = main_window
        self.index = index
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # highlight drawing state
        self._drawing = False
        self._last_pt = QPointF()
        self.rectangles_successful = []
        self.rectangles_failed = []
        self.rectangles_merged = []
        self.rectangles_user_added = []

        # Flags for visibility
        self.rectangles_successful_visible_flag = True
        self.rectangles_failed_visible_flag = True
        self.rectangles_merged_visible_flag = True
        self.rectangles_user_added_visible_flag = True

        # Flag to Track Image Type
        self.show_preprocessed_image = False

        # Connect selectionChanged signal to update highlighting and emit custom signal
        self.scene().selectionChanged.connect(self.on_selection_changed)

    def wheelEvent(self, event):
        mouse_position = self.mapToScene(event.position().toPoint())
        old_transform = self.transform()
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        delta = self.transform().map(mouse_position) - old_transform.map(mouse_position)
        self.translate(delta.x(), delta.y())
        self.main_window.sync_views(self)

    def mousePressEvent(self, event):
        mw = self.main_window
        if (mw.highlight_mode or mw.erase_mode) and event.button() == Qt.LeftButton:
            self._drawing  = True
            scene_pt       = self.mapToScene(event.position().toPoint())
            self._path_pts = [scene_pt]
            self._last_pt  = scene_pt
            return
        if self.main_window.add_mode:
            if event.button() == Qt.LeftButton:
                scene_pos = self.mapToScene(event.position().toPoint())
                x, y = scene_pos.x(), scene_pos.y()
                width = self.main_window.rectangle_width
                height = self.main_window.rectangle_height
                rect_coords = [
                    (x - width / 2, y - height / 2),
                    (x + width / 2, y - height / 2),
                    (x + width / 2, y + height / 2),
                    (x - width / 2, y + height / 2),
                    (x - width / 2, y - height / 2)  # Close the rectangle
                ]
                self.main_window.temp_rectangle = {
                    'center': (x, y),
                    'coords': rect_coords
                }
                self.main_window.update_temp_rectangle()
        elif self.main_window.delete_mode:
            if event.button() == Qt.LeftButton:
                scene_pos = self.mapToScene(event.position().toPoint())
                item = self.scene().itemAt(scene_pos, self.transform())
                if item and isinstance(item, QGraphicsPolygonItem):
                    self.main_window.delete_graphics_item(item)
        elif self.main_window.merge_mode:
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.position().toPoint())
                item = self.scene().itemAt(pos, self.transform())
                if item and isinstance(item, QGraphicsPolygonItem):
                    # Toggle selection state
                    item.setSelected(not item.isSelected())
            else:
                super().mousePressEvent(event)

        elif self.main_window.local_density_mode:
            if event.button() == Qt.LeftButton:
                if hasattr(self, 'local_density_rect_item') and self.local_density_rect_item:
                    self.scene().removeItem(self.local_density_rect_item)
                    self.local_density_rect_item = None

                scene_pos = self.mapToScene(event.position().toPoint())
                self.local_density_drag_start = (scene_pos.x(), scene_pos.y())
            else:
                super().mousePressEvent(event)


        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        mw = self.main_window
        if self._drawing:
            pt = self.mapToScene(event.position().toPoint())
            if (pt - self._last_pt).manhattanLength() < 2:
                return
            if mw.area_mode:                       # fill as you go
                self._path_pts.append(pt)
                self._draw_fill_path(self._path_pts)
            else:                                  # simple stroke
                self._draw_line(self._last_pt, pt)
            self._last_pt = pt
            return

        if self.main_window.local_density_mode and event.buttons() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            if hasattr(self, 'local_density_drag_start') and self.local_density_drag_start is not None:
                x0, y0 = self.local_density_drag_start
                x1, y1 = scene_pos.x(), scene_pos.y()

                # Remove old item
                if hasattr(self, 'local_density_rect_item') and self.local_density_rect_item:
                    self.scene().removeItem(self.local_density_rect_item)

                # Draw dashed rectangle
                pen = QPen(QColor("cyan"), 2, Qt.DashLine)
                brush = QBrush(Qt.NoBrush)
                coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
                self.local_density_rect_item = QGraphicsPolygonItem(QPolygonF([QPointF(px, py) for px, py in coords]))
                self.local_density_rect_item.setPen(pen)
                self.local_density_rect_item.setBrush(brush)
                self.scene().addItem(self.local_density_rect_item)
        else:
            super().mouseMoveEvent(event)
            self.main_window.sync_views(self)


    def mouseReleaseEvent(self, event):
        if self._drawing:
            self._drawing = False
            return

        if self.main_window.local_density_mode and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            if hasattr(self, 'local_density_drag_start') and self.local_density_drag_start is not None:
                x0, y0 = self.local_density_drag_start
                x1, y1 = scene_pos.x(), scene_pos.y()
                sel_min_x, sel_max_x = sorted([x0, x1])
                sel_min_y, sel_max_y = sorted([y0, y1])
                selection_area = abs((sel_max_x - sel_min_x) * (sel_max_y - sel_min_y))

                inside_count = 0
                # exclude “unknown” from the local tally
                for rect in self.main_window.rectangles:
                    if rect.get("classification_label") == "unknown":
                        continue  # skip
                    cx, cy = rect['polygon'].centroid.x, rect['polygon'].centroid.y
                    if sel_min_x <= cx <= sel_max_x and sel_min_y <= cy <= sel_max_y:
                        inside_count += 1

                if selection_area > 0:
                    local_density = (inside_count / selection_area) * 100
                else:
                    local_density = 0
                self.main_window.local_density_label.setText(f"Local Density: {local_density:.2f}")

                # reset drag info
                self.local_density_drag_start = None
        else:
            super().mouseReleaseEvent(event)

    def _painter(self, erase=False):
        mw = self.main_window
        if mw.highlight_image is None:
            return None
        p = QPainter(mw.highlight_image)
        if erase:
            p.setCompositionMode(QPainter.CompositionMode_Clear)
        return p

    def _draw_line(self, p0, p1):
        mw = self.main_window
        painter = self._painter(erase=mw.erase_mode)
        if painter is None:
            return
        pen = QPen(QColor(mw.highlight_colour), mw.highlight_size,
                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p0, p1)
        painter.end()
        mw.highlight_dirty = True
        mw._refresh_highlight_pixmaps()
        mw._mark_unsaved()

    def _draw_fill_path(self, pts):
        """
        Fills the polygon enclosed by *pts*, closing it automatically so the
        region is coloured in as you drag.
        """
        mw = self.main_window
        painter = self._painter(erase=mw.erase_mode)
        if painter is None or len(pts) < 3:
            return
        colour = QColor(mw.highlight_colour)
        colour.setAlpha(200)
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        painter.drawPolygon(QPolygonF(pts + [pts[0]]))   # close path
        painter.end()
        mw.highlight_dirty = True
        mw._refresh_highlight_pixmaps()
        mw._mark_unsaved()


    def _draw_highlight(self, p0: QPointF, p1: QPointF):
        mw = self.main_window
        if mw.highlight_image is None:
            return
        painter = QPainter(mw.highlight_image)
        if mw.erase_mode:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            colour = QColor(0, 0, 0, 0)
        else:
            colour = QColor(mw.highlight_colour)
            colour.setAlpha(200)
        pen = QPen(colour, mw.highlight_size, Qt.SolidLine,
                Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p0, p1)
        painter.end()
        mw._refresh_highlight_pixmaps()


    def keyPressEvent(self, event):
        # Forward the key press event to the main window
        self.main_window.keyPressEvent(event)

    def toggle_drawings(self, drawing_type):
        if drawing_type == "rectangles_successful":
            self.rectangles_successful_visible_flag = not self.rectangles_successful_visible_flag
        elif drawing_type == "rectangles_failed":
            self.rectangles_failed_visible_flag = not self.rectangles_failed_visible_flag
        elif drawing_type == "rectangles_merged":
            self.rectangles_merged_visible_flag = not self.rectangles_merged_visible_flag
        elif drawing_type == "rectangles_user_added":
            self.rectangles_user_added_visible_flag = not self.rectangles_user_added_visible_flag
        # Refresh the view
        self.main_window.update_view(self, self.index)

    def clear_drawings(self, storage_list):
        storage_list.clear()

    def on_selection_changed(self):
        if self.main_window.merge_mode:
            self.update_highlighting()
        # Emit the custom signal to notify the main window
        self.selection_changed_signal.emit()

    def update_highlighting(self):
        scene = self.scene()
        for item in scene.items():
            # Skip non-rect items
            if not isinstance(item, QGraphicsPolygonItem):
                continue
            if item.data(0) == "temp":
                continue  # skip the temp rectangle

            if item.isSelected():
                # e.g. highlight in yellow
                pen = item.pen()
                pen.setColor(QColor("yellow"))
                pen.setWidth(3)
                item.setPen(pen)

    def paintEvent(self, event):
        super().paintEvent(event)
        # Now draw the grid over the view
        if self.main_window.show_grid:
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(Qt.white, 2, Qt.SolidLine)
            painter.setPen(pen)
            rect = self.viewport().rect()
            width = rect.width()
            height = rect.height()
            # Draw vertical lines (creating 3 columns)
            for i in range(1, 3):
                x = width * i / 3
                painter.drawLine(QPointF(x, 0), QPointF(x, height))
            # Draw horizontal lines (creating 3 rows)
            for i in range(1, 3):
                y = height * i / 3
                painter.drawLine(QPointF(0, y), QPointF(width, y))
            painter.end()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.viewport().update()  # Trigger repaint to refresh the grid


class TIFFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF File Viewer with Synchronized Views")
        self.setGeometry(100, 100, 1400, 900)
        self.rectangles = []  # Store rectangles data
        self.overlapping_indices = set()
        self.delete_mode = False  # Flag to indicate if delete mode is active
        self.rectangle_id_counter = 0  # Unique identifier for rectangles
        self.deleted_rectangles = []  # Stack for storing deleted rectangles
        self.add_mode = False  # Flag for Add Dislocation mode
        self.temp_rectangle = None  # Store the temporary rectangle being added
        self.rectangle_width = 30  # Default width for new rectangles
        self.rectangle_height = 30  # Default height for new rectangles
        self.merge_mode = False  # Flag for Merge mode
        self.preprocessed_pixmap_view0 = None  # Preprocessed pixmap for View 1
        self.preprocessed_pixmap_view1 = None  # Preprocessed pixmap for View 2
        self.preprocessed_pixmap_view3 = None  # Preprocessed pixmap for View 4
        self.preprocessed_image_view3_np = None  # Preprocessed image for View 4 as NumPy array
        self.preprocessed_pixmap_view3_processed = None  # Processed image for View 4
        self.loaded_pixmap = None  # Original pixmap
        self.action_stack = []  # Stack to store actions for Undo functionality
        self.image_np = None  # Store the original image as a numpy array
        self.combined_image_pixmap = None  # Store the combined image pixmap
        self.show_grid = False  # Flag to show/hide grid overlay
        self.need_to_update_overlaps = False  # Flag to indicate if overlaps need to be recomputed
        self.continuous_update = False  # Flag to enable/disable continuous update
        self.need_recompute_images = False  # Flag to indicate if images need to be recomputed
        self.local_density_mode = False
        self.last_dir = QDir.homePath()
        self.pending_adds = []
        self.pending_deletes = []
        self.pending_merges = []
        self.unsaved_changes = False
        # highlight layer state
        self.highlight_image   = None      # shared transparent QImage
        self.highlight_pixmaps = {}        # view → QGraphicsPixmapItem
        self.highlight_mode    = False
        self.area_mode      = False 
        self.erase_mode        = False
        self.highlight_size    = 40        # brush diameter, px
        self.highlight_colour  = QColor("yellow")
        self.highlight_visible = True
        self.highlight_dirty    = False    # True once drawn on since last load/save
        self.initUI()

    # Same colors as the PNG export below (paper_classification_style) so the
    # live view and the saved image always agree - single source of truth.
    classification_color_map = {
        'p3_pole':        '#E69F00',  # orange
        'p4_pole':        '#56B4E9',  # sky blue
        'dislocation':    '#009E73',  # bluish green
        'multiple':       '#0072B2',  # blue
        'no_dislocation': '#D55E00',  # vermillion
        'overlapping':    '#CC79A7',  # reddish purple
        'faint':          '#F0E442',  # yellow
        'unknown':        '#FFFFFF',  # white
        # any other labels => fallback to 'black'
    }

    # Derived from classification_color_map above so the live view and the
    # saved PNG always use identical colors - no grey or black anywhere,
    # since those blend into the dark parts of a grayscale SEM image and
    # become invisible. All solid EXCEPT 'unknown', which stays white/dashed
    # (see save_image()).
    paper_classification_style = {label: {'color': color} for label, color in classification_color_map.items()}
    paper_merged_style = {'color': '#FF1493'}    # deep pink - manual correction / merged, deliberately distinct
    paper_fallback_style = {'color': '#FFFFFF'}  # unlabeled rectangles treated the same as 'unknown'

    def initUI(self):
        from PySide6.QtWidgets import (
            QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QCheckBox,
            QLabel, QPushButton, QSlider, QSpacerItem, QSizePolicy
        )
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen

        # overall layout: left panel | center panel | right panel, with a bottom panel spanning the full width
        main_hlayout = QHBoxLayout()         # Horizontal for left/center/right
        center_vlayout = QVBoxLayout()       # The center area will hold top bar + 2×2 grid + (some margin if you like)

        # 1) Left panel (for toggles of View 0 & 2)
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Box)
        left_panel.setStyleSheet("background-color: #f0f0f0; color: #000000; border: 1px solid #c0c0c0;")
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(5, 5, 5, 5)

        # 2) Center panel: a QFrame containing a vertical layout with the thin top bar + 2×2 grid
        center_frame = QFrame()
        center_frame_layout = QVBoxLayout(center_frame)
        center_frame_layout.setContentsMargins(0, 0, 0, 0)

        # 2a) Top bar (Load, Save, Settings)
        top_bar_frame = QFrame()
        top_bar_layout = QHBoxLayout(top_bar_frame)
        top_bar_layout.setContentsMargins(5, 5, 5, 5)

        load_btn = QPushButton("Load [Ctrl+L]")
        load_btn.setShortcut("Ctrl+L")
        load_btn.clicked.connect(self.open_files)
        top_bar_layout.addWidget(load_btn)

        self.save_data_button = QPushButton("Save [Ctrl+S]")
        self.save_data_button.setShortcut("Ctrl+S")
        self.save_data_button.clicked.connect(self.save_data)
        top_bar_layout.addWidget(self.save_data_button)

        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_image)
        top_bar_layout.addWidget(self.save_image_button)

        self.settings_button = QPushButton("Settings")
        top_bar_layout.addWidget(self.settings_button)

        # highlight tool bar
        def add_btn(text, slot, check=False):
            b = QPushButton(text)
            b.setCheckable(check)
            b.clicked.connect(slot)
            top_bar_layout.addWidget(b)
            return b

        self.switch_area = ToggleSwitch()
        self.switch_area.toggled.connect(self.toggle_area_mode)
        top_bar_layout.addWidget(self.switch_area)
        self.switch_area.setCursor(Qt.PointingHandCursor)
        self.switch_area.setToolTip("Stroke  ↔  Area-fill")
        self.switch_area.setChecked(False)                 # left = Stroke
        self.switch_area.setFixedSize(40, 20)

        # style: show our two tiny svgs
        self.switch_area.setStyleSheet(f"""
        QCheckBox::indicator {{
            width: 40px; height: 20px;
        }}
        QCheckBox::indicator:unchecked {{
            image: url("{SVG_SWITCH_OFF}");
        }}
        QCheckBox::indicator:checked {{
            image: url("{SVG_SWITCH_ON}");
        }}
        """)

        self.switch_area.toggled.connect(self.toggle_area_mode)
        top_bar_layout.addWidget(self.switch_area)

        self.btn_pen  = add_btn("🖍", self.toggle_pen,  True)
        self.btn_ers  = add_btn("🧼", self.toggle_erase, True)
        self.btn_eye  = add_btn("👁", self.toggle_highlight_vis, True); self.btn_eye.setChecked(True)
        self.btn_clr  = add_btn("🗑", self.clear_highlight)


        self.hl_size_slider = QSlider(Qt.Horizontal)
        self.hl_size_slider.setRange(20, 150)
        self.hl_size_slider.setValue(self.highlight_size)
        self.hl_size_slider.setFixedWidth(100)
        self.hl_size_slider.valueChanged.connect(
            lambda v: setattr(self, "highlight_size", v))
        top_bar_layout.addWidget(self.hl_size_slider)

        # quick colour swatches
        for n, c in [("Y", "yellow"), ("G", "lime"), ("C", "cyan"), ("M", "magenta")]:
            b = QPushButton(n)
            b.setFixedWidth(24)
            b.setStyleSheet(f"background:{c}; color: #000000;")
            b.clicked.connect(lambda _, col=c: self.set_highlight_colour(QColor(col)))
            top_bar_layout.addWidget(b)


        top_bar_layout.addStretch(1)  # push them left
        center_frame_layout.addWidget(top_bar_frame, 0)

        # 2b) The 2×2 image grid
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_container = QFrame()
        grid_container.setLayout(grid_layout)
        center_frame_layout.addWidget(grid_container, 1)

        # 3) Right panel (for toggles of View 1 & 3)
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Box)
        right_panel.setStyleSheet("background-color: #f0f0f0; color: #000000; border: 1px solid #c0c0c0;")
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(5, 5, 5, 5)

        # Add [left_panel | center_frame | right_panel] to main_hlayout
        main_hlayout.addWidget(left_panel, 0)
        main_hlayout.addWidget(center_frame, 1)
        main_hlayout.addWidget(right_panel, 0)

        # 4) Bottom panel across full width
        bottom_panel_frame = QFrame()
        bottom_panel_layout = QHBoxLayout(bottom_panel_frame)
        bottom_panel_layout.setContentsMargins(5, 5, 5, 5)

        left_bottom_container = QVBoxLayout()
        right_bottom_container = QVBoxLayout()
        bottom_panel_layout.addLayout(left_bottom_container, 2)
        bottom_panel_layout.addLayout(right_bottom_container, 1)

        # Combine everything in a vertical layout
        center_vlayout.addLayout(main_hlayout, 1)         # top row with 2x2 images
        center_vlayout.addWidget(bottom_panel_frame, 0)   # bottom row (global controls)
        container = QWidget()
        container.setLayout(center_vlayout)
        self.setCentralWidget(container)

        # four synchronized views in a 2x2 grid; visibility toggles for V0/V2 go in
        # the left panel and V1/V3 in the right panel
        default_visibility = [
            {'rectangles_successful': False, 'rectangles_failed': False, 'rectangles_user_added': False},  # V0
            {'rectangles_successful': True,  'rectangles_failed': True,  'rectangles_user_added': True},   # V1
            {'rectangles_successful': False, 'rectangles_failed': True,  'rectangles_user_added': True},   # V2
            {'rectangles_successful': False, 'rectangles_failed': True,  'rectangles_user_added': True},   # V3
        ]
        self.views = []
        self.image_toggle_checkboxes = []

        # Create sub-layouts for toggles in the left panel

        left_panel_layout.addStretch(2)
        toggles_v0_layout = QVBoxLayout()
        left_panel_layout.addLayout(toggles_v0_layout)
        left_panel_layout.addStretch(2)
        toggles_v2_layout = QVBoxLayout()
        left_panel_layout.addLayout(toggles_v2_layout)
        left_panel_layout.addStretch(1)

        # Similarly for right panel
        right_panel_layout.addStretch(2)
        toggles_v1_layout = QVBoxLayout()
        right_panel_layout.addLayout(toggles_v1_layout)
        right_panel_layout.addStretch(2)
        toggles_v3_layout = QVBoxLayout()
        right_panel_layout.addLayout(toggles_v3_layout)
        right_panel_layout.addStretch(1)

        # Now build the 4 views into grid_layout:
        for i in range(4):
            view = SynchronizedGraphicsView(self, i)
            self.views.append(view)

            row = i // 2
            col = i % 2
            grid_layout.addWidget(view, row, col)

            # Set default visibility
            vis = default_visibility[i]
            view.rectangles_successful_visible_flag = vis['rectangles_successful']
            view.rectangles_failed_visible_flag = vis['rectangles_failed']
            view.rectangles_user_added_visible_flag = vis['rectangles_user_added']

            # Build toggles
            toggle_box = QVBoxLayout()
            cb_suc = QCheckBox(f"View {i+1}: Successful")
            cb_suc.setChecked(vis['rectangles_successful'])
            cb_suc.stateChanged.connect(self.create_toggle_callback(view, "rectangles_successful"))
            toggle_box.addWidget(cb_suc)

            cb_fail = QCheckBox(f"View {i+1}: Failed")
            cb_fail.setChecked(vis['rectangles_failed'])
            cb_fail.stateChanged.connect(self.create_toggle_callback(view, "rectangles_failed"))
            toggle_box.addWidget(cb_fail)

            cb_user = QCheckBox(f"View {i+1}: User Added")
            cb_user.setChecked(vis['rectangles_user_added'])
            cb_user.stateChanged.connect(self.create_toggle_callback(view, "rectangles_user_added"))
            toggle_box.addWidget(cb_user)

            # Preproc image toggles for top-left or top-right only
            if i in [0, 1]:
                cb_img = QCheckBox(f"View {i+1}: Preproc Img")
                cb_img.setChecked(True)
                view.show_preprocessed_image = True
                cb_img.toggled.connect(self.create_image_toggle_callback(view))
                toggle_box.addWidget(cb_img)
                self.image_toggle_checkboxes.append(cb_img)

            # Place toggle_box in the correct sub-layout
            if i == 0:
                toggles_v0_layout.addLayout(toggle_box)
            elif i == 1:
                toggles_v1_layout.addLayout(toggle_box)
            elif i == 2:
                toggles_v2_layout.addLayout(toggle_box)
            else:
                toggles_v3_layout.addLayout(toggle_box)

        # bottom panel: global controls on the left, classification toggles on the right
        # Row 1
        row1 = QHBoxLayout()
        self.continuous_update_checkbox = QCheckBox("Continuously Update [U]")
        self.continuous_update_checkbox.setChecked(False)
        self.continuous_update_checkbox.setShortcut('U')
        self.continuous_update_checkbox.stateChanged.connect(self.toggle_continuous_update)
        row1.addWidget(self.continuous_update_checkbox)

        self.refresh_button = QPushButton("Refresh [R]")
        self.refresh_button.setShortcut('R')
        self.refresh_button.clicked.connect(self.handle_refresh_button)
        row1.addWidget(self.refresh_button)

        overlap_label = QLabel("Overlap %:")
        row1.addWidget(overlap_label)

        self.overlap_slider = QSlider(Qt.Horizontal)
        self.overlap_slider.setMinimum(0)
        self.overlap_slider.setMaximum(100)
        self.overlap_slider.setValue(0)
        self.overlap_slider.setTickPosition(QSlider.TicksBelow)
        self.overlap_slider.setTickInterval(10)
        self.overlap_slider.valueChanged.connect(self.update_overlap_threshold)
        row1.addWidget(self.overlap_slider)

        overlap_val_label = QLabel("0%")
        self.overlap_slider.valueChanged.connect(lambda v: overlap_val_label.setText(f"{v}%"))
        row1.addWidget(overlap_val_label)

        self.show_overlapping_checkbox = QCheckBox("Show Only Overlapping [O]")
        self.show_overlapping_checkbox.stateChanged.connect(self.update_views)
        row1.addWidget(self.show_overlapping_checkbox)

        self.grid_toggle_checkbox = QCheckBox("Grid [G]")
        self.grid_toggle_checkbox.setChecked(self.show_grid)
        self.grid_toggle_checkbox.toggled.connect(self.toggle_grid)
        row1.addWidget(self.grid_toggle_checkbox)

        left_bottom_container.addLayout(row1)

        # Row 2
        row2 = QHBoxLayout()
        self.merge_button = QPushButton("Merge [M]")
        self.merge_button.setCheckable(True)
        self.merge_button.setShortcut('M')
        self.merge_button.clicked.connect(self.toggle_merge_mode)
        row2.addWidget(self.merge_button)

        self.delete_button = QPushButton("Delete [D]")
        self.delete_button.setCheckable(True)
        self.delete_button.setShortcut('D')
        self.delete_button.clicked.connect(self.toggle_delete_mode)
        row2.addWidget(self.delete_button)

        undo_btn = QPushButton("Undo [Ctrl+Z]")
        undo_btn.setShortcut('Ctrl+Z')
        undo_btn.clicked.connect(self.undo_last_action)
        row2.addWidget(undo_btn)

        left_bottom_container.addLayout(row2)

        # Row 3
        row3 = QHBoxLayout()
        self.add_button = QPushButton("Add Dislocation [A]")
        self.add_button.setCheckable(True)
        self.add_button.setShortcut('A')
        self.add_button.clicked.connect(self.toggle_add_mode)
        row3.addWidget(self.add_button)

        w_lbl = QLabel("Width [Arrows]:")
        row3.addWidget(w_lbl)
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setMinimum(5)
        self.width_slider.setMaximum(100)
        self.width_slider.setValue(30)
        self.width_slider.valueChanged.connect(self.update_rectangle_width)
        row3.addWidget(self.width_slider)

        h_lbl = QLabel("Height [Arrows]:")
        row3.addWidget(h_lbl)
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setMinimum(5)
        self.height_slider.setMaximum(100)
        self.height_slider.setValue(30)
        self.height_slider.valueChanged.connect(self.update_rectangle_height)
        row3.addWidget(self.height_slider)

        left_bottom_container.addLayout(row3)

        # Classification toggles on the right side
        self.classification_toggles = {}
        class_grid = QGridLayout()
        row_idx = 0
        col_idx = 0
        for label_name, color_str in self.classification_color_map.items():
            color_label = QLabel()
            color_label.setFixedSize(15, 15)
            color_label.setStyleSheet(f"background-color: {color_str}; border: 1px solid black;")

            cb = QCheckBox(label_name)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_views)
            self.classification_toggles[label_name] = cb

            class_grid.addWidget(color_label, row_idx, col_idx)
            class_grid.addWidget(cb,          row_idx, col_idx + 1)

            if col_idx == 0:
                col_idx = 2
            else:
                col_idx = 0
                row_idx += 1

        right_bottom_container.addLayout(class_grid)

        # floating counters/density readout top-left, instructions top-right
        self.counters_label = QLabel("", self)
        self.counters_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 200);
                color: #000000;
                border: 1px solid gray;
                padding: 4px;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        self.counters_label.setWordWrap(True)
        self.counters_label.setVisible(False)

        self.counters_frame = QFrame(self)
        self.counters_frame.setStyleSheet(self.counters_label.styleSheet())
        self.counters_frame.setGeometry(200, 200, 250, 300)
        self.counters_frame.setVisible(False)

        # The layout inside that frame
        counters_frame_layout = QVBoxLayout(self.counters_frame)
        counters_frame_layout.setContentsMargins(2, 2, 2, 2)

        # Add the main counters label (which shows total dislocations, successful, etc.)
        counters_frame_layout.addWidget(self.counters_label)

        # Add a small line or spacer
        spacer = QFrame()
        spacer.setFrameShape(QFrame.HLine)
        spacer.setFrameShadow(QFrame.Sunken)
        counters_frame_layout.addWidget(spacer)

        # Add local density sublayout
        density_layout = QHBoxLayout()
        pix = QPixmap(16, 16)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setPen(QPen(Qt.black, 2, Qt.DashLine))
        painter.drawRect(2, 2, 12, 12)
        painter.end()

        self.local_density_button = QPushButton()
        self.local_density_button.setCheckable(True)
        self.local_density_button.setFixedSize(28, 28)
        self.local_density_button.setIcon(QIcon(pix))
        self.local_density_button.setIconSize(pix.size())
        self.local_density_button.setToolTip("Draw a rectangle to compute local density")
        self.local_density_button.clicked.connect(self.toggle_local_density_mode)
        density_layout.addWidget(self.local_density_button)

        self.local_density_label = QLabel("Local Dens: -")
        density_layout.addWidget(self.local_density_label)

        counters_frame_layout.addLayout(density_layout)

        # Instructions top-right
        self.instruction_label = QLabel("", self)
        self.instruction_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 200);
                color: #000000;
                border: 1px solid gray;
                padding: 4px;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setVisible(True)
        # We can keep it around 200 wide:
        self.instruction_label.setGeometry(self.width() - 210, 10, 200, 100)

        # Connect the selection_changed_signal
        for v in self.views:
            v.selection_changed_signal.connect(self.update_instructions)

        # Initialize instructions
        self.update_instructions()

    # highlight layer helpers
    def set_highlight_colour(self, col: QColor):
        self.highlight_colour = col

    def _activate_tool(self, pen=False, erase=False):
        self.highlight_mode = pen
        self.erase_mode     = erase
        self.btn_pen.setChecked(pen)
        self.btn_ers.setChecked(erase)
        if pen or erase:
            QApplication.setOverrideCursor(Qt.CrossCursor)
        else:
            self._clear_override_cursor()

    def toggle_pen(self):
        self._activate_tool(pen=self.btn_pen.isChecked())

    def toggle_erase(self):
        self._activate_tool(erase=self.btn_ers.isChecked())

    def toggle_area_mode(self, checked=False):
        self.area_mode = checked if isinstance(checked, bool) else self.switch_area.isChecked()




    def _init_highlight_layer(self, w, h):
        """(re)build one shared transparent QImage and a per-view pixmap item"""
        # create / resize the backing image
        if self.highlight_image is None or self.highlight_image.size() != QSize(w, h):
            self.highlight_image = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            self.highlight_image.fill(Qt.transparent)

        # recreate a pixmap item in every scene
        for v in self.views:
            # remove the old overlay if it still exists
            old = self.highlight_pixmaps.get(v)
            if old and old.scene():       # guard against “already deleted”
                v.scene().removeItem(old)

            pm = v.scene().addPixmap(QPixmap.fromImage(self.highlight_image))
            pm.setPos(0, 0)            # important – anchor at the scene origin
            pm.setOpacity(0.45)        # see-through
            pm.setZValue(-0.5)         # between base image (-1) and rectangles (0)
            pm.setAcceptedMouseButtons(Qt.NoButton)
            pm.setFlag(QGraphicsItem.ItemIsSelectable, False)
            pm.setFlag(QGraphicsItem.ItemIsMovable,   False)
            pm.setVisible(self.highlight_visible)
            self.highlight_pixmaps[v] = pm

    def _refresh_highlight_pixmaps(self):
        qpm = QPixmap.fromImage(self.highlight_image)
        for pm in self.highlight_pixmaps.values():
            pm.setPixmap(qpm)

    def _clear_override_cursor(self):
        # pop everything that is left on the QApplication stack
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        # make sure every view shows its own drag-cursor again
        for v in self.views:
            v.viewport().unsetCursor()

    def toggle_highlight_vis(self):
        self.highlight_visible = self.btn_eye.isChecked()
        for pm in self.highlight_pixmaps.values():
            pm.setVisible(self.highlight_visible)

    def clear_highlight(self):
        if self.highlight_image:
            self.highlight_image.fill(Qt.transparent)
            self.highlight_dirty = True
            self._refresh_highlight_pixmaps()
            self._mark_unsaved()

    # highlight persistence (survives close/reopen)
    def _highlight_path(self):
        """PNG path used to persist the highlighter layer for the current TIFF."""
        if not getattr(self, 'tiff_path', None):
            return None
        base_dir = os.path.dirname(self.tiff_path)
        tiff_basename = os.path.splitext(os.path.basename(self.tiff_path))[0]
        return os.path.join(base_dir, f"{tiff_basename}_highlights.png")

    def _save_highlight_layer(self):
        # Skip writing a file for images that were never actually drawn on.
        if not self.highlight_dirty:
            return
        path = self._highlight_path()
        if not path or self.highlight_image is None:
            return
        if not self.highlight_image.save(path, "PNG"):
            print(f"[Save Highlights] Failed to save highlight layer to: {path}")
        else:
            print(f"[Save Highlights] Saved highlight layer to: {path}")
            self.highlight_dirty = False

    def _load_highlight_layer(self):
        """Restore the highlighter layer saved for the current TIFF, or start blank."""
        path = self._highlight_path()
        loaded = QImage(path) if path and os.path.exists(path) else None
        if loaded is not None and not loaded.isNull():
            self.highlight_image = loaded.convertToFormat(QImage.Format_ARGB32_Premultiplied)
            print(f"[Load Highlights] Restored highlight layer from: {path}")
        else:
            self.highlight_image = None
        self.highlight_dirty = False


    def resizeEvent(self, event):
        super().resizeEvent(event)

        # For example, narrower but taller: x=10, y=10, width=160, height=180
        self.counters_frame.setGeometry(10, 10, 160, 180)

        # Keep instructions top-right:
        margin = 10
        w_ = 200
        h_ = 100
        x_ = self.width() - w_ - margin
        y_ = margin
        self.instruction_label.setGeometry(x_, y_, w_, h_)

    def _mark_unsaved(self):
        self.unsaved_changes = True

    def _maybe_save_changes(self):
        """
        If there are unsaved edits, ask the user whether to save
        before continuing.  Returns True if the caller can proceed
        (Yes‑saved, No‑discard, or nothing to save); returns False
        if the user hit Cancel.
        """
        if not self.unsaved_changes:
            return True    # nothing to ask

        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setIcon(QMessageBox.Question)
        box.setText("Save changes before continuing?")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Yes)
        choice = box.exec()

        if choice == QMessageBox.Yes:
            self.save_data()
            return True
        elif choice == QMessageBox.No:
            return True
        else:  # Cancel
            return False


    def save_data(self):
        """
        Overwrite the original CSV with new data, then rename it to
        <tiff_basename>_data_<X>_dislocations.csv,
        where <X> is the total # of dislocations.
        """
        import os
        import pandas as pd
        import numpy as np
        from PySide6.QtWidgets import QMessageBox

        if not hasattr(self, 'csv_path') or not self.csv_path:
            print("[Save Data] No original CSV path found; not saving.")
            return

        if not hasattr(self, 'tiff_path') or not self.tiff_path:
            print("[Save Data] No TIFF path found; using CSV base name instead.")

        base_dir = os.path.dirname(self.csv_path)
        # Use TIFF basename if available, else the CSV
        tiff_basename = os.path.splitext(os.path.basename(self.tiff_path))[0] if hasattr(self, 'tiff_path') else os.path.splitext(os.path.basename(self.csv_path))[0]
        # 'unknown' boxes aren't confirmed dislocations, so they shouldn't
        # count toward the total - matches the on-screen counters.
        total_dislocations = sum(
            1 for r in self.rectangles if r.get('classification_label') != 'unknown'
        )

        new_csv_name = f"{tiff_basename}_data_{total_dislocations}_dislocations.csv"
        new_csv_path = os.path.join(base_dir, new_csv_name)

        rows = []
        for rect in self.rectangles:
            # Convert 'User_Added' -> 'Failed_fit' if needed
            status = rect['status']

            
            coords_str = str(rect.get('coords', []))
            patch_val = rect.get('patch_locations', None)
            patch_str = str(patch_val) if patch_val else 'None'
            popt_val = rect.get('popt', None)
            if popt_val is not None:
                popt_val = np.array(popt_val, dtype=float)
                popt_str = np.array2string(
                    popt_val,
                    separator=' ',
                    precision=8,
                    floatmode='fixed',
                    suppress_small=False,
                    max_line_width=9999999
                )
            else:
                popt_str = 'None'
            pixel_size = rect.get('pixel_size', None) or 'None'

            row_dict = {
                'rect': coords_str,
                'classification_label': rect.get('classification_label', 'None'),  # <--- NEW
                'status': status,
                'patch_location': patch_str,
                'popt': popt_str,
                'pixel_size': pixel_size
            }
            rows.append(row_dict)

        df = pd.DataFrame(rows)

        # 1) Overwrite the old CSV in-place
        df.to_csv(self.csv_path, index=False)

        # 2) Rename it to the new filename
        try:
            os.rename(self.csv_path, new_csv_path)
            print(f"[Save Data] Renamed {self.csv_path} -> {new_csv_path}")
        except Exception as e:
            print(f"[Save Data] Rename failed: {e}")

        # Persist the highlighter layer alongside the data so it survives
        # closing and reopening the app.
        self._save_highlight_layer()

        self.unsaved_changes = False

        # 3) Show success message
        QMessageBox.information(
            self,
            "Save Successful",
            f"Successfully saved data to:\n{new_csv_path}"
        )

    def save_image(self):
        """
        Exports the loaded image with every rectangle overlaid as a
        standalone PNG for a paper, at the image's native pixel resolution
        (no resampling) so crops/zooms afterward stay sharp. All lines are
        solid and colorblind-safe (Okabe-Ito palette), except 'unknown'
        which stays white/dashed to match the live view. Fit status
        (Failed_fit) is shown via alpha instead of a dashed/dotted style, so
        it doesn't make lines look "broken up".
        """
        import matplotlib.pyplot as plt
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        if getattr(self, 'image_np', None) is None:
            QMessageBox.warning(self, "No image loaded", "Load a TIFF before saving an image.")
            return
        if not self.rectangles:
            QMessageBox.warning(self, "No rectangles", "There are no rectangles to draw.")
            return

        tiff_basename = (os.path.splitext(os.path.basename(self.tiff_path))[0]
                          if getattr(self, 'tiff_path', None) else 'image')
        default_dir = os.path.dirname(self.tiff_path) if getattr(self, 'tiff_path', None) else os.path.expanduser("~")
        default_path = os.path.join(default_dir, f"{tiff_basename}_annotated.png")

        path, _ = QFileDialog.getSaveFileName(self, "Save Annotated Image", default_path, "PNG Image (*.png)")
        if not path:
            return

        height, width = self.image_np.shape[:2]
        dpi = 300
        # figsize picked so width/height in inches * dpi == the image's actual
        # pixel dimensions exactly - the base image is drawn 1:1, never up-
        # or down-sampled, regardless of what dpi value is chosen.
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # fills the whole figure, no margins
        ax.imshow(self.image_np, cmap='gray', interpolation='none')
        ax.set_axis_off()

        # matplotlib linewidth is in points (1/72 inch), not pixels - convert
        # so every box's outline is exactly 2 pixels wide in the saved raster,
        # regardless of dpi.
        line_width_px = 2.0
        line_width = line_width_px * 72.0 / dpi
        # 'unknown' (and unlabeled/'other') stay white & dashed, matching the
        # live view's convention.
        dashed_keys = {'unknown', 'other'}

        used_styles = {}
        for rect in self.rectangles:
            is_merged = rect.get('merged') or rect.get('status') == 'User_Added'
            if is_merged:
                style = self.paper_merged_style
                key = 'merged / user-added'
            else:
                label = rect.get('classification_label')
                style = self.paper_classification_style.get(label, self.paper_fallback_style)
                key = label if label in self.paper_classification_style else 'other'

            linestyle = 'dashed' if key in dashed_keys else 'solid'
            # Fading already-faint white dashed lines further would make them
            # disappear, so only classified (solid, colored) lines fade.
            alpha = 1.0 if linestyle == 'dashed' else (0.4 if rect.get('status') == 'Failed_fit' else 1.0)

            coords = rect['coords']
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color=style['color'], linestyle=linestyle, linewidth=line_width, alpha=alpha)
            used_styles[key] = (style['color'], linestyle)

        legend_elements = [
            plt.Line2D([0], [0], color=color, linestyle=ls, linewidth=2, label=key)
            for key, (color, ls) in used_styles.items()
        ]
        # Illustrate the alpha-fade convention using an already-used class
        # color rather than introducing a grey/black swatch just for this.
        if used_styles:
            demo_color = next(iter(used_styles.values()))[0]
            legend_elements.append(
                plt.Line2D([0], [0], color=demo_color, linewidth=2, alpha=0.4, label='faded = failed fit'))
        # Legend drawn outside the image axes - fig.savefig's bbox_inches='tight'
        # below expands the saved canvas to fit it without touching the image's
        # own pixel scale.
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0, 0.3, 1),
                  fontsize=max(8, int(width / 250)))

        fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

        QMessageBox.information(self, "Saved", f"Saved annotated image to:\n{path}")

    def update_counters(self):
        if not self.rectangles:
            self.counters_label.setVisible(False)
            self.counters_frame.setVisible(False)
            return


        # — 1.   Work only with non‑merged, non‑unknown rectangles
        valid_rects = [
            r for r in self.rectangles
            if not r["merged"]
            and r.get("classification_label") != "unknown"
        ]

        total       = len(valid_rects)
        successful  = sum(r["status"] == "Successful_fit" for r in valid_rects)
        failed      = sum(r["status"] == "Failed_fit"    for r in valid_rects)
        user_added  = sum(r["status"] == "User_Added"    for r in valid_rects)

        # — 2.   Compose the text
        counters_text = (
            f"Total Dislocations: {total}\n"
            f"Successful Fits: {successful}\n"
            f"Failed Fits: {failed}\n"
            f"User Added: {user_added}\n"
        )

        # — 3.   Global density (same rule: unknowns excluded)
        if self.image_np is not None:
            h, w = self.image_np.shape[:2]
            area = h * w
            density = (total / area) * 100 if area else 0
            counters_text += f"Global Density/100 px²: {density:.2f}\n"

        # — 4.   Show it
        self.counters_label.setText(counters_text)
        self.counters_label.setVisible(True)
        self.counters_frame.setVisible(True)



    def toggle_grid(self, checked):
        self.show_grid = checked
        for view in self.views:
            view.viewport().update()  # Trigger repaint
        self.update_instructions()

    def update_instructions(self):
        instructions = ""
        # Check if files are loaded
        if self.loaded_pixmap is None:
            instructions = "Please open a TIFF and a CSV file to begin."
        else:
            # Check current mode
            if self.add_mode:
                instructions = (
                    "Add Mode:\n"
                    "Click on an area to add a rectangle,\n"
                    "use arrow keys to change rectangle size,\n"
                    "press [Space] or [Enter] to confirm.\n"
                    "Press [A] again or [Esc] to exit Add Mode."
                )
            elif self.merge_mode:
                instructions = (
                    "Merge Mode:\n"
                    "Select rectangles to merge by clicking on them,\n"
                    "press [Space] or [Enter] to confirm the merge.\n"
                    "Press [M] again or [Esc] to exit Merge Mode."
                )
            elif self.delete_mode:
                instructions = (
                    "Delete Mode:\n"
                    "Click on a rectangle to delete it. \nPress [D] again or [Esc] to exit Delete Mode."
                )
            else:
                # Default instructions
                instructions = "Click and drag to move the view,\nscroll wheel to zoom. If actions take too long, try disabling 'Continuous Updating'."
                # Check if any rectangle is selected across all views
                any_selected = False
                for view in self.views:
                    for item in view.scene().selectedItems():
                        if isinstance(item, QGraphicsPolygonItem):
                            any_selected = True
                            break
                    if any_selected:
                        break
                if any_selected:
                    instructions += "\nUse the Delete key to remove the selected rectangle."

        # Update the instruction label
        self.instruction_label.setText(instructions)

        # Show or hide based on whether there is instruction to show
        if instructions:
            self.instruction_label.setVisible(True)
        else:
            self.instruction_label.setVisible(False)

    def create_toggle_callback(self, view, drawing_type):
        def toggle_callback(state):
            view.toggle_drawings(drawing_type)
            print(f"Toggle '{drawing_type}' set to {'Visible' if state == Qt.Checked else 'Hidden'} for View {view.index + 1}")
            self.update_instructions()  # Update instructions when visibility changes
        return toggle_callback

    def create_image_toggle_callback(self, view):
        def image_toggle_callback(checked):
            view.show_preprocessed_image = checked
            print(f"Checkbox toggled: show_preprocessed = {checked}")  # Debug statement
            # Update the view
            self.update_view(view, view.index)
            self.update_instructions()  # Update instructions when image toggle changes
        return image_toggle_callback



    def open_files(self):
        if not self._maybe_save_changes():
            return

        tiff_path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF File", self.last_dir, "TIFF Files (*.tiff *.tif)"
        )
        if tiff_path:
            self.last_dir = os.path.dirname(tiff_path)

        csv_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", self.last_dir, "CSV Files (*.csv)"
        )
        if csv_path:
            self.last_dir = os.path.dirname(csv_path)

        if tiff_path and csv_path:
            self.load_image_and_rectangles(tiff_path, csv_path)
            self.unsaved_changes = False        # clean slate
        else:
            QMessageBox.warning(
                self, "File Selection Incomplete",
                "Please select both TIFF and CSV files."
            )
            self.update_instructions()

    def closeEvent(self, event):
        """
        Qt calls this when the main window is about to close.
        Offer to save if there are unsaved changes.
        """
        if self._maybe_save_changes():
            event.accept()    # quit
        else:
            event.ignore()    # stay open


    def load_image_and_rectangles(self, tiff_path, csv_path):
        self.csv_path = csv_path  # store the original CSV
        self.tiff_path = tiff_path
        try:
            # Load original TIFF image using PIL
            image = Image.open(tiff_path).convert('L')
            image_np = np.array(image)
            self.image_np = image_np  # Store the image numpy array
            q_image = QImage(
                image_np.data, image_np.shape[1], image_np.shape[0],
                image_np.strides[0], QImage.Format_Grayscale8
            )
            self.loaded_pixmap = QPixmap.fromImage(q_image)
            print(f"Original pixmap loaded with size: {self.loaded_pixmap.size()}")
        except Exception as e:
            print(f"Error loading TIFF file: {e}")
            QMessageBox.critical(self, "Error Loading TIFF", f"An error occurred while loading the TIFF file:\n{e}")
            self.update_instructions()  # Update instructions to reflect error
            return

        # Restore any highlighter drawing saved for this TIFF (or start blank,
        # rather than carrying over another image's highlights).
        self._load_highlight_layer()

        # Preprocess the image for the top-left view (index=0)
        preprocess_steps_view0 = ['normalize']#, 'denoise', 'adaptive_smoothing']  # Define your preprocessing steps for View 1
        try:
            preprocessed_image_view0, noise_estimation_view0 = self.preprocess_image(image_np.copy(), preprocess_steps_view0)
            # Convert preprocessed image to QPixmap
            q_image_preprocessed_view0 = QImage(
                preprocessed_image_view0.data, preprocessed_image_view0.shape[1], preprocessed_image_view0.shape[0],
                preprocessed_image_view0.strides[0], QImage.Format_Grayscale8
            )
            self.preprocessed_pixmap_view0 = QPixmap.fromImage(q_image_preprocessed_view0)
            print(f"Preprocessed pixmap for View 1 created with size: {self.preprocessed_pixmap_view0.size()}")
        except Exception as e:
            print(f"Error preprocessing image for View 1: {e}")
            QMessageBox.critical(self, "Error Preprocessing Image", f"An error occurred during image preprocessing for View 1:\n{e}")
            self.preprocessed_pixmap_view0 = self.loaded_pixmap  # Fallback to original if preprocessing fails
            print("Preprocessed pixmap for View 1 fallback to original pixmap.")

        # Preprocess the image for the top-right view (index=1)
        preprocess_steps_view1 = ['normalize', 'denoise', 'adaptive_smoothing']  # Define your preprocessing steps for View 2
        try:
            preprocessed_image_view1, noise_estimation_view1 = self.preprocess_image(image_np.copy(), preprocess_steps_view1)
            # Convert preprocessed image to QPixmap
            q_image_preprocessed_view1 = QImage(
                preprocessed_image_view1.data, preprocessed_image_view1.shape[1], preprocessed_image_view1.shape[0],
                preprocessed_image_view1.strides[0], QImage.Format_Grayscale8
            )
            self.preprocessed_pixmap_view1 = QPixmap.fromImage(q_image_preprocessed_view1)
            print(f"Preprocessed pixmap for View 2 created with size: {self.preprocessed_pixmap_view1.size()}")
        except Exception as e:
            print(f"Error preprocessing image for View 2: {e}")
            QMessageBox.critical(self, "Error Preprocessing Image", f"An error occurred during image preprocessing for View 2:\n{e}")
            self.preprocessed_pixmap_view1 = self.loaded_pixmap  # Fallback to original if preprocessing fails
            print("Preprocessed pixmap for View 2 fallback to original pixmap.")

        # Preprocess the image for the bottom-right view (index=3)
        preprocess_steps_view3 = ['normalize', 'denoise', 'adaptive_smoothing']  # Define your preprocessing steps for View 4
        try:
            preprocessed_image_view3, noise_estimation_view3 = self.preprocess_image(image_np.copy(), preprocess_steps_view3)
            # Store the preprocessed image as a NumPy array for modification
            self.preprocessed_image_view3_np = preprocessed_image_view3.copy()
            # Convert preprocessed image to QPixmap
            q_image_preprocessed_view3 = QImage(
                preprocessed_image_view3.data, preprocessed_image_view3.shape[1], preprocessed_image_view3.shape[0],
                preprocessed_image_view3.strides[0], QImage.Format_Grayscale8
            )
            self.preprocessed_pixmap_view3 = QPixmap.fromImage(q_image_preprocessed_view3)
            print(f"Preprocessed pixmap for View 4 created with size: {self.preprocessed_pixmap_view3.size()}")
        except Exception as e:
            print(f"Error preprocessing image for View 4: {e}")
            QMessageBox.critical(self, "Error Preprocessing Image", f"An error occurred during image preprocessing for View 4:\n{e}")
            self.preprocessed_pixmap_view3 = self.loaded_pixmap  # Fallback to original if preprocessing fails
            print("Preprocessed pixmap for View 4 fallback to original pixmap.")

        # Load CSV file
        try:
            df = pd.read_csv(csv_path)
            print(f"CSV file loaded with {len(df)} rows.")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            QMessageBox.critical(self, "Error Loading CSV", f"An error occurred while loading the CSV file:\n{e}")
            self.update_instructions()  # Update instructions to reflect error
            return

        # Store loaded image and CSV data for toggling purposes
        self.loaded_df = df

        # Clear existing rectangles
        self.rectangles.clear()
        self.rectangle_id_counter = 0  # Reset the rectangle ID counter

        # Create rectangle data
        for idx, row in df.iterrows():
            status = row['status']
            if status == 'Low_gradient':
                continue  # Skip Low Gradient rectangles

            rect_coords = row['rect']
            try:
                rect_coords = ast.literal_eval(rect_coords)
            except Exception as e:
                print(f"Error parsing 'rect' at index {idx}: {e}")
                continue  # Skip this rectangle

            # Process rect_coords to get a list of (x, y) tuples
            if isinstance(rect_coords, (list, tuple)):
                if len(rect_coords) == 2 and all(isinstance(lst, (list, tuple)) for lst in rect_coords):
                    # rect_coords is a tuple/list of two lists: x_coords and y_coords
                    x_coords, y_coords = rect_coords
                    rect_coords = list(zip(x_coords, y_coords))
                else:
                    # rect_coords is a list of coordinate tuples
                    # Extract only x and y from each tuple
                    rect_coords = [(pt[0], pt[1]) for pt in rect_coords if len(pt) >= 2]
            else:
                print(f"Unexpected format for rect_coords at index {idx}: {rect_coords}")
                continue  # Skip this rectangle

            # Ensure the polygon has non-zero width and height
            x_values = [x for x, y in rect_coords]
            y_values = [y for x, y in rect_coords]
            width = max(x_values) - min(x_values)
            height = max(y_values) - min(y_values)
            if width <= 0 or height <= 0:
                print(f"Skipping rectangle at index {idx} due to zero width or height.")
                continue  # Skip this rectangle

            # Ensure the polygon is closed
            if rect_coords[0] != rect_coords[-1]:
                rect_coords.append(rect_coords[0])

            # Parse 'popt' using numpy.fromstring
            popt_str = row.get('popt', None)
            if pd.notna(popt_str) and popt_str != 'None':
                try:
                    # Remove square brackets and newlines, then parse
                    popt_str_clean = popt_str.strip('[]').replace('\n', ' ').replace('  ', ' ')
                    popt = np.fromstring(popt_str_clean, sep=' ')
                    if popt.size == 0:
                        popt = None  # Empty array, set to None
                except Exception as e:
                    print(f"Error parsing 'popt' at index {idx}: {e}")
                    popt = None
            else:
                popt = None
            # retrieve patch locations
            patch_location = row.get('patch_location', None)
            if pd.notna(patch_location) and patch_location != 'None':
                try:
                    patch_location = ast.literal_eval(patch_location)
                    if not isinstance(patch_location, (list, tuple)):
                        raise ValueError(f"Expected list or tuple, got {type(patch_location)}")
                except Exception as e:
                    print(f"Error parsing 'patch_location' at index {idx}: {e}")
                    patch_location = None
                else:
                    print(f"Successfully parsed 'patch_locations' at index {idx}: {patch_location}")

            # retrieve pixel_size values
            pixel_size = row.get('pixel_size', None)
            if pd.notna(pixel_size) and pixel_size != 'None':
                try:
                    pixel_size = float(pixel_size)
                except Exception as e:
                    print(f"Error parsing 'pixel_size' at index {idx}: {e}")
                    pixel_size = None

            # retrieve classifcation labels
            #label options:
#             classification_label_map = {
            #     p3_pole_class: 'p3_pole',
            #     p4_pole_class: 'p4_pole',
            #     dislocation_class: 'dislocation',
            #     faint_dislocation_class: 'faint',
            #     multiple_dislocations_class: 'multiple',
            #     no_dislocation_class: 'no_dislocation',
            #     overlapping_class: 'overlapping'
            # }
            classification_label = row.get('classification_label', None)
            if pd.notna(classification_label) and classification_label != 'None':
                classification_label = str(classification_label).strip()
                print(f"Successfully read 'classification_label' at index {idx}: {classification_label}")
            else:
                classification_label = None
 


                
            # Create shapely polygon
            shapely_polygon = Polygon(rect_coords)
            self.rectangles.append({
                'coords': rect_coords,
                'status': status,
                'polygon': shapely_polygon,
                'merged': False,
                'id': self.rectangle_id_counter,
                'popt': popt,
                'patch_locations': patch_location,
                'pixel_size': pixel_size,
                'classification_label': classification_label  # Use the correct variable
            })
            self.rectangle_id_counter += 1

        print(f"Total rectangles loaded: {len(self.rectangles)}")
        # Print popt values for debugging
        num_rects_with_popt = sum(1 for rect in self.rectangles if rect.get('popt') is not None)
        print(f"Rectangles with 'popt' parameters: {num_rects_with_popt}")

        # if self.continuous_update:
        self.continuous_refresh()

        # Update the counters label
        self.update_counters()

        self.counters_frame.setVisible(True)

        # Draw on different views
        for i, view in enumerate(self.views):
            self.update_view(view, i)
            self._recenter_views()

    def _recenter_views(self):
        """
        Reset zoom/pan and centre every view on its new pixmap.
        Call this at the end of load_image_and_rectangles().
        """
        for v in self.views:
            # 1) forget any previous zoom/pan
            v.resetTransform()

            # 2) make sure the scroll‑bars start at the top‑left
            v.horizontalScrollBar().setValue(0)
            v.verticalScrollBar().setValue(0)

            # 3) centre (or scale‑to‑fit) the whole scene
            if v.scene().items():
                v.setSceneRect(v.scene().itemsBoundingRect())
                # pick ONE of the following lines:
                v.centerOn(v.scene().itemsBoundingRect().center())      # just centre
                # v.fitInView(v.scene().itemsBoundingRect(), Qt.KeepAspectRatio)  # scale to fit


    def generate_combined_image(self):
        print("Generating combined image...")
        combined_image = np.ones_like(self.image_np, dtype=np.float32) * 0.5  # Initialize with mid-gray
        num_rectangles_with_popt = 0
        for rect in self.rectangles:
            if rect.get('popt') is not None:
                num_rectangles_with_popt += 1

                popt = rect['popt']
                # Get patch coordinates
                patch = rect['patch_locations']
                print(f"Patch locations: {patch}")
                min_x, min_y, max_x, max_y = patch

                pixel_size = rect['pixel_size']
                patch_width = max_x - min_x
                patch_height = max_y - min_y

                # Create grid over the rectangle's region
                # Center the patch around 0,0
                x = (np.arange(patch_width) - patch_width // 2) * pixel_size
                y = (np.arange(patch_height) - patch_height // 2) * pixel_size
                x_grid_full, y_grid_full = np.meshgrid(x, y, indexing='xy')
                xy = np.vstack((x_grid_full.ravel(), y_grid_full.ravel()))
                # Compute model output
                try:
                    model_output = model(xy, *popt)
                except Exception as e:
                    print(f"Error computing model for rectangle {rect['id']}: {e}")
                    continue
                # Reshape model_output to (height, width)
                height = max_y - min_y
                width = max_x - min_x
                if model_output.size != height * width:
                    print(f"Model output size does not match rectangle size for rectangle {rect['id']}")
                    continue  # Skip this rectangle
                model_output = model_output.reshape((height, width))

                # **Apply Masking Logic Here**
                # Unpack parameters from popt
                try:
                    (translation_factor_x, translation_factor_y, stretch_factor_x, stretch_factor_y, theta_x, theta_y,
                     theta_z, amplitude, height_factor, baseline, shear_factor_x, shear_factor_y) = popt
                except ValueError as ve:
                    print(f"Error unpacking popt for rectangle {rect['id']}: {ve}")
                    continue  # Skip this rectangle if popt does not have expected parameters

                # Create initial mask based on baseline threshold
                mask_threshold = 0.08  # Adjust as needed
                mask = np.logical_and(model_output >= baseline - mask_threshold, model_output <= baseline + mask_threshold)

                # Compute gradients along each axis
                grad_y, grad_x = np.gradient(model_output)
                # Compute gradient magnitude
                grad_mag = np.hypot(grad_x, grad_y)
                # Apply gradient threshold to create gradient mask
                gradient_threshold = 0.05  # Adjust as needed
                mask_gradient = grad_mag >= gradient_threshold
                mask_gradient = np.logical_not(mask_gradient)

                # Combine the baseline mask and gradient mask
                mask_combined = np.logical_and(mask, mask_gradient)


                # Optional: Check if the mask has only one connected region
                from scipy.ndimage import label
                s = [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]]
                labeled_mask, num_features = label(mask_combined, structure=s)
                if num_features > 1:
                    print(f"Mask has {num_features} separate features for rectangle {rect['id']}. Skipping.")
                    continue  # Skip this rectangle

                # Apply the mask to the model output
                model_output_masked = np.ma.array(model_output, mask=mask_combined)

                # Normalize model_output_masked to have values between 0 and 1
                min_val = model_output_masked.min()
                max_val = model_output_masked.max()
                if max_val - min_val > 0:
                    model_output_masked = (model_output_masked - min_val) / (max_val - min_val) - 0.5
                else:
                    model_output_masked = model_output_masked * 0  # Zero array

                # Add the masked model output to combined_image at the correct location
                combined_image[min_y:max_y, min_x:max_x] += model_output_masked.filled(0)
            else:
                print(f"No 'popt' for rectangle {rect['id']}")
        print(f"Number of rectangles with 'popt': {num_rectangles_with_popt}")
        if num_rectangles_with_popt == 0:
            print("No rectangles with 'popt' parameters found. Combined image will be empty.")
            # Set fallback so the bottom-left image always has something to display
            self.combined_image_pixmap = self.loaded_pixmap

        else:
            # After addition, combined_image may have values outside [0,1], so we need to clip
            combined_image = np.clip(combined_image, 0, 1)
            # Convert to 0-255 and uint8
            combined_image = combined_image * 255
            combined_image = combined_image.astype(np.uint8)
            # Convert to QPixmap
            height, width = combined_image.shape
            q_image = QImage(
                combined_image.data, width, height, combined_image.strides[0], QImage.Format_Grayscale8
            )
            self.combined_image_pixmap = QPixmap.fromImage(q_image)
            print("Combined image generated.")




    def process_view4_image(self):
        print("Processing View 4 image...")
        processed_image = self.preprocessed_image_view3_np.copy()
        num_rectangles_with_popt = 0

        for rect in self.rectangles:
            if rect.get('popt') is not None and rect['status'] == 'Successful_fit':
                num_rectangles_with_popt += 1

                popt = rect['popt']
                # Get patch coordinates
                patch = rect['patch_locations']
                print(f"Patch locations: {patch}")
                min_x, min_y, max_x, max_y = patch

                # Ensure coordinates are integers
                min_x = int(round(min_x))
                min_y = int(round(min_y))
                max_x = int(round(max_x))
                max_y = int(round(max_y))

                # Check that coordinates are within image bounds
                if min_x < 0 or min_y < 0 or max_x > processed_image.shape[1] or max_y > processed_image.shape[0]:
                    print(f"Patch location out of bounds for rectangle {rect['id']}")
                    continue

                # Get the raw patch from the preprocessed image
                raw_patch = processed_image[min_y:max_y, min_x:max_x]

                # Create a grid for the model
                pixel_size = rect['pixel_size']
                height = max_y - min_y
                width = max_x - min_x

                # Create x and y coordinates in physical units
                x = (np.arange(width) - width // 2) * pixel_size
                y = (np.arange(height) - height // 2) * pixel_size
                x_grid, y_grid = np.meshgrid(x, y, indexing='xy')
                xy = np.vstack((x_grid.ravel(), y_grid.ravel()))

                # Compute model output z
                try:
                    z = model(xy, *popt)
                except Exception as e:
                    print(f"Error computing model for rectangle {rect['id']}: {e}")
                    continue

                # Reshape z to match the raw_patch shape
                z = z.reshape(raw_patch.shape)

                # Compute x_center and y_center in image coordinates
                x_center = min_x + width // 2
                y_center = min_y + height // 2

                # Call smooth_and_add_noise with the correct parameters
                try:
                    smoothed_patch = smooth_and_add_noise(
                        image=raw_patch,
                        z=z,
                        x_center=x_center,
                        y_center=y_center,
                        x_min=min_x,
                        y_min=min_y,
                        x_max=max_x,
                        y_max=max_y,
                        gradient_threshold=0.03, # 0.01
                        laplacian_threshold=0.03, # 0.01
                        smoothing_sigma=20, # 15
                        boundary_blur_sigma=0.5, # 0.5
                        noise_sigma_estimation=0.3, # 0.3
                        noise_type='gaussian', # guassian
                        structuring_element_size=5, # 5
                        dilation_iterations=3, # 3
                        grey_dilation_iterations=3, # 3
                        beta=1.0, # 1
                        gamma=1.0, # 1
                        min_smoothing_value=0.1, # 0.1
                        show_plots=False
                    )
                except Exception as e:
                    print(f"Error in smooth_and_add_noise for rectangle {rect['id']}: {e}")
                    continue

                # # Replace the area in processed_image with the smoothed_patch
                processed_image[min_y:max_y, min_x:max_x] = smoothed_patch
                # processed_image[min_y:max_y, min_x:max_x] = preprocessed_residuals


        # After processing all rectangles, update self.preprocessed_pixmap_view3_processed
        self.preprocessed_image_view3_processed_np = processed_image

        # Update preprocessed_pixmap_view3_processed
        height, width = self.preprocessed_image_view3_processed_np.shape
        q_image = QImage(
            self.preprocessed_image_view3_processed_np.data, width, height, self.preprocessed_image_view3_processed_np.strides[0], QImage.Format_Grayscale8
        )
        self.preprocessed_pixmap_view3_processed = QPixmap.fromImage(q_image)

        print(f"Processed View 4 image with {num_rectangles_with_popt} rectangles.")


    def update_view(self, view, index):

        classification_color_map = self.classification_color_map
        print(f"Updating view {index + 1}...")
        scene = view.scene()

        # Clear previous drawings and storage lists before clearing the scene
        view.clear_drawings(view.rectangles_successful)
        view.clear_drawings(view.rectangles_failed)
        view.clear_drawings(view.rectangles_merged)
        view.clear_drawings(view.rectangles_user_added)

        # keep base image & high-light layer, remove only rectangles
        for it in scene.items():
            if it.zValue() >= 0:          #  rectangles & temp items
                scene.removeItem(it)

        # Create the highlight-overlay **once**, not every refresh
        if not self.highlight_pixmaps:    # first time only
            self._init_highlight_layer(self.loaded_pixmap.width(),
                                       self.loaded_pixmap.height())

        self._init_highlight_layer(self.loaded_pixmap.width(), self.loaded_pixmap.height())

        # Determine which pixmap to use based on the view's index and flags
        if view.index == 0:
            if view.show_preprocessed_image and self.preprocessed_pixmap_view0:
                pixmap_to_use = self.preprocessed_pixmap_view0
                print(f"View {index + 1} is using the preprocessed pixmap (View 1).")
            else:
                pixmap_to_use = self.loaded_pixmap
                print(f"View {index + 1} is using the original pixmap (View 1).")
        elif view.index == 1:
            if view.show_preprocessed_image and self.preprocessed_pixmap_view1:
                pixmap_to_use = self.preprocessed_pixmap_view1
                print(f"View {index + 1} is using the preprocessed pixmap (View 2).")
            else:
                pixmap_to_use = self.loaded_pixmap
                print(f"View {index + 1} is using the original pixmap (View 2).")
        elif view.index == 2:
            # For view index 2 (bottom-left), set pixmap to combined image pixmap
            if self.combined_image_pixmap:
                pixmap_to_use = self.combined_image_pixmap
                print(f"View {index + 1} is using the combined image pixmap.")
            else:
                pixmap_to_use = self.loaded_pixmap
                print(f"Combined image pixmap not available. Using original pixmap for View {index + 1}.")
        elif view.index == 3:
            # For view index 3 (bottom-right), set pixmap to processed image
            if self.preprocessed_pixmap_view3_processed:
                pixmap_to_use = self.preprocessed_pixmap_view3_processed
                print(f"View {index + 1} is using the processed pixmap (View 4).")
            else:
                pixmap_to_use = self.preprocessed_pixmap_view3
                print(f"Processed pixmap not available. Using preprocessed pixmap for View {index + 1}.")
        else:
            # For other views (if any), use the original pixmap
            pixmap_to_use = self.loaded_pixmap
            print(f"View {index + 1} is using the original pixmap (Other View).")



        # Add the pixmap to the scene
        pixmap_item = scene.addPixmap(pixmap_to_use)
        pixmap_item.setZValue(-1)  # Send pixmap to back
        pixmap_item.setAcceptedMouseButtons(Qt.NoButton)  # Disable mouse events
        pixmap_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        pixmap_item.setFlag(QGraphicsItem.ItemIsMovable, False)

        # ensure the existing overlay is visible in this scene
        if view in self.highlight_pixmaps:
            self.highlight_pixmaps[view].setVisible(self.highlight_visible)



        # Add rectangles based on visibility flags and overlapping settings
        show_only_overlapping = self.show_overlapping_checkbox.isChecked()

        for rect in self.rectangles:
            rect_id = rect['id']

            # Skip this rectangle if user wants only overlaps but it's not overlapping
            if show_only_overlapping and rect_id not in self.overlapping_indices:
                continue

            if rect['merged']:
                if not view.rectangles_merged_visible_flag:
                    continue
            elif rect['status'] == 'Successful_fit':
                if not view.rectangles_successful_visible_flag:
                    continue
            elif rect['status'] == 'Failed_fit':
                if not view.rectangles_failed_visible_flag:
                    continue
            elif rect['status'] == 'User_Added':
                if not view.rectangles_user_added_visible_flag:
                    continue


            # 1) Base color from classification_label (if any)
            classification_label = rect.get('classification_label', None)

            # is toggled OFF, then skip drawing it:
            if classification_label in self.classification_toggles:
                if not self.classification_toggles[classification_label].isChecked():
                    continue

            if classification_label:
                base_color = classification_color_map.get(classification_label, '#FFFFFF')
            else:
                base_color = '#FFFFFF'

            # 2) If user_added or merged => override color (matches paper_merged_style in save_image())
            if rect['merged'] or rect['status'] == 'User_Added':
                base_color = '#FF1493'

            # 3) Decide line style for failed vs others. White boxes
            # (unknown / no classification) are dashed - same convention as
            # save_image() - so they don't get lost as a plain solid white
            # line against a bright part of the image.
            if rect['status'] == 'Failed_fit':
                line_style = Qt.DotLine
            elif base_color == '#FFFFFF':
                line_style = Qt.DashLine
            else:
                line_style = Qt.SolidLine

            # 4) Create pen and add polygon
            pen = QPen(QColor(base_color))
            pen.setWidth(2)
            pen.setStyle(line_style)

            coords = rect['coords']
            graphics_item = scene.addPolygon(
                QPolygonF([QPointF(x, y) for x, y in coords]),
                pen,
                QBrush(Qt.NoBrush)
            )
            graphics_item.setData(0, rect_id)
            graphics_item.setZValue(0)
            graphics_item.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)


        # Draw temporary rectangle if it exists
        if self.temp_rectangle:
            coords = self.temp_rectangle['coords']
            pen = QPen(QColor("yellow"), 2, Qt.DashLine)  # Temporary rectangle style
            pen.setWidth(2)  # Set line thickness to 2
            graphics_item = scene.addPolygon(
                QPolygonF([QPointF(x, y) for x, y in coords]),
                pen,
                QBrush(Qt.NoBrush)
            )
            graphics_item.setData(0, "temp")  # Mark as temporary
            graphics_item.setZValue(1)  # Ensure it's above the pixmap but below other items

        # Update highlighting
        view.update_highlighting()

    def sync_views(self, source_view):
        for view in self.views:
            if view is not source_view:
                view.setTransform(source_view.transform())
                view.horizontalScrollBar().setValue(source_view.horizontalScrollBar().value())
                view.verticalScrollBar().setValue(source_view.verticalScrollBar().value())

    def update_views(self):
        if self.need_to_update_overlaps and self.show_overlapping_checkbox.isChecked():
            if ENABLE_OVERLAP_ON_OPEN:
                self.find_overlapping_rectangles()
        self.need_to_update_overlaps = False

        for i, view in enumerate(self.views):
            self.update_view(view, i)

    def update_overlap_threshold(self, value):
        # Always do the first find_overlapping_rectangles
        if ENABLE_OVERLAP_ON_OPEN:
            self.find_overlapping_rectangles()

        # Then, if continuous...
        if self.continuous_update:
            # Instead of refresh_all, call continuous_refresh
            self.continuous_refresh()

        self.update_instructions()

    def find_overlapping_rectangles(self):
        self.overlapping_indices = set()
        overlap_threshold = self.overlap_slider.value()
        print(f"Overlap threshold set to: {overlap_threshold}%")

        for i, rect1 in enumerate(self.rectangles):
            if rect1['merged']:
                continue
            poly1 = rect1['polygon']
            area1 = poly1.area

            for j in range(i + 1, len(self.rectangles)):
                rect2 = self.rectangles[j]
                if rect2['merged']:
                    continue
                poly2 = rect2['polygon']
                area2 = poly2.area
                intersection = poly1.intersection(poly2)

                if not intersection.is_empty:
                    intersection_area = intersection.area
                    overlap_percentage = (intersection_area / min(area1, area2)) * 100

                    if overlap_percentage >= overlap_threshold:
                        self.overlapping_indices.add(rect1['id'])  # Add rect1's id
                        self.overlapping_indices.add(rect2['id'])  # Add rect2's id
                        print(f"Rectangles {rect1['id']} and {rect2['id']} overlap by {overlap_percentage:.2f}%")

        print(f"Total overlapping rectangles: {len(self.overlapping_indices)}")
        self.update_instructions()  # Update instructions if overlapping affects visibility

    def toggle_merge_mode(self):
        self.merge_mode = self.merge_button.isChecked()
        if self.merge_mode:
            # Deactivate other modes
            if self.add_mode:
                self.add_mode = False
                self.add_button.setChecked(False)
                QApplication.restoreOverrideCursor()
                self.temp_rectangle = None  # Clear temporary rectangle
                # Remove temporary rectangle from all views
                for view in self.views:
                    scene = view.scene()
                    for item in scene.items():
                        if item.data(0) == "temp":
                            scene.removeItem(item)
            if self.delete_mode:
                self.delete_mode = False
                self.delete_button.setChecked(False)
                QApplication.restoreOverrideCursor()
            # Set cursor and selection mode
            QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
            for view in self.views:
                view.setDragMode(QGraphicsView.RubberBandDrag)
        else:
            QApplication.restoreOverrideCursor()
            for view in self.views:
                view.setDragMode(QGraphicsView.ScrollHandDrag)
                # Clear any selection and remove highlighting
                view.scene().clearSelection()
                view.update_highlighting()
            # Automatically confirm the merge upon exiting Merge Mode
            self.confirm_merge()

        self.update_instructions()  # Update instructions after mode change


    def confirm_merge(self):
        """
        Merges the currently selected rectangles in all scenes into a single bounding rectangle.
        Immediately updates self.rectangles, logs an undo action, and partially redraws polygons.
        Does a full image refresh only if continuous_update is on.
        """
        # 1) Only proceed if merge_mode is active
        if not self.merge_mode:
            return

        # 2) Gather selected rectangle IDs
        selected_rect_ids = set()
        for view in self.views:
            for item in view.scene().selectedItems():
                if isinstance(item, QGraphicsPolygonItem):
                    rid = item.data(0)
                    if rid is not None and rid != "temp":
                        selected_rect_ids.add(int(rid))

        if not selected_rect_ids:
            QMessageBox.information(self, "No Selection", "No rectangles selected for merging.")
            return

        # 3) Find them in self.rectangles
        selected_rects = [r for r in self.rectangles if r["id"] in selected_rect_ids]
        if not selected_rects:
            QMessageBox.warning(self, "Invalid Selection", "Selected rectangles are invalid or already merged.")
            return

        # 4) Compute bounding rectangle
        min_x = min(min(pt[0] for pt in rect["coords"][:-1]) for rect in selected_rects)
        min_y = min(min(pt[1] for pt in rect["coords"][:-1]) for rect in selected_rects)
        max_x = max(max(pt[0] for pt in rect["coords"][:-1]) for rect in selected_rects)
        max_y = max(max(pt[1] for pt in rect["coords"][:-1]) for rect in selected_rects)

        new_coords = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y)  # close polygon
        ]

        # 5) Create the new rectangle
        new_rect = {
            "coords": new_coords,
            "status": "User_Added",
            "polygon": Polygon(new_coords),
            "merged": False,
            "id": self.rectangle_id_counter,
            "popt": None,
            "patch_locations": [min_x, min_y, max_x, max_y],
            "pixel_size": None
        }

        # Make a copy of the original rectangles for Undo
        original_rects_data = [rect.copy() for rect in selected_rects]

        # 6) Remove old rectangles from data
        self.rectangles = [r for r in self.rectangles if r["id"] not in selected_rect_ids]

        # 7) Insert the new bounding rectangle
        new_id = self.rectangle_id_counter
        self.rectangles.append(new_rect)
        self.rectangle_id_counter += 1
        print(f"[Merge] Created rect {new_id} by merging {selected_rect_ids}.")

        # 8) Record an undo action
        self.action_stack.append({
            "action": "merge",
            "merged_rectangle": new_rect.copy(),
            "original_rectangles": original_rects_data
        })

        # 9) Partially remove old polygons from all scenes, partially add the new one
        for old_rect in selected_rects:
            self.draw_rectangle_in_all_scenes(old_rect, remove=True)
        self.draw_rectangle_in_all_scenes(new_rect, remove=False)

        self._mark_unsaved()

        # 10) If continuous update is on, do heavy steps
        if self.continuous_update:
            self.continuous_refresh()

        # 11) Conclude merge mode
        self.merge_mode = False
        self.merge_button.setChecked(False)
        QApplication.restoreOverrideCursor()
        for view in self.views:
            view.setDragMode(QGraphicsView.ScrollHandDrag)
            view.scene().clearSelection()
            view.update_highlighting()

        self.update_counters()
        self.update_instructions()



    def toggle_delete_mode(self):
        self.delete_mode = self.delete_button.isChecked()
        if self.delete_mode:
            # Deactivate other modes
            if self.add_mode:
                self.add_mode = False
                self.add_button.setChecked(False)
                QApplication.restoreOverrideCursor()
            if self.merge_mode:
                self.merge_mode = False
                self.merge_button.setChecked(False)
                QApplication.restoreOverrideCursor()
                for view in self.views:
                    view.setDragMode(QGraphicsView.ScrollHandDrag)
            # Set cursor to cross for deletion
            QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
        else:
            QApplication.restoreOverrideCursor()

        self.update_instructions()  # Update instructions after mode change


    def draw_rectangle_in_all_scenes(self, rect, remove=False):
        """
        If remove=False, draws rect's polygon in all scenes.
        If remove=True, removes rect's polygon from all scenes (by rect['id']).
        This is a partial redraw for quick visual feedback.
        """
        rect_id = rect['id']
        coords = rect['coords']
        status = rect['status']
        # Decide color from status
        if rect.get('merged', False):
            color = "blue"
        elif status == "Successful_fit":
            color = "green"
        elif status == "Failed_fit":
            color = "red"
        elif status == "User_Added":
            color = "purple"
        else:
            color = "black"

        for view in self.views:
            scene = view.scene()
            if remove:
                # Remove any QGraphicsPolygonItem with data(0) == rect_id
                for item in scene.items():
                    if isinstance(item, QGraphicsPolygonItem) and item.data(0) == rect_id:
                        scene.removeItem(item)
            else:
                # Add a polygon for this rect
                pen = QPen(QColor(color))
                pen.setWidth(2)
                polygon_item = scene.addPolygon(
                    QPolygonF([QPointF(x, y) for x, y in coords]),
                    pen,
                    QBrush(Qt.NoBrush)
                )
                polygon_item.setData(0, rect_id)
                polygon_item.setZValue(0)
                polygon_item.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)

    def delete_graphics_item(self, graphics_item):
        """
        Deletes a rectangle from self.rectangles and provides visual feedback 
        without requiring a full image refresh, unless continuous_update is on.
        """
        rect_id = graphics_item.data(0)
        if rect_id is None:
            return

        rect_id = int(rect_id)
        # 1) Find the rectangle in self.rectangles
        rect_to_remove = next((r for r in self.rectangles if r["id"] == rect_id), None)
        if not rect_to_remove:
            print(f"[Delete] Rectangle {rect_id} not found in self.rectangles.")
            return

        # 2) Remove it from the data
        self.rectangles.remove(rect_to_remove)
        print(f"[Delete] Rectangle {rect_id} removed from data.")
        self._mark_unsaved()

        # 3) Record an undo action
        self.action_stack.append({
            "action": "delete",
            "rectangle": rect_to_remove.copy()
        })

        # 4) Partially remove it from the scenes (visual)
        self.draw_rectangle_in_all_scenes(rect_to_remove, remove=True)

        # (Optionally also remove the clicked item directly, if you want.)
        scene = graphics_item.scene()
        if scene:
            scene.removeItem(graphics_item)

        # 5) If continuous update is on, do the heavy computations
        if self.continuous_update:
            self.continuous_refresh()

        # 6) Finally, update counters & instructions
        self.update_counters()
        self.update_instructions()
        print(f"[Delete] Rectangle {rect_id} fully handled.")



    def undo_last_action(self):
        if not self.action_stack:
            QMessageBox.information(self, "Undo Failed", "No actions to undo.")
            return

        last_action = self.action_stack.pop()
        action_type = last_action['action']

        if action_type == "add":
            rect = last_action['rectangle']
            # 1) Remove from self.rectangles
            self.rectangles = [r for r in self.rectangles if r['id'] != rect['id']]
            print(f"Undo Add: Rectangle {rect['id']} removed from data.")

            # 2) Partially remove from scenes
            self.draw_rectangle_in_all_scenes(rect, remove=True)

        elif action_type == "delete":
            rect = last_action['rectangle']
            # 1) Re-add to self.rectangles
            self.rectangles.append(rect)
            print(f"Undo Delete: Rectangle {rect['id']} restored.")

            # 2) Partially draw it again
            self.draw_rectangle_in_all_scenes(rect, remove=False)

        elif action_type == "merge":
            merged_rect = last_action["merged_rectangle"]
            original_rects = last_action["original_rectangles"]

            # 1) Remove the merged rect from data
            self.rectangles = [r for r in self.rectangles if r['id'] != merged_rect['id']]
            print(f"Undo Merge: merged rect {merged_rect['id']} removed.")

            self._mark_unsaved()

            # 2) Re-add the original rects
            self.rectangles.extend(original_rects)
            for r in original_rects:
                print(f"Undo Merge: original rect {r['id']} restored.")

            # 3) Partially remove the merged from scenes, partially add the old rects
            self.draw_rectangle_in_all_scenes(merged_rect, remove=True)
            for old_rect in original_rects:
                self.draw_rectangle_in_all_scenes(old_rect, remove=False)

        else:
            QMessageBox.warning(self, "Undo Failed", f"Unknown action type: {action_type}")
            return

        # We skip big find_overlapping, generate_combined_image calls here
        # for performance. If user wants updated images, they can press Refresh 
        # or rely on continuous_update.

        print(f"Undo Action: {action_type} undone.")
        self.update_counters()
        self.update_instructions()


    def toggle_add_mode(self):
        self.add_mode = self.add_button.isChecked()
        if self.add_mode:
            # Deactivate other modes
            if self.delete_mode:
                self.delete_mode = False
                self.delete_button.setChecked(False)
                QApplication.restoreOverrideCursor()
            if self.merge_mode:
                self.merge_mode = False
                self.merge_button.setChecked(False)
                QApplication.restoreOverrideCursor()
                for view in self.views:
                    view.setDragMode(QGraphicsView.ScrollHandDrag)
            # Set cursor to cross for adding
            QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
        else:
            QApplication.restoreOverrideCursor()
            self.temp_rectangle = None  # Clear temporary rectangle
            # Remove temporary rectangle from all views
            for view in self.views:
                scene = view.scene()
                for item in scene.items():
                    if item.data(0) == "temp":
                        scene.removeItem(item)
        self.update_instructions()  # Update instructions after mode change

    def update_rectangle_width(self, value):
        self.rectangle_width = value
        self.update_temp_rectangle()
        self.update_instructions()  # Update instructions if size changes

    def update_rectangle_height(self, value):
        self.rectangle_height = value
        self.update_temp_rectangle()
        self.update_instructions()  # Update instructions if size changes

    def update_temp_rectangle(self):
        # Remove previous temporary rectangle from all views and redraw
        for view in self.views:
            scene = view.scene()
            # Remove previous temporary rectangle
            for item in scene.items():
                if item.data(0) == "temp":
                    scene.removeItem(item)

            if self.temp_rectangle:
                # Update the size of the temporary rectangle
                x, y = self.temp_rectangle['center']
                width = self.rectangle_width
                height = self.rectangle_height
                rect_coords = [
                    (x - width / 2, y - height / 2),
                    (x + width / 2, y - height / 2),
                    (x + width / 2, y + height / 2),
                    (x - width / 2, y + height / 2),
                    (x - width / 2, y - height / 2)  # Close the rectangle
                ]
                self.temp_rectangle['coords'] = rect_coords

                # Create new temporary rectangle
                pen = QPen(QColor("yellow"), 2, Qt.DashLine)  # Temporary rectangle style
                pen.setWidth(2)  # Set line thickness to 2
                graphics_item = scene.addPolygon(
                    QPolygonF([QPointF(px, py) for px, py in rect_coords]),
                    pen,
                    QBrush(Qt.NoBrush)
                )
                graphics_item.setData(0, "temp")  # Mark as temporary
                graphics_item.setZValue(1)  # Ensure it's above the pixmap but below other items

    def finalize_temp_rectangle(self):
        if not self.temp_rectangle:
            return

        # Calculate bounding coords
        x_coords = [pt[0] for pt in self.temp_rectangle['coords']]
        y_coords = [pt[1] for pt in self.temp_rectangle['coords']]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Build new rectangle
        new_rect = {
            'coords': self.temp_rectangle['coords'],
            'status': 'User_Added',
            'polygon': Polygon(self.temp_rectangle['coords']),
            'merged': False,
            'id': self.rectangle_id_counter,
            'popt': None,
            'patch_locations': [min_x, min_y, max_x, max_y],
            'pixel_size': None
        }

        # 1) Insert into self.rectangles
        self.rectangles.append(new_rect)
        rect_id = self.rectangle_id_counter
        self.rectangle_id_counter += 1
        print(f"[Add] Rectangle {rect_id} added to self.rectangles.")

        # 2) Record the Add action for undo
        self.action_stack.append({
            "action": "add",
            "rectangle": new_rect.copy()
        })
        self._mark_unsaved()

        # 3) Partially draw the new rectangle (visual confirmation)
        self.draw_rectangle_in_all_scenes(new_rect, remove=False)

        # 2) Clear the temporary rectangle from all views
        for view in self.views:
            scene = view.scene()
            for item in scene.items():
                if item.data(0) == "temp":
                    scene.removeItem(item)
        self.temp_rectangle = None

        # 3) If continuous update is ON, run heavy computations
        if self.continuous_update:
            self.continuous_refresh()
        
        # 4) Now redraw
        self.update_views()
        self.update_counters()
        self.update_instructions()

    def apply_pending_actions(self):
        """
        Actually update self.rectangles with merges, adds, and deletes
        that were only done visually while in non-continuous mode.
        """
        # 1) handle all local deletes
        for rect_id in self.pending_deletes:
            rect_obj = next((r for r in self.rectangles if r['id'] == rect_id), None)
            if rect_obj:
                self.rectangles.remove(rect_obj)
                print(f"[REAL DELETE] Rectangle {rect_id} removed from data.")
        self.pending_deletes.clear()

        # 2) handle all local adds
        for rect_obj in self.pending_adds:
            rect_obj['id'] = self.rectangle_id_counter
            self.rectangle_id_counter += 1
            self.rectangles.append(rect_obj)
            print(f"[REAL ADD] Rectangle {rect_obj['id']} added to data.")
        self.pending_adds.clear()

        # 3) handle merges
        for merge_info in self.pending_merges:
            # 3a) remove old ones from data
            for old_id in merge_info["original_ids"]:
                old_obj = next((r for r in self.rectangles if r['id'] == old_id), None)
                if old_obj:
                    self.rectangles.remove(old_obj)
                    print(f"[REAL MERGE] Old rect {old_id} removed.")
            # 3b) add the new merged rect
            new_rect_obj = merge_info["new_rect_data"]
            new_rect_obj['id'] = self.rectangle_id_counter
            self.rectangle_id_counter += 1
            self.rectangles.append(new_rect_obj)
            print(f"[REAL MERGE] Merged rect {new_rect_obj['id']} added.")
        self.pending_merges.clear()

        # 4) Mark that we need to recalc images/overlaps
        if self.pending_deletes or self.pending_adds or self.pending_merges:
            self.need_recompute_images = True
            self.need_to_update_overlaps = True


    def keyPressEvent(self, event):
        """
        Global shortcuts.

        * ESC leaves Add/Delete/Merge *and* quits Pen/Eraser,
        but **never touches the stroke⇆fill switch** (area_mode).
        """
        k = event.key()

        # Delete selected
        if k == Qt.Key_Delete:
            for v in self.views:
                for it in v.scene().selectedItems():
                    if isinstance(it, QGraphicsPolygonItem):
                        self.delete_graphics_item(it)
            self.update_views()
            return

        # Confirm actions
        if k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            if self.add_mode and self.temp_rectangle:
                self.finalize_temp_rectangle()
            elif self.merge_mode:
                self.confirm_merge()
            return

        # Escape
        if k == Qt.Key_Escape:
            # quit Pen / Eraser only
            if self.highlight_mode or self.erase_mode:
                self._activate_tool(False, False)
                return

            # leave Add / Delete / Merge modes
            if self.add_mode:
                self.add_button.setChecked(False)
                self.toggle_add_mode();   return
            if self.delete_mode:
                self.delete_button.setChecked(False)
                self.toggle_delete_mode();   return
            if self.merge_mode:
                self.merge_button.setChecked(False)
                self.toggle_merge_mode();   return

            # otherwise just clear selection
            for v in self.views:
                v.scene().clearSelection()
            self.update_instructions()
            return

        # Temp-rect size (Add mode)
        if self.add_mode:
            if k == Qt.Key_Left:
                self.width_slider.setValue(
                    max(self.width_slider.value()-1, self.width_slider.minimum()))
                return
            if k == Qt.Key_Right:
                self.width_slider.setValue(
                    min(self.width_slider.value()+1, self.width_slider.maximum()))
                return
            if k == Qt.Key_Up:
                self.height_slider.setValue(
                    min(self.height_slider.value()+1, self.height_slider.maximum()))
                return
            if k == Qt.Key_Down:
                self.height_slider.setValue(
                    max(self.height_slider.value()-1, self.height_slider.minimum()))
                return

        # One-key toggles
        if k == Qt.Key_G:
            self.grid_toggle_checkbox.toggle();   return
        if k == Qt.Key_O:
            self.show_overlapping_checkbox.toggle();   return

        # default
        super().keyPressEvent(event)


    def preprocess_image(self, image_array, preprocess_steps, clip_top=20, clip_bottom=50, clip_left=5, clip_right=5,
                         max_sigma=50, increment=1, max_gradient_ratio=0.2, max_allowed_percentage=0.8):

        print("Starting preprocessing...")

        def adaptively_smooth_image(image, max_sigma=50, increment=1, max_gradient_ratio=0.2, max_allowed_percentage=0.8):
            def calculate_gradient(image):
                gx, gy = np.gradient(image)
                gradient_magnitude = np.sqrt(gx**2 + gy**2)
                return gradient_magnitude, gx, gy

            original_gradient, _, _ = calculate_gradient(image)
            max_allowed_gradient = max_gradient_ratio * np.mean(original_gradient)
            num_exceeding_pixels = np.sum(original_gradient > max_allowed_gradient)

            print(f"Original gradient mean: {np.mean(original_gradient):.4f}, max_allowed_gradient: {max_allowed_gradient:.4f}")
            print(f"Number of pixels exceeding max allowed gradient: {num_exceeding_pixels}")

            for sigma in np.arange(1, max_sigma + 1, increment):
                smoothed_image = gaussian_filter(image, sigma=sigma)
                current_gradient, _, _ = calculate_gradient(smoothed_image)
                num_exceeding_pixels_after = np.sum(current_gradient > max_allowed_gradient)
                reduction_percentage = (num_exceeding_pixels - num_exceeding_pixels_after) / num_exceeding_pixels * 100

                print(f"Sigma: {sigma}, Reduction Percentage: {reduction_percentage:.2f}%")

                if reduction_percentage >= max_allowed_percentage:
                    print(f"Adaptive smoothing applied with sigma: {sigma}")
                    return smoothed_image, sigma

            print(f"Adaptive smoothing not achieved, using max_sigma: {max_sigma}")
            return gaussian_filter(image, sigma=max_sigma), max_sigma

        noise_estimation = 0

        # Clipping the image
        if 'clip' in preprocess_steps:
            original_height, original_width = image_array.shape
            image_array = image_array[clip_top:original_height - clip_bottom,
                                      clip_left:original_width - clip_right]
            print(f"Image clipped to shape: {image_array.shape}")

        for step in preprocess_steps:
            if step == 'normalize':
                arr_min = np.min(image_array)
                arr_max = np.max(image_array)
                if arr_max - arr_min == 0:
                    print("Normalization skipped due to zero variance.")
                    image_array = image_array * 0  # Set to zero
                else:
                    image_array = (image_array - arr_min) / (arr_max - arr_min)
                print("Normalization applied.")
            elif step == 'denoise':
                noise_estimation = estimate_sigma(image_array, average_sigmas=True)
                image_array = denoise_wavelet(image_array, sigma=0.5 * noise_estimation, wavelet='sym4', method='BayesShrink', mode='soft')
                print(f"Denoising applied with estimated noise: {noise_estimation:.4f}")
            elif step == 'adaptive_smoothing':
                image_array, smoothing_sigma = adaptively_smooth_image(image_array, max_sigma, increment, max_gradient_ratio, max_allowed_percentage)
                print(f"Adaptive smoothing completed with sigma: {smoothing_sigma}")

        # Apply light Gaussian blur to fill any remaining specs or noise
        image_array = gaussian_filter(image_array, sigma=1)
        print("Light Gaussian blur applied.")

        # Normalize image back to 0-255 for saving as uint8
        arr_min = np.min(image_array)
        arr_max = np.max(image_array)
        if arr_max - arr_min > 0:
            image_array = (image_array - arr_min) / (arr_max - arr_min) * 255
        else:
            image_array = np.zeros_like(image_array)
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        print("Image normalized back to 0-255 and converted to uint8.")

        return image_array, noise_estimation

    def toggle_continuous_update(self, checked: bool):
        self.continuous_update = checked
        print(f"[DEBUG] continuous_update set to: {self.continuous_update}")
    
    def toggle_local_density_mode(self):
        self.local_density_mode = self.local_density_button.isChecked()
        if self.local_density_mode:
            QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
        else:
            QApplication.restoreOverrideCursor()
            # Optionally clear any leftover rectangle
            for view in self.views:
                if hasattr(view, 'local_density_rect_item') and view.local_density_rect_item:
                    view.scene().removeItem(view.local_density_rect_item)
                    view.local_density_rect_item = None



    def handle_refresh_button(self):
        print("[DEBUG] Refresh button was clicked!")  # <--- debugging
        self.continuous_refresh()

    def continuous_refresh(self):
        # Apply any merges/adds/deletes not done in continuous mode
        self.apply_pending_actions()

        # Then do all the heavy computations
        if ENABLE_OVERLAP_ON_OPEN:
            self.find_overlapping_rectangles()
        self.generate_combined_image()
        self.process_view4_image()
        self.update_views()
        QApplication.processEvents()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TIFFViewer()
    viewer.show()
    sys.exit(app.exec())

