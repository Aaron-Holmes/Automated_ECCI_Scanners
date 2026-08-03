import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, QMainWindow, QHBoxLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QKeySequence
from tensorflow.keras.models import load_model

class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the dislocation locator model
        self.dislocation_locator_model = load_model('bounding_box_model_gh.h5')

        # Initialize variables
        self.image_paths = []
        self.processed_images = set()  # To keep track of already processed images
        self.current_image_index = 0
        self.current_image = None
        self.rect_center = None
        self.rect_width = 10
        self.rect_height = 10
        self.window_width = 600
        self.window_height = 600
        self.scale_factor = 1.0
        self.save_file_path = "image_annotations.txt"  # Name of the save file

        # Load the already processed images if file exists
        self.load_processed_images()

        # Setup UI
        self.setWindowTitle("Image Annotator")
        self.setGeometry(100, 100, self.window_width, self.window_height)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.window_width, self.window_height)
        self.layout.addWidget(self.image_label)

        # Sliders
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setTickPosition(QSlider.TicksBelow)
        self.width_slider.setTickInterval(5)
        self.layout.addWidget(QLabel("Rectangle Width"))
        self.layout.addWidget(self.width_slider)

        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setTickPosition(QSlider.TicksBelow)
        self.height_slider.setTickInterval(5)
        self.layout.addWidget(QLabel("Rectangle Height"))
        self.layout.addWidget(self.height_slider)

        # Buttons
        # Create a horizontal layout for buttons
        self.button_layout = QHBoxLayout()

        self.save_bounds_button = QPushButton("Save Bounds (Q)", self)
        self.save_bounds_button.clicked.connect(self.save_bounds)
        self.save_bounds_button.setShortcut(QKeySequence("Q"))
        self.button_layout.addWidget(self.save_bounds_button)

        self.no_dislocation_button = QPushButton("No Dislocation (W)", self)
        self.no_dislocation_button.clicked.connect(self.no_dislocation)
        self.no_dislocation_button.setShortcut(QKeySequence("W"))
        self.button_layout.addWidget(self.no_dislocation_button)

        self.unsure_button = QPushButton("Unsure (E)", self)
        self.unsure_button.clicked.connect(self.unsure)
        self.unsure_button.setShortcut(QKeySequence("E"))
        self.button_layout.addWidget(self.unsure_button)

        self.undo_button = QPushButton("Undo (Z)", self)
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setShortcut(QKeySequence("Z"))
        self.button_layout.addWidget(self.undo_button)

        self.layout.addLayout(self.button_layout)

        # Open folder dialog
        self.open_folder_button = QPushButton("Select Folder", self)
        self.open_folder_button.clicked.connect(self.select_folder)
        self.layout.addWidget(self.open_folder_button)

        # Bind sliders and actions
        self.width_slider.valueChanged.connect(self.update_rectangle)
        self.height_slider.valueChanged.connect(self.update_rectangle)
        self.image_label.mousePressEvent = self.on_click

        # Ensure the focus is on the main window for key press events
        self.central_widget.setFocusPolicy(Qt.StrongFocus)

    def load_processed_images(self):
        """Load processed images from the save file to skip already processed ones."""
        if os.path.exists(self.save_file_path):
            with open(self.save_file_path, 'r') as file:
                for line in file:
                    image_path = line.strip().split(',')[0]
                    self.processed_images.add(image_path)
        print(f"Loaded {len(self.processed_images)} processed images.")

    def save_processed_image(self, image_path, minx, miny, maxx, maxy, label):
        """Save the processed image path, bounding box, and label to the save file."""
        with open(self.save_file_path, 'a') as file:
            file.write(f"{image_path},{minx},{miny},{maxx},{maxy},{label}\n")
        print(f"Saved bounds for image: {image_path}")

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with TIFF Images")
        if folder_path:
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.tiff')]
            # Filter out already processed images
            self.image_paths = [path for path in self.image_paths if path not in self.processed_images]
            if self.image_paths:
                self.load_image()
            else:
                print("All images in this folder have already been processed.")

    def load_image(self):
        if self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            self.current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_height, img_width = self.current_image.shape
            self.rect_center = None
            self.calculate_scale_factor()

            # Set the sliders' ranges based on image dimensions
            self.width_slider.setRange(1, img_width)
            self.height_slider.setRange(1, img_height)

            # Use the model to predict the bounding box
            self.predict_bounding_box()
            self.display_image()
        else:
            print("No more images to annotate.")
            self.close()

    def calculate_scale_factor(self):
        img_height, img_width = self.current_image.shape
        scale_x = self.window_width / img_width
        scale_y = self.window_height / img_height
        self.scale_factor = min(scale_x, scale_y)

    def normalize_and_resize(self, image):
        """Resize to 60x60 for the model."""
        # Since your model was trained without normalization, we only resize
        image_resized = cv2.resize(image, (60, 60), interpolation=cv2.INTER_AREA)
        return image_resized

    def predict_bounding_box(self):
        """Use the model to predict the bounding box and update the rectangle and sliders."""
        if self.current_image is not None:
            # Resize the image for the model
            image_for_model = self.normalize_and_resize(self.current_image)
            # Expand dimensions to match the model's expected input shape
            image_input = np.expand_dims(image_for_model, axis=0)  # Add batch dimension
            image_input = np.expand_dims(image_input, axis=-1)     # Add channel dimension
            # Predict bounding box (coordinates relative to 60x60 image)
            prediction = self.dislocation_locator_model.predict(image_input)
            # Extract predicted bounding box coordinates
            minx_pred, miny_pred, maxx_pred, maxy_pred = prediction[0]
            # Ensure the predicted values are within the valid range
            minx_pred = np.clip(minx_pred, 0, 60)
            miny_pred = np.clip(miny_pred, 0, 60)
            maxx_pred = np.clip(maxx_pred, 0, 60)
            maxy_pred = np.clip(maxy_pred, 0, 60)
            # Scale to the original image size
            img_height, img_width = self.current_image.shape
            minx = int(minx_pred / 60 * img_width)
            miny = int(miny_pred / 60 * img_height)
            maxx = int(maxx_pred / 60 * img_width)
            maxy = int(maxy_pred / 60 * img_height)
            # Ensure coordinates are within image bounds
            minx = np.clip(minx, 0, img_width - 1)
            miny = np.clip(miny, 0, img_height - 1)
            maxx = np.clip(maxx, 0, img_width - 1)
            maxy = np.clip(maxy, 0, img_height - 1)
            # Update rectangle center and size
            x_center = (minx + maxx) // 2
            y_center = (miny + maxy) // 2
            width = maxx - minx
            height = maxy - miny
            # Ensure width and height are positive
            width = max(width, 1)
            height = max(height, 1)
            # Update rectangle center and size
            self.rect_center = (int(x_center), int(y_center))
            self.rect_width = int(width)
            self.rect_height = int(height)
            # Update sliders to match the predicted rectangle size
            self.width_slider.blockSignals(True)
            self.height_slider.blockSignals(True)
            self.width_slider.setValue(self.rect_width)
            self.height_slider.setValue(self.rect_height)
            self.width_slider.blockSignals(False)
            self.height_slider.blockSignals(False)
            # Debugging information
            print(f"Predicted bounding box (60x60 image): {minx_pred}, {miny_pred}, {maxx_pred}, {maxy_pred}")
            print(f"Scaled bounding box (original image): minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")
            print(f"Rectangle center: {self.rect_center}, width: {self.rect_width}, height: {self.rect_height}")
        else:
            self.rect_center = None

    def display_image(self):
        if self.current_image is not None:
            img_resized = cv2.resize(self.current_image, (int(self.current_image.shape[1] * self.scale_factor),
                                                          int(self.current_image.shape[0] * self.scale_factor)))
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            if self.rect_center is not None:
                self.draw_rectangle(img_resized)
            else:
                print("No rectangle to draw.")
            # Convert to QImage for display
            height, width, channel = img_resized.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
        else:
            print("No image to display.")

    def on_click(self, event):
        # If user clicks on the screen, allow manual rectangle drawing
        x = int(event.pos().x() / self.scale_factor)
        y = int(event.pos().y() / self.scale_factor)
        self.rect_center = (x, y)
        self.update_rectangle()

    def update_rectangle(self):
        self.rect_width = self.width_slider.value()
        self.rect_height = self.height_slider.value()
        self.display_image()

    def draw_rectangle(self, img):
        if self.rect_center is None:
            return
        x, y = self.rect_center
        w = self.rect_width // 2
        h = self.rect_height // 2
        minx = max(int((x - w) * self.scale_factor), 0)
        miny = max(int((y - h) * self.scale_factor), 0)
        maxx = min(int((x + w) * self.scale_factor), img.shape[1] - 1)
        maxy = min(int((y + h) * self.scale_factor), img.shape[0] - 1)
        cv2.rectangle(img, (minx, miny), (maxx, maxy), (255, 0, 0), 2)

    def skip_image(self):
        print(f"Skipped image: {self.image_paths[self.current_image_index]}")
        self.processed_images.add(self.image_paths[self.current_image_index])
        self.current_image_index += 1
        self.load_image()

    def no_dislocation(self):
        """Handle 'No Dislocation' action."""
        image_path = self.image_paths[self.current_image_index]
        self.save_processed_image(image_path, 0, 0, 0, 0, 'no_dislocation')
        self.processed_images.add(image_path)
        self.current_image_index += 1
        self.load_image()

    def unsure(self):
        """Handle 'Unsure' action."""
        image_path = self.image_paths[self.current_image_index]
        self.save_processed_image(image_path, 0, 0, 0, 0, 'unsure')
        self.processed_images.add(image_path)
        self.current_image_index += 1
        self.load_image()

    def save_bounds(self):
        """Save the bounding box of the current image."""
        if self.rect_center is None:
            print("No rectangle selected.")
            return
        x, y = self.rect_center
        w = self.rect_width // 2
        h = self.rect_height // 2
        minx = max(x - w, 0)
        miny = max(y - h, 0)
        maxx = min(x + w, self.current_image.shape[1] - 1)
        maxy = min(y + h, self.current_image.shape[0] - 1)

        image_path = self.image_paths[self.current_image_index]
        self.save_processed_image(image_path, minx, miny, maxx, maxy, 'dislocation')

        self.processed_images.add(image_path)
        self.current_image_index += 1
        self.load_image()

    def undo(self):
        """Undo the last action and return to the previous image."""
        if self.current_image_index > 0:
            # Remove last annotation from the file
            if os.path.exists(self.save_file_path):
                with open(self.save_file_path, 'r') as file:
                    lines = file.readlines()
                if lines:
                    # Remove the last line
                    last_line = lines[-1]
                    lines = lines[:-1]
                    # Write back the remaining lines
                    with open(self.save_file_path, 'w') as file:
                        file.writelines(lines)
                    # Remove the image from processed_images
                    last_image_path = self.image_paths[self.current_image_index - 1]
                    self.processed_images.discard(last_image_path)
                    # Decrement current_image_index
                    self.current_image_index -= 1
                    # Load previous image
                    self.load_image()
                    print(f"Undo: Removed annotation for image {last_image_path}")
                else:
                    print("No annotations to undo.")
            else:
                print("Annotations file does not exist.")
        else:
            print("No previous image to undo to.")

    def keyPressEvent(self, event):
        # Handle left and right keys for adjusting width
        if event.key() == Qt.Key_Left:
            current_value = self.width_slider.value()
            self.width_slider.setValue(max(current_value - 1, self.width_slider.minimum()))
        elif event.key() == Qt.Key_Right:
            current_value = self.width_slider.value()
            self.width_slider.setValue(min(current_value + 1, self.width_slider.maximum()))

        # Handle up and down keys for adjusting height
        elif event.key() == Qt.Key_Up:
            current_value = self.height_slider.value()
            self.height_slider.setValue(min(current_value + 1, self.height_slider.maximum()))
        elif event.key() == Qt.Key_Down:
            current_value = self.height_slider.value()
            self.height_slider.setValue(max(current_value - 1, self.height_slider.minimum()))

        # Save the rectangle with 'Q' key
        elif event.key() == Qt.Key_Q:
            self.save_bounds()

        # 'W' key for 'No Dislocation'
        elif event.key() == Qt.Key_W:
            self.no_dislocation()

        # 'E' key for 'Unsure'
        elif event.key() == Qt.Key_E:
            self.unsure()

        # 'Z' key for Undo
        elif event.key() == Qt.Key_Z:
            self.undo()

        # Update rectangle whenever keys change values
        self.update_rectangle()

if __name__ == "__main__":
    app = QApplication([])
    window = ImageAnnotator()
    window.show()
    app.exec()
