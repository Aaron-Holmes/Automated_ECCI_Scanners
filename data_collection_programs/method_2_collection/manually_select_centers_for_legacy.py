import sys
import os
import numpy as np
import tifffile as tiff
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
import cv2

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout)
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PySide6.QtCore import Qt

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.click_callback = None

    def mousePressEvent(self, event):
        if self.click_callback:
            self.click_callback(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load models
        self.find_bounds_of_dislocation = load_model('bounding_box_model_gh.h5')

        # Initialize UI components
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dislocation Center Finder')

        # Create buttons
        self.select_folder_button = QPushButton('Select Folder')
        self.select_folder_button.clicked.connect(self.select_folder)

        self.skip_button = QPushButton('Skip [S]')
        self.skip_button.clicked.connect(self.skip_image)
        self.skip_button.setEnabled(False)

        self.cannot_find_button = QPushButton("Cannot Find [F]")
        self.cannot_find_button.clicked.connect(self.cannot_find)
        self.cannot_find_button.setEnabled(False)

        # Create image display area
        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.click_callback = self.image_clicked

        # Layouts
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.select_folder_button)
        top_layout.addWidget(self.skip_button)
        top_layout.addWidget(self.cannot_find_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
    def keyPressEvent(self, event):
        """Capture key press events."""
        if event.key() == Qt.Key_S:
            self.skip_image()
        elif event.key() == Qt.Key_F:
            self.cannot_find()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')

        if folder:
            self.folder = folder
            self.image_files = [f for f in os.listdir(folder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
            self.current_image_index = -1
            self.skip_button.setEnabled(True)
            self.cannot_find_button.setEnabled(True)
            self.process_next_image()

    def process_next_image(self):
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_files):
            # No more images
            self.image_label.setText('No more images.')
            self.skip_button.setEnabled(False)
            self.cannot_find_button.setEnabled(False)
            return

        image_file = self.image_files[self.current_image_index]
        image_path = os.path.join(self.folder, image_file)
        self.current_image_file = image_file
        self.current_image_path = image_path

        # Load the image
        self.image = self.load_tiff_image(image_path)

        # Process the image to get the bounding box
        self.process_image()

    def load_tiff_image(self, tiff_path):
        img = load_img(tiff_path, color_mode='grayscale')
        image_np = img_to_array(img)
        image_np = np.squeeze(image_np)
        return image_np

    def process_image(self):
        # Resize the image to the required input size for the model
        model_input_shape = (60, 60)

        # Check if image size is less than model input size
        if self.image.shape[0] < model_input_shape[0] or self.image.shape[1] < model_input_shape[1]:
            # Skip this image
            self.process_next_image()
            return

        # Resize image
        image_resized = cv2.resize(self.image, model_input_shape, interpolation=cv2.INTER_LINEAR)

        # Preprocess the image
        image_resized = np.expand_dims(image_resized, axis=0)
        image_resized = np.expand_dims(image_resized, axis=-1)

        # Predict bounding box
        bounds = self.find_bounds_of_dislocation.predict(image_resized)
        x_min, y_min, x_max, y_max = bounds[0]

        # Convert bounding box coordinates back to original image size
        scale_x = self.image.shape[1] / model_input_shape[1]
        scale_y = self.image.shape[0] / model_input_shape[0]

        x_min = x_min * scale_x
        y_min = y_min * scale_y
        x_max = x_max * scale_x
        y_max = y_max * scale_y

        # Ensure bounding box is within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.image.shape[1], x_max)
        y_max = min(self.image.shape[0], y_max)

        self.bounding_box = (x_min, y_min, x_max, y_max)

        # Display the image with bounding box
        self.display_image()

    def display_image(self):
        # Convert image to QPixmap
        # First, normalize image to 0-255 and convert to uint8
        image_norm = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))
        image_uint8 = (image_norm * 255).astype(np.uint8)

        # Convert to QImage
        height, width = image_uint8.shape
        bytes_per_line = width
        q_image = QImage(image_uint8.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # Create a QPainter to draw the bounding box
        painter = QPainter(pixmap)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)

        x_min, y_min, x_max, y_max = self.bounding_box

        # Draw rectangle
        painter.drawRect(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        painter.end()

        # Get the size of the image_label
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # Scale the pixmap to fit the label
        pixmap_scaled = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio)

        # Store scaled pixmap dimensions
        self.pixmap_scaled_width = pixmap_scaled.width()
        self.pixmap_scaled_height = pixmap_scaled.height()

        # Calculate scaling factors
        self.scale_factor_x = pixmap.width() / self.pixmap_scaled_width
        self.scale_factor_y = pixmap.height() / self.pixmap_scaled_height

        # Display the pixmap
        self.image_label.setPixmap(pixmap_scaled)

    def image_clicked(self, event):
        # Get the position where the user clicked
        x = event.pos().x()
        y = event.pos().y()

        # Calculate offsets (margins)
        x_offset = (self.image_label.width() - self.pixmap_scaled_width) / 2
        y_offset = (self.image_label.height() - self.pixmap_scaled_height) / 2

        # Adjust the click position to account for offsets
        x_adjusted = x - x_offset
        y_adjusted = y - y_offset

        # Ensure the click is within the displayed image
        if x_adjusted < 0 or x_adjusted >= self.pixmap_scaled_width or y_adjusted < 0 or y_adjusted >= self.pixmap_scaled_height:
            print("Clicked outside the image area.")
            return

        # Map back to image coordinates
        image_x = x_adjusted * self.scale_factor_x
        image_y = y_adjusted * self.scale_factor_y

        # Get the bounding box
        x_min, y_min, x_max, y_max = self.bounding_box

        # Calculate the largest side
        width = x_max - x_min
        height = y_max - y_min
        largest_side = int(max(width, height) * 1.5)

        # Create a square that surrounds the bounding box, with side length = largest_side
        # Adjust x_min, y_min to make the square centered on the bounding box
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        x_square_min = x_center - largest_side / 2
        y_square_min = y_center - largest_side / 2
        x_square_max = x_center + largest_side / 2
        y_square_max = y_center + largest_side / 2

        # Ensure the square is within the image bounds
        x_square_min = max(0, x_square_min)
        y_square_min = max(0, y_square_min)
        x_square_max = min(self.image.shape[1], x_square_max)
        y_square_max = min(self.image.shape[0], y_square_max)

        # Convert to integer indices
        x_square_min_int = int(x_square_min)
        y_square_min_int = int(y_square_min)
        x_square_max_int = int(x_square_max)
        y_square_max_int = int(y_square_max)

        # Ensure indices are valid
        if x_square_min_int >= x_square_max_int or y_square_min_int >= y_square_max_int:
            print("Invalid square dimensions, skipping.")
            self.process_next_image()
            return

        # Crop the square from the image
        square_image = self.image[y_square_min_int:y_square_max_int, x_square_min_int:x_square_max_int]

        # Adjust the click coordinates to be relative to the square image
        click_x_in_square = image_x - x_square_min
        click_y_in_square = image_y - y_square_min

        # Now, save the square image and the coordinates
        self.save_image_and_coordinates(square_image, click_x_in_square, click_y_in_square, found=1)

        # Move to next image
        self.process_next_image()

    def cannot_find(self):
        # When the user indicates that the center cannot be found, save the state with "found = 0"
        self.save_image_and_coordinates(None, -1, -1, found=0)
        self.process_next_image()

    def save_image_and_coordinates(self, square_image, click_x_in_square, click_y_in_square, found):
        # Create a directory to save the data if it doesn't already exist
        save_folder = 'data/dislocation_center_finding_data'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # If square_image is provided, save it
        if square_image is not None and square_image.size > 0:
            save_image_path = os.path.join(save_folder, f'{os.path.splitext(self.current_image_file)[0]}_square.tif')

            # Normalize the image data to 0-255 and convert to uint8
            min_val = np.min(square_image)
            max_val = np.max(square_image)
            if max_val > min_val:
                square_image_norm = (square_image - min_val) / (max_val - min_val)
                square_image_uint8 = (square_image_norm * 255).astype(np.uint8)
            else:
                # Handle the case where the image has a constant value
                square_image_uint8 = np.zeros_like(square_image, dtype=np.uint8)

            # Save the square image using tifffile
            tiff.imwrite(save_image_path, square_image_uint8)
            print(f"Saved image: {save_image_path}")
        else:
            print(f"Error: Invalid or empty square image for {self.current_image_file}")

        # Save the coordinates and the "found" status
        save_coords_path = os.path.join(save_folder, f'{os.path.splitext(self.current_image_file)[0]}_coords.txt')
        with open(save_coords_path, 'w') as f:
            f.write(f'{click_x_in_square} {click_y_in_square} {found}')
            print(f"Saved coordinates: {save_coords_path}")

    def skip_image(self):
        self.process_next_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

