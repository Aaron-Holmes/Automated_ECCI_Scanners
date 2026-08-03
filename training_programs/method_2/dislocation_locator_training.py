#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split

# Load the true bounds data
def load_true_bounds(file_path):
    data = pd.read_csv(file_path, header=None, names=["image_path", "minx", "miny", "maxx", "maxy"])
    return data

# Augment the data with 8 unique transformations (flips and rotations)
def augment_data(images, bounds):
    augmented_images = []
    augmented_bounds = []
    transformation_labels = []

    # Define each transformation explicitly with the corresponding bounding box transformation
    transformations = [
        # Original (no transformation)
        ("Original", lambda img, b: (img, b)),
        
        # Horizontal Flip (flip along the vertical axis)
        ("Horizontal Flip", lambda img, b: (
            cv2.flip(img, 1),  
            (60 - b[2], b[1], 60 - b[0], b[3])  
        )),

        # Vertical Flip (flip along the horizontal axis)
        ("Vertical Flip", lambda img, b: (
            cv2.flip(img, 0),  
            (b[0], 60 - b[3], b[2], 60 - b[1])  
        )),

        # Horizontal + Vertical Flip (equivalent to a 180° rotation)
        ("Horizontal+Vertical Flip (180° Rotation)", lambda img, b: (
            cv2.flip(img, -1),  
            (60 - b[2], 60 - b[3], 60 - b[0], 60 - b[1])  
        )),

        # 90° Rotation (clockwise)
        ("90° Rotation", lambda img, b: (
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),  
            (60 - b[3], b[0], 60 - b[1], b[2])  
        )),

        # 90° Rotation + Horizontal Flip
        ("90° Rotation + Horizontal Flip", lambda img, b: (
            cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1),  
            (b[1], b[0], b[3], b[2])  # FIXED: Maps to (y, x)
        )),

        # 270° Rotation (equivalent to 90° counterclockwise)
        ("270° Rotation", lambda img, b: (
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),  
            (b[1], 60 - b[2], b[3], 60 - b[0])  
        )),

        # 270° Rotation + Horizontal Flip
        ("270° Rotation + Horizontal Flip", lambda img, b: (
            cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1),  
            (60 - b[3], 60 - b[2], 60 - b[1], 60 - b[0])  
        )),
    ]

    # Apply each transformation explicitly
    for img, b in zip(images, bounds):
        for label, transform in transformations:
            augmented_img, augmented_b = transform(img.copy(), list(b))  
            augmented_images.append(augmented_img)
            augmented_bounds.append(augmented_b)
            transformation_labels.append(label)

    return augmented_images, augmented_bounds, transformation_labels


# Prepare the data for training
def prepare_data(data):
    images = []
    bounds = []
    for index, row in data.iterrows():
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is not None and img.shape == (60, 60):  
            images.append(img)
            bounds.append((row["minx"], row["miny"], row["maxx"], row["maxy"]))

    return np.array(images), np.array(bounds)

# Visualize the augmented images with bounding boxes in a 3x3 grid with labels
def visualize_augmentations(augmented_images, augmented_bounds, transformation_labels, num_images_to_show=4):
    sets_per_image = 8  # Each image has 8 unique augmentations

    for i in range(num_images_to_show):
        plt.figure(figsize=(10, 10))
        for j in range(sets_per_image):
            img_idx = i * sets_per_image + j
            img = augmented_images[img_idx].copy()
            bounds = augmented_bounds[img_idx]
            label = transformation_labels[img_idx]

            # Draw bounding box on the image
            cv2.rectangle(img, (int(bounds[0]), int(bounds[1])),
                          (int(bounds[2]), int(bounds[3])), (255, 0, 0), 1)

            # Plot each augmented image in a 3x3 grid
            plt.subplot(3, 3, j + 1)
            plt.imshow(img, cmap='gray')
            plt.title(label)
            plt.axis('off')

        plt.show()  

# Create the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(60, 60, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='linear', kernel_initializer='he_normal', bias_initializer=Constant([30, 30, 30, 30]))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    
# Function to calculate the IoU (Intersection over Union) between two bounding boxes
def calculate_iou(true_box, pred_box):
    xA = max(true_box[0], pred_box[0])
    yA = max(true_box[1], pred_box[1])
    xB = min(true_box[2], pred_box[2])
    yB = min(true_box[3], pred_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
    boxBArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# Visualize validation set with both ground truth and model-predicted bounding boxes, and IoU (confidence score)
def visualize_validation_predictions(images, true_bounds, model, num_images_to_show=25):
    predicted_bounds = model.predict(images)  

    num_images = min(num_images_to_show, len(images))
    plt.figure(figsize=(12, 12))  

    for i in range(num_images):
        img = images[i].reshape(60, 60)  
        true_b = true_bounds[i]
        pred_b = predicted_bounds[i]

        iou_score = calculate_iou(true_b, pred_b)

        plt.subplot(5, 5, i + 1)  
        plt.imshow(img, cmap='gray')
        
        ax = plt.gca()

        # True bounding box (white)
        true_rect = patches.Rectangle((true_b[0], true_b[1]), true_b[2] - true_b[0], true_b[3] - true_b[1],
                                      linewidth=1, edgecolor='white', facecolor='none')
        ax.add_patch(true_rect)

        # Predicted bounding box (black)
        pred_rect = patches.Rectangle((pred_b[0], pred_b[1]), pred_b[2] - pred_b[0], pred_b[3] - pred_b[1],
                                      linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(pred_rect)

        plt.text(1, 5, f'IoU: {iou_score:.2f}', color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        plt.axis('off')  

    plt.tight_layout()
    plt.show()  

# Main function
def main():
    # Flag to control whether to perform augmentation
    perform_augmentation = True
    see_augmentations = False

    # UPDATE THIS: Ensure the path is relative or points to the correct active directory
    data_path = "../../data/training_data/method_2_training_data/locator_training data/hand_picked_location.txt"
    data = load_true_bounds(data_path)

    # Prepare the data
    images, bounds = prepare_data(data)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, bounds, test_size=0.1, random_state=42)

    # Perform data augmentation on the training data
    if perform_augmentation:
        augmented_X_train, augmented_y_train, transformation_labels = augment_data(X_train, y_train)
    else:
        augmented_X_train, augmented_y_train = X_train, y_train

    # Visualize the augmentations before training
    if see_augmentations:
        visualize_augmentations(augmented_X_train, augmented_y_train, transformation_labels)

    # Reshape for the CNN model
    X_train = np.array(augmented_X_train).reshape(-1, 60, 60, 1)
    y_train = np.array(augmented_y_train)

    X_val = X_val.reshape(-1, 60, 60, 1)
    y_val = np.array(y_val)

    # Create and train the model
    model = create_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=32)

    # Save the model
    model.save('bounding_box_model_gh_e1000_bs32.h5')
    print("Model saved as 'bounding_box_model_gh_e1000_bs32.h5'")

    # Visualize validation set with true and predicted bounding boxes
    visualize_validation_predictions(X_val, y_val, model, num_images_to_show=25)

if __name__ == "__main__":
    main()