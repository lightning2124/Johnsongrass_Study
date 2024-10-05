"""
Dataset Splitting Script for Machine Learning

This script splits a dataset into training and validation sets for machine learning tasks.
It is designed to work with datasets organized in the YOLO format, with separate 'images' and 'labels' folders.

Features:
- Splits dataset into train and validation sets based on a specified ratio
- Maintains the correspondence between images and their label files
- Creates separate train and validation folders for both images and labels
- Randomly shuffles the dataset before splitting to ensure unbiased distribution

Usage:
Modify the 'dataset_folder' and 'train_ratio' variables at the bottom of the script,
then run the script using: python script_name.py

Note: This script assumes that the dataset is organized with 'images' and 'labels' 
folders in the root directory, and that image and label files have corresponding names.
"""

import os
import random
import shutil

def split_dataset(dataset_folder, train_ratio=0.8):
    """
    Splits a YOLO dataset into train and validation sets.

    Args:
        dataset_folder: Path to the root dataset folder (containing "images" and "labels").
        train_ratio: Ratio of the dataset to be used for training (default 0.8).
    """
    

    # Check if the dataset folder exists
    if not os.path.exists(dataset_folder):
        raise ValueError(f"Dataset folder not found: {dataset_folder}")

    # Create train and validation folders within "images" and "labels"
    images_train_folder = os.path.join(dataset_folder, "images", "train")
    images_val_folder = os.path.join(dataset_folder, "images", "val")
    labels_train_folder = os.path.join(dataset_folder, "labels", "train")
    labels_val_folder = os.path.join(dataset_folder, "labels", "val")

    os.makedirs(images_train_folder, exist_ok=True)
    os.makedirs(images_val_folder, exist_ok=True)
    os.makedirs(labels_train_folder, exist_ok=True)
    os.makedirs(labels_val_folder, exist_ok=True)

    # Get image and label filenames
    images_folder = os.path.join(dataset_folder, "images")
    filenames = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))]

    # Shuffle the filenames
    random.shuffle(filenames)


    # Split into train and validation sets
    train_size = int(len(filenames) * train_ratio)

    # Move images and labels to the respective folders
    for i, filename in enumerate(filenames):
        if i < train_size:  # Check if it belongs to the training set
            target_image_folder = images_train_folder
            target_label_folder = labels_train_folder
        else:  # Otherwise, it belongs to the validation set
            target_image_folder = images_val_folder
            target_label_folder = labels_val_folder

        image_path = os.path.join(images_folder, filename)
        label_path = os.path.join(os.path.join(dataset_folder, "labels"), filename[:-4] + '.txt')
        shutil.move(image_path, os.path.join(target_image_folder, filename))
        shutil.move(label_path, os.path.join(target_label_folder, filename[:-4] + '.txt'))


if __name__ == "__main__":
    dataset_folder = "sortedDataset"  # Replace with your dataset folder path
    train_ratio = 0.8  # Adjust as needed (e.g., 0.7 for 70% train)
    split_dataset(dataset_folder, train_ratio)