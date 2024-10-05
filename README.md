# Plant Detection Project for Johnsongrass

## Overview

This project focuses on developing and evaluating a plant detection model specifically for **Johnsongrass**, an invasive species known for its aggressive growth and ability to outcompete native vegetation. The project includes scripts for data preprocessing, model training, and performance evaluation.

## Repository Structure

- `EDA.ipynb`: Jupyter notebook for dataset preprocessing and visualization
- `evaluation.ipynb`: Script for model evaluation and performance analysis
- `split.py`: Script for splitting the dataset into training and validation sets
- `training.ipynb`: Jupyter notebook for model configuration and training

## Detailed Description

### EDA.ipynb

This notebook contains functions for preprocessing and visualizing an image dataset with bounding box annotations. It's designed to convert annotations from a CSV file into YOLO format and prepare the dataset for object detection tasks.

Key functions:
- `extract_boxes(row_data)`: Extracts bounding box coordinates from a row of data.
- `display_image_with_boxes(row_number, df, image_folder)`: Displays an image with its corresponding bounding boxes.
- `clean_data(df, image_folder, output_folder="dataset")`: Main function for preprocessing the dataset.

### evaluation.ipynb

This notebook is responsible for model evaluation and performance analysis. It includes:

1. Model and data configuration
2. Helper function to check for plant detection
3. Evaluation process
4. Confusion matrix calculation
5. Results visualization

### split.py

This script splits the dataset into training and validation sets for machine learning tasks. It works with datasets organized in the YOLO format.

Features:
- Splits dataset into train and validation sets based on a specified ratio
- Maintains correspondence between images and their label files
- Creates separate train and validation folders for both images and labels
- Randomly shuffles the dataset before splitting

### training.ipynb

This notebook is used for model configuration and training. It includes:

- Model Initialization: Initializing the YOLOv8x model for transfer learning.
- Hyperparameter Configuration: The training process includes modifications to various hyperparameters such as epochs, batch size, image size, confidence threshold, IOU threshold, learning rates, data augmentation settings, and more. These adjustments are made to optimize the model's performance for Johnsongrass detection.

## Importance of Johnsongrass Detection

Johnsongrass is categorized as a noxious weed in many regions due to its ability to form dense colonies that restrict crop growth and outcompete native plant life. Its rapid spread through rhizomes makes it difficult to control, leading to significant ecological and economic impacts.

## Usage

1. Start with `EDA.ipynb` to preprocess your dataset.
2. Use `split.py` to divide your dataset into training and validation sets.
3. Configure and run the training process using `training.ipynb`.
4. Evaluate the model's performance with `evaluation.py`.
