# U-Net Model for MRI Image Segmentation

This project implements a U-Net model for **MRI image segmentation**. The model is designed to segment medical images, specifically brain MRI scans, to delineate regions of interest (e.g., tumors, lesions, or other abnormalities).

## Overview

U-Net is a convolutional neural network (CNN) architecture specifically designed for **semantic segmentation** tasks. The model consists of an encoder-decoder structure with skip connections, which allows for precise segmentation of image regions.

This implementation is tailored for MRI data, where the goal is to classify each pixel in the image as either belonging to the region of interest (e.g., a tumor) or background. The model utilizes **Dice Coefficient Loss** and **IoU** as evaluation metrics to handle class imbalance and improve segmentation accuracy.

## Features

- **U-Net Architecture**: Designed for semantic segmentation of medical images with an encoder-decoder structure and skip connections.
- **Dice Coefficient Loss**: Optimized for segmentation tasks, particularly in the presence of imbalanced datasets.
- **Evaluation Metrics**: Binary accuracy, Intersection over Union (IoU), and Dice Coefficient to assess model performance.
- **Callbacks**: Model checkpointing during training to save the best-performing model.

## Requirements

To run this project, you'll need the following Python libraries:

- `tensorflow` (2.x or later)
- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`
- `opencv-python` (for image preprocessing)
- `scikit-image` (for image processing)

You can install the required packages via `pip`:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python scikit-image
