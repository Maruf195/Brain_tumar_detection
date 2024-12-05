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

# Brain MRI Segmentation using U-Net

## Introduction
This project focuses on segmenting brain tumors from MRI scans using the **U-Net architecture**. The dataset consists of MRI images with corresponding segmentation masks that mark tumor regions. The goal is to develop a robust model for semantic segmentation, achieving high accuracy and overlap between predicted and actual tumor regions.

---

## Dataset
### Description
The dataset contains pairs of:
- **MRI Scans**: Grayscale or RGB images of brain scans.
- **Segmentation Masks**: Binary masks where pixel values indicate tumor regions.


### Features
- **MRI Images**: High-resolution brain scans.
- **Masks**: Binary masks (0 for background, 1 for tumor region).

### Directory Structure
The dataset is structured as:
- **Images**: Folder containing MRI scans.
- **Masks**: Folder containing segmentation masks for each MRI scan.

### Kaggle Link 
   - https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

---

## Data Preprocessing
1. **Resizing**: Images and masks resized to 256x256 pixels.
2. **Normalization**: Pixel values scaled between 0 and 1.
3. **Mask Binarization**: Thresholded such that values > 0.5 are set to 1, others to 0.
4. **Data Splitting**: Dataset split into:
   - 90% for training
   - 5% for validation
   - 5% for testing

---

## Feature Engineering
### Data Augmentation
To improve model generalization, the following augmentations were applied:
- **Rotation**: Random rotations for orientation variability.
- **Shifts**: Horizontal and vertical translations.
- **Zoom and Shear**: Introduces scale variations.
- **Horizontal Flip**: Adds variability in left-right orientations.

---

## Model Development
### U-Net Architecture
The **U-Net** model is used for segmentation:
1. **Encoder**: Downsampling path with convolution and max-pooling layers.
2. **Bottleneck**: Captures abstract features.
3. **Decoder**: Upsampling path with transpose convolutions and skip connections.

### Loss Function and Metrics
- **Loss Function**: Dice Coefficient Loss to maximize overlap between predicted and ground truth masks.
- **Metrics**:
  - **Dice Coefficient**: Measures overlap between predicted and actual masks.
  - **IoU (Intersection over Union)**: Evaluates segmentation accuracy.
  - **Binary Accuracy**: Pixel-wise accuracy.

---

## Training
### Hyperparameters
- **Learning Rate**: `1e-4`
- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: Adam optimizer.

### Callbacks
- **Model Checkpoint**: Saves the best model based on validation loss.

---

## Results and Discussion
### Evaluation
The model is evaluated on the validation set using:
- Dice Coefficient
- IoU
- Binary Accuracy

### Visualizations
- **Loss and Accuracy Curves**: Training and validation performance.
- **Predicted Masks**: Examples of MRI images, ground truth masks, and predicted masks.

---

## Conclusion
This project demonstrates the effectiveness of the U-Net architecture for brain MRI segmentation. Future improvements could include:
- Experimenting with different architectures (e.g., Attention U-Net).
- Tuning hyperparameters for better performance.
- Using a larger or more diverse dataset.


---

## Improving Performance with Threads and Processes
### Use of Threads and Processes
To optimize data preprocessing, training, and augmentation, multithreading and multiprocessing can be employed to utilize all available CPU cores effectively. This approach minimizes bottlenecks, such as loading and augmenting images during training.

### Implementation Strategies
1. **Multithreading for Data Loading**:
   - Use TensorFlow/Keras `data_generator` with `use_multiprocessing=True` and `workers=N` (where `N` is the number of CPU cores) to parallelize data loading and augmentation.
   - Benefit: Reduces GPU idle time during training.



