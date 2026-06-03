# SmartVision AI - Intelligent Multi-Class Object Recognition System

## Project Overview

SmartVision AI is a Computer Vision and Deep Learning project developed to perform both Image Classification and Object Detection using state-of-the-art deep learning models.

The system can identify and locate multiple objects from 25 different categories selected from the COCO Dataset. It combines Transfer Learning-based CNN models for classification and YOLOv8 for real-time object detection.

The final solution is deployed as a web application using Streamlit and Hugging Face Spaces, making it accessible through a browser without requiring local installation.

### Live Demo

https://saahil292-smartvision-ai.hf.space

---

# Problem Statement

Many industries require intelligent systems capable of automatically identifying objects in images and videos.

Traditional manual monitoring is slow, expensive, and prone to human error.

The goal of this project is to build a smart vision system capable of:

* Detecting multiple objects in an image
* Classifying objects accurately
* Providing real-time predictions
* Working across multiple domains
* Deploying on cloud platforms

---

# Project Objectives

1. Build an image classification system using Transfer Learning.
2. Train and compare multiple CNN architectures.
3. Build a YOLOv8 object detection model.
4. Compare classification and detection performance.
5. Create an interactive Streamlit web application.
6. Deploy the application on Hugging Face Spaces.

---

# Dataset

## Dataset Used

COCO (Common Objects in Context) Dataset

The COCO dataset is one of the most popular datasets used in Computer Vision research.

Originally:

* 122,000+ Images
* 80 Categories
* Millions of Object Annotations

For this project, we created a custom subset containing:

* 25 Classes
* 100 Images per Class
* Total Dataset Size: 2500 Images

---

# Selected Classes

### Vehicles

* Car
* Truck
* Bus
* Motorcycle
* Bicycle
* Airplane

### Person

* Person

### Outdoor Objects

* Traffic Light
* Stop Sign
* Bench

### Animals

* Dog
* Cat
* Horse
* Bird
* Cow
* Elephant

### Food & Kitchen

* Bottle
* Cup
* Bowl
* Pizza
* Cake

### Indoor Objects

* Chair
* Couch
* Bed
* Potted Plant

---

# Project Architecture

Input Image

↓

Preprocessing

↓

Classification Models

(VGG16, ResNet50, MobileNetV2, EfficientNetB0)

↓

YOLOv8 Object Detection

↓

Prediction Results

↓

Streamlit Dashboard

↓

Hugging Face Deployment

---

# Phase 1: Data Collection and Preprocessing

## Dataset Loading

The dataset was loaded from Hugging Face using streaming mode.

Streaming allows downloading only required samples without downloading the entire COCO dataset.

Benefits:

* Faster processing
* Lower storage usage
* Efficient data collection

---

## Data Cleaning

The collected images were checked for:

* Corrupted files
* Invalid annotations
* Duplicate images
* Missing labels

---

## Data Preprocessing

For Classification:

1. Extract bounding boxes.
2. Crop individual objects.
3. Resize images to 224×224.
4. Normalize pixel values.
5. Organize into class folders.

For Detection:

1. Convert annotations into YOLO format.
2. Create image-label structure.
3. Generate data.yaml file.

---

## Data Augmentation

To improve generalization, augmentation techniques were applied:

* Horizontal Flip
* Rotation
* Zoom
* Brightness Adjustment
* Contrast Adjustment
* Color Jitter

Benefits:

* Prevents overfitting
* Improves model robustness
* Increases dataset diversity

---

# Phase 2: Image Classification Using Transfer Learning

## What is Transfer Learning?

Transfer Learning means using a pre-trained model that has already learned features from millions of images.

Instead of training from scratch, we reuse learned knowledge and fine-tune it for our custom dataset.

Advantages:

* Faster Training
* Higher Accuracy
* Less Data Required

---

# Model 1: VGG16

## Why VGG16?

* Simple architecture
* Easy to understand
* Excellent feature extraction

### Architecture

Input Image

↓

13 Convolution Layers

↓

5 Max Pooling Layers

↓

Fully Connected Layers

↓

Softmax Output Layer

### Results

* Good accuracy
* Larger model size
* Slower inference

---

# Model 2: ResNet50

## Why ResNet50?

Deep networks suffer from vanishing gradients.

ResNet introduces Skip Connections.

### Skip Connection

Output = Input + Learned Features

Benefits:

* Solves vanishing gradient problem
* Enables deeper networks
* Better feature learning

### Results

* Higher accuracy than VGG16
* Better feature extraction

---

# Model 3: MobileNetV2

## Why MobileNetV2?

Designed for mobile and edge devices.

Uses:

Depthwise Separable Convolution

Instead of:

Standard Convolution

Benefits:

* Faster inference
* Smaller model size
* Less memory usage

### Results

* Fastest classification model
* Lightweight architecture

---

# Model 4: EfficientNetB0

## Why EfficientNet?

Uses Compound Scaling.

Scales:

* Width
* Depth
* Resolution

Together.

Benefits:

* Better accuracy
* Efficient computation

### Results

* Best overall classification performance
* Best accuracy-to-size ratio

---

# Hyperparameter Tuning

Several hyperparameters were optimized:

### Learning Rate

Controls how much weights are updated.

### Batch Size

Number of images processed before updating weights.

### Epochs

Number of complete passes through the dataset.

### Dropout

Randomly disables neurons during training.

Purpose:

* Prevent overfitting
* Improve generalization

### Optimizer

Adam Optimizer was used.

Functions:

* Updates model weights
* Minimizes loss
* Speeds up convergence

---

# Evaluation Metrics

## Accuracy

Measures correct predictions.

Accuracy = Correct Predictions / Total Predictions

---

## Precision

Measures prediction quality.

Precision = TP / (TP + FP)

---

## Recall

Measures ability to find all objects.

Recall = TP / (TP + FN)

---

## F1 Score

Balance between Precision and Recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

# Phase 3: YOLOv8 Object Detection

## What is YOLO?

YOLO stands for:

You Only Look Once

Unlike traditional detectors:

YOLO detects objects in a single forward pass.

Benefits:

* Real-time detection
* High accuracy
* Fast inference

---

# YOLOv8 Workflow

Input Image

↓

Feature Extraction

↓

Bounding Box Prediction

↓

Class Prediction

↓

Confidence Score

↓

Non-Maximum Suppression

↓

Final Detection

---

# Non-Maximum Suppression (NMS)

Sometimes multiple boxes are predicted for the same object.

NMS removes duplicate boxes and keeps the highest confidence prediction.

---

# YOLO Performance

Metrics Used:

* mAP@0.5
* Precision
* Recall
* Confidence Score

YOLO achieved strong object detection performance across the selected 25 classes.

---

# Streamlit Application

The project was developed as a multi-page Streamlit application.

## Features

### Home Page

* Project overview
* Instructions

### Classification Page

* Upload image
* Predict object class
* Display confidence score

### Detection Page

* Detect multiple objects
* Draw bounding boxes

### Performance Page

* Accuracy comparison
* Model evaluation metrics

### About Page

* Project details
* Technologies used

---

# Deployment

The application was deployed using:

* Streamlit
* Hugging Face Spaces
* GitHub

Deployment Workflow:

Code

↓

GitHub Repository

↓

Hugging Face Space

↓

Public Web Application

---

# Technologies Used

* Python
* TensorFlow
* Keras
* PyTorch
* OpenCV
* YOLOv8
* Streamlit
* Hugging Face
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

# Business Applications

1. Smart Traffic Monitoring
2. Retail Analytics
3. Security Surveillance
4. Wildlife Monitoring
5. Healthcare Monitoring
6. Smart Homes
7. Agriculture
8. Logistics and Warehousing

---

# Conclusion

SmartVision AI successfully demonstrates the power of Deep Learning and Computer Vision by combining Image Classification and Object Detection into a single intelligent system.

Using Transfer Learning models (VGG16, ResNet50, MobileNetV2, EfficientNetB0) and YOLOv8, the project achieves accurate object recognition while maintaining real-time performance.

The deployment on Hugging Face Spaces makes the solution accessible, scalable, and ready for real-world applications.
