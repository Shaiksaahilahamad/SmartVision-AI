# SmartVision AI - Intelligent Multi-Class Object Recognition System

## Live Demo
🔗 Hugging Face Deployment:
https://saahil292-smartvision-ai.hf.space

---

# Project Overview
SmartVision AI is a Computer Vision and Deep Learning project used for:

- Image Classification
- Object Detection
- Multi-object Recognition

The system uses Transfer Learning models and YOLOv8 to detect and classify objects from images in real time.

---

# Technologies Used

- Python
- PyTorch
- TensorFlow
- OpenCV
- Streamlit
- YOLOv8
- VGG16
- ResNet50
- MobileNetV2
- EfficientNetB0
- Hugging Face Spaces

---

# Dataset Used

Dataset Name:
COCO 2017 Dataset (25 Class Subset)

Total Classes:
25

Total Images:
2500 Images

Dataset Source:
https://cocodataset.org/

---

# Selected Classes

- person
- bicycle
- car
- motorcycle
- airplane
- bus
- truck
- traffic light
- stop sign
- bench
- dog
- cat
- horse
- bird
- cow
- elephant
- bottle
- cup
- bowl
- pizza
- cake
- chair
- couch
- bed
- potted plant

---

# Project Workflow

## Phase 1: Data Preprocessing

- Load COCO Dataset
- Filter 25 classes
- Resize images
- Normalize images
- Create train/validation/test split
- Apply data augmentation

---

## Phase 2: Image Classification

### Models Used

1. VGG16
2. ResNet50
3. MobileNetV2
4. EfficientNetB0

### Expected Accuracy

| Model | Accuracy |
|-------|----------|
| VGG16 | 80-85% |
| ResNet50 | 85-90% |
| MobileNetV2 | 82-87% |
| EfficientNetB0 | 88-93% |

---

# Phase 3: Object Detection

YOLOv8 is used for:

- Multi-object detection
- Bounding box prediction
- Real-time object detection

### Expected Performance

- mAP@0.5: 85-90%
- FPS: 30-50
- Processing Time: <2 seconds

---

# Streamlit Application Pages

1. Home Page
2. Image Classification Page
3. Object Detection Page
4. Model Performance Dashboard
5. Live Webcam Detection
6. About Page

---

# Deployment

Deployment Platform:
Hugging Face Spaces

Deployment Link:
https://saahil292-smartvision-ai.hf.space

Deployment Steps:

1. Create GitHub Repository
2. Upload Project Files
3. Add requirements.txt
4. Connect Hugging Face Space
5. Deploy Streamlit App

---

# Business Use Cases

- Smart Traffic Monitoring
- Retail Analytics
- Security Surveillance
- Wildlife Monitoring
- Healthcare Monitoring
- Smart Home Automation
- Agriculture Monitoring
- Warehouse Management

---

# Project Deliverables

- Jupyter Notebooks
- Trained Models
- YOLO Model
- Streamlit Web App
- Documentation
- Deployment Link

---

# Conclusion

SmartVision AI is an intelligent computer vision system capable of:

- Detecting multiple objects
- Classifying images accurately
- Performing real-time predictions
- Deploying on cloud platforms

The project combines Deep Learning, Transfer Learning, and YOLOv8 to create a scalable AI-powered object recognition system.
