# Improving Object Detection Robustness via Adversarial Weather Simulation

## Overview
This project investigates improving the robustness of object detection models under adverse weather conditions using adversarial fog simulation. We implement a physics-based fog model and integrate it into a min–max optimization framework to train detectors that are resilient to challenging visibility conditions.

Currently, we are focusing on establishing a strong baseline by fine-tuning a DETR (ResNet-50) object detection model on the KITTI Tracking dataset.

---

## Project Structure
```
├── detr-finetuning
│   ├── src/
│   │  ├── train.py # Training + validation loop
│   │  ├── dataset.py # KITTI dataset loader
│   │
│   ├── test_scripts/
│   │  ├── test_setup.py
│   │  ├── inference.py 
│   │
│   ├── submit_detr.sh # job submission script
│
└── README.md
```
The CNN based model can be found in a seperate github, and will be implemented and combined soon. 
Link: https://github.com/MemoOV2002/CNN---Improving-Object-Detection-Robustness-via-Adversarial-Weather-Simulation-/tree/main

---

## Dataset

We are using the KITTI Tracking dataset.

### Data Split
- Train: sequences 0–12  
- Validation: sequences 13–16  
- Test: sequences 17–20  

This split is done at the sequence level to prevent temporal leakage.

### Preprocessing
- Images resized to 640 × 384
- KITTI labels converted to COCO format
- Classes used:
  - Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc

---

## Model

We use DETR (Detection Transformer) with a ResNet-50 backbone via HuggingFace Transformers.

### Training Details
- Optimizer: AdamW  
- Learning rate:
  - 1e-4 (transformer)
  - 1e-5 (backbone)  
- Batch size: 4  
- Gradient clipping: 0.1  
- Input size: 640 × 384

---
## Current Status
- DETR and Resnet fine-tuned on KITTI
- Training and validation pipeline complete
- Inference and visualization working
## In Progress
- Finalization of training models
- Adversarial fog simulation
- Min–max optimization over fog parameter β
- Compare robustness across DETR and CNN-based detectors

---
## Contributors
David DeAcereto
Guillermo Ortega Vallejo
Serhat Tay
Michelle Zhang

