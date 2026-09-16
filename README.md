# Sofa Project – YOLO Segmentation Model Training

## Overview

This repository contains the YOLO segmentation model training stage of an AI-based Sofa Cost Estimation project.

The purpose of this stage is to detect and segment the major sofa components from an input sofa image.

The segmentation output will later be used for:

- Sofa shape and design analysis
- Geometry analysis
- Engineering calculations
- Customer-provided dimensions
- Parametric CAD generation
- Bill of Materials (BOM)
- Cost estimation

## Current Model

- Model: YOLO11n-Seg
- Task: Instance Segmentation
- Number of classes: 6
- Training epochs: 100
- Image size: 640 × 640
- Batch size: 4
- Device: CPU
- Training images: 536
- Validation images: 144

## Sofa Components

The model detects the following six components:

1. `back_cushion`
2. `base`
3. `left_arm`
4. `legs`
5. `right_arm`
6. `seat_cushion`

## Dataset

The dataset was cleaned, repaired and verified before training.

Final dataset:

- Training images: 536
- Validation images: 144
- Training polygons: 3800
- Validation polygons: 1080
- Annotation errors after cleaning: 0

The unwanted sofa-level classes were removed from the training annotations so that the model focuses on the required sofa components.

## Training Environment

Training was performed using:

- CPU: AMD Ryzen 5 7520U
- RAM: 16 GB
- Python: 3.13.13
- Ultralytics: 8.4.135
- PyTorch: 2.13.0+cpu
- Operating System: Windows 11

The model was trained locally using VS Code and Ultralytics YOLO.

## Final Validation Results

The final validation results obtained from the best model were:

| Metric | Result |
|---|---:|
| Box Precision | 88.2% |
| Box Recall | 87.1% |
| Box mAP50 | 90.3% |
| Box mAP50-95 | 70.2% |
| Mask Precision | 89.0% |
| Mask Recall | 81.2% |
| Mask mAP50 | 85.9% |
| Mask mAP50-95 | 65.1% |

### Class-wise Results

| Class | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| back_cushion | 93.3% | 88.6% | 92.5% | 69.2% |
| base | 84.3% | 83.5% | 86.6% | 72.4% |
| left_arm | 89.8% | 86.2% | 90.9% | 72.7% |
| legs | 79.6% | 54.8% | 65.5% | 32.5% |
| right_arm | 93.8% | 79.7% | 86.7% | 61.3% |
| seat_cushion | 93.2% | 94.4% | 93.4% | 82.3% |

The `legs` class showed weaker segmentation performance compared with the other components and requires further testing and improvement.

## Model Files

The trained model checkpoints are stored using Git LFS.

```text
runs/
└── segment/
    └── models/
        └── sofa_yolo11n_seg_v1-2/
            └── weights/
                ├── best.pt
                └── last.pt