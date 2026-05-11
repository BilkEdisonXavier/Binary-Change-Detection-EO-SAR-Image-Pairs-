# Binary-Change-Detection-EO-SAR-Image-Pairs-
# EO-SAR Binary Change Detection

This project detects changes between two satellite images:
- Pre-event image (before disaster/event)
- Post-event image (after disaster/event)

The model predicts:
- 1 → Changed area
- 0 → No change

Example:
- Building destroyed
- Flooded area
- Damaged roads
- Urban changes

---

# Project Goal

The goal is to train an AI model that compares:
- EO images (normal satellite images)
- SAR images (radar satellite images)

and finds changed regions automatically.

---

# Dataset Structure

Place dataset like this:

project/
│
├── dataset/
│   ├── train/
│   │   ├── pre_event/
│   │   ├── post_event/
│   │   └── target/
│   │
│   ├── val/
│   │   ├── pre_event/
│   │   ├── post_event/
│   │   └── target/
│   │
│   └── test/
│       ├── pre_event/
│       ├── post_event/
│       └── target/

---

# Install Requirements

Open terminal inside project folder.

Run:

```bash
pip install -r requirements.txt

Libraries Used:
-PyTorch
-OpenCV
-Albumentations
-segmentation_models_pytorch
-NumPy
-Matplotlib
Model Used:
U-Net + ResNet34

Why?
-Good segmentation performance
-Fast training
-Easy to understand
-Works well for satellite images
-Input to Model

The model receives:

Pre-event image
Post-event image

Both images are combined together.

Loss Function:
-BCE + Dice Loss

Why?

-BCE helps pixel classification
-Dice helps detect small changed regions

This improves segmentation quality.

Data Augmentation:

The following augmentations are used:

-Horizontal Flip
-Vertical Flip
-Rotation
-Resize
-Normalize

Purpose:

-Improve model generalization
-Reduce overfitting

Training:
The model automatically:

-loads dataset
-trains model
-validates model
-saves best checkpoint

Evaluation:
Metrics calculated:

-IoU
-Precision
-Recall
-F1 Score

Inference:
-Ground truth mask
-Predicted mask

Metrics Explanation:
1) IoU
-Measures overlap between:

2) predicted change area
-real change area
-Higher is better.

3) Precision
-Measures:
"How many predicted changes are actually correct?"

4) Recall
-Measures:
"How many real changes were detected?"

5) F1 Score
-Balance between:
 precision
 recall

Challenges in This Project:

Some difficult problems:
-Small changed regions
-Noisy SAR images
-Class imbalance
-False positives
-Illumination differences

Future Improvements:

Possible future work:
-Transformer models
-Attention U-Net
-Better augmentation
-Multi-scale learning
-Test-time augmentation

Project Structure:

project/
│
├── dataset/
├── datasets/
├── models/
├── losses/
├── utils/
├── checkpoints/
├── outputs/
│
├── train.py
├── eval.py
├── inference.py
│
├── config.yaml
├── requirements.txt
└── README.md

How This Project Works:

Step-by-step:

1)Load pre-event image
2)Load post-event image
3)Combine both images
4)Send into U-Net model
5)Predict changed pixels
6)Compare prediction with target mask
7)Update model weights
8)Save best model

Author:
-BILK EDISON X

References:
-U-Net Paper
-PyTorch Documentation
-segmentation_models_pytorch
-Albumentations
