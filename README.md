# Histopathologic Cancer Detection

Binary classification of 96x96 histopathology patches — detect metastatic cancer in tissue scans.

Members: Vache Harutyunyan, Mari Hayrapetyan, Liza Baroyan, Erik Shahbazyan

**Models:** Custom CNN, ResNet-34, EfficientNet-B0, DenseNet-121
**Dataset:** [PatchCamelyon (Kaggle)](https://www.kaggle.com/competitions/histopathologic-cancer-detection)

---

## Overview

This project tackles the PatchCamelyon (PCam) Kaggle competition — a binary image classification task where the goal is to identify metastatic cancer in small patches of histopathology tissue scans.

Each image is a **96×96 RGB patch** extracted from a whole-slide image of lymph node tissue. A patch is labeled **positive (cancer)** if the center 32×32 pixel region contains at least one pixel of tumor tissue. The dataset contains over **220,000 labeled images**, making it a large-scale benchmark for medical image classification.

The clinical motivation is significant: manual inspection of tissue slides is time-consuming and error-prone. An automated, high-accuracy classifier can assist pathologists in prioritizing and verifying cases.

---

## Approach

We trained and compared **four deep learning architectures** of increasing complexity:

- **Custom CNN** — lightweight baseline built from scratch with 3 convolutional blocks and batch normalization
- **ResNet-34** — residual network pretrained on ImageNet, fully fine-tuned
- **EfficientNet-B0** — compound-scaled efficient architecture, pretrained on ImageNet
- **DenseNet-121** — densely connected network with feature reuse, pretrained on ImageNet

All models use **binary cross-entropy loss** with a sigmoid output, **AdamW optimizer**, and a **cosine annealing scheduler**. Training includes data augmentation (random flips, rotation, color jitter) and early stopping based on validation AUC.

The primary evaluation metric is **AUC-ROC**, chosen because it measures discrimination ability independently of classification threshold.

---
## Results

| Model | AUC | Accuracy |
|---|---|---|
| Custom CNN | 0.9572 | 88.39% |
| ResNet-34 | 0.9949 | 97.21% |
| EfficientNet-B0 | 0.9956 | 97.41% |
| DenseNet-121 | **0.9967** | **97.85%** |

**Best model: DenseNet-121** — Precision: 0.98 · Recall: 0.98 · F1: 0.98

---

## Training History

### Custom CNN

| Epoch | T-Loss | V-Loss | V-Acc | V-AUC |
|---|---|---|---|---|
| 1 | 0.4685 | 0.4305 | 0.7984 | 0.8960 |
| 2 | 0.4288 | 0.3744 | 0.8339 | 0.9173 |
| 3 | 0.3987 | 0.3385 | 0.8499 | 0.9302 |
| 4 | 0.3723 | 0.3092 | 0.8690 | 0.9408 |
| 5 | 0.3526 | 0.2962 | 0.8754 | 0.9472 |
| 6 | 0.3360 | 0.2762 | 0.8816 | 0.9556 |
| 7 | 0.3211 | 0.2735 | 0.8842 | 0.9562 |
| 8 | 0.3129 | 0.2784 | 0.8818 | 0.9562 |
| 9 | 0.3056 | 0.2730 | 0.8839 | 0.9572 |
| 10 | 0.3041 | 0.2877 | 0.8763 | 0.9550 |

### ResNet-34

| Epoch | T-Loss | V-Loss | V-Acc | V-AUC |
|---|---|---|---|---|
| 1 | 0.2336 | 0.1582 | 0.9419 | 0.9828 |
| 2 | 0.1776 | 0.1397 | 0.9490 | 0.9858 |
| 3 | 0.1563 | 0.1320 | 0.9509 | 0.9891 |
| 4 | 0.1399 | 0.1155 | 0.9598 | 0.9907 |
| 5 | 0.1267 | 0.1045 | 0.9636 | 0.9920 |
| 6 | 0.1132 | 0.0947 | 0.9665 | 0.9933 |
| 7 | 0.1013 | 0.0916 | 0.9690 | 0.9940 |
| 8 | 0.0916 | 0.0861 | 0.9697 | 0.9944 |
| 9 | 0.0835 | 0.0820 | 0.9717 | 0.9949 |
| 10 | 0.0792 | 0.0822 | 0.9721 | 0.9949|


### EfficientNet-B0

| Epoch | T-Loss | V-Loss | V-Acc | V-AUC |
|---|---|---|---|---|
| 1 | 0.2434 | 0.1395 | 0.9502 | 0.9863 |
| 2 | 0.1683 | 0.1155 | 0.9587 | 0.9906 |
| 3 | 0.1423 | 0.1011 | 0.9647 | 0.9926 |
| 4 | 0.1254 | 0.0928 | 0.9677 | 0.9936 |
| 5 | 0.1117 | 0.0884 | 0.9694 | 0.9941 |
| 6 | 0.1009 | 0.0857 | 0.9714 | 0.9945 |
| 7 | 0.0933 | 0.0788 | 0.9732 | 0.9953 |
| 8 | 0.0857 | 0.0773 | 0.9740 | 0.9955 |
| 9 | 0.0810 | 0.0775 | 0.9745 | 0.9955 |
| 10 | 0.0789 | 0.0766 | 0.9741 | 0.9956|

### DenseNet-121

| Epoch | T-Loss | V-Loss | V-Acc | V-AUC |
|---|---|---|---|---|
| 1 | 0.2103 | 0.1438 | 0.9484 | 0.9867 |
| 2 | 0.1507 | 0.1135 | 0.9614 | 0.9910 |
| 3 | 0.1292 | 0.0997 | 0.9648 | 0.9928 |
| 4 | 0.1144 | 0.0981 | 0.9645 | 0.9937 |
| 5 | 0.0997 | 0.0869 | 0.9700 | 0.9946 |
| 6 | 0.0878 | 0.0767 | 0.9741 | 0.9955 |
| 7 | 0.0771 | 0.0739 | 0.9747 | 0.9958 |
| 8 | 0.0664 | 0.0784 | 0.9741 | 0.9961 |
| 9 | 0.0601 | 0.0683 | 0.9780 | 0.9965 |
| 10 | 0.0550 | 0.0668 | 0.9785 | **0.9967** ✓ |

---

## Project Structure

```
├── config/
│   └── config.yaml          # all hyperparameters & paths
├── src/
│   ├── __init__.py           # config loader (CFG singleton)
│   ├── dataset.py            # CancerDataset, TestDataset, loaders
│   ├── model.py              # model architectures & registry
│   ├── train.py              # training loop with early stopping & logging
│   ├── evaluate.py           # AUROC, accuracy, confusion matrix, plots
│   └── predict.py            # single-image & batch inference
├── scripts/
│   ├── train.sh              # launch training
│   ├── eval.sh               # run evaluation
│   └── run_app.sh            # launch Gradio web UI
├── app/
│   └── app.py                # Gradio web interface
├── data/                     # dataset (download from Kaggle)
│   ├── train/                # training .tif images
│   ├── test/                 # test .tif images
│   └── train_labels.csv
├── checkpoints/              # trained model weights (.pt)
├── logs/                     # training logs
├── notebook/                 # EDA notebook with data distribution before training
├── outputs/                  # plots & submission CSVs
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it in `data/`.

## Usage

```bash
# Evaluate trained models on validation set
bash scripts/eval.sh

# Launch Gradio web UI
bash scripts/run_app.sh

# Train from scratch (not needed — if checkpoints included)

#You can find the checkpoints in this Google Drive link below:
# https://drive.google.com/drive/folders/1XYd7Zcw2ZAqQl61ZpDIeyU1CM013iS7w?usp=drive_link
bash scripts/train.sh
```

## Checkpoints

| Model | File |
|---|---|
| CustomCNN | `checkpoints/Custom_CNN.pt` |
| ResNet-34 | `checkpoints/ResNet_34.pt` |
| EfficientNet-B0 | `checkpoints/EfficientNet_B0.pt` |
| DenseNet-121 | `checkpoints/DenseNet_121.pt` |
