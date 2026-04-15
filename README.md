# AFA-Trans-ICL: Self-Supervised Masked Representation Learning for Open-Set IoT Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/)

Official PyTorch implementation of the paper **"AFA-Trans-ICL: Self-Supervised Masked Representation Learning for Open-Set IoT Anomaly Detection"**.

## 🧠 Model Architecture

<p align="center">
  <img src="assets/architecture.png" alt="AFA-Trans-ICL Architecture" width="100%">
</p>

> **Figure 1:** The overall architecture of the proposed AFA-Trans-ICL framework. It consists of three synergistic stages: (1) Adaptive Feature Attention (AFA) layer, (2) Transformer-based Masked Autoencoder (MAE), and (3) Inductive Clustering Learning (ICL) decision module.

## 📝 Abstract
The rapid proliferation of the Internet of Things (IoT) has significantly expanded the attack surface for zero-day threats. Open-Set Anomaly Detection (OSAD) addresses this challenge by identifying unknown attacks using only normal training data. In this paper, we propose **AFA-Trans-ICL**, a unified self-supervised framework for robust open-set IoT anomaly detection. The framework seamlessly integrates an **Adaptive Feature Attention (AFA)** module, a **Transformer-based Masked Autoencoder (MAE)**, and an **Inductive Clustering Learning (ICL)** module to enforce compact decision boundaries in the latent space.

Experiments on the realistic Edge-IIoTset benchmark demonstrate strong performance, achieving an **AUROC of 0.9965** and an **AP of 0.9979**.

## 📁 Repository Structure
```text
AFA-Trans-ICL/
├── assets/             # Architecture diagrams and images
├── data/               # Data processing and robust scaling scripts
├── models/             # Core architecture (AFA, Transformer MAE)
├── utils/              # Joint loss function and seed settings
├── checkpoints/        # Directory for pre-trained models (.pth, .pkl)
├── train.py            # Phase 1 & 2: Pre-training and Boundary Optimization
└── test.py             # Phase 3: Zero-Day Inference and Evaluation
````

## 🚀 Quick Start

### 1 Environment Setup

Clone this repository and install the required dependencies:

```bash

git clone [https://github.com/wen12-debug/AFA-Trans-ICL.git](https://github.com/wen12-debug/AFA-Trans-ICL.git)
cd AFA-Trans-ICL
pip install -r requirements.txt
```

### 2 Dataset Preparation
The **Edge-IIoTset** dataset is officially hosted on [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications). 

Please download the dataset and extract the unified tabular data file. Ensure the file is named `DNN-Edge-IIoT-dataset.csv` and place it inside the root directory or the `data/` folder before running the training scripts.

### 3\. Training (Phase 1 & 2)

To train the model from scratch using only normal traffic data (OSAD protocol):

```bash
python train.py --data_path DNN-Edge-IIoT-dataset.csv --epochs 30 --icl_epochs 50 --mask_ratio 0.3 --lambda_val 0.001
```

*Pre-trained weights and the robust scaler will be automatically saved to the `./checkpoints/` directory.*

### 4\. Evaluation (Phase 3)

To evaluate the model against zero-day out-of-distribution attacks using the leave-one-out protocol:

```bash
python test.py --data_path DNN-Edge-IIoT-dataset.csv --save_dir ./checkpoints
```

## 📊 Core Results (Edge-IIoTset)

AFA-Trans-ICL demonstrates exceptional robustness and generalization capabilities in strictly Open-Set Anomaly Detection (OSAD) scenarios.

### 1. Overall Performance
Under a rigorous leave-one-out zero-day evaluation protocol, the full framework extracts highly discriminative normal prototypes and achieves state-of-the-art detection metrics:
- **AUROC:** 0.9965
- **Average Precision (AP):** 0.9979

### 2. Excellence in Zero-Day Network Threats
The framework exhibits near-perfect generalization against unseen, structurally distinct network-layer and volumetric attacks. By effectively capturing the intrinsic macroscopic boundaries of normal network behavior, drastic topological deviations trigger significant latent space violations. Highlighted zero-day detection results include:

| Unseen Attack Type | AUROC $\uparrow$ | AP $\uparrow$ |
| :--- | :---: | :---: |
| **DDoS (TCP)** | 0.9998 | 0.9943 |
| **DDoS (ICMP)** | 0.9972 | 0.9632 |
| **Backdoor** | 0.9954 | 0.8400 |
| **MITM** | 0.9777 | 0.7704 |
| **DDoS (UDP)** | 0.9839 | 0.6776 |

### 3. Ablation Study: The Value of Synergistic Design
Every component of AFA-Trans-ICL is critical for maintaining high performance in heterogeneous IoT environments. Removing core modules (such as the adaptive feature calibration or the masked token representation) significantly degrades the detection capability:

| Configuration | AUROC $\uparrow$ | AP $\uparrow$ |
| :--- | :---: | :---: |
| **Full Framework** | **0.9965** | **0.9979** |
| w/o Joint Loss | 0.9749 | 0.9885 |
| w/o AFA Layer | 0.9621 | 0.9657 |
| w/o Masking (Standard AE) | 0.7391 | 0.7471 |

*Note: For further details, including orthogonal detection visualizations (dual-space mapping of reconstruction error vs. latent distance) and feature-level interpretability analysis (AFA attention weights), please refer to Section V of our paper.*

## 🔗 Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@article{wen2026afa,
  title={AFA-Trans-ICL: Self-Supervised Masked Representation Learning for Open-Set IoT Anomaly Detection},
  author={Wen, Yuchen and Zhang, Yue and Sun, Kexin},
  journal={Course Assignment: IoT Security and Privacy Protection},
  year={2026}
}
```

