# 🧠 NeuroFusion 3D: A Multimodal Clinical Decision Support System for Brain Tumor Segmentation and Preoperative Visualization

NeuroFusion 3D is a research-oriented Medical AI prototype developed as a **Clinical Decision Support System (CDSS)** for brain tumor analysis. The system combines volumetric 3D MRI imaging with Clinical Text Reports using a multimodal deep learning architecture to support tumor segmentation, volumetric assessment, AI-assisted clinical interpretation, and interactive 3D visualization.

> **Disclaimer:** This project is intended exclusively for research and educational purposes. It is not approved for clinical diagnosis or medical decision-making.

---

## 🚀 Key Features

* 3D volumetric MRI preprocessing using MONAI
* 3D U-Net semantic segmentation
* Multimodal fusion of MRI and clinical reports
* DistilBERT clinical text encoder
* Interactive Streamlit dashboard
* Automated tumor volume estimation
* AI-assisted clinical report generation
* Modular research-oriented architecture

---

## 🖼️ User Interface & Visual Assets

### 1. Interactive Dashboard
The Streamlit dashboard provides real-time visualization of volumetric MRI scans, segmentation masks, tumor measurements, and AI-assisted clinical reports.
![Interactive Dashboard](docs/dashboard.png)

### 2. Segmentation Results (Raw MRI vs. Predicted Segmentation)
![Segmentation Results](docs/segmentation.png)

### 3. Training Performance & Validation Loss
![Training Performance](docs/training_curves.png)

### 4. System Architecture
![System Architecture](docs/architecture.png)

---

## 📊 Evaluation Strategy

The current repository represents the complete software architecture of the proposed system. Performance metrics will be reported after model training on an appropriate research dataset.

The segmentation model is configured to optimize a **Hybrid Loss Function**:
* Dice Loss
* Cross Entropy Loss

The following evaluation metrics are planned:
* Dice Similarity Coefficient (DSC) & Intersection over Union (IoU)
* Precision & Recall & F1-Score
* Hausdorff Distance (95HD) & Average Symmetric Surface Distance (ASSD)
* ROC-AUC (Classification) & Confusion Matrix

---

## 🧬 Target Dataset

The pipeline is designed to support publicly available multimodal brain tumor datasets such as **BraTS (2023/2024)**. Supported MRI modalities include:
* **T1** / **T1ce** (Contrast Enhanced)
* **T2** / **FLAIR** (Fluid Attenuated Inversion Recovery)

---

## 📁 Repository Structure

```text
NeuroFusion-3D/
│
├── app.py                     # Streamlit Main Dashboard
├── train.py                   # Model Training Loop Script
├── inference.py               # Evaluation & Prediction Script
├── build_model.py             # 3D U-Net Architecture Definition
├── multimodal_fusion.py       # DistilBERT Embedding Fusion Layer
├── prepare_pipeline.py        # MONAI Preprocessing Pipeline
├── utils/                     # Helper Functions
├── requirements.txt           # Project Dependencies
└── docs/                      # Visual Assets and Plots
    ├── dashboard.png
    ├── segmentation.png
    ├── architecture.png
    └── training_curves.png
---
---

# Installation

```bash
git clone https://github.com/elosely/NeuroFusion-3D.git

cd NeuroFusion-3D

pip install -r requirements.txt
```

---

# Run

```bash
python train.py

streamlit run app.py
```

---

# Future Work

- Explainable AI (Grad-CAM)
- DICOM support
- Multi-class tumor classification
- Longitudinal patient follow-up
- PDF clinical report generation
- Docker deployment
- REST API integration
- PACS compatibility
- Federated Learning
- Cloud deployment

---

# Developer

**Youssef Elosely**

Medical Biophysics Student

Faculty of Science

Focus Areas:

- Medical Imaging
- Artificial Intelligence
- Computer Vision
- Medical Physics
- Biomedical Signal & Image Processing

---

# License

This project is released under the MIT License.
