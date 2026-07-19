# 🧠 NeuroFusion 3D
### A Multimodal Clinical Decision Support System for Brain Tumor Segmentation and Preoperative Visualization

NeuroFusion 3D is a research-oriented Medical AI prototype developed as a **Clinical Decision Support System (CDSS)** for brain tumor analysis.

The system combines volumetric **3D MRI** imaging with **Clinical Text Reports** using a multimodal deep learning architecture to support tumor segmentation, volumetric assessment, AI-assisted clinical interpretation, and interactive 3D visualization.

> **Disclaimer**
>
> This project is intended exclusively for research and educational purposes. It is not approved for clinical diagnosis or medical decision-making.

---

# Key Features

- 3D volumetric MRI preprocessing using MONAI
- 3D U-Net semantic segmentation
- Multimodal fusion of MRI and clinical reports
- DistilBERT clinical text encoder
- Interactive Streamlit dashboard
- Automated tumor volume estimation
- AI-assisted clinical report generation
- Modular research-oriented architecture

---

# User Interface & Visual Assets

## Interactive Dashboard

The Streamlit dashboard provides real-time visualization of volumetric MRI scans, segmentation masks, tumor measurements, and AI-assisted clinical reports.

![Dashboard](docs/dashboard.png)

---

## Segmentation Results

| Raw MRI Slice | Predicted Tumor Segmentation |
|---------------|------------------------------|
| ![](docs/raw_slice.png) | ![](docs/segmented_slice.png) |

---

## 3D Brain Reconstruction

Interactive visualization of the reconstructed brain volume with segmented tumor regions.

![3D Reconstruction](docs/brain_3d.png)

---

## Clinical Report Generation

Automatically generated AI-assisted clinical report.

![Clinical Report](docs/report.png)

---

## Training Performance

### Training & Validation Loss

![Loss Curves](docs/loss_curves.png)

---

## System Architecture

![Architecture](docs/architecture.png)

---

# System Architecture

```
                    3D MRI Volume
                          │
                    Preprocessing
                          │
                    MONAI Pipeline
                          │
                     3D U-Net Encoder
                          │
                  Spatial Feature Maps
                          │
                          ├──────────────────────┐
                          │                      │
                          │                      ▼
                          │            Clinical Report
                          │                      │
                          │                DistilBERT
                          │                      │
                          │               Text Embeddings
                          └──────────────┬───────┘
                                         │
                                 Multimodal Fusion
                                         │
                              Classification Head
                                         │
                          Tumor Segmentation + Report
                                         │
                              Streamlit Dashboard
```

---

# Technologies

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Medical Imaging | MONAI |
| NLP | Hugging Face Transformers (DistilBERT) |
| Medical Image Handling | NiBabel |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Dashboard | Streamlit |

---

# Evaluation Strategy

The current repository represents the complete software architecture of the proposed system.

Performance metrics will be reported after model training on an appropriate research dataset.

The segmentation model is configured to optimize a Hybrid Loss Function:

- Dice Loss
- Cross Entropy Loss

The following evaluation metrics are planned:

- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Precision
- Recall
- F1-Score
- Hausdorff Distance (95HD)
- Average Symmetric Surface Distance (ASSD)
- ROC-AUC (Classification)
- Confusion Matrix

---

# Target Dataset

The pipeline is designed to support publicly available multimodal brain tumor datasets such as:

- BraTS
- BraTS 2023
- BraTS 2024

Supported MRI modalities include:

- T1
- T1ce
- T2
- FLAIR

---

# Repository Structure

```
NeuroFusion-3D/
│
├── app.py
├── train.py
├── inference.py
├── build_model.py
├── multimodal_fusion.py
├── prepare_pipeline.py
├── utils/
├── docs/
│   ├── dashboard.png
│   ├── segmentation.png
│   ├── architecture.png
│   └── training_curves.png
├── requirements.txt
├── LICENSE
└── README.md
```

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
