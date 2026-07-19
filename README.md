# NeuroFusion 3D: A Multimodal Clinical Decision Support System

NeuroFusion 3D is a Multimodal Medical Imaging prototype developed as a Clinical Decision Support System (CDSS) and a Preoperative Visualization tool for brain tumor analysis.

---

## UI & Visual Assets

| Raw Structural MRI Slice | 3D Segmentation Overlay |
|---|---|
| ![Raw Slice](docs/raw_mri.png) | ![Segmented Slice](docs/segmented_mri.png) |

![Training & Validation Performance](docs/loss_curves.png)

---

## Evaluation Strategy

The 3D segmentation module is based on a 3D U-Net architecture implemented with MONAI and PyTorch. The network optimizes a Hybrid Loss Function: Dice Loss + Cross-Entropy Loss.

---

## Data Flow Diagram

[ 3D Volumetric MRI ] ---> (3D U-Net Encoder) ---> [ Spatial Features ] --->
                                                                           [ Multimodal Fusion Layer ] ---> [ Streamlit Dashboard ]
[ Clinical Text Report ] ---> (DistilBERT Encoder) ---> [ Text Embeddings ] --->

---

## Repository Structure

NeuroFusion-3D/
├── app.py                     # Main Streamlit Dashboard
├── train.py                   # Core PyTorch/MONAI training loop
├── generate_report.py         # Utility script for assets
├── requirements.txt           # Project dependencies
└── docs/                      # UI Screenshots and plots

---

## Installation & Execution

1. Clone the repository:
git clone https://github.com/elosely/NeuroFusion-3D.git
cd NeuroFusion-3D

2. Install Dependencies:
pip install torch torchvision torchaudio monai numpy matplotlib streamlit transformers nibabel pandas

3. Generate Assets & Run Dashboard:
python generate_report.py
streamlit run app.py

---

## Developer Context

* Youssef Elosely
* Medical Biophysics Department, Faculty of Science.
* Focus Area: Computer Vision in Radiotherapy Planning & AI-Driven Medical Imaging.
