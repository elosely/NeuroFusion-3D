# 🧠 NeuroFusion 3D: Multimodal 3D Brain Tumor Segmentation & Diagnosis System

An advanced Multimodal Medical AI platform designed for automated brain tumor analysis and surgical planning. It bridges the gap between clinical data and medical imaging by fusing **3D Volumetric MRI scans** with **Unstructured Clinical Text Reports** to deliver precise tumor segmentation, volumetric sizing, and automated diagnostic grading.

---

## 🚀 Key Features

* **3D Volumetric Processing & Augmentation:** Built with **MONAI** and **PyTorch** to handle spatial medical imaging features directly from NIfTI (`.nii.gz`) data arrays.
* **3D U-Net Semantic Segmentation:** High-fidelity localization of mass boundaries across multiple MRI sequences (FLAIR, T1ce, T2).
* **Clinical Multimodal Fusion:** Extracted deep textual features from patient charts using **Hugging Face Transformers (DistilBERT)** and concatenated them with 3D imaging features for accurate classification.
* **Biophysical Volumetric Tracking:** Automatically calculates the estimated tumor mass volume in cubic centimeters ($cm^3$ / cc) to assist in stereotactic radiotherapy planning.
* **Real-Time Fluid Dashboard:** Designed an ultra-fast, real-time UI with **Streamlit** that utilizes native arrays to deliver zero-lag 3D slice navigation (Z-axis) with live segmentation overlays.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Medical AI Framework:** MONAI (Medical Open Network for AI), PyTorch
* **Natural Language Processing (NLP):** Hugging Face Transformers (DistilBERT)
* **Image Processing & Handling:** Nibabel, NumPy
* **Deployment & Visualization:** Streamlit, Matplotlib

---

## 📁 System Architecture & Pipeline

```text
[ 3D MRI Volumetric Scan ] ───> (MONAI Dictionary Transforms) ───> [ 3D U-Net Encoder ] ───┐
                                                                                             ├──> [ Multimodal Fusion Layer ] ───> [ Streamlit Dashboard ]
[ Clinical Patient Report ] ───> (DistilBERT Text Encoder)    ───> [ Text Embeddings ]   ───┘