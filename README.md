## 📊 Architecture Benchmark & Evaluation Strategy

This repository represents a **research-oriented software prototype** for a Multimodal Clinical Decision Support System (CDSS). The current implementation focuses on the complete system architecture, data processing pipeline, multimodal feature fusion, and interactive visualization framework. As the deep learning models have **not yet been trained and validated on a complete research dataset**, no performance metrics are reported in order to maintain scientific accuracy and reproducibility.

The architecture is fully prepared for end-to-end training and quantitative evaluation using established medical imaging benchmarks once an appropriate annotated dataset is integrated.

---

### 1. Segmentation Pipeline Evaluation

The 3D segmentation module is based on a **3D U-Net** architecture implemented with **MONAI** and **PyTorch**. To address the severe class imbalance commonly encountered in brain tumor segmentation, the network is configured to optimize a hybrid loss function combining Dice Loss and Cross-Entropy Loss:

[
\mathcal{L}*{total} =
\mathcal{L}*{Dice} +
\mathcal{L}_{CrossEntropy}
]

The segmentation model will be evaluated using widely accepted quantitative metrics, including:

* Dice Similarity Coefficient (DSC)
* Intersection over Union (IoU)
* Precision
* Recall (Sensitivity)
* F1-Score
* Hausdorff Distance (95HD)
* Average Symmetric Surface Distance (ASSD)

These metrics provide complementary measurements of volumetric overlap, boundary accuracy, and segmentation robustness.

---

### 2. Multimodal Classification Evaluation

Clinical text reports are encoded using **DistilBERT**, while volumetric MRI features are extracted through the 3D convolutional encoder. The resulting latent representations are fused within a multimodal feature space to support downstream classification tasks.

The classification component is configured to optimize **Cross-Entropy Loss** and is designed to support research applications such as:

* Tumor subtype prediction
* Diagnostic decision support
* Risk stratification
* WHO tumor grade estimation (when supported by appropriately labeled datasets)

Performance will be assessed using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

---

### 3. Target Benchmark Dataset

The preprocessing pipeline, tensor dimensions, and volumetric data flow are designed to be fully compatible with the **Brain Tumor Segmentation (BraTS)** benchmark datasets, including multimodal MRI sequences:

* T1
* T1ce
* T2
* FLAIR

The architecture is intended for training and evaluation on publicly available research datasets containing expert-annotated tumor segmentation masks.

---

### 4. Reproducibility

To ensure scientific reproducibility, the complete training environment will be documented upon model training, including:

* Python version
* PyTorch version
* MONAI version
* CUDA version
* GPU configuration
* Random seed initialization
* Training hyperparameters
* Data preprocessing configuration

---

### 5. Planned Evaluation Protocol

Following model training, the system will be benchmarked with respect to:

* Segmentation accuracy (DSC, IoU)
* Boundary agreement (95HD, ASSD)
* Classification performance
* Tumor volume estimation accuracy
* Inference time per MRI volume
* GPU memory utilization
* End-to-end pipeline latency

This evaluation strategy is intended to provide a comprehensive assessment of both the medical imaging and multimodal learning components while maintaining transparency, reproducibility, and scientific rigor.
                   ├──> [ Multimodal Fusion Layer ] ───> [ Streamlit Dashboard ]
[ Clinical Patient Report ] ───> (DistilBERT Text Encoder)    ───> [ Text Embeddings ]   ───┘
