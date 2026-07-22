import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import torch
import os
from dicom_nifti_loader import ClinicalDataLoader
from mesh_builder import Medical3DMeshBuilder
from pdf_generator import ClinicalPDFReport
from unet_model import AIInferenceEngine, NeuroFusion3DUNet

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroFusion 3D - AI-Powered Clinical CDSS",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroFusion 3D: Clinical Decision Support System")
st.markdown("### Powered by 3D U-Net Deep Learning & Multi-Planar Volumetric Visualization")
st.markdown("---")

# --- SIDEBAR: CLINICAL FILE UPLOADER & CONTROLS ---
st.sidebar.header("📁 Clinical Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload 3D Volumetric MRI Scan (.nii / .nii.gz)", 
    type=["nii", "gz"]
)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Segmentation Pipeline")
segmentation_mode = st.sidebar.radio(
    "Select Segmentation Engine:",
    ["🤖 3D U-Net Deep Learning Model", "🎛️ Manual Intensity Thresholding"]
)

st.sidebar.markdown("---")

# --- DATA PROCESSING & VIEWPORT ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load 3D Volume
        volume_data, meta = ClinicalDataLoader.load_nifti(tmp_path)
        norm_volume = ClinicalDataLoader.normalize_intensity(volume_data)
        os.remove(tmp_path)

        dim_x, dim_y, dim_z = norm_volume.shape

        # Sidebar Sliders for Navigation
        st.sidebar.header("🕹️ Slice Navigation")
        slice_axial = st.sidebar.slider("Axial Slice (Z-Axis)", 0, dim_z - 1, dim_z // 2)
        slice_coronal = st.sidebar.slider("Coronal Slice (Y-Axis)", 0, dim_y - 1, dim_y // 2)
        slice_sagittal = st.sidebar.slider("Sagittal Slice (X-Axis)", 0, dim_x - 1, dim_x // 2)

        # SEGMENTATION EXECUTION
        if "3D U-Net" in segmentation_mode:
            st.toast("⚡ Running 3D U-Net Deep Learning Inference...", icon="🧠")
            vol_tensor = torch.from_numpy(norm_volume)
            pred_prob = AIInferenceEngine.run_inference(vol_tensor)
            
            # Extract distinct regions
            core_mask = (pred_prob > 0.5).astype(np.float32)
            edema_mask = ((norm_volume > 0.65) & (core_mask == 0)).astype(np.float32)
            total_mask = ((core_mask == 1) | (edema_mask == 1)).astype(np.float32)
        else:
            core_mask = (norm_volume > 0.82).astype(np.float32)
            edema_mask = ((norm_volume > 0.65) & (norm_volume <= 0.82)).astype(np.float32)
            total_mask = (norm_volume > 0.65).astype(np.float32)

        # Calculate exact volume
        vol_stats = ClinicalDataLoader.compute_exact_tumor_volume(total_mask, meta["voxel_volume_cm3"])

        # PDF Report Button
        st.sidebar.markdown("---")
        st.sidebar.header("📄 Medical Report")
        pdf_bytes = ClinicalPDFReport.generate_report(meta, vol_stats)
        st.sidebar.download_button(
            label="📥 Download Clinical PDF Report",
            data=pdf_bytes,
            file_name="NeuroFusion_Clinical_Report.pdf",
            mime="application/pdf"
        )

        # Dashboard Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Format", meta["format"])
        col2.metric("Matrix Dimensions", f"{dim_x} x {dim_y} x {dim_z}")
        col3.metric("Segmentation Engine", "3D U-Net AI" if "3D U-Net" in segmentation_mode else "Manual")
        col4.metric("AI Lesion Mass", f"{vol_stats['calculated_volume_cm3']} cm³")

        st.markdown("---")

        tab1, tab2 = st.tabs(["🖼️ Multi-Color 2D Viewport", "🧊 Multi-Spectral 3D Surface Rendering"])

        with tab1:
            st.subheader("Orthogonal Viewport with Anatomical Contour Overlays")
            st.caption("🔴 Red: Active Core | 🟡 Yellow: Boundary/Edema Boundary Lines")
            
            v_col1, v_col2, v_col3 = st.columns(3)

            def plot_slice(slice_data, c_mask, e_mask, title):
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(slice_data, cmap="gray")
                ax.imshow(np.ma.masked_where(e_mask == 0, e_mask), cmap="YlOrBr", alpha=0.5)
                ax.imshow(np.ma.masked_where(c_mask == 0, c_mask), cmap="Reds", alpha=0.7)
                if np.sum(e_mask) > 0:
                    ax.contour(e_mask, colors='yellow', linewidths=0.8, levels=[0.5])
                if np.sum(c_mask) > 0:
                    ax.contour(c_mask, colors='red', linewidths=1.0, levels=[0.5])
                ax.axis("off")
                ax.set_title(title, fontsize=10, fontweight='bold', color='#1A365D')
                return fig

            with v_col1:
                fig_ax = plot_slice(norm_volume[:, :, slice_axial], core_mask[:, :, slice_axial], edema_mask[:, :, slice_axial], f"Axial Slice #{slice_axial}")
                st.pyplot(fig_ax)

            with v_col2:
                fig_cor = plot_slice(norm_volume[:, slice_coronal, :], core_mask[:, slice_coronal, :], edema_mask[:, slice_coronal, :], f"Coronal Slice #{slice_coronal}")
                st.pyplot(fig_cor)

            with v_col3:
                fig_sag = plot_slice(norm_volume[slice_sagittal, :, :], core_mask[slice_sagittal, :, :], edema_mask[slice_sagittal, :, :], f"Sagittal Slice #{slice_sagittal}")
                st.pyplot(fig_sag)

        with tab2:
            st.subheader("Interactive Multi-Spectral 3D Surface Mesh")
            st.caption("Inspect the 3D reconstructed AI segmentation bounds.")
            
            with st.spinner("Generating Multi-Color 3D Surface Mesh..."):
                fig_3d = Medical3DMeshBuilder.create_3d_volume_mesh(norm_volume, core_mask, edema_mask)
                st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as e:
        st.error(f"Error executing AI pipeline: {e}")

else:
    st.info("💡 **Clinical Workflow Ready:** Upload a 3D MRI scan (`.nii` or `.nii.gz`) to execute the 3D U-Net Deep Learning inference pipeline!")