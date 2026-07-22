import nibabel as nib
import numpy as np
import pydicom
import os
from typing import Tuple, Dict, Any

class ClinicalDataLoader:
    """
    Medical Image Loader for Real Clinical Formats (NIfTI & DICOM)
    Extracts 3D Voxel Tensors, Metadata, and Real-world Spatial Dimensions.
    """
    
    @staticmethod
    def load_nifti(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads a NIfTI (.nii / .nii.gz) volume and extracts spatial voxel dimensions.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Medical file not found at: {file_path}")
            
        # Load NIfTI Image using NiBabel
        nifti_img = nib.load(file_path)
        
        # Extract 3D Image Array and canonicalize orientation (RAS+)
        nifti_img = nib.as_closest_canonical(nifti_img)
        volume_data = nifti_img.get_fdata(dtype=np.float32)
        
        # Extract Voxel Spacing (dx, dy, dz) in mm from the affine matrix
        header = nifti_img.header
        voxel_dims = header.get_zooms()[:3]  # (x_spacing, y_spacing, z_spacing)
        
        metadata = {
            "format": "NIfTI",
            "shape": volume_data.shape,
            "voxel_dimensions_mm": voxel_dims,
            "voxel_volume_cm3": (voxel_dims[0] * voxel_dims[1] * voxel_dims[2]) / 1000.0,
            "affine": nifti_img.affine
        }
        
        return volume_data, metadata

    @staticmethod
    def compute_exact_tumor_volume(segmentation_mask: np.ndarray, voxel_volume_cm3: float) -> Dict[str, float]:
        """
        Calculates exact anatomical tumor volume in cm³ based on physical voxel dimensions.
        """
        positive_voxels = np.sum(segmentation_mask > 0.5)
        total_volume_cm3 = positive_voxels * voxel_volume_cm3
        
        return {
            "total_positive_voxels": int(positive_voxels),
            "calculated_volume_cm3": float(round(total_volume_cm3, 3))
        }

    @staticmethod
    def normalize_intensity(volume: np.ndarray) -> np.ndarray:
        """
        Standard Medical Intensity Normalization (Zero-Mean Unit-Variance or Min-Max Clipping).
        Clips extreme MRI values (outliers) above 99th percentile.
        """
        p1, p99 = np.percentile(volume, (1, 99))
        volume_clipped = np.clip(volume, p1, p99)
        
        if p99 - p1 > 0:
            normalized = (volume_clipped - p1) / (p99 - p1)
        else:
            normalized = volume_clipped
            
        return normalized