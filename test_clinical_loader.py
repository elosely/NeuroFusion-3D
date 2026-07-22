import numpy as np
import nibabel as nib
from dicom_nifti_loader import ClinicalDataLoader

# 1. إنشاء ملف NIfTI وهمي يحاكي أبعاد جهاز الأشعة الحقيقي (1mm x 1mm x 1mm)
synthetic_mri = np.random.rand(128, 128, 64).astype(np.float32)
# نعمل بقعة تشبه الورم في المنتصف
synthetic_mri[50:70, 50:70, 20:30] += 2.0 

# حفظه كملف NIfTI طبي حقيقي
affine = np.diag([1.0, 1.0, 1.0, 1.0]) # Voxel spacing = 1mm x 1mm x 1mm
nifti_test = nib.Nifti1Image(synthetic_mri, affine)
nib.save(nifti_test, "sample_mri.nii.gz")

print("✅ Saved real NIfTI sample: sample_mri.nii.gz")

# 2. تشغيل الـ ClinicalDataLoader
loaded_volume, meta = ClinicalDataLoader.load_nifti("sample_mri.nii.gz")

print("\n--- 📊 Extracted Clinical Metadata ---")
print(f"Format: {meta['format']}")
print(f"Volume Matrix Shape: {meta['shape']}")
print(f"Voxel Spacing (mm): {meta['voxel_dimensions_mm']}")
print(f"Single Voxel Volume (cm³): {meta['voxel_volume_cm3']}")

# 3. محاكاة حساب حجم الورم الحقيقي
dummy_mask = np.zeros_like(loaded_volume)
dummy_mask[50:70, 50:70, 20:30] = 1.0  # (20x20x10 = 4000 voxels)

vol_stats = ClinicalDataLoader.compute_exact_tumor_volume(dummy_mask, meta['voxel_volume_cm3'])

print("\n--- 🧠 Anatomical Tumor Measurement ---")
print(f"Total Tumor Voxels: {vol_stats['total_positive_voxels']}")
print(f"Calculated Tumor Mass Volume: {vol_stats['calculated_volume_cm3']} cm³")