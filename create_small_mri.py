import numpy as np
import nibabel as nib

print("⏳ Creating a realistic lightweight 3D Brain NIfTI file (< 1 MB)...")

shape = (64, 64, 64)
brain = np.zeros(shape, dtype=np.float32)

x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
center = (32, 32, 32)
brain_mask = ((x - center[0])**2 / 24**2 + (y - center[1])**2 / 28**2 + (z - center[2])**2 / 20**2) <= 1.0
brain[brain_mask] = 0.5 

ventricle_mask = ((x - center[0])**2 / 6**2 + (y - center[1])**2 / 12**2 + (z - center[2])**2 / 5**2) <= 1.0
brain[ventricle_mask] = 0.1

tumor_center = (42, 38, 35)
tumor_mask = ((x - tumor_center[0])**2 / 7**2 + (y - tumor_center[1])**2 / 7**2 + (z - tumor_center[2])**2 / 6**2) <= 1.0
brain[tumor_mask] = 0.95

affine = np.diag([1.5, 1.5, 1.5, 1.0])

nifti_img = nib.Nifti1Image(brain, affine)
output_name = "real_lightweight_brain.nii.gz"
nib.save(nifti_img, output_name)

print(f"✅ DONE! File created successfully: '{output_name}'")