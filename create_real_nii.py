import numpy as np
import nibabel as nib
import os

print("⏳ جاري إنشاء وتصدير ملف أشعة طبية (.nii.gz) حقيقي على جهازك...")

# 1. هنصنع أبعاد حجمية طبية حقيقية (128x128x90) ومعاها الـ 4 قنوات للأشعة
shape = (128, 128, 90, 4)
data = np.zeros(shape, dtype=np.float32)

# 2. بناء شكل بيوفيزيائي للمخ والورم بالمعادلات
x, y, z = np.ogrid[:128, :128, :90]
brain_zone = ((x - 64)**2 / 55**2 + (y - 64)**2 / 45**2 + (z - 45)**2 / 35**2) <= 1
tumor_zone = ((x - 75)**2 + (y - 60)**2 + (z - 50)**2) <= 14**2

# نملا القنوات الأربعة بنويز وإشارات رنين محاكية
for c in range(4):
    data[brain_zone, c] = np.random.uniform(0.3, 0.7, size=np.sum(brain_zone)) * (c + 1)
    if c == 2: # قناة T1ce بنخلي الورم فيها ينور جامد بالصبغة
        data[tumor_zone, c] += 1.2
    else: # باقي القنوات بيبان فيها أثر الورم بشكل مختلف
        data[tumor_zone, c] += 0.4

# 3. تحويل المصفوفة لملف NIfTI طبي حقيقي باستخدام مصفوفة تحويل (Affine Matrix)
# المصفوفة دي هي اللي بتعرف أجهزة الأشعة الاتجاهات الجسدية (يمين، شمال، فوق، تحت)
affine = np.eye(4)
nii_image = nib.Nifti1Image(data, affine)

# 4. حفظ الملف في فولدر المشروع
output_path = "patient_001_mri.nii.gz"
nib.save(nii_image, output_path)

print(f"✅ نجحنا! الملف جاهز دلوقتي في فولدر المشروع باسم: {os.path.abspath(output_path)}")
print("💡 الحجم خفيف جداً والأبعاد حقيقية 100% وجاهز للرفع على الموقع.")