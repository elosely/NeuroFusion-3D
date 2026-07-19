import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd
)

print("🧪 جاري بناء خط تجهيز البيانات (MONAI Dictionary Transforms)...")

# 1. بنجهز عينة وهمية على شكل "قاموس" (Dictionary) لأن MONAI بتحب الداتا متسمية
# هانعتبر عندنا الأشعة (image) والماسك اللي فيه تحديد الدكتور للورم (label)
np.random.seed(42)
fake_image = np.random.randint(0, 255, size=(120, 120, 90, 4)).astype(np.float32)
fake_label = np.zeros((120, 120, 90, 1), dtype=np.float32)
# هنحط ورم وهمي في الماسك
fake_label[50:70, 50:70, 40:50, 0] = 1

data_dict = {"image": fake_image, "label": fake_label}

# 2. بناء الـ Pipeline العبقري من MONAI
# حرف الـ 'd' في آخر كل دالة معناه (Dictionary transform) يعني بيطبق على الـ keys اللي بنحددها
medical_transforms = Compose([
    # جعل القنوات (Channels) هي البُعد الأول في المصفوفة وده اللي PyTorch بتحبه
    # هيحول الأشعة من (120,120,90,4) إلى (4,120,120,90)
    EnsureChannelFirstd(keys=["image", "label"], channel_dim=-1),
    
    # توحيد الإضاءة وقيم الإشارات للأشعة فقط (مش للماسك)
    NormalizeIntensityd(keys=["image"]),
    
    # اقتصاص عشوائي بحجم ثابت (مثلاً 64x64x64) عشان الموديل يتدرب على أجزاء مركزة ومياخدش ميموري ضخمة
    RandSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 64), random_size=False),
    
    # قلب الأشعة عشوائياً كنوع من الـ Augmentation
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0)
])

# 3. تشغيل الـ Pipeline على الداتا بتاعتنا
transformed_data = medical_transforms(data_dict)

print("\n🚀 تم تجهيز الداتا بنجاح!")
print(f"📐 أبعاد الأشعة قبل التجهيز: {fake_image.shape}")
print(f"📐 أبعاد الأشعة بعد تجهيز MONAI (تجهيز للـ 3D Model): {transformed_data['image'].shape}")
print(f"📐 أبعاد ماسك الورم بعد التجهيز: {transformed_data['label'].shape}")