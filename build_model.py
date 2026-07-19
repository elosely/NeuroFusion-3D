import torch
from monai.networks.nets import UNet

print("🧠 جاري بناء شبكة الـ 3D U-Net المخصصة للأورام الدماغية...")

# 1. تعريف الموديل الـ 3D باستخدام MONAI
# هنحدد له المدخلات والمخرجات بناءً على أبعاد الداتا بتاعتنا
model_3d = UNet(
    spatial_dims=3,          # إحنا شغالين 3D مش 2D
    in_channels=4,           # عدد قنوات الأشعة (Flair, T1, T1ce, T2)
    out_channels=1,          # المخرج هو قناة واحدة (ماسك الورم: ورم أو مش ورم)
    channels=(16, 32, 64),   # عدد الفلاتر في كل طبقة (كل ما ننزل في حرف الـ U بنزود الميزات)
    strides=(2, 2),          # خطوات تصغير الحجم أثناء الـ Downsampling
)

print("✅ تم بناء هيكل الموديل بنجاح!")

# 2. محاكاة "Batch" كاملة كأننا بنعمل Training
# البايروتش بتحب الداتا تدخل في شكل: (Batch_Size, Channels, H, W, D)
# هنفترض إن الـ Batch Size = 1 مريض
fake_batch_image = torch.randn(1, 4, 64, 64, 64)

print(f"\n📥 حجم الأشعة المدخلة للموديل (Batch): {fake_batch_image.shape}")

# 3. تمرير الأشعة جوه الموديل (Forward Pass)
model_3d.eval() # وضع التقييم عشان مفيش تدريب حالياً
with torch.no_grad():
    model_output = model_3d(fake_batch_image)

print(f"📤 حجم مخرجات الموديل (الماسك المتوقع): {model_output.shape}")

if model_output.shape == torch.Size([1, 1, 64, 64, 64]):
    print("\n🎉 يا عيني على الجمال! الموديل طلع مخرجات متوافقة 100% مع حجم الماسك المطلوبة.")
    print("السيستم البرمجي الأساسي للمشروع الخيالي بتاعنا كده بقى جاهز تماماً!")
else:
    print("❌ هناك اختلاف في الأبعاد، محتاجين نراجع الـ Architecture.")