import numpy as np
import matplotlib.pyplot as plt

print("🧠 جاري توليد أشعة مخ ثلاثية الأبعاد افتراضية في الذاكرة...")

# هنصنع مكعب أشعة وهمي أبعاده (120 شريحة طول × 120 عرض × 90 شريحة عمق)
# وهنحط فيه 4 قنوات (الـ 4 أنواع أشعة: Flair, T1, T1ce, T2)
# الأبعاد دي محاكية جداً للأشعة الطبية الحقيقية
np.random.seed(42)
height, width, slices, channels = 120, 120, 90, 4

# صنع مصفوفة من الأصفار (مخ فاضي)
mri_data = np.zeros((height, width, slices, channels))

# هنصنع شكل بيضاوي في النص يمثل "المخ" جوه الأشعة
x, y, z = np.ogrid[:height, :width, :slices]
brain_mask = ((x - height//2)**2 / (height//2.5)**2 + 
              (y - width//2)**2 / (width//2.5)**2 + 
              (z - slices//2)**2 / (slices//2.5)**2) <= 1

# هنملا مكان المخ بشوية إشارات (Signals) محاكية للرنين مع شوية نويز طبيعية
for c in range(channels):
    mri_data[brain_mask, c] = np.random.uniform(0.5, 1.0, size=np.sum(brain_mask)) * (c + 1)
    # إضافة نويز خفيفة في الخلفية
    mri_data[:, :, :, c] += np.random.normal(0, 0.05, size=mri_data[:, :, :, c].shape)

# هنحط بقعة منورة في النص تمثل "الورم الافتراضي" جوه القناة الثالثة (T1ce اللي بتنور بالصبغة)
tumor_mask = ((x - 70)**2 + (y - 60)**2 + (z - 45)**2) <= 15**2
mri_data[tumor_mask, 2] += 1.5  

print("\n✅ تم توليد الأشعة بنجاح وبدون استهلاك 1 ميجا من الهارد!")
print(f"📐 الأبعاد الحجمية للأشعة (Shape): {mri_data.shape}")
print("💡 التفسير: (الطول: 120, العرض: 120, عدد الشرائح: 90, عدد أنواع الأشعة: 4)")

# 4. عرض الشريحة رقم 45 (اللي في نص المخ بالظبط) للأربع أنواع أشعة جنب بعض
mid_slice = slices // 2
titles = ['Flair (حدود الورم)', 'T1 (التشريح)', 'T1ce (الورم بالصبغة)', 'T2 (السوائل)']

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for i in range(channels):
    axes[i].imshow(mri_data[:, :, mid_slice, i], cmap='bone')
    axes[i].set_title(titles[i])
    axes[i].axis('off')

print("🖼️ جاري فتح نافذة المقارنة بين أنواع الأشعة الأربعة...")
plt.show()