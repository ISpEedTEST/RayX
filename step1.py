import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

print("جاري البحث عن صورة أشعة...")

# تحديد مسار مجلد صور الرئة المصابة (Pneumonia)
pneumonia_dir = "./chest_xray/train/PNEUMONIA"

# سحب أول صورة موجودة في المجلد
img_path = glob.glob(os.path.join(pneumonia_dir, "*.jpeg"))[0]

# قراءة الصورة وعرضها
img = mpimg.imread(img_path)
plt.imshow(img, cmap='gray')
plt.title("X-Ray: Pneumonia (Infected)")
plt.axis('off')

print("تم العثور على الصورة! انظر للنافذة الجديدة التي ظهرت.")
plt.show()