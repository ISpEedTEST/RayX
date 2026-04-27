import kagglehub
import shutil
import os

print("جاري تحميل البيانات، يرجى الانتظار (حجمها حوالي 1.2 جيجا)...")

# تحميل البيانات
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# تحديد اسم المجلد اللي راح نحفظ فيه الصور داخل مشروعنا
destination = "./chest_xray"

# أمر لنقل الملفات من المسار المخفي إلى مجلد مشروعك
if not os.path.exists(destination):
    print("تم التحميل! جاري نقل الملفات إلى مجلد مشروعك...")
    shutil.copytree(path, destination)
    print("تم الحفظ بنجاح! مجلد الصور جاهز الآن داخل مشروعك.")
else:
    print("المجلد موجود مسبقاً عندك!")