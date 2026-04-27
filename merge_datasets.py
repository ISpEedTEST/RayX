import pandas as pd
import os

# أسماء الأمراض الـ 15 المعتمدة
classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

print("Generating the Unified Smart Map...")

# 1. معالجة بيانات NIH الأصلية
nih_df = pd.read_csv('Data_Entry_2017.csv')
records = []

for index, row in nih_df.iterrows():
    img_name = row['Image Index']
    labels_text = row['Finding Labels']
    
    row_dict = {'Image_Name': img_name}
    for cls in classes:
        row_dict[cls] = 0.0 # الافتراضي صفر
        
    for label in labels_text.split('|'):
        if label in classes:
            row_dict[label] = 1.0 # نضع 1 للمرض الموجود
            
    records.append(row_dict)

# 2. معالجة صورك القديمة مباشرة من مجلد chest_xray
# تأكد أن أسماء المجلدات تطابق ما هو مكتوب هنا بالضبط (حالة الأحرف مهمة)
ext_dir = './chest_xray/train'
pneumonia_dir = os.path.join(ext_dir, 'PNEUMONIA')
normal_dir = os.path.join(ext_dir, 'NORMAL')

# إضافة مرضى الالتهاب الرئوي القدامى (مع تفعيل خدعة الـ -1)
if os.path.exists(pneumonia_dir):
    for img_name in os.listdir(pneumonia_dir):
        row_dict = {'Image_Name': img_name}
        for cls in classes:
            row_dict[cls] = -1.0 # نضع -1 لأننا لا نعلم عن باقي الأمراض
            
        row_dict['Pneumonia'] = 1.0 # متأكدون من الالتهاب
        row_dict['No Finding'] = 0.0 # ومؤكد أنه ليس سليماً
        records.append(row_dict)
    print(f"Added External Pneumonia images with Masking (-1.0).")
else:
    print(f"Folder not found: {pneumonia_dir}")

# إضافة المرضى السليمين القدامى
if os.path.exists(normal_dir):
    for img_name in os.listdir(normal_dir):
        row_dict = {'Image_Name': img_name}
        for cls in classes:
            row_dict[cls] = 0.0 # سليم يعني لا يوجد أي مرض
            
        row_dict['No Finding'] = 1.0
        records.append(row_dict)
    print(f"Added External Normal images.")
else:
    print(f"Folder not found: {normal_dir}")

# 3. حفظ الخريطة الجديدة
final_df = pd.DataFrame(records)
final_df.to_csv('unified_labels.csv', index=False)
print("-" * 40)
print("SUCCESS: unified_labels.csv created successfully!")