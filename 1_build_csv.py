import pandas as pd
import os

print("🚀 Starting Data Pipeline: Building Clean Unified Map...")

# المسارات الأساسية
nih_csv = 'Data_Entry_2017.csv'
dir_nih = './nih_images'
dir_ext_normal = './chest_xray/train/NORMAL'
dir_ext_sick = './chest_xray/train/PNEUMONIA'

classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

valid_ext = ('.png', '.jpg', '.jpeg')
records = []

# -----------------------------------------
# 1. جلب صور NIH الموجودة فعلياً في المجلد
# -----------------------------------------
nih_actual_images = set([f for f in os.listdir(dir_nih) if f.lower().endswith(valid_ext)]) if os.path.exists(dir_nih) else set()
nih_df = pd.read_csv(nih_csv)

nih_count = 0
for index, row in nih_df.iterrows():
    img_name = row['Image Index']
    if img_name in nih_actual_images:
        labels_text = row['Finding Labels']
        row_dict = {'Image_Name': img_name, 'Source': 'NIH'}
        
        for cls in classes:
            row_dict[cls] = 0.0 
            
        for label in labels_text.split('|'):
            if label in classes:
                row_dict[label] = 1.0 
                
        records.append(row_dict)
        nih_count += 1

# -----------------------------------------
# 2. جلب صور Kaggle (السليمة)
# -----------------------------------------
normal_count = 0
if os.path.exists(dir_ext_normal):
    for img_name in os.listdir(dir_ext_normal):
        if img_name.lower().endswith(valid_ext):
            row_dict = {'Image_Name': img_name, 'Source': 'Kaggle_Normal'}
            for cls in classes:
                row_dict[cls] = 0.0 
            row_dict['No Finding'] = 1.0
            records.append(row_dict)
            normal_count += 1

# -----------------------------------------
# 3. جلب صور Kaggle (التهاب رئوي) بتقنية الإخفاء
# -----------------------------------------
sick_count = 0
if os.path.exists(dir_ext_sick):
    for img_name in os.listdir(dir_ext_sick):
        if img_name.lower().endswith(valid_ext):
            row_dict = {'Image_Name': img_name, 'Source': 'Kaggle_Sick'}
            for cls in classes:
                row_dict[cls] = -1.0  # إخفاء كل الأمراض
            
            row_dict['Pneumonia'] = 1.0 # تأكيد الالتهاب
            row_dict['No Finding'] = 0.0
            records.append(row_dict)
            sick_count += 1

# -----------------------------------------
# حفظ الخريطة النهائية
# -----------------------------------------
final_df = pd.DataFrame(records)
final_df.to_csv('unified_labels.csv', index=False)

print("-" * 50)
print("✅ Unified Map Generated Successfully: unified_labels.csv")
print(f"📁 NIH Images Found: {nih_count}")
print(f"📁 Kaggle Normal Images Found: {normal_count}")
print(f"📁 Kaggle Pneumonia Images Found: {sick_count}")
print(f"🔥 TOTAL READY IMAGES: {nih_count + normal_count + sick_count}")
print("-" * 50)