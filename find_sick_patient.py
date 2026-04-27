import pandas as pd
import os

# قراءة الملف
data_frame = pd.read_csv('Data_Entry_2017.csv')
image_dir = './nih_images'

# فلترة الصور الموجودة في جهازك فقط
downloaded_images = set(os.listdir(image_dir))
df_downloaded = data_frame[data_frame['Image Index'].isin(downloaded_images)]

# البحث عن مريض مصاب بتضخم القلب أو الالتهاب
sick_patients = df_downloaded[df_downloaded['Finding Labels'].str.contains('Cardiomegaly|Pneumonia|Effusion', regex=True)]

print(f"Found {len(sick_patients)} sick patients in your folder.")
if len(sick_patients) > 0:
    print("Here are 3 sick images you can test:")
    print("-" * 40)
    for i in range(min(3, len(sick_patients))):
        img_name = sick_patients.iloc[i]['Image Index']
        disease = sick_patients.iloc[i]['Finding Labels']
        print(f"Image Name: {img_name} | Real Disease: {disease}")
    print("-" * 40)