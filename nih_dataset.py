# import pandas as pd
# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset

# print("Loading medical diagnoses file...")

# # 1. Paths
# csv_path = 'Data_Entry_2017.csv'
# image_dir = './nih_images' # مسار الصور الصحيح المباشر

# # 2. Read the full CSV
# data_frame = pd.read_csv(csv_path)

# # 3. Filter dataframe to only keep images you actually downloaded
# downloaded_images = set(os.listdir(image_dir))
# data_frame = data_frame[data_frame['Image Index'].isin(downloaded_images)]

# # 4. Extract unique diseases from your specific downloaded subset
# all_diseases = set()
# for labels in data_frame['Finding Labels']:
#     for label in labels.split('|'):
#         all_diseases.add(label)

# classes = sorted(list(all_diseases))

# # 5. Build the Custom Dataset Class
# class NIHDistribution(Dataset):
#     def __init__(self, dataframe, image_dir, classes, transform=None):
#         self.dataframe = dataframe
#         self.image_dir = image_dir
#         self.classes = classes
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         # Get image name and path
#         img_name = self.dataframe.iloc[idx]['Image Index']
#         img_path = os.path.join(self.image_dir, img_name)
        
#         # Open image
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
            
#         # Convert text labels into a mathematical array
#         labels_text = self.dataframe.iloc[idx]['Finding Labels']
#         label_matrix = torch.zeros(len(self.classes))
        
#         for label in labels_text.split('|'):
#             label_idx = self.classes.index(label)
#             label_matrix[label_idx] = 1.0
            
#         return image, label_matrix







import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

print("Loading The Ultimate Mixed Dataset (NIH + Kaggle) in a Cocktail...")

# 1. المسارات (سنقرأ من كل الأماكن)
csv_path = 'unified_labels.csv'
dir_nih = './nih_images'
dir_ext_normal = './chest_xray/train/NORMAL'
dir_ext_sick = './chest_xray/train/PNEUMONIA'

# 2. قراءة الخريطة الموحدة
data_frame = pd.read_csv(csv_path)

# 3. فلترة الخريطة: جلب الصور الحقيقية من كل المجلدات
valid_images = set()
valid_extensions = ('.png', '.jpg', '.jpeg')

# جلب صور NIH
if os.path.exists(dir_nih):
    for f in os.listdir(dir_nih):
        if f.lower().endswith(valid_extensions):
            valid_images.add(f)

# جلب صور Kaggle السليمة
if os.path.exists(dir_ext_normal):
    for f in os.listdir(dir_ext_normal):
        if f.lower().endswith(valid_extensions):
            valid_images.add(f)

# جلب صور Kaggle الملتهبة
if os.path.exists(dir_ext_sick):
    for f in os.listdir(dir_ext_sick):
        if f.lower().endswith(valid_extensions):
            valid_images.add(f)

data_frame = data_frame[data_frame['Image_Name'].isin(valid_images)]

classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

class NIHDistribution(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row['Image_Name']
        
        # كشاف ذكي: يبحث عن الصورة في كل الرفوف
        if os.path.exists(os.path.join(dir_nih, img_name)):
            img_path = os.path.join(dir_nih, img_name)
        elif os.path.exists(os.path.join(dir_ext_normal, img_name)):
            img_path = os.path.join(dir_ext_normal, img_name)
        else:
            img_path = os.path.join(dir_ext_sick, img_name)
            
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        label_matrix = torch.tensor([row[cls] for cls in self.classes], dtype=torch.float32)
            
        return image, label_matrix