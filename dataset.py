# dataset.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMG_SIZE, USE_CLACHE, USE_NOISE

# Only these 5 diseases (order matters)
DESIRED_LABELS = ['Pneumonia', 'Effusion', 'Atelectasis', 'Cardiomegaly', 'Infiltration']

class NIHChestXrayDataset(Dataset):
    def __init__(self, df, img_root, transform=None, is_train=True):
        self.img_root = img_root
        self.is_train = is_train
        self.label_names = DESIRED_LABELS

        # Recursively index all images (supports subfolders)
        print("Indexing images recursively...")
        self.image_paths = {}
        for root, dirs, files in os.walk(self.img_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths[file] = os.path.join(root, file)

        available_images = set(self.image_paths.keys())
        print(f"Found {len(available_images)} unique image files")

        # Filter dataframe to only images that exist
        df = df[df['Image Index'].isin(available_images)].reset_index(drop=True)
        print(f"Dataset after filtering: {len(df)} samples")

        self.df = df
        self.transform = transform if transform else self._get_default_transform(is_train)

    def _get_default_transform(self, is_train):
        if is_train:
            transforms = [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, border_mode=0, p=0.3),
            ]
            if USE_CLACHE:
                transforms.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3))
            if USE_NOISE:
                transforms.append(A.GaussNoise(var_limit=(10.0, 30.0), p=0.15))
            transforms.extend([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            return A.Compose(transforms)
        else:
            return A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = self.image_paths.get(img_name)
        if img_path is None:
            raise FileNotFoundError(f"{img_name} not found after filtering")

        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Parse labels – only the 5 desired diseases
        finding_str = row['Finding Labels']
        if finding_str == 'No Finding':
            present_set = set()
        else:
            present_set = set(finding_str.split('|'))

        labels = [1 if label in present_set else 0 for label in DESIRED_LABELS]
        labels = np.array(labels, dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(labels, dtype=torch.float32)