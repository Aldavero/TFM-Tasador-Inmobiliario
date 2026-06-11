import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class PropertyImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, is_train=False):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame con las propiedades (para evitar Data Leakage).
            root_dir (str): Directorio raíz del proyecto donde está la carpeta 'data'.
            is_train (bool): Si es True, aplica Data Augmentation.
        """
        self.root_dir = Path(root_dir)
        df = dataframe.copy()
        
        # Filtrar los que no tienen error de API/Seguridad o Nulos
        valid_states = ["A reformar", "Buen estado", "Lujo"]
        df = df[df['estado_conservacion'].isin(valid_states)]
        
        # Mapeo a números
        self.label_map = {"A reformar": 0, "Buen estado": 1, "Lujo": 2}
        
        # "Explotar" el dataset aislando correctamente por casas
        self.samples = []
        for _, row in df.iterrows():
            label = self.label_map[row['estado_conservacion']]
            for img_col in ['local_img_1', 'local_img_2', 'local_img_3', 'local_img_4', 'local_img_5']:
                img_path_rel = str(row.get(img_col, ''))
                if pd.notna(img_path_rel) and img_path_rel != 'None' and img_path_rel.strip() != '':
                    full_path = self.root_dir / "data_pipeline" / "data" / img_path_rel
                    if full_path.exists() and full_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        self.samples.append((full_path, label))
                        
        self.is_train = is_train
        
        # Transformaciones (Data Augmentation vs Normal)
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5), # Evitar overfitting
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
