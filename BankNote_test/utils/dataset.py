import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class BanknoteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, subfolder in enumerate(['genuine', 'counterfeit']):
            folder_path = os.path.join(root_dir, subfolder)
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(folder_path, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
