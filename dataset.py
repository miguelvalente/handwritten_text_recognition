import os
import random
from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class Notebook(Dataset):
    ''' Dataset class for my notebook text'''
    def __init__(self, path, transform=None, loader=default_loader):
        self.path = path
        self.loader = default_loader
        self.transform = transform
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=3)])

        try:
            self._load_data()
        except Exception:
            return False

    def _load_data(self):
        self.data = sorted(Path(self.path).iterdir(), key=os.path.getmtime)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.loader(self.data[idx])
        if self.transform:
            img = self.transform(img)

        return img

class Crops(Dataset):
    ''' Dataset class for the crops of the notebook text'''
    def __init__(self, path, transform=None, loader=default_loader, pairs=None):
        self.path = path
        self.loader = default_loader
        self.transform = transform
        self.pairs = pairs
        if pairs:
            self.different_by_fg, self.similar_by_fg = pairs

        try:
            self._load_data()
        except Exception:
            return False

        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.pairs:
            idx1, idx2, pair_type = self.data[idx]
            img1 = self.loader(self.crops[idx1])
            img2 = self.loader(self.crops[idx2])
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return img1, img2
        else:
            img = self.loader(self.data[idx])
            if self.transform:
                img = self.transform(img)
            return img

    def _load_data(self):
        self.data = sorted(Path(self.path).iterdir(), key=os.path.getmtime)
        if self.pairs:
            pass
