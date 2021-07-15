import os
from os import listdir
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from pathlib import Path


class Notebook(Dataset):
    ''' Dataset class for my notebook text'''
    def __init__(self, path, transform=None, loader=default_loader):
        self.path = path
        self.loader = default_loader
        self.transform = transform

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
