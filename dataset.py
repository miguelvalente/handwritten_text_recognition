import os
import random
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


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

class Crops(Dataset):
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

        #idx_a, idx_b = random.sample(range(self.__len__()), 2)
#
        #img_a = self.loader(self.data[idx_a])
        #img_b = self.loader(self.data[idx_b])
#
        #if self.transform:
        #    img_a = self.transform(img_a)
        #    img_b = self.transform(img_b)
#
        #return img_a, img_b
