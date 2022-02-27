import torch
from model import Siamese
import numpy as np
import torchvision.transforms as transforms
from dataset import Crops
import pandas as pd


path = 'data/crops_30'

similar = np.genfromtxt('data/similar_by_fg.csv', delimiter=',', dtype=int)
similar = np.column_stack((similar, np.zeros(len(similar)))).astype(int)

different = np.genfromtxt('data/different_by_fg.csv', delimiter=',', dtype=int)
different = np.column_stack((different, np.zeros(len(different)))).astype(int)

different[np.random.choice(different.shape[0], 1, replace=False), :]
different[np.random.choice(different.shape[0], 2, replace=False), :]

pairs = (1, 2)
crops = Crops(path=path, pairs=pairs)

train_loader = torch.utils.data.DataLoader(crops, batch_size=64, shuffle=True, pin_memory=True)

print()