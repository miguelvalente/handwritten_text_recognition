import torch
import torch.nn as nn
from model import Siamese
import numpy as np
import torchvision.transforms as transforms
from dataset import Crops
import pandas as pd


path = 'data/crops_30'

similar = np.genfromtxt('data/similar_by_fg.csv', delimiter=',', dtype=int)
similar = np.column_stack((similar, np.zeros(len(similar)))).astype(int)

different = np.genfromtxt('data/different_by_fg.csv', delimiter=',', dtype=int)
different = np.column_stack((different, np.ones(len(different)))).astype(int)

different[np.random.choice(different.shape[0], 1, replace=False), :]
similar[np.random.choice(different.shape[0], similar, replace=False), :]

pairs = (1, 2)
crops = Crops(path=path, pairs=None)

train_loader = torch.utils.data.DataLoader(crops, batch_size=64, shuffle=True, pin_memory=True)

network = Siamese(3)

network.train()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)
optimizer.zero_grad()

train_loss = []
loss_val = 0
loss_fn = nn.BCELoss()

for img1, img2 in pairs:
    optimizer.zero_grad()
    output = network(img1, img2)
    loss = loss_fn(output)
    loss_val += loss.item()
    loss.backward()
    optimizer.step()
print()