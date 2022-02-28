import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm

from dataset import Crops
from model import Siamese


def get_loaders(rng, config):
    crops_per_img = config['crops_per_img']
    path = f"data/{config['folder_name']}_{config['crops_per_img']}"

    similar = np.genfromtxt(f'{path}/similar_by_fg.csv', delimiter=',', dtype=int)
    similar = np.column_stack((similar, np.zeros(len(similar)))).astype(int)

    different = np.genfromtxt(f'{path}/different_by_fg.csv', delimiter=',', dtype=int)
    different = np.column_stack((different, np.ones(len(different)))).astype(int)

    dif_pairs = different[rng.choice(different.shape[0], 40000, replace=False), :]
    sim_pairs = similar[rng.choice(different.shape[0], 40000, replace=False), :]

    dif_train = dif_pairs[:int(np.floor(40000 * 0.80)), :]
    dif_val = dif_pairs[int(np.floor(40000 * 0.80)):, :]
    sim_train = sim_pairs[:int(np.floor(40000 * 0.80)), :]
    sim_val = sim_pairs[int(np.floor(40000 * 0.80)):, :]

    train = np.concatenate((dif_train, sim_train))
    val = np.concatenate((dif_val, sim_val))

    crop_train = Crops(path=path, pairs=train)
    crop_val = Crops(path=path, pairs=val)

    train_loader = torch.utils.data.DataLoader(crop_train, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(crop_val, batch_size=64, pin_memory=True)

    return train_loader, val_loader


if __name__ == "__main__":
    rng = np.random.default_rng(1337)

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train_loader, val_loader = get_loaders(rng, config)

    network = Siamese(3)
    network.cuda()
    cudnn.benchmark = True

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    loss_fn = nn.BCEWithLogitsLoss()

    network.train()
    for img1, img2, y in tqdm(train_loader, desc="Epochs"):
        optimizer.zero_grad()
        output = network(img1.cuda(), img2.cuda())
        loss = loss_fn(output, y.float().cuda())
        # loss_val += loss.item()
        loss.backward()
        optimizer.step()
