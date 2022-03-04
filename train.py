import numpy as np
import os
import wandb
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm, trange
from ignite.metrics import Accuracy
jkk
from dataset import Crops
from model import Siamese


def get_loaders(rng, config):
    crops_per_img = config['crops_per_img']
    path = f"data/{config['folder_name']}_{config['crops_per_img']}"

    similar = np.genfromtxt(f'{path}/similar_by_fg.csv', delimiter=',', dtype=int)
    similar = np.column_stack((similar, np.zeros(len(similar)))).astype(int)

    different = np.genfromtxt(f'{path}/different_by_fg.csv', delimiter=',', dtype=int)
    different = np.column_stack((different, np.ones(len(different)))).astype(int)

    size = 20000  # min(len(different), len(similar))

    dif_pairs = different[rng.choice(different.shape[0], size, replace=False), :]
    sim_pairs = similar[rng.choice(similar.shape[0], size, replace=False), :]

    dif_train = dif_pairs[:int(np.floor(size * 0.80)), :]
    dif_val = dif_pairs[int(np.floor(size * 0.80)):, :]
    sim_train = sim_pairs[:int(np.floor(size * 0.80)), :]
    sim_val = sim_pairs[int(np.floor(size * 0.80)):, :]

    train = np.concatenate((dif_train, sim_train))
    val = np.concatenate((dif_val, sim_val))

    crop_train = Crops(path=path, pairs=train)
    crop_val = Crops(path=path, pairs=val)

    train_loader = torch.utils.data.DataLoader(crop_train, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(crop_val, batch_size=config.batch_size, pin_memory=True)

    return train_loader, val_loader


if __name__ == "__main__":
    os.environ['WANDB_MODE'] = 'online'
    run = wandb.init(project="unsupervised_line_detection",
                     config="configs/config.yaml",
                     entity="mvalente")

    config = wandb.config
    rng = np.random.default_rng(1337)

    train_loader, val_loader = get_loaders(rng, config)

    model = Siamese(3)
    model.cuda()
    cudnn.benchmark = True
        model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in trange(config.epochs):
        train_loss = []
        y_pred = []
        y_ground = []
        for img1, img2, y in tqdm(train_loader, desc=f"Epoch: {epoch}"):
            optimizer.zero_grad()
            output = model(img1.cuda(), img2.cuda())
            loss = loss_fn(output, y.float().cuda())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss.append(loss.item())
                y_pred.append(output.detach().cpu().numpy())
                y_ground.append(y.numpy())

        y_pred = np.where(np.concatenate(y_pred) > 0.5, 1, 0)
        y_ground = np.concatenate(y_ground)
        accuracy = sum(y_pred == y_ground) / len(y_pred)

        model.eval()
        eval_loss = []
        y_pred = []
        y_ground = []
        for img1_eval, img2_eval, y_eval in (tqdm(val_loader, desc="Validating")):
            with torch.no_grad():
                output = model(img1_eval.cuda(), img2_eval.cuda())
                eval_loss.append(loss_fn(output, y_eval.float().cuda()).item())

                y_pred.append(output.detach().cpu().numpy())
                y_ground.append(y_eval.numpy())

        y_pred = np.where(np.concatenate(y_pred) > 0.5, 1, 0)
        y_ground = np.concatenate(y_ground)
        accuracy_eval = sum(y_pred == y_ground) / len(y_pred)

        model.train()

        run.log({"loss": sum(train_loss) / len(train_loss),
                 "loss_eval": sum(eval_loss) / len(eval_loss),
                 "acc": accuracy,
                 "acc_eval": accuracy_eval,
                 "epoch": epoch})

    run.finish()
