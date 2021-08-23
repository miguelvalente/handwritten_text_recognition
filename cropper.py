import cv2
import torchvision.transforms as transforms
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from dataset import Crops, Notebook
from utils import Rotate


def crop(config):
    transform = transforms.Compose([transforms.ToTensor(), Rotate()])
    croper = transforms.RandomCrop(size=(config['height'], config['width']))

    notebook = Notebook(path='data/notebook', transform=transform)
    crops = []
    for idx, img in enumerate(tqdm(notebook, desc='Reading Imgs')):
        for i in range(config['crops_per_image']):
            cropped_img = croper(img)
            save_image(cropped_img, f'data/crops_30/crop_{idx}_{i}.jpg')

def s_score(imgs, write=False):
    ''' Calculates s score
        Inputs:
         - imgs: dataset
        returns: a darray with all the scores
    '''

    lower_val = np.array([0, 0, 0])
    upper_val = np.array([100, 100, 100])

    fg_pixels_a2 = []
    bg_pixels = []
    for img_a2 in tqdm(imgs, desc='Calculating Imgs Pixels'):
        img_a2 = np.array(img_a2)
        mask = cv2.inRange(img_a2, lower_val, upper_val)
        fg_pixels_a2.append(mask[mask != 0].shape[0])
        bg_pixels = mask[mask == 0].shape[0]

    mins = []
    maxs = []
    for fg_pixel_a1 in tqdm(fg_pixels_a2, desc='Calculating Max and Min'):
        mins.append(np.minimum(fg_pixel_a1, fg_pixels_a2))
        maxs.append(np.maximum(fg_pixel_a1, fg_pixels_a2))

    scores = np.round(np.stack(mins) / np.stack(maxs), 2)
    scores[np.isnan(scores)] = 0

    if write:
        with open('crops_30_2.txt', 'w') as file:
            for line in scores:
                file.write(','.join([str(i) for i in line]))
                file.write('\n')

    return scores, bg_pixels

def main(config, write=False):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
    path = 'data/crops_30'
    crops = Crops(path=path, transform=None)

    lower_val = np.array([0, 0, 0])
    upper_val = np.array([100, 100, 100])

    scores, _ = s_score(crops)

    similar_by_fg = np.argwhere(scores >= 0.7)
    different_by_fg = np.argwhere(scores <= 0.4)

    np.savetxt(f'data/similar_by_fg.csv', similar_by_fg, delimiter=',', fmt="%i")
    np.savetxt(f'data/different_by_fg.csv', different_by_fg, delimiter=',', fmt="%i")


if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # crop(config)
    main(config)
