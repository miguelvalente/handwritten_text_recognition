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

def s_score(img_a1, imgs):
    ''' Calculates s score according'''

    lower_val = np.array([0, 0, 0])
    upper_val = np.array([100, 100, 100])

    img_a1 = np.array(img_a1)
    mask = cv2.inRange(img_a1, lower_val, upper_val)
    fg_pixels_a1 = mask[mask != 0].shape[0]
    # bg_pixels_a1 = mask[mask == 0].shape[0]

    scores = []
    for img_a2 in imgs:
        img_a2 = np.array(img_a2)
        mask = cv2.inRange(img_a2, lower_val, upper_val)
        fg_pixels_a2 = mask[mask != 0].shape[0]
        # bg_pixels_a2 = mask[mask == 0].shape[0]
        score = np.min([fg_pixels_a1, fg_pixels_a2]) / np.max([fg_pixels_a1, fg_pixels_a2])

        scores.append(0 if np.isnan(score) else round(score, 2))

    return scores


def calculate_scores(config):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
    crops = Crops(path='data/crops_30', transform=None)

    lower_val = np.array([0, 0, 0])
    upper_val = np.array([100, 100, 100])

    all_scores = []
    for crop in tqdm(crops, desc='Calculating Scores'):
        all_scores.append(s_score(crop, crops))

    with open('year.txt', 'w') as file:
        for scores in all_scores:
            file.write(f'{scores}')

    # img = np.array(crops[2])
    # mask = cv2.inRange(img, lower_val, upper_val)
    # plt.imshow(mask)
    # plt.show()
    print()

if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    crop(config)
    calculate_scores(config)
