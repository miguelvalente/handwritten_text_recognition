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
            save_image(cropped_img, f'data/crops/crop_{idx}_{i}.jpg')

def s_score(img_a1, img_a2):
    img_a1 = np.array(img_a1)
    img_a2 = np.array(img_a2)

    lower_val = np.array([0, 0, 0])
    upper_val = np.array([100, 100, 100])

    mask = cv2.inRange(img_a1, lower_val, upper_val)
    fg_pixels_a1 = mask[mask != 0].shape[0]
    bg_pixels_a1 = mask[mask == 0].shape[0]

    mask = cv2.inRange(img_a2, lower_val, upper_val)
    fg_pixels_a2 = mask[mask != 0].shape[0]
    bg_pixels_a2 = mask[mask == 0].shape[0]

    return np.min([fg_pixels_a1, fg_pixels_a2]) / np.max([fg_pixels_a1, fg_pixels_a2])

def temp(config):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
    crops = Crops(path='data/crops', transform=None)

    mean = 0
    maxim = 0
    minim = 0
    # for img in tqdm(crops):
    #     img = np.array(img)
    #     mean += img.mean(axis=0).mean(axis=0)
    #     minim += img.min(axis=0).mean(axis=0)
    #     maxim += img.max(axis=0).mean(axis=0)

    mean /= crops.__len__()
    maxim /= crops.__len__()
    minim /= crops.__len__()

    print(f'max:{maxim} | min:{minim} | mean:{mean}')
#    count = cv2.countNonZero(np.array(img_a.squeeze()))
#    plt.imshow(img_a.permute(1, 2, 0))
#    plt.show()
    lower_val = np.array([0, 0, 0])
    upper_val = np.array([100, 100, 100])

    s_score(crops[0], crops[2])
    mask = cv2.inRange(img, lower_val, upper_val)
    plt.imshow(mask)
    plt.show()


    print()

if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # crop(config)
    temp(config)
