import cv2
import os
import torchvision.transforms as transforms
import yaml
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import torch 

import matplotlib.pyplot as plt
import numpy as np
from dataset import Crops, Notebook
from utils import Rotate

class Cropper():
    def __init__(self, config):
        self.config = config
        self.lower = np.array(config['lower_treshold'])
        self.upper = np.array(config['upper_treshold'])
        self.crops_per_img = config['crops_per_img']
        self.height = config['width']
        self.width = config['width']
        self.path = f"data/{config['folder_name']}_{config['crops_per_img']}"

    def crop(self, show_img=False):
        '''
            Keeps Cropping until the foreground pixels are above 6000 if 500by500; 2000 if 250by250
            6000 was a the value selcted for this "toy dataset" looking at crops with no text in it
            essentially makes sure that its not saving white space crrops
            To find values for anotehr use case False to save_crops
        '''
        fg_tresh = 2000
        transform = transforms.Compose([transforms.ToTensor(), Rotate()])
        croper = transforms.RandomCrop(size=(self.height, self.width))

        notebook = Notebook(path='data/notebook', transform=transform)

        crops = []
        for idx, img in enumerate(tqdm(notebook, desc='Reading Imgs')):
            for i in range(self.crops_per_img):
                cropped_img = np.array(croper(img))
                mask = cv2.inRange(cropped_img, self.lower, self.upper)
                fg_pixels = (mask[mask != 0].shape[0])

                if show_img:
                    bg_pixels = mask[mask == 0].shape[0]
                    print(f'Bg: {bg_pixels}   Fg: {fg_pixels}')
                    cv2.imshow('Image', cropped_img)
                    k = cv2.waitKey(0)
                    if k == 27:    # Esc key to stop
                        break
                    else:
                        continue
                else:
                    if fg_pixels < fg_tresh:
                        i -= 1
                    else:
                        save_image(to_tensor(cropped_img), f'{self.path}/crop_{idx}_{i}.jpg')

    def s_score(self, imgs, write=False):
        '''
            Calculates s score
            Inputs:
            - imgs: dataset
            returns: a darray with all the scores
        '''

        fg_pixels_a2 = []
        for img in tqdm(imgs, desc='Calculating Imgs Pixels'):
            img = np.array(img)
            mask = cv2.inRange(img, self.lower, self.upper)
            fg_pixels_a2.append(mask[mask != 0].shape[0])

        mins = []
        maxs = []
        for fg_pixel_a1 in tqdm(fg_pixels_a2, desc='Calculating Max and Min'):
            mins.append(np.minimum(fg_pixel_a1, fg_pixels_a2))
            maxs.append(np.maximum(fg_pixel_a1, fg_pixels_a2))

        scores = np.round(np.stack(mins) / np.stack(maxs), 2)
        scores[np.isnan(scores)] = 0

        if write:
            with open(f"crops_{self.crops_per_imge}.txt", 'w') as file:
                for line in scores:
                    file.write(','.join([str(i) for i in line]))
                    file.write('\n')

        return scores


if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    cropper = Cropper(config)
    path = f"data/{config['folder_name']}_{config['crops_per_img']}"

    # Crops based on amount of "crops_per_image" or if dir empty due to an error
    if not os.path.exists(path):
        os.makedirs(path)
        cropper.crop()
    elif len(os.listdir(path=path)) == 0:
        cropper.crop()

    crops = Crops(path=path)

    scores = cropper.s_score(crops)

    similar_by_fg = np.argwhere(scores >= 0.7)
    different_by_fg = np.argwhere(scores <= 0.4)

    np.savetxt(f'{path}/similar_by_fg.csv', similar_by_fg, delimiter=',', fmt="%i")
    np.savetxt(f'{path}/different_by_fg.csv', different_by_fg, delimiter=',', fmt="%i")
