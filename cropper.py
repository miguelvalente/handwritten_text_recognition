import torchvision.transforms as transforms
from torchvision.utils import save_image
import yaml
from tqdm import tqdm

from dataset import Notebook
from utils import Rotate


with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


transform = transforms.Compose([transforms.ToTensor(), Rotate()])
croper = transforms.RandomCrop(size=(config['height'], config['width']))

notebook = Notebook(path='data/notebook', transform=transform)

crops = []
for idx, img in enumerate(tqdm(notebook, desc='Reading Imgs')):
    for i in range(config['crops_per_image']):
        cropped_img = croper(img)
        save_image(cropped_img, f'data/crops/crop_{idx}_{i}.jpg')
