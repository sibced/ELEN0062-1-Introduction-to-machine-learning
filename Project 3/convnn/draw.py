import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from .model import Model
from .dataloader import ToImage, CSVDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = Model()
model.load_state_dict(torch.load('small3.pth'), strict=True)
model = model.to(device)

dataloader = DataLoader(CSVDataset.test_set(transforms.Compose([ToImage()])),
                        batch_size=1, shuffle=False)

plt.rcParams['figure.figsize'] = [12, 8]

def multiplot(lines, rows, images, title):
    "make a plot with multiple images"
    plt.figure(figsize=(20, 10))

    for i in range(lines*rows):
        plt.subplot(lines, rows, i+1)
        plt.imshow(images[i])
        plt.title(title[i])
        plt.xticks([])
        plt.yticks([])
        
    plt.show()

model.eval()

with torch.no_grad():
    for data in dataloader:
        image = data['image']
        label = data['label']

        image = image.view(image.shape[1:]).permute([1, 2, 0])

        out_img = data['output'].cpu()[0, 0, :, :]
        pad = torch.zeros_like(out_img)
        comparison = torch.stack([out_img, pad, label[0, 0, :, :]], dim=-1)
        
        multiplot(1, 2, (image, comparison), ('input', 'comp'))

        if input('') == 'q':
            break
