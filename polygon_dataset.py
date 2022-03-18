import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from torchvision.io import read_image

import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image


class PolygonImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform =transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = Image.open(img_path)
        
        image = image.resize((64,64))
        image = image.convert('RGB')
        image = transforms.ToTensor()(image)
        # image = read_image(img_path)
        # image.resize(1, image.shape[1], image.shape[2])
        
        # image = TF.center_crop(image, (64,64))
        # image = F.interpolate(image, size=64)
        # image.reshape(4, 64, 64)
        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
from torch.utils.data import DataLoader

# train_dataloader = DataLoader(PolygonImageDataset(annotations_file='./images/labels.csv', img_dir='./images'), batch_size=3, shuffle=True)



# train_features, train_labels = next(iter(train_dataloader))
# print(train_features.size())
# print(train_labels.size())
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img.permute(1,2,0))
# plt.show()
# print(label)