import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def norm_points(points,width,height):

    x = 1/width
    y = 1/height

    pair = [x,y]
    arr = np.tile(pair,21)
    norm = arr*points
    # norm = norm.astype(int)
    return norm

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        # return 20
    
    def __getitem__(self, idx):
        # print(self.data.iloc[1,0])
        # print(idx)
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert('RGB') 
        width, height = image.size
        points = self.data.iloc[idx, 1]
        points = points.strip('[]')
        points=points.split(',')
        points = np.array(points, dtype=np.float32)
        points = norm_points(points,width,height)
        # class_label = self.data.iloc[idx, 2]
        class_label = torch.tensor(self.data.iloc[idx, 2])
        # convert class label to a number or tensor if needed
        
        sample = {'image': image, 'points': points, 'class': class_label}
        # print(image.size)
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
