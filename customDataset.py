import os
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image


# df = pd.read_csv("./railroads_dataset/railroads_validation.csv")

# Cleaning up the data
# print(df)
# df.drop('class index', inplace=True, axis=1)
# df.drop('Unnamed: 0', inplace=True, axis=1)
# df.drop('Unnamed: 0.1', inplace=True, axis=1)
# df.drop('data set', inplace=True, axis=1)
# df.to_csv("./railroads_dataset/railroads_validation.csv", index=False)
# print("\n")
# print(df)

# Custom dataset class for railroads dataset
class RailroadsDataset(Dataset):
    def __init__(self, csv, path_dir, transform=None):
        self.annotations = pd.read_csv(csv)
        self.path_dir = path_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.path_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        if self.transform:
            image = self.transform(image)
        
        return image, y_label
        