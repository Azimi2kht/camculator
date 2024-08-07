from glob import glob
from re import findall

import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

labels = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "w": 10,
    "x": 11,
    "y": 12,
    "z": 13,
    "dot": 14,
    "minus": 15,
    "plus": 16,
    "slash": 17,
}


class CamculatorDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.glob = glob(self.data_path + "/*")
        self.transform = transform

    def __len__(self):
        return len(self.glob)

    def __getitem__(self, idx):
        img_path = self.glob[idx]

        pattern = r"[^/]+(?=-)"
        symbol = findall(pattern, img_path)[0]

        label = labels[symbol]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            aug = self.transform(image=image)
            image = aug["image"]

        return image, label
