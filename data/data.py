from torch.utils import data
import numpy as np
import os
from PIL import Image
import cv2
import sys


class ImgSeg(data.Dataset):
    """
    Image Segmentation DataLoad
    """

    def __init__(self, jpg_path, seg_path, transform=None):
        self.jpg_path = jpg_path
        self.seg_path = seg_path
        self.trans = transform
        self.filenames = os.listdir(self.jpg_path)

    def __getitem__(self, index):
        # img = Image.open(os.path.join(self.jpg_path, self.filenames[index])).convert('RGB')
        # seg = Image.open(os.path.join(self.seg_path, self.filenames[index].replace('.jpg', '_mask.jpg'))).convert('1')
        img = cv2.imread(os.path.join(self.jpg_path, self.filenames[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.imread(os.path.join(self.seg_path, self.filenames[index].replace('.jpg', '_mask.jpg')), 0)

        if self.trans is not None:
            aug= self.trans(image=img, mask=seg)

        # print(aug["image"].shape, aug["mask"].shape)

        return aug["image"], aug["mask"]

    def __len__(self):
        return len(self.filenames)
