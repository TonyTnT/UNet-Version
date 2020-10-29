import os

import cv2
from torch.utils import data


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
        img = cv2.imread(os.path.join(self.jpg_path, self.filenames[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.imread(os.path.join(self.seg_path, self.filenames[index].replace('.jpg', '.png')), 0)

        if self.trans is not None:
            aug = self.trans(image=img, mask=seg)

        return aug["image"], aug["mask"]

    def __len__(self):
        return len(self.filenames)
