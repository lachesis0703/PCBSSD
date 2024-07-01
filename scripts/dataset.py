import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from PIL import Image
from torchvision import transforms
import torch

class ImageWithObbDataset(Dataset):
    def __init__(self, img_path, lab_path, transform=None):
        super().__init__()
        # self.ext = '.png'
        self.ext = '.jpeg'
        self.lab_list = os.listdir(img_path if lab_path is None else lab_path)
        self.img_path = img_path
        self.lab_path = lab_path
        self.transform = transform

    def __getitem__(self, index):
        basename = os.path.os.path.splitext(self.lab_list[index])[0]
        try:
            img = 255-read_image(os.path.join(self.img_path, basename + self.ext), ImageReadMode.RGB)
            gt = []
            if self.lab_path is not None:
                with open(os.path.join(self.lab_path, basename + '.txt')) as f:
                    shapes = f.readlines()
                for shape in shapes:
                    # paras = shape.split(' ')  # pcbmo
                    paras = shape.split('\t')
                    if len(paras) != 3:     # 取决于label文件的列数 pcbmo:6 pcb:3
                        continue
                    gt.append(tuple(float(x) for x in paras))
            if gt == []:
                gt.append([0.,0.,0.])
            gt = np.float32(gt)
        except BaseException:
            errstr = basename
            print(f'Load failed: {errstr}')

        if self.transform is not None:
            # img, gt = self.transform(img, gt)
            img = self.transform(img)
            # print(gt,basename)
        
        return img, gt, basename

    def __len__(self):
        return len(self.lab_list)

class ImageNoLabDataset(Dataset):
    def __init__(self, img_path, transform=None):
        super().__init__()
        self.img_list = os.listdir(img_path)
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        basename = os.path.os.path.splitext(self.img_list[index])[0]
        # img = transforms.ToTensor()(Image.open(os.path.join(self.img_path, self.img_list[index])).convert('RGB'))
        img = 255-read_image(os.path.join(self.img_path, basename + '.png'), ImageReadMode.RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, basename

    def __len__(self):
        return len(self.img_list)

class ImageWithObbDataset2(Dataset):
    def __init__(self, img_path, lab_path, transform=None):
        super().__init__()
        self.ext = '.png'
        self.lab_list = os.listdir(img_path if lab_path is None else lab_path)
        self.img_path = img_path
        self.lab_path = lab_path
        self.transform = transform

    def __getitem__(self, index):
        basename = os.path.os.path.splitext(self.lab_list[index])[0]
        try:
            img = 255-read_image(os.path.join(self.img_path, basename + self.ext), ImageReadMode.RGB)
            gt = []
            if self.lab_path is not None:
                with open(os.path.join(self.lab_path, basename + '.txt')) as f:
                    shapes = f.readlines()
                for shape in shapes:
                    paras = shape.split(' ')  # pcbmo
                    if len(paras) != 6:     # 取决于label文件的列数 pcbmo:6 pcb:3
                        continue
                    gt.append(tuple(float(x) for x in paras))
            gt = np.float32(gt)
        except BaseException:
            errstr = basename
            print(f'Load failed: {errstr}')

        if self.transform is not None:
            # img, gt = self.transform(img, gt)
            img = self.transform(img)

        return img, gt, basename

    def __len__(self):
        return len(self.lab_list)
