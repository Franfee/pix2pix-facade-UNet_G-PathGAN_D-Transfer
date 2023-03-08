# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 9:54
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

from torch.utils.data import DataLoader
import torchvision.transforms.functional as f

from utils.get_parser import get_parser
from utils.FacadeDatasetClass import *

opt = get_parser()

# Configure dataloaders
data_transforms = [
    transforms.Resize((opt.img_height, opt.img_width), f.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


def get_data_loader(mode="train"):
    if mode == "train":
        train_dataloader = DataLoader(
            FacadeImageDataset(root="datasets/%s" % opt.dataset_name, transforms_=data_transforms, mode="train"),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
        return train_dataloader
    else:
        val_dataloader = DataLoader(
            FacadeImageDataset(root="datasets/%s" % opt.dataset_name, transforms_=data_transforms, mode="test"),
            batch_size=10,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
        return val_dataloader
