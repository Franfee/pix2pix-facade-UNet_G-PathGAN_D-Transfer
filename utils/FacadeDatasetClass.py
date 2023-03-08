
import os
import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FacadeImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="None"):
        self.transform = transforms.Compose(transforms_)

        # self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        self.filesA = sorted(glob.glob(os.path.join(root, mode) + "/*.jpg"))
        self.filesB = sorted(glob.glob(os.path.join(root, mode) + "/*.png"))
        if mode == "train":
            # self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
            self.filesA.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.jpg")))
            self.filesB.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.png")))

    def __getitem__(self, index):
        # img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        # w, h = img.size
        # img_A = img.crop((0, 0, w // 2, h))
        # img_B = img.crop((w // 2, 0, w, h))
        img_A = Image.open(self.filesA[index % len(self.filesA)]).convert("RGB")
        img_B = Image.open(self.filesB[index % len(self.filesB)]).convert("RGB")

        if np.random.random() < 0.5:
            np_img_A = np.array(img_A)[:, ::-1, :]
            np_img_B = np.array(img_B)[:, ::-1, :]
            img_A = Image.fromarray(np_img_A, "RGB")
            img_B = Image.fromarray(np_img_B, "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        # return len(self.files)
        return len(self.filesA)
