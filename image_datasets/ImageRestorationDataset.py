import os

from PIL import Image
from torch.utils.data import Dataset


class ImageRestorationDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir, transform=None):
        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir
        self.transform = transform

        self.degraded_files = os.listdir(self.degraded_dir)
        self.gt_files = os.listdir(self.gt_dir)

    def __len__(self):
        return len(self.degraded_files)

    def __getitem__(self, index):
        degraded_path = os.path.join(self.degraded_dir, self.degraded_files[index])
        gt_path = os.path.join(self.gt_dir, self.gt_files[index])

        degraded_img = Image.open(degraded_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        if self.transform:
            degraded_img = self.transform(degraded_img)
            gt_img = self.transform(gt_img)

        return degraded_img, gt_img