import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
    def __init__(self, damaged_dir, gt_dir, mask_dir, transform=None):
        self.damaged_dir = damaged_dir
        self.gt_dir = gt_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.degraded_files = os.listdir(self.damaged_dir)
        self.gt_files = os.listdir(self.gt_dir)
        self.mask_files = os.listdir(self.mask_dir)

        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.degraded_files)

    def __getitem__(self, index):
        damaged_path = os.path.join(self.damaged_dir, self.degraded_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])
        gt_path = os.path.join(self.gt_dir, self.gt_files[index])

        damaged_img = Image.open(damaged_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        mask_img = Image.open(mask_path)

        if self.transform:
            damaged_img = self.transform(damaged_img)
            gt_img = self.transform(gt_img)
            mask_img = self.transform(mask_img)

        return damaged_img, gt_img, mask_img
