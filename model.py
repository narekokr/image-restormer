import argparse
import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from models.ImageRestorationModel import ImageRestorationModel


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
args = parser.parse_args()
batch_size = args.batch_size

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


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Create the model object
model = ImageRestorationModel()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a dataset object
train_dataset = ImageRestorationDataset('train_data/degraded', 'train_data/original', transform=transform)
# Create a dataloader object
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def psnr_loss(original, restored, max_val=1.0):
    mse = F.mse_loss(original, restored)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return -psnr  # Return negative PSNR loss since we want to minimize it


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('Started training')
start = time.time()
# Train the model for some number of epochs
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # get the inputs
        inputs, original = data
        inputs, original = inputs.to(device), original.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(original, outputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 100 == 99:  # print every 100 mini-batches
        if i > -1:  # print every 100 mini-batches
            taken = datetime.timedelta(seconds=(time.time() - start))
            formatted_time = str(taken).split('.')[0]
            print('[%d, %5d] loss: %.3f, time taken:' % (epoch + 1, i + 1, loss.item()), formatted_time)
            # running_loss = 0.0

print('Finished Training')

torch.save(model.state_dict(), 'pretrained_models/latest.pt')

def test(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        psnr_total = 0.0
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            plt.imshow(outputs[0].permute(1, 2, 0))
            plt.show()
            mse = criterion(outputs, labels)
            psnr = 10 * torch.log10(1 / mse)
            psnr_total += psnr.item()
        print('Average PSNR: %.2f dB' % (psnr_total / len(test_loader)))


test_data = ImageRestorationDataset('train_data/degraded', 'train_data/original', transform=transform)
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test(model, dataloader)
