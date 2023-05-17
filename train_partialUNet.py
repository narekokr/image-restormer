import argparse
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from image_datasets.DiffusionDataset import DiffusionDataset
from piqa import SSIM

from models.PartialConvUNet import PartialConvUNet
from util import EarlyStopper

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--load_optimizer', action='store_true')

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
checkpoint = args.checkpoint
load_optimizer = args.load_optimizer

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Create the model object
model = PartialConvUNet()
early_stopper = EarlyStopper(patience=3, min_delta=0.005)

# Define the loss function and optimizer
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


# criterion = SSIMLoss().cuda() if torch.cuda.is_available() else SSIMLoss()

# Create a dataset object
dataset = DiffusionDataset('train_data/imgs', 'train_data/original', 'train_data/masks', transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
# Create a dataloader object
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

def psnr_loss(original, restored, max_val=1.0):
    mse = F.mse_loss(original, restored)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return 1 / psnr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model)
model.to(device)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

if checkpoint:
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) if load_optimizer else None
    scheduler.load_state_dict(checkpoint['scheduler_state_dict']) if load_optimizer else None
    start_epoch = checkpoint['epoch']
    model.eval()
    model.train()

print('Started training')
start = time.time()
# Train the model for some number of epochs
for epoch in range(epochs):
    for i, data in enumerate(train_dataloader):
        # get the inputs
        inputs, original, masks = data
        inputs, original, masks = inputs.to(device), original.to(device), masks.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, masks)
        with torch.no_grad():
            plt.imshow(outputs[0].permute(1, 2, 0), cmap=plt.cm.gray)
            plt.show()
        loss = criterion(original, outputs)
        loss.backward()
        optimizer.step()

        # print statistics
        # if i % 100 == 99:  # print every 100 mini-batches
        taken = datetime.timedelta(seconds=(time.time() - start))
        formatted_time = str(taken).split('.')[0]
        print('[%d, %5d] loss: %.3f, time taken:' % (epoch + 1, i + 1, loss.item()), formatted_time)
        # running_loss = 0.0

    with torch.no_grad():
        losses = []
        for i, data in enumerate(val_loader):
            inputs, original, masks = data
            inputs, original, masks = inputs.to(device), original.to(device), masks.to(device)

            outputs = model(inputs, masks)
            loss = criterion(outputs, original)
            losses.append(loss.item())

            taken = datetime.timedelta(seconds=(time.time() - start))
            formatted_time = str(taken).split('.')[0]
            print('validate [%d, %5d] loss: %.3f, time taken:' % (epoch + 1, i + 1, loss.item()), formatted_time)
        if early_stopper.early_stop(np.mean(losses)):
            break
        scheduler.step()

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'scheduler_state_dict': scheduler.state_dict()
            }, f'checkpoints/model_{epoch}.pt')

torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'scheduler_state_dict': scheduler.state_dict()
}, 'checkpoints/model.pt')

print('Finished Training')

# torch.save(model.state_dict(), 'pretrained_models/latest.pt')


# def test(model, test_loader):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
#     with torch.no_grad():
#         psnr_total = 0.0
#         for i, (inputs, labels) in enumerate(test_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             plt.imshow(outputs[0].permute(1, 2, 0))
#             plt.show()
#             mse = criterion(outputs, labels)
#             psnr = 10 * torch.log10(1 / mse)
#             psnr_total += psnr.item()
#         print('Average PSNR: %.2f dB' % (psnr_total / len(test_loader)))
#
#
# test_data = ImageRestorationDataset('train_data/degraded', 'train_data/original', transform=transform)
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test(model, train_dataloader)
