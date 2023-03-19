import torch
import torchvision.datasets
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.ImageRestorationModel import ImageRestorationModel

model = ImageRestorationModel()
model = nn.DataParallel(model)

PATH = 'pretrained_models/latest(5).pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(PATH, map_location=device))

transform = transforms.Compose([
    # transforms.Resize((512, 512)),
    transforms.ToTensor()
])

with torch.no_grad():

    test_data = torchvision.datasets.ImageFolder('test_data', transform=transform)
    loader = DataLoader(test_data, batch_size=1)
    for image in loader:
        image = image[0]
        res = model(image)
        plt.imshow(res[0].permute(1, 2, 0))
        plt.show()
        print(image.shape, res.shape)
        print(nn.MSELoss()(image, res))
    img = Image.open('test.jpg')  # Load image as PIL.Image
    img = transform(img)
    res = model(img)
    print(res.shape)
    plt.imshow(res.permute(1, 2, 0))
    plt.show()