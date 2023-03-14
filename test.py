import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from models.ImageRestorationModel import ImageRestorationModel

model = ImageRestorationModel()

PATH = 'pretrained_models/latest.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(PATH, map_location=device))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
img = Image.open('test.jpg')  # Load image as PIL.Image
img = transform(img)
res = model(img)
print(res.shape)
plt.imshow(img.permute(1, 2, 0))
plt.show()