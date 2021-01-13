import torchvision.models as models
import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import random

SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet18-classification.pt')

test_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
                       ])

test_data = datasets.ImageFolder('/home/trung/working/projects/BoneClassification/dataset/test', test_transforms)
device = torch.device('cuda')
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=random.uniform(5, 10)),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
                       ])

train_data = datasets.ImageFolder('/home/trung/working/projects/BoneClassification/dataset/train', train_transforms)

BATCH_SIZE = 32

test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=4).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

if torch.cuda.is_available():
    model.cuda()


for i, data in enumerate(test_iterator):
    images, labels = data
    plt.imshow(images[0].permute(1, 2, 0))
    plt.show()

    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()
    out = model(images)
    _, predicted = torch.max(out, 1)
    c = (predicted == labels).squeeze()

    predicted_label = train_data.classes[predicted[0].item()]

    print('Item', i, 'Label:', train_data.classes[labels[0].item()], ', Predicted:', predicted_label)