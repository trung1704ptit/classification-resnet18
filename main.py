import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import random
import numpy as np
from PIL import ImageFile
from visualization import plot_accuracies, plot_lrs, plot_losses
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 1234
EPOCHS = 100
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet18-classification.pt')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=random.uniform(5, 10)),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
                       ])

test_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
                       ])

train_data = datasets.ImageFolder('/home/trung/working/projects/BoneClassification/dataset/train', train_transforms)
valid_data = datasets.ImageFolder('/home/trung/working/projects/BoneClassification/dataset/val', test_transforms)
test_data = datasets.ImageFolder('/home/trung/working/projects/BoneClassification/dataset/test', test_transforms)


BATCH_SIZE = 32

train_iterator = torch.utils.data.DataLoader(train_data, num_workers=8, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, num_workers=8,  batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_data, num_workers=8, batch_size=BATCH_SIZE)

device = torch.device('cuda')

print(device)

model = models.resnet34(pretrained=True).to(device)

print(model)

for param in model.parameters():
    param.requires_grad = False

print(model.fc)

model.fc = nn.Linear(in_features=512, out_features=4).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_iterator), epochs=EPOCHS)
criterion = nn.CrossEntropyLoss().to(device)


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    lrs = []
    
    model.train()
    # start = time.perf_counter()
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        end = time.perf_counter()
        # print(end - start)
        optimizer.zero_grad()
                
        fx = model(x)
        
        loss = criterion(fx, y)
        
        acc = calculate_accuracy(fx, y)
        
        loss.backward()

        optimizer.step()
        # Record and update learning rate
        lrs.append(get_lr(optimizer))
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # start = time.perf_counter()
    return epoch_loss / len(iterator), epoch_acc / len(iterator), lrs

def evaluate(model, device, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

history = []

for epoch in range(EPOCHS):
    train_loss, train_acc, lrs = train(model, device, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, device, valid_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:05.2f}% |')
    result = {
        "train_loss": train_loss,
        "lrs": lrs,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc
    }

    history.append(result)

plot_accuracies(history)
plot_losses(history)
plot_lrs(history)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss, test_acc = evaluate(model, device, valid_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:05.2f}% |')