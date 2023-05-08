import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

from torchvision import models
resnet18 = models.resnet18(pretrained=True)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_val = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = datasets.CIFAR10(root='./../data', train=True,
                                        download=True, transform=transform_train)
valset = datasets.CIFAR10(root='./../data', train=False,
                                        download=True, transform=transform_val)

resnet18.fc = nn.Linear(512, 10)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 40
lr = 0.001
early_stop = 5
optimizer = torch.optim.SGD(resnet18.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

model = resnet18.to(device)
criterion = nn.CrossEntropyLoss().to(device)

former_loss_val = np.inf
no_progress = 0
for epoch in range(epochs):
    model.train()
    print(f'epoch : {epoch}')
    loss_tr = 0
    correct_tr = 0
    data_count_tr = 0
    for inputs,targets in trainloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        loss_tr += loss_item*len(targets)
        _, predicted = torch.max(outputs.data, 1)
        correct_tr += (predicted == targets).sum().item()
        data_count_tr += targets.size(0)
    print(f'train loss: {loss_tr/data_count_tr}, train acc: {correct_tr/data_count_tr}')

    loss_val = 0
    correct_val = 0
    data_count_val = 0
    with torch.no_grad():
        model.eval()
        for inputs, targets in valloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_item = loss.item()
            loss_val += loss_item*len(targets)
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == targets).sum().item()
            data_count_val += targets.size(0)
    print(f'val loss: {loss_val/data_count_val}, val acc: {correct_val/data_count_val}')
    
    if former_loss_val < loss_val:
        no_progress += 1
    else:
        no_progress = 0
    if no_progress >= early_stop:
        break
    former_loss_val = loss_val


# save 
filepath = 'cifar10_resnet18.pt' # or pth
torch.save(model.state_dict(), filepath) # model 자체를 save할 수 있음(같은 확장자)