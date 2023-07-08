import torch
from torchvision import datasets, models, transforms


def get_cifar10_dataloader(path, input_size=224, batch_size=16, num_workers=4):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    imagenet_data_train = datasets.CIFAR10(path, train=True, download=True, transform=data_transforms['train'])
    imagenet_data_val = datasets.CIFAR10(path, train=False, download=True, transform=data_transforms['val'])

    dataloaders_dict = {
    'train' : torch.utils.data.DataLoader(imagenet_data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'val' : torch.utils.data.DataLoader(imagenet_data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    return dataloaders_dict

def get_cifar100_dataloader(path, input_size=224, batch_size=16, num_workers=4):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ]),
    }
    imagenet_data_train = datasets.CIFAR100(path, train=True, download=True, transform=data_transforms['train'])
    imagenet_data_val = datasets.CIFAR100(path, train=False, download=True, transform=data_transforms['val'])

    dataloaders_dict = {
    'train' : torch.utils.data.DataLoader(imagenet_data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'val' : torch.utils.data.DataLoader(imagenet_data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    return dataloaders_dict