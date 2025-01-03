import torch
import torchvision #type: ignore
import torchvision.transforms as transforms # type: ignore
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10 #type: ignore
from typing import Tuple

def load_data(batch_size:int = 32) ->Tuple[DataLoader, DataLoader]:
    transform : torchvision.transforms.transforms.Compose = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
            )
            ])
    trainset : CIFAR10 = CIFAR10(root = './data', 
                                       train=True,
                                       download=True,
                                       transform=transform)
    testset : CIFAR10 = CIFAR10(root = './data', 
                                       train=False,
                                       download=True,
                                       transform=transform)
    trainloader:DataLoader = DataLoader(trainset, batch_size=batch_size, 
                             shuffle=True)
    testloader :DataLoader = DataLoader(testset, batch_size=batch_size, 
                             shuffle=True)

    return (trainloader, testloader)

