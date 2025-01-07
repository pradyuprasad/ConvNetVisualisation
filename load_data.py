import torchvision.transforms as transforms  # type: ignore
from torch.utils.data import DataLoader
from torchvision.datasets import STL10  # type: ignore
from typing import Tuple, Dict, List
import torch
import json
import os

STL_METADATA_FILE_NAME = 'STL_meta.json'

def load_mean_std_dev_STL() -> Tuple[List[float], List[float]]:
    if not os.path.exists(STL_METADATA_FILE_NAME):
        calculate_mean_and_std_values()

    with open(STL_METADATA_FILE_NAME, 'r') as f:
        metadata = json.load(f)



    return metadata['mean'], metadata['std_dev']

def load_test_data_for_plotting(batch_size: int = 32) -> DataLoader:
    mean, std = load_mean_std_dev_STL()
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

    testset: STL10 = STL10(root='./data', split='train', download=True, transform=transform)

    testloader: DataLoader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return testloader



def load_data(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    # Define training data transformations
    mean, std = load_mean_std_dev_STL()
    train_transform: transforms.Compose = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=mean)
    ])

    # Define testing data transformations
    test_transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=mean)
    ])

    # Create training dataset
    trainset: STL10 = STL10(root='./data', split='train', download=True, transform=train_transform)

    # Create testing dataset
    testset: STL10 = STL10(root='./data', split='test', download=True, transform=test_transform)

    # Create DataLoaders
    trainloader: DataLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader: DataLoader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Return the DataLoaders
    return trainloader, testloader


def calculate_mean_and_std_values() -> None:
    trainset: STL10 = STL10(root='./data', split='train', download=True, transform=transforms.ToTensor())

    batch_size:int = 64

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches:int = 0

    for data, _ in train_loader:
        channels_sum += torch.sum(data, dim=[0,2,3]) / (96 * 96)
        channels_squared_sum += torch.sum(data**2, dim=[0,2,3]) / (96 * 96)
        num_batches += batch_size

    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    output: Dict[str, List[float]] = {
        'mean': mean.tolist(),
        'std_dev': std.tolist()
    }

    with open(STL_METADATA_FILE_NAME, 'w') as f:
        json.dump(output, f)




if __name__ == "__main__":
    calculate_mean_and_std_values()
