import torchvision.transforms as transforms  # type: ignore
from torch.utils.data import DataLoader
from torchvision.datasets import STL10  # type: ignore
from typing import Tuple

def load_test_data_for_plotting(batch_size: int = 32) -> DataLoader:
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor()
    ])

    testset: STL10 = STL10(root='./data', split='train', download=True, transform=transform)

    testloader: DataLoader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return testloader



def load_data(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    # Define training data transformations
    train_transform: transforms.Compose = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])

    # Define testing data transformations
    test_transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
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
