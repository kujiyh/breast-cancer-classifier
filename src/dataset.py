import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

def get_loaders(data_dir = 'C:/Users/brian/BCC/data', batch_size=64): # load data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # converting to tensor and standardizing data
    ])

    train_transform = transforms.Compose([ # more data
        transforms.Resize((224, 224)),   
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size 

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True) # nonrandom test data

    return train_loader, test_loader, datasets.ImageFolder(data_dir).classes




