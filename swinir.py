import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import sys
import cv2

batch_size = 64
num_workers = 4
channels = 3

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=channels),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_jpeg_datasets(QF, transform):
    jpeg_train_dir = f'./datasets/CIFAR10/jpeg{QF}/train'
    jpeg_test_dir = f'./datasets/CIFAR10/jpeg{QF}/test'

    train_dataset = datasets.ImageFolder(jpeg_train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=True)

    return train_dataset, test_dataset, train_dataloader, test_dataloader


# _, _, jpeg20_train_dataloader, jpeg20_test_dataloader = load_jpeg_datasets(20, transform)
# _, _, jpeg40_train_dataloader, jpeg40_test_dataloader = load_jpeg_datasets(40, transform)


jpeg20_removal_swinir = torch.load('./SwinIR/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth')

jpeg20_output = jpeg20_removal_swinir()

# jpeg40_removal_swinir = torch.load('./SwinIR/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth')

# jpeg40_output = jpeg40_removal_swinir(jpeg40_test_dataloader)
