from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import datasets
import numpy as np
import torch
import os, sys, re
import logging
import cv2
import tqdm
import time


def make_and_save_mini_imagenet_each_qf(QF):
    mini_imagenet_path = os.path.join(os.getcwd(), "datasets", "mini-imagenet")
    transform = T.Compose([T.ToTensor(), T.Resize((224, 224))])

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(
        os.path.join(mini_imagenet_path, "original", "train"),
        transform=transform,
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(mini_imagenet_path, "original", "test"),
        transform=transform,
    )

    output_train_path = os.path.join(mini_imagenet_path, "_original", "train")
    output_test_path = os.path.join(mini_imagenet_path, "_original", "test")
    for i in range(1000):
        os.makedirs(os.path.join(output_train_path, str(i)), exist_ok=True)
        os.makedirs(os.path.join(output_test_path, str(i)), exist_ok=True)

    for idx, (img, label) in enumerate(train_dataset):
        image = T.ToPILImage()(img)
        image_filename = os.path.join(
            output_train_path, str(label), f"image_{int(idx):05d}.png"
        )
        image.save(image_filename, "PNG")

        if idx % 5000 == 0:
            print(f"{idx} training images saved...")

    for idx, (img, label) in enumerate(test_dataset):
        image = T.ToPILImage()(img)
        image_filename = os.path.join(
            output_test_path, str(label), f"image_{int(idx):05d}.png"
        )
        image.save(image_filename, "PNG")

        if idx % 5000 == 0:
            print(f"{idx} testing images saved...")


if __name__ == "__main__":
    QFs = [100, 80, 60, 40, 20]
    make_and_save_mini_imagenet_each_qf(100)
