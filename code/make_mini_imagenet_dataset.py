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


def change_imagenet_dir_name():
    # 변경할 디렉토리 경로 지정
    for name in ["train", "test"]:
        base_dir = (
            f"./datasets/mini-imagenet/original/{name}"  # test도 동일하게 적용 가능
        )

        # 폴더 목록 가져오기 및 정렬
        folders = sorted(
            [
                f
                for f in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, f))
            ]
        )

        # 0부터 시작해서 5자리로 패딩된 이름으로 변경
        for idx, folder in enumerate(folders):
            new_name = f"{idx:05d}"
            src = os.path.join(base_dir, folder)
            dst = os.path.join(base_dir, new_name)
            os.rename(src, dst)
            print(f"{folder} -> {new_name}")


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
    change_imagenet_dir_name()
    make_and_save_mini_imagenet_each_qf(100)
