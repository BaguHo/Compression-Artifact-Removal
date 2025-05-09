from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from torch import nn
import torch
import os, sys, re
import logging
import cv2
import tqdm
import time
import lpips

num_classes = 100

class CustomDataset(Dataset):
    def __init__(self, input_images, target_images):
        self.input_images = input_images
        self.target_images = target_images

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        input_image = torch.from_numpy(input_image).float()/255.0
        target_image = torch.from_numpy(target_image).float()/255.0
        return input_image, target_image
    
    
def load_test_dataset_and_dataloader_32x32_y_each_qf(QF, batch_size, num_workers):  
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "original_size")

    test_input_dataset = []
    test_target_dataset = []

    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    # 테스트 데이터 로드
    for i in tqdm.tqdm(range(num_classes), desc=f"Loading test data (QF {QF})"):
        test_path = os.path.join(test_input_dir, f"{i:03d}")
        target_test_path = os.path.join(target_test_dataset_dir, f"{i:03d}")

        # test_path 내 파일을 정렬된 순서로 불러오기
        sorted_test_files = sorted(os.listdir(test_path))
        sorted_target_test_files = sorted(os.listdir(target_test_path))

        # 두 디렉토리의 파일명이 같은지 확인하며 로드
        for input_file, target_file in zip(sorted_test_files, sorted_target_test_files):
            if input_file.replace(".jpeg", ".png") == target_file:
                # input 이미지 로드
                test_image_path = os.path.join(test_path, input_file)
                test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
                test_image = np.expand_dims(test_image, axis=0)
                test_input_dataset.append(test_image)

                # target 이미지 로드
                target_image_path = os.path.join(target_test_path, target_file)
                target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
                target_image = np.expand_dims(target_image, axis=0)
                test_target_dataset.append(target_image)

            else:
                print(
                    f"Warning: Mismatched files in testing set: {input_file} and {target_file}"
                )

    # Dataset과 DataLoader 생성
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataset, test_loader


def load_train_dataset_and_dataloader_32x32_y_all_qf(batch_size, num_workers):
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "original_size")

    train_input_dataset = []
    train_target_dataset = []

    QFs = [100, 80, 60, 40, 20]
    # QFs = [100]
    for QF in QFs:
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")

        # 학습 데이터 로드
        for i in tqdm.tqdm(
            range(num_classes), desc=f"Loading train data (QF {QF})", total=num_classes
        ):
            train_path = os.path.join(train_input_dir, f"{i:03d}")
            target_train_path = os.path.join(target_train_dataset_dir, f"{i:03d}")

            # train_path 내 파일을 정렬된 순서로 불러오기
            sorted_train_files = sorted(os.listdir(train_path))
            sorted_target_train_files = sorted(
                os.listdir(target_train_path)
            )

            # 두 디렉토리의 파일명이 같은지 확인하며 로드
            for input_file, target_file in zip(
                sorted_train_files, sorted_target_train_files
            ):
                if input_file.replace("jpeg", "png") == target_file:
                    # input 이미지 로드
                    train_image_path = os.path.join(train_path, input_file)
                    train_image = cv2.imread(train_image_path, cv2.IMREAD_GRAYSCALE)
                    # [h,w] --> [1,h,w] 
                    train_image = np.expand_dims(train_image, axis=0)
                    train_input_dataset.append(train_image)

                    # target 이미지 로드
                    target_image_path = os.path.join(target_train_path, target_file)
                    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
                    # [h,w] --> [1,h,w] 
                    target_image = np.expand_dims(target_image, axis=0)
                    train_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in training set: {input_file} and {target_file}"
                    )

    # Dataset과 DataLoader 생성
    train_dataset = CustomDataset(train_input_dataset, train_target_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataset, train_loader

def load_train_dataset_and_dataloader_8x8_ycrcb_all_qf(batch_size, num_workers):
    QFs = [100, 80, 60, 40, 20]
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8_images")

    train_input_dataset = []
    train_target_dataset = []

    for QF in QFs:
        # input images
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")

        # target images (original)
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")

        # 학습 데이터 로드
        for i in tqdm.tqdm(
            range(num_classes), desc=f"Loading train data (QF {QF})", total=num_classes
        ):
            train_path = os.path.join(train_input_dir, f"{i:03d}")
            target_train_path = os.path.join(target_train_dataset_dir, f"{i:03d}")

            # train_path 내 파일을 정렬된 순서로 불러오기
            sorted_train_files = sorted(os.listdir(train_path))
            sorted_target_train_files = sorted(os.listdir(target_train_path))

            # 두 디렉토리의 파일명이 같은지 확인하며 로드
            for train_file, target_file in zip(
                sorted_train_files, sorted_target_train_files
            ):
                if train_file == target_file:
                    # input 이미지 로드
                    train_image_path = os.path.join(train_path, train_file)
                    train_image = cv2.imread(train_image_path)  
                    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2YCrCb)
                    if train_image is None:
                        print(f"Warning: Failed to load input image: {train_image_path}")
                        sys.exit(1)
                    else:
                        train_input_dataset.append(train_image)

                    # target 이미지 로드
                    target_image_path = os.path.join(target_train_path, target_file)
                    target_image = cv2.imread(target_image_path)
                    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2YCrCb)
                    if target_image is None:
                        print(f"Warning: Failed to load target image: {target_image_path}")
                        sys.exit(1)
                    else:
                        train_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in training set: {train_file} and {target_file}"
                    )
    train_dataset = CustomDataset(train_input_dataset, train_target_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataset, train_loader


def load_train_dataset_and_dataloader_8x8_y_all_qf(batch_size, num_workers):
    QFs = [100,80,60,40,20]
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8_images")

    train_input_dataset = []
    train_target_dataset = []

    for QF in QFs:
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")

        for i in tqdm.tqdm(
            range(num_classes), desc=f"Loading train data (QF {QF})", total=num_classes
        ):
            train_path = os.path.join(train_input_dir, f"{i:03d}")
            target_train_path = os.path.join(target_train_dataset_dir, f"{i:03d}")

            sorted_train_files = sorted(os.listdir(train_path))
            sorted_target_train_files = sorted(os.listdir(target_train_path))

            for train_file, target_file in zip(
                sorted_train_files, sorted_target_train_files
            ):
                if train_file == target_file:
                    train_image_path = os.path.join(train_path, train_file)
                    train_image = cv2.imread(train_image_path, cv2.IMREAD_GRAYSCALE)
                    # [32,32] --> [1,32,32]
                    train_image = np.expand_dims(train_image, axis=0)
                    if train_image is None:
                        print(f"Warning: Failed to load input image: {train_image_path}")
                        sys.exit(1)
                    else:
                        train_input_dataset.append(train_image)

                    target_image_path = os.path.join(target_train_path, target_file)
                    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
                    # [32,32] --> [1,32,32]
                    target_image = np.expand_dims(target_image, axis=0)
                    if target_image is None:
                        print(f"Warning: Failed to load target image: {target_image_path}")
                        sys.exit(1)
                    else:
                        train_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in training set: {train_file} and {target_file}"
                    )
    train_dataset = CustomDataset(train_input_dataset, train_target_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataset, train_loader

def load_test_dataset_and_dataloader_8x8_y_each_qf(QF, batch_size, num_workers):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8_images")
    test_input_dataset = []
    test_target_dataset = []

    # input images
    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")
    # target images (original)
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    # 테스트 데이터 로드
    for i in tqdm.tqdm(range(num_classes), desc=f"Loading test data (QF {QF})"):
        test_path = os.path.join(test_input_dir, f"{i:03d}")
        target_test_path = os.path.join(target_test_dataset_dir, f"{i:03d}")

        # test_path 내 파일을 정렬된 순서로 불러오기
        sorted_test_files = sorted(os.listdir(test_path))
        sorted_target_test_files = sorted(os.listdir(target_test_path))

        # 두 디렉토리의 파일명이 같은지 확인하며 로드
        for test_file, target_file in zip(sorted_test_files, sorted_target_test_files):
            if test_file == target_file:
                # input 이미지 로드
                test_image_path = os.path.join(test_path, test_file)
                test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
                # [8,8] --> [1,8,8]
                test_image = np.expand_dims(test_image, axis=0)
                test_input_dataset.append(test_image)

                # target 이미지 로드
                target_image_path = os.path.join(target_test_path, target_file)
                target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
                target_image = np.expand_dims(target_image, axis=0)
                test_target_dataset.append(target_image)
            else:
                print(
                    f"Warning: Mismatched files in testing set: {test_file} and {target_file}"
                )
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataset, test_loader