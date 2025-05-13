from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
import os, sys, re
import logging
import cv2
import tqdm
import time
import lpips

logging.basicConfig(filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s")
num_classes = 100
QFs = [100,80,60,40,20]
# QFs = [100]

class CustomDataset(Dataset):
    def __init__(self, input_images, target_images):
        self.input_images = input_images
        self.target_images = target_images

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        # print("input_image.shape, target_image.shape", input_image.shape, target_image.shape)
        # input()
        # [3, h, w] 에서 [3,h/255,w/255]로 변환
        # input_image = torch.from_numpy(input_image)
        # target_image = torch.from_numpy(target_image)
        return input_image, target_image

def load_images_from_dirctory(input_dir, target_dir, color_space=None):
    if color_space is None:
        raise ValueError("color_space must be specified")
    
    input_images = []
    target_images = []
    
    for i in range(num_classes):
        input_path = os.path.join(input_dir, f"{i:03d}")
        target_path = os.path.join(target_dir, f"{i:03d}")
        
        # input_path 내 파일을 정렬된 순서로 불러오기
        sorted_input_files = sorted(os.listdir(input_path))
        sorted_target_files = sorted(os.listdir(target_path))
        
        # 두 디렉토리의 파일명이 같은지 확인하며 로드
        for input_file, target_file in zip(sorted_input_files, sorted_target_files):
            if input_file.replace("jpeg", "png") == target_file:
                # input 이미지 로드
                input_image_path = os.path.join(input_path, input_file)
                target_image_path = os.path.join(target_path, target_file)

                if color_space == "y":
                    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
                    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
                elif color_space == "ycrcb":
                    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
                    target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
                    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2YCrCb)
                elif color_space == "bgr":
                    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)    
                    target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
                
                if color_space == "y":
                    # [h,w] -> [1,h,w]
                    input_image = np.expand_dims(input_image, axis=0)
                    target_image = np.expand_dims(target_image, axis=0)
                    input_image = input_image.astype(np.float32)/255
                    target_image = target_image.astype(np.float32)/255
                else:
                    # [h,w,c] -> [c,h,w]
                    input_image = input_image.transpose(2, 0, 1).astype(np.float32)/255
                    target_image = target_image.transpose(2, 0, 1).astype(np.float32)/255
                
                input_images.append(input_image)
                target_images.append(target_image)
            else:
                print(f"Warning: Mismatched files in training set: {input_file} and {target_file}")
    
    return input_images, target_images

def load_dataset_and_dataloader_all_qf(is_train=None, color_space=None, size=None,batch_size=64, num_workers=64):  
    if color_space is None or size is None or is_train is None:
        raise ValueError("color_space, size, and is_train must be specified")
    
    input_images = []
    target_images = [] 
    for QF in QFs:  
        print(f"Loading QF {QF} dataset...")
        cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", size)

        if is_train:    
            input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
            target_dir = os.path.join(cifar100_path, "original", "train")
        else:
            input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")
            target_dir = os.path.join(cifar100_path, "original", "test")

        input_images_from_each_qf, target_images_from_each_qf = load_images_from_dirctory(input_dir, target_dir, color_space=color_space)
        input_images.extend(input_images_from_each_qf)
        target_images.extend(target_images_from_each_qf)
    
    # Dataset과 DataLoader 생성
    dataset = CustomDataset(input_images, target_images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataset, loader

def load_dataset_and_dataloader_each_qf(QF, is_train=None, color_space=None, size=None,batch_size=64, num_workers=64):  
    if color_space is None or size is None or is_train is None:
        raise ValueError("color_space, size, and is_train must be specified")
    
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", size)
    print(f"Loading QF {QF} dataset... from {cifar100_path}")

    if is_train:    
        input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        target_dir = os.path.join(cifar100_path, "original", "train")
    else:
        input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")
        target_dir = os.path.join(cifar100_path, "original", "test")

    input_dataset, target_dataset = load_images_from_dirctory(input_dir, target_dir, color_space=color_space)
    
    # Dataset과 DataLoader 생성
    dataset = CustomDataset(input_dataset, target_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataset, loader

# ==============================================================================================

def load_train_dataset_and_dataloader_8x8_ycrcb_all_qf(batch_size, num_workers):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8")

    train_input_dataset = []
    train_target_dataset = []

    for QF in QFs:
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")

        train_input_images, train_target_images = load_images_from_dirctory(train_input_dir, target_train_dataset_dir, color_space="ycrcb")
        train_input_dataset.extend(train_input_images)
        train_target_dataset.extend(train_target_images)
    
    train_dataset = CustomDataset(train_input_dataset, train_target_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataset, train_loader

def load_test_dataset_and_dataloader_8x8_ycrcb_each_qf(QF, batch_size, num_workers):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8")
    test_input_dataset = []
    test_target_dataset = []

    # input images
    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")
    # target images (original)
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    test_input_images, test_target_images = load_images_from_dirctory(test_input_dir, target_test_dataset_dir, color_space="ycrcb")
    test_input_dataset.extend(test_input_images)
    test_target_dataset.extend(test_target_images)

    test_dataset = CustomDataset(test_input_dataset, test_target_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataset, test_loader

def load_train_dataset_and_dataloader_32x32_ycrcb_all_qf(batch_size, num_workers):
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "32x32")

    train_input_dataset = []
    train_target_dataset = []

    for QF in QFs:
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")

        train_input_images, train_target_images = load_images_from_dirctory(train_input_dir, target_train_dataset_dir, color_space="ycrcb")
        train_input_dataset.extend(train_input_images)
        train_target_dataset.extend(train_target_images)
    
    # Dataset과 DataLoader 생성
    train_dataset = CustomDataset(train_input_dataset, train_target_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataset, train_loader

def load_train_dataset_and_dataloader_32x32_ycrcb_each_qf(QF, batch_size, num_workers):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "32x32")
    train_input_dataset = []
    train_target_dataset = []

    train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
    target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")

    train_input_images, train_target_images = load_images_from_dirctory(train_input_dir, target_train_dataset_dir, color_space="ycrcb")
    train_input_dataset.extend(train_input_images)
    train_target_dataset.extend(train_target_images)

    train_dataset = CustomDataset(train_input_dataset, train_target_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, train_loader

def load_test_dataset_and_dataloader_32x32_ycrcb_each_qf(QF, batch_size, num_workers):
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "32x32")

    test_input_dataset = []
    test_target_dataset = []

    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    # 테스트 데이터 로드
    test_input_images, test_target_images = load_images_from_dirctory(test_input_dir, target_test_dataset_dir, color_space="ycrcb")
    test_input_dataset.extend(test_input_images)
    test_target_dataset.extend(test_target_images)

    # Dataset과 DataLoader 생성
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataset, test_loader

def load_train_dataset_and_dataloader_32x32_bgr_all_qf(batch_size, num_workers):
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "32x32")

    train_input_dataset = []
    train_target_dataset = []

    for QF in QFs:
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")

        train_input_images, train_target_images = load_images_from_dirctory(train_input_dir, target_train_dataset_dir, color_space="bgr")
        train_input_dataset.extend(train_input_images)
        train_target_dataset.extend(train_target_images)
    
    # Dataset과 DataLoader 생성
    train_dataset = CustomDataset(train_input_dataset, train_target_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataset, train_loader

def load_test_dataset_and_dataloader_32x32_bgr_each_qf(QF, batch_size, num_workers):
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "32x32")

    test_input_dataset = []
    test_target_dataset = []

    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    # 테스트 데이터 로드
    test_input_images, test_target_images = load_images_from_dirctory(test_input_dir, target_test_dataset_dir, color_space="bgr")
    test_input_dataset.extend(test_input_images)
    test_target_dataset.extend(test_target_images)

    # Dataset과 DataLoader 생성
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataset, test_loader

def load_test_dataset_and_dataloader_32x32_y_each_qf(QF, batch_size, num_workers):  
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "original_size")

    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    test_input_dataset, test_target_dataset = load_images_from_dirctory(test_input_dir, target_test_dataset_dir, color_space="gray")
    
    # Dataset과 DataLoader 생성
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataset, test_loader

def load_train_dataset_and_dataloader_32x32_y_all_qf(batch_size, num_workers):
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "original_size")

    train_input_dataset = []
    train_target_dataset = []

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

def load_train_dataset_and_dataloader_8x8_y_all_qf(batch_size, num_workers):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8_images")

    train_input_dataset = []
    train_target_dataset = []

    for QF in QFs:
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")

        for i in range(num_classes):
            train_path = os.path.join(train_input_dir, f"{i:03d}")
            target_train_path = os.path.join(target_train_dataset_dir, f"{i:03d}")
            print(f"QF{QF} {i}")
            sorted_train_files = sorted(os.listdir(train_path))
            sorted_target_train_files = sorted(os.listdir(target_train_path))

            for train_file, target_file in zip(sorted_train_files, sorted_target_train_files):
                if train_file == target_file:
                    train_image_path = os.path.join(train_path, train_file)
                    train_image = cv2.imread(train_image_path, cv2.IMREAD_GRAYSCALE)
                    # [8,8] --> [1,8,8]
                    train_image = np.expand_dims(train_image, axis=0)
                    train_image = train_image.astype(np.float32)/255
                    if train_image is None:
                        print(f"Warning: Failed to load input image: {train_image_path}")
                        sys.exit(1)
                    else:
                        train_input_dataset.append(train_image)

                    target_image_path = os.path.join(target_train_path, target_file)
                    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
                    # [8,8] --> [1,8,8]
                    target_image = np.expand_dims(target_image, axis=0)
                    target_image = target_image.astype(np.float32)/255
                    if target_image is None:
                        print(f"Warning: Failed to load target image: {target_image_path}")
                        sys.exit(1)
                    else:
                        train_target_dataset.append(target_image)
                else:
                    print(f"Warning: Mismatched files in training set: {train_file} and {target_file}")
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
                test_image = test_image.astype(np.float32)/255
                test_input_dataset.append(test_image)

                # target 이미지 로드
                target_image_path = os.path.join(target_test_path, target_file)
                target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
                target_image = np.expand_dims(target_image, axis=0)
                target_image = target_image.astype(np.float32)/255
                test_target_dataset.append(target_image)
            else:
                print(f"Warning: Mismatched files in testing set: {test_file} and {target_file}")
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataset, test_loader

def load_div2k_train_dataset_and_dataloader_224x224_rgb_all_qf(batch_size, num_workers):
    train_input_dataset = []
    train_target_dataset = []
    
    for qf in QFs:
        div2k_train_input_path = os.path.join(os.getcwd(), "datasets", "DIV2K", "224x224", f"jpeg{qf}", "train")
        div2k_train_target_path = os.path.join(os.getcwd(), "datasets", "DIV2K", "224x224", "original", "train")
        
        # 폴더에 있는 이미지들 imread로 로드
        for input_file_name, target_file_name in tqdm.tqdm(zip(sorted(os.listdir(div2k_train_input_path)), sorted(os.listdir(div2k_train_target_path))), desc=f"Loading train input and target data (QF {qf})", total=len(os.listdir(div2k_train_input_path))):
            input_img_path = os.path.join(div2k_train_input_path, input_file_name)
            target_img_path = os.path.join(div2k_train_target_path, target_file_name)
            input_img = cv2.imread(input_img_path)
            target_img = cv2.imread(target_img_path)
            if input_img is None or target_img is None:
                print(f"Error: Could not read image {input_img_path} or {target_img_path}")
                logging.error(f"Error: Could not read image {input_img_path} or {target_img_path}")
                continue
            input_img = input_img.transpose((2, 0, 1))
            input_img = input_img.astype(np.float32)/255.0
            train_input_dataset.append(input_img)
            target_img = target_img.transpose((2, 0, 1))
            target_img = target_img.astype(np.float32)/255.0
            train_target_dataset.append(target_img)
    
    if len(train_input_dataset) != len(train_target_dataset):
        print(f"Error: Number of train input and target images do not match")
        print("train_input_dataset", len(train_input_dataset))
        print("train_target_dataset", len(train_target_dataset))
        logging.error(f"Error: Number of train input and target images do not match")
        logging.error("train_input_dataset", len(train_input_dataset), "train_target_dataset", len(train_target_dataset))
    
    # Dataset과 DataLoader 생성
    train_dataset = CustomDataset(train_input_dataset, train_target_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_dataset, train_loader

def load_div2k_valid_dataset_and_dataloader_224x224_rgb_each_qf(QF, batch_size, num_workers):
    valid_input_dataset = []
    valid_target_dataset = []
    div2k_valid_input_path = os.path.join(".", "datasets", "DIV2K", "224x224", f"jpeg{QF}", "valid")
    div2k_valid_target_path = os.path.join(".", "datasets", "DIV2K", "224x224", "original", "valid")
    
    # 폴더에 있는 이미지들 imread로 로드
    for input_file_name, target_file_name in tqdm.tqdm(zip(sorted(os.listdir(div2k_valid_input_path)), sorted(os.listdir(div2k_valid_target_path))), desc=f"Loading valid input data (QF {QF})", total=len(os.listdir(div2k_valid_input_path))):
        input_img_path = os.path.join(div2k_valid_input_path, input_file_name)
        target_img_path = os.path.join(div2k_valid_target_path, target_file_name)
        input_img = cv2.imread(input_img_path)
        target_img = cv2.imread(target_img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # [h,w,c] -> [c,h,w]
        if input_img is None or target_img is None:
            print(f"Error: Could not read image {input_img_path} or {target_img_path}")
            logging.error(f"Error: Could not read image {input_img_path} or {target_img_path}")
            continue
        input_img = input_img.transpose((2, 0, 1))
        input_img = input_img.astype(np.float32)/255.0  
        valid_input_dataset.append(input_img)
    
        target_img = target_img.transpose((2, 0, 1))
        target_img = target_img.astype(np.float32)/255.0
        valid_target_dataset.append(target_img)
    
    if len(valid_input_dataset) != len(valid_target_dataset):
        print(f"Error: Number of valid input and target images do not match")
        print("valid_input_dataset", len(valid_input_dataset))
        print("valid_target_dataset", len(valid_target_dataset))
        logging.error(f"Error: Number of valid input and target images do not match")
        logging.error("valid_input_dataset", len(valid_input_dataset), "valid_target_dataset", len(valid_target_dataset))
    
    # Dataset과 DataLoader 생성
    valid_dataset = CustomDataset(valid_input_dataset, valid_target_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return valid_dataset, valid_loader