# =============================================================================
#  @article{zhang2017beyond,
#    title={Compression Artifacts Reduction by a Deep Convolutional Network},
#    author={Chao Dong, Yubin Deng, Chen Change Loy, and Xiaoou Tang},
#    journal={Proceedings of the  IEEE International Conference on Computer Vision (ICCV)},
#    year={2015}
#  }
# =============================================================================

# 메인 터미널에서 실행
# avg PSNR: 33.3950
# avg SSIM: 0.9090

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import re
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import csv, os
from tqdm import tqdm
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)

model_output_dir = "./post-processing/models/"

num_epochs = 2
num_classes = 20
QFs = [80, 60, 40, 20]
batch_size = 128

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def sort_key(filename):
    image_match = re.search(r"image_(\d+)", filename)
    crop_match = re.search(r"crop_(\d+)", filename)

    image_number = int(image_match.group(1)) if image_match else float("inf")
    crop_number = int(crop_match.group(1)) if crop_match else float("inf")

    return (image_number, crop_number)


class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU(),
        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x


class CIFAR100Dataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        # print(np.array(self.input_images).shape)
        # print(np.array(self.target_images).shape)

        input_image = Image.fromarray(input_image)
        target_image = Image.fromarray(target_image)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def load_images_from_original_size():
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "original_size")

    train_input_dataset = []
    test_input_dataset = []
    train_target_dataset = []
    test_target_dataset = []

    for QF in QFs:
        # input images
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

        # target images (original)
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")
        target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

        # 학습 데이터 로드
        for i in range(num_classes):
            train_path = os.path.join(train_input_dir, str(i))
            target_train_path = os.path.join(target_train_dataset_dir, str(i))

            # train_path 내 파일을 정렬된 순서로 불러오기
            sorted_train_file_names = sorted(os.listdir(train_path), key=sort_key)
            sorted_target_train_file_names = sorted(
                os.listdir(target_train_path), key=sort_key
            )

            for train_file, target_file in zip(
                sorted_train_file_names, sorted_target_train_file_names
            ):
                # input 이미지 로드
                train_image_path = os.path.join(train_path, train_file)
                train_image = Image.open(train_image_path).convert("YCbCr")
                train_image = np.array(train_image, dtype=np.uint8)
                train_input_dataset.append(train_image)

                # target 이미지 로드
                target_image_path = os.path.join(target_train_path, target_file)
                target_image = Image.open(target_image_path).convert("YCbCr")
                target_image = np.array(target_image, dtype=np.uint8)
                train_target_dataset.append(target_image)

        # 테스트 데이터 로드
        for i in range(num_classes):
            test_path = os.path.join(test_input_dir, str(i))
            target_test_path = os.path.join(target_test_dataset_dir, str(i))

            # test_path 내 파일을 정렬된 순서로 불러오기
            sorted_test_files = sorted(os.listdir(test_path), key=sort_key)
            sorted_target_test_files = sorted(
                os.listdir(target_test_path), key=sort_key
            )

            # 두 디렉토리의 파일명이 같은지 확인하며 로드
            for test_file, target_file in zip(
                sorted_test_files, sorted_target_test_files
            ):
                # input 이미지 로드
                test_image_path = os.path.join(test_path, test_file)
                test_image = Image.open(test_image_path).convert("YCbCr")
                test_image = np.array(test_image, dtype=np.uint8)
                test_input_dataset.append(np.array(test_image))

                # target 이미지 로드
                target_image_path = os.path.join(target_test_path, target_file)
                test_target_images = Image.open(target_image_path).convert("YCbCr")
                test_target_images = np.array(test_target_images, dtype=np.uint8)
                test_target_dataset.append(np.array(test_target_images))

    # Dataset과 DataLoader 생성
    train_dataset = CIFAR100Dataset(
        train_input_dataset, train_target_dataset, transform=transform
    )
    test_dataset = CIFAR100Dataset(
        test_input_dataset, test_target_dataset, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    train_dataset, test_dataset, train_loader, test_loader = (
        load_images_from_original_size()
    )

    model = ARCNN()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    best_val_loss = float("inf")

    # TODO: Train
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for jpeg_input_images, original_target_images in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            jpeg_input_images, original_target_images = jpeg_input_images.to(
                device
            ), original_target_images.to(device)
            output = model(jpeg_input_images)
            loss = criterion(output, original_target_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss / len(train_loader):.4f}"
        )

        model.eval()
        val_loss = 0.0

        with torch.nograd():
            for jpeg_input_images_val, original_target_images_val in tqdm(
                test_loader, desc="Validation"
            ):
                jpeg_input_images_val, original_target_images_val = (
                    jpeg_input_images_val.to(device),
                    original_target_images_val.to(device),
                )
                outputs_val = model(original_target_images_val)
                loss = criterion(outputs_val, jpeg_input_images_val)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(model_output_dir, "Best_ARCNN_Model.pth"),
            )
            print(f"New best model saved with validation loss: {val_loss:.4f}")

    # TODO: Test
    model.load_state_dict(
        torch.load(os.path.join(model_output_dir, "Best_ARCNN_Model.pth"))
    )

    model.eval()

    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for jpeg_input_images_val, original_target_images in tqdm(
            test_loader, desc="Testing"
        ):
            jpeg_input_images_val, original_target_images = jpeg_input_images_val.to(
                device
            ), original_target_images.to(device)
            outputs_test = model(original_target_images)
            # show outputs image
            plt.imshow(outputs_test[0].cpu().detach().numpy().transpose(1, 2, 0))
            plt.show()
            input()

            outputs_test = outputs_test.cpu().numpy()
            jpeg_input_images_val = jpeg_input_images_val.cpu().numpy()

            for i in range(len(outputs_test)):
                log_dir = "./psnr_log"
                log_file = os.path.join(log_dir, "PxT.csv")
                os.makedirs(log_dir, exist_ok=True)
                file_exists = os.path.isfile(log_file)
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["PSNR", "SSIM"])
                    writer.writerow(
                        [
                            psnr(jpeg_input_images_val[i], outputs_test[i]),
                            ssim(
                                jpeg_input_images_val[i],
                                outputs_test[i],
                                channel_axis=0,
                                win_size=3,
                                data_range=1.0,
                            ),
                        ]
                    )
                psnr_scores.append(psnr(jpeg_input_images_val[i], outputs_test[i]))
                ssim_scores.append(
                    ssim(
                        jpeg_input_images_val[i],
                        outputs_test[i],
                        channel_axis=0,
                        win_size=3,
                        data_range=1.0,
                    )
                )

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Average PSNR: 33.0046
    # Average SSIM: 0.9213
