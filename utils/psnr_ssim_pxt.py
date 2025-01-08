# PSNR and SSIM --> original - combined_ycbcr(PxT)
# avg PSNR: 13.9376
# avg SSIM: 0.3672
import torch
import numpy as np
import csv
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

dataset_name = "combined_ycbcr"
QFs = [80, 60, 40, 20]
batch_size = 128
num_classes = 5
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
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name)

    train_input_dataset = []
    test_input_dataset = []
    train_target_dataset = []
    test_target_dataset = []

    for QF in QFs:
        # input images
        train_input_dir = os.path.join(cifar100_path, f"QF_{QF}", "train")
        test_input_dir = os.path.join(cifar100_path, f"QF_{QF}", "test")

        # target images (original)
        target_train_dataset_dir = os.path.join(
            os.getcwd(), "datasets", "CIFAR100", "original_size", "original", "train"
        )
        target_test_dataset_dir = os.path.join(
            os.getcwd(), "datasets", "CIFAR100", "original_size", "original", "test"
        )

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
    psnr_scores, ssim_scores = [], []
    _, _, train_loader, test_loader = load_images_from_original_size()

    for jpeg_input_images_val, original_target_images in tqdm(
        test_loader, desc="Calculating PSNR and SSIM"
    ):
        jpeg_input_images_val, original_target_images = jpeg_input_images_val.to(
            device
        ), original_target_images.to(device)
        jpeg_input_images_val = jpeg_input_images_val.cpu().numpy()
        original_target_images = original_target_images.cpu().numpy()

        for i in range(len(jpeg_input_images_val)):
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
                        psnr(
                            jpeg_input_images_val[i],
                            original_target_images[i],
                        ),
                        ssim(
                            jpeg_input_images_val[i],
                            original_target_images[i],
                            channel_axis=0,
                            win_size=7,
                            data_range=1.0,
                        ),
                    ]
                )
            psnr_scores.append(
                psnr(
                    jpeg_input_images_val[i],
                    original_target_images[i],
                )
            )
            ssim_scores.append(
                ssim(
                    jpeg_input_images_val[i],
                    original_target_images[i],
                    channel_axis=0,
                    win_size=7,
                    data_range=1.0,
                )
            )

    print(f"PSNR: {np.mean(psnr_scores):.4f}")
    print(f"SSIM: {np.mean(ssim_scores):.4f}")
