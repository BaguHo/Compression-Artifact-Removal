from skimage.metrics import structural_similarity, peak_signal_noise_ratio
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

if len(sys.argv) < 3:
    print("Usage: python script.py <batch_size> <num_workers> ")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

dataset_name = "CIFAR100"
slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)

batch_size = int(sys.argv[1])
num_workers = int(sys.argv[2])
num_classes = 1000


def sort_key(filename):
    image_match = re.search(r"image_(\d+)", filename)
    crop_match = re.search(r"crop_(\d+)", filename)

    image_number = int(image_match.group(1)) if image_match else float("inf")
    crop_number = int(crop_match.group(1)) if crop_match else float("inf")

    return (image_number, crop_number)


def calculate_psnr_ssim_with_original_jpeg():
    QFs = [100, 80, 60, 40, 20]
    dataset_name = "mini-imagenet"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name)

    test_input_dataset = []
    test_target_dataset = []

    for QF in QFs:
        # input images
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

        # target images (original)
        target_train_dataset_dir = os.path.join(cifar100_path, "_original", "train")
        target_test_dataset_dir = os.path.join(cifar100_path, "_original", "test")

        # 테스트 데이터 로드
        for i in tqdm.tqdm(range(num_classes), desc=f"Loading test data (QF {QF})"):
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
                if test_file.replace("jpeg", "png") == target_file:
                    # input 이미지 로드
                    test_file.replace("png", "jpeg")
                    test_image_path = os.path.join(test_path, test_file)
                    test_image = cv2.imread(test_image_path)
                    test_input_dataset.append(test_image)

                    # target 이미지 로드
                    target_image_path = os.path.join(target_test_path, target_file)
                    target_image = cv2.imread(target_image_path)
                    test_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in testing set: {test_file} and {target_file}"
                    )

        # test_input_dataaset, test_target_dataset의 psnr, ssim 계산 후 original_jpeg.csv에 저장
        psnr_values = []
        ssim_values = []
        for i in range(len(test_input_dataset)):
            input_image = test_input_dataset[i]
            target_image = test_target_dataset[i]

            psnr_values.append(
                peak_signal_noise_ratio(input_image, target_image, data_range=255)
            )
            ssim_values.append(
                structural_similarity(
                    input_image,
                    target_image,
                    data_range=255,
                    multichannel=True,
                    channel_axis=2,
                )
            )
            logging.info(
                f"original-jpeg Image {i}: PSNR = {psnr_values[-1]}, SSIM = {ssim_values[-1]}"
            )
        print(f"Avg PSNR: {np.mean(psnr_values)}, Avg SSIM: {np.mean(ssim_values)}")
        logging.info(
            f"Avg PSNR: {np.mean(psnr_values)}, Avg SSIM: {np.mean(ssim_values)}"
        )
        os.makedirs("metrics", exist_ok=True)

        # Save the results to a CSV file
        with open("metrics/original_jpeg.csv", "w") as f:
            f.write("PSNR,SSIM\n")
            for psnr, ssim in zip(psnr_values, ssim_values):
                f.write(f"{psnr},{ssim}\n")
        logging.info("Results saved to original_jpeg.csv")


if __name__ == "__main__":
    calculate_psnr_ssim_with_original_jpeg()
