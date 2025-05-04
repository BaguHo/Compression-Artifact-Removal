import numpy as np
import os, sys, logging
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
import tqdm
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize(224,224)
    ]
)

batch_size = 1024
num_workers = 64
num_classes = 100

class CustomDataset(Dataset):
    def __init__(self, input_images, target_images, transform=transform):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image
    
    
def load_cifar100_test_dataset_and_dataloader(QF, dataset_name):
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "original_size")

    test_input_dataset = []
    test_target_dataset = []

    test_input_dir = os.path.join(f"./datasets/{dataset_name}/jpeg{QF}/test")
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    # 테스트 데이터 로드
    for i in tqdm.tqdm(range(num_classes), desc=f"Loading test data (QF {QF})"):
        test_path = os.path.join(test_input_dir, str(i).zfill(3))
        target_test_path = os.path.join(target_test_dataset_dir, str(i).zfill(3))

        # test_path 내 파일을 정렬된 순서로 불러오기
        sorted_test_files = sorted(os.listdir(test_path))
        sorted_target_test_files = sorted(os.listdir(target_test_path))

        # 두 디렉토리의 파일명이 같은지 확인하며 로드
        for input_file, target_file in zip(sorted_test_files, sorted_target_test_files):
            # if input_file == target_file:
            # input 이미지 로드
            test_image_path = os.path.join(test_path, input_file)
            test_image = cv2.imread(test_image_path)
            test_input_dataset.append(test_image)

            # target 이미지 로드
            target_image_path = os.path.join(target_test_path, target_file)
            target_image = cv2.imread(target_image_path)
            test_target_dataset.append(target_image)

            # else:
            #     print(
            #         f"Warning: Mismatched files in testing set: {input_file} and {target_file}"
            #     )

    # Dataset과 DataLoader 생성
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataset, test_loader


def calculate_psnr_ssim_lpips(QF, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset, test_loader = load_cifar100_test_dataset_and_dataloader(QF, dataset_name)
    
    psnr_values = []
    ssim_values = []
    lpips_values = []
    lpips_model = lpips.LPIPS(net="alex").to(device)
    
    with torch.no_grad():
        for input_images, target_images in tqdm.tqdm(test_loader, desc=f"Calculating PSNR and SSIM for QF {QF}"):
            for i in range(len(input_images)):
                input_image, target_image = input_images[i], target_images[i]
                lpips_alex = lpips_model(
                    torch.tensor(target_image).to(device), torch.tensor(input_image).to(device), normalize=True
                ).cpu().item()

                target_image = target_images[i].cpu().numpy()
                output_image = input_images[i].cpu().numpy()

                # Calculate PSNR
                psnr_value = psnr(
                    target_image, output_image, data_range=1.0
                )

                # Calculate SSIM
                ssim_value = ssim(
                    target_image, output_image, multichannel=True, data_range=1.0, channel_axis=0
                )

                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                lpips_values.append(lpips_alex)
    
    print(f"{dataset_name} QF {QF}: PSNR = {np.mean(psnr_values):.2f}, SSIM = {np.mean(ssim_values):.4f}, LPIPS = {np.mean(lpips_values):.4f}")
    logging.info(f"{dataset_name} QF {QF}: PSNR = {np.mean(psnr_values):.2f}, SSIM = {np.mean(ssim_values):.4f}, LPIPS = {np.mean(lpips_values):.4f}")
    return np.mean(psnr_values), np.mean(ssim_values), np.mean(lpips_values)

if __name__ == "__main__":
    QFs = ["100", "80", "60", "40", "20"]
    dataset_names = ["ARCNN_cifar100", "DnCNN_cifar100", "BlockCNN_cifar100", "PxT_v2_cifar100"]
    for QF in QFs:
        for dataset_name in dataset_names:  
            calculate_psnr_ssim_lpips(QF, dataset_name)
