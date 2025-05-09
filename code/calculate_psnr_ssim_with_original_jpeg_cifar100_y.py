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
import load_dataset

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)
batch_size = 4096
num_workers = 64
num_classes = 100

def calculate_psnr_ssim_lpips(QF):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset, test_loader = load_dataset.load_test_dataset_and_dataloader_8x8_y_each_qf(QF, batch_size, num_workers)
    
    psnr_values = []
    ssim_values = []
    lpips_values = []
    lpips_model = lpips.LPIPS(net="alex").to(device)
    
    with torch.no_grad():
        for input_images, target_images in tqdm.tqdm(test_loader, desc=f"Calculating PSNR and SSIM for QF {QF}"):
            for i in range(len(input_images)):
                input_image, target_image = input_images[i], target_images[i]
                # print("input_image.shape, target_image.shape", input_image.shape, target_image.shape)
                # input()
                # lpips_alex = lpips_model(
                #     torch.tensor(target_image).to(device), torch.tensor(input_image).to(device), normalize=True
                # ).cpu().item()

                target_image = target_images[i].cpu().numpy()
                output_image = input_images[i].cpu().numpy()
                
                target_image = (target_image*255.0).astype(np.uint8)
                output_image = (output_image*255.0).astype(np.uint8)
                # Calculate PSNR
                psnr_value = psnr(
                    target_image, output_image, data_range=255.0
                )

                # Calculate SSIM
                ssim_value = ssim(
                    target_image, output_image, data_range=255.0, channel_axis=0
                )

                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                # lpips_values.append(lpips_alex)
    
    print(f"CIFAR100 original-jpeg QF {QF}: PSNR = {np.mean(psnr_values):.2f}, SSIM = {np.mean(ssim_values):.4f}")
    logging.info(f"CIFAR100 original-jpeg QF {QF}: PSNR = {np.mean(psnr_values):.2f}, SSIM = {np.mean(ssim_values):.4f}")
    return np.mean(psnr_values), np.mean(ssim_values), None

if __name__ == "__main__":
    QFs = ["100", "80", "60", "40", "20"]
    for QF in QFs:
        calculate_psnr_ssim_lpips(QF)
