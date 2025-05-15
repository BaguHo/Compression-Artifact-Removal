import numpy as np
import os, sys, logging
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
from load_dataset import CustomDataset
from load_dataset import load_dataset_and_dataloader_each_qf

if len(sys.argv) < 2:
    print("Usage: python script.py <color_space(bgr, ycrcb, y)>")
    sys.exit(1)

color_space = sys.argv[1]
logging.basicConfig(filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s")
batch_size = 1024
num_workers = 24
num_classes = 100

def calculate_psnr_ssim_lpips(QF):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset, test_loader = load_dataset_and_dataloader_each_qf(QF, is_train=False, color_space=color_space, size="32x32", batch_size=batch_size, num_workers=num_workers)
    
    psnr_values = []
    ssim_values = []
    lpips_values = []   
    lpips_model = lpips.LPIPS(net="alex").to(device)
    
    with torch.no_grad():
        for input_images, target_images in tqdm.tqdm(test_loader, desc=f"Calculating PSNR and SSIM for QF {QF}"):
            for i in range(len(input_images)):
                input_image, target_image = input_images[i], target_images[i]
                # print("1",input_image.shape, target_image.shape)
                # [3, 32,32]
                input_image = (input_image.cpu().numpy()*255).astype(np.uint8).transpose((1,2,0))
                target_image = (target_image.cpu().numpy()*255).astype(np.uint8).transpose((1,2,0))
                
                lpips_alex = lpips_model(torch.tensor(target_image).permute(2,0,1).to(device), torch.tensor(input_image).permute(2,0,1).to(device), normalize=True).cpu().item()

                psnr_value = psnr(target_image, input_image, data_range=255)
                ssim_value = ssim(target_image, input_image, multichannel=True, data_range=255, channel_axis=2)

                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                lpips_values.append(lpips_alex)
    
    print(f"CIFAR100 original-jpeg {color_space} QF {QF}: PSNR = {np.mean(psnr_values):.2f}, SSIM = {np.mean(ssim_values):.4f}, LPIPS = {np.mean(lpips_values):.4f}")
    logging.info(f"CIFAR100 original-jpeg {color_space} QF {QF}: PSNR = {np.mean(psnr_values):.2f}, SSIM = {np.mean(ssim_values):.4f}, LPIPS = {np.mean(lpips_values):.4f}")
    return np.mean(psnr_values), np.mean(ssim_values), np.mean(lpips_values)

if __name__ == "__main__":
    QFs = ["100", "80", "60", "40", "20"]
    for QF in QFs:
        calculate_psnr_ssim_lpips(QF)
