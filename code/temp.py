import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import os

target_images = []
output_images = []
psnr_values = []
ssim_values = []
QFs = [100,80,60,40,20]
img_dir1 = "datasets/PxT_8x8_ycrcb_target/test"
# img_dir2 = "datasets/PxT_8x8_ycrcb_input/jpeg100/test"
# img_dir2 = "datasets/PxT_8x8_ycrcb_output/jpeg60/test"
# img_dir2 = "datasets/BlockCNN_rgb_cifar100/jpeg100"
for QF in QFs:
    img_dir2 = f"datasets/BlockCNN_rgb_cifar100/jpeg{QF}"
    psnr_values = []
    ssim_values = []
    for i in range(100):
        img1_files = sorted(os.listdir(f"{img_dir1}/{i:03d}"))
        img2_files = sorted(os.listdir(f"{img_dir2}/{i:03d}"))
        for img1_file, img2_file in zip(img1_files, img2_files):
            img1_path = os.path.join(f"{img_dir1}/{i:03d}", img1_file)
            img2_path = os.path.join(f"{img_dir2}/{i:03d}", img2_file)
            target_img = cv2.imread(img1_path, cv2.IMREAD_COLOR)
            output_img = cv2.imread(img2_path, cv2.IMREAD_COLOR)
            target_img = target_img / 255.0
            output_img = output_img / 255.0
            target_images.append(target_img)
            output_images.append(output_img)

            target_ycrcb = target_img
            output_ycrcb = output_img

            psnr_value = psnr(target_ycrcb, output_ycrcb, data_range=1.0)
            ssim_value = ssim(target_ycrcb, output_ycrcb, data_range=1.0, channel_axis=2)

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
    print(f"QF {QF}: AVG PSNR: {np.mean(psnr_values):.2f} dB | AVG SSIM: {np.mean(ssim_values):.4f}")
