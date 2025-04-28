import numpy as np
import os, sys, logging
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips


def calculate_psnr_ssim_lpips(QF):
    original_dataset_dir  = os.path.join("datasets/mini-imagenet/_original/test")
    jpeg_dataset_dir = os.path.join(f"datasets/mini-imagenet/jpeg{QF}/test")

    original_images = []
    jpeg_images = []

    for i in range(1000):
        original_image_path = os.path.join(original_dataset_dir, str(i))
        jpeg_image_path = os.path.join(jpeg_dataset_dir, str(i))
        original_image_names = os.listdir(original_image_path)
        jpeg_image_names = os.listdir(jpeg_image_path)

        # print(original_images)
        # print(jpeg_images)

        for original_image_name, jpeg_image_name in zip(original_image_names, jpeg_image_names):
            # original_image_path =  os.path.join(original_image_path, original_image_name)
            # jpeg_image_path = os.path.join(jpeg_image_path, jpeg_image_name)
            original_image = Image.open(original_image_path)
            jpeg_image = Image.open(jpeg_image_path)
            original_images.append(original_image)
            jpeg_images.append(jpeg_image)

    lpips_metric = lpips.LPIPS(net="alex")
    lpips_values = []
    psnr_values = []
    ssim_values = []

    for i in range(len(original_images)):
        original_image = original_images[i]
        jpeg_image = jpeg_images[i]

        original_image = np.array(original_image)
        jpeg_image = np.array(jpeg_image)

        lpips_values.append(lpips_metric(original_image, jpeg_image))
        psnr_values.append(psnr(original_image, jpeg_image))
        ssim_values.append(ssim(original_image, jpeg_image))

    print(f"QF {QF}: PSNR = {np.mean(psnr_values)}, SSIM = {np.mean(ssim_values)}, LPIPS = {np.mean(lpips_values)}")
    logging.info(f"QF {QF}: PSNR = {np.mean(psnr_values)}, SSIM = {np.mean(ssim_values)}, LPIPS = {np.mean(lpips_values)}")
    return np.mean(psnr_values), np.mean(ssim_values), np.mean(lpips_values)

if __name__ == "__main__":
    QFs = ["100", "80", "60", "40", "20"]
    for QF in QFs:
        calculate_psnr_ssim_lpips(QF)
