import numpy as np
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch

def calculate_avg_psnr_ssim(original_image_dir, removed_image_dir):
    lpips_model = lpips.LPIPS(net='alex')
    psnr_scores_removed = []  # original vs removed
    ssim_scores_removed = []  # original vs removed
    lpips_scores_removed = []  # original vs removed

    # Get list of files in each directory
    original_files = sorted(os.listdir(original_image_dir))
    removed_files = sorted(os.listdir(removed_image_dir))

    for original_file, removed_file in zip(original_files, removed_files):
        original_path = os.path.join(original_image_dir, original_file)
        removed_path = os.path.join(removed_image_dir, removed_file)

        # Load images and convert to numpy arrays
        original_image = np.array(Image.open(original_path))
        removed_image = np.array(Image.open(removed_path))

        # Calculate metrics between original and removed
        psnr_removed = psnr(original_image, removed_image)
        ssim_removed = ssim(
            original_image, 
            removed_image,
            win_size=3,  # 작은 이미지를 위해 win_size를 줄임
            channel_axis=2  # RGB 이미지의 경우 채널이 마지막 축에 있음
        )

        # Calculate LPIPS between original and removed
        lpips_removed = lpips_model(torch.from_numpy(original_image), torch.from_numpy(removed_image))

        psnr_scores_removed.append(psnr_removed)
        ssim_scores_removed.append(ssim_removed)
        lpips_scores_removed.append(lpips_removed)

    # Calculate averages
    avg_psnr_removed = np.mean(psnr_scores_removed)
    avg_ssim_removed = np.mean(ssim_scores_removed)
    avg_lpips_removed = np.mean(lpips_scores_removed)

    return avg_psnr_removed, avg_ssim_removed, avg_lpips_removed

    
if __name__ == "__main__":
    QFs = [100, 80, 60, 40, 20]
    num_classes = 1000

    for QF in QFs:
        for i in range(num_classes):
            original_image_dir = os.path.join(
                os.getcwd(),
                "datasets",
                "mini-imagenet",
                f"_original",
                "test",
                str(i),
            )
            removed_image_dir = os.path.join(
                os.getcwd(), "datasets", "mini-imagenet", f"jpeg{QF}", "test", str(i)
            )

            avg_psnr_removed, avg_ssim_removed, avg_lpips_removed = (
                calculate_avg_psnr_ssim(original_image_dir, removed_image_dir)
            )

            with open(
                os.path.join(os.getcwd(), "results", "psnr_ssim_results.csv"), "a"
            ) as f:
                if (
                    os.path.getsize(
                        os.path.join(os.getcwd(), "results", "psnr_ssim_results.csv")
                    )
                    == 0
                ):
                    f.write(
                        "QF,class,avg_psnr_removed,avg_ssim_removed,avg_lpips_removed\n"
                    )
                f.write(
                    f"{QF},{i},{avg_psnr_removed},{avg_ssim_removed},{avg_lpips_removed}\n"
                )
            print(
                f"QF: {QF}, class: {i}, avg_psnr_removed: {avg_psnr_removed}, avg_ssim_removed: {avg_ssim_removed}, avg_lpips_removed: {avg_lpips_removed}"
            )
