import numpy as np
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_avg_psnr_ssim(original_image_dir, removed_image_dir):
    psnr_scores_original = []  # original vs original
    ssim_scores_original = []  # original vs original
    psnr_scores_removed = []  # original vs removed
    ssim_scores_removed = []  # original vs removed

    # Get list of files in each directory
    original_files = sorted(os.listdir(original_image_dir))
    removed_files = sorted(os.listdir(removed_image_dir))

    for original_file, removed_file in zip(original_files, removed_files):
        original_path = os.path.join(original_image_dir, original_file)
        removed_path = os.path.join(removed_image_dir, removed_file)

        # Load images and convert to numpy arrays
        original_image = np.array(Image.open(original_path))
        removed_image = np.array(Image.open(removed_path))

        # Calculate metrics between original and itself
        psnr_original = psnr(original_image, original_image)
        ssim_original = ssim(
            original_image, 
            original_image,
            win_size=3,  # 작은 이미지를 위해 win_size를 줄임
            channel_axis=2  # RGB 이미지의 경우 일반적으로 채널이 마지막 축(2)에 있음
        )

        # Calculate metrics between original and removed
        psnr_removed = psnr(original_image, removed_image)
        ssim_removed = ssim(
            original_image, 
            removed_image,
            win_size=3,  # 작은 이미지를 위해 win_size를 줄임
            channel_axis=2  # RGB 이미지의 경우 채널이 마지막 축에 있음
        )

        psnr_scores_original.append(psnr_original)
        ssim_scores_original.append(ssim_original)
        psnr_scores_removed.append(psnr_removed)
        ssim_scores_removed.append(ssim_removed)

    # Calculate averages
    avg_psnr_original = np.mean(psnr_scores_original)
    avg_ssim_original = np.mean(ssim_scores_original)
    avg_psnr_removed = np.mean(psnr_scores_removed)
    avg_ssim_removed = np.mean(ssim_scores_removed)

    return avg_psnr_original, avg_ssim_original, avg_psnr_removed, avg_ssim_removed


if __name__ == "__main__":
    QFs = [80, 60, 40, 20]
    num_classes = 20

    for QF in QFs:
        for i in range(num_classes):
            original_image_dir = os.path.join(
                os.getcwd(),
                "datasets",
                "CIFAR100",
                "original_size",
                f"jpeg{QF}",
                "test",
                str(i),
            )
            removed_image_dir = os.path.join(
                os.getcwd(), "datasets", "combined_ycbcr", f"QF_{QF}", "test", str(i)
            )

            avg_psnr_original, avg_ssim_original, avg_psnr_removed, avg_ssim_removed = (
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
                        "QF,class,avg_psnr_original,avg_ssim_original,avg_psnr_removed,avg_ssim_removed\n"
                    )
                f.write(
                    f"{QF},{i},{avg_psnr_original},{avg_ssim_original},{avg_psnr_removed},{avg_ssim_removed}\n"
                )
            print(
                f"QF: {QF}, class: {i}, avg_psnr_original: {avg_psnr_original}, avg_ssim_original: {avg_ssim_original}, avg_psnr_removed: {avg_psnr_removed}, avg_ssim_removed: {avg_ssim_removed}"
            )
