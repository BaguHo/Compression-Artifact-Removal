import os
import cv2
import numpy as np

def combine_8x8_patches_to_32x32(image_patches):
    combined_image = np.zeros((32, 32, 3), dtype=np.uint8)
    for j in range(4):
        for k in range(4):
            idx = j * 4 + k
            combined_image[
                j * 8 : (j + 1) * 8, k * 8 : (k + 1) * 8
            ] = image_patches[idx]
    return combined_image

def save_combined_images(QF, image_name_idx, output_dir_path, image):
    os.makedirs(output_dir_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir_path,f"{image_name_idx:05d}.png"),image)
