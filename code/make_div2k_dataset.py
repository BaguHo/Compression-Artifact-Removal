import torch
import torchvision
import os
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from PIL import Image
import tqdm

dataset_name = "DIV2K"
# num_classes = 100
QFs = [100, 80, 60, 40, 20]

def crop_image(image, crop_size=224):
    width, height = image.size
    cropped_images = []

    for top in range(0, height, crop_size):
        for left in range(0, width, crop_size):
            right = min(left + crop_size, width)
            bottom = min(top + crop_size, height)
            cropped_img = image.crop((left, top, right, bottom))
            
            # If the cropped image is smaller than crop_size, pad it
            if cropped_img.size[0] < crop_size or cropped_img.size[1] < crop_size:
                # Create a new image with the target size
                # padded_img = Image.new("RGB", (crop_size, crop_size), color=(0, 0, 0))
                # # Calculate padding sizes
                # pad_left = (crop_size - cropped_img.size[0]) // 2
                # pad_top = (crop_size - cropped_img.size[1]) // 2
                # # Paste the original image in the center of the padded image
                # padded_img.paste(cropped_img, (pad_left, pad_top))
                # cropped_img = padded_img
                continue
            
            cropped_images.append(cropped_img)

    return cropped_images


def save_DIV2K():
    # datasets/DIV2K_train_HR,datasets/DIV2K_valid_HR 에서 불러오기
    train_div2k_file_path = os.path.join(".", "datasets", "DIV2K", "original_size", "original","train")
    valid_div2k_file_path = os.path.join(".", "datasets", "DIV2K", "original_size", "original","valid")
    train_output_path = os.path.join(".", "datasets", "DIV2K", "224x224", "original","train")
    valid_output_path = os.path.join(".", "datasets", "DIV2K", "224x224", "original","valid")
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(valid_output_path, exist_ok=True)
    div2k_train_files = os.listdir(train_div2k_file_path)
    div2k_valid_files = os.listdir(valid_div2k_file_path)
    
    # 파일 불러와서 224x224로 자르고 저장. 나눠 떨어지지 않으면 0패딩
    for file_name in tqdm.tqdm(div2k_train_files, desc=f"Saving original and 224x224 train images"):
        img_path = os.path.join(train_div2k_file_path, file_name)
        img = Image.open(img_path)
        if img is not None:
            width, height = img.size
            padded_img = Image.new("RGB", (max(width, 224), max(height, 224)))
            padded_img.paste(img)
            cropped_images = crop_image(padded_img)
            for idx, cropped_img in enumerate(cropped_images):
                if cropped_img is not None:
                    cropped_img.save(os.path.join(train_output_path,f"{os.path.splitext(file_name)[0]}_crop_{idx:05d}.png"))

    for file_name in tqdm.tqdm(div2k_valid_files, desc=f"Saving original and 224x224 valid images"):
        img_path = os.path.join(valid_div2k_file_path, file_name)
        img = Image.open(img_path)
        if img is not None:
            width, height = img.size
            padded_img = Image.new("RGB", (max(width, 224), max(height, 224)))
            padded_img.paste(img)
            cropped_images = crop_image(padded_img)
            for idx, cropped_img in enumerate(cropped_images):
                if cropped_img is not None:
                    cropped_img.save(os.path.join(valid_output_path,f"{os.path.splitext(file_name)[0]}_crop_{idx:05d}.png"))

    # 파일을 불러와서 jpeg으로 저장
    for QF in QFs:
        jpeg_224x224_train_output_dir = os.path.join(os.getcwd(), "datasets", "DIV2K", "224x224", f"jpeg{QF}", "train") 
        jpeg_224x224_valid_output_dir = os.path.join(os.getcwd(), "datasets", "DIV2K", "224x224", f"jpeg{QF}", "valid") 
        jpeg_original_train_output_dir = os.path.join(os.getcwd(), "datasets", "DIV2K", "original_size", f"jpeg{QF}", "train") 
        jpeg_original_valid_output_dir = os.path.join(os.getcwd(), "datasets", "DIV2K", "original_size", f"jpeg{QF}", "valid") 
        os.makedirs(jpeg_224x224_train_output_dir, exist_ok=True)
        os.makedirs(jpeg_224x224_valid_output_dir, exist_ok=True)
        os.makedirs(jpeg_original_train_output_dir, exist_ok=True)
        os.makedirs(jpeg_original_valid_output_dir, exist_ok=True)

        # 파일 불러와서 224x224로 자르고 각 jpeg qf에 맞게 저장. 나눠 떨어지지 않으면 0패딩
        for file_name in tqdm.tqdm(div2k_train_files, desc=f"Saving jpeg{QF} train images"):
            img_path = os.path.join(train_div2k_file_path, file_name)
            img = Image.open(img_path)
            if img is not None:
                img.save(os.path.join(jpeg_original_train_output_dir,f"{os.path.splitext(file_name)[0]}.jpeg"), "JPEG", quality=QF)
                width, height = img.size
                padded_img = Image.new("RGB", (max(width, 224), max(height, 224)), color=(0, 0, 0))
                padded_img.paste(img)
                cropped_images = crop_image(padded_img)
                for idx, cropped_img in enumerate(cropped_images):
                    if cropped_img is not None:
                        cropped_img.save(os.path.join(jpeg_224x224_train_output_dir,f"{os.path.splitext(file_name)[0]}_crop_{idx:05d}.jpeg"), "JPEG", quality=QF)

        for file_name in tqdm.tqdm(div2k_valid_files, desc=f"Saving jpeg{QF} valid images"):
            img_path = os.path.join(valid_div2k_file_path, file_name)
            img = Image.open(img_path)
            if img is not None:
                img.save(os.path.join(jpeg_original_valid_output_dir,f"{os.path.splitext(file_name)[0]}.jpeg"), "JPEG", quality=QF)
                width, height = img.size
                padded_img = Image.new("RGB", (max(width, 224), max(height, 224)), color=(0, 0, 0))
                padded_img.paste(img)
                cropped_images = crop_image(padded_img)
                for idx, cropped_img in enumerate(cropped_images):
                    if cropped_img is not None:
                        cropped_img.save(os.path.join(jpeg_224x224_valid_output_dir,f"{os.path.splitext(file_name)[0]}_crop_{idx:05d}.jpeg"), "JPEG", quality=QF)


    print("####################################################################")


if __name__ == "__main__":
    # save_CIFAR100()
    save_DIV2K()