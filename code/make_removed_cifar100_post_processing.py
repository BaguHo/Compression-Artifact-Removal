from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import os, sys
import logging
import cv2
import tqdm
import time
from load_dataset import load_dataset_and_dataloader_each_qf
from models import DnCNN, BlockCNN,ARCNN

if len(sys.argv) < 4:
    print("Usage: python script.py <color_space> <batch_size> <num_workers> ")
    sys.exit(1)

logging.basicConfig(filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s")

QFs = [100, 80, 60, 40, 20]

model_names = [
    "ARCNN",
    "DnCNN",
    "BlockCNN",
]
color_space = sys.argv[1]
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = 100
seed = 42
torch.manual_seed(seed)

if __name__ == "__main__":
    for QF in QFs:
        # Load the dataset
        train_dataset, train_loader = load_dataset_and_dataloader_each_qf(QF= QF,is_train=True, color_space=color_space, size="32x32", batch_size=batch_size, num_workers=num_workers)
        test_dataset, test_loader = load_dataset_and_dataloader_each_qf(QF= QF,is_train=False, color_space=color_space, size="32x32", batch_size=batch_size, num_workers=num_workers)
        
        # Initialize the model
        for model_name in model_names:
            if model_name == "ARCNN":
                model = ARCNN()
            elif model_name == "DnCNN":
                model = DnCNN()
            elif model_name == "BlockCNN":
                model = BlockCNN()
            print(model.__class__.__name__)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # use multiple GPUs if available
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                print(f"Using {torch.cuda.device_count()} GPUs")

            model.to(device)
            print(f"Model device: {device}")

            # !load model
            model.load_state_dict(torch.load(f"./models/{model_name}_20.pth", map_location=device))

            model.eval()
            train_loss = 0.0
            test_loss = 0.0
            psnr_values = []
            ssim_values = []
            psnr_b_values = []

            with torch.no_grad():
                image_idx = 0
                class_idx = 0

                for input_images, target_images in tqdm.tqdm(train_loader, desc=f"Making Train Images QF {QF}"):
                    input_images = input_images.to(device)
                    target_images = target_images.to(device)

                    # Forward pass
                    outputs = model(input_images)

                    for i in range(len(outputs)):
                        rgb_output = outputs[i].cpu().numpy()
                        rgb_output = (rgb_output.transpose(1, 2, 0) * 255).astype(np.uint8)
                        image_idx += 1
                        output_directory_path = f"datasets/{model_name}_{color_space}_cifar100/jpeg{QF}/train/{class_idx:03d}"
                        os.makedirs(output_directory_path,exist_ok=True)
                        if color_space == "rgb":
                            bgr_output = cv2.cvtColor(rgb_output, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f"{output_directory_path}/image_{image_idx:05d}.png", bgr_output)
                        if image_idx % 500 == 0 and image_idx > 0:
                            image_idx = 0
                            class_idx += 1
                image_idx = 0
                class_idx = 0
                for input_images, target_images in tqdm.tqdm(test_loader, desc=f"Making Test Images QF {QF}"):
                    input_images = input_images.to(device)
                    target_images = target_images.to(device)
                    outputs = model(input_images)

                    for i in range(len(outputs)):
                        rgb_output = outputs[i].cpu().numpy()
                        rgb_output = (rgb_output.transpose(1, 2, 0) * 255).astype(np.uint8)
                        image_idx += 1
                        output_directory_path = f"datasets/{model_name}_{color_space}_cifar100/jpeg{QF}/test/{class_idx:03d}"
                        os.makedirs(output_directory_path,exist_ok=True)
                        if color_space == "rgb":
                            bgr_output = cv2.cvtColor(rgb_output, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f"{output_directory_path}/image_{image_idx:05d}.png", bgr_output)
                        if image_idx % 100 == 0 and image_idx > 0:
                            image_idx = 0
                            class_idx += 1
                
                print(f"make removed images for {model_name} QF {QF}")
                logging.info(f"make removed images for {model_name} QF {QF}")

