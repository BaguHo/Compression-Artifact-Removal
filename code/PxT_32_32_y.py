from skimage.metrics import (structural_similarity as ssim, peak_signal_noise_ratio as psnr)
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from torch import nn
import torch
import os, sys, re
import logging
import tqdm
import time
import lpips
import load_dataset
from models import PxT_32x32_y, PxT_32x32_y_improved
import cv2
from torchmetrics.functional import structural_similarity_index_measure
from vit_base import vit_base

if len(sys.argv) < 4:
    print("Usage: python script.py <epoch> <batch_size> <num_workers>")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

dataset_name = "CIFAR100"
slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)
QFs = [100,80,60,40,20]
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = 100
model_name = "vit_base"

# def ssim_loss(output, target):
#     # SSIM 값이 1에 가까울수록 유사, 0에 가까울수록 다름
#     return 1 - structural_similarity_index_measure(output, target)

if __name__ == "__main__":
    # Load the dataset
    train_dataset, train_loader = load_dataset.load_train_dataset_and_dataloader_32x32_y_all_qf(batch_size, num_workers)

    model = vit_base()
    # print(model)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    model.to(device)
    print(f"Model device: {device}")

    # ! models{model_name}_20.pth 불러오기
    # model.load_state_dict(torch.load(os.path.join("models", f"{model_name}_20.pth")))

    # train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    print(f"Training started at {time.ctime(start_time)}")
    logging.info(f"Training started at {time.ctime(start_time)}")
    print(f"Training for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (input_images, target_images) in enumerate(
            tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")
        ):
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_images)
            loss = criterion(outputs, target_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
        logging.info(
            f"{model_name} Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}"
        )
        # Save the model
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join("models", f"{model_name}_{epoch+1:03d}.pth"),
            )
            print(f"{model_name} Model saved at epoch {epoch+1}")
            logging.info(f"{model_name} Model saved at epoch {epoch+1}")
    end_time = time.time()
    print(f"Training finished at {time.ctime(end_time)}")
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    logging.info(f"Training finished at {time.ctime(end_time)}")
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Test the model
    model.eval()
    image_name_idx = 0
    combined_image_idx = 0
    combined_output_images = []
    test_loss = 0.0
    psnr_values = []
    ssim_values = []
    lpips_loss_alex = lpips.LPIPS(net="alex").to(device)
    lpips_alex_loss_values = []

    for QF in QFs:
        test_dataset, test_loader = load_dataset.load_test_dataset_and_dataloader_32x32_y_each_qf(QF, batch_size, num_workers)
        with torch.no_grad():
            combined_target_images = []
            combined_output_images = []
            rgb_target_patches = []
            rgb_output_patches = []
            patch_idx = 0
            image_name_idx = 0
            class_idx = 0

            for input_images, target_images in tqdm.tqdm(test_loader, desc="Testing"):
                input_images = input_images.to(device)
                target_images = target_images.to(device)

                # Forward pass
                outputs = model(input_images)

                # Calculate SSIM loss
                loss = criterion(outputs, target_images)
                test_loss += loss.item()

                for i in range(len(outputs)):
                    y_target = target_images[i].cpu().numpy()
                    y_output = outputs[i].cpu().numpy()
                    y_input = input_images[i].cpu().numpy()
                    # [c,h,w] --> [h,w,c] 
                    # print("before transpose", y_target.shape, y_output.shape, y_input.shape)
                    y_target = (y_target * 255).astype(np.uint8).transpose(1,2,0)
                    y_output = (y_output * 255).astype(np.uint8).transpose(1,2,0)
                    y_input = (y_input * 255).astype(np.uint8).transpose(1,2,0)
                    # print("after transpose", y_target.shape, y_output.shape, y_input.shape)
                    # Calculate PSNR and SSIM
                    psnr_value = psnr(
                        y_target,
                        y_output,
                        data_range=255,
                    )
                    ssim_value = ssim(
                        y_target,
                        y_output,
                        data_range=255,
                        channel_axis=2,
                    )
                    lpips_alex_loss = lpips_loss_alex(
                        torch.from_numpy(y_output)
                        .permute(2,0,1)
                        .to(device),
                        torch.from_numpy(y_target)
                        .permute(2,0,1)
                        .to(device),
                    )

                    lpips_alex_loss_values.append(lpips_alex_loss.item())
                    psnr_values.append(psnr_value)
                    ssim_values.append(ssim_value)

                    image_name_idx += 1
                    
                    os.makedirs(
                        os.path.join(
                            "datasets",
                            f"{model_name}_input",
                            f"jpeg{QF}",
                            "test",
                            f"{class_idx:03d}",
                        ),
                        exist_ok=True,
                        )
                    os.makedirs(
                        os.path.join(
                            "datasets",
                            f"{model_name}_target",
                            f"original",
                            "test",
                            f"{class_idx:03d}",
                        ),
                        exist_ok=True,
                        )
                    os.makedirs(
                        os.path.join(
                            "datasets",
                            f"{model_name}_output",
                            f"jpeg{QF}",
                            "test",
                            f"{class_idx:03d}",
                        ),
                        exist_ok=True,
                        )
                    
                    input_image_path = os.path.join(
                        "datasets",
                        f"{model_name}_input",
                        f"jpeg{QF}",
                        "test",
                        f"{class_idx:03d}",
                        f"input_{image_name_idx:05d}.png",
                        )
                    
                    target_image_path = os.path.join(
                        "datasets",
                        f"{model_name}_target",
                        f"jpeg{QF}",
                        "test",
                        f"{class_idx:03d}",
                        f"target_{image_name_idx:05d}.png",
                        )
                    
                    output_image_path = os.path.join(
                        "datasets",
                        f"{model_name}_output",
                        f"jpeg{QF}",
                        "test",
                        f"{class_idx:03d}",
                        f"output_{image_name_idx:05d}.png",
                        )
                    cv2.imwrite(target_image_path, y_target)
                    cv2.imwrite(input_image_path, y_input)
                    cv2.imwrite(output_image_path, y_output)
                    
                    if image_name_idx % 100 == 0 and image_name_idx > 0:
                        class_idx += 1
                        image_name_idx = 0

        # Calculate average metrics
        avg_test_loss = test_loss / len(test_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_lpips_alex = np.mean(lpips_alex_loss_values)

        print(
            f"{model_name} QF: {QF} | Test Loss: {avg_test_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.4f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.4f}"
        )
        logging.info(
            f"{model_name} QF:{QF} | Test Loss: {avg_test_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.4f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.4f}"
        )
