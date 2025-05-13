from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from torch import nn
import torch
import os, sys, re
import logging
import cv2
import tqdm
import time
import lpips
from load_dataset import load_train_dataset_and_dataloader_8x8_ycrcb_all_qf, load_test_dataset_and_dataloader_8x8_ycrcb_each_qf
from utils import combine_8x8_patches_to_32x32, save_combined_images
from models import PxT_8x8_ycrcb

if len(sys.argv) < 4:
    print("Usage: python script.py <epoch> <batch_size> <num_workers>")
    sys.exit(1)

logging.basicConfig(filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s")

seed = 42
torch.manual_seed(seed)
QFs = [100, 80, 60, 40, 20]
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = 100
model_name = "PxT_8x8_ycrcb"

if __name__ == "__main__":
    # Load the dataset

    # Initialize the model
    best_loss = float('inf')
    model = PxT_8x8_ycrcb()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    model.to(device)
    print(f"Model device: {device}")

    # train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    #! load model
    # model.load_state_dict(torch.load(os.path.join("models", f"{model_name}_best.pth")))
    start_time = time.time()
    print(f"Training started at {time.ctime(start_time)}")
    logging.info(f"Training started at {time.ctime(start_time)}")

    # load train dataset
    train_dataset, train_loader = load_train_dataset_and_dataloader_8x8_ycrcb_all_qf(batch_size=batch_size,num_workers=num_workers)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (input_images, target_images) in enumerate(tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")):
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_images)
            loss = criterion(outputs, target_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * (input_images.size(0)*1.0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
        # loss, PxT의 파라미터 logging
        logging.info(f"{model_name} Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
        # Save the model if it's the best loss so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join("models", f"{model_name}_best.pth"),)
            print(f"{model_name} Model saved at epoch {epoch+1} with best loss: {epoch_loss:.6f}")
            logging.info(f"{model_name} Model saved at epoch {epoch+1} with best loss: {epoch_loss:.6f}")
        if epoch + 1 == epochs:
            torch.save(model.state_dict(), os.path.join("models", f"{model_name}_final.pth"),)
            print(f"{model_name} Model saved at epoch {epoch+1}")
            logging.info(f"{model_name} Model saved at epoch {epoch+1}")
    end_time = time.time()
    print(f"Training finished at {time.ctime(end_time)}")
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    logging.info(f"Training finished at {time.ctime(end_time)}")
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    for QF in QFs:
        model.eval()
        image_name_idx = 0
        combined_image_idx = 0
        lpips_loss_alex = lpips.LPIPS(net="alex").to(device)
        lpips_alex_loss_values = []
        test_loss = 0.0
        psnr_values = []
        ssim_values = []
        test_dataset, test_loader = load_test_dataset_and_dataloader_8x8_ycrcb_each_qf(QF,batch_size=batch_size,num_workers=num_workers)
        with torch.no_grad():
            y_target_patches = []
            y_output_patches = []
            y_input_patches = []
            patch_idx = 0
            class_idx = 0
            image_name_idx = 0
            
            for input_images, target_images in tqdm.tqdm(test_loader, desc=f"Testing QF {QF}"):
                input_images = input_images.to(device)
                target_images = target_images.to(device)

                # Forward pass
                outputs = model(input_images)

                # Calculate MSE loss
                loss = criterion(outputs, target_images)
                test_loss += loss.item() * input_images.size(0)

                for i in range(len(outputs)):
                    # [c,h,w]
                    y_target = target_images[i].cpu().numpy()
                    y_output = outputs[i].cpu().numpy()
                    y_input = input_images[i].cpu().numpy()
                    np.clip(y_target, 0, 1, out=y_target)
                    np.clip(y_output, 0, 1, out=y_output)
                    np.clip(y_input, 0, 1, out=y_input)
                    # [c,h,w] --> [h,w,c]
                    y_target = (y_target * 255).astype(np.uint8).transpose(1, 2, 0)
                    y_output = (y_output * 255).astype(np.uint8).transpose(1, 2, 0)
                    y_input = (y_input * 255).astype(np.uint8).transpose(1, 2, 0)
                    y_target_patches.append(y_target)
                    y_output_patches.append(y_output)
                    y_input_patches.append(y_input)
                    patch_idx += 1

                    # 8x8 이미지들을 32x32로 합치기
                    if patch_idx % 16 == 0 and patch_idx > 0:
                        patch_idx = 0
                        combined_target_image = combine_8x8_patches_to_32x32(y_target_patches)
                        combined_output_image = combine_8x8_patches_to_32x32(y_output_patches)
                        combined_input_image = combine_8x8_patches_to_32x32(y_input_patches)

                        # Calculate PSNR and SSIM
                        psnr_value = psnr(
                            combined_target_image,
                            combined_output_image,
                            data_range=255,
                        )
                        ssim_value = ssim(
                            combined_target_image,
                            combined_output_image,
                            data_range=255,
                            channel_axis=2,
                        )
                        lpips_alex_loss = lpips_loss_alex(
                            torch.from_numpy(combined_output_image).permute(2, 0, 1).to(device),
                            torch.from_numpy(combined_target_image).permute(2, 0, 1).to(device),
                        )

                        lpips_alex_loss_values.append(lpips_alex_loss.item())
                        psnr_values.append(psnr_value)
                        ssim_values.append(ssim_value)
                        
                        image_name_idx += 1
                        combined_input_image =cv2.cvtColor(combined_input_image, cv2.COLOR_YCrCb2BGR)
                        combined_output_image =cv2.cvtColor(combined_output_image, cv2.COLOR_YCrCb2BGR)
                        combined_target_image =cv2.cvtColor(combined_target_image, cv2.COLOR_YCrCb2BGR)
                        
                        save_combined_images(QF, class_idx, image_name_idx, os.path.join("datasets",f"{model_name}_input",f"jpeg{QF}","train",f"{class_idx:03d}"), combined_input_image)
                        save_combined_images(QF, class_idx, image_name_idx, os.path.join("datasets",f"{model_name}_output",f"jpeg{QF}","train",f"{class_idx:03d}"), combined_output_image)
                        save_combined_images(QF, class_idx, image_name_idx, os.path.join("datasets",f"{model_name}_target",f"train",f"{class_idx:03d}"), combined_target_image)
                        
                        if image_name_idx % 100 == 0 and image_name_idx > 0:
                            class_idx += 1
                            image_name_idx = 0

        # Calculate average metrics
        avg_test_loss = test_loss / len(test_loader.dataset)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_lpips_alex = np.mean(lpips_alex_loss_values)

        print(f"{model_name} QF: {QF} | Test Loss: {avg_test_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.4f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.4f}")
        logging.info(f"{model_name} QF:{QF} | Test Loss: {avg_test_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.4f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.4f}")
