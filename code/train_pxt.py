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
from load_dataset import load_train_dataset_and_dataloader_8x8_ycrcb_all_qf, load_test_dataset_and_dataloader_8x8_ycrcb_each_qf, load_dataset_and_dataloader_all_qf, load_dataset_and_dataloader_each_qf
from utils import combine_8x8_patches_to_32x32, save_combined_images
from models import PxT_8x8_ycrcb

if len(sys.argv) < 5:
    print("Usage: python script.py <color_space(bgr,ycrcb,y)> <epoch> <batch_size> <num_workers>")
    sys.exit(1)

logging.basicConfig(filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s")

seed = 42
torch.manual_seed(seed)
QFs = [100, 80, 60, 40, 20]
color_space = sys.argv[1]
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
num_workers = int(sys.argv[4])
num_classes = 100
model_name = "PxT_8x8"

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
    # model.load_state_dict(torch.load(os.path.join("models", f"{model_name}_{color_space}_final.pth")))
    # model.load_state_dict(torch.load(os.path.join("models", f"{model_name}_{color_space}_3.pth")))
    
    
    start_time = time.time()
    print(f"Training started at {time.ctime(start_time)}")
    logging.info(f"Training started at {time.ctime(start_time)}")

    # load train dataset
    # train_dataset, train_loader = load_train_dataset_and_dataloader_8x8_ycrcb_all_qf(color_space=color_space,batch_size=batch_size,num_workers=num_workers)
    train_dataset, train_loader = load_dataset_and_dataloader_all_qf(is_train=True, color_space=color_space, size="8x8", batch_size=batch_size,num_workers=num_workers)
    
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
            torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{color_space}_best.pth"),)
            print(f"{model_name} Model saved at epoch {epoch+1} with best loss: {epoch_loss:.6f}")
            logging.info(f"{model_name} Model saved at epoch {epoch+1} with best loss: {epoch_loss:.6f}")
        if epoch + 1 == epochs:
            torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{color_space}_{epochs}.pth"),)
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
        # test_dataset, test_loader = load_test_dataset_and_dataloader_8x8_ycrcb_each_qf(QF,color_space=color_space,batch_size=batch_size,num_workers=num_workers)
        test_dataset, test_loader = load_dataset_and_dataloader_each_qf(QF,is_train=False,color_space=color_space,size="8x8",batch_size=batch_size,num_workers=num_workers)
        with torch.no_grad():
            target_patches = []
            output_patches = []
            input_patches = []
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
                    target_image = target_images[i].cpu().numpy()
                    output_image = outputs[i].cpu().numpy()
                    input_image = input_images[i].cpu().numpy()
                    np.clip(target_image, 0, 1, out=target_image)
                    np.clip(output_image, 0, 1, out=output_image)
                    np.clip(input_image, 0, 1, out=input_image)
                    # [c,h,w] --> [h,w,c]
                    target_image = (target_image * 255).astype(np.uint8).transpose(1, 2, 0)
                    output_image = (output_image * 255).astype(np.uint8).transpose(1, 2, 0)
                    input_image = (input_image * 255).astype(np.uint8).transpose(1, 2, 0)
                    target_patches.append(target_image)
                    output_patches.append(output_image)
                    input_patches.append(input_image)
                    patch_idx += 1

                    # 8x8 이미지들을 32x32로 합치기
                    if patch_idx % 16 == 0 and patch_idx > 0:
                        patch_idx = 0
                        combined_target_image = combine_8x8_patches_to_32x32(target_patches)
                        combined_output_image = combine_8x8_patches_to_32x32(output_patches)
                        combined_input_image = combine_8x8_patches_to_32x32(input_patches)
                        target_patches.clear()
                        output_patches.clear()
                        input_patches.clear()

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

                        if color_space == "ycrcb":
                            combined_input_image =cv2.cvtColor(combined_input_image, cv2.COLOR_YCrCb2BGR)
                            combined_output_image =cv2.cvtColor(combined_output_image, cv2.COLOR_YCrCb2BGR)
                            combined_target_image =cv2.cvtColor(combined_target_image, cv2.COLOR_YCrCb2BGR)

                        image_name_idx += 1
                        save_combined_images(QF, image_name_idx, os.path.join("datasets",f"{model_name}_{color_space}_input",f"jpeg{QF}","test",f"{class_idx:03d}"), combined_input_image)
                        save_combined_images(QF, image_name_idx, os.path.join("datasets",f"{model_name}_{color_space}_output",f"jpeg{QF}","test",f"{class_idx:03d}"), combined_output_image)
                        save_combined_images(QF, image_name_idx, os.path.join("datasets",f"{model_name}_{color_space}_target",f"test",f"{class_idx:03d}"), combined_target_image)
                        
                        if image_name_idx % 100 == 0 and image_name_idx > 0:
                            class_idx += 1
                            image_name_idx = 0

        # Calculate average metrics
        avg_test_loss = test_loss / len(test_loader.dataset)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_lpips_alex = np.mean(lpips_alex_loss_values)

        print(f"{model_name} {color_space} QF: {QF} | Epoch: {epochs} | Test Loss: {avg_test_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.4f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.4f}")
        logging.info(f"{model_name} {color_space} QF:{QF} | Epoch: {epochs} | Test Loss: {avg_test_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.4f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.4f}")
