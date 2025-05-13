from skimage.metrics import structural_similarity, peak_signal_noise_ratio
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
import cv2
from load_dataset import load_test_dataset_and_dataloader_32x32_ycbcr_each_qf,load_train_dataset_and_dataloader_32x32_ycbcr_all_qf
from models import PxT_32x32_ycbcr

if len(sys.argv) < 4:
    print("Usage: python script.py <epoch> <batch_size> <num_workers>")
    sys.exit(1)

logging.basicConfig(filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s")

QFs = [100, 80, 60, 40, 20]
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
model_name = "PxT_32x32_ycbcr"

if __name__ == "__main__":
    train_dataset, train_loader = load_train_dataset_and_dataloader_32x32_ycbcr_all_qf(batch_size, num_workers)

    model = PxT_32x32_ycbcr()
    # print(model.__class__.__name__)

    device = torch.device("cuda"if torch.cuda.is_available()else "mps" if torch.backends.mps.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    model.to(device)
    print(f"Model device: {device}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.load_state_dict(torch.load(os.path.join("models", f"{model_name}_20.pth")))
    # start_time = time.time()
    # print(f"Training started at {time.ctime(start_time)}")
    # logging.info(f"Training started at {time.ctime(start_time)}")
    # print(f"Training for {epochs} epochs")

    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for i, (input_images, target_images) in enumerate(tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")):
    #         # [c,h,w]
    #         input_images = input_images.to(device)
    #         target_images = target_images.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(input_images)
    #         loss = criterion(outputs, target_images)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item() * input_images.size(0)
    #     epoch_loss = running_loss / len(train_loader.dataset)
    #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    #     logging.info(f"{model_name} Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    #     # Save the model
    #     if (epoch + 1) % 10 == 0:
    #         torch.save(model.state_dict(),os.path.join("models", f"{model_name}_{epoch+1}.pth"),)
    #         print(f"{model_name} Model saved at epoch {epoch+1}")
    #         logging.info(f"{model_name} Model saved at epoch {epoch+1}")
    # end_time = time.time()
    # print(f"Training finished at {time.ctime(end_time)}")
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.2f} seconds")
    # logging.info(f"Training finished at {time.ctime(end_time)}")
    # logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

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
        test_dataset, test_loader = load_test_dataset_and_dataloader_32x32_ycbcr_each_qf(QF, batch_size, num_workers)
        with torch.no_grad():
            combined_target_images = []
            combined_output_images = []
            rgb_target_patches = []
            rgb_output_patches = []
            patch_idx = 0
            image_name_idx = 0
            class_idx = 0

            for input_images, target_images in tqdm.tqdm(test_loader, desc="Testing"):
                #[c,h,w]
                input_images = input_images.to(device)
                target_images = target_images.to(device)

                outputs = model(input_images)
                loss = criterion(outputs, target_images)
                test_loss += loss.item() * input_images.size(0)

                for i in range(len(outputs)):
                    ycrcb_input = input_images[i].cpu().numpy()
                    ycrcb_output = outputs[i].cpu().numpy()
                    ycrcb_target = target_images[i].cpu().numpy()
                    # [3,32,32] --> [32,32,3] 
                    ycrcb_input = ycrcb_input.transpose(1, 2, 0) 
                    ycrcb_output = ycrcb_output.transpose(1, 2, 0) 
                    ycrcb_target = ycrcb_target.transpose(1, 2, 0) 
                    
                    psnr = peak_signal_noise_ratio(ycrcb_target,ycrcb_output,data_range=1.0)
                    ssim = structural_similarity(ycrcb_target,ycrcb_output,data_range=1.0,channel_axis=2)
                    lpips_alex_loss = lpips_loss_alex(torch.from_numpy(ycrcb_output).permute(2, 0, 1).to(device),torch.from_numpy(ycrcb_target).permute(2, 0, 1).to(device))

                    # convert ycrcb to bgr
                    bgr_target = cv2.cvtColor(ycrcb_target, cv2.COLOR_YCrCb2BGR)
                    bgr_output = cv2.cvtColor(ycrcb_output, cv2.COLOR_YCrCb2BGR)
                    bgr_input = cv2.cvtColor(ycrcb_input, cv2.COLOR_YCrCb2BGR)
                    
                    lpips_alex_loss_values.append(lpips_alex_loss.item())
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)

                    image_name_idx += 1
                    
                    os.makedirs(os.path.join("datasets",f"{model_name}_input",f"jpeg{QF}","test",f"{class_idx:03d}",),exist_ok=True)
                    os.makedirs(os.path.join("datasets",f"{model_name}_output",f"jpeg{QF}","test",f"{class_idx:03d}",),exist_ok=True)
                    os.makedirs(os.path.join("datasets",f"{model_name}_target",f"jpeg{QF}","test",f"{class_idx:03d}",),exist_ok=True)
                    
                    input_image_path = os.path.join("datasets",f"{model_name}_input",f"jpeg{QF}","test",f"{class_idx:03d}",f"input{image_name_idx:05d}.png",)
                    output_image_path = os.path.join("datasets",f"{model_name}_output",f"jpeg{QF}","test",f"{class_idx:03d}",f"output{image_name_idx:05d}.png",)
                    target_image_path = os.path.join("datasets",f"{model_name}_target",f"jpeg{QF}","test",f"{class_idx:03d}",f"target{image_name_idx:05d}.png",)
                    
                    # [0,1] --> [0,255]
                    bgr_input, bgr_output, bgr_target = (bgr_input*255).astype(np.uint8), (bgr_output*255).astype(np.uint8), (bgr_target*255).astype(np.uint8)
                    
                    cv2.imwrite(input_image_path, bgr_input)
                    cv2.imwrite(output_image_path, bgr_output)
                    cv2.imwrite(target_image_path, bgr_target)
                    
                    if image_name_idx % 100 == 0 and image_name_idx > 0:
                        class_idx += 1
                        image_name_idx = 0

        # Calculate average metrics
        avg_test_loss = test_loss / len(test_loader.dataset)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_lpips_alex = np.mean(lpips_alex_loss_values)

        print(f"{model_name} QF: {QF} | Test Loss: {avg_test_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.6f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.6f}")
        logging.info(f"{model_name} QF:{QF} | Test Loss: {avg_test_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.6f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.6f}")
