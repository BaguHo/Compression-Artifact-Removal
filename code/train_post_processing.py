from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import nn
import torch
import torchvision.transforms as transforms
import numpy as np
import os, sys
import logging
import cv2
import tqdm
import time
import lpips
from load_dataset import load_train_dataset_and_dataloader_32x32_bgr_all_qf, load_test_dataset_and_dataloader_32x32_bgr_each_qf, load_dataset_and_dataloader_all_qf, load_dataset_and_dataloader_each_qf
from models import ARCNN, DnCNN, BlockCNN

if len(sys.argv) < 5:
    print("Usage: python script.py <color_space(bgr,rgb,ycrcb,y)> <epoch> <batch_size> <num_workers>")
    sys.exit(1)

logging.basicConfig(filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s")

dataset_name = "CIFAR100"
slack_webhook_url = "https://hooks.slack.com/services/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"
color_space = sys.argv[1]
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
num_workers = int(sys.argv[4])
num_classes = 100
seed = 42
torch.manual_seed(seed)
today_date = time.strftime("%Y%m%d")

if __name__ == "__main__":
    QFs = [100, 80, 60, 40, 20]

    model_names = [
        # "BlockCNN",
        # "ARCNN",
        "DnCNN",
    ]

    for model_name in model_names:
        if model_name == "ARCNN":
            model = ARCNN()
            batch_size = 4096
        elif model_name == "DnCNN":
            model = DnCNN()
            batch_size = 512
        elif model_name == "BlockCNN":
            model = BlockCNN()
            batch_size = 4096
        print(model.__class__.__name__)

        # load training dataset and dataloader
        train_dataset, train_loader = load_dataset_and_dataloader_all_qf(is_train=True, color_space=color_space, size="32x32", batch_size=batch_size, num_workers=num_workers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPUs")

        model.to(device)
        print(f"Model device: {device}")

        # train the model
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if os.path.exists(os.path.join("models", f"{model_name}_{color_space}_{today_date}_checkpoint.pth")):
            model.load_state_dict(torch.load(os.path.join("models", f"{model_name}_{color_space}_{today_date}_checkpoint.pth")))
            print("Model loaded from checkpoint")
        start_time = time.time()
        print(f"Training started at {time.ctime(start_time)}")
        logging.info(f"{model_name} Training started at {time.ctime(start_time)}")
        print(f"Training for {epochs} epochs")
        best_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (input_images, target_images) in enumerate(tqdm.tqdm(train_loader, desc=f"{model_name} Train Epoch {epoch+1}/{epochs}")):
                input_images = input_images.to(device)
                target_images = target_images.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(input_images)
                loss = criterion(outputs, target_images)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * input_images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
            logging.info(f"{model_name} Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{color_space}_{today_date}_best.pth"))
                print(f"{model_name} Model saved at epoch {epoch+1} with best loss: {epoch_loss:.6f}")
                logging.info(f"{model_name} Model saved at epoch {epoch+1} with best loss: {epoch_loss:.6f}")
            # Save the model
            torch.save(model.state_dict(), os.path.join("models", f"{model_name}_{color_space}_{today_date}_checkpoint.pth"))

        end_time = time.time()
        print(f"Training finished at {time.ctime(end_time)}")
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        logging.info(f"Training finished at {time.ctime(end_time)}")
        logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        # test for each JPEG QF
        for QF in QFs:
            # test_dataset, test_dataloader = load_test_dataset_and_dataloader_32x32_bgr_each_qf(QF, batch_size=batch_size, num_workers=num_workers)
            test_dataset, test_dataloader = load_dataset_and_dataloader_each_qf(is_train=False, color_space=color_space, size="32x32", QF=QF, batch_size=batch_size, num_workers=num_workers)
            model.eval()
            idx = 0
            test_loss = 0.0
            psnr_values = []
            ssim_values = []
            lpips_alex_values = []
            lpips_alex_model = lpips.LPIPS(net="alex").to(device)
            class_idx = 0
            image_name_idx = 0
            with torch.no_grad():
                for input_images, target_images in tqdm.tqdm(test_dataloader, desc=f"{model_name} Testing QF{QF}"):
                    input_images = input_images.to(device)
                    target_images = target_images.to(device)

                    # Forward pass
                    outputs = model(input_images)

                    # Calculate MSE loss
                    loss = criterion(outputs, target_images)
                    test_loss += loss.item() * input_images.size(0)

                    for i in range(len(outputs)):
                        # Calculate LPIPS
                        lpips_value = lpips_alex_model(target_images[i], outputs[i], normalize=True).item()
                        target_img = target_images[i].cpu().numpy()
                        output_img = outputs[i].cpu().numpy()
                        input_img = input_images[i].cpu().numpy()
                        np.clip(target_img, 0, 1, out=target_img)
                        np.clip(output_img, 0, 1, out=output_img)
                        np.clip(input_img, 0, 1, out=input_img)
                        # [c,h,w] --> [h,w,c]
                        target_img = (target_img * 255).astype(np.uint8).transpose(1, 2, 0)
                        output_img = (output_img * 255).astype(np.uint8).transpose(1, 2, 0)
                        input_img = (input_img * 255).astype(np.uint8).transpose(1, 2, 0)

                        # Calculate PSNR and SSIM
                        psnr_value = psnr(target_img, output_img, data_range=255)
                        ssim_value = ssim(target_img, output_img, data_range=255, channel_axis=2)
                        lpips_alex_values.append(lpips_value)
                        psnr_values.append(psnr_value)
                        ssim_values.append(ssim_value)

                        test_output_image_dir = os.path.join("datasets", f"{model_name}_{color_space}_{today_date}",f"jpeg{QF}", "test",f"{class_idx:03d}")
                        # [c,h,w] --> [h,w,c]
                        if color_space == "ycrcb":
                            output_img = cv2.cvtColor(output_img, cv2.COLOR_YCrCb2BGR)
                        elif color_space == "rgb":
                            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                        os.makedirs(test_output_image_dir, exist_ok=True)
                        image_name_idx += 1
                        cv2.imwrite(os.path.join(test_output_image_dir, f"output_{image_name_idx:05d}.png"), output_img)
                        if image_name_idx % 100 == 0:
                            image_name_idx = 0
                            class_idx +=1

            # Calculate average metrics
            avg_test_loss = test_loss / len(test_dataloader.dataset)
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            avg_lpips_alex = np.mean(lpips_alex_values)

            print(f"Model: {model_name}_{color_space}, Epoch: {epochs}, QF: {QF}, Test Loss: {avg_test_loss:.4f}, Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}, Average LPIPS Alex: {avg_lpips_alex:.4f}")
            logging.info(f"Model: {model_name}_{color_space}, Epoch: {epochs}, QF: {QF}, Test Loss: {avg_test_loss:.4f}, Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}, Average LPIPS Alex: {avg_lpips_alex:.4f}")

        for QF in QFs:
            train_dataset, train_loader = load_dataset_and_dataloader_each_qf(is_train=True, color_space=color_space, size="32x32", QF=QF, batch_size=batch_size, num_workers=num_workers)
            class_idx = 0
            image_name_idx = 0
            with torch.no_grad():
                for input_images, target_images in tqdm.tqdm(train_loader, desc=f"{model_name} Make training images QF{QF}"):
                    input_images = input_images.to(device)
                    target_images = target_images.to(device)

                    # Forward pass
                    outputs = model(input_images)
                    for i in range(len(outputs)):
                        # Calculate LPIPS
                        output_img = outputs[i].cpu().numpy()
                        np.clip(output_img, 0, 1, out=output_img)
                        # [c,h,w] --> [h,w,c]
                        output_img = (output_img * 255).astype(np.uint8).transpose(1, 2, 0)

                        train_output_image_dir = os.path.join("datasets", f"{model_name}_{color_space}_{today_date}",f"jpeg{QF}", "train",f"{class_idx:03d}")
                        # [c,h,w] --> [h,w,c]
                        if color_space == "ycrcb":
                            output_img = cv2.cvtColor(output_img, cv2.COLOR_YCrCb2BGR)
                        elif color_space == "rgb":
                            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                        os.makedirs(train_output_image_dir, exist_ok=True)
                        image_name_idx += 1
                        cv2.imwrite(os.path.join(train_output_image_dir, f"output_{image_name_idx:05d}.png"), output_img)
                        if image_name_idx % 500 == 0:
                            image_name_idx = 0
                            class_idx +=1