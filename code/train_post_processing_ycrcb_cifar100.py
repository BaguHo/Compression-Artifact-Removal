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
from load_dataset import load_train_dataset_and_dataloader_32x32_ycrcb_all_qf, load_test_dataset_and_dataloader_32x32_ycrcb_each_qf
from models import ARCNN, DnCNN, BlockCNN

if len(sys.argv) < 4:
    print("Usage: python script.py <epoch> <batch_size> <num_workers>")
    sys.exit(1)

logging.basicConfig(filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s")

dataset_name = "CIFAR100"
slack_webhook_url = "https://hooks.slack.com/services/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"
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = 100
seed = 42
torch.manual_seed(seed)

if __name__ == "__main__":
    QFs = [100, 80, 60, 40, 20]

    # Load the dataset
    train_dataset, train_loader = load_train_dataset_and_dataloader_32x32_ycrcb_all_qf(batch_size=batch_size, num_workers=num_workers)
    
    model_names = [
        "BlockCNN",
        "ARCNN",
        "DnCNN",
    ]

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
            model = torch.nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPUs")

        model.to(device)
        print(f"Model device: {device}")

        # train the model
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # # !load model
        # model.load_state_dict(torch.load(f"./models/{type(model).__name__}_30.pth"))

        start_time = time.time()
        print(f"Training started at {time.ctime(start_time)}")
        logging.info(f"{model_name} Training started at {time.ctime(start_time)}")
        print(f"Training for {epochs} epochs")
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

            # Save the model
            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                torch.save(model.state_dict(), os.path.join("models", f"{model_name}_ycrcb_cifar100_{epoch+1}.pth"))
                print(f"Model saved at epoch {epoch+1}")
                logging.info(f"Model {model_name} saved at epoch {epoch+1}")

        end_time = time.time()
        print(f"Training finished at {time.ctime(end_time)}")
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        logging.info(f"Training finished at {time.ctime(end_time)}")
        logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        # Save the final model
        torch.save(model.state_dict(), os.path.join("models", f"{model_name}_ycrcb_cifar100_final.pth"))
        print(f"Final model saved as {model_name}_ycrcb_cifar100_final.pth")
        logging.info(f"Final model saved as {model_name}_ycrcb_cifar100_final.pth")

        # test for each JPEG QF
        for QF in QFs:
            test_dataset, test_dataloader = load_test_dataset_and_dataloader_32x32_ycrcb_each_qf(QF, batch_size=batch_size, num_workers=num_workers)
            model.eval()
            idx = 0
            test_loss = 0.0
            psnr_values = []
            ssim_values = []
            lpips_alex_values = []
            lpips_alex_model = lpips.LPIPS(net="alex").to(device)

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
                        ycrcb_target = target_images[i].cpu().numpy()
                        ycrcb_output = outputs[i].cpu().numpy()
                        ycrcb_input = input_images[i].cpu().numpy()
                        # [c,h,w] --> [h,w,c]
                        ycrcb_target = (ycrcb_target * 255).astype(np.uint8).transpose(1, 2, 0)
                        ycrcb_output = (ycrcb_output * 255).astype(np.uint8).transpose(1, 2, 0)
                        ycrcb_input = (ycrcb_input * 255).astype(np.uint8).transpose(1, 2, 0)

                        # Calculate PSNR and SSIM
                        psnr_value = psnr(ycrcb_target, ycrcb_output, data_range=255)
                        ssim_value = ssim(ycrcb_target, ycrcb_output, data_range=255, channel_axis=2)

                        lpips_alex_values.append(lpips_value)
                        psnr_values.append(psnr_value)
                        ssim_values.append(ssim_value)

            # Calculate average metrics
            avg_test_loss = test_loss / len(test_dataloader.dataset)
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            avg_lpips_alex = np.mean(lpips_alex_values)

            print(f"Model: {model_name}, Epoch: {epochs}, QF: {QF}, Training Time: {time.ctime(end_time)}, Test Loss: {avg_test_loss:.4f}, Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}, Average LPIPS Alex: {avg_lpips_alex:.4f}")
            logging.info(f"Model: {model_name}, Epoch: {epochs}, QF: {QF}, Training Time: {time.ctime(end_time)}, Test Loss: {avg_test_loss:.4f}, Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}, Average LPIPS Alex: {avg_lpips_alex:.4f}")
