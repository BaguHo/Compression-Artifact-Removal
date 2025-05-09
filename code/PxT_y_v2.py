from skimage.metrics import structural_similarity, peak_signal_noise_ratio
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
import load_dataset

import argparse

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

def get_args():
    parser = argparse.ArgumentParser(description="PxT Training Arguments")
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--img_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--mlp_dim', type=int, default=384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_name', type=str, default='PxT_y')
    return parser.parse_args()

dataset_name = "CIFAR100"
slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)
QFs = [100, 80, 60, 40, 20]
num_classes = 100


class CustomDataset(Dataset):
    def __init__(self, input_images, target_images, transform=transforms.ToTensor()):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        input_image = torch.from_numpy(input_image).float()/255.0
        target_image = torch.from_numpy(target_image).float()/255.0
        return input_image, target_image


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(Encoder, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        # Multi-head self-attention with residual connection
        y = self.ln1(x)
        y, _ = self.mhsa(y, y, y)
        x = x + y

        # MLP with residual connection
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + y
        return x

#  loss 값이 0.0001이 되도록 학습
class PxT(nn.Module):
    def __init__(
        self,
        img_size=8,
        patch_size=1,
        in_channels=1,
        embed_dim=12, # 192,
        num_heads=12,
        num_layers=12,
        mlp_dim=384,
    ):
        super(PxT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        print('pos embedding:', self.position_embeddings)

        self.transformer = nn.ModuleList(
            [Encoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        self.decoder = nn.Sequential(nn.Linear(embed_dim, self.patch_dim))

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, self.patch_dim)

        x = self.patch_embed(x)
        x = x + self.position_embeddings

        x = x.permute(1, 0, 2)
        for layer in self.transformer:
            x = layer(x)
        x = x.permute(1, 0, 2)

        x = self.decoder(x)

        x = x.view(
            batch_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, self.in_channels, self.img_size, self.img_size)
        return x


if __name__ == "__main__":
    args = get_args()
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    model_name = args.model_name

    # Load the dataset
    train_dataset, train_loader = load_dataset.load_train_dataset_and_dataloader_8x8_y_all_qf(batch_size, num_workers)

    # Initialize the model with argparse params
    model = PxT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_dim=args.mlp_dim
    )
    print(model)

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

    # train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
        # loss, PxT의 파라미터 logging
        logging.info(
            f"{model_name} Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, img_size: {args.img_size}, patch_size: {args.patch_size}, in_channels: {args.in_channels}, embed_dim: {args.embed_dim}, num_heads: {args.num_heads}, num_layers: {args.num_layers}, mlp_dim: {args.mlp_dim}"
        )
        # Save the best model with all hyperparameters in filename
        if epoch == 0:
            best_loss = epoch_loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            param_str = (
                f"img{args.img_size}_patch{args.patch_size}_in{args.in_channels}_ed{args.embed_dim}"
                f"_nh{args.num_heads}_nl{args.num_layers}_mlp{args.mlp_dim}_lr{args.lr}"
            )
            model_save_name = f"{model_name}_{param_str}_best.pth"
            save_path = os.path.join("models", model_save_name)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved: {save_path} (Loss: {best_loss:.6f})")
            logging.info(f"Best model saved: {save_path} (Loss: {best_loss:.6f})")

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
        test_dataset, test_loader = load_dataset.load_test_dataset_and_dataloader_8x8_y_each_qf(QF,batch_size,num_workers)
        with torch.no_grad():
            combined_target_images = []
            combined_output_images = []
            y_target_patches = []
            y_output_patches = []
            patch_idx = 0
            class_idx = 0
            image_name_idx = 0
            
            for input_images, target_images in tqdm.tqdm(test_loader, desc="Testing"):
                input_images = input_images.to(device)
                target_images = target_images.to(device)

                # Forward pass
                outputs = model(input_images)

                # Calculate MSE loss
                loss = criterion(outputs, target_images)
                test_loss += loss.item()

                for i in range(len(outputs)):
                    y_target = target_images[i].cpu().numpy()
                    y_output = outputs[i].cpu().numpy()
                    np.clip(y_target, 0, 1, out=y_target)
                    np.clip(y_output, 0, 1, out=y_output)
                    y_target = np.transpose(y_target, (1, 2, 0)) * 255
                    y_output = np.transpose(y_output, (1, 2, 0)) * 255
                    y_target_patches.append(y_target)
                    y_output_patches.append(y_output)
                    patch_idx += 1

                    # 8x8 이미지들을 32x32로 합치기
                    if patch_idx % 16 == 0 and patch_idx > 0:
                        patch_idx = 0
                        combined_target_image = np.zeros((32, 32, 1), dtype=np.uint8)
                        combined_output_image = np.zeros((32, 32, 1), dtype=np.uint8)

                        for j in range(4):
                            for k in range(4):
                                idx = j * 4 + k
                                combined_target_image[
                                    j * 8 : (j + 1) * 8, k * 8 : (k + 1) * 8
                                ] = y_target_patches[idx]
                        y_target_patches.clear()

                        for j in range(4):
                            for k in range(4):
                                idx = j * 4 + k
                                combined_output_image[
                                    j * 8 : (j + 1) * 8, k * 8 : (k + 1) * 8
                                ] = y_output_patches[idx]

                        y_output_patches.clear()
                        combined_target_images.append(combined_target_image)
                        combined_output_images.append(combined_output_image)

                        # Calculate PSNR and SSIM
                        psnr = peak_signal_noise_ratio(
                            combined_target_image,
                            combined_output_image,
                            data_range=255,
                        )
                        ssim = structural_similarity(
                            combined_target_image,
                            combined_output_image,
                            data_range=255,
                        )
                        lpips_alex_loss = lpips_loss_alex(
                            torch.from_numpy(combined_output_image)
                            .permute(2, 0, 1)
                            .to(device),
                            torch.from_numpy(combined_target_image)
                            .permute(2, 0, 1)
                            .to(device),
                        )

                        lpips_alex_loss_values.append(lpips_alex_loss.item())
                        psnr_values.append(psnr)
                        ssim_values.append(ssim)

                        image_name_idx += 1
                        os.makedirs(
                            os.path.join(
                                "datasets",
                                f"{model_name}",
                                f"jpeg{QF}",
                                "test",
                                f"{class_idx:03d}",
                            ),
                            exist_ok=True,
                        )
                        combined_output_image_path = os.path.join(
                            "datasets",
                            f"{model_name}",
                            f"jpeg{QF}",
                            "test",
                            f"{class_idx:03d}",
                            f"combined_output{image_name_idx}.png",
                        )
                        cv2.imwrite(combined_output_image_path, combined_output_image)
                        if image_name_idx % 100 == 0 and image_name_idx > 0:
                            class_idx += 1
                            image_name_idx = 0

        # Calculate average metrics
        avg_test_loss = test_loss / len(test_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_lpips_alex = np.mean(lpips_alex_loss_values)

        print(
            f"{model_name} QF: {QF} | Test Loss: {avg_test_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.6f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.6f}"
        )
        logging.info(
            f"{model_name} QF:{QF} | Test Loss: {avg_test_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {np.mean(ssim_values):.6f} | LPIPS Alex: {np.mean(lpips_alex_loss_values):.6f}"
        )
