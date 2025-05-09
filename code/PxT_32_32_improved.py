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
from PIL import Image

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
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = 100
model_name = "PxT_32_32_improved_ycbcr"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class CustomDataset(Dataset):
    def __init__(self, input_images, target_images, transform=transform):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def laod_cifar100_train_dataset_and_dataloader():
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "original_size")

    train_input_dataset = []
    train_target_dataset = []

    QFs = [100, 80, 60, 40, 20]
    for QF in QFs:
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")

        # 학습 데이터 로드
        for i in tqdm.tqdm(
            range(num_classes), desc=f"Loading train data (QF {QF})", total=num_classes
        ):
            train_path = os.path.join(train_input_dir, f"{i:03d}")
            target_train_path = os.path.join(target_train_dataset_dir, f"{i:03d}")

            # train_path 내 파일을 정렬된 순서로 불러오기
            sorted_train_files = sorted(os.listdir(train_path))
            sorted_target_train_files = sorted(
                os.listdir(target_train_path)
            )
            # 두 디렉토리의 파일명이 같은지 확인하며 로드
            for input_file, target_file in zip(
                sorted_train_files, sorted_target_train_files
            ):
                if input_file.replace("jpeg", "png") == target_file:
                    # input 이미지 로드
                    train_image_path = os.path.join(train_path, input_file)
                    train_image = Image.open(train_image_path)
                    train_image = train_image.convert("YCbCr")
                    train_input_dataset.append(train_image)

                    # target 이미지 로드
                    target_image_path = os.path.join(target_train_path, target_file)
                    target_image = Image.open(target_image_path)
                    target_image = target_image.convert("YCbCr")
                    train_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in training set: {input_file} and {target_file}"
                    )

    # Dataset과 DataLoader 생성
    train_dataset = CustomDataset(train_input_dataset, train_target_dataset, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, train_loader


def load_cifar100_test_dataset_and_dataloader(QF):
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "original_size")

    test_input_dataset = []
    test_target_dataset = []

    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    # 테스트 데이터 로드
    for i in tqdm.tqdm(range(num_classes), desc=f"Loading test data (QF {QF})"):
        test_path = os.path.join(test_input_dir, f"{i:03d}")
        target_test_path = os.path.join(target_test_dataset_dir, f"{i:03d}")

        # test_path 내 파일을 정렬된 순서로 불러오기
        sorted_test_files = sorted(os.listdir(test_path))
        sorted_target_test_files = sorted(os.listdir(target_test_path))

        # 두 디렉토리의 파일명이 같은지 확인하며 로드
        for input_file, target_file in zip(sorted_test_files, sorted_target_test_files):
            if input_file.replace(".jpeg", ".png") == target_file:
                # input 이미지 로드
                test_image_path = os.path.join(test_path, input_file)
                test_image = Image.open(test_image_path)
                test_image = test_image.convert("YCbCr")
                test_input_dataset.append(test_image)

                # target 이미지 로드
                target_image_path = os.path.join(target_test_path, target_file)
                target_image = Image.open(target_image_path)
                target_image = target_image.convert("YCbCr")
                test_target_dataset.append(target_image)

            else:
                print(
                    f"Warning: Mismatched files in testing set: {input_file} and {target_file}"
                )

    # Dataset과 DataLoader 생성
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataset, test_loader


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

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(Decoder, self).__init__()
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

class Improved_PxT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=384,
        num_heads=16,
        num_layers=16,
        mlp_dim=256,
        dropout=0.1,
    ):
        super(Improved_PxT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # 인코더
        self.encoder_layers = nn.ModuleList(
            [Encoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        
        # 병목 층
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        
        # 디코더
        self.decoder_layers = nn.ModuleList(
            [Decoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers//2)]
        )
        
        # 최종 출력 레이어
        self.final_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.patch_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 이미지를 패치로 분할
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, self.patch_dim)

        # 패치 임베딩
        x = self.patch_embed(x)
        x = x + self.position_embeddings

        # 인코더 처리 추가
        x = x.permute(1, 0, 2)  # (n_patches, batch_size, embed_dim)
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 병목 처리 추가
        x = x.permute(1, 0, 2)  # (batch_size, n_patches, embed_dim)
        x = self.bottleneck(x)
        x = x.permute(1, 0, 2)  # (n_patches, batch_size, embed_dim)
        
        # 디코더 처리
        for layer in self.decoder_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # (batch_size, n_patches, embed_dim)

        # 최종 출력
        x = self.final_layer(x)

        # 패치를 다시 이미지로 재구성
        x = x.view(
            batch_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            3,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, 3, self.img_size, self.img_size)
        return x


    

if __name__ == "__main__":
    # Load the dataset
    train_dataset, train_loader = laod_cifar100_train_dataset_and_dataloader()

    # Initialize the model
    model = Improved_PxT()
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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        logging.info(
            f"{model_name} Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}"
        )
        # Save the model
        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                os.path.join("models", f"{model_name}_{epoch+1}.pth"),
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
    test_loss = 0.0
    psnr_values = []
    ssim_values = []
    lpips_loss_alex = lpips.LPIPS(net="alex").to(device)
    lpips_alex_loss_values = []

    for QF in [100, 80, 60, 40, 20]:
        test_dataset, test_loader = load_cifar100_test_dataset_and_dataloader(QF)
        with torch.no_grad():
            rgb_target_patches = []
            rgb_output_patches = []
            patch_idx = 0

            for input_images, target_images in tqdm.tqdm(test_loader, desc="Testing"):
                input_images = input_images.to(device)
                target_images = target_images.to(device)

                # Forward pass
                outputs = model(input_images)

                # Calculate MSE loss
                loss = criterion(outputs, target_images)
                test_loss += loss.item()

                for i in range(len(outputs)):
                    rgb_target = target_images[i].cpu().numpy()
                    rgb_output = outputs[i].cpu().numpy()
                    np.clip(rgb_target, 0, 1, out=rgb_target)
                    np.clip(rgb_output, 0, 1, out=rgb_output)
                    rgb_target = np.transpose(rgb_target, (1, 2, 0)) * 255
                    rgb_output = np.transpose(rgb_output, (1, 2, 0)) * 255
                    rgb_target_patches.append(rgb_target)
                    rgb_output_patches.append(rgb_output)
                    patch_idx += 1

                    # If we've collected 16 patches, combine and compute metrics
                    if len(rgb_target_patches) == 16:
                        def combine_patches(patches):
                            # patches: list of 16 (8,8,3) arrays
                            rows = []
                            for j in range(4):
                                row = np.concatenate(patches[j*4:(j+1)*4], axis=1)
                                rows.append(row)
                            return np.concatenate(rows, axis=0)
                        combined_target = combine_patches(rgb_target_patches)
                        combined_output = combine_patches(rgb_output_patches)

                        # Calculate metrics on the combined 32x32 images
                        psnr = peak_signal_noise_ratio(
                            combined_target.transpose(2, 0, 1),
                            combined_output.transpose(2, 0, 1),
                            data_range=255,
                        )
                        ssim = structural_similarity(
                            combined_target.transpose(2, 0, 1),
                            combined_output.transpose(2, 0, 1),
                            multichannel=True,
                            data_range=255,
                            channel_axis=0,
                        )
                        lpips_alex_loss = lpips_loss_alex(
                            torch.from_numpy(combined_output).permute(2, 0, 1).to(device),
                            torch.from_numpy(combined_target).permute(2, 0, 1).to(device),
                        )
                        lpips_alex_loss_values.append(lpips_alex_loss.item())
                        psnr_values.append(psnr)
                        ssim_values.append(ssim)

                        # Save the combined output image
                        os.makedirs(f"./datasets/{model_name}_output", exist_ok=True)
                        combined_image_path = os.path.join(
                            f"./datasets/{model_name}_output",
                            f"output{image_name_idx:03d}.png",
                        )
                        img = Image.fromarray(combined_output.astype(np.uint8), 'YCbCr')
                        img = img.convert('RGB')
                        img.save(combined_image_path)   
                        image_name_idx += 1

                        # Clear patch lists for next image
                        rgb_target_patches = []
                        rgb_output_patches = []


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
