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

if len(sys.argv) < 5:
    print("Usage: python script.py <epoch> <batch_size> <num_workers> <num_classes>")
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
num_classes = int(sys.argv[4])


def sort_key(filename):
    image_match = re.search(r"image_(\d+)", filename)
    crop_match = re.search(r"crop_(\d+)", filename)

    image_number = int(image_match.group(1)) if image_match else float("inf")
    crop_number = int(crop_match.group(1)) if crop_match else float("inf")

    return (image_number, crop_number)


class CIFAR100Dataset(Dataset):
    def __init__(self, input_images, target_images, transform=transforms.ToTensor()):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        # input_image = cv2.imread(input_image)
        # target_image = cv2.imread(target_image)
        # ! warning: The following lines are commented out to avoid PIL dependency
        # input_image = Image.fromarray(input_image)
        # target_image = Image.fromarray(target_image)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def load_images():
    QFs = [80, 60, 40, 20]
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8_images")

    train_input_dataset = []
    test_input_dataset = []
    train_target_dataset = []
    test_target_dataset = []

    for QF in QFs:
        # input images
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

        # target images (original)
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")
        target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

        # 학습 데이터 로드
        for i in tqdm.tqdm(
            range(num_classes), desc=f"Loa,ding train data (QF {QF})", total=num_classes
        ):
            train_path = os.path.join(train_input_dir, str(i))
            target_train_path = os.path.join(target_train_dataset_dir, str(i))

            # train_path 내 파일을 정렬된 순서로 불러오기
            sorted_train_files = sorted(os.listdir(train_path), key=sort_key)
            sorted_target_train_files = sorted(
                os.listdir(target_train_path), key=sort_key
            )

            # 두 디렉토리의 파일명이 같은지 확인하며 로드
            for train_file, target_file in zip(
                sorted_train_files, sorted_target_train_files
            ):
                if train_file == target_file:
                    # input 이미지 로드
                    train_image_path = os.path.join(train_path, train_file)
                    train_image = cv2.imread(train_image_path)
                    train_input_dataset.append(train_image)

                    # target 이미지 로드
                    target_image_path = os.path.join(target_train_path, target_file)
                    target_image = cv2.imread(target_image_path)
                    train_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in training set: {train_file} and {target_file}"
                    )

        # 테스트 데이터 로드
        for i in tqdm.tqdm(range(num_classes), desc=f"Loading test data (QF {QF})"):
            test_path = os.path.join(test_input_dir, str(i))
            target_test_path = os.path.join(target_test_dataset_dir, str(i))

            # test_path 내 파일을 정렬된 순서로 불러오기
            sorted_test_files = sorted(os.listdir(test_path), key=sort_key)
            sorted_target_test_files = sorted(
                os.listdir(target_test_path), key=sort_key
            )

            # 두 디렉토리의 파일명이 같은지 확인하며 로드
            for test_file, target_file in zip(
                sorted_test_files, sorted_target_test_files
            ):
                if test_file == target_file:
                    # input 이미지 로드
                    test_image_path = os.path.join(test_path, test_file)
                    test_image = cv2.imread(test_image_path)
                    test_input_dataset.append(test_image)

                    # target 이미지 로드
                    target_image_path = os.path.join(target_test_path, target_file)
                    target_image = cv2.imread(target_image_path)
                    test_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in testing set: {test_file} and {target_file}"
                    )

    # Dataset과 DataLoader 생성
    train_dataset = CIFAR100Dataset(train_input_dataset, train_target_dataset)
    test_dataset = CIFAR100Dataset(test_input_dataset, test_target_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


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


class PxT(nn.Module):
    def __init__(
        self,
        img_size=8,
        patch_size=1,
        in_channels=3,
        embed_dim=64,
        num_heads=16,
        num_layers=8,
        mlp_dim=128,
    ):
        super(PxT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

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
            3,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, 3, self.img_size, self.img_size)
        return x


# test를 돌릴 때 psnr, ssim 를 평균으로 저장하는 함수 (.csv로 저장)
def save_metrics(metrics, filename):
    with open(filename, "w") as f:
        f.write("PSNR,SSIM\n")
        for i in range(len(metrics["PSNR"])):
            f.write(f"{metrics['PSNR'][i]},{metrics['SSIM'][i]}\n")
    print(f"Metrics saved to {filename}")


if __name__ == "__main__":
    # Load the dataset
    train_dataset, test_dataset, train_loader, test_loader = load_images()

    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Initialize the model
    model = PxT()
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
            f"{type(model).__name__} Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}"
        )
        # Save the model
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join("models", f"{type(model).__name__}_{epoch+1}.pth"),
            )
            print(f"{type(model).__name__} Model saved at epoch {epoch+1}")
            logging.info(f"{type(model).__name__} Model saved at epoch {epoch+1}")

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
    lpips_loss_vgg = lpips.LPIPS(net="vgg").to(device)
    lpips_alex_loss_values = []
    lpips_vgg_loss_values = []

    with torch.no_grad():
        combined_target_images = []
        combined_output_images = []
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

                # 8x8 이미지들을 32x32로 합치기
                if patch_idx % 16 == 0 and patch_idx > 0:
                    patch_idx = 0
                    combined_target_image = np.zeros((32, 32, 3), dtype=np.uint8)
                    combined_output_image = np.zeros((32, 32, 3), dtype=np.uint8)

                    for j in range(4):
                        for k in range(4):
                            idx = j * 4 + k
                            combined_target_image[
                                j * 8 : (j + 1) * 8, k * 8 : (k + 1) * 8, :
                            ] = rgb_target_patches[idx]
                    rgb_target_patches.clear()

                    for j in range(4):
                        for k in range(4):
                            idx = j * 4 + k
                            combined_output_image[
                                j * 8 : (j + 1) * 8, k * 8 : (k + 1) * 8, :
                            ] = rgb_output_patches[idx]

                    rgb_output_patches.clear()
                    combined_target_images.append(combined_target_image)
                    combined_output_images.append(combined_output_image)

                    # Calculate PSNR and SSIM
                    psnr = peak_signal_noise_ratio(
                        combined_target_image.transpose(2, 0, 1),
                        combined_output_image.transpose(2, 0, 1),
                        data_range=255,
                    )
                    ssim = structural_similarity(
                        combined_target_image.transpose(2, 0, 1),
                        combined_output_image.transpose(2, 0, 1),
                        multichannel=True,
                        data_range=255,
                        channel_axis=0,
                    )
                    lpips_alex_loss = lpips_loss_alex(
                        torch.from_numpy(combined_output_image)
                        .permute(2, 0, 1)
                        .to(device),
                        torch.from_numpy(combined_target_image)
                        .permute(2, 0, 1)
                        .to(device),
                    )
                    lpips_vgg_loss = lpips_loss_alex(
                        torch.from_numpy(combined_output_image)
                        .permute(2, 0, 1)
                        .to(device),
                        torch.from_numpy(combined_target_image)
                        .permute(2, 0, 1)
                        .to(device),
                    )

                    lpips_alex_loss_values.append(lpips_alex_loss.item())
                    lpips_vgg_loss_values.append(lpips_vgg_loss.item())
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)

                    logging.info(
                        f"{type(model).__name__}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}"
                    )

                    # 합쳐진 이미지 저장
                    os.makedirs(f"{type(model).__name__}_output", exist_ok=True)
                    combined_image_path = os.path.join(
                        f"{type(model).__name__}_output",
                        f"combined_output{image_name_idx}.png",
                    )
                    cv2.imwrite(combined_image_path, combined_output_image)
                    logging.info(
                        f"{type(model).__name__} Combined image saved at {combined_image_path}"
                    )
                    image_name_idx += 1

    # Calculate average metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips_alex = np.mean(lpips_alex_loss_values)
    avg_lpips_vgg = np.mean(lpips_vgg_loss_values)

    print(
        f"{type(model).__name__} Test Loss: {avg_test_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {np.mean(ssim_values):.4f}, LPIPS Alex: {np.mean(lpips_alex_loss_values):.4f}, LPIPS VGG: {np.mean(lpips_vgg_loss_values):.4f}"
    )
    logging.info(
        f"{type(model).__name__} Test Loss: {avg_test_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {np.mean(ssim_values):.4f}, LPIPS Alex: {np.mean(lpips_alex_loss_values):.4f}, LPIPS VGG: {np.mean(lpips_vgg_loss_values):.4f}"
    )
