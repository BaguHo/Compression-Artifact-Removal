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

if len(sys.argv) < 6:
    print(
        "Usage: python script.py <model_path> <epoch> <batch_size> <num_workers> <num_classes>"
    )
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

dataset_name = "CIFAR100"
slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)

model_path = sys.argv[1]
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
num_workers = int(sys.argv[4])
num_classes = int(sys.argv[5])


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

    # Load the model
    torch.load(
        os.path.join("models", f"{type(model).__name__}_final.pth"),
    )

    # Test the model
    model.eval()
    idx = 0
    test_loss = 0.0
    psnr_values = []
    ssim_values = []
    psnr_b_values = []

    with torch.no_grad():
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

                # Calculate PSNR
                psnr = peak_signal_noise_ratio(rgb_target, rgb_output, data_range=1.0)

                # Calculate SSIM
                ssim = structural_similarity(
                    rgb_target,
                    rgb_output,
                    multichannel=True,
                    data_range=1.0,
                    channel_axis=0,
                )

                psnr_values.append(psnr)
                ssim_values.append(ssim)
                # print(f"{type(model).__name__}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
                logging.info(
                    f"{type(model).__name__}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}"
                )

                # save the output images
                os.makedirs(f"{type(model).__name__}_output", exist_ok=True)
                output_image_path = os.path.join(
                    f"{type(model).__name__}_output", f"output{idx}.png"
                )
                rgb_output = outputs[i].permute(1, 2, 0).cpu().numpy()

                cv2.imwrite(
                    output_image_path,
                    rgb_output,
                )
                logging.info(
                    f"{type(model).__name__} Output image saved at {output_image_path}"
                )
                idx += 1

    # Calculate average metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = np.mean(psnr_values)

    print(
        f"{type(model).__name__} Test Loss: {avg_test_loss:.4f}, PSNR: {avg_psnr:.2f} dB"
    )
    logging.info(
        f"{type(model).__name__} Test Loss: {avg_test_loss:.4f}, PSNR: {avg_psnr:.2f} dB"
    )

    # Save metrics
    metrics = {
        "Test Loss": [avg_test_loss],
        "PSNR": psnr_values,
        "SSIM": ssim_values,
    }
    save_metrics(metrics, f"{type(model).__name__}_metrics.csv")
    # Save the final model
    torch.save(
        model.state_dict(),
        os.path.join("datasets", f"{type(model).__name__}_final.pth"),
    )
    print(f"Final model saved as {type(model).__name__}_final.pth")
    logging.info(f"Final model saved as {type(model).__name__}_final.pth")
