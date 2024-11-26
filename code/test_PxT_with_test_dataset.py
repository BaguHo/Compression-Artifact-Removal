import os
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
import torch
from torch import nn
from torchvision.transforms import ToPILImage

QFs = [80, 60, 40, 20]
batch_size = 1
num_classes = 20
model_name = "PxT_50_epoch.pth"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


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


class ViT(nn.Module):
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
        super(ViT, self).__init__()
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


class CIFAR100Dataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        # print(np.array(self.input_images).shape)
        # print(np.array(self.target_images).shape)

        input_image = Image.fromarray(input_image)
        target_image = Image.fromarray(target_image)

        # print(f'input image shape: {input_image.shape}')
        # print(f'target image shape: {target_image.shape}')

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def sort_key(filename):
    image_match = re.search(r"image_(\d+)", filename)
    crop_match = re.search(r"crop_(\d+)", filename)

    image_number = int(image_match.group(1)) if image_match else float("inf")
    crop_number = int(crop_match.group(1)) if crop_match else float("inf")

    return (image_number, crop_number)


def load_images_from_8x8(QF):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8_images")

    test_input_dataset = []
    test_target_dataset = []

    # input images
    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

    # target images (original)
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    # 테스트 데이터 로드
    for i in range(num_classes):
        test_path = os.path.join(test_input_dir, str(i))
        target_test_path = os.path.join(target_test_dataset_dir, str(i))

        # test_path 내 파일을 정렬된 순서로 불러오기
        sorted_test_files = sorted(os.listdir(test_path), key=sort_key)
        sorted_target_test_files = sorted(os.listdir(target_test_path), key=sort_key)

        # 두 디렉토리의 파일명이 같은지 확인하며 로드
        for test_file, target_file in zip(sorted_test_files, sorted_target_test_files):
            if test_file == target_file:
                # input 이미지 로드
                test_image_path = os.path.join(test_path, test_file)
                test_image = Image.open(test_image_path).convert("RGB")
                test_input_dataset.append(np.array(test_image))

                # target 이미지 로드
                target_image_path = os.path.join(target_test_path, target_file)
                target_image = Image.open(target_image_path).convert("RGB")
                test_target_dataset.append(np.array(target_image))
            else:
                print(
                    f"Warning: Mismatched files in testing set: {test_file} and {target_file}"
                )

    # Dataset과 DataLoader 생성
    test_dataset = CIFAR100Dataset(
        test_input_dataset, test_target_dataset, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataset, test_loader


if __name__ == "__main__":
    # Check if GPUs are available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for QF in QFs:
        # Load test dataset and dataloader
        _, test_loader = load_images_from_8x8(QF)

        # Load model and move it to the appropriate device (GPU/CPU)
        model = torch.load("./models/PxT_50_epoch.pth", map_location=device)
        model = model.to(device)  # Move model to GPU if available

        # Enable multi-GPU support with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        model.eval()

        # Output directory for saving results
        output_dir = os.path.join(
            os.getcwd(), "datasets", "removed_images_50_epoch_each_QF", f"QF_{QF}"
        )
        criterion = nn.MSELoss()

        with torch.no_grad():
            image_idx = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                #  모델 output 이미지 저장 (8x8 이미지)
                images = outputs
                idx = 0
                for image in images:
                    image = ToPILImage()(image)

                    os.makedirs(os.path.join(output_dir, str(labels)), exist_ok=True)
                    image.save(
                        os.path.join(
                            output_dir,
                            str(labels),
                            f"image_{image_idx}_idx_{idx}.jpeg",
                        )
                    )
                    idx += 1
                image_idx += 1
