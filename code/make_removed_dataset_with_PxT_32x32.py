import os
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import re
import torch
from torch import nn
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from knockknock import slack_sender

slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)


# TODO: test 데이터셋에 train데이터셋이 저자오디고 있음.  --> 수정 필요

QFs = [80, 60, 40, 20]
batch_size = 1
num_classes = 20
model_name = "PxT_y_channel_32x32.pth"


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
            1,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, 1, self.img_size, self.img_size)
        return x


class NaturalSortImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 파일 경로를 자연스러운 순서로 정렬
        self.samples = natsorted(self.samples, key=lambda x: x[0])
        self.imgs = self.samples

        # 0~4 클래스만 필터링
        filtered_samples = []
        for path, label in self.samples:
            if 0 <= label < num_classes:
                filtered_samples.append((path, label))
        self.samples = filtered_samples
        self.imgs = self.samples


def load_images_from_original_size(QF):
    cifar100_path = os.path.join(os.getcwd(), "datasets", "CIFAR100", "original_size")

    train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

    train_dataset = NaturalSortImageFolder(
        root=train_input_dir,
        transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("YCbCr")),
                transforms.Lambda(
                    lambda x: Image.fromarray(np.array(x)[:, :, 0], mode="L")
                ),
                transforms.ToTensor(),
            ]
        ),
    )

    print(f'load train_dataset from "{train_input_dir}"')

    test_dataset = NaturalSortImageFolder(
        root=test_input_dir,
        transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("YCbCr")),
                transforms.Lambda(
                    lambda x: Image.fromarray(np.array(x)[:, :, 0], mode="L")
                ),
                transforms.ToTensor(),
            ]
        ),
    )

    print(f'load test_dataset from "{test_input_dir}"')

    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    # print(f"Test dataset classes: {test_dataset.classes}")

    return train_loader, test_loader


if __name__ == "__main__":
    # device = "cpu"
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load model and process images
    for QF in QFs:
        image_indexs = [0 for i in range(num_classes)]
        train_loader, test_loader = load_images_from_original_size(QF)

        # model = torch.load(f"./models/{model_name}", map_location="cpu")
        model = torch.load(f"./models/{model_name}")
        model = model.to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        train_output_dir = os.path.join(
            os.getcwd(),
            "datasets",
            "removed_image_PxT_y_channel_32x32",
            f"QF_{QF}",
            "train",
        )

        test_output_dir = os.path.join(
            os.getcwd(),
            "datasets",
            "removed_image_PxT_y_channel_32x32",
            f"QF_{QF}",
            "test",
        )

        os.makedirs(test_output_dir, exist_ok=True)
        os.makedirs(train_output_dir, exist_ok=True)

        img_idx = 0

        for images, labels in tqdm(train_loader, desc="Processing Train Data"):
            images, labels = images.to(device), labels.to(device)
            # print(f"train labels: {(labels)}")
            outputs = model(images)

            images = outputs

            for image, label in zip(images, labels):
                image = ToPILImage()(image.cpu())
                class_dir = os.path.join(train_output_dir, f"{label}")
                os.makedirs(class_dir, exist_ok=True)

                image_path = os.path.join(
                    class_dir,
                    f"image_{img_idx}.jpeg",
                )
                img_idx += 1
                image.save(image_path)
                # print(f"Saved {image_path}")

        img_idx = 0
        for images, labels in tqdm(test_loader, desc="Processing Test Data"):
            images, labels = images.to(device), labels.to(device)
            # print(f"test labels: {(labels)}")
            outputs = model(images)
            images = outputs

            for image, label in zip(images, labels):
                image = ToPILImage()(image.cpu())
                class_dir = os.path.join(test_output_dir, f"{label}")
                os.makedirs(class_dir, exist_ok=True)

                image_path = os.path.join(
                    class_dir,
                    f"image_{img_idx}.jpeg",
                )
                img_idx += 1
                image.save(image_path)

    # Combine Y channel with CbCr channels
    for QF in QFs:
        for i in range(num_classes):
            y_images = []
            cbcr_images = []

            for mode in ["train", "test"]:
                y_image_dir = os.path.join(
                    ".",
                    "datasets",
                    "removed_image_PxT_y_channel_32x32",
                    f"QF_{QF}",
                    mode,
                    str(i),
                )

                cbcr_image_dir = os.path.join(
                    ".",
                    "datasets",
                    "CIFAR100",
                    "original_size",
                    f"jpeg{QF}",
                    mode,
                    str(i),
                )

                y_image_names = natsorted(os.listdir(y_image_dir))
                for image_name in y_image_names:
                    img = Image.open(os.path.join(y_image_dir, image_name))
                    img_array = np.array(img)[
                        :, :, np.newaxis
                    ]  # Add channel dimension for Y
                    # print(f"img_array shape: {img_array.shape}")
                    # input()
                    y_images.append(img_array)

                cbcr_image_names = natsorted(os.listdir(cbcr_image_dir))
                for image_name in cbcr_image_names:
                    img = Image.open(os.path.join(cbcr_image_dir, image_name)).convert(
                        "YCbCr"
                    )
                    # print(f"img shape: {np.array(img).shape}")
                    _, cb, cr = img.split()
                    # print(f"np.dstack((cb, cr)) shape: {np.dstack((cb, cr)).shape}")
                    # input()
                    cbcr_images.append(np.dstack((cb, cr)))

                output_path = os.path.join(
                    ".", "datasets", "combined_ycbcr_32x_32", f"QF_{QF}", mode, str(i)
                )
                os.makedirs(output_path, exist_ok=True)

                img_idx = 0
                for idx, (y_img, cbcr) in enumerate(zip(y_images, cbcr_images)):
                    y_array = np.array(y_img)
                    ycbcr = np.dstack((y_array, cbcr))
                    combined_img = Image.fromarray(ycbcr, mode="YCbCr")

                    # # show combined iamge
                    # plt.imshow(combined_img)
                    # plt.show()

                    rgb_img = combined_img.convert("RGB")
                    output_filename = f"combined_{img_idx}.png"
                    img_idx += 1
                    rgb_img.save(os.path.join(output_path, output_filename))
                    print(f"saved {os.path.join(output_path, output_filename)}")
