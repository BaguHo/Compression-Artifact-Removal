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
from natsort import natsorted

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


class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, selected_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        if selected_classes is not None:
            self.classes = [cls for cls in self.classes if cls in selected_classes]

        self.image_paths = []
        self.labels = []

        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                file_names = natsorted(os.listdir(class_dir))
                for file_name in file_names:
                    file_path = os.path.join(class_dir, file_name)
                    self.image_paths.append(file_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def load_images_from_8x8(QF):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8_images")

    train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    selected_classes = [str(i) for i in range(num_classes)]

    train_dataset = CustomImageFolder(
        root_dir=train_input_dir, transform=transform, selected_classes=selected_classes
    )

    test_dataset = CustomImageFolder(
        root_dir=test_input_dir, transform=transform, selected_classes=selected_classes
    )

    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    print(f"Test dataset classes: {test_dataset.classes}")

    return train_loader, test_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for QF in QFs:
        train_loader, test_loader = load_images_from_8x8(QF)

        model = torch.load("./models/PxT_50_epoch.pth", map_location=device)
        model = model.to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        model.eval()

        output_dir = os.path.join(
            os.getcwd(), "datasets", "removed_images_50_epoch_each_QF", f"QF_{QF}"
        )

        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            batch_idx = 0
            image_idx = 0

            for batch_images, batch_labels in test_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(
                    device
                )

                batch_outputs = model(batch_images)

                for i, (image, label) in enumerate(zip(batch_outputs, batch_labels)):
                    image = ToPILImage()(image.cpu())

                    class_dir = os.path.join(output_dir, str(label.item()))
                    os.makedirs(class_dir, exist_ok=True)

                    image_path = os.path.join(
                        class_dir,
                        f"batch_{batch_idx}_image_{image_idx}_within_batch_{i}.jpeg",
                    )
                    image.save(image_path)
                    print(f"Saved {image_path}")

                    image_idx += 1

                batch_idx += 1
