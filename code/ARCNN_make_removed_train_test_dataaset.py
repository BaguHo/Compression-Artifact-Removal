# TODO: YCbCr 형태로 잘 훈련되는지 확인 필요 <-- 색 값이 이상하게 나옴
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

QFs = [80, 60, 40, 20]
batch_size = 1
num_classes = 5
model_name = "ARCNN.pth"


class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU(),
        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
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


def load_images_from_original(QF):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "original_size")

    train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

    train_dataset = NaturalSortImageFolder(
        root=train_input_dir,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    print(f'load train_dataset from "{train_input_dir}"')

    test_dataset = NaturalSortImageFolder(
        root=test_input_dir,
        transform=transforms.Compose(
            [
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

    for QF in QFs:
        image_indexs = [0 for i in range(num_classes)]
        train_loader, test_loader = load_images_from_original(QF)

        # model = torch.load(f"./models/{model_name}", map_location="cpu")
        model = ARCNN()
        model.load_state_dict(torch.load(f"./post-processing/models/{model_name}"))
        model = model.to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        train_output_dir = os.path.join(
            os.getcwd(),
            "datasets",
            "removed_image_ARCNN",
            f"QF_{QF}",
            "train",
        )

        test_output_dir = os.path.join(
            os.getcwd(),
            "datasets",
            "removed_image_ARCNN",
            f"QF_{QF}",
            "test",
        )

        os.makedirs(test_output_dir, exist_ok=True)
        os.makedirs(train_output_dir, exist_ok=True)

        image_idx = 0

        for images, labels in tqdm(train_loader, desc="Processing Train Data"):
            images, labels = images.to(device), labels.to(device)
            # print(f"train labels: {(labels)}")
            outputs = model(images)

            # show outputs image
            plt.imshow(outputs[0].cpu().detach().numpy().transpose(1, 2, 0))
            plt.show()
            input()

            images = outputs

            for image, label in zip(images, labels):
                image = ToPILImage()(image.cpu())
                class_dir = os.path.join(train_output_dir, f"{label}")
                os.makedirs(class_dir, exist_ok=True)

                image_path = os.path.join(
                    class_dir,
                    f"image_{image_indexs[label]}.jpeg",
                )
                image.save(image_path)
                # print(f"Saved {image_path}")

                image_indexs[label] += 1

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
                    f"image_{image_indexs[label]}.jpeg",
                )
                image.save(image_path)
                # print(f"Saved {image_path}")

                image_indexs[label] += 1
