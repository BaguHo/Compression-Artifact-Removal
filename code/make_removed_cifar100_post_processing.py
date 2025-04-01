from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from torch import nn, functional as F
import torch
import os, sys, re
import logging
import cv2
import tqdm
import time

if len(sys.argv) < 4:
    print("Usage: python script.py <batch_size> <num_workers> <num_classes>")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

dataset_name = "CIFAR100"
slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)

model_names = [
    "ARCNN",
    "DnCNN",
    "BlockCNN",
]
batch_size = int(sys.argv[1])
num_workers = int(sys.argv[2])
num_classes = int(sys.argv[3])


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
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "original_size")

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


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


class FastARCNN(nn.Module):
    def __init__(self):
        super(FastARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=2, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.PReLU(),
        )
        self.last = nn.ConvTranspose2d(
            64, 3, kernel_size=9, stride=2, padding=4, output_padding=1
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x


class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = [
            nn.Sequential(
                nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
        ]
        for i in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        return y - residual


class BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = F.relu(out)

        return out


class BlockCNN(nn.Module):
    def __init__(self):
        super(BlockCNN, self).__init__()
        k = 64
        self.conv_1 = nn.Conv2d(3, k, (3, 5), (1, 1), padding=(1, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(k)

        self.layer_1 = BottleNeck(k, k)
        self.layer_2 = BottleNeck(k, k)

        self.conv_2 = nn.Conv2d(k, k * 2, (3, 5), (1, 1), padding=(1, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(k * 2)

        self.layer_3 = BottleNeck(k * 2, k * 2)

        self.conv_3 = nn.Conv2d(
            k * 2, k * 4, (1, 5), (1, 1), padding=(0, 2), bias=False
        )
        self.bn3 = nn.BatchNorm2d(k * 4)

        self.layer_4 = BottleNeck(k * 4, k * 4)
        self.layer_5 = BottleNeck(k * 4, k * 4)

        self.conv_4 = nn.Conv2d(
            k * 4, k * 8, (1, 1), (1, 1), padding=(0, 0), bias=False
        )
        self.bn4 = nn.BatchNorm2d(k * 8)

        self.layer_6 = BottleNeck(k * 8, k * 8)

        self.conv_5 = nn.Conv2d(k * 8, k * 4, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(k * 4)

        self.layer_7 = BottleNeck(k * 4, k * 4)

        self.conv_6 = nn.Conv2d(k * 4, k * 2, 1, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm2d(k * 2)

        self.layer_8 = BottleNeck(k * 2, k * 2)

        self.conv_7 = nn.Conv2d(k * 2, k, 1, 1, 0, bias=False)
        self.bn7 = nn.BatchNorm2d(k)

        self.layer_9 = BottleNeck(k, k)

        # self.conv_8 = nn.Conv2d(k*2, COLOR_CHANNELS, 1, 1, 0, bias=False)

        self.conv_8 = nn.Conv2d(k, 3, 1, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.squeeze(1)
        out = F.relu(self.bn1(self.conv_1(x)))
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = F.relu(self.bn2(self.conv_2(out)))
        out = self.layer_3(out)
        out = F.relu(self.bn3(self.conv_3(out)))
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = F.relu(self.bn4(self.conv_4(out)))
        out = self.layer_6(out)
        out = F.relu(self.bn5(self.conv_5(out)))
        out = self.layer_7(out)
        out = F.relu(self.bn6(self.conv_6(out)))
        out = self.layer_8(out)
        out = F.relu(self.bn7(self.conv_7(out)))
        out = self.layer_9(out)
        out = self.conv_8(out)
        out = self.sig(out)
        # out = out * 255

        # out = torch.sigmoid(self.conv_8(out))

        return out


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
    for model_name in model_names:
        if model_name == "ARCNN":
            model = ARCNN()
        elif model_name == "FastARCNN":
            model = FastARCNN()
        elif model_name == "DnCNN":
            model = DnCNN()
        elif model_name == "BlockCNN":
            model = BlockCNN()
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

        # # train the model
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # # !load model
        model.load_state_dict(torch.load(f"./models/{type(model).__name__}_30.pth"))

        # Test the model
        model.eval()
        test_loss = 0.0
        psnr_values = []
        ssim_values = []
        psnr_b_values = []

        with torch.no_grad():
            rgb_train_images = []
            rgb_test_images = []
            image_idx = 0

            for input_images, target_images in tqdm.tqdm(
                train_loader, desc="Making Train Images"
            ):
                input_images = input_images.to(device)
                target_images = target_images.to(device)

                # Forward pass
                outputs = model(input_images)

                # Calculate MSE loss
                loss = criterion(outputs, target_images)
                test_loss += loss.item()

                for i in range(len(outputs)):
                    rgb_output = outputs[i].cpu().numpy()
                    np.clip(rgb_output, 0, 1, out=rgb_output)
                    rgb_output = np.transpose(rgb_output, (1, 2, 0)) * 255
                    rgb_train_images.append(rgb_output)
                    image_idx += 1
                    os.makedirs(f"{type(model).__name__}_cifar100/train", exist_ok=True)
                    cv2.imwrite(
                        f"{type(model).__name__}_cifar100/train/image_{image_idx}.png",
                        rgb_output,
                    )
                    if image_idx % 500 == 0 and image_idx > 0:
                        image_idx = 0

                for input_images, target_images in tqdm.tqdm(
                    test_loader, desc="Making Train Images"
                ):
                    input_images = input_images.to(device)

                    # Forward pass
                    outputs = model(input_images)

                    for i in range(len(outputs)):
                        rgb_output = outputs[i].cpu().numpy()
                        np.clip(rgb_output, 0, 1, out=rgb_output)
                        rgb_output = np.transpose(rgb_output, (1, 2, 0)) * 255
                        rgb_test_images.append(rgb_output)
                        image_idx += 1
                        os.makedirs(
                            f"{type(model).__name__}_cifar100/test", exist_ok=True
                        )
                        cv2.imwrite(
                            f"{type(model).__name__}_cifar100/test/image_{image_idx}.png",
                            rgb_output,
                        )
                        if image_idx % 500 == 0 and image_idx > 0:
                            image_idx = 0
