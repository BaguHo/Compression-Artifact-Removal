import torchvision
from torch.autograd import Variable
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch.nn.init
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, confusion_matrix
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
import timm
from PIL import Image
import cv2
import re

channels = 3
learning_rate = 0.001
epochs = 100
batch_size = 8
dataset_name = "CIFAR100"
model_name = "ViT"
num_workers = 4
image_type = "RGB"
num_classes = 5
QFs = [80, 60, 40, 20]


# 디렉토리 생성
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# save model
def save_model(model, path, filename):
    makedir(path)

    model_path = os.path.join(path, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# model training
def train(model, train_loader, criterion, optimizer):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")


# evaluate model
def test(model, test_loader, msg):
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # TODO: [8,8,8]이 나옴 --> [8,3,8,8] 이 나와야 함
            print(f"predicted: {predicted.shape}")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    # precision_per_class = precision_score(all_targets, all_predictions, average=None)
    precision_avg = precision_score(all_targets, all_predictions, average="macro")

    print(f"Accuracy of the model on the test images -- {msg}: {accuracy:.2f}%")

    return accuracy, precision_avg


# save result
def save_result(
    model_name=model_name,
    train_dataset=None,
    test_dataset=None,
    accuracy=None,
    precision=None,
    QF=None,
):
    results_df = pd.DataFrame(
        {
            "Model Name": [model_name],
            "Channel": [channels],
            "Train Dataset": [train_dataset],
            "Test Dataset": [test_dataset],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Epoch": [epochs],
            "Batch Size": [batch_size],
            "QF": [QF],
        }
    )
    file_path = os.path.join(os.getcwd(), "result.csv")

    if os.path.isfile(file_path):
        results_df.to_csv(file_path, mode="a", index=False, header=False)
    else:
        results_df.to_csv(file_path, mode="w", index=False)

    print("Results saved to './result.csv'")


def save_CIFAR100():
    # CIFAR-100 데이터셋 다운로드 및 변환 설정
    transform = transforms.ToTensor()  # 이미지를 Tensor로 변환

    # CIFAR-100 학습 및 테스트 데이터셋 다운로드
    train_dataset = datasets.CIFAR100(
        root="./datasets", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root="./datasets", train=False, download=True, transform=transform
    )

    # CIFAR-100 클래스 이름 가져오기
    class_names = train_dataset.classes

    # 이미지를 저장할 루트 디렉토리 설정
    output_dir = os.path.join(
        os.getcwd(), "datasets", "CIFAR100", "original_size", "original"
    )

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(class_names)):
        train_class_dir = os.path.join(output_dir, "train", str(i))
        test_class_dir = os.path.join(output_dir, "test", str(i))
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

    print("Saving training images...")
    for idx, (image, label) in enumerate(train_dataset):
        image = transforms.ToPILImage()(image)

        image_filename = os.path.join(
            output_dir, "train", str(label), f"image_{idx}_laebl_{label}.png"
        )
        image.save(image_filename, "PNG")

        if idx % 5000 == 0:
            print(f"{idx} training images saved...")

    print("Saving test images...")
    for idx, (image, label) in enumerate(test_dataset):
        image = transforms.ToPILImage()(image)

        image_filename = os.path.join(
            output_dir, "test", str(label), f"image_{idx}_laebl_{label}.png"
        )

        image.save(image_filename, "PNG")

        if idx % 2000 == 0:
            print(f"{idx} test images saved...")

    for QF in QFs:
        jpeg_output_dir = os.path.join(
            os.getcwd(), "datasets", "CIFAR100", "original_size", f"jpeg{QF}"
        )
        os.makedirs(jpeg_output_dir, exist_ok=True)

        for i in range(len(class_names)):
            train_class_dir = os.path.join(jpeg_output_dir, "train", str(i))
            test_class_dir = os.path.join(jpeg_output_dir, "test", str(i))
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

        print(f"Saving jpeg{QF} training images...")
        for idx, (image, label) in enumerate(train_dataset):
            image = transforms.ToPILImage()(image)

            image_filename = os.path.join(
                jpeg_output_dir, "train", str(label), f"image_{idx}_laebl_{label}.png"
            )
            image.save(image_filename, "PNG")

            if idx % 5000 == 0:
                print(f"{idx} jpeg training images saved...")

        print(f"Saving jpeg {QF} test images...")
        for idx, (image, label) in enumerate(test_dataset):
            image = transforms.ToPILImage()(image)

            image_filename = os.path.join(
                jpeg_output_dir, "test", str(label), f"image_{idx}_laebl_{label}.png"
            )

            image.save(image_filename, "PNG")

            if idx % 2000 == 0:
                print(f"{idx} jpeg test images saved...")

    print("All jpeg images have been saved successfully.")


def extract_label(file_name):
    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("_")
    label = parts[-1]
    return label


# jpeg 이미지 생성
def make_jpeg_datasets(QF):
    train_output_dir = os.path.join(
        os.getcwd(), "datasets", dataset_name, "original_size", f"jpeg{QF}", "train"
    )
    test_output_dir = os.path.join(
        os.getcwd(), "datasets", dataset_name, "original_size", f"jpeg{QF}", "test"
    )

    makedir(train_output_dir)
    makedir(test_output_dir)

    train_input_dir = os.path.join(
        os.getcwd(), "datasets", dataset_name, "original_size", "original", "train"
    )
    test_input_dir = os.path.join(
        os.getcwd(), "datasets", dataset_name, "original_size", "original", "test"
    )

    # original dataset load
    original_train_dataset = datasets.ImageFolder(train_input_dir)
    original_test_dataset = datasets.ImageFolder(test_input_dir)

    # original_train_dataset = datasets.CIFAR100(
    #     root=os.path.join(os.getcwd(), "datasets"), train=True, download=True
    # )
    # original_test_dataset = datasets.CIFAR100(
    #     root=os.path.join(os.getcwd(), "datasets"), train=False, download=True
    # )

    for i in range(100):
        files = os.listdir(os.path.join(train_input_dir, str(i)))
        for file in files:
            file_label = extract_label(file)
            temp_path = os.path.join(train_output_dir, str(file_label))
            os.makedirs(temp_path, exist_ok=True)
            file_output_path = os.path.join(train_output_dir, str(file_label), file)
            image = Image.open(os.path.join(train_input_dir, str(i), file))
            image.save(file_output_path, "JPEG", qaulity=QF)

        files = os.listdir(os.path.join(test_input_dir, str(i)))
        for file in files:
            file_label = extract_label(file)
            temp_path = os.path.join(test_output_dir, str(file_label))
            os.makedirs(temp_path, exist_ok=True)
            file_output_path = os.path.join(test_output_dir, str(file_label), file)
            image = Image.open(os.path.join(test_input_dir, str(i), file))
            image.save(file_output_path, "JPEG", qaulity=QF)


# JPEG 데이터셋 로드
def load_jpeg_datasets(QF):
    jpeg_train_dir = os.path.join(
        os.getcwd(), "datasets", dataset_name, f"jpeg{QF}", "train"
    )
    jpeg_test_dir = os.path.join(
        os.getcwd(), "datasets", dataset_name, f"jpeg{QF}", "test"
    )

    train_dataset = datasets.ImageFolder(jpeg_train_dir)
    test_dataset = datasets.ImageFolder(jpeg_test_dir)

    # print(f'train_dataset shape: {train_dataset}')
    # print(f'test_dataset shape: {test_dataset}')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader


def tensor_to_image(tensor):
    tensor = tensor * 0.5 + 0.5
    image_np = tensor.numpy().squeeze() * 255
    image_np = image_np.astype(np.uint8)
    return image_np


def show_images(dataset, dataloader, length=5):
    random_indices = torch.randperm(len(dataloader))[:length]

    # 데이터 로더에서 배치 가져오기
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # 클래스 이름 정의 (예시)
    class_names = dataset.classes

    # 이미지 시각화 함수 정의
    def imshow(img, label):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(class_names[label])
        plt.show()

    # 몇 개의 이미지 시각화
    num_images_to_show = 5
    for i in range(num_images_to_show):
        imshow(images[i], labels[i].item())


def crop_image(image, crop_size=8):
    width, height = image.size
    cropped_images = []

    for top in range(0, height, crop_size):
        for left in range(0, width, crop_size):
            right = min(left + crop_size, width)
            bottom = min(top + crop_size, height)
            cropped_img = image.crop((left, top, right, bottom))
            cropped_images.append(cropped_img)

    return cropped_images


# 이미지 처리 및 저장 함수 정의
def process_and_save_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # input_dir 내의 모든 이미지 파일 처리
    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)

    with Image.open(img_path) as img:
        # 이미지를 8x8로 자름
        cropped_images = crop_image(img)
        # 잘린 이미지를 output_dir에 저장
        for idx, cropped_img in enumerate(cropped_images):
            cropped_img.save(
                os.path.join(
                    output_dir, f"{os.path.splitext(img_file)[0]}_crop_{idx}.jpeg"
                )
            )


def make_8x8_jpeg_image(QF):
    for i in range(100):
        train_dir = os.path.join(
            os.getcwd(),
            "datasets",
            dataset_name,
            "original_size",
            f"jpeg{QF}",
            "train",
            str(i),
        )
        test_dir = os.path.join(
            os.getcwd(),
            "datasets",
            dataset_name,
            "original_size",
            f"jpeg{QF}",
            "test",
            str(i),
        )

        output_train_dir = os.path.join(
            os.getcwd(),
            "datasets",
            dataset_name,
            "8x8_images",
            f"jpeg{QF}",
            "train",
            str(i),
        )
        output_test_dir = os.path.join(
            os.getcwd(),
            "datasets",
            dataset_name,
            "8x8_images",
            f"jpeg{QF}",
            "test",
            str(i),
        )

        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(output_test_dir, exist_ok=True)

        process_and_save_images(train_dir, output_train_dir)
        process_and_save_images(test_dir, output_test_dir)


def make_8x8_image_from_original_dataset():
    temp_path = os.path.join(os.getcwd(), "datasets", dataset_name)

    for i in range(100):
        input_train_dir = os.path.join(
            temp_path,
            "original_size",
            "original",
            "train",
            str(i),
        )

        input_test_dir = os.path.join(
            temp_path,
            "original_size",
            "original",
            "test",
            str(i),
        )

        output_train_dir = os.path.join(
            temp_path,
            "8x8_images",
            f"original",
            "train",
            str(i),
        )
        output_test_dir = os.path.join(
            temp_path,
            "8x8_images",
            f"original",
            "test",
            str(i),
        )

        makedir(output_train_dir)
        makedir(output_test_dir)

        process_and_save_images(input_train_dir, output_train_dir)
        process_and_save_images(input_test_dir, output_test_dir)


class Encoder(nn.Module):
    def __init__(self, embed_size=64, num_heads=3, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.attention = nn.MultiheadAttention(
            embed_size, num_heads, dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 인코더 블록 내부 처리
        x = self.ln1(x)
        x = x + self.attention(x, x, x)[0]
        x = x + self.ff(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_encoders=6,
        embed_size=64,
        img_size=(8, 8),
        patch_size=8,
        num_heads=4,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        # num_tokens: 패치 수 계산
        num_tokens = (img_size[0] * img_size[1]) // (patch_size**2)

        # 패치 임베딩 레이어
        self.patch_embedding = nn.Linear(in_channels * patch_size**2, embed_size)

        # 위치 임베딩
        self.pos_embedding = nn.Parameter(
            torch.randn((num_tokens, embed_size)), requires_grad=True
        )

        # 인코더 블록 생성
        self.encoders = nn.ModuleList(
            [
                Encoder(embed_size=embed_size, num_heads=num_heads)
                for _ in range(num_encoders)
            ]
        )

        # MLP 헤드: 패치를 다시 이미지로 변환
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size), nn.Linear(embed_size, in_channels * patch_size**2)
        )

    def forward(self, x):
        batch_size, channel_size, height, width = x.shape

        # 이미지를 패치로 분할 (unfold 사용)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(
            batch_size, -1, self.in_channels * self.patch_size * self.patch_size
        )

        # 패치 임베딩 적용
        x = self.patch_embedding(patches)

        # 위치 임베딩 적용
        pos_embedding = self.pos_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pos_embedding

        # 인코더 블록 통과
        for encoder in self.encoders:
            x = encoder(x)

        # MLP Head 적용하여 패치 재구성
        x = self.mlp_head(x)

        # x를 [batch_size, num_patches, in_channels, patch_size, patch_size]로 리셰이프
        x = x.view(batch_size, -1, self.in_channels, self.patch_size, self.patch_size)

        # 높이와 너비 방향의 패치 수 계산
        num_patches_height = self.img_size[0] // self.patch_size
        num_patches_width = self.img_size[1] // self.patch_size

        # x를 [batch_size, num_patches_height, num_patches_width, in_channels, patch_size, patch_size]로 리셰이프
        x = x.view(
            batch_size,
            num_patches_height,
            num_patches_width,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )

        # 차원 재배열: [batch_size, in_channels, img_size[0], img_size[1]]로 변환
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()

        # 최종 리셰이프: [batch_size, in_channels, img_size[0], img_size[1]]
        x = x.view(batch_size, self.in_channels, self.img_size[0], self.img_size[1])

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


# 정렬 기준 함수 정의
def sort_key(filename):
    # 'image_2_label_0_crop_7' 형식에서 image와 crop의 숫자를 추출
    image_match = re.search(r"image_(\d+)", filename)
    crop_match = re.search(r"crop_(\d+)", filename)

    # image와 crop 뒤에 있는 숫자를 정수로 반환
    image_number = int(image_match.group(1)) if image_match else float("inf")
    crop_number = int(crop_match.group(1)) if crop_match else float("inf")

    return (image_number, crop_number)


def load_images_from_8x8(QF):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "8x8_images")

    # input images
    train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

    # target images (original)
    target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    train_input_dataset = []
    test_input_dataset = []
    train_target_dataset = []
    test_target_dataset = []

    # 학습 데이터 로드
    for i in range(100):
        train_path = os.path.join(train_input_dir, str(i))
        target_train_path = os.path.join(target_train_dataset_dir, str(i))

        # train_path 내 파일을 정렬된 순서로 불러오기
        sorted_train_files = sorted(os.listdir(train_path), key=sort_key)
        sorted_target_train_files = sorted(os.listdir(target_train_path), key=sort_key)

        # 두 디렉토리의 파일명이 같은지 확인하며 로드
        for train_file, target_file in zip(
            sorted_train_files, sorted_target_train_files
        ):
            if train_file == target_file:
                # input 이미지 로드
                train_image_path = os.path.join(train_path, train_file)
                train_image = cv2.imread(train_image_path)
                train_input_dataset.append(np.array(train_image))

                # target 이미지 로드
                target_image_path = os.path.join(target_train_path, target_file)
                target_image = cv2.imread(target_image_path)
                train_target_dataset.append(np.array(target_image))
            else:
                print(
                    f"Warning: Mismatched files in training set: {train_file} and {target_file}"
                )

    # 테스트 데이터 로드
    for i in range(100):
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
                test_image = cv2.imread(test_image_path)
                test_input_dataset.append(np.array(test_image))

                # target 이미지 로드
                target_image_path = os.path.join(target_test_path, target_file)
                target_image = cv2.imread(target_image_path)
                test_target_dataset.append(np.array(target_image))
            else:
                print(
                    f"Warning: Mismatched files in testing set: {test_file} and {target_file}"
                )

    # Dataset과 DataLoader 생성
    train_dataset = CIFAR100Dataset(
        train_input_dataset, train_target_dataset, transform=transform
    )
    test_dataset = CIFAR100Dataset(
        test_input_dataset, test_target_dataset, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


# training & testing for each QF
def training_testing():
    save_CIFAR100()
    make_8x8_image_from_original_dataset()

    for QF in QFs:
        # make jpeg dataset
        # print("making the jpeg dataaset...")
        # make_jpeg_datasets(QF)
        # print("Done")

        # jpeg image 8x8로 저장
        print("making the 8x8 image..")
        make_8x8_jpeg_image(QF)
        print("done")

        # load dataset [training, target] = [jpeg, original] as 8x8
        print("Loading dataset and dataloader...")
        train_dataset, test_dataset, train_loader, test_loader = load_images_from_8x8(
            QF
        )
        # print(f'train loader: {train_loader}')
        print(f"test loader: {test_loader}")

        print("Done")

        # print(f'''train shape: {train_dataset.shape}''')
        # print(f'''test shape: {test_dataset.shape}''')

        removal_model = ViT().to(device)

        # removal  model 손실함수 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(removal_model.parameters(), lr=learning_rate)

        # train the removal model
        print(f"[train removal model QF:{QF}]")
        train(removal_model, train_loader, criterion, optimizer)

        print(
            "#############################################################################"
        )
        print(f"[test removal model]")
        accuracy, precision = test(removal_model, test_loader, f"Removal {QF}")
        save_result(model_name, dataset_name, dataset_name, accuracy, precision, QF)
        print(
            "#############################################################################"
        )


################################################################################################################
################################################################################################################

if __name__ == "__main__":

    # transform 정의
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # gpu 설정
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(device)

    training_testing()
