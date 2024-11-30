# TODO: it should be run in the root directory of the project
import os
import re
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
import timm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import models
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm

PxT_output_path = os.path.join(os.getcwd(), "models", "PxT.pth")
learning_rate = 0.001
epochs = 50
batch_size = 512
dataset_name = "CIFAR100"
model_name = "PxT_y_channel"
num_workers = 128
image_type = "YCbCr"
num_classes = 20
QFs = [80, 60, 40, 20]


# 디렉토리 생성
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# save model
def save_model(model, path, filename):
    makedir(path)

    model_path = os.path.join(path, filename)
    torch.save(model, model_path)
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
            # print(outputs)
            # print(labels)
            # input()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.8f}")

        # 5 에포크마다 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                "checkpoints", f"model_checkpoint_epoch_{epoch + 1}.pth"
            )
            save_model(
                model, "checkpoints", f"{model_name}_checkpoint_epoch_{epoch + 1}.pth"
            )
            print(f"Checkpoint saved at epoch {epoch + 1}")

    # 마지막 모델 저장
    save_model(model, "checkpoints", model_name)


# evaluate model
def test(model, test_loader, criterion, msg):
    removed_images_path = os.path.join(os.getcwd(), "datasets", "removed_images")
    os.makedirs(removed_images_path, exist_ok=True)

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        image_idx = 0
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            #  모델 output 이미지 저장 (8x8 이미지)
            images = outputs
            idx = 0
            for image in images:
                image = ToPILImage()(image)

                os.makedirs(
                    os.path.join(removed_images_path, str(image_idx)), exist_ok=True
                )
                # TODO: remove saving images
                # image.save(
                #     os.path.join(
                #         removed_images_path,
                #         f"{str(image_idx)}",
                #         f"image_{image_idx}_idx_{idx}.jpeg",
                #     )
                # )
                # image.save(os.path.join(removed_images_path, f"{idx}.jpeg"))
                idx += 1
            image_idx += 1

            # print(f"outputs shape: {outputs.shape}")
            # print(f"labels shape: {labels.shape}")
            running_loss += loss.item()

    avg_loss = running_loss / len(test_loader)
    print(f"Average loss of the model on the test images -- {msg}: {avg_loss:.8f}")

    return avg_loss


# # save result
# def save_result(
#     model_name=model_name,
#     train_dataset=None,
#     test_dataset=None,
#     accuracy=None,
#     precision=None,
#     QF=None,
# ):
#     results_df = pd.DataFrame(
#         {
#             "Model Name": [model_name],
#             "Channel": [channels],
#             "Train Dataset": [train_dataset],
#             "Test Dataset": [test_dataset],
#             "Accuracy": [accuracy],
#             "Precision": [precision],
#             "Epoch": [epochs],
#             "Batch Size": [batch_size],
#             "QF": [QF],
#         }
#     )
#     file_path = os.path.join(os.getcwd(), "result.csv")

#     if os.path.isfile(file_path):
#         results_df.to_csv(file_path, mode="a", index=False, header=False)
#     else:
#         results_df.to_csv(file_path, mode="w", index=False)

#     print("Results saved to './result.csv'")


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

    for i in range(num_classes):
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


# Paramater
# Patch Embeding Paramaters: patch_size * patch_size * channels + embediing_dim
# + Positional embeding = 1*64*64
# + Layer norm Paramaters = 2 * embeding_dim = 2 * 64
# MultiheadAttention Paramaters =  3 * 64 (3×(64×64+64))+(64×64+64) = 16640
# MLP block = (64×128+128)+(128×64+64) = 16576
# Encoder Paramaters = 4×33472=133888
# Decoder Paramaters = (64×3)+3=192+3=195
# Total Paramaters = Patch Embedding: 256 + Position Embeddings: 4096 + Encoder Layers:  133888 + Decoder Layer:  195 = 138435


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        in_channels=1,
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


def load_images_from_8x8():
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
        for i in range(num_classes):
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
                    train_image = Image.open(train_image_path).convert("YCbCr")
                    y_channel_train_image = np.array(train_image)[:, :, 0]
                    train_input_dataset.append(np.array(y_channel_train_image))

                    # target 이미지 로드
                    target_image_path = os.path.join(target_train_path, target_file)
                    target_image = Image.open(target_image_path).convert("YCbCr")
                    y_channel_target_image = np.array(target_image)[:, :, 0]
                    train_target_dataset.append(np.array(y_channel_target_image))
                else:
                    print(
                        f"Warning: Mismatched files in training set: {train_file} and {target_file}"
                    )

        # 테스트 데이터 로드
        for i in range(num_classes):
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
                    test_image = Image.open(test_image_path).convert("YCbCr")
                    y_channel_test_image = np.array(test_image)[:, :, 0]
                    test_input_dataset.append(np.array(y_channel_test_image))

                    # target 이미지 로드
                    target_image_path = os.path.join(target_test_path, target_file)
                    target_image = Image.open(target_image_path).convert("YCbCr")
                    y_channel_target_image = np.array(target_image)[:, :, 0]
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


# training & testing for each QF
def training_testing():
    # save_CIFAR100()
    # make_8x8_image_from_original_dataset()

    # for QF in QFs:
    #     # jpeg image 8x8로 저장
    #     print("making the 8x8 image..")
    #     make_8x8_jpeg_image(QF)
    #     print("Done")

    # FIx random seed
    torch.manual_seed(0)

    # load dataset [training, target] = [jpeg, original] as 8x8
    print("Loading dataset and dataloader...")
    train_dataset, test_dataset, train_loader, test_loader = load_images_from_8x8()
    # print(f'train loader: {train_loader}')
    # print(f"test loader: {test_loader}")
    print("Done")

    # print(f'''train shape: {train_dataset.shape}''')
    # print(f'''test shape: {test_dataset.shape}''')

    removal_model = PxT().to(device)
    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        removal_model = nn.DataParallel(removal_model)

    print(f"Total number of parameters: {count_parameters(removal_model)}")

    # removal  model 손실함수 정의
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(removal_model.parameters(), lr=learning_rate)

    # train the removal model
    print(f"[train removal model]")
    train(removal_model, train_loader, criterion, optimizer)

    test_loss = test(removal_model, test_loader, criterion, f"Removal ")
    save_model(
        removal_model,
        os.path.join(os.getcwd(), "output_models"),
        f"PxT.pth",
    )

    print(
        "#############################################################################"
    )

    # print(f"[test removal model]")
    # accuracy, precision = test(removal_model, test_loader, criterion, f"Removal {QF}")
    # save_result(model_name, dataset_name, dataset_name, accuracy, precision, QF)
    # print(
    #     "#############################################################################"
    # )


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
