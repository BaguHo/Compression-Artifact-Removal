import torchvision
from torch.autograd import Variable
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.init
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, confusion_matrix
import os
import shutil
import cv2
import random
import numpy as np
import time
from sklearn.model_selection import train_test_split
from timm import create_model

# TODO: GPU에 따라 다르게 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device('mps')
print(device)

channels = 1
learning_rate = 0.001
epochs = 1
batch_size = 64
QF = 60
dataset_name = "Tufts Face Database"
model_name = "ViT"
num_workers = 4
image_type = 'RGB'

# 디렉토리 생성


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 모델 저장


def save_model(model, path, filename):
    makedir(path)

    model_path = os.path.join(path, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# 모델 학습 함수


def train(model, train_loader, criterion, optimizer):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # if len(images.shape) == 3:
            #     images = images.unsqueeze(0)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')


# 모델 평가
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
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    precision_per_class = precision_score(all_targets, all_predictions, average=None)
    precision_avg = precision_score(all_targets, all_predictions, average='macro')

    print(f'Accuracy of the model on the test images -- {msg}: {accuracy:.2f}%')

    return accuracy, precision_avg

# 결과 저장


def save_result(model_name=model_name,  train_dataset=None, test_dataset=None, accuracy=None, precision=None, QF=QF):
    results_df = pd.DataFrame({
        'Model Name': [model_name],
        "Channel": [channels],
        'Train Dataset': [train_dataset],
        'Test Dataset': [test_dataset],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'QF': [QF]
    })
    file_path = './result.csv'

    if os.path.isfile(file_path):
        results_df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        results_df.to_csv(file_path, mode='w', index=False)

    print("Results saved to './result.csv'")


# jpeg 이미지 생성
def make_jpeg_datasets(QF):
    train_output_dir = f'./datasets/{dataset_name}/jpeg{QF}/train/'
    test_output_dir = f'./datasets/{dataset_name}/jpeg{QF}/test/'

    makedir(train_output_dir)
    makedir(test_output_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # original dataset load
    original_dataset_path = './datasets/thermal_cropped_images'
    original_dataset = datasets.ImageFolder(original_dataset_path, transform=transform)
    original_dataset_train, original_dataset_test = train_test_split(original_dataset, test_size=0.2, random_state=42)

    # make JPEG dataset
    for i in range(len(5)):
        makedir(os.path.join(train_output_dir, "class_" + str(i)))
        makedir(os.path.join(test_output_dir, "class_" + str(i)))

    for idx, (image, label) in enumerate(original_dataset_train):
        file_name = f"image_{idx}_label_{label}.jpg"
        output_file_path = os.path.join(train_output_dir, "class_" + str(label), file_name)
        image.convert(image_type).save(output_file_path, 'JPEG', quality=QF)

    for idx, (image, label) in enumerate(original_dataset_test):
        file_name = f"image_{idx}_label_{label}.jpg"
        output_file_path = os.path.join(test_output_dir, "class_" + str(label), file_name)
        image.convert(image_type).save(output_file_path, 'JPEG', quality=QF)

# JPEG 데이터셋 로드
# TODO: dataset에  따라 변경 필요함


def load_jpeg_datasets(QF, transform):
    jpeg_train_dir = f'./datasets/{dataset_name}/jpeg{QF}/train'
    jpeg_test_dir = f'./datasets/{dataset_name}/jpeg{QF}/test'

    train_dataset = datasets.ImageFolder(jpeg_train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=True)

    return train_dataset, test_dataset, train_dataloader, test_dataloader


def tensor_to_image(tensor):
    tensor = tensor * 0.5 + 0.5
    image_np = tensor.numpy().squeeze() * 255
    image_np = image_np.astype(np.uint8)
    return image_np


def show_images(dataset, length=5):
    random_indices = torch.randperm(len(dataset))[:length]

    plt.figure(figsize=(10, 2))

    for i, idx in enumerate(random_indices):
        image_tensor, label = dataset[idx]
        image_np = tensor_to_image(image_tensor)

        plt.subplot(1, length, i + 1)
        plt.imshow(image_np, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ViT 모델 정의


class Encoder(nn.Module):
    def __init__(self, embed_size=768, num_heads=3, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attention(x, x, x)[0]
        x = x + self.ff(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, in_channels=3, num_encoders=6, embed_size=768, img_size=(128, 128), patch_size=16, num_classes=10, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_tokens = (img_size[0]*img_size[1])//(patch_size**2)
        self.class_token = nn.Parameter(torch.randn((embed_size,)), requires_grad=True)
        self.patch_embedding = nn.Linear(in_channels*patch_size**2, embed_size)
        self.pos_embedding = nn.Parameter(torch.randn((num_tokens+1, embed_size)), requires_grad=True)
        self.encoders = nn.ModuleList([
            Encoder(embed_size=embed_size, num_heads=num_heads) for _ in range(num_encoders)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        batch_size, channel_size = x.shape[:2]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(x.size(0), -1, channel_size*self.patch_size*self.patch_size)
        x = self.patch_embedding(patches)
        class_token = self.class_token.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.pos_embedding.unsqueeze(0)
        for encoder in self.encoders:
            x = encoder(x)
        x = x[:, 0, :].squeeze()
        x = self.mlp_head(x)
        return x


# training & testing for each QF
def training_testing():
    QFs = [20, 40, 60, 80, 100]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # original dataset load
    original_dataset_path = './datasets/thermal_cropped_images'
    original_dataset = datasets.ImageFolder(original_dataset_path, transform=transform)
    original_dataset_train, original_dataset_test = train_test_split(original_dataset, test_size=0.2, random_state=42)

    original_dataset_train_loader = DataLoader(
        original_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    original_dataset_test_loader = DataLoader(
        original_dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_sample, _ = next(iter(original_dataset_train_loader))
    print(f'train_shape: {train_sample.shape}')

    test_sample, _ = next(iter(original_dataset_test_loader))
    print(f'test_shape: {test_sample.shape}')

    # original model
    original_model = ViT().to(device)
    # original_model = create_model('vit_base_patch16_224', pretrained=False, num_classes=5, img_size=[128, 128])
    # original_model.patch_embed.proj = nn.Conv2d(3, original_model.embed_dim, kernel_size=(16, 16), stride=(8, 8))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(original_model.parameters(), lr=learning_rate)

    # original dataset model tarining
    train(original_model, original_dataset_train_loader, criterion, optimizer)

    # save original model
    save_model(original_model, './models', 'original_model.pth')

    for QF in QFs:
        # JPEG dataset 생성
        make_jpeg_datasets(QF)

        # load JPEG  datasets
        _, _,  jpeg_train, jpeg_test = load_jpeg_datasets(QF, transform)

        # test with original dataset test dataset
        accuracy, precision = test(original_model, original_dataset_test_loader, 'original - original')
        save_result(model_name, dataset_name,  dataset_name, accuracy, precision, QF)

        #  test with JPEG test dataset
        accuracy, precision = test(original_model, jpeg_test, f'original - jpeg {QF}')
        save_result(model_name, dataset_name, f'JPEG', accuracy, precision)

        # Tarining with JPEG dataset.
        jpeg_model = ViT().to(device)
        # jpeg_model = create_model('vit_base_patch16_224', pretrained=False, num_classes=5, img_size=[128, 128])

        # jpeg_model.patch_embed.proj = nn.Conv2d(3, jpeg_model.embed_dim, kernel_size=(16, 16), stride=(8, 8))

        # 손실함수 정의
        optimizer = optim.Adam(jpeg_model.parameters(), lr=learning_rate)

        # train the jpeg modelYou gonna learn Looking at that. How to maintain a look at the deal Yeah. who also pulled up. into the Now you guys
        train(jpeg_model, jpeg_train, criterion, optimizer)

        # save jpeg model
        save_model(jpeg_model, './models', 'jpeg_model.pth')

        # Test with JPEG test dataset
        accuracy, precision = test(jpeg_model, jpeg_test, f'jpeg {QF} - jpeg {QF}')
        save_result(model_name, f'JPEG', f'JPEG', accuracy, precision)

        # test with original  test dataset
        accuracy, precision = test(jpeg_model, original_dataset_test, f'jpeg {QF} - original')
        save_result(model_name, f'JPEG', dataset_name, accuracy, precision)


################################################################################################################
################################################################################################################

if __name__ == "__main__":
    training_testing()