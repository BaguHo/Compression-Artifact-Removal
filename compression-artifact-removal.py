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

# TODO: GPU에 따라 다르게 설정
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('mps')
print(device)

channels = 1
learning_rate = 0.001
epochs = 15
batch_size = 64
QF = 60
dataset_name = "CIFAR10"
model_name = "CNN"
num_workers = 2

# dataloader 생성 함수


# def set_dataloader(train_dataset_path, test_dataset_path):
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     train_dataset = datasets.ImageFolder(train_dataset_path, transform=transform)
#     test_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader


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


# cfg 설정
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# VGG16


class VGG16(nn.Module):
    def __init__(self, vgg_name):
        self.features = self._make_layers(cfg[vgg_name])
        super(VGG16, self).__init__()
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# CNN model


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 모델 학습 함수
def train(model, train_loader, criterion, optimizer):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

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


def save_result(model_name="CNN",  train_dataset=None, test_dataset=None, accuracy=None, precision=None):
    results_df = pd.DataFrame({
        'Model Name': [model_name],
        "Channel": [channels],
        'Training Dataset': [train_dataset],
        'Test Dataset': [test_dataset],
        'Accuracy': [accuracy],
        'Precision': [precision]
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

    # CIFAR10  데이터셋 로드
    dataset_train = datasets.CIFAR10(root="./datasets/", train=True, download=True)
    dataset_test = datasets.CIFAR10(root="./datasets/", train=False, download=True)

    # class name 저장
    class_names = dataset_test.classes

    for i in range(len(class_names)):
        makedir(os.path.join(train_output_dir, "class_" + str(i)))
        makedir(os.path.join(test_output_dir, "class_" + str(i)))

    for idx, (image, label) in enumerate(dataset_train):
        file_name = f"image_{idx}_label_{label}.jpg"
        output_file_path = os.path.join(train_output_dir, "class_" + str(label), file_name)
        image.convert('RGB').save(output_file_path, 'JPEG', quality=QF)

    for idx, (image, label) in enumerate(dataset_test):
        file_name = f"image_{idx}_label_{label}.jpg"
        output_file_path = os.path.join(test_output_dir, "class_" + str(label), file_name)
        image.convert('RGB').save(output_file_path, 'JPEG', quality=QF)

# JPEG 데이터셋 로드


def load_jpeg_datasets(QF, transform):
    jpeg_train_dir = f'./datasets/CIFAR10/jpeg{QF}/train'
    jpeg_test_dir = f'./datasets/CIFAR10/jpeg{QF}/test'

    train_dataset = datasets.ImageFolder(jpeg_train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=True)

    return train_dataset, test_dataset, train_dataloader, test_dataloader


def tensor_to_image(tensor):
    tensor = tensor * 0.5 + 0.5
    # Convert to NumPy array and scale to [0, 255]
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

################################################################################################################
################################################################################################################


if __name__ == "__main__":
    # TODO: 처음에 데이터 다운 받아서 JPEG 이미지 생성
    # JPEG QF dataset 생성
    make_jpeg_datasets(QF)

    # transform 정의
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=channels),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST model 정의
    original_model = CNN().to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(original_model.parameters(), lr=learning_rate)

    # original dataset load
    original_dataset_train = datasets.CIFAR10(root="./datasets/", train=True, transform=transforms.ToTensor(),
                                              target_transform=None, download=True)
    original_dataset_test = datasets.CIFAR10(root="./datasets/", train=False, transform=transforms.ToTensor(),
                                             target_transform=None, download=True)

    original_dataset_train_loader = DataLoader(
        original_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    original_dataset_test_loader = DataLoader(
        original_dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    # load JPEG 90 datasets
    jpeg_train_dataset, jpeg_test_dataset, jpeg_train_loader, jpeg_test_loader = load_jpeg_datasets(
        60, transform)

    # JPEG training dataset 출력해서 확인
    # show_images(jpeg_train_dataset, 10)

    # JPEG 90 testing dataset 출력해서 확인
    # show_images(jpeg_test_dataset, 5)

    # original dataset model tarining
    train(original_model, original_dataset_train_loader, criterion, optimizer)
    # save original model
    save_model(original_model, './models', 'original_model.pth')
    # test with original dataset test dataset
    accuracy, precision = test(original_model, original_dataset_test_loader, 'original - original')
    save_result(model_name, "CIFAR10",  "CIFAR10", accuracy, precision)

    #  test with JPEG test dataset
    accuracy, precision = test(original_model, jpeg_test_loader, 'original - jpeg 60')
    save_result(model_name, "CIFAR10", f'JPEG {QF}', accuracy, precision)

   # Tarining with JPEG dataset
    jpeg_model = CNN().to(device)
    # 손실함수 정의
    optimizer = optim.Adam(jpeg_model.parameters(), lr=learning_rate)
    # train the jpeg model
    train(jpeg_model, jpeg_train_loader, criterion, optimizer)
    # save jpeg model
    save_model(jpeg_model, './models', 'jpeg_model.pth')

    # Test with JPEG test dataset
    accuracy, precision = test(jpeg_model, jpeg_test_loader, 'jpeg 60 - jpeg 60')
    save_result(model_name, f'JPEG {QF}', f'JPEG {QF}', accuracy, precision)

    # test with original  test dataset
    accuracy, precision = test(jpeg_model, original_dataset_test_loader, 'jpeg 60 - original')
    save_result(model_name, f'JPEG {QF}', "CIFAR10", accuracy, precision)
