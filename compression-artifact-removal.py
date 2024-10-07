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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

learning_rate = 0.001
epochs = 15
batch_size = 64

# dataloader 생성 함수


def set_dataloader(train_dataset_path, test_dataset_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(train_dataset_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 디렉토리 생성
def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# CNN model


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# #  VGG16 model
# class VGG16(nn.Module):
#     def __init__(self):
#         super(VGG16, self).__init__()
#         vgg16 = models.vgg16(pretrained=False)

#         vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

#         self.features = vgg16.features
#         self.classifier = vgg16.classifier

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


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
def test(model, test_loader):
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

    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

    return accuracy, precision_avg

# 결과 저장


def save_result(model_name="CNN", train_dataset=None, test_dataset=None, accuracy=None, precision=None):
    results_df = pd.DataFrame({
        'Model Name': [model_name],
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
    train_output_dir = f'./datasets/MNIST/jpeg{QF}/train/'
    test_output_dir = f'./datasets/MNIST/jpeg{QF}/test/'

    makedir(train_output_dir)
    makedir(test_output_dir)

    # MNIST 데이터셋 로드
    mnist_dataset_train = datasets.MNIST(root="./datasets/", train=True, download=True)
    mnist_dataset_test = datasets.MNIST(root="./datasets/", train=False, download=True)

    for i in range(10):
        makedir(os.path.join(train_output_dir, str(i)))
        makedir(os.path.join(test_output_dir, str(i)))

    for idx, (image, label) in enumerate(mnist_dataset_train):
        file_name = f"image_{idx}_label_{label}.jpg"
        output_file_path = os.path.join(train_output_dir, "class", str(label), file_name)
        image.convert('RGB').save(output_file_path, 'JPEG', quality=QF)

    for idx, (image, label) in enumerate(mnist_dataset_test):
        file_name = f"image_{idx}_label_{label}.jpg"
        output_file_path = os.path.join(test_output_dir, "class", str(label), file_name)
        image.convert('RGB').save(output_file_path, 'JPEG', quality=QF)

# JPEG 데이터셋 로드


def load_jpeg_datasets(QF, transform):
    jpeg_train_dir = './datasets/MNIST/jpeg90/train'
    jpeg_test_dir = './datasets/MNIST/jpeg90/test'

    train_dataset = datasets.ImageFolder(jpeg_train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
    # JPEG QF 90 dataset 생성
    make_jpeg_datasets(90)

    # transform 정의
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST model 정의
    MNIST_model = CNN().to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(MNIST_model.parameters(), lr=learning_rate)

    mnist_train = datasets.MNIST(root="./datasets/", train=True, transform=transforms.ToTensor(),
                                 target_transform=None, download=True)
    mnist_test = datasets.MNIST(root="./datasets/", train=False, transform=transforms.ToTensor(),
                                target_transform=None, download=True)

    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    # load JPEG 90 datasets
    jpeg_90_train_dataset, jpeg_90_test_dataset, jpeg_90_train_loader, jpeg_90_test_loader = load_jpeg_datasets(
        90, transform)

    # JPEG 90 training dataset 확인
    # show_images(jpeg_90_train_dataset, 10)

    # JPEG 90 testing dataset 확인
    # show_images(jpeg_90_test_dataset, 5)

    # MNIST model tarining
    train(MNIST_model, mnist_train_loader, criterion, optimizer)

    # test with MNIST test dataset
    accuracy, precision = test(MNIST_model, mnist_test_loader)
    save_result("CNN", "MNIST",  "MNIST", accuracy, precision)

    #  test with JPEG 90 test dataset
    accuracy, precision = test(MNIST_model, jpeg_90_test_loader)
    save_result("CNN", "MNIST", "JPEG 90", accuracy, precision)

    # Tarining with JPEG 90 dataset
    jpeg_90_model = CNN().to(device)

    # Test with JPEG 90 test dataset
    accuracy, precision = test(jpeg_90_model, jpeg_90_test_loader)
    save_result("CNN", "JPEG 90", "JPEG 90", accuracy, precision)

    # test with MNIST test dataset
    accuracy, precision = test(jpeg_90_model, mnist_test_loader)
    save_result("CNN", "JPEG 90", "MNIST", accuracy, precision)
