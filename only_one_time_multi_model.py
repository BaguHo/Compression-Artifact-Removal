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
import numpy as np
import time
from sklearn.model_selection import train_test_split
import timm
import time
import wget

channels = 1
learning_rate = 0.001
epochs = 100
batch_size = 4
dataset_name = "Tufts_Face_Database"
model_list = ['efficientnet_b3', 'resnet18.a1_in1k', 'resnet50', 'resnet101', 'mobilenetv2', 'vgg19', 'resnext50_32x4d', 'resnext101_32x4d', 'inception_v3']
num_workers = 3
image_type = 'RGB'

# 디렉토리 생성
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Download and unzip the dataset
def download_dataset():
    # load the dataset
    file_name = 'thermal_crroped_images'
    url = f'wget –no-check-certificate ‘https://docs.google.com/uc?export=download&id=14VBOjguAx_j0TE-X_i0PchCIMymgrM0B’ -O {file_name}'
    output_path = './datasets'

    if not os.path.exist(os.path.join(output_path, file_name)):
        wget.download(url, dest_path)

        # unzip the dataset
        os.system('unzip ' + file_name + '-d '+ output_path)

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
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')


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
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    precision_per_class = precision_score(all_targets, all_predictions, average=None)
    precision_avg = precision_score(all_targets, all_predictions, average='macro')

    print(f'Accuracy of the model on the test images -- {msg}: {accuracy:.2f}%')

    return accuracy, precision_avg

# save result
def save_result(model_name=None,  train_dataset=None, test_dataset=None, accuracy=None, precision=None, epochs=None, batch_size=None, QF=None):
    results_df = pd.DataFrame({
        'Model Name': [model_name],
        "Channel": [channels],
        'Train Dataset': [train_dataset],
        'Test Dataset': [test_dataset],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Epoch': [epochs],
        'Batch Size': [batch_size],
        'QF': [QF]
    })
    file_path = os.path.join(os.getcwd(), 'result.csv')

    if os.path.isfile(file_path):
        results_df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        results_df.to_csv(file_path, mode='w', index=False)

    print("Results saved to './result.csv'")


# jpeg 이미지 생성
def make_jpeg_datasets(QF):
    train_output_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}', 'train')
    test_output_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}', 'test')

    makedir(train_output_dir)
    makedir(test_output_dir)

    # original dataset load
    original_dataset_path = os.path.join(os.getcwd(), 'datasets', 'thermal_cropped_images')
    original_dataset = datasets.ImageFolder(original_dataset_path, transform=transform)
    original_dataset_train, original_dataset_test = train_test_split(original_dataset, test_size=0.2, random_state=42)

    # make JPEG dataset
    for i in range(5):
        makedir(os.path.join(train_output_dir, str(i + 1)))
        makedir(os.path.join(test_output_dir, str(i + 1)))

    # Define a transform to convert tensors to PIL images
    to_pil = transforms.ToPILImage()

    # Save train images as JPEG
    for idx, (image_tensor, label) in enumerate(original_dataset_train):
        image = to_pil(image_tensor)
        file_name = f"{idx}_label_{label + 1}.jpg"
        output_file_path = os.path.join(train_output_dir, str(label + 1), file_name)
        image.save(output_file_path, 'JPEG', quality=QF)
        # print(f'saved {output_file_path}')

    # Save test images as JPEG
    for idx, (image_tensor, label) in enumerate(original_dataset_test):
        image = to_pil(image_tensor)
        file_name = f"image_{idx}_label_{label + 1}.jpg"
        output_file_path = os.path.join(test_output_dir, str(label + 1), file_name)
        image.save(output_file_path, 'JPEG', quality=QF)
        # print(f'saved {output_file_path}')

    # for idx, (image, label) in enumerate(original_dataset_train):
    #     file_name = f"image_{idx}_label_{label}.jpg"
    #     output_file_path = os.path.join(train_output_dir, str(label), file_name)
    #     image.convert(image_type).save(output_file_path, 'JPEG', quality=QF)

    # for idx, (image, label) in enumerate(original_dataset_test):
    #     file_name = f"image_{idx}_label_{label}.jpg"
    #     output_file_path = os.path.join(test_output_dir, "class_" + str(label), file_name)
    #     image.convert(image_type).save(output_file_path, 'JPEG', quality=QF)

# JPEG 데이터셋 로드
def load_jpeg_datasets(QF, transform):
    jpeg_train_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}', 'train')
    jpeg_test_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}', 'test')

    train_dataset = datasets.ImageFolder(jpeg_train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    # print(f'train_dataset shape: {train_dataset}')
    # print(f'test_dataset shape: {test_dataset}')

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


# train and test the models for each QF
def training_testing():
    QFs = [100, 80, 60, 40, 20]

    # original dataset load
    original_dataset_path = os.path.join(os.getcwd(), 'datasets', 'thermal_cropped_images')
    original_dataset = datasets.ImageFolder(original_dataset_path, transform=transform)
    original_dataset_train, original_dataset_test = train_test_split(original_dataset, test_size=0.2, random_state=42)

    original_dataset_train_loader = DataLoader(
        original_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    original_dataset_test_loader = DataLoader(
        original_dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # JPEG dataset 생성
    print('make jpeg dataset')
    print('#############################################################################')
    for QF in QFs:
        make_jpeg_datasets(QF)

    # check the shape of dataset
    # train_sample, train_labels = next(iter(original_dataset_train_loader))
    # print(f'train_shape: {train_sample.shape}')
    # print(f'train labels:  {train_labels}')

    # test_sample, test_labels = next(iter(original_dataset_test_loader))
    # print(f'test_shape: {test_sample.shape}')
    # print(f'test labels:  {test_labels}')

    for current_model_name in model_list:
        print(f'[Current Model: {current_model_name}]')
        original_model = timm.create_model(current_model_name, pretrained=True, num_classes=5).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(original_model.parameters(), lr=learning_rate)

        # original dataset model tarining
        print('[train the original model]')
        train(original_model, original_dataset_train_loader, criterion, optimizer)
        print('#############################################################################')

        for QF in QFs:
            # load JPEG  datasets
            jpeg_train_dataset, jpeg_test_dataset,  jpeg_train_loader, jpeg_test_loader = load_jpeg_datasets(QF, transform)

            # test with original dataset test dataset
            accuracy, pricision = test(original_model, original_dataset_test_loader, 'original - original')
            save_result(current_model_name, dataset_name,  dataset_name, accuracy, precision, epochs, batch_size, QF=QF)

            #  test with JPEG test dataset
            print('test original model with JPEG test dataset')
            accuracy, pricision = test(original_model, jpeg_test_loader, f'original - jpeg {QF}')
            save_result(current_model_name, dataset_name, f'JPEG', accuracy, precision, epochs, batch_size, QF=QF)

            # Tarining with JPEG dataset.
            print(f'[Current Model: {current_model_name}]')
            jpeg_model = timm.create_model(current_model_name, pretrained=True, num_classes=5).to(device)

            # JPEG model 손실함수 정의
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(jpeg_model.parameters(), lr=learning_rate)

            # train the jpeg model
            print('[train the jpeg model]')
            train(jpeg_model, jpeg_train_loader, criterion, optimizer)

            # test with original test dataset
            print('test jpeg model with original  test dataset')
            accuracy, pricision = test(jpeg_model, original_dataset_test_loader, f'jpeg {QF} - original')
            save_result(current_model_name, f'JPEG', dataset_name, accuracy, precision, epochs, batch_size, QF=QF)

            # Test with JPEG test dataset
            print('test jpeg model with JPEG test dataset')
            accuracy, pricision = test(jpeg_model, jpeg_test_loader, f'jpeg {QF} - jpeg {QF}')
            save_result(current_model_name, f'JPEG', f'JPEG', accuracy, precision, epochs, batch_size, QF=QF)
            print('#############################################################################')

################################################################################################################
################################################################################################################

if __name__ == "__main__":

    download_dataset()
    # transform 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # gpu 설정
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(device)

    training_testing()

