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
from PIL import Image

channels = 3
learning_rate = 0.001
epochs = 100
batch_size = 256
dataset_name = "CIFAR100"
model_name = "ViT"
num_workers = 4
image_type = 'RGB'
num_classes = 5

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
    # precision_per_class = precision_score(all_targets, all_predictions, average=None)
    precision_avg = precision_score(all_targets, all_predictions, average='macro')

    print(f'Accuracy of the model on the test images -- {msg}: {accuracy:.2f}%')

    return accuracy, precision_avg

# save result
def save_result(model_name=model_name,  train_dataset=None, test_dataset=None, accuracy=None, precision=None, QF=None):
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

def save_CIFAR100():
    # CIFAR-100 데이터셋 다운로드 및 변환 설정
    transform = transforms.ToTensor()  # 이미지를 Tensor로 변환

# CIFAR-100 학습 및 테스트 데이터셋 다운로드
    train_dataset = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform)

# CIFAR-100 클래스 이름 가져오기
    class_names = train_dataset.classes

# 이미지를 저장할 루트 디렉토리 설정
    output_dir = os.path.join(os.getcwd(), 'datasets', 'CIFAR100', '8x8_images', 'original')
    os.makedirs(output_dir, exist_ok=True)

# 각 클래스를 위한 디렉토리 생성 (학습 및 테스트용)
    for i in range(len(class_names)):
        train_class_dir = os.path.join(output_dir, 'train', str(i))
        test_class_dir = os.path.join(output_dir, 'test', str(i))
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

# 학습 데이터 저장하기
    print("Saving training images...")
    for idx, (image, label) in enumerate(train_dataset):
        # PIL 이미지로 변환 (ToTensor의 반대 작업)
        image = transforms.ToPILImage()(image)

        image_filename = os.path.join(output_dir, 'train',str(label), f'image_{idx}_laebl{label}.png')

        print(image_filename)
        # 이미지 저장
        image.save(image_filename, 'PNG')

        if idx % 5000 == 0:
            print(f'{idx} training images saved...')

# 테스트 데이터 저장하기
    print("Saving test images...")
    for idx, (image, label) in enumerate(test_dataset):
        # PIL 이미지로 변환 (ToTensor의 반대 작업)
        image = transforms.ToPILImage()(image)

        # 이미지 파일 이름 설정 (예: 'class_name_00001.png')
        image_filename = os.path.join(output_dir, 'test', str(label),  f'image_{idx}_laebl{label}.png')
        print(image_filename)

        # 이미지 저장
        image.save(image_filename, 'PNG')

        if idx % 2000 == 0:
            print(f'{idx} test images saved...')

    print("All images have been saved successfully.")

# jpeg 이미지 생성
def make_jpeg_datasets(QF):
    train_output_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}', 'train')
    test_output_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}', 'test')

    makedir(train_output_dir)
    makedir(test_output_dir)

    # original dataset load
    original_train_dataset = datasets.CIFAR100(root=os.path.join(os.getcwd(), 'datasets'), train=True, download=True)
    original_test_dataset = datasets.CIFAR100(root=os.path.join(os.getcwd(), 'datasets'), train=False, download=True)

    # Save train images as JPEG
    for idx, (image, label) in enumerate(original_train_dataset):
        file_name = f"image_{idx}_label_{label}.jpg"
        output_file_path = os.path.join(train_output_dir, str(label), file_name)
        temp_path = os.path.join(train_output_dir, str(label))
        makedir(temp_path)
        image.save(output_file_path, 'JPEG', quality=QF)

    # Save test images as JPEG
    for idx, (image, label) in enumerate(original_test_dataset):
        file_name = f"image_{idx}_label_{label}.jpg"
        output_file_path = os.path.join(test_output_dir, str(label), file_name)
        temp_path = os.path.join(test_output_dir, str(label))
        makedir(temp_path)
        image.save(output_file_path, 'JPEG', quality=QF)

# JPEG 데이터셋 로드
def load_jpeg_datasets(QF, transform):
    jpeg_train_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}', 'train')
    jpeg_test_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}', 'test')

    train_dataset = datasets.ImageFolder(jpeg_train_dir)
    test_dataset = datasets.ImageFolder(jpeg_test_dir)

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
            cropped_img.save(os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_crop_{idx}.jpeg"))

def make_8x8_image(train_path, test_path, QF):
    output_train_length = 0
    output_test_length = 0

    for i in range(100):
        #train_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}','train', str(i))
        #test_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, f'jpeg{QF}','test' , str(i))
        train_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, 'train', str(i))
        test_dir = os.path.join(os.getcwd(), 'datasets', dataset_name,'test' , str(i))

        output_train_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, '8x8_images', 'original','train', str(i))
        output_test_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, '8x8_images', 'original','test', str(i))

        makedir(output_train_dir)
        makedir(output_test_dir)

        process_and_save_images(train_dir, output_train_dir)
        process_and_save_images(test_dir, output_test_dir)

# ViT 모델 정의
class Encoder(nn.Module):
    def __init__(self, embed_size=192, num_heads=3, dropout=0.1):
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
            def __init__(self, in_channels=3, num_encoders=6, embed_size=192, img_size=(8,8), patch_size=8, num_classes=192, num_heads=4):
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
    QFs = [100, 80, 60, 40, 20]

    # original dataset load
    original_train = datasets.CIFAR100(os.path.join(os.getcwd(), 'datasets'), train=True, download=True)
    original_test = datasets.CIFAR100(os.path.join(os.getcwd(), 'datasets'), train=False, download=True)

    original_train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(root=os.path.join(os.getcwd(), 'datasets'), train=True, transform=transform, download=True, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True))
    original_test_loader = torch.utils.data.DataLoader(datasets.CIFAR100(root=os.path.join(os.getcwd(), 'datasets'), train=False, transform=transform, download=True, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True))

    # original model
    original_model = ViT().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(original_model.parameters(), lr=learning_rate)

    # original dataset model tarining
    print('[train the original model]')
    train(original_model, original_dataset_train_loader, criterion, optimizer)

    # save original model
    print('save original model')
    save_model(original_model, os.path.join(os.getcwd(), 'models'), 'original_model.pth')

    for QF in QFs:
        # JPEG dataset 생성
        print('make jpeg dataset')
        print('#############################################################################')
        make_jpeg_datasets(QF)

        # load JPEG  datasets
        jpeg_train_dataset, jpeg_test_dataset,  jpeg_train_loader, jpeg_test_loader = load_jpeg_datasets(QF, transform)

        # test with original dataset test dataset
        accuracy, precision = test(original_model, original_dataset_test_loader, 'original - original')
        save_result(model_name, dataset_name,  dataset_name, accuracy, precision, QF=QF)

        #  test with JPEG test dataset
        print('test original model with JPEG test dataset')
        accuracy, precision = test(original_model, jpeg_test_loader, f'original - jpeg {QF}')
        save_result(model_name, dataset_name, f'JPEG', accuracy, precision, QF=QF)

        # Tarining with JPEG dataset.
        # jpeg_model = ViT().to(device)
        jpeg_model = timm.create_model('resnet50', pretrained=True, num_classes=5).to(device)

        # jpeg_model = create_model('vit_base_patch16_224', pretrained=False, num_classes=5, img_size=[128, 128])

        # jpeg_model.patch_embed.proj = nn.Conv2d(3, jpeg_model.embed_dim, kernel_size=(16, 16), stride=(8, 8))

        # JPEG model 손실함수 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(jpeg_model.parameters(), lr=learning_rate)

        # train the jpeg model
        print('[train the jpeg model]')
        train(jpeg_model, jpeg_train_loader, criterion, optimizer)

        # save jpeg model
    save_model(jpeg_model, os.path.join(os.getcwd(), 'models'), 'jpeg_model.pth')

    # Test with JPEG test dataset
    print('test jpeg model with JPEG test dataset')
    accuracy, precision = test(jpeg_model, jpeg_test_loader, f'jpeg {QF} - jpeg {QF}')
    save_result(model_name, f'JPEG', f'JPEG', accuracy, precision, QF=QF)

    # test with original  test dataset
    print('test jpeg model with original  test dataset')
    accuracy, precision = test(jpeg_model, original_dataset_test_loader, f'jpeg {QF} - original')
    save_result(model_name, f'JPEG', dataset_name, accuracy, precision, QF=QF)
    print('#############################################################################')


################################################################################################################
################################################################################################################

if __name__ == "__main__":

    # transform 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # gpu 설정
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(device)

    # training_testing()


    # original dataset load
    original_train = datasets.CIFAR100(os.path.join(os.getcwd(), 'datasets'), train=True, download=True)
    original_test = datasets.CIFAR100(os.path.join(os.getcwd(), 'datasets'), train=False, download=True)

    temp_qf = 60

    # make_jpeg_datasets(temp_qf)

    # jpeg_train_dataset, jpeg_test_dataset, jpeg_train_loader, jpeg_test_loader = load_jpeg_datasets(temp_qf, transform)

    # make_8x8_image(temp_qf)

    save_CIFAR100()
