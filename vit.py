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

channels = 3
learning_rate = 0.001
epochs = 100
batch_size = 8
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
            # TODO: [8,8,8]이 나옴 --> [8,3,8,8] 이 나와야 함
            print(f'predicted: {predicted.shape}')
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
    train_output_dir = os.path.join(os.getcwd(), 'datasets', dataset_name,'original_size' , f'jpeg{QF}', 'train')
    test_output_dir = os.path.join(os.getcwd(), 'datasets', dataset_name,'original_size' , f'jpeg{QF}', 'test')

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

def make_8x8_jpeg_image(QF):
    for i in range(100):
        train_dir = os.path.join(os.getcwd(), 'datasets', dataset_name,'original_size', f'jpeg{QF}','train', str(i))
        test_dir = os.path.join(os.getcwd(), 'datasets', dataset_name,'original_size',  f'jpeg{QF}','test' , str(i))

        output_train_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, '8x8_images', f'jpeg{QF}','train',  str(i))
        output_test_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, '8x8_images', f'jpeg{QF}', 'test', str(i))

        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(output_test_dir, exist_ok=True)

        process_and_save_images(train_dir, output_train_dir)
        process_and_save_images(test_dir, output_test_dir)

def make_8x8_image_from_original_dataset():
    for i in range(100):
        train_dir = os.path.join(os.getcwd(), 'datasets', dataset_name,'original_size', 'original','train', str(i))
        test_dir = os.path.join(os.getcwd(), 'datasets', dataset_name,'original_size',  'original','test' , str(i))

        output_train_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, '8x8_images', f'original','train',  str(i))
        output_test_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, '8x8_images', f'original', 'test', str(i))

        makedir(output_train_dir)
        makedir(output_test_dir)

        process_and_save_images(train_dir, output_train_dir)
        process_and_save_images(test_dir, output_test_dir)

class Encoder(nn.Module):
    def __init__(self, embed_size=64, num_heads=3, dropout=0.1):
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
    def __init__(self, in_channels=3, num_encoders=6, embed_size=64, img_size=(8,8), patch_size=8, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_tokens = (img_size[0]*img_size[1])//(patch_size**2)

        # Class token 제거
        # 패치 임베딩
        self.patch_embedding = nn.Linear(in_channels*patch_size**2, embed_size)

        # 위치 임베딩
        self.pos_embedding = nn.Parameter(torch.randn((num_tokens+1, embed_size)), requires_grad=True)

        # 인코더 블록 생성
        self.encoders = nn.ModuleList([
            Encoder(embed_size=embed_size, num_heads=num_heads) for _ in range(num_encoders)
        ])

        # MLP 헤드 수정: 임베딩을 다시 이미지로 복원하는 구조
        # 여기서 최종적으로 (in_channels * img_height * img_width) 크기로 변환
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, in_channels * img_size[0] * img_size[1])
        )

    def forward(self, x):
        print(f"Input tensor: {x}")
        print(f"Input shape: {x.shape}")
        batch_size, channel_size = x.shape[:2]

        # 이미지를 패치로 나누기
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(x.size(0), -1, channel_size*self.patch_size*self.patch_size)

        # 패치 임베딩 적용
        x = self.patch_embedding(patches)

        # 위치 임베딩 적용
        pos_embedding = self.pos_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pos_embedding

        # 인코더 블록 통과
        for encoder in self.encoders:
            x = encoder(x)

        # MLP 헤드를 통해 이미지를 복원하는 과정
        x = self.mlp_head(x[:, 0])

        # 이미지를 (batch size, channels=3, height=8, width=8) 형태로 변환
        x = x.view(batch_size, channel_size, *self.img_size)
        
        # 출력 텐서의 shape을 출력 (추가된 부분)
        # print(f"Output shape: {x.shape}")

        return x


## ViT 모델 정의
#class Encoder(nn.Module):
#    def __init__(self, embed_size=64, num_heads=3, dropout=0.1):
#        super().__init__()
#        self.ln1 = nn.LayerNorm(embed_size)
#        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout, batch_first=True)
#        self.ln2 = nn.LayerNorm(embed_size)
#        self.ff = nn.Sequential(
#            nn.Linear(embed_size, 4 * embed_size),
#            nn.GELU(),
#            nn.Dropout(dropout),
#            nn.Linear(4 * embed_size, embed_size),
#            nn.Dropout(dropout)
#        )
#        self.dropout = nn.Dropout(dropout)
#
#        def forward(self, x):
#            x = self.ln1(x)
#            x = x + self.attention(x, x, x)[0]
#            x = x + self.ff(self.ln2(x))
#            return x
#
#class ViT(nn.Module):
#    def __init__(self, in_channels=3, num_encoders=6, embed_size=64, img_size=(8,8), patch_size=8, num_classes=192, num_heads=4):
#        super().__init__()
#        self.img_size = img_size
#        self.patch_size = patch_size
#        num_tokens = (img_size[0]*img_size[1])//(patch_size**2)
#        self.class_token = nn.Parameter(torch.randn((embed_size,)), requires_grad=True)
#        self.patch_embedding = nn.Linear(in_channels*patch_size**2, embed_size)
#        self.pos_embedding = nn.Parameter(torch.randn((num_tokens+1, embed_size)), requires_grad=True)
#        self.encoders = nn.ModuleList([
#            Encoder(embed_size=embed_size, num_heads=num_heads) for _ in range(num_encoders)
#        ])
#        self.mlp_head = nn.Sequential(
#            nn.LayerNorm(embed_size),
#            nn.Linear(embed_size, num_classes)
#        )
#
#    def forward(self, x):
#        batch_size, channel_size = x.shape[:2]
#        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#        patches = patches.contiguous().view(x.size(0), -1, channel_size*self.patch_size*self.patch_size)
#        x = self.patch_embedding(patches)
#        class_token = self.class_token.unsqueeze(0).repeat(batch_size, 1, 1)
#        x = torch.cat([class_token, x], dim=1)
#        x = x + self.pos_embedding.unsqueeze(0)
#        for encoder in self.encoders:
#            x = encoder(x)
#            x = x[:, 0, :].squeeze()
#            x = self.mlp_head(x)
#            return x

#class ViT(nn.Module):
#    def __init__(self, img_size=8, patch_size=2, in_channels=3, embed_dim=64, num_heads=4, num_layers=6):
#        super(ViT, self).__init__()
#
#        self.img_size = img_size
#        self.patch_size = patch_size
#        self.num_patches = (img_size // patch_size) ** 2
#        self.patch_dim = in_channels * patch_size * patch_size
#
#        # Patch Embedding Layer
#        self.patch_embedding = nn.Linear(self.patch_dim, embed_dim)
#
#        # Positional Embedding
#        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
#
#        # Transformer Layers (Encoder)
#        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
#        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#        # Output Layer to map back to image patches
#        self.output_layer = nn.Linear(embed_dim, self.patch_dim)
#
#    def forward(self, x):
#        # Step 1: Divide image into patches
#        batch_size = x.size(0)
#        x = x.view(batch_size, 3, self.img_size, self.img_size)  # B x C x H x W
#
#        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#        patches = patches.contiguous().view(batch_size, -1, self.patch_dim)  # B x num_patches x patch_dim
#
#        # Step 2: Patch Embedding + Positional Encoding
#        patches_embedded = self.patch_embedding(patches) + self.positional_embedding
#
#        # Step 3: Transformer Encoder
#        encoded_patches = self.transformer_encoder(patches_embedded)
#
#        # Step 4: Output Layer to reconstruct image patches
#        output_patches = self.output_layer(encoded_patches)
#
#        # Step 5: Reshape back to image format
#        output_patches = output_patches.view(batch_size, -1, 3, self.patch_size, self.patch_size)
#
#        output_img = output_patches.permute(0, 2, 1, 3).contiguous()   # B x C x H/P x W/P -> B x C x H x W
#
#        return output_img

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


        #print(np.array(self.input_images).shape)
        #print(np.array(self.target_images).shape)

        input_image = Image.fromarray(input_image)
        target_image = Image.fromarray(target_image)

        # print(f'input image shape: {input_image.shape}')
        # print(f'target image shape: {target_image.shape}')

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def load_images_from_8x8(QF):
    dataset_name = 'CIFAR100'
    cifar100_path = os.path.join(os.getcwd(), 'datasets', dataset_name, '8x8_images')

    # input images
    train_dir = os.path.join(cifar100_path, f'jpeg{QF}', 'train')
    test_dir = os.path.join(cifar100_path, f'jpeg{QF}', 'test')

    # target images (original)
    target_train_dataset_dir = os.path.join(cifar100_path, 'original', 'train')
    target_test_dataset_dir = os.path.join(cifar100_path, 'original', 'test')

    train_input_dataset = []
    test_input_dataset = []
    train_target_dataset = []
    test_target_dataset = []

    # Load training images and corresponding original images
    for i in range(100):
        train_path = os.path.join(train_dir, str(i))
        target_train_path = os.path.join(target_train_dataset_dir, str(i))

        for image_file in os.listdir(train_path):
            image_path = os.path.join(train_path, image_file)
            image = cv2.imread(image_path)
            train_input_dataset.append(np.array(image))

        for image_file in os.listdir(target_train_path):
            image_path = os.path.join(target_train_path, image_file)
            image = cv2.imread(image_path)
            train_target_dataset.append(np.array(image))

    # Load test images and corresponding original images
    for i in range(100):
        test_path = os.path.join(test_dir, str(i))
        target_test_path = os.path.join(target_test_dataset_dir, str(i))

        for image_file in os.listdir(test_path):
            image_path = os.path.join(test_path, image_file)
            image = cv2.imread(image_path)
            test_input_dataset.append(np.array(image))

        for image_file in os.listdir(target_test_path):
            image_path = os.path.join(target_test_path, image_file)
            image = cv2.imread(image_path)
            test_target_dataset.append(np.array(image))

    train_dataset = CIFAR100Dataset(train_input_dataset, train_target_dataset, transform=transform)
    test_dataset = CIFAR100Dataset(test_input_dataset, test_target_dataset, transform=transform)
    
    print(train_dataset[0])
    print(train_dataset[0][0])
    print(train_dataset[0][0][0])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#    print(np.array(test_loader))
    #images, labels = next(iter(train_loader))
    #print('train shape: ', images.shape,'train label shape' ,labels.shape)

    #images, labels = next(iter(test_loader))
    #print('train shape: ', images.shape,'train label shape' ,labels.shape)
#    for inputs, targets in train_loader:
#        print(f'Input batch shape: {inputs.shape}')
#        print(f'Target batch shape: {targets.shape}')
#
#    for inputs, targets in test_loader:
#        print(f'Input batch shape: {inputs.shape}')
#        print(f'Target batch shape: {targets.shape}')

    return train_dataset, test_dataset, train_loader, test_loader

# training & testing for each QF
def training_testing():
    QFs = [80, 60, 40, 20]

    make_8x8_image_from_original_dataset()

    for QF in QFs:
        # make jpeg dataset
        print('making the jpeg dataaset...')
        make_jpeg_datasets(QF)
        print('Done')

        # jpeg image 8x8로 저장
        print('making the 8x8 image..')
        make_8x8_jpeg_image(QF)
        print('done')

        # load dataset [training, target] = [jpeg, original] as 8x8
        print('Loading dataset and dataloader...')
        train_dataset, test_dataset, train_loader, test_loader = load_images_from_8x8(QF)
        #print(f'train loader: {train_loader}')
        print(f'test loader: {test_loader}')

        print('Done')

        # print(f'''train shape: {train_dataset.shape}''')
        # print(f'''test shape: {test_dataset.shape}''')

        removal_model = ViT().to(device)

        # removal  model 손실함수 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(removal_model.parameters(), lr=learning_rate)

        # train the removal model
        print(f'[train removal model QF:{QF}]')
        train(removal_model, train_loader, criterion, optimizer)

        print('#############################################################################')
        print(f'[test removal model]')
        accuracy, precision = test(removal_model, test_loader, f'Removal {QF}')
        save_result(model_name, dataset_name, dataset_name, accuracy, precision, QF)
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

    training_testing()
    # jpeg_train_dataset, jpeg_test_dataset, jpeg_train_loader, jpeg_test_loader = load_jpeg_datasets(temp_qf, transform)

    # make_8x8_image(temp_qf)

    # save_CIFAR100()

    # make jpeg 8x8 images for


