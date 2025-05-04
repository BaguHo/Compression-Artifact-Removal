import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b3, mobilenet_v2, vgg19
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import sys, os
import logging
import re
import numpy as np
from torchvision.utils import save_image
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models.efficientnet import EfficientNet_B3_Weights
from torchvision.models.mobilenet import MobileNet_V2_Weights
from torchvision.models.vgg import VGG19_Weights

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

seed = 42
torch.manual_seed(seed)

if len(sys.argv) < 4:
    print(
        "Usage: python train_multiple_classification_models.py <epochs> <batch_size> <num_workers>"
    )
    sys.exit(1)

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = 1000
# 데이터 준비
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def load_original_train_test_dataloader():
    train_dataset = datasets.ImageFolder(
        os.path.join("datasets", "mini-imagenet", "_original", "train"),
        transform=transform,
    )
    test_dataset = datasets.ImageFolder(
        os.path.join("datasets", "mini-imagenet", "_original", "test"),
        transform=transform,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataloader, test_dataloader


# 모델 선택 함수
def get_model(model_name):
    if model_name == "efficientnet_b3":
        model = efficientnet_b3(
            weights=EfficientNet_B3_Weights.DEFAULT
        )  
        # model.head.fc = nn.Linear(model.head.fc.in_features, 100)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenetv2_100":
        model = mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT
        )  
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg19":
        model = vgg19(weights=VGG19_Weights.DEFAULT)  
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(
            "Invalid model name! Choose from ['efficientnet_b3', 'mobilenetv2_100', 'vgg19']"
        )
    return model


# 훈련 함수
def train_model(model_name, epochs=epochs):
    original_train_loader, original_test_loader = load_original_train_test_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(get_model(model_name))

    print(f"Using {torch.cuda.device_count()} GPUs")
    model = get_model(model_name)
    model.to(device)

    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    if model_name == "efficientnet_b3":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif model_name == "mobilenetv2_100":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif model_name == "vgg19":
        optimizer = optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )

    # 훈련 루프
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(original_train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(original_train_loader)}")

    # 테스트 정확도 계산
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(original_test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # predicted = nn.Softmax(dim=1)(outputs).argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # model save
    torch.save(model.state_dict(), f"models/{model_name}_mini_imagenet.pth")
    print(
        f"Accuracy of {model_name} on original mini-imagenet: {100 * correct / total:.2f}%"
    )
    logging.info(
        f"Accuracy of {model_name} on original mini-imagenet: {100 * correct / total:.2f}%"
    )

    # original JPEG 데이터셋, post-processed JPEG 데이터셋에 대한 정확도 계산
    QFs = [100, 80, 60, 40, 20]
    # 저장한 JPEG 데이터셋 불러와서 테스트
    for QF in QFs:
        jpeg_test_dataset = datasets.ImageFolder(
            f"datasets/mini-imagenet/jpeg{QF}/test", transform=transform
        )
        jpeg_test_loader = DataLoader(
            jpeg_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(jpeg_test_loader, desc=f"Testing JPEG QF={QF}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = nn.Softmax(dim=1)(outputs).argmax(1)
                # _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of {model_name} on mini-imagenet JPEG compressed QF={QF}: {100 * correct / total:.2f}%"
        )
        logging.info(
            f"Accuracy of {model_name} on mini-imagenet JPEG compressed QF={QF}: {100 * correct / total:.2f}%"
        )

    # post-processed dataset accuracy
    QFs = [100, 80, 60, 40, 20]
    post_processed_dataset_name = ["ARCNN_mini_imagenet", "BlockCNN_mini_imagenet", "DnCNN_mini_imagenet"]
    for dataset_name in post_processed_dataset_name:
        for QF in QFs:
            post_processed_jpeg_test_dataset = datasets.ImageFolder(
                f"datasets/{dataset_name}/jpeg{QF}/test", transform=transform
            )
            post_processed_jpeg_test_loader = DataLoader(
                post_processed_jpeg_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(post_processed_jpeg_test_loader, desc=f"Testing post-processed JPEG QF={QF}"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    predicted = nn.Softmax(dim=1)(outputs).argmax(1)
                    # _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(
                f"Accuracy of {model_name} on {dataset_name} post-processed JPEG compressed QF={QF}: {100 * correct / total:.2f}%"
            )
            logging.info(
                f"Accuracy of {model_name} on {dataset_name} post-processed JPEG compressed QF={QF}: {100 * correct / total:.2f}%"
            )


# 실행 예시: 원하는 모델 이름을 입력하여 훈련 시작
if __name__ == "__main__":
    model_names = ["efficientnet_b3", "mobilenetv2_100", "vgg19"]
    for model_name in model_names:
        print(f"Training {model_name}...")
        logging.info(f"Training {model_name}...")
        train_model(model_name=model_name)
