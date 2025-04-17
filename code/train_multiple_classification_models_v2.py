import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
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
from torchvision.datasets import CIFAR100
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.models.efficientnet import EfficientNet_B3_Weights

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

if len(sys.argv) < 4:
    print(
        "Usage: python train_multiple_classification_models.py <epochs> <batch_size> <num_workers>"
    )
    sys.exit(1)

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])

# 데이터 준비
transform = transforms.Compose([transforms.ToTensor()])


# Define function to save CIFAR100 as PNG
def save_cifar100_as_png(dataset, split):
    output_dir = os.path.join("datasets", "cifar100_png", split)
    os.makedirs(output_dir, exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        # Create class directory if it doesn't exist
        class_dir = os.path.join(output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        # Convert to PIL Image if it's a tensor or numpy array
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Save as PNG using PIL
        if idx < 10:
            img_path = os.path.join(class_dir, f"img_0000{idx}.png")
        elif idx < 100:
            img_path = os.path.join(class_dir, f"img_000{idx}.png")
        elif idx < 1000:
            img_path = os.path.join(class_dir, f"img_00{idx}.png")
        elif idx < 10000:
            img_path = os.path.join(class_dir, f"img_0{idx}.png")
        else:
            img_path = os.path.join(class_dir, f"img_{idx}.png")

        img.save(img_path, "PNG")

        if idx % 1000 == 0:
            print(f"Processed {idx} images for {split} set")


# Original CIFAR100 datasets
cifar100_train = CIFAR100(
    root="./datasets", train=True, download=True, transform=transform
)
cifar100_test = CIFAR100(
    root="./datasets", train=False, download=True, transform=transform
)

# Save as PNG if not already done
png_train_dir = os.path.join("datasets", "cifar100_png", "train")
png_test_dir = os.path.join("datasets", "cifar100_png", "test")

if not os.path.exists(png_train_dir):
    save_cifar100_as_png(cifar100_train, "train")

if not os.path.exists(png_test_dir):
    save_cifar100_as_png(cifar100_test, "test")

# Load PNG datasets
train_dataset = datasets.ImageFolder(png_train_dir, transform=transform)
test_dataset = datasets.ImageFolder(png_test_dir, transform=transform)

print("cifar100 train dataset[0]", cifar100_train[0][0])
print("cifar100_train[0]", cifar100_train[0][1])
print("train_dataset[0]", train_dataset[0][0])
print("train_dataset[0]", train_dataset[0][1])

# plt.imshow(to_pil_image((cifar100_train[0][0])))
# plt.show()
# plt.imshow(to_pil_image((train_dataset[0][0])))
# plt.show()
# input()

# input()

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)


# Function to save CIFAR-100 dataset with different JPEG quality factors
def save_cifar100_with_different_qf(dataset, split, qfs=[100, 80, 60, 40, 20]):
    import PIL.Image as Image

    for qf in qfs:
        output_dir = os.path.join("datasets", "cifar100_jpeg", f"JPEG{qf}", split)
        os.makedirs(output_dir, exist_ok=True)

        for idx, (img, label) in enumerate(dataset):
            # Create class directory if it doesn't exist
            class_dir = os.path.join(output_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)

            # Convert to PIL Image if it's a tensor or numpy array
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            # Save with specific JPEG quality
            img_path = os.path.join(class_dir, f"img_{idx}.jpg")
            img.save(img_path, "JPEG", quality=qf)

            if idx % 1000 == 0:
                print(f"Processed {idx} images with QF={qf} for {split} set")


if not os.path.exists("datasets/cifar100_jpeg/"):
    print("Saving CIFAR-100 jpeg test dataset with different quality factors...")
    save_cifar100_with_different_qf(test_dataset, "test")


# 모델 선택 함수
def get_model(model_name):
    if model_name == "efficientnet_b3":
        model = efficientnet_b3(
            weights=EfficientNet_B3_Weights
        )  # Pretrained weights 사용 가능
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Dropout(0.4), nn.Linear(1024, 100)
        )
    elif model_name == "mobilenetv2_100":
        model = mobilenet_v2(weights=None)  # Pretrained weights 사용 가능
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Dropout(0.4), nn.Linear(1024, 100)
        )
    elif model_name == "vgg19":
        model = vgg19(weights=None)  # Pretrained weights 사용 가능
        model.classifier[6] = nn.Linear(4096, 100)  # CIFAR-100에 맞게 수정
    else:
        raise ValueError(
            "Invalid model name! Choose from ['efficientnet_b3', 'mobilenetv2_100', 'vgg19']"
        )
    return model


# 훈련 함수
def train_model(model_name, epochs=epochs):
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
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    # 테스트 정확도 계산
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # model save
    torch.save(model.state_dict(), f"models/{model_name}_cifar100.pth")
    print(f"Accuracy of {model_name} on CIFAR-100: {100 * correct / total:.2f}%")
    logging.info(f"Accuracy of {model_name} on CIFAR-100: {100 * correct / total:.2f}%")

    # JPEG 데이터셋에 대한 정확도 계산
    QFs = [100, 80, 60, 40, 20]

    # 저장한 JPEG 데이터셋 불러와서 테스트
    for QF in QFs:
        jpeg_test_dataset = datasets.ImageFolder(
            "datasets/cifar100_jpeg/JPEG" + str(QF) + "/test", transform=transform
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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of {model_name} on CIFAR-100 JPEG QF={QF}: {100 * correct / total:.2f}%"
        )
        logging.info(
            f"Accuracy of {model_name} on CIFAR-100 JPEG QF={QF}: {100 * correct / total:.2f}%"
        )


# 실행 예시: 원하는 모델 이름을 입력하여 훈련 시작
if __name__ == "__main__":
    model_names = ["efficientnet_b3", "mobilenetv2_100", "vgg19"]
    for model_name in model_names:
        print(f"Training {model_name}...")
        train_model(model_name=model_name)
