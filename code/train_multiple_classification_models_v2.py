import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torchvision.models import efficientnet_b3, mobilenet_v2, vgg19
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import sys

if len(sys.argv) < 3:
    print("Usage: python train_multiple_classification_models.py <epochs> <batch_size>")
    sys.exit(1)

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

# 데이터 준비
transform = transforms.Compose(
    [
        transforms.Resize(224),  # 모든 모델에 동일한 입력 크기로 조정
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = CIFAR100(
    root="./datasets", train=True, download=True, transform=transform
)
test_dataset = CIFAR100(
    root="./datasets", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 모델 선택 함수
def get_model(model_name):
    if model_name == "efficientnet_b3":
        model = efficientnet_b3(weights=IMAGENET1K_V1)  # Pretrained weights 사용 가능
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 100
        )  # CIFAR-100에 맞게 수정
    elif model_name == "mobilenetv2_100":
        model = mobilenet_v2(weights=None)  # Pretrained weights 사용 가능
        model.classifier[1] = nn.Linear(
            model.last_channel, 100
        )  # CIFAR-100에 맞게 수정
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
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    # 테스트 정확도 계산
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of {model_name} on CIFAR-100: {100 * correct / total:.2f}%")


# 실행 예시: 원하는 모델 이름을 입력하여 훈련 시작
if __name__ == "__main__":
    model_names = ["efficientnet_b3", "mobilenetv2_100", "vgg19"]
    for model_name in model_names:
        print(f"Training {model_name}...")
        train_model(model_name=model_name)
