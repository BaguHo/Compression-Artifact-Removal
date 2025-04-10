import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.init
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score
import os, sys, re
import timm
import logging

if len(sys.argv) < 5:
    print("Usage: python script.py <epochs> <batch_size> <num_workers> <num_classes>")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

learning_rate = 0.001
dataset_names = [
    "ARCNN_cifar100",
    "BlockCNN_cifar100",
    "DnCNN_cifar100",
    "PxT_cifar100",
]
model_list = ["efficientnet_b3", "mobilenetv2_100", "vgg19"]
QFs = [80, 60, 40, 20]

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = int(sys.argv[4])
num_classes = 100


# JPEG 데이터셋 로드
def load_jpeg_datasets(QF, transform):
    jpeg_test_dir = os.path.join(
        os.getcwd(),
        "datasets",
        "CIFAR100",
        "original_size",
        f"jpeg{QF}",
        "test",
    )

    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_dataset, test_dataloader


# JPEG 데이터셋 로드
def load_post_processed_jpeg_datasets(QF, transform, dataset_name):
    jpeg_test_dir = os.path.join(
        os.getcwd(), "datasets", "removed_cifar100", dataset_name, f"JPEG{QF}", "test"
    )

    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_dataset, test_dataloader


def sort_key(filename):
    image_match = re.search(r"image_(\d+)", filename)
    crop_match = re.search(r"crop_(\d+)", filename)

    image_number = int(image_match.group(1)) if image_match else float("inf")
    crop_number = int(crop_match.group(1)) if crop_match else float("inf")

    return (image_number, crop_number)


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

    # cifar100 데이터셋 로드
    cifar100_train_dir = os.path.join(
        os.getcwd(), "datasets", "CIFAR100", "original_size", "original", "train"
    )
    cifar100_test_dir = os.path.join(
        os.getcwd(), "datasets", "CIFAR100", "original_size", "original", "test"
    )
    cifar100_train = datasets.ImageFolder(cifar100_train_dir, transform=transform)
    cifar100_test = datasets.ImageFolder(cifar100_test_dir, transform=transform)
    cifar100_train_loader = DataLoader(
        cifar100_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    cifar100_test_loader = DataLoader(
        cifar100_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # cifar100_train = datasets.CIFAR100(
    #     root=os.path.join(os.getcwd(), "datasets"),
    #     train=True,
    #     download=True,
    #     transform=transform,
    # )
    # cifar100_test = datasets.CIFAR100(
    #     root=os.path.join(os.getcwd(), "datasets"),
    #     train=False,
    #     download=True,
    #     transform=transform,
    # )
    # cifar100_train_loader = DataLoader(
    #     cifar100_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    # )
    # cifar100_test_loader = DataLoader(
    #     cifar100_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    # )

    for current_model in model_list:
        print(f"[Current Model: {current_model}]")
        logging.info(f"[Current Model: {current_model}]")
        print(
            "#############################################################################"
        )

        # cifar100 모델 정의
        model = timm.create_model(
            current_model, pretrained=True, num_classes=num_classes
        ).to(device)

        # cifar100 모델 손실함수 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        # cifar100 모델 학습
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            # tqdm을 배치 루프에 적용 (에포크당 진행률 표시)
            batch_iter = tqdm(
                cifar100_train_loader,
                desc=f"{current_model} Epoch {epoch+1}/{epochs}",
                leave=True,
            )

            for images, labels in batch_iter:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 배치별 실시간 손실 표시
                batch_iter.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

            # 에포크 종료 후 평균 손실 계산
            epoch_loss = running_loss / len(cifar100_train_loader)
            print(f"\nEpoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
            logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

            # 모델 저장 (마지막 에포크에서만)
            if epoch + 1 == epochs:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        os.getcwd(),
                        "models",
                        f"{current_model}_cifar100_epoch_{epoch + 1}.pth",
                    ),
                )

        # cifar100 모델 테스트
        model.eval()
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in tqdm(
                cifar100_test_loader, desc=f"{current_model} testing", leave=False
            ):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_targets.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        accuracy = 100 * correct / total
        precision = precision_score(all_targets, all_predictions, average="macro")
        os.makedirs("metrics", exist_ok=True)
        results_df = pd.DataFrame(
            {
                "model": [current_model],
                "dataset_name": ["CIFAR100"],
                "QF": ["original"],
                "accuracy": [accuracy],
                "precision": [precision],
            }
        )
        results_df.to_csv(
            "metrics/classification_results.csv",
            mode="a+",
            header=not os.path.exists("metrics/classification_results.csv"),
            index=False,
        )
        logging.info(
            f"Model: {current_model}, Dataset: CIFAR100, Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}"
        )
        print(
            f"Model: {current_model}, Dataset: CIFAR100, Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}"
        )

        # JPEG 데이터셋 로드 및 모델 테스트
        for QF in QFs:
            test_dataset, test_loader = load_jpeg_datasets(QF, transform)
            model.eval()
            correct = 0
            total = 0
            all_targets = []
            all_predictions = []
            with torch.no_grad():
                for images, labels in tqdm(
                    test_loader,
                    desc=f"{current_model} jpeg compressed {QF} testing",
                    leave=False,
                ):
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_targets.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
            accuracy = 100 * correct / total
            precision = precision_score(all_targets, all_predictions, average="macro")
            # model, dataset_name, QF, accuracy, precision르ㄹ metrics/classification_results.csv에 저장
            results_df = pd.DataFrame(
                {
                    "model": [current_model],
                    "dataset_name": ["JPEG compressed"],
                    "QF": [QF],
                    "accuracy": [accuracy],
                    "precision": [precision],
                }
            )
            results_df.to_csv(
                "metrics/classification_results.csv",
                mode="a+",
                header=not os.path.exists("metrics/classification_results.csv"),
                index=False,
            )
            logging.info(
                f"Model: {current_model}, Dataset: JPEG compressed QF: {QF}, Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}"
            )
            print(
                f"Model: {current_model}, Dataset: JPEG compressed QF: {QF}, Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}"
            )

            for dataset_name in dataset_names:
                test_dataset, test_loader = load_post_processed_jpeg_datasets(
                    QF, transform, dataset_name
                )
                model.eval()
                correct = 0
                total = 0
                all_targets = []
                all_predictions = []
                with torch.no_grad():
                    for images, labels in tqdm(
                        test_loader,
                        desc=f"{current_model} jpeg {QF}testing",
                        leave=False,
                    ):
                        images, labels = images.to(device), labels.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        all_targets.extend(labels.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())
                accuracy = 100 * correct / total
                precision = precision_score(
                    all_targets, all_predictions, average="macro"
                )
                # model, dataset_name, QF, accuracy, precision르ㄹ metrics/classification_results.csv에 저장
                os.makedirs("metrics", exist_ok=True)
                results_df = pd.DataFrame(
                    {
                        "model": [current_model],
                        "dataset_name": [dataset_name],
                        "QF": [QF],
                        "accuracy": [accuracy],
                        "precision": [precision],
                    }
                )
                results_df.to_csv(
                    "metrics/classification_results.csv",
                    mode="a+",
                    header=not os.path.exists("metrics/classification_results.csv"),
                    index=False,
                )
                logging.info(
                    f"Model: {current_model}, Dataset:{dataset_name} QF: {QF}, Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}"
                )
                print(
                    f"Model: {current_model}, Dataset:{dataset_name} QF: {QF}, Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}"
                )
