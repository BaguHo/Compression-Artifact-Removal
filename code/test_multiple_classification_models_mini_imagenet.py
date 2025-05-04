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

if len(sys.argv) < 3:
    print("Usage: python script.py  <batch_size> <num_workers>")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

learning_rate = 0.001
dataset_names = [
    "ARCNN_mini_imagenet",
    "BlockCNN_mini_imagenet",
    "DnCNN_mini_imagenet",
    # "PxT_mini_imagenet",
]
model_list = ["efficientnet_b3", "mobilenetv2_100", "vgg19"]
QFs = [100, 80, 60, 40, 20]

batch_size = int(sys.argv[1])
num_workers = int(sys.argv[2])
num_classes = 1000

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# JPEG 데이터셋 로드
def load_jpeg_datasets(QF, transform):
    jpeg_test_dir = os.path.join(
        os.getcwd(),
        "datasets",
        "mini-imagenet",
        f"jpeg{QF}",
        "test",
    )

    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_dataset, test_dataloader


# jpeg artifact removed 데이터셋 로드
def load_post_processed_jpeg_datasets(QF, transform, dataset_name):
    jpeg_test_dir = os.path.join(
        os.getcwd(), "datasets",  dataset_name, f"jpeg{QF}", "test"
    )

    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_dataset, test_dataloader


################################################################################################################
################################################################################################################

if __name__ == "__main__":
    # gpu 설정
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(device)

    mini_imagenet_test_dir = os.path.join(
        os.getcwd(), "datasets", "mini-imagenet", "_original", "test"
    )
    mini_imagenet_test = datasets.ImageFolder(mini_imagenet_test_dir, transform=transform)
    mini_imagenet_test_loader = DataLoader(
        mini_imagenet_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    for current_model in model_list:
        print(f"[Current Model: {current_model}]")
        logging.info(f"[Current Model: {current_model}]")
        print(
            "#############################################################################"
        )

        # '[./models/ARCNN_mini_imagenet.pth', './models/BlockCNN_mini_imagenet.pth', './models/DnCNN_mini_imagenet.pth'] load model
        model = torch.load(os.path.join(os.getcwd(), "models", f"{current_model}_mini_imagenet.pth"))

        # mini_imagenet 모델 손실함수 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

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
