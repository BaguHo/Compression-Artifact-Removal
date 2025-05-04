import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b3, mobilenet_v2, vgg19
import torch.nn.init
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score
import os, sys, re
import logging
from torchvision.models.efficientnet import EfficientNet_B3_Weights
from torchvision.models.mobilenet import MobileNet_V2_Weights
from torchvision.models.vgg import VGG19_Weights
from knockknock import slack_sender

seed = 42
torch.manual_seed(seed)

if len(sys.argv) < 4:
    print("Usage: python script.py <epochs> <batch_size> <num_workers> ")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)
slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)
learning_rate = 0.001
dataset_names = [
    "ARCNN_cifar100",
    "BlockCNN_cifar100",
    "DnCNN_cifar100",
    "PxT_v2_cifar100",
]
model_list = ["efficientnet_b3", "mobilenetv2_100", "vgg19"]
QFs = [100, 80, 60, 40, 20]

epochs = int(sys.argv[1])
epoch = epochs
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = 100

# transform 정의
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ]
)

@slack_sender(slack_webhook_url, channel="Jiho Eum")
def send_slack_message(message):
    pass    

# jpeg 데이터셋 로드
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


# jpeg 데이터셋 로드
def load_post_processed_jpeg_datasets(QF, transform, dataset_name):
    jpeg_test_dir = os.path.join(
        os.getcwd(), "datasets", dataset_name, f"jpeg{QF}", "test"
    )

    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_dataset, test_dataloader

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

    for current_model in model_list:
        print(f"[Current Model: {current_model}]")
        logging.info(f"[Current Model: {current_model}]")
        print(
            "#############################################################################"
        )

        # cifar100 모델 정의
        model = get_model(current_model)
        model.to(device)

        # cifar100 모델 손실함수 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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


        # # TODO: load model
        # model.load_state_dict(
        #     torch.load(
        #         os.path.join(
        #             os.getcwd(),
        #             "models",
        #             f"{current_model}_cifar100_epoch_5.pth",
        #         ),
        #         map_location=device,
        #     )
        # )
    
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

        # jpeg 데이터셋 로드 및 모델 테스트
        for QF in QFs:
            jpeg_test_dataset, jpeg_test_loader = load_jpeg_datasets(QF, transform)
            model.eval()
            jpeg_correct = 0
            jpeg_total = 0
            jpeg_all_targets = []
            jpeg_all_predictions = []
            with torch.no_grad():
                for images, labels in tqdm(
                    jpeg_test_loader,
                    desc=f"{current_model} QF {QF} testing",
                    leave=False,
                ):
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    jpeg_total += labels.size(0)
                    jpeg_correct += (predicted == labels).sum().item()

                    jpeg_all_targets.extend(labels.cpu().numpy())
                    jpeg_all_predictions.extend(predicted.cpu().numpy())
            jpeg_accuracy = 100 * jpeg_correct / jpeg_total
            jpeg_precision = precision_score(jpeg_all_targets, jpeg_all_predictions, average="macro")
            # model, dataset_name, QF, accuracy, precision를 metrics/classification_results.csv에 저장
            results_df = pd.DataFrame(
                {
                    "model": [current_model],
                    "dataset_name": ["jpeg compressed"],
                    "QF": [QF],
                    "accuracy": [jpeg_accuracy],
                    "precision": [jpeg_precision],
                }
            )
            results_df.to_csv(
                "metrics/classification_results.csv",
                mode="a+",
                header=not os.path.exists("metrics/classification_results.csv"),
                index=False,
            )
            logging.info(
                f"Model: {current_model}, Dataset: jpeg compressed QF: {QF}, Epoch: {epoch + 1}, Accuracy: {jpeg_accuracy:.2f}%, Precision: {jpeg_precision:.2f}"
            )
            print(
                f"Model: {current_model}, Dataset: jpeg compressed QF: {QF}, Epoch: {epoch + 1}, Accuracy: {jpeg_accuracy:.2f}%, Precision: {jpeg_precision:.2f}"
            )
            
            for dataset_name in dataset_names:
                post_processed_test_dataset, post_processed_test_loader = load_post_processed_jpeg_datasets(
                    QF, transform, dataset_name
                )
                model.eval()
                post_processed_correct = 0
                post_processed_total = 0
                post_processed_all_targets = []
                post_processed_all_predictions = []
                with torch.no_grad():
                    for images, labels in tqdm(
                        post_processed_test_loader,
                        desc=f"{current_model} {dataset_name} jpeg {QF}testing",
                        leave=False,
                    ):
                        images, labels = images.to(device), labels.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        post_processed_total += labels.size(0)
                        post_processed_correct += (predicted == labels).sum().item()

                        post_processed_all_targets.extend(labels.cpu().numpy())
                        post_processed_all_predictions.extend(predicted.cpu().numpy())
                post_processed_accuracy = 100 * post_processed_correct / post_processed_total
                post_processed_precision = precision_score(
                    post_processed_all_targets, post_processed_all_predictions, average="macro"
                )
                # model, dataset_name, QF, accuracy, precision를 metrics/classification_results.csv에 저장
                # os.makedirs("metrics", exist_ok=True)
                # results_df = pd.DataFrame(
                #     {
                #         "model": [current_model],
                #         "dataset_name": [dataset_name],
                #         "QF": [QF],
                #         "accuracy": [accuracy],
                #         "precision": [precision],
                #     }
                # )
                # results_df.to_csv(
                #     "metrics/classification_results.csv",
                #     mode="a+",
                #     header=not os.path.exists("metrics/classification_results.csv"),
                #     index=False,
                # )
                logging.info(
                    f"Model: {current_model}, Dataset:{dataset_name} QF: {QF}, Epoch: {epoch + 1}, Accuracy: {post_processed_accuracy:.2f}%, Precision: {post_processed_precision:.2f}"
                )
                print(
                    f"Model: {current_model}, Dataset:{dataset_name} QF: {QF}, Epoch: {epoch + 1}, Accuracy: {post_processed_accuracy:.2f}%, Precision: {post_processed_precision:.2f}"
                )
                
    send_slack_message("Classification training completed.")
