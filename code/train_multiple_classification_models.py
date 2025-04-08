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
from knockknock import slack_sender
import logging

slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)

if len(sys.argv) < 5:
    print("Usage: python script.py <epochs> <batch_size> <num_workers> <num_classes>")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

learning_rate = 0.001
dataset_name = ["ARCNN_cifar100", "BlockCNN_cifar100", "DnCNN_cifar100"]
model_list = ["efficientnet_b3", "mobilenetv2_100", "vgg19"]
QFs = [80, 60, 40, 20]
image_type = "RGB"


epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = int(sys.argv[4])


# 디렉토리 생성
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# save model
# def save_model(model, filename):
#     path = os.path.join(os.getcwd(), "models")
#     os.makedirs(path, exist_ok=True)
#     model_path = os.path.join(path, filename)
#     # torch.save(model, model_path)
#     print(f"Model saved to {model_path}")


# model training
@slack_sender(webhook_url=slack_webhook_url, channel="Jiho Eum")
def train(current_model_name, model, train_loader, criterion, optimizer):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"{current_model_name} Epoch {epoch+1}/{epochs}"
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # save the original model on 'classification_models' directory as state_dict
        # save_model(
        #     model.state_dict(),
        #     f"{current_model_name}_epoch_{epochs}.pth",
        # )
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")


# evaluate model
@slack_sender(webhook_url=slack_webhook_url, channel="Jiho Eum")
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
    precision_avg = precision_score(all_targets, all_predictions, average="macro")

    print(f"Accuracy of the model on the test images -- {msg}: {accuracy:.2f}%")

    return accuracy, precision_avg


# save result
def save_result(
    model_name=None,
    train_dataset=None,
    test_dataset=None,
    accuracy=None,
    precision=None,
    epochs=None,
    batch_size=None,
    QF=None,
):
    results_df = pd.DataFrame(
        {
            "Model Name": [model_name],
            "Channel": [3],
            "Train Dataset": [train_dataset],
            "Test Dataset": [test_dataset],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Epoch": [epochs],
            "Batch Size": [batch_size],
            "QF": [QF],
        }
    )
    file_path = os.path.join(
        os.getcwd(), "results", f"num_classes_{num_classes}_epochs_{epochs}_result.csv"
    )

    if os.path.isfile(file_path):
        results_df.to_csv(file_path, mode="a", index=False, header=False)
    else:
        results_df.to_csv(file_path, mode="w", index=False)

    # print(f"Results saved to '{file_path}'")


# JPEG 데이터셋 로드
def load_jpeg_datasets(QF, transform):
    jpeg_train_dir = os.path.join(
        os.getcwd(), "datasets", dataset_name, f"QF_{QF}", "train"
    )
    jpeg_test_dir = os.path.join(
        os.getcwd(), "datasets", dataset_name, f"QF_{QF}", "test"
    )

    train_dataset = datasets.ImageFolder(jpeg_train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(jpeg_test_dir, transform=transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader


def sort_key(filename):
    image_match = re.search(r"image_(\d+)", filename)
    crop_match = re.search(r"crop_(\d+)", filename)

    image_number = int(image_match.group(1)) if image_match else float("inf")
    crop_number = int(crop_match.group(1)) if crop_match else float("inf")

    return (image_number, crop_number)


# train and test the models for each QF
@slack_sender(webhook_url=slack_webhook_url, channel="Jiho Eum")
def training_testing():
    QFs = [80, 60, 40, 20]

    # load original dataset
    # original_train_dir = os.path.join(
    #     os.getcwd(), "datasets", "CIFAR100", "original_size", "original", "train"
    # )
    # original_test_dir = os.path.join(
    #     os.getcwd(), "datasets", "CIFAR100", "original_size", "original", "test"
    # )
    # original_train_dataset = datasets.ImageFolder(
    #     original_train_dir, transform=transform
    # )
    # original_test_dataset = datasets.ImageFolder(original_test_dir, transform=transform)

    # # keep only 0~4 folder classes
    # train_indices = [
    #     i for i, (_, c) in enumerate(original_train_dataset.samples) if c < 5
    # ]
    # test_indices = [
    #     i for i, (_, c) in enumerate(original_test_dataset.samples) if c < 5
    # ]

    for current_model_name in model_list:
        print(f"[Current Model: {current_model_name}]")
        # original_model = timm.create_model(
        #     current_model_name, pretrained=True, num_classes=num_classes
        # ).to(device)

        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(original_model.parameters(), lr=learning_rate)

        # # original dataset model tarining
        # print("[train the original model]")
        # train(
        #     current_model_name,
        #     original_model,
        #     original_train_loader,
        #     criterion,
        #     optimizer,
        # )
        print(
            "#############################################################################"
        )

        for QF in QFs:
            accuracies = [0.0] * 4
            pricisions = [0.0] * 4
            # load JPEG  datasets
            _, _, jpeg_train_loader, jpeg_test_loader = load_jpeg_datasets(
                QF, transform
            )

            # # test with original dataset test dataset
            # print("[original - original]")
            # acc, prec = test(
            #     original_model, original_test_loader, "original - original"
            # )
            # accuracies[0] = acc
            # pricisions[0] = prec
            # save_result(current_model_name, dataset_name,  dataset_name, accuracy, precision, epochs, batch_size, QF=QF)

            # #  test with JPEG test dataset
            # print("[original - jpeg]")
            # acc, prec = test(original_model, jpeg_test_loader, f"original - jpeg {QF}")
            # accuracies[1] = acc
            # pricisions[1] = prec
            # save_result(current_model_name, dataset_name, f'JPEG', accuracy, precision, epochs, batch_size, QF=QF)

            # Tarining with JPEG dataset.
            print(f"[Current Model: {current_model_name}]")
            jpeg_model = timm.create_model(
                current_model_name, pretrained=True, num_classes=5
            ).to(device)

            # JPEG model 손실함수 정의
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(jpeg_model.parameters(), lr=learning_rate)

            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                jpeg_model = nn.DataParallel(jpeg_model)

            # train the jpeg model
            print("[train the jpeg model]")
            train(
                current_model_name, jpeg_model, jpeg_train_loader, criterion, optimizer
            )

            # # test with original test dataset
            # print("[jpeg - original]")
            # acc, prec = test(jpeg_model, original_test_loader, f"jpeg {QF} - original")
            # accuracies[2] = acc
            # pricisions[2] = prec
            # # save_result(current_model_name, f'JPEG', dataset_name, accuracy, precision, epochs, batch_size, QF=QF)

            # Test with JPEG test dataset
            print("[jpeg - jpeg]")
            acc, prec = test(jpeg_model, jpeg_test_loader, f"jpeg {QF} - jpeg {QF}")
            accuracies[3] = acc
            pricisions[3] = prec
            # save_result(current_model_name, f'JPEG', f'JPEG', accuracy, precision, epochs, batch_size, QF=QF)
            print(
                "#############################################################################"
            )

            # save_result(
            #     current_model_name,
            #     dataset_name,
            #     dataset_name,
            #     accuracies[0],
            #     pricisions[0],
            #     epochs,
            #     batch_size,
            #     QF,
            # )
            # save_result(
            #     current_model_name,
            #     dataset_name,
            #     "JPEG",
            #     accuracies[1],
            #     pricisions[1],
            #     epochs,
            #     batch_size,
            #     QF,
            # )
            # save_result(
            #     current_model_name,
            #     "JPEG",
            #     dataset_name,
            #     accuracies[2],
            #     pricisions[1],
            #     epochs,
            #     batch_size,
            #     QF,
            # )
            save_result(
                current_model_name,
                "JPEG",
                "JPEG",
                accuracies[3],
                pricisions[3],
                epochs,
                batch_size,
                QF,
            )
            print("saved results")


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

    training_testing()
