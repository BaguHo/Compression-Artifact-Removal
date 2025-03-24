from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from torch import nn
import torch
import os, sys, re
import logging
import cv2
import tqdm
import time

if len(sys.argv) < 5:
    print("Usage: python script.py <epoch> <batch_size> <num_workers> <num_classes>")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

dataset_name = "CIFAR100"
slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = int(sys.argv[4])


def sort_key(filename):
    image_match = re.search(r"image_(\d+)", filename)
    crop_match = re.search(r"crop_(\d+)", filename)

    image_number = int(image_match.group(1)) if image_match else float("inf")
    crop_number = int(crop_match.group(1)) if crop_match else float("inf")

    return (image_number, crop_number)


class CIFAR100Dataset(Dataset):
    def __init__(self, input_images, target_images, transform=transforms.ToTensor()):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        # input_image = cv2.imread(input_image)
        # target_image = cv2.imread(target_image)
        # ! warning: The following lines are commented out to avoid PIL dependency
        # input_image = Image.fromarray(input_image)
        # target_image = Image.fromarray(target_image)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def load_images():
    QFs = [80, 60, 40, 20]
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "original_size")

    train_input_dataset = []
    test_input_dataset = []
    train_target_dataset = []
    test_target_dataset = []

    for QF in QFs:
        # input images
        train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
        test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

        # target images (original)
        target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")
        target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

        # 학습 데이터 로드
        for i in range(num_classes):
            train_path = os.path.join(train_input_dir, str(i))
            target_train_path = os.path.join(target_train_dataset_dir, str(i))

            # train_path 내 파일을 정렬된 순서로 불러오기
            sorted_train_files = sorted(os.listdir(train_path), key=sort_key)
            sorted_target_train_files = sorted(
                os.listdir(target_train_path), key=sort_key
            )

            # 두 디렉토리의 파일명이 같은지 확인하며 로드
            for train_file, target_file in tqdm.tqdm(
                zip(sorted_train_files, sorted_target_train_files),
                desc=f"Loading class {i} train data (QF {QF})",
                total=len(sorted_train_files),
            ):
                if train_file.replace("jpeg", "png") == target_file:
                    # input 이미지 로드
                    train_image_path = os.path.join(train_path, train_file)
                    train_image = cv2.imread(train_image_path)
                    train_input_dataset.append(train_image)

                    # target 이미지 로드
                    target_image_path = os.path.join(target_train_path, target_file)
                    target_image = cv2.imread(target_image_path)
                    train_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in training set: {train_file} and {target_file}"
                    )

        # 테스트 데이터 로드
        for i in tqdm.tqdm(range(num_classes), desc=f"Loading test data (QF {QF})"):
            test_path = os.path.join(test_input_dir, str(i))
            target_test_path = os.path.join(target_test_dataset_dir, str(i))

            # test_path 내 파일을 정렬된 순서로 불러오기
            sorted_test_files = sorted(os.listdir(test_path), key=sort_key)
            sorted_target_test_files = sorted(
                os.listdir(target_test_path), key=sort_key
            )

            # 두 디렉토리의 파일명이 같은지 확인하며 로드
            for test_file, target_file in zip(
                sorted_test_files, sorted_target_test_files
            ):
                if test_file.replace("jpeg", "png") == target_file:
                    # input 이미지 로드
                    test_image_path = os.path.join(test_path, test_file)
                    test_image = cv2.imread(test_image_path)
                    test_input_dataset.append(test_image)

                    # target 이미지 로드
                    target_image_path = os.path.join(target_test_path, target_file)
                    target_image = cv2.imread(target_image_path)
                    test_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in testing set: {test_file} and {target_file}"
                    )

    # Dataset과 DataLoader 생성
    train_dataset = CIFAR100Dataset(train_input_dataset, train_target_dataset)
    test_dataset = CIFAR100Dataset(test_input_dataset, test_target_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU(),
        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x


class FastARCNN(nn.Module):
    def __init__(self):
        super(FastARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=2, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.PReLU(),
        )
        self.last = nn.ConvTranspose2d(
            64, 3, kernel_size=9, stride=2, padding=4, output_padding=1
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x


# test를 돌릴 때 psnr, ssim 를 평균으로 저장하는 함수 (.csv로 저장)
def save_metrics(metrics, filename):
    with open(filename, "w") as f:
        f.write("PSNR,SSIM\n")
        for i in range(len(metrics["PSNR"])):
            f.write(f"{metrics['PSNR'][i]},{metrics['SSIM'][i]}\n")
    print(f"Metrics saved to {filename}")


if __name__ == "__main__":
    # Load the dataset
    train_dataset, test_dataset, train_loader, test_loader = load_images()

    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Initialize the model
    model = ARCNN()
    print(model)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    model.to(device)
    print(f"Model device: {device}")

    # train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    print(f"Training started at {time.ctime(start_time)}")
    logging.info(f"Training started at {time.ctime(start_time)}")
    print(f"Training for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (input_images, target_images) in enumerate(
            tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")
        ):
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_images)
            loss = criterion(outputs, target_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        # Save the model
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{type(model).__name__}_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")
            logging.info(f"Model saved at epoch {epoch+1}")

    end_time = time.time()
    print(f"Training finished at {time.ctime(end_time)}")
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    logging.info(f"Training finished at {time.ctime(end_time)}")
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Test the model
    model.eval()
    test_loss = 0.0
    psnr_values = []
    ssim_values = []
    psnr_b_values = []

    with torch.no_grad():
        for input_images, target_images in tqdm.tqdm(test_loader, desc="Testing"):
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            # Forward pass
            outputs = model(input_images)

            # Calculate MSE loss
            loss = criterion(outputs, target_images)
            test_loss += loss.item()

            idx = 0
            for i in range(len(outputs)):

                rgb_target = target_images[i].cpu().numpy()
                rgb_output = outputs[i].cpu().numpy()

                # Calculate PSNR
                psnr = peak_signal_noise_ratio(rgb_target, rgb_output, data_range=1.0)

                # Calculate SSIM
                ssim = structural_similarity(
                    rgb_target,
                    rgb_output,
                    multichannel=True,
                    data_range=1.0,
                    channel_axis=0,
                )

                psnr_values.append(psnr)
                ssim_values.append(ssim)
                print(f"{type(model).__name__}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
                logging.info(
                    f"{type(model).__name__}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}"
                )

                # save the output images
                os.makedirs(f"{type(model).__name__}_output", exist_ok=True)
                output_image_path = os.path.join(
                    f"{type(model).__name__}_output", f"output{idx}.png"
                )
                rgb_output = outputs[i].permute(1, 2, 0).cpu().numpy()

                cv2.imwrite(
                    output_image_path,
                    rgb_output,
                )
                logging.info(
                    f"{type(model).__name__} Output image saved at {output_image_path}"
                )
                idx += 1

    # Calculate average metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = np.mean(psnr_values)

    print(f"Test Loss: {avg_test_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
    logging.info(f"Test Loss: {avg_test_loss:.4f}, PSNR: {avg_psnr:.2f} dB")

    # Save metrics
    metrics = {
        "Test Loss": [avg_test_loss],
        "PSNR": psnr_values,
        "SSIM": ssim_values,
    }
    save_metrics(metrics, f"{type(model).__name__}_metrics.csv")
    # Save the final model
    torch.save(
        model.state_dict(),
        os.path.join("datasets", f"{type(model).__name__}_final.pth"),
    )
    print(f"Final model saved as {type(model).__name__}_final.pth")
    logging.info(f"Final model saved as {type(model).__name__}_final.pth")

    # Send slack notification
    message = f"Model training completed. Elapsed time: {elapsed_time:.2f} seconds"
