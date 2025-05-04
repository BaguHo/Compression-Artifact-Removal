from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import nn
import torch
import torchvision.transforms as transforms
import numpy as np
import os, sys
import logging
import cv2
import tqdm
import time
import lpips

# !warning: BlockCNN does not work

if len(sys.argv) < 4:
    print("Usage: python script.py <epoch> <batch_size> <num_workers>")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

dataset_name = "mini-imagenet"
slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
num_classes = 1000

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

class CustomDataset(Dataset):
    def __init__(self, input_images, target_images, transform=transforms.ToTensor()):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

def laod_mini_imagenet_train_dataset_and_dataloader():
    mini_imagenet_path = os.path.join(os.getcwd(), "datasets", "mini-imagenet")
    

    train_input_dataset = []
    train_target_dataset = []

    QFs = [100, 80, 60, 40, 20]
    for QF in QFs:
        train_input_dir = os.path.join(mini_imagenet_path, f"jpeg{QF}", "train")
        target_train_dataset_dir = os.path.join(
            mini_imagenet_path, "_original", "train"
        )

        # 학습 데이터 로드
        for i in tqdm.tqdm(
            range(num_classes), desc=f"Loading train data (QF {QF})", total=num_classes
        ):
            train_path = os.path.join(train_input_dir, str(i))
            target_train_path = os.path.join(target_train_dataset_dir, str(i))

            # train_path 내 파일을 정렬된 순서로 불러오기
            sorted_train_files = sorted(os.listdir(train_path))
            sorted_target_train_files = sorted(
                os.listdir(target_train_path)
            )

            # 두 디렉토리의 파일명이 같은지 확인하며 로드
            for input_file, target_file in zip(
                sorted_train_files, sorted_target_train_files
            ):
                if input_file.replace("jpeg", "png") == target_file:
                    # input 이미지 로드
                    train_image_path = os.path.join(train_path, input_file)
                    train_image = cv2.imread(train_image_path)
                    train_input_dataset.append(train_image)

                    # target 이미지 로드
                    target_image_path = os.path.join(target_train_path, target_file)
                    target_image = cv2.imread(target_image_path)
                    train_target_dataset.append(target_image)
                else:
                    print(
                        f"Warning: Mismatched files in training set: {input_file} and {target_file}"
                    )

    # Dataset과 DataLoader 생성
    train_dataset = CustomDataset(train_input_dataset, train_target_dataset, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, train_loader


def load_mini_imagenet_test_dataset_and_dataloader(QF):
    mini_imagenet_path = os.path.join(os.getcwd(), "datasets", "mini-imagenet")

    test_input_dataset = []
    test_target_dataset = []

    test_input_dir = os.path.join(mini_imagenet_path, f"jpeg{QF}", "test")
    target_test_dataset_dir = os.path.join(mini_imagenet_path, "_original", "test")

    # 테스트 데이터 로드
    for i in tqdm.tqdm(range(num_classes), desc=f"Loading test data (QF {QF})"):
        test_path = os.path.join(test_input_dir, str(i))
        target_test_path = os.path.join(target_test_dataset_dir, str(i))

        # test_path 내 파일을 정렬된 순서로 불러오기
        sorted_test_files = sorted(os.listdir(test_path))
        sorted_target_test_files = sorted(os.listdir(target_test_path))

        # 두 디렉토리의 파일명이 같은지 확인하며 로드
        for input_file, target_file in zip(sorted_test_files, sorted_target_test_files):
            if input_file.replace(".jpeg", ".png") == target_file:
                # input 이미지 로드
                test_image_path = os.path.join(test_path, input_file)
                test_image = cv2.imread(test_image_path)
                test_input_dataset.append(test_image)

                # target 이미지 로드
                target_image_path = os.path.join(target_test_path, target_file)
                target_image = cv2.imread(target_image_path)
                test_target_dataset.append(target_image)

            else:
                print(
                    f"Warning: Mismatched files in testing set: {input_file} and {target_file}"
                )

    # Dataset과 DataLoader 생성
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataset, test_loader


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


class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = [
            nn.Sequential(
                nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
        ]
        for i in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        return y - residual


class BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = F.relu(out)

        return out


class BlockCNN(nn.Module):
    def __init__(self):
        super(BlockCNN, self).__init__()
        k = 64
        self.conv_1 = nn.Conv2d(3, k, (3, 5), (1, 1), padding=(1, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(k)

        self.layer_1 = BottleNeck(k, k)
        self.layer_2 = BottleNeck(k, k)

        self.conv_2 = nn.Conv2d(k, k * 2, (3, 5), (1, 1), padding=(1, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(k * 2)

        self.layer_3 = BottleNeck(k * 2, k * 2)

        self.conv_3 = nn.Conv2d(
            k * 2, k * 4, (1, 5), (1, 1), padding=(0, 2), bias=False
        )
        self.bn3 = nn.BatchNorm2d(k * 4)

        self.layer_4 = BottleNeck(k * 4, k * 4)
        self.layer_5 = BottleNeck(k * 4, k * 4)

        self.conv_4 = nn.Conv2d(
            k * 4, k * 8, (1, 1), (1, 1), padding=(0, 0), bias=False
        )
        self.bn4 = nn.BatchNorm2d(k * 8)

        self.layer_6 = BottleNeck(k * 8, k * 8)

        self.conv_5 = nn.Conv2d(k * 8, k * 4, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(k * 4)

        self.layer_7 = BottleNeck(k * 4, k * 4)

        self.conv_6 = nn.Conv2d(k * 4, k * 2, 1, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm2d(k * 2)

        self.layer_8 = BottleNeck(k * 2, k * 2)

        self.conv_7 = nn.Conv2d(k * 2, k, 1, 1, 0, bias=False)
        self.bn7 = nn.BatchNorm2d(k)

        self.layer_9 = BottleNeck(k, k)

        # self.conv_8 = nn.Conv2d(k*2, COLOR_CHANNELS, 1, 1, 0, bias=False)

        self.conv_8 = nn.Conv2d(k, 3, 1, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.squeeze(1)
        out = F.relu(self.bn1(self.conv_1(x)))
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = F.relu(self.bn2(self.conv_2(out)))
        out = self.layer_3(out)
        out = F.relu(self.bn3(self.conv_3(out)))
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = F.relu(self.bn4(self.conv_4(out)))
        out = self.layer_6(out)
        out = F.relu(self.bn5(self.conv_5(out)))
        out = self.layer_7(out)
        out = F.relu(self.bn6(self.conv_6(out)))
        out = self.layer_8(out)
        out = F.relu(self.bn7(self.conv_7(out)))
        out = self.layer_9(out)
        out = self.conv_8(out)
        out = self.sig(out)
        # out = out * 255

        # out = nn.sigmoid(self.conv_8(out))

        return out


# test를 돌릴 때 psnr, ssim 를 평균으로 저장하는 함수 (.csv로 저장)
def save_metrics(metrics, filename):
    with open(filename, "w") as f:
        f.write("PSNR,SSIM,LPIPS Alex\n")
        for i in range(len(metrics["PSNR"])):
            f.write(
                f"{metrics['PSNR'][i]},{metrics['SSIM'][i]},{metrics['LPIPS Alex'][i]}\n"
            )
    print(f"Metrics saved to {filename}")


if __name__ == "__main__":
    QFs = [100, 80, 60, 40, 20]

    # Load the dataset
    # train_dataset, test_dataset, train_loader, test_loader = load_images()
    train_dataset, train_loader = laod_mini_imagenet_train_dataset_and_dataloader()

    model_names = [
        "BlockCNN",
    ]

    for model_name in model_names:
        if model_name == "ARCNN":
            model = ARCNN()
        elif model_name == "FastARCNN":
            model = FastARCNN()
        elif model_name == "DnCNN":
            model = DnCNN()
        elif model_name == "BlockCNN":
            model = BlockCNN()
        print(model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPUs")

        model.to(device)
        print(f"Model device: {device}")

        # # train the model
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # # !load model
        # model.load_state_dict(torch.load(f"./models/{type(model).__name__}_30.pth"))

        start_time = time.time()
        print(f"Training started at {time.ctime(start_time)}")
        logging.info(f"{model_name} Training started at {time.ctime(start_time)}")
        print(f"Training for {epochs} epochs")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (input_images, target_images) in enumerate(
                tqdm.tqdm(train_loader, desc=f"Train Epoch {model_name} {epoch+1}/{epochs}")
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
            logging.info(
                f"{model_name} Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}"
            )

            # Save the model
            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                torch.save(
                    model.state_dict(),
                    os.path.join("models", f"{model_name}_{epoch+1}.pth"),
                )
                print(f"Model saved at epoch {epoch+1}")
                logging.info(f"Model {model_name} saved at epoch {epoch+1}")

        end_time = time.time()
        print(f"Training finished at {time.ctime(end_time)}")
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        logging.info(f"Training finished at {time.ctime(end_time)}")
        logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        # Save the final model
        torch.save(
            model.state_dict(),
            os.path.join("models", f"{model_name}_final.pth"),
        )
        print(f"Final model saved as {model_name}_final.pth")
        logging.info(f"Final model saved as {model_name}_final.pth")

        # test for each JPEG QF
        for QF in QFs:
            test_dataset, test_dataloader = (
                laod_mini_imagenet_test_dataset_and_dataloader(QF)
            )
            model.eval()
            idx = 0
            test_loss = 0.0
            psnr_values = []
            ssim_values = []
            lpips_alex_values = []
            lpips_alex_model = lpips.LPIPS(net="alex").to(device)

            with torch.no_grad():
                for input_images, target_images in tqdm.tqdm(
                    test_dataloader, desc="Testing"
                ):
                    input_images = input_images.to(device)
                    target_images = target_images.to(device)

                    # Forward pass
                    outputs = model(input_images)

                    # Calculate MSE loss
                    loss = criterion(outputs, target_images)
                    test_loss += loss.item()

                    for i in range(len(outputs)):
                        # Calculate LPIPS
                        lpips_alex = lpips_alex_model(
                            target_images[i], outputs[i], normalize=True
                        ).item()

                        target_image = target_images[i].cpu().numpy()
                        output_image = outputs[i].cpu().numpy()

                        # Calculate PSNR
                        psnr = peak_signal_noise_ratio(
                            target_image, output_image, data_range=1.0
                        )

                        # Calculate SSIM
                        ssim = structural_similarity(
                            target_image,
                            output_image,
                            multichannel=True,
                            data_range=1.0,
                            channel_axis=0,
                        )

                        lpips_alex_values.append(lpips_alex)
                        psnr_values.append(psnr)
                        ssim_values.append(ssim)

                        # # save the output images
                        # os.makedirs(f"{model_name}_mini_imagenet/jpeg{QF}/test", exist_ok=True)
                        # output_image_path = os.path.join(
                        #     f"{model_name}_mini_imagenet",
                        #     f"jpeg{QF}",
                        #     f"output{idx:05d}.png",
                        # )
                        # np.clip(output_image, 0, 1, out=output_image)
                        # output_image = np.transpose(output_image, (1, 2, 0)) * 255
                        # cv2.imwrite(output_image_path, output_image)
                        # idx += 1

            # Calculate average metrics
            avg_test_loss = test_loss / len(test_dataloader)
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            avg_lpips_alex = np.mean(lpips_alex_values)

            print(
                f"Model: {model_name}, Epoch: {epochs}, QF: {QF}, Training Time: {time.ctime(end_time)}, Test Loss: {avg_test_loss:.4f}, Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {np.mean(ssim_values):.4f}, Average LPIPS Alex: {avg_lpips_alex:.4f}"
            )
            logging.info(
                f"Model: {model_name}, Epoch: {epochs}, QF: {QF}, Training Time: {time.ctime(end_time)}, Test Loss: {avg_test_loss:.4f}, Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {np.mean(ssim_values):.4f}, Average LPIPS Alex: {avg_lpips_alex:.4f}"
            )

            # Save metrics
            metrics = {
                "QF": QF,
                "Test Loss": [avg_test_loss],
                "PSNR": psnr_values,
                "SSIM": ssim_values,
                "LPIPS Alex": lpips_alex_values,
            }
            os.makedirs("metrics", exist_ok=True)
            save_metrics(
                metrics,
                os.path.join("metrics", f"{model_name}_mini_imagenet_metrics.csv"),
            )
