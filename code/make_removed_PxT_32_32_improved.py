from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage as to_pil
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import requests
import os, sys, re
import logging
import cv2
import tqdm
import time
from knockknock import slack_sender
from PIL import Image

if len(sys.argv) < 3:
    print("Usage: python script.py <batch_size> <num_workers> ")
    sys.exit(1)

logging.basicConfig(
    filename="data.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

dataset_name = "CIFAR100"
slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)
QFs = [100, 80, 60, 40, 20]

model_names = [
    "MAE_improved_PxT"
]
batch_size = int(sys.argv[1])
num_workers = int(sys.argv[2])
num_classes = 100

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class CustomDataset(Dataset):
    def __init__(self, input_images, target_images, transform=transform):
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


def load_images(QF):
    dataset_name = "CIFAR100"
    cifar100_path = os.path.join(os.getcwd(), "datasets", dataset_name, "original_size")

    train_input_dataset = []
    test_input_dataset = []
    train_target_dataset = []
    test_target_dataset = []

    # input images
    train_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "train")
    test_input_dir = os.path.join(cifar100_path, f"jpeg{QF}", "test")

    # target images (original)
    target_train_dataset_dir = os.path.join(cifar100_path, "original", "train")
    target_test_dataset_dir = os.path.join(cifar100_path, "original", "test")

    # # 학습 데이터 로드
    # for i in tqdm.tqdm(
    #     range(num_classes), desc=f"Loading train data (QF {QF})", total=num_classes
    # ):
    #     train_path = os.path.join(train_input_dir, str(i).zfill(3))
    #     target_train_path = os.path.join(target_train_dataset_dir, str(i).zfill(3))

    #     # train_path 내 파일을 정렬된 순서로 불러오기
    #     sorted_train_files = sorted(os.listdir(train_path))
    #     sorted_target_train_files = sorted(os.listdir(target_train_path))

    #     # 두 디렉토리의 파일명이 같은지 확인하며 로드
    #     for train_file, target_file in zip(
    #         sorted_train_files, sorted_target_train_files
    #     ):
    #         if train_file.replace("jpeg", "png") == target_file:
    #             # input 이미지 로드
    #             train_image_path = os.path.join(train_path, train_file)
    #             train_image = Image.open(train_image_path)
    #             train_input_dataset.append(train_image)

    #             # target 이미지 로드
    #             target_image_path = os.path.join(target_train_path, target_file)
    #             target_image = Image.open(target_image_path)
    #             train_target_dataset.append(target_image)
    #         else:
    #             print(
    #                 f"Warning: Mismatched files in training set: {train_file} and {target_file}"
    #             )

    # 테스트 데이터 로드
    for i in tqdm.tqdm(range(num_classes), desc=f"Loading test data (QF {QF})"):
        test_path = os.path.join(test_input_dir, str(i).zfill(3))
        target_test_path = os.path.join(target_test_dataset_dir, str(i).zfill(3))

        # test_path 내 파일을 정렬된 순서로 불러오기
        sorted_test_files = sorted(os.listdir(test_path))
        sorted_target_test_files = sorted(os.listdir(target_test_path))

        # 두 디렉토리의 파일명이 같은지 확인하며 로드
        for test_file, target_file in zip(sorted_test_files, sorted_target_test_files):
            if test_file.replace("jpeg", "png") == target_file:
                # input 이미지 로드
                test_image_path = os.path.join(test_path, test_file)
                test_image = Image.open(test_image_path)
                test_input_dataset.append(test_image)

                # target 이미지 로드
                target_image_path = os.path.join(target_test_path, target_file)
                target_image = Image.open(target_image_path)
                test_target_dataset.append(target_image)
            else:
                print(
                    f"Warning: Mismatched files in testing set: {test_file} and {target_file}"
                )

    # Dataset과 DataLoader 생성
    # train_dataset = CustomDataset(train_input_dataset, train_target_dataset)
    test_dataset = CustomDataset(test_input_dataset, test_target_dataset)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return None, test_dataset, None, test_loader


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


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(Encoder, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        # Multi-head self-attention with residual connection
        y = self.ln1(x)
        y, _ = self.mhsa(y, y, y)
        x = x + y

        # MLP with residual connection
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + y
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(Decoder, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, embed_dim)
        )
        
    def forward(self, x):
        # Multi-head self-attention with residual connection
        y = self.ln1(x)
        y, _ = self.mhsa(y, y, y)
        x = x + y
        
        # MLP with residual connection
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + y
        return x

class MAE_Improved_PxT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=256,       # 임베딩 차원 증가
        encoder_num_heads=8,
        decoder_num_heads=4, # 디코더는 인코더보다 작은 헤드 수
        encoder_layers=12,   # 인코더 레이어 수 증가
        decoder_layers=4,    # 디코더 레이어 수 감소
        encoder_mlp_dim=1024,# 인코더 MLP 차원 증가
        decoder_mlp_dim=512, # 디코더 MLP 차원 (인코더보다 작게)
        masking_ratio=0.25,  # 마스킹 비율 (25%)
        dropout=0.1,
    ):
        super(MAE_Improved_PxT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim
        self.masking_ratio = masking_ratio
        
        # 패치 임베딩
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        
        # 위치 임베딩
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 마스크 토큰 (디코더에서 사용)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 인코더
        self.encoder_layers = nn.ModuleList(
            [Encoder(embed_dim, encoder_num_heads, encoder_mlp_dim) for _ in range(encoder_layers)]
        )
        
        # 디코더 (인코더보다 경량화)
        self.decoder_layers = nn.ModuleList(
            [Decoder(embed_dim, decoder_num_heads, decoder_mlp_dim) for _ in range(decoder_layers)]
        )
        
        # 최종 출력 레이어 (패치 재구성)
        self.decoder_pred = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.patch_dim)
        )
        
        # 초기화
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def random_masking(self, x, mask_ratio):
        """랜덤 마스킹: 토큰의 일부를 마스킹하고 마스크 인덱스 반환"""
        B, N, D = x.shape  # 배치, 토큰 수, 차원
        
        # 마스킹할 토큰 수
        len_keep = int(N * (1 - mask_ratio))
        
        # 각 배치별로 서로 다른 랜덤 순열 생성
        noise = torch.rand(B, N, device=x.device)  # 노이즈 생성
        ids_shuffle = torch.argsort(noise, dim=1)  # 정렬된 인덱스
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 복원 인덱스
        
        # 유지할 토큰의 인덱스 (앞부분만 유지)
        ids_keep = ids_shuffle[:, :len_keep]
        
        # 마스킹: 유지할 토큰만 선택
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # 마스크 생성 (마스킹된 위치는 True)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        # 이미지를 패치로 변환 (B, C, H, W) -> (B, N, C*P*P)
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, self.patch_dim)
        
        # 패치 임베딩 + 위치 임베딩
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # 랜덤 마스킹
        x, mask, ids_restore = self.random_masking(x, self.masking_ratio)
        
        # 인코더 통과 (보이는 패치만)
        x = x.permute(1, 0, 2)  # (N_vis, B, D)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # (B, N_vis, D)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        B, N_vis, D = x.shape  # 배치, 보이는 토큰 수, 차원
        N = self.num_patches  # 전체 패치 수
        
        # 마스크 토큰 확장
        mask_tokens = self.mask_token.repeat(B, N - N_vis, 1)
        
        # 전체 시퀀스에 마스크 토큰과 인코더 출력 결합
        x_ = torch.cat([x, mask_tokens], dim=1)  # (B, N, D)
        
        # ids_restore를 사용하여 원래 위치로 복원
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # 위치 임베딩 추가
        x = x_ + self.pos_embed
        
        # 디코더 통과
        x = x.permute(1, 0, 2)  # (N, B, D)
        for layer in self.decoder_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # (B, N, D)
        
        # 패치 예측
        x = self.decoder_pred(x)
        
        return x
    
    def forward(self, x, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.masking_ratio
        
        # 인코딩
        latent, mask, ids_restore = self.forward_encoder(x)
        
        # 디코딩
        pred = self.forward_decoder(latent, ids_restore)
        
        # 이미지 재구성
        pred = pred.view(
            x.shape[0],
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            3,
            self.patch_size,
            self.patch_size,
        )
        pred = pred.permute(0, 3, 1, 4, 2, 5).contiguous()
        pred = pred.view(x.shape[0], 3, self.img_size, self.img_size)
        
        return pred, mask


if __name__ == "__main__":
    for QF in QFs:
        # Load the dataset
        train_dataset, test_dataset, train_loader, test_loader = load_images(QF)

        # Initialize the model
        for model_name in model_names:
            if model_name == "ARCNN":
                model = ARCNN()
            elif model_name == "DnCNN":
                model = DnCNN()
            elif model_name == "BlockCNN":
                model = BlockCNN()
            elif model_name == "MAE_improved_PxT":
                model = MAE_Improved_PxT()
            
            if QF == 100:
                print(model)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # use multiple GPUs if available
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                print(f"Using {torch.cuda.device_count()} GPUs")

            model.to(device)
            print(f"Model device: {device}")

            # train the model
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.load_state_dict(
                torch.load(
                    f"./models/{model_name}_20.pth", map_location=device
                )
            )
            # !load model
            # model.load_state_dict(
            #     torch.load(
            #         f"./models/{model_name}.pth", map_location=device
            #     )
            # )

            # Test the model
            model.eval()
            test_loss = 0.0
            psnr_values = []
            ssim_values = []
            psnr_b_values = []

            with torch.no_grad():
                rgb_train_images = []
                rgb_test_images = []
                temp_image_name_idx = 0
                image_idx = 0
                class_idx = 0

                # for input_images, target_images in tqdm.tqdm(
                #     train_loader, desc="Making Train Images"
                # ):
                #     input_images = input_images.to(device)
                #     target_images = target_images.to(device)

                #     # Forward pass
                #     outputs = model(input_images)

                #     # Calculate MSE loss
                #     loss = criterion(outputs, target_images)
                #     test_loss += loss.item()

                #     for i in range(len(outputs)):
                #         rgb_output = outputs[i].cpu().numpy()
                #         np.clip(rgb_output, 0, 1, out=rgb_output)
                #         # rgb_output = np.transpose(rgb_output, (1, 2, 0)) * 255
                #         # rgb_output = rgb_output.astype(np.uint8)
                #         rgb_train_images.append(rgb_output)
                #         image_idx += 1
                #         os.makedirs(
                #             f"datasets/{model_name}_cifar100_v2/jpeg{QF}/train/{class_idx:03d}",
                #             exist_ok=True,
                #         )
                #         # print(rgb_output.shape)
                #         # input()
                #         img = to_pil()(rgb_output.transpose(1, 2, 0))
                #         # print(np.array(img).shape)
                #         # input()
                #         img.save(
                #             f"datasets/{model_name}_cifar100_v2/jpeg{QF}/train/{class_idx:03d}/image_{image_idx:05d}.png"
                #         )
                #         if image_idx % 500 == 0 and image_idx > 0:
                #             image_idx = 0
                #             class_idx += 1

                image_idx = 0
                class_idx = 0
                for input_images, target_images in tqdm.tqdm(
                    test_loader, desc="Making Test Images"
                ):
                    input_images = input_images.to(device)
                    target_images = target_images.to(device)
                
                    # Forward pass
                    outputs, mask = model(input_images)  # 튜플 언패킹 필요
                
                    # 재구성된 픽셀과 원본 픽셀 간의 차이 계산
                    pixel_loss = (outputs - target_images) ** 2

                    # 마스킹된 위치(mask=True)에 대해서만 손실 계산
                    # 8x8 공간 차원으로 재구성 후 4x4 픽셀 단위로 확장
                    expanded_mask = mask.view(-1, 8, 8)  # (B, 8, 8)
                    expanded_mask = expanded_mask.unsqueeze(1)  # (B, 1, 8, 8)
                    expanded_mask = expanded_mask.repeat(1, 3, 1, 1)  # (B, 3, 8, 8)
                    expanded_mask = expanded_mask.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)  # (B, 3, 32, 32)
                    masked_loss = pixel_loss * expanded_mask
                    loss = masked_loss.mean() 

                    # input_image와 target이미지를 각각 "./PxT_32_32_patch_size_8_test_input", "./PxT_32_32_test_target"에 이미지로 저장
                    # input_images = input_images.cpu().numpy()
                    # target_images = target_images.cpu().numpy()
                    # for i in range(len(input_images)):
                    #     input_image = input_images[i]
                    #     target_image = target_images[i]
                    #     input_image = np.transpose(input_image, (1, 2, 0)) * 255
                    #     target_image = np.transpose(target_image, (1, 2, 0)) * 255
                    #     input_image = Image.fromarray(input_image.astype(np.uint8))
                    #     target_image = Image.fromarray(target_image.astype(np.uint8))
                    #     os.makedirs(f"./{model_name}_test_input", exist_ok=True)
                    #     os.makedirs(f"./{model_name}_test_target", exist_ok=True)
                    #     input_image.save(
                    #         f"./{model_name}_test_input/{temp_image_name_idx:05d}.png"
                    #     )
                    #     target_image.save(
                    #         f"./{model_name}_test_target/{temp_image_name_idx:05d}.png"
                    #     )
                    #     temp_image_name_idx += 1

                    for i in range(len(outputs)):
                        rgb_output = outputs[i].cpu().numpy()
                        np.clip(rgb_output, 0, 1, out=rgb_output)
                        # rgb_output = np.transpose(rgb_output, (1, 2, 0)) * 255
                        # rgb_output = rgb_output.astype(np.uint8)
                        rgb_test_images.append(rgb_output)
                        image_idx += 1
                        os.makedirs(
                            f"datasets/{model_name}_cifar100/jpeg{QF}/test/{class_idx:03d}",
                            exist_ok=True,
                        )
                        img = to_pil()(rgb_output.transpose(1, 2, 0))
                        # print(np.array(img).shape)
                        # input()
                        img.save(
                            f"datasets/{model_name}_cifar100/jpeg{QF}/test/{class_idx:03d}/image_{image_idx:05d}.png"
                        )
                        if image_idx % 100 == 0 and image_idx > 0:
                            image_idx = 0
                            class_idx += 1
                            
    # send_slack_message("make_removed_cifar100_post_processing_v2 is done")

