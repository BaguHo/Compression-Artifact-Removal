import torch
import torch.nn as nn


class Autoencoder_JPEGArtifactRemoval(nn.Module):
    """
    Autoencoder 기반 JPEG 압축 아티팩트 제거 모델 (32x32, Y채널 입력)
    - 입력: (batch, 1, 32, 32)
    - 8x8 블록 단위의 downsampling/upsampling 구조
    - Encoder-Decoder 구조
    """

    def __init__(self, in_channels=1, base_dim=64):
        super(Autoencoder_JPEGArtifactRemoval, self).__init__()
        self.encoder = nn.Sequential(
            # 8x8 블록 단위로 다운샘플링 (출력: 64채널, 4x4)
            nn.Conv2d(in_channels, base_dim, kernel_size=8, stride=8),
            nn.ReLU(inplace=True),
            # 추가적인 feature 압축
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            # feature 복원
            nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 8x8 블록 단위로 업샘플링 (출력: 1채널, 32x32)
            nn.ConvTranspose2d(base_dim, in_channels, kernel_size=8, stride=8),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        out = self.decoder(z)
        return out
