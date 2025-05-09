import torch
import torch.nn as nn


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


class PxT_32x32_y_improved(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=1,
        embed_dim=384,
        num_heads=16,
        num_layers=16,
        mlp_dim=256,
    ):
        super(PxT_32x32_y_improved, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # Convolutional stem for local feature extraction
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self.transformer = nn.ModuleList(
            [Encoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, self.patch_dim),
        )

        # CNN block for artifact suppression after transformer
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        inp = x  # Save input for skip connection

        # Convolutional stem
        x = self.conv_stem(x)

        batch_size = x.size(0)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, self.patch_dim)

        x = self.patch_embed(x)
        x = x + self.position_embeddings

        x = x.permute(1, 0, 2)
        for layer in self.transformer:
            x = layer(x)
        x = x.permute(1, 0, 2)

        x = self.decoder(x)

        x = x.view(
            batch_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, self.in_channels, self.img_size, self.img_size)

        # CNN refinement
        x = self.refine(x)

        # Residual connection
        x = x + inp

        return x


class PxT_32x32_y(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=1,
        embed_dim=384,
        num_heads=16,
        num_layers=16,
        mlp_dim=256,
    ):
        super(PxT_32x32_y, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self.transformer = nn.ModuleList(
            [Encoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        self.decoder = nn.Sequential(nn.Linear(embed_dim, self.patch_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, self.patch_dim)

        x = self.patch_embed(x)
        x = x + self.position_embeddings

        x = x.permute(1, 0, 2)
        for layer in self.transformer:
            x = layer(x)
        x = x.permute(1, 0, 2)

        x = self.decoder(x)

        x = x.view(
            batch_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, self.in_channels, self.img_size, self.img_size)
        return x


class PxT_JPEGArtifactRemoval(nn.Module):
    """
    JPEG 압축 아티팩트 제거를 위한 트랜스포머 기반 모델 (YUV 포맷의 Y 채널 전용).
    - 입력: Y 채널(휘도)만 사용 (in_channels=1)
    - Conv head/tail로 국소적 아티팩트 보정
    - 트랜스포머로 전역 문맥 처리
    """

    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=1,  # Y 채널만 사용
        embed_dim=384,
        num_heads=16,
        num_layers=16,
        mlp_dim=256,
        conv_head_dim=64,
    ):
        super(PxT_JPEGArtifactRemoval, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # Y 채널 입력에 맞춘 컨볼루션 헤드
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, conv_head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_head_dim, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self.transformer = nn.ModuleList(
            [Encoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        self.decoder = nn.Sequential(nn.Linear(embed_dim, self.patch_dim))

        # Y 채널 출력에 맞춘 컨볼루션 테일
        self.conv_tail = nn.Sequential(
            nn.Conv2d(in_channels, conv_head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_head_dim, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Conv head: Y 채널 입력의 국소 특징 추출
        x = self.conv_head(x)

        # Patch embedding
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, self.patch_dim)

        x = self.patch_embed(x)
        x = x + self.position_embeddings

        # Transformer layers: 전역 문맥 처리
        x = x.permute(1, 0, 2)
        for layer in self.transformer:
            x = layer(x)
        x = x.permute(1, 0, 2)

        # Patch decoding
        x = self.decoder(x)
        x = x.view(
            batch_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, self.in_channels, self.img_size, self.img_size)

        # Conv tail: Y 채널 출력의 국소적 복원
        x = self.conv_tail(x)
        return x


class autoencoder(nn.Module):
    """
    Autoencoder 기반 JPEG 압축 아티팩트 제거 모델 (32x32, Y채널 입력)
    - 입력: (batch, 1, 32, 32)
    - 8x8 블록 단위의 downsampling/upsampling 구조
    - Encoder-Decoder 구조
    """

    def __init__(self, in_channels=1, base_dim=64):
        super(autoencoder, self).__init__()
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
