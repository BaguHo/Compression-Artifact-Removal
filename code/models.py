import torch
import torch.nn as nn
import torch.nn.functional as F

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

class PxT_8x8_y(nn.Module):
    """
        ViT 기본 파라미터
        img_size = 224
        patch_size = 16
        in_channels = 3
        embed_dim = 768
        num_heads = 12
        num_layers = 12
        mlp_dim = 3072
    """
    def __init__(
        self,
        img_size=8,
        patch_size=1,
        in_channels=1,
        embed_dim=64,
        num_heads=8,
        num_layers=8,
        mlp_dim=128,
    ):
        super(PxT_8x8_y, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.transformer = nn.ModuleList([Encoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.decoder = nn.Sequential(nn.Linear(embed_dim, self.patch_dim))

    def forward(self, x):
        batch_size = x.size(0)

        # [B, C, H, W ] --> [B, N, D]
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, self.patch_dim)

        x = self.patch_embed(x)
        # ! position id가 없음
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

class PxT_8x8_ycrcb(nn.Module):
    """
        ViT 기본 파라미터
        img_size = 224
        patch_size = 16
        in_channels = 3
        embed_dim = 768
        num_heads = 12
        num_layers = 12
        mlp_dim = 3072
    """
    def __init__(
        self,
        img_size=8,
        patch_size=1,
        in_channels=3,
        embed_dim=256,
        num_heads=8,
        num_layers=8,
        mlp_dim=512,
    ):
        super(PxT_8x8_ycrcb, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.transformer = nn.ModuleList([Encoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.decoder = nn.Sequential(nn.Linear(embed_dim, self.patch_dim))

    def forward(self, x):
        batch_size = x.size(0)

        # [B, C, H, W ] --> [B, N, D]
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, self.patch_dim)

        x = self.patch_embed(x)
        # ! position id가 없음
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

# 14.5M parameter needs 총 3억 훈련 토큰
class PxT_32x32_ycbcr(nn.Module):
    """
    32x32 YCbCr 입력 JPEG 아티팩트 제거 모델
    - 8x8 블록 단위의 패치Embedding
    - 3색상 채널을 384차원으로 임베딩
    - 16개의 attention layer를 가진 16-layer transformer encoder
    - 256차원으로 2x2 Downsampling
    - 384차원으로 2x2 Upsampling
    - 3색상 채널로 32x32 출력
    """

    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=384,
        num_heads=12,
        num_layers=12, 
        mlp_dim=1572   # 더 큰 차원으로 수정
    ):
        super(PxT_32x32_ycbcr, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        # 8x8 블록 단위의 패치Embedding
        # 2D 이미지를 patch개수, patch_size로 변경
        # R(h*w*c) --> R(N*(P^2*C))
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        # 포지션 임베딩
        # 각 패치의 위치 정보를 학습할 수 있도록 패치 개수 x 임베딩 차원으로 설정
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # 16개의 attention layer를 가진 16-layer transformer encoder
        self.transformer = nn.ModuleList([Encoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])

        # Decoder 구조 개선
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_dim),  
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),  
            nn.GELU(),
            nn.Linear(mlp_dim, self.patch_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, self.patch_dim)

        shallow_feature = self.patch_embed(x)  # [B, N, D]
        x = shallow_feature + self.position_embeddings

        # permute(1, 0, 2): Transformer의 입력 형식에 맞게 차원 순서 변경
        # (batch_size, num_patches, embed_dim) -> (num_patches, batch_size, embed_dim)
        # PyTorch MultiheadAttention이 이 형식을 기대함
        x = x.permute(1, 0, 2)
        for layer in self.transformer:
            x = layer(x)
        # permute(1, 0, 2): 원래의 차원 순서로 복원
        # (num_patches, batch_size, embed_dim) -> (batch_size, num_patches, embed_dim)
        x = x.permute(1, 0, 2)
        combined = torch.cat([shallow_feature, x], dim=-1)
        x = self.decoder(combined)

        x = x.view(
            batch_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            3,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, 3, self.img_size, self.img_size)
        return x
    
class PxT_224x224_ycbcr(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=8,
        in_channels=3,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_dim=3072,
    ):
        super(PxT_224x224_ycbcr, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 784
        self.patch_dim = in_channels * patch_size * patch_size  # 192
        self.embed_dim = embed_dim

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
            3,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, 3, self.img_size, self.img_size)
        return x
    
# post processing models
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

        # out = torch.sigmoid(self.conv_8(out))

        return out