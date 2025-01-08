import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    def __init__(self, depth=17, filters=64, image_channels=3, use_bnorm=True):
        """
        depth       : 전체 레이어 수 (Keras에서 depth와 동일)
        filters     : 중간 레이어의 채널 수 (Keras의 filters와 동일)
        image_channels: 입력/출력 채널 수 (RGB=3 등)
        use_bnorm   : BatchNormalization 적용 여부
        """
        super(DnCNN, self).__init__()

        layers = []

        # -------------------------
        # 1) 첫 번째 conv + ReLU
        # -------------------------
        # Keras:
        # x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='Orthogonal')(inpt)
        # x = Activation('relu')(x)
        #
        # PyTorch:
        layers.append(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=filters,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # -------------------------------------------
        # 2) 중간(conv + BN + ReLU) 레이어를 (depth-2)번 반복
        # -------------------------------------------
        for i in range(depth - 2):
            # conv (use_bnorm에 따라 bias 유무 조절)
            conv = nn.Conv2d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=3,
                padding=1,
                bias=not use_bnorm,
            )
            layers.append(conv)

            # BatchNormalization (옵션)
            if use_bnorm:
                bn = nn.BatchNorm2d(
                    num_features=filters,
                    momentum=0.0,  # Keras momentum=0.0과 유사하게 설정
                    eps=1e-4,
                )  # Keras epsilon=1e-4
                layers.append(bn)

            # ReLU
            layers.append(nn.ReLU(inplace=True))

        # --------------------------
        # 3) 마지막 conv (출력 채널로)
        # --------------------------
        # Keras:
        # x = Conv2D(image_channels, kernel_size=3, padding='same', use_bias=False)(x)
        #
        # PyTorch:
        conv = nn.Conv2d(
            in_channels=filters,
            out_channels=image_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        layers.append(conv)

        # nn.Sequential로 묶기
        self.dncnn = nn.Sequential(*layers)

        # Keras에서 kernel_initializer='Orthogonal'을 사용했던 것에 맞춰
        # PyTorch에서도 Orthogonal 초기화를 진행
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Keras의 x = inpt - x(최종 conv 출력) 과 동일하게,
        PyTorch에서는 out = x - self.dncnn(x)를 통해 residual 학습을 수행.
        """
        out = self.dncnn(x)
        # 최종 출력은 입력 x에서 예측된 노이즈 out을 빼준 값
        return x - out
