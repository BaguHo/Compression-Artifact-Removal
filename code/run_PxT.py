import torch
import torch.nn as nn
import PIL
from PIL import Image
import numpy as np
import os
import cv2
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

QF = 60


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
        y = self.ln1(x)
        y, _ = self.mhsa(y, y, y)
        x = x + y
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + y
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=8,
        patch_size=1,
        in_channels=3,
        embed_dim=128,
        num_heads=16,
        num_layers=16,
        mlp_dim=128,
    ):
        super(ViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
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


def load_sample_data():
    smaple_data_path = os.path.join(
        ".", "datasets", "CIFAR100", "8x8_images", f"jpeg{QF}", "test", "0"
    )
    # Load 16 images from sample data path
    images = []
    image_names = sorted(os.listdir(smaple_data_path))
    image_names = image_names[:16]

    image_paths = []
    for i in range(len(image_names)):
        image_path = os.path.join(smaple_data_path, image_names[i])
        image_paths.append(image_path)
        image = PIL.Image.open(image_path)
        images.append(image)
        # show image
        # image.show()
        # image = np.array(image)
        # images.append(image)
    images[0].show()
    print(f"image path: {image_paths[0]}")

    return images


if __name__ == "__main__":
    # Use GPU if available, otherwise fallback to MPS or CPU
    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else ("mps" if torch.backends.mps.is_available() else "cpu")
    # )
    device = "cpu"
    print(f"Using device: {device}")

    images = load_sample_data()

    # print(np.array(images).shape)
    model = ViT().to(device)
    model.load_state_dict(
        torch.load(os.path.join(".", "models", "PxT.pth"), map_location=device),
        strict=False,
    )

    model.eval()
    print(f'before transform: {images[0]}')
    images = [np.array(image) for image in images]
    print(f'after transform: {images[0]}')
    images = torch.tensor(images).permute(0, 3, 1, 2).float().to(device)
    print(f'after transform: {images[0]}')
    output_tensor_image = model(images[0].unsqueeze(0))
    print(f'model output: {output_tensor_image}')
    output_image = to_pil_image(output_tensor_image[0].cpu())


    plt.figure()
    plt.imshow(output_image)
    plt.show()
    
    
    # images = [
    #     torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    #     for image in images
    # ]
    # for image in images:
    #     print(np.array(image).shape)
    #     plt.imshow(np.array(image))
    #     print(image)
    #     output_image = model(np.array(image))
    #     cv2.imshow("Output Image", output_image)
