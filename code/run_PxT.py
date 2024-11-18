import torch
import torch.nn as nn
import PIL
from PIL import Image
import numpy as np
import os

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


# MS COCO dataset
def load_sample_data():
    smaple_data_path = os.path.join(
        ".", "datasets", "CIFAR100", "8x8_images", f"jpeg{QF}", "test", "0"
    )
    # Load 16 images from sample data path
    images = []
    image_names = sorted(os.listdir(smaple_data_path))
    image_names = image_names[:16]

    for i in range(len(image_names)):
        image = PIL.Image.open(os.path.join(smaple_data_path, image_names[i]))
        image = np.array(image)
        images.append(image)

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

    # Load sample data
    images = load_sample_data()

    # Convert images to tensor and move them to the device
    images_tensor = [
        torch.tensor(image).permute(2, 0, 1).float() / 255 for image in images
    ]  # Normalize between 0 and 1

    # Stack into a batch of size (16, 3, H, W)
    images_tensor = torch.stack(images_tensor)

    # Load model and move it to the appropriate device (CPU in this case)
    model = ViT().to(device)
    model.load_state_dict(
        torch.load(os.path.join(".", "models", "PxT.pth"), map_location=device),
        strict=False,
    )

    # Set model to evaluation mode (important for inference)
    model.eval()

    output = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for image in images_tensor:
            # Add batch dimension and pass through model
            output.append(model(image.unsqueeze(0)).detach().cpu().numpy())

    #  show the output images through PIL as a grid
    output = np.array(output)
    output = output.squeeze()
    output = output.transpose(0, 2, 3, 1)
    output = output.reshape(4, 4, 8, 8, 3)
    output = output.transpose(0, 2, 1, 3, 4)
    output = output.reshape(4 * 8, 4 * 8, 3)
    output = (output).astype(np.uint8)
    output = Image.fromarray(output)
    output.show()
    
