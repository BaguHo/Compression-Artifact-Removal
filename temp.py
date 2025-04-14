import numpy as np
from PIL import Image

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data from the CSV file
# file_path = "metrics/classification_results.csv"
# try:
#     data = pd.read_csv(file_path)
# except FileNotFoundError:
#     print(f"File not found: {file_path}")
#     exit()

# # Filter unique models
# models = data["model"].unique()
# datasets = [
#     "JPEG compressed",
#     "ARCNN_cifar100",
#     "BlockCNN_cifar100",
#     "DnCNN_cifar100",
#     "PxT_cifar100",
# ]

# # Plot for each model
# for model in models:
#     plt.figure(figsize=(10, 6), dpi=100)
#     model_data = data[data["model"] == model]

#     # Plot each dataset
#     for dataset in datasets:
#         dataset_data = model_data[model_data["dataset_name"] == dataset]
#         plt.plot(
#             dataset_data["QF"], dataset_data["accuracy"], marker="o", label=dataset
#         )

#     # Configure plot
#     plt.title(f"{model}", fontsize=20)
#     plt.xlabel("QF (Quality Factor)", fontsize=18)
#     plt.ylabel("Accuracy", fontsize=18)
#     plt.legend(title="Dataset")
#     plt.grid(True)
#     plt.show()

# Load the images
image_paths = ['mobilenet.png', 'efficientnet.png', 'vgg19.png']
images = [Image.open(path) for path in image_paths]

# Get the dimensions of all images
widths, heights = zip(*(img.size for img in images))

# Calculate the total width and the maximum height
total_width = sum(widths)
max_height = max(heights)

# Create a new image with the appropriate dimensions
merged_image = Image.new('RGB', (total_width, max_height))

# Paste the images side by side
x_offset = 0
for img in images:
    merged_image.paste(img, (x_offset, 0))
    x_offset += img.size[0]

# Save the merged image
merged_image.save('merged_models.png')
print("Images successfully merged and saved as merged_models.png")