import os
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from PIL import Image

num_classes = 20
QF = 60

for class_number in range(num_classes):
    path = f"./datasets/removed_and_merged_images_50_epoch/{class_number}"
    output_path = f"./datasets/rgb_16x16/{QF}/{class_number}"
    os.makedirs(output_path, exist_ok=True)

    image_names = natsorted(os.listdir(path))
    output_images = []

    idx = 0
    for image_name in image_names:
        image = Image.open(os.path.join(path, image_name))
        np_image = np.array(image)
        # print(np_image.shape)
        np_image[:, :, 0], np_image[:, :, 2] = (
            np_image[:, :, 2].copy(),
            np_image[:, :, 0].copy(),
        )
        idx += 1
        output_image = Image.fromarray(np_image)
        output_image.save(os.path.join(output_path, f"image{idx}.png"))
