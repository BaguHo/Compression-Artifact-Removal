import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted

big_image = np.zeros((32, 32, 3), dtype=np.uint8)
original_big_image = np.zeros((32, 32, 3), dtype=np.uint8)

n = 0
input_image_path = os.path.join(".", "datasets", "removed_images", "1")
original_image_path = os.path.join(".", "datasets", "CIFAR100", '8x8_images', "original", 'test', '1')

# original_image_names = natsorted(os.listdir(original_image_path))
# original_imaage_names = original_image_names[16 * n : 16 * (n + 1)]
images_names = natsorted(os.listdir(input_image_path))
images_names = images_names[16 * n : 16 * (n + 1)]
# print(original_image_names)
print(images_names)

for i in range(16):
    image_path = os.path.join(input_image_path, images_names[i])
    image = plt.imread(image_path)
    image = np.array(image)
    big_image[(i // 4) * 8 : (i // 4) * 8 + 8, (i % 4) * 8 : (i % 4) * 8 + 8, :] = image

output_path = os.path.join('.', 'datasets', 'removed_and_merged_images', '1')

for i in range(len(os.listdir(input_image_path))//16):
    big_image = np.zeros((32, 32, 3), dtype=np.uint8)
    original_big_image = np.zeros((32, 32, 3), dtype=np.uint8)
    original_image_names = natsorted(os.listdir(original_image_path))
    original_imaage_names = original_image_names[16 * i : 16 * (i + 1)]
    images_names = natsorted(os.listdir(input_image_path))
    images_names = images_names[16 * i : 16 * (i + 1)]
    for j in range(16):
        image_path = os.path.join(input_image_path, images_names[j])
        image = plt.imread(image_path)
        image = np.array(image)
        big_image[(j // 4) * 8 : (j // 4) * 8 + 8, (j % 4) * 8 : (j % 4) * 8 + 8, :] = image

    plt.imshow(big_image)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path,f'output_{i}.png'))
    plt.show()


# for i in range(16):
#     image_path = os.path.join(original_image_path, original_image_names[i])
#     image = plt.imread(image_path)
#     image = np.array(image)
#     original_big_image[(i // 4) * 8 : (i // 4) * 8 + 8, (i % 4) * 8 : (i % 4) * 8 + 8, :] = image

# plt.imshow(big_image)
# plt.show()

# plt.imshow(original_big_image)
# plt.show()
