import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted

big_image = np.zeros((32, 32, 3), dtype=np.uint8)

num_classes = 20
QFs = [80, 60, 40, 20]

if __name__ == "__main__":
    for QF in QFs:
        for i in range(num_classes):
            input_image_path = os.path.join(
                ".", "datasets", "removed_images_50_epoch_each_QF", f"QF_{QF}", str(i)
            )
            output_path = os.path.join(
                ".",
                "datasets",
                "removed_and_merged_images_50_epoch_QF",
                f"QF_{QF}",
                str(i),
            )
            images_names = natsorted(os.listdir(input_image_path))
            print(images_names)

            for j in range(len(os.listdir(input_image_path)) // 16):
                slice_images_names = images_names[16 * j : 16 * (j + 1)]
                for k in range(16):
                    image_path = os.path.join(input_image_path, slice_images_names[k])
                    image = plt.imread(image_path)
                    image = np.array(image)
                    big_image[
                        (k // 4) * 8 : (k // 4) * 8 + 8,
                        (k % 4) * 8 : (k % 4) * 8 + 8,
                        :,
                    ] = image

                big_image = np.zeros((32, 32, 3), dtype=np.uint8)
                slice_images_names = natsorted(os.listdir(input_image_path))
                slice_images_names = slice_images_names[16 * j : 16 * (j + 1)]
                for k in range(16):
                    image_path = os.path.join(input_image_path, slice_images_names[k])
                    image = plt.imread(image_path)
                    image = np.array(image)
                    big_image[
                        (k // 4) * 8 : (k // 4) * 8 + 8,
                        (k % 4) * 8 : (k % 4) * 8 + 8,
                        :,
                    ] = image

                plt.imshow(big_image)
                os.makedirs(output_path, exist_ok=True)
                plt.savefig(os.path.join(output_path, f"output_{j}.png"))
                print(f"Saved {output_path}/output_{j}.png")
