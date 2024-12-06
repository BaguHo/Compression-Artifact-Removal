import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted
from PIL import Image
import time

num_classes = 20
QFs = [80, 60, 40, 20]

if __name__ == "__main__":
    start_time = time.time()
    input_images = []

    for QF in QFs:
        for i in range(num_classes):
            output_images = []
            output_image_names = []
            print(f"QF: {QF}, class: {i}")

            input_image_path = os.path.join(
                ".",
                "datasets",
                "removed_image_PxT_y_channel",
                f"QF_{QF}",
                "train",
                str(i),
            )
            print(f"input_image_path: {input_image_path}")
            output_path = os.path.join(
                ".",
                "datasets",
                "merged_image_PxT_y_channel",
                f"QF_{QF}",
                "train",
                str(i),
            )
            print(f"output_path: {output_path}")

            images_names = natsorted(os.listdir(input_image_path))

            for image_name in images_names:
                image = plt.imread(os.path.join(input_image_path, image_name))
                input_images.append(image)

            image_length = len(os.listdir(input_image_path)) // 16

            for j in range(image_length):
                big_image = np.zeros((32, 32), dtype=np.uint8)
                for k in range(16):
                    image = input_images[j * 16 + k]
                    image = np.array(image, dtype=np.uint8)

                    big_image[
                        (k // 4) * 8 : (k // 4) * 8 + 8, (k % 4) * 8 : (k % 4) * 8 + 8
                    ] = image

                output_image = Image.fromarray(big_image)

                # plt.imshow(output_image)
                # plt.show()
                # input()
                os.makedirs(output_path, exist_ok=True)
                output_images.append(output_image)
                output_image_names.append(os.path.join(output_path, f"output_{j}.png"))

            print("[images names]")
            for name in output_image_names:
                print(name)

            for j in range(image_length):
                output_images[j].save(output_image_names[j])
                # print(f"Saved {output_path}/output_{j}.png")
            # input()
