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
    output_images = []
    output_image_names = []
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

            for image_name in images_names:
                image = plt.imread(os.path.join(input_image_path, image_name))
                image = (image * 255).astype(np.uint8)
                plt.imshow(image)
                plt.show()
                input_images.append(image)

            image_length = len(os.listdir(input_image_path)) // 16

            for j in range(image_length):
                big_image = np.zeros((32, 32, 3), dtype=np.uint8)
                for k in range(16):
                    image = input_images[j * 16 + k]
                    image = np.array(image)

                    big_image[
                        (k // 4) * 8 : (k // 4) * 8 + 8,
                        (k % 4) * 8 : (k % 4) * 8 + 8,
                        :,
                    ] = image
                    print(big_image)

                output_image = Image.fromarray(big_image)

                plt.imshow(output_image)
                plt.show()
                input()
                os.makedirs(output_path, exist_ok=True)
                output_images.append(output_image)
                output_image_names.append(os.path.join(output_path, f"output_{j}.png"))
                output_images[j].save(output_image_names[j])
                print(f"Saved {output_path}/output_{j}.png")

            # for j in range(len(os.listdir(input_image_path)) // 16):
            # print(f"slice index: {16*j} : {16*(j + 1)}")
            # slice_images_names = images_names[16 * j : 16 * (j + 1)]
            # print(slice_images_names)

            # for k in range(16):
            #     small_image = os.path.join(input_image_path, slice_images_names[k])
            #     image = plt.imread(small_image)
            #     image = np.array(image)
            #     big_image[
            #         (k // 4) * 8 : (k // 4) * 8 + 8,
            #         (k % 4) * 8 : (k % 4) * 8 + 8,
            #         :,
            #     ] = image
            # print(j, k)
            # output_image = Image.fromarray(big_image)
            # os.makedirs(output_path, exist_ok=True)
            # output_images.append(output_image)
            # output_image_names.append(os.path.join(output_path, f"output_{j}.png"))
            # output_image.save(os.path.join(output_path, f"output_{j}.png"))
            # if j == 10:
            #     end_time = time.time()
            #     print(f"time: {end_time - start_time}")

        # for i in range(len(output_images)):
        #     output_images[i].save(output_image_names[i])
        # print(f"Saved {output_path}/output_{j}.png")
