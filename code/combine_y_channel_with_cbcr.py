import os
from PIL import Image
import numpy as np
from natsort import natsorted
from matplotlib import pyplot as plt

QFs = [80, 60, 40, 20]
num_classes = 20


for QF in QFs:
    for i in range(num_classes):
        y_images = []
        cbcr_images = []

        for mode in ["train", "test"]:
            y_image_dir = os.path.join(
                ".",
                "datasets",
                "merged_image_PxT_y_channel",
                f"QF_{QF}",
                mode,
                str(i),
            )

            cbcr_image_dir = os.path.join(
                ".",
                "datasets",
                "CIFAR100",
                "original_size",
                f"jpeg{QF}",
                mode,
                str(i),
            )

            y_image_names = natsorted(os.listdir(y_image_dir))
            for image_name in y_image_names:
                img = Image.open(os.path.join(y_image_dir, image_name))
                img_array = np.array(img)[
                    :, :, np.newaxis
                ]  # Add channel dimension for Y
                # print(f"img_array shape: {img_array.shape}")
                # input()
                y_images.append(img_array)

            cbcr_image_names = natsorted(os.listdir(cbcr_image_dir))
            for image_name in cbcr_image_names:
                img = Image.open(os.path.join(cbcr_image_dir, image_name)).convert(
                    "YCbCr"
                )
                # print(f"img shape: {np.array(img).shape}")
                _, cb, cr = img.split()
                # print(f"np.dstack((cb, cr)) shape: {np.dstack((cb, cr)).shape}")
                # input()
                cbcr_images.append(np.dstack((cb, cr)))

            output_path = os.path.join(
                ".", "datasets", "combined_ycbcr", f"QF_{QF}", mode, str(i)
            )
            os.makedirs(output_path, exist_ok=True)

            for idx, (y_img, cbcr) in enumerate(zip(y_images, cbcr_images)):
                y_array = np.array(y_img)
                ycbcr = np.dstack((y_array, cbcr))
                combined_img = Image.fromarray(ycbcr, mode="YCbCr")

                # show combined iamge
                plt.imshow(combined_img)
                plt.show()

                rgb_img = combined_img.convert("RGB")

                output_filename = f"combined_{idx}.png"
                rgb_img.save(os.path.join(output_path, output_filename))
                print(f"saved {os.path.join(output_path, output_filename)}")
