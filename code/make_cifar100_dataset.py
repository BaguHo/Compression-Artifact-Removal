import torch
import torchvision
import os
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from PIL import Image
from knockknock import slack_sender

dataset_name = "CIFAR100"
num_classes = 100

slack_webhook_url = (
    "https://hooks.slack.com/services/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"
)


def crop_image(image, crop_size=8):
    width, height = image.size
    cropped_images = []

    for top in range(0, height, crop_size):
        for left in range(0, width, crop_size):
            right = min(left + crop_size, width)
            bottom = min(top + crop_size, height)
            cropped_img = image.crop((left, top, right, bottom))
            cropped_images.append(cropped_img)

    return cropped_images


def save_CIFAR100():
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR100(
        root=os.path.join(".", "datasets"),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR100(
        root=os.path.join(".", "datasets"),
        train=False,
        download=True,
        transform=transform,
    )

    class_names = train_dataset.classes

    output_dir = os.path.join(".", "datasets", "CIFAR100", "original_size", "original")

    # save original dataset
    os.makedirs(output_dir, exist_ok=True)
    print(f"make output dir  {output_dir}")

    for i in range(len(class_names)):
        train_class_dir = os.path.join(output_dir, "train", str(i))
        test_class_dir = os.path.join(output_dir, "test", str(i))
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

    print(f'Saving to {os.path.join(output_dir, "train")}')
    print("Saving training images...")
    for idx, (image, label) in enumerate(train_dataset):
        image = transforms.ToPILImage()(image)

        image_filename = os.path.join(
            output_dir, "train", str(label), f"image_{idx}_laebl_{label}.png"
        )
        image.save(image_filename, "PNG")

        if idx % 5000 == 0:
            print(f"{idx} training images saved...")

    print(f'Saving to {os.path.join(output_dir, "test")}')
    print("Saving test images...")
    for idx, (image, label) in enumerate(test_dataset):
        image = transforms.ToPILImage()(image)

        image_filename = os.path.join(
            output_dir, "test", str(label), f"image_{idx}_laebl_{label}.png"
        )

        image.save(image_filename, "PNG")

        if idx % 2000 == 0:
            print(f"{idx} test images saved...")

    # make and save jepg datsaet for each QF
    for QF in QFs:
        jpeg_output_dir = os.path.join(
            os.getcwd(), "datasets", "CIFAR100", "original_size", f"jpeg{QF}"
        )

        os.makedirs(jpeg_output_dir, exist_ok=True)

        for i in range(len(class_names)):
            train_class_dir = os.path.join(jpeg_output_dir, "train", str(i))
            test_class_dir = os.path.join(jpeg_output_dir, "test", str(i))
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

        print(f"Saving jpeg{QF} training images...")
        for idx, (image, label) in enumerate(train_dataset):
            image = transforms.ToPILImage()(image)

            image_filename = os.path.join(
                jpeg_output_dir,
                "train",
                str(label),
                f"image_{idx}_laebl_{label}.jpeg",
            )
            image.save(image_filename, "JPEG", quality=QF)

            if idx % 5000 == 0:
                print(f"{idx} jpeg training images saved...")

        print(f"Saving jpeg {QF} test images...")
        for idx, (image, label) in enumerate(test_dataset):
            image = transforms.ToPILImage()(image)

            image_filename = os.path.join(
                jpeg_output_dir,
                "test",
                str(label),
                f"image_{idx}_laebl_{label}.jpeg",
            )

            image.save(image_filename, "JPEG", quality=QF)

            if idx % 2000 == 0:
                print(f"{idx} jpeg test images saved...")

    print("All jpeg images have been saved successfully.")
    print("####################################################################")


# 이미지 처리 및 저장 함수 정의
def process_and_save_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # input_dir 내의 모든 이미지 파일 처리
    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)

        with Image.open(img_path) as img:
            # 이미지를 8x8로 자름
            cropped_images = crop_image(img)
            # 잘린 이미지를 output_dir에 저장
            for idx, cropped_img in enumerate(cropped_images):
                cropped_img.save(
                    os.path.join(
                        output_dir, f"{os.path.splitext(img_file)[0]}_crop_{idx}.jpeg"
                    )
                )


def make_8x8_image_from_original_dataset():
    temp_path = os.path.join(os.getcwd(), "datasets", dataset_name)

    for i in range(num_classes):
        input_train_dir = os.path.join(
            temp_path,
            "original_size",
            "original",
            "train",
            str(i),
        )

        input_test_dir = os.path.join(
            temp_path,
            "original_size",
            "original",
            "test",
            str(i),
        )

        output_train_dir = os.path.join(
            temp_path,
            "8x8_images",
            f"original",
            "train",
            str(i),
        )
        output_test_dir = os.path.join(
            temp_path,
            "8x8_images",
            f"original",
            "test",
            str(i),
        )

        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(output_test_dir, exist_ok=True)

        process_and_save_images(input_train_dir, output_train_dir)
        process_and_save_images(input_test_dir, output_test_dir)


def make_8x8_jpeg_image(QF):
    for i in range(num_classes):
        train_dir = os.path.join(
            os.getcwd(),
            "datasets",
            dataset_name,
            "original_size",
            f"jpeg{QF}",
            "train",
            str(i),
        )
        test_dir = os.path.join(
            os.getcwd(),
            "datasets",
            dataset_name,
            "original_size",
            f"jpeg{QF}",
            "test",
            str(i),
        )

        output_train_dir = os.path.join(
            os.getcwd(),
            "datasets",
            dataset_name,
            "8x8_images",
            f"jpeg{QF}",
            "train",
            str(i),
        )
        output_test_dir = os.path.join(
            os.getcwd(),
            "datasets",
            dataset_name,
            "8x8_images",
            f"jpeg{QF}",
            "test",
            str(i),
        )

        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(output_test_dir, exist_ok=True)

        process_and_save_images(train_dir, output_train_dir)
        process_and_save_images(test_dir, output_test_dir)


if __name__ == "__main__":
    QFs = [80, 60, 40, 20]
    save_CIFAR100()
    make_8x8_image_from_original_dataset()
    for QF in QFs:
        # jpeg image 8x8로 저장
        print("making the 8x8 image..")
        make_8x8_jpeg_image(QF)
        print("Done")
