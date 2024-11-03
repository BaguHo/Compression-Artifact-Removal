import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

QFs = [80, 60, 40, 20]


def save_CIFAR100():
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR100(
        root="./datasets", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root="./datasets", train=False, download=True, transform=transform
    )

    class_names = train_dataset.classes

    output_dir = os.path.join(
        os.getcwd(), "datasets", "CIFAR100", "original_size", "original"
    )

    # save original dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(class_names)):
            train_class_dir = os.path.join(output_dir, "train", str(i))
            test_class_dir = os.path.join(output_dir, "test", str(i))
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

        print("Saving training images...")
        for idx, (image, label) in enumerate(train_dataset):
            image = transforms.ToPILImage()(image)

            image_filename = os.path.join(
                output_dir, "train", str(label), f"image_{idx}_laebl_{label}.png"
            )
            image.save(image_filename, "PNG")

            if idx % 5000 == 0:
                print(f"{idx} training images saved...")

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

        if not os.path.exists(jpeg_output_dir):
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
                    f"image_{idx}_laebl_{label}.png",
                )
                image.save(image_filename, "PNG")

                if idx % 5000 == 0:
                    print(f"{idx} jpeg training images saved...")

            print(f"Saving jpeg {QF} test images...")
            for idx, (image, label) in enumerate(test_dataset):
                image = transforms.ToPILImage()(image)

                image_filename = os.path.join(
                    jpeg_output_dir,
                    "test",
                    str(label),
                    f"image_{idx}_laebl_{label}.png",
                )

                image.save(image_filename, "PNG")

                if idx % 2000 == 0:
                    print(f"{idx} jpeg test images saved...")

    print("All jpeg images have been saved successfully.")
