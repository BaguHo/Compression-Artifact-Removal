import cv2
import os

jpeg_image = cv2.imread("./datasets/CIFAR100/original_size/jpeg80/train/0/image_00002_laebl_000.jpeg")
removed_iamge = cv2.imread("./datasets/BlockCNN_cifar100/jpeg80/train/000/image_00001.png")

print(jpeg_image.shape, removed_iamge.shape)
print("dtype:", jpeg_image.dtype, removed_iamge.dtype)
print("max:", jpeg_image.max(), removed_iamge.max())
print("min:", jpeg_image.min(), removed_iamge.min())
