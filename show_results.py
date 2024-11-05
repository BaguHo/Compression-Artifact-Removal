import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("TKAgg")


df = pd.read_csv("./result.csv")

dataset_name_1 = "Tufts Face Database"
dataset_name_2 = "JPEG"
model_name = "resnet50"

cifar10_cifar10 = df[
    (df["Train Dataset"] == dataset_name_1)
    & (df["Test Dataset"] == dataset_name_1)
    & (df["Model Name"] == model_name)
]
jpeg_jpeg = df[
    (df["Train Dataset"] == dataset_name_2)
    & (df["Test Dataset"] == dataset_name_2)
    & (df["Model Name"] == model_name)
]
cifar10_jpeg = df[
    (df["Train Dataset"] == dataset_name_1)
    & (df["Test Dataset"] == dataset_name_2)
    & (df["Model Name"] == model_name)
]
jpeg_cifar10 = df[
    (df["Train Dataset"] == dataset_name_2)
    & (df["Test Dataset"] == dataset_name_1)
    & (df["Model Name"] == model_name)
]

plt.figure(figsize=(10, 6))
a = np.arange(0, 100, 20)
plt.xticks(a)

plt.plot(
    cifar10_cifar10["QF"],
    cifar10_cifar10["Accuracy"],
    marker="o",
    label="original-original",
)
plt.plot(jpeg_jpeg["QF"], jpeg_jpeg["Accuracy"], marker="o", label="jpeg-jpeg")
plt.plot(
    cifar10_jpeg["QF"], cifar10_jpeg["Accuracy"], marker="o", label="original-jpeg"
)
plt.plot(
    jpeg_cifar10["QF"], jpeg_cifar10["Accuracy"], marker="o", label="jpeg-original"
)

plt.title("Accuracy with Different JPEG Quality Factor (QF)")
plt.xlabel("Quality Factor (QF)")
plt.ylabel("Accuracy")
plt.legend(fontsize="large")
plt.grid(True)

plt.show()
