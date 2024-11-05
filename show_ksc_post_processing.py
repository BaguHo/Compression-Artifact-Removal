import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv("./ksc_post_processing.csv")

# Define dataset and model names
dataset_name_1 = "Tufts Face Database"
dataset_name_2 = "JPEG"
dataset_name_3 = "jpeg2000_removed"
dataset_name_4 = "compressed_video_enhancement"
model_name = "resnet50"

# Filter data based on conditions
tufts_tufts = df[
    (df["Train Dataset"] == dataset_name_1)
    & (df["Test Dataset"] == dataset_name_1)
    & (df["Model Name"] == model_name)
]
jpeg2000_jpeg2000 = df[
    (df["Train Dataset"] == dataset_name_3)
    & (df["Test Dataset"] == dataset_name_3)
    & (df["Model Name"] == model_name)
]

video_video = df[
    (df["Train Dataset"] == dataset_name_4)
    & (df["Test Dataset"] == dataset_name_4)
    & (df["Model Name"] == model_name)
]

# Plotting
plt.figure(figsize=(10, 6))
a = np.arange(0, 100, 20)
plt.xticks(a)

# plt.plot(tufts_tufts["QF"], tufts_tufts["Accuracy"], marker="o", label="tufts-tufts")
plt.plot(
    jpeg2000_jpeg2000["QF"],
    jpeg2000_jpeg2000["Accuracy"],
    marker="o",
    label="JPEG2000-JPEG2000",
)
plt.plot(
    tufts_tufts["QF"],
    tufts_tufts["Accuracy"],
    marker="o",
    label="tufts-tufts",
)
plt.plot(
    video_video["QF"],
    video_video["Accuracy"],
    marker="o",
    label="video_enhancement-video_enhancement",
)

# plt.plot(tufts_jpeg["QF"], tufts_jpeg["Accuracy"], marker="o", label="tufts-jpeg")
# plt.plot(jpeg_tufts["QF"], jpeg_tufts["Accuracy"], marker="o", label="jpeg-tufts")

# plt.plot(d3_3["QF"], d3_3["Accuracy"], marker="o", label="jpeg2000-jpeg2000")
# plt.plot(d3_j["QF"], d3_j["Accuracy"], marker="o", label="jpeg2000-jpeg")
# plt.plot(dj_3["QF"], dj_3["Accuracy"], marker="o", label="jpeg-jpeg2000")

# plt.plot(
#     d4_4["QF"],
#     d4_4["Accuracy"],
#     marker="o",
#     label="video_enhancement-video_enhancement",
# )
# plt.plot(d4_j["QF"], d4_j["Accuracy"], marker="o", label="video_enhancement-jpeg")
# plt.plot(dj_4["QF"], dj_4["Accuracy"], marker="o", label="jpeg-video_enhancement")

plt.title("Accuracy with Different JPEG Quality Factor (QF)")
plt.xlabel("Quality Factor (QF)")
plt.ylabel("Accuracy")
plt.legend(fontsize="large")
plt.grid(True)

plt.show()
