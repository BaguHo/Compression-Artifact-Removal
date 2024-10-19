import os
import shutil
import re

source_path = f'./datasets/thermal-face-128x128'
destination_path = f'./datasets/thermal_cropped_images'
files = os.listdir(source_path)

for file in files:
    print(file)
    match = re.search(r'E-(\d+)', file)
    if not match:
        continue
    image_label = match.group(0)
    image_label = image_label.replace('E-', '')
    print(f'image label: {image_label}')
    src_path = os.path.join(source_path, file)
    dest_path  = os.path.join(destination_path, image_label, file)
    print(dest_path)

    folder_path = os.path.join(destination_path, image_label)
    if not os.path.exists(folder_path):
        os.makedirs(os.path.join(destination_path, image_label))
    shutil.copyfile(src_path, dest_path)
