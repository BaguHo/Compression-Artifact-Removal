import os

def rename_to_zero_padded(folder_path, num_digits=3):
    for name in os.listdir(folder_path):
        old_path = os.path.join(folder_path, name)
        if os.path.isdir(old_path) and name.isdigit():
            new_name = name.zfill(num_digits)
            new_path = os.path.join(folder_path, new_name)
            if old_path != new_path:
                print(f"Renaming: {old_path} -> {new_path}")
                os.rename(old_path, new_path)

# 사용 예시: test 폴더 경로를 아래에 입력하세요.
names = ["jpeg100", "jpeg80", "jpeg60", "jpeg40", "jpeg20", "original"]
for name in names:
    train_folder = "/home/say7fish/Compression-Artifact-Removal/datasets/CIFAR100/original_size/" + name + "/train"
    rename_to_zero_padded(train_folder)