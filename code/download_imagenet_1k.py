from huggingface_hub.hf_api import HfFolder
from datasets import load_dataset
import os
from resize_and_save import resize_and_save_images

if not os.path.exists("datasets/imagenet-1k"):
    #dataset = "ILSVRC/imagenet-1k"
    HfFolder.save_token("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")
    dataset = load_dataset("imagenet-1k")
    # dataset save 
    dataset.save_to_disk("datasets/imagenet-1k")

    resize_and_save_images()
    