from huggingface_hub.hf_api import HfFolder
from datasets import load_dataset
import os

if not os.path.exists("datasets/imagenet-1k"):
    # dataset = "ILSVRC/imagenet-1k"
    dataset = load_dataset("imagenet-1k")
    # dataset save 
    dataset.save_to_disk("datasets/imagenet-1k")