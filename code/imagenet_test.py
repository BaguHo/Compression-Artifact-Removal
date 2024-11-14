import torchvision.datasets
import os

def load_imagenet():
    train = datasets.ImageNet(os.path.join('.','datasets'), train=True))
    test = datasets.ImageNet(os.path.join('.', 'datasets', train=False))

    return train,test


if __name__ == __main__:
    load_imagenet()
