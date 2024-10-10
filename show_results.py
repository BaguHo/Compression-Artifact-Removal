import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('./results.csv')

cifar10_cifar10 = df[(df['Train Dataset'] == 'CIFAR10') & (
    df['Test Dataset'] == 'CIFAR10') & (df['Model Name'] == 'VGG16')]
jpeg_jpeg = df[(df['Train Dataset'] == 'JPEG') & (df['Test Dataset'] == 'JPEG') & (df['Model Name'] == 'VGG16')]
cifar10_jpeg = df[(df['Train Dataset'] == 'CIFAR10') & (df['Test Dataset'] == 'JPEG') & (df['Model Name'] == 'VGG16')]
jpeg_cifar10 = df[(df['Train Dataset'] == 'JPEG') & (df['Test Dataset'] == 'CIFAR10') & (df['Model Name'] == 'VGG16')]

plt.figure(figsize=(10, 6))
a = np.arange(0, 100, 20)
plt.xticks(a)

plt.plot(cifar10_cifar10['QF'], cifar10_cifar10['Accuracy'], marker='o', label='original-original')
plt.plot(jpeg_jpeg['QF'], jpeg_jpeg['Accuracy'], marker='o', label='jpeg-jpeg')
plt.plot(cifar10_jpeg['QF'], cifar10_jpeg['Accuracy'], marker='o', label='original-jpeg')
plt.plot(jpeg_cifar10['QF'], jpeg_cifar10['Accuracy'], marker='o', label='jpeg-original')

plt.title('Accuracy with Different JPEG Quality Factor (QF)')
plt.xlabel('Quality Factor (QF)')
plt.ylabel('Accuracy')
plt.legend(fontsize='large')
plt.grid(True)

plt.show()
