import glob
import pandas as pd

metrics_path = './metrics/*.csv'

files = glob.glob(metrics_path)

file_names = ["ARCNN_test_metrics.csv", "original_jpeg.csv", "BlockCNN_test_metrics.csv", "DnCNN_test_metrics.csv"]
psnr_avg = 0
ssim_avg = 0
for file_name in file_names:
    df = pd.read_csv(f'./metrics/{file_name}')
    psnr_avg = df['PSNR'].mean()
    ssim_avg = df['SSIM'].mean()
    
    print(file_name)
    print(f'PSNR AVG: {psnr_avg:.2f}')
    print(f'SSIM AVG: {ssim_avg:.4f}')
