from skimage import train_test_split
from torch.utils.data import DataLoader



train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

test_loader = DataLoader(test_data, shuffle=False)
for original_test, denoised_test in test_loader:
    original_test, denoised_test = original_test, denoised_test

    outputs_test = model(denoised_test)

    outputs_test = outputs_test.numpy()
    original_test = original_test.numpy()

    for i in range(len(outputs_test)):

        psnr_scores.append(psnr(original_test[i], outputs_test[i]))

        patch_size = min(outputs_test[i].shape[0], outputs_test[i].shape[1])
        win_size = min(7, patch_size)

        if win_size >= 3:
            ssim_val = ssim(original_test[i], outputs_test[i], win_size=win_size, channel_axis=-1, data_range=1.0)
            ssim_scores.append(ssim_val)
        else:
            print(f"Skipping SSIM for patch {i} due to insufficient size")


avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
