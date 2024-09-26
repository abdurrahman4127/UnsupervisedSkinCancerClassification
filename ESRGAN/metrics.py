import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

# Function to calculate PSNR

def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')
    R = 255.0
    psnr = 10 * np.log10((R ** 2) / mse)
    return psnr

# Function to calculate SSIM

def calculate_ssim(original, enhanced, win_size=11):
    score, _ = ssim(original, enhanced, full=True, multichannel=True, win_size=win_size, channel_axis=-1)
    return score

# Function to calculate Perceptual Index (PI)

def calculate_pi(original, enhanced):
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, original.shape[:2][::-1])

    # Calculate the Ma's Metric
    img_diff = np.abs(original.astype(np.float32) - enhanced.astype(np.float32))
    ma_score = np.exp(-np.mean(img_diff))
    return ma_score

# Function to calculate Spatial Correlation Coefficient (SCC)

def calculate_scc(original, enhanced):
    original_flat = original.reshape(-1)
    enhanced_flat = enhanced.reshape(-1)
    scc_score, _ = pearsonr(original_flat, enhanced_flat)
    return scc_score

# Function to calculate Naturalness Image Quality Evaluator (NIQE)

def calculate_niqe(enhanced):
    if len(enhanced.shape) == 3 and enhanced.shape[2] == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    mu = np.mean(enhanced)
    sigma = np.std(enhanced)
    kurtosis = np.mean((enhanced - mu) ** 4) / sigma ** 4
    skewness = np.mean((enhanced - mu) ** 3) / sigma ** 3
    niqe_score = 0.5 * (np.abs(kurtosis - 3) + np.abs(skewness))
    return niqe_score

original_images_dir = '/content/dataset'
enhanced_images_dir = '/content/generated'

# original_images_dir = '/content/ham10000_dataset'
# enhanced_images_dir = '/content/ham10000_generated'

original_images_paths = [os.path.join(original_images_dir, f) for f in os.listdir(original_images_dir)]
enhanced_images_paths = [os.path.join(enhanced_images_dir, f) for f in os.listdir(enhanced_images_dir)]

psnr_values = []
ssim_values = []
pi_values = []
scc_values = []
niqe_values = []

for i in range(len(original_images_paths)):
    original_image = cv2.imread(original_images_paths[i])
    enhanced_image = cv2.imread(enhanced_images_paths[i])

    target_size = (original_image.shape[1], original_image.shape[0])
    enhanced_image = cv2.resize(enhanced_image, target_size)

    psnr_value = calculate_psnr(original_image, enhanced_image)
    ssim_value = calculate_ssim(original_image, enhanced_image)
    pi_value = calculate_pi(original_image, enhanced_image)
    scc_value = calculate_scc(original_image, enhanced_image)
    niqe_value = calculate_niqe(enhanced_image)

    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)
    pi_values.append(pi_value)
    scc_values.append(scc_value)
    niqe_values.append(niqe_value)

results = {
    'Metric': ['PSNR', 'SSIM', 'Perceptual Index', 'SCC', 'NIQE'],
    'Value': [np.mean(psnr_values), np.mean(ssim_values), np.mean(pi_values), np.mean(scc_values), np.mean(niqe_values)]
}

results_df = pd.DataFrame(results)
print(results_df)
# results_df.to_csv('isic_image_enhancement_evaluation.csv', index=False)

# results = {
#     'Metric': ['PSNR', 'SSIM', 'Perceptual Index', 'SCC', 'NIQE'],
#     'Value': [np.mean(psnr_values), np.mean(ssim_values), np.mean(pi_values), np.mean(scc_values), np.mean(niqe_values)]
# }

# results_df = pd.DataFrame(results)
# print(results_df)
# results_df.to_csv('ham10000_image_enhancement_evaluation.csv', index=False)
