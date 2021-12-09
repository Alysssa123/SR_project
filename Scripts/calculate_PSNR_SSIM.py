import cv2
import os
import math
from datetime import datetime
import numpy as np
from PIL import Image
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as sk_cpt_ssim
import json
import sys

# hr_dir = '/content/drive/MyDrive/SR_Code/dataset/Flickr_face/srgan_hr/'
# sr_dir = '/content/drive/MyDrive/SR_Code/dataset/Flickr_face/srgan_sr/'
# psnr_ssim_score_file = '/content/drive/MyDrive/SR_Code/Eval/PSNR_SSIM_srgan_face.json'

hr_dir = sys.argv[1]
sr_dir = sys.argv[2]
psnr_ssim_score_file = sys.argv[3]

# python3 test.pyfrom torchvision.utils import make_grid

def psnr(img1, img2):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2, multichannel=True):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    return sk_cpt_ssim(img1, img2, multichannel=multichannel)

# calculate PSNR and SSIM
PSNR_all = []
SSIM_all = []

a = 0

for i in os.listdir(hr_dir):
  if not i.endswith('.png'):
    continue
  try:
    hr_img = cv2.imread(hr_dir + i)
    sr_img = cv2.imread(sr_dir + i)

    # sr_img = cv2.imread(sr_dir + i.split('.', 1 )[0] + "_x4_SR.png")
    PSNR = psnr(sr_img, hr_img)
    SSIM = ssim(sr_img, hr_img)

  except Exception as err:
    print("error: ", i, err)
    exit(-1)

  if PSNR == float('inf'):
    print('identical img' + i)

  a += 1

  if a % 1000 ==0:
    print('processing' + i)

# print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(i + 1, base_name, PSNR, SSIM))
  PSNR_all.append(PSNR)
  SSIM_all.append(SSIM)

# save PSNR and SSIM
new_dict = {'PSNR':PSNR_all, 'SSIM':SSIM_all}
with open(psnr_ssim_score_file,'w') as f:
  json.dump(new_dict, f)

# print(PSNR_all)
# print(SSIM_all)
print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
    sum(PSNR_all) / len(PSNR_all),
    sum(SSIM_all) / len(SSIM_all)))

