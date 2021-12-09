
import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import time
from PIL import Image
import math
from datetime import datetime
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.utils import make_grid


# model_path = '/content/drive/MyDrive/ColabNotebooks/Testing/script/test/model/SR_model.pth'
# test_img_folder = '/content/drive/MyDrive/ColabNotebooks/dataset/collected_imgs/LR_SCI/'

model_path = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]

def tensor2img_np(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received tensor with dimension: %d' % n_dim)
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img_np(img_np, img_path, mode='RGB'):
    if img_np.ndim == 2:
        mode = 'L'
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)


model = arch.SRResNet(3, 3, 64, 16, upscale=4, norm_type=None, act_type='relu', mode='CNA', res_scale=1, upsample_mode='pixelshuffle')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.cuda()

print('testing...')

total_time = 0
idx = 0
for path in glob.glob(input_dir + '/*'):
    idx += 1
    basename = os.path.basename(path)
    base = os.path.splitext(basename)[0]
    print("test ",idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img * 1.0 / 255
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # matlab imresize
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.cuda()

    start = time.time()
    with torch.no_grad():
        output = model(img_LR).data
    total_time += time.time() - start
    output = tensor2img_np(output.squeeze())
    save_img_np(output, os.path.join(output_dir, base + '.png'))

print('Time for each image: {:.2e}s'.format(total_time/100))