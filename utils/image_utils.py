#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from skimage.metrics import structural_similarity

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2, mask=None): # batch, channel, height, width
    if mask is None:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True) # batch, 1
        return 20 * torch.log10(1.0 / torch.sqrt(mse)) # batch, 1
    else:
        batch, channel = img1.shape[0], img1.shape[1]
        psnrs = torch.tensor([], device="cuda")
        for i in range(batch):
            num_valid = mask[i].sum() * channel
            mse = ((((img1 - img2)) ** 2) * mask).sum() / num_valid
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            psnrs = torch.cat((psnrs, psnr.unsqueeze(0)),dim=0)
        return psnrs # batch, 1
    
def calculate_ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True, data_range=1.0,channel_axis=2)
    # _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True)
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid