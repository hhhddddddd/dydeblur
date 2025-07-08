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
import sys
import cv2
from datetime import datetime
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from typing import NamedTuple

class TrackObservations(NamedTuple):
    xyz: torch.Tensor
    visibles: torch.Tensor
    invisibles: torch.Tensor
    confidences: torch.Tensor
    colors: torch.Tensor

class StaticObservations(NamedTuple):
    xyz: torch.Tensor
    normals: torch.Tensor
    colors: torch.Tensor

def inverse_sigmoid(x):
    x = torch.clamp(x, 1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))

def gumbel_sigmoid(input, temperature=1, eps = 1e-10):
    with torch.no_grad():
        uniform1 = torch.rand(input.size())
        uniform2 = torch.rand(input.size())
        gumbel_noise = -torch.log(torch.log(uniform1 + eps)/torch.log(uniform2 + eps) + eps).cuda()
    reparam = (input + gumbel_noise)/temperature
    ret = torch.sigmoid(reparam)
    return ret, gumbel_noise

def inverse_gumbel_sigmoid(output, gumbel_noise, temperature=1, eps = 1e-10):
    return torch.log(output / (1 - output)) * temperature - gumbel_noise

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution) # 800, 800
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0 # 800, 800, 3; MARK: 0~255 -> 0~1
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1) # 3, 800, 800
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1) # 1, 800, 800


def ArrayToTorch(array, resolution):
    # resized_image = np.resize(array, resolution)
    resized_image_torch = torch.from_numpy(array)

    if len(resized_image_torch.shape) == 3:
        return resized_image_torch.permute(2, 0, 1)
    else:
        return resized_image_torch.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0: # never happen
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r): # rotate quaternion -> rotation matrix
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda") # [gs_num, 3, 3]
    R = build_rotation(r) # rotate quaternion -> rotation matrix

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L # R @ S
    return L


def safe_state(silent): # print function
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))

def Pseudocolorization(depth):
    depth = depth - depth.min()
    depth = depth / depth.max()

    depth *= 255

    # cv2.COLORMAP_JET, blue represents a higher depth value, and red represents a lower depth value
    # The value of alpha in the cv.convertScaleAbs() function is related to the effective distance in the depth map. If like me, all the depth values
    # in the default depth map are within the effective distance, and the 16-bit depth has been manually converted to 8-bit depth. , then alpha can be set to 1.
    depth=cv2.applyColorMap(cv2.convertScaleAbs(depth.cpu().squeeze(0).numpy() ,alpha=1),cv2.COLORMAP_JET)

    return torch.tensor(depth).permute(2, 0, 1)

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def get_2d_emb(batch_size, x, y, out_ch, device):
    out_ch = int(np.ceil(out_ch / 4) * 2) # 8
    inv_freq = 1.0 / (10000 ** (torch.arange(0, out_ch, 2).float() / out_ch)) # (4,)
    pos_x = torch.arange(x, device=device).type(inv_freq.type())*2*np.pi/x # (288,)
    pos_y = torch.arange(y, device=device).type(inv_freq.type())*2*np.pi/y # (512,)
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq) # (288, 4)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq) # (512, 4)
    emb_x = get_emb(sin_inp_x).unsqueeze(1) # 288, 1, 8
    emb_y = get_emb(sin_inp_y) # 512, 8
    emb = torch.zeros((x, y, out_ch * 2), device=device) # 288, 512, 16
    emb[:, :, : out_ch] = emb_x
    emb[:, :, out_ch : 2 * out_ch] = emb_y
    return emb[None, :, :, :].repeat(batch_size, 1, 1, 1) # 1, 288, 512, 16

def knn(x: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = x.cpu().numpy()
    knn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x)
    distances, indices = knn_model.kneighbors(x)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

def positional_encoding(mask, num_frequencies=8):
    mask = torch.tensor(mask, dtype=torch.float32)  # (512, 512, 1)
    freqs = torch.linspace(1.0, 10.0, num_frequencies)  
    encodings = [mask]  # remain mask
    for f in freqs:
        encodings.append(torch.sin(mask * f * torch.pi))  # sin 
        encodings.append(torch.cos(mask * f * torch.pi))  # cos 
    return torch.stack(encodings, dim=-1)  # (512, 512, 16)

# mask = torch.rand(512, 512, 1)  # 生成随机 mask
# encoded_mask = positional_encoding(mask)  # 输出形状 (512, 512, 16)
