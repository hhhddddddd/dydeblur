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
import math
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

def l1_loss(network_output, gt, mask=None):
    if mask is None:
        return torch.abs((network_output - gt)).mean()
    else:
        return (torch.abs((network_output - gt)) * mask).mean()

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1) # 11, 1; window_size == 11
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # 1, 1, 11, 11
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()) # 3, 1, 11, 11
    return window


def ssim(img1, img2, mask=None, window_size=11, size_average=True):

    # img1 = torch.clamp(img1, min=0, max=1)
    # img2 = torch.clamp(img2, min=0, max=1)

    channel = img1.size(-3) # 3
    window = create_window(window_size, channel) # 3, 1, 11, 11

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel) # window: 3, 1, 11, 11; [out_channels, in_channels/groups, kernel_height, kernel_width]
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2) # 3, 400, 940
    mu2_sq = mu2.pow(2) # 3, 400, 940
    mu1_mu2 = mu1 * mu2 # 3, 400, 940

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq # 3, 400, 940
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq # 3, 400, 940
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2  # 3, 400, 940

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)) # 3, 400, 940

    if size_average:
        if mask is None:
            return ssim_map.mean()
        else:
            num_valid = mask.sum() * img1.shape[1]
            return (ssim_map * mask).sum() / num_valid
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def adaptive_binary_crossentropy(dynamic):
    target = (dynamic > 0.5).float()
    loss = - (target * torch.log(dynamic) + (1 - target) * torch.log(1 - dynamic))
    return loss.mean()

def pearson_depth_loss(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co


def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
        # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
        num_box_h = math.floor(depth_src.shape[0]/box_p)
        num_box_w = math.floor(depth_src.shape[1]/box_p)
        max_h = depth_src.shape[0] - box_p
        max_w = depth_src.shape[1] - box_p
        _loss = torch.tensor(0.0,device='cuda')
        n_corr = int(p_corr * num_box_h * num_box_w)
        x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
        y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
        x_1 = x_0 + box_p
        y_1 = y_0 + box_p
        _loss = torch.tensor(0.0,device='cuda')
        for i in range(len(x_0)):
            _loss += pearson_depth_loss(depth_src[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
        return _loss/n_corr


def tv_loss(grids, mask=None): # grids.shape: batch, channel, height, width
    """
    https://github.com/apchenstu/TensoRF/blob/4ec894dc1341a2201fe13ae428631b58458f105d/utils.py#L139

    Args:
        grids: stacks of explicit feature grids (stacked at dim 0)
    Returns:
        average total variation loss for neighbor rows and columns.
    """
    if mask is None:
        number_of_grids = grids.shape[0] # batch
        h_tv_count = grids[:, :, 1:, :].shape[1] * grids[:, :, 1:, :].shape[2] * grids[:, :, 1:, :].shape[3] # c * (h-1) * w
        w_tv_count = grids[:, :, :, 1:].shape[1] * grids[:, :, :, 1:].shape[2] * grids[:, :, :, 1:].shape[3] # c * h * (w-1)
        h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2).sum() # b, c, h-1, w -> torch.Size([])
        w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2).sum() # b, c, h, w-1 -> torch.Size([])
        return 2 * (h_tv / h_tv_count + w_tv / w_tv_count) / number_of_grids

    else:
        number_of_grids, channel = grids.shape[0], grids.shape[1] # batch, channel
        h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2).view(number_of_grids, channel, -1) # b, c, (h-1) * w; 399 * 940 = 375060
        w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2).view(number_of_grids, channel, -1) # b, c, h * (w-1); 400 * 939 = 375600
        h_mask = mask[:, :, :-1, :].clone().view(number_of_grids, 1, -1) # b, 1, (h-1) * w
        w_mask = mask[:, :, :, :-1].clone().view(number_of_grids, 1, -1) # b, 1, h * (w-1)
        h_tv_masks = torch.tensor([], device="cuda")
        w_tv_masks = torch.tensor([], device="cuda")
        h_dynamics = 0
        w_dynamics = 0
        for i in range(number_of_grids):
            h_dynamics += h_mask[i].sum()
            w_dynamics += w_mask[i].sum()
            h_tv_mask = h_tv[i] * h_mask[i] # c, num_h_mask; 399 * 940 = 375060
            w_tv_mask = w_tv[i] * w_mask[i] # c, num_w_mask; 400 * 939 = 375600
            h_tv_masks = torch.cat((h_tv_masks, h_tv_mask), dim=1) # c, num_h_masks
            w_tv_masks = torch.cat((w_tv_masks, w_tv_mask), dim=1) # c, num_w_masks

        return 2 * ((h_tv_masks.sum() / (3 * h_dynamics)) + (w_tv_masks.sum() / (3 * w_dynamics)))

def align_loss(kernel_weights, kernel, center=1.0): # batch, kernel*kernel, 400, 940
    kernel_center = kernel_weights[:, ((kernel * kernel) // 2), :, :] # batch, 400, 940
    target = (torch.ones(kernel_weights.shape[0], kernel_weights.shape[-2], kernel_weights.shape[-1]) * center.cpu()).cuda() # batch, 400, 940
    alignloss = (target - kernel_center).abs().mean() # batch, 400, 940 -> torch.Size([])
    return alignloss

def align_loss_center(iteration, init=1.0, final=0.5, start_iteration=10000):
    if iteration < start_iteration:
        return init
    else:
        t = np.clip((iteration - start_iteration) / (35000 - start_iteration), 0, 1)
        log_center = np.exp(np.log(init) * (1 - t) + np.log(final) * t)
        return log_center
    
def project_2d_tracks(tracks_3d_w, Ks, T_cw, return_depth=False):
    """
    :param tracks_3d_w (torch.Tensor): (T, N, 3)
    :param Ks (torch.Tensor): (T, 3, 3)
    :param T_cw (torch.Tensor): (T, 4, 4)
    :returns tracks_2d (torch.Tensor): (T, N, 2)
    """
    tracks_3d_c = torch.einsum(
        "tij,tnj->tni", T_cw, F.pad(tracks_3d_w, (0, 1), value=1)
    )[..., :3]
    tracks_3d_v = torch.einsum("tij,tnj->tni", Ks, tracks_3d_c)
    if return_depth:
        return (
            tracks_3d_v[..., :2] / torch.clamp(tracks_3d_v[..., 2:], min=1e-5),
            tracks_3d_v[..., 2],
        )
    return tracks_3d_v[..., :2] / torch.clamp(tracks_3d_v[..., 2:], min=1e-5)

def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss

def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])
        
def compute_z_acc_loss(means_ts_nb: torch.Tensor, w2cs: torch.Tensor):
    """
    :param means_ts (G, 3, B, 3)
    :param w2cs (B, 4, 4)
    return (float)
    """
    camera_center_t = torch.linalg.inv(w2cs)[:, :3, 3]  # (B, 3)
    ray_dir = F.normalize(
        means_ts_nb[:, 1] - camera_center_t, p=2.0, dim=-1
    )  # [G, B, 3]
    # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, B, 3]
    # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    acc_loss = (
        ((means_ts_nb[:, 1] - means_ts_nb[:, 0]) * ray_dir).sum(dim=-1) ** 2
    ).mean() + (
        ((means_ts_nb[:, 2] - means_ts_nb[:, 1]) * ray_dir).sum(dim=-1) ** 2
    ).mean()
    return acc_loss


def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (K, T, 3)
    :param motion_rots (K, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    accel = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    loss = accel.norm(dim=-1).mean()
    return loss

def compute_gradient_loss(pred, gt, mask, quantile=0.98):
    """
    Compute gradient loss
    pred: (batch_size, H, W, D) or (batch_size, H, W)
    gt: (batch_size, H, W, D) or (batch_size, H, W)
    mask: (batch_size, H, W), bool or float
    """
    # NOTE: messy need to be cleaned up
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
    pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
    gt_grad_x = gt[:, :, 1:] - gt[:, :, :-1]
    gt_grad_y = gt[:, 1:, :] - gt[:, :-1, :]
    loss = masked_l1_loss(
        pred_grad_x[mask_x][..., None], gt_grad_x[mask_x][..., None], quantile=quantile
    ) + masked_l1_loss(
        pred_grad_y[mask_y][..., None], gt_grad_y[mask_y][..., None], quantile=quantile
    )
    return loss