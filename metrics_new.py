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

from pathlib import Path
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '7' # MARK: GPU
from PIL import Image
import torch
from math import exp
import torchvision.transforms.functional as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import lpips
import json
import models
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser
from skimage.metrics import structural_similarity

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

def readImages(render_files, gt_files, mask_files):
    renders = []
    gts = []
    masks = None
    image_names = []
    for file in render_files:
        render = Image.open(file)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(os.path.basename(file))
    for file in gt_files:
        gt = Image.open(file)
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
    if mask_files is not None:
        masks = []
        for file in mask_files:
            mask = Image.open(file)
            masks.append(tf.to_tensor(mask).unsqueeze(0)[:, :1, :, :].cuda())
    return renders, gts, masks, image_names


def evaluate(render_files, gt_files, mask_files, model, save_path):
    full_dict = {}
    per_view_dict = {}

    renders, gts, masks, image_names = readImages(render_files, gt_files, mask_files) # image_num, 1, 3, 400, 940

    ssims = []
    psnrs = []
    lpipss = []


    if masks is not None: # use mask
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):

            psnrs.append(psnr(renders[idx], gts[idx], masks[idx]))

            use_structural_similarity = False
            if use_structural_similarity:
                ssims.append(structural_similarity(gts[idx].cpu().numpy(), renders[idx].cpu().numpy(), 
                                                channel_axis=-1, data_range=1))
            else:
                ssims.append(ssim(renders[idx], gts[idx]))

            use_alex = True
            if use_alex:
                with torch.no_grad():
                    lpips_value = model.forward(renders[idx], gts[idx], masks[idx])
                    lpipss.append(lpips_value.item())
            else:
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
    else: # not use mask
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):

            psnrs.append(psnr(renders[idx], gts[idx]))

            use_structural_similarity = False
            if use_structural_similarity:
                ssims.append(structural_similarity(gts[idx].cpu().numpy(), renders[idx].cpu().numpy(), 
                                                channel_axis=-1, data_range=1))
            else:
                ssims.append(ssim(renders[idx], gts[idx]))


            use_alex = True
            if use_alex:
                with torch.no_grad():
                    lpips_value = model.forward(renders[idx], gts[idx])
                    lpipss.append(lpips_value.item())
            else:
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item()})
    per_view_dict.update(
        {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

    with open(os.path.join(save_path, "results.json"), 'w') as fp:
        json.dump(full_dict, fp, indent=True)        
    with open(os.path.join(save_path, "per_view.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)   

    return torch.tensor(psnrs).mean().item(), torch.tensor(lpipss).mean().item(), torch.tensor(ssims).mean().item()


def calculate_average_metrics(json_file):

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f) # read dict

    total_scenes = 0
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_lpips = 0.0

    for scene_name, metrics in data.items():
        if not isinstance(metrics, dict):
            continue  
        if scene_name == "Average":
            continue  

        sum_psnr += metrics.get("PSNR", 0)
        sum_ssim += metrics.get("SSIM", 0)
        sum_lpips += metrics.get("LPIPS", 0)
        total_scenes += 1

    avg_psnr = sum_psnr / total_scenes
    avg_ssim = sum_ssim / total_scenes
    avg_lpips = sum_lpips / total_scenes

    data.update({"Average":{
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips,  
    }}) # update dict

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)  # write dict

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    ###--------- LPIPS ---------###
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    with torch.no_grad():
        model = models.PerceptualLoss(model='net-lin',net='alex', use_gpu=True, version=0.1)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--render_path', '-r', type=str, default='None')
    parser.add_argument('--gt_path', '-g', type=str, default='None')
    parser.add_argument('--mask_path', '-m', type=str, default='None')
    parser.add_argument('--save_path', '-s', type=str, default='None')
    parser.add_argument('--aggregate_path', '-a', type=str, default='None')
    parser.add_argument('--name', '-n', type=str, default='None')
    args = parser.parse_args()

    # select correct file if need
    render_files = sorted(glob(os.path.join(args.render_path, "*.png")) + glob(os.path.join(args.render_path, "*.jpg")))
    gt_files = sorted(glob(os.path.join(args.gt_path, "*.png")) + glob(os.path.join(args.gt_path, "*.jpg")))
    if args.mask_path != 'None':
        mask_files = sorted(glob(os.path.join(args.mask_path, "*.png")) + glob(os.path.join(args.mask_path, "*.jpg")))
    else:
        mask_files = None
    # TODO

    # compute PSNR LPSPS SSIM
    psnr_, lpips_, ssim_ = evaluate(render_files, gt_files, mask_files, model, args.save_path)

    # aggregate results
    try:
        with open(args.aggregate_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # read dict
    except FileNotFoundError:
        print("Create:", args.aggregate_path)
        data = {}

    data.update({args.name:{
        "PSNR": psnr_,
        "SSIM": ssim_,
        "LPIPS": lpips_,  
    }}) # update dict

    with open(args.aggregate_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)  # write dict

    # average aggregate results
    calculate_average_metrics(args.aggregate_path)
