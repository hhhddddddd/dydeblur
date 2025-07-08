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

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # MARK: GPU
import torch
from scene import Scene, DeformModel, Blur, MotionBases
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_som
import torchvision
from utils.general_utils import safe_state, get_2d_emb
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time

def render_set(model_path, not_use_dynamic_mask, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, motion_model, blur, kernel=9, train=False):
    if name == 'train': name = "deblur"
    render_path = os.path.join(model_path, "low", name, "ours_{}".format(iteration), "render") # name: train or test
    gts_path = os.path.join(model_path, "low", name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, "low", name, "ours_{}".format(iteration), "depth")
    conv_path = os.path.join(model_path, "low", name, "ours_{}".format(iteration), "conv")
    mask_path = os.path.join(model_path, "low", name, "ours_{}".format(iteration), "mask")
    blur_path = os.path.join(model_path, "low", name, "ours_{}".format(iteration), "blur")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(conv_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)
    makedirs(blur_path, exist_ok=True)

    # sharp_path = os.path.join(model_path, name, "ours_{}".format(iteration), "sharp")
    # makedirs(sharp_path, exist_ok=True)

    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")): # views: TrainCameras or TestCameras
        if load2gpu_on_the_fly:
            view.load2device()
        if motion_model is None:
            fid = view.fid
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling, _ = deform.step(xyz.detach(), time_input)

            results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, train=0)
            rendering, depth, dynamic_mask = results["render"], results["depth"], results["dynamic_mask"]
        else:
            t_start = time.time()
            results = render_som(view, gaussians, pipeline, background, motion_model, train=0)
            torch.cuda.synchronize()
            t_end = time.time()
            t_list.append(t_end - t_start)
            rendering, depth, dynamic_mask = results["render"], results["depth"],  results["dynamic_mask"]
        

        if train:
            shuffle_rgb = rendering.unsqueeze(0) # 1, 3, 400, 940
            shuffle_depth = depth.unsqueeze(0) - depth.min() # 1, 1, 400, 940
            shuffle_depth = shuffle_depth/shuffle_depth.max()
            pos_enc = get_2d_emb(1, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1], 16, torch.device(0)) # 1, 400, 940, 16

            if not_use_dynamic_mask:
                kernel_weights, mask = blur(view.uid, pos_enc, torch.cat([shuffle_rgb,shuffle_depth],1).detach()) # kernel_weights: 1, 81, 400, 940; mask: 1, 1, 400, 940
            else:
                dmask = dynamic_mask.unsqueeze(0).unsqueeze(0) # 1, 1, 400, 940
                kernel_weights, mask = blur(view.uid, pos_enc, torch.cat([shuffle_rgb.detach(),shuffle_depth.detach(),dmask.detach()],1)) # kernel_weights: 1, 81, 400, 940; mask: 1, 1, 400, 940
            
            # kernel_weights, mask = blur(view.uid, pos_enc, torch.cat([shuffle_rgb,shuffle_depth],1).detach()) # kernel_weights: 1, 81, 400, 940; mask: 1, 1, 400, 940
            unfold_ss = torch.nn.Unfold(kernel_size=(kernel, kernel), padding=kernel // 2).cuda()
            patches = unfold_ss(shuffle_rgb) # 1, 9 * 9 * 3, 400 * 940
            patches = patches.view(1, 3, kernel ** 2, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1]) # 1, 3, 81, 400, 940
            kernel_weights = kernel_weights.unsqueeze(1) # 1, 1, 81, 400, 940
            rgb = torch.sum(patches * kernel_weights, 2)[0] # 3, 400, 940
            mask = mask[0] # 1, 400, 940

            blur_final = mask*rgb + (1-mask)*rendering

        depth = depth / (depth.max() + 1e-5) # normalization for visual
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rgb, os.path.join(conv_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mask, os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(blur_final, os.path.join(blur_path, '{0:05d}'.format(idx) + ".png"))
        

        # torchvision.utils.save_image(sharp, os.path.join(sharp_path, '{0:05d}'.format(idx) + ".png"))
    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(name,'fps:',fps)

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, train=False) # MARK: load gaussians
        
        dataset.not_use_dynamic_mask = False # NOTE dmask
        camera_train_num = len(scene.getTrainCameras())
        image_h, image_w = scene.getTrainCameras()[0].original_image.shape[1:]
        print("Number Train Camera = {}, image size: {} x {}".format(camera_train_num, image_h, image_w))
        blur = Blur(camera_train_num, image_h, image_w, ks=9, not_use_rgbd=False, not_use_pe=False, \
             not_use_dynamic_mask=dataset.not_use_dynamic_mask).to("cuda")
            #  not_use_dynamic_mask=True).to("cuda")
        print(dataset.not_use_dynamic_mask)
        blur.load_weights(dataset.model_path, dataset.operate, dataset.time) # MARK: load blurkerel weights

        som = True
        # som = False
        motion_model = None
        deform = None
        if som:
            init_rots = torch.rand(20, camera_train_num, 6)
            init_transls = torch.rand(20, camera_train_num, 3)
            motion_model = MotionBases(init_rots, init_transls)
            # motion_model = MotionBases()
            motion_model.load_weights(dataset.model_path, dataset.operate, dataset.time)
            motion_model = motion_model.to('cuda')
        else:
            deform = DeformModel(dataset.is_blender, dataset.is_6dof)
            deform.load_weights(dataset.model_path, dataset.operate, dataset.time) # MARK: load deformable field weights

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set

        if not skip_train:
            render_func(dataset.model_path, dataset.not_use_dynamic_mask, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform, motion_model, blur, kernel=9, train=True)

        # if not skip_test:
        #     render_func(dataset.model_path, dataset.not_use_dynamic_mask, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
        #                 scene.getTestCameras(), gaussians, pipeline,
        #                 background, deform, motion_model, blur, kernel=9, train=False)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser) # combined command line arguments and config file arguments
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
