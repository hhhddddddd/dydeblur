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
# os.environ["CUDA_VISIBLE_DEVICES"] = '6' # MARK: GPU
os.environ["OMP_NUM_THREADS"] = '4' # MARK: THREAD
# os.environ["export CUDA_LAUNCH_BLOCKING"] = '1'
import sys
import time
import json
import pdbr
import lpips
import models
import torch
import torch.nn as nn
import torch.optim as optim
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss, align_loss, align_loss_center
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, DeformModel, MLP, Blur
from utils.general_utils import safe_state, get_linear_noise_func, get_2d_emb
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

def handle_exception(exc_type, exc_value, exc_traceback):
    # if exc_type is not KeyboardInterrupt:
    print(f"Exception: {exc_value}")  
    pdbr.post_mortem(exc_traceback)
         
sys.excepthook = handle_exception

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True # MARK: switch tb_writer
    # TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, clean_iterations): # lp: dataset,  op: opt,  pp: pipe,
    starttime = time.strftime("%Y-%m-%d_%H:%M:%S") # train start time
    tb_writer = prepare_output_and_logger(dataset, starttime) # TensorBoard
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof, dataset.jdeform_spatial_lr_scale) # DeformModel contains DeformNetwork, scene.cameras_extent
    deform.train_setting(opt) # Initialize the optimizer and the learning rate scheduler
    # dynamic_mlp = MLP(input_ch=3, output_ch=1)
    # dynamic_mlp.train_setting(opt)

    camera_train_num = len(scene.getTrainCameras())
    image_h, image_w = scene.getTrainCameras()[0].original_image.shape[1:]
    print("Number Train Camera = {}, image size: {} x {}".format(camera_train_num, image_h, image_w))
    blur = Blur(camera_train_num, image_h, image_w, ks=dataset.kernel, not_use_rgbd=False, not_use_pe=False, not_use_gt_rgbd=dataset.not_use_gt_rgbd, skip_connect=True).to("cuda") # single scale
    blur.train_setting()
    unfold_ss = torch.nn.Unfold(kernel_size=(dataset.kernel, dataset.kernel), padding=dataset.kernel // 2).cuda()

    gaussians.create_GTnet(hidden=3, width=64, pos_delta=0, num_moments=1)
    gaussians.training_setup(opt) # Initialize the optimizer and the learning rate scheduler
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    with torch.no_grad():
        model = models.PerceptualLoss(model='net-lin',net='alex', use_gpu=True, version=0.1)
    if dataset.source_path.split('/')[-1] == 'dense':
        use_alex = False
    else:
        use_alex = True
    print("use_alex:", use_alex)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)   # record start time
    iter_end = torch.cuda.Event(enable_timing=True)     # record end time

    viewpoint_stack = None
    unseen_viewpoint_stack = None # NOTE
    ema_loss_for_log = 0.0  # exponential moving average loss
    best_psnr_test = 0.0
    best_psnr_train = 0.0
    best_psnr_virtual = 0.0 # NOTE
    best_iteration_test = 0
    best_iteration_train = 0
    best_iteration_virtual = 0 # NOTE
    best_ssim_test = 0.0
    best_lpips_test = 0.0
    progress_bar = tqdm(range(opt.iterations))
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000) # Novelty: learning rate decay function
    loss_texts = []
    gs_texts = []
    print(opt.zmask_loss_alpha, opt.align_loss_alpha, opt.densify_grad_threshold, opt.for_min_opacity, opt.prune_cameras_extent)
    for iteration in range(1, opt.iterations + 1):

        iter_start.record() # record start time

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        if not unseen_viewpoint_stack: # NOTE 
            unseen_viewpoint_stack = scene.getVirtualCameras().copy()

        if (iteration % opt.virtual_interval == 0) and (iteration > opt.warm_up) and (iteration < opt.virtual_max_steps): # NOTE
            total_frame = len(unseen_viewpoint_stack) % camera_train_num  # Monocular Dynamic Scene, current total frame, before pop
            time_interval = 1 / (total_frame + 1)     # time_interval is used for add noise
            viewpoint_cam = unseen_viewpoint_stack.pop(randint(0, len(unseen_viewpoint_stack)-1))
        else:
            total_frame = len(viewpoint_stack)  # Monocular Dynamic Scene, current total frame, before pop
            time_interval = 1 / total_frame     # time_interval is used for add noise
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # len(viewpoint_stack) deincrease
  
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()     # accelerate: move camera data to the GPU

        fid = viewpoint_cam.fid # input time
        
        # opt.warm_up = 1000
        if iteration < opt.warm_up:         # warm_up == 3000; maybe for static region
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]  # current gaussian quantity, eg: 10587
            time_input = fid.unsqueeze(0).expand(N, -1) # Expand the input in the time dimension, eg:torch.Size([10587,1])
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration) # for time_input; why do this?
            d_xyz, d_rotation, d_scaling, _ = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise) # MARK: .detach()

            
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_histogram("scene/d_xyz", d_xyz.abs(), iteration)
                tb_writer.add_histogram("scene/d_rotation", d_rotation.abs(), iteration)
                tb_writer.add_histogram("scene/d_scaling", d_scaling.abs(), iteration)
            
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, train=0, lambda_s=opt.lambda_s, lambda_p=opt.lambda_p, max_clamp=opt.max_clamp)              
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg_re["render"], render_pkg_re["viewspace_points"], \
            render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["depth"]

        maskloss = 0.
        maskloss_sparse = 0.
        depthloss = 0.
        tvloss = 0.
        alignloss = 0.
        if iteration > opt.blur_iteration: # in order to stable train

            shuffle_rgb = image.unsqueeze(0) # 1, 3, 400, 940
            shuffle_depth = depth.unsqueeze(0) - depth.min() # 1, 1, 400, 940
            shuffle_depth = shuffle_depth/shuffle_depth.max()
            pos_enc = get_2d_emb(1, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1], 16, torch.device(0)) # 1, 400, 940, 16

            if dataset.not_use_gt_rgbd:
                kernel_weights, mask = blur(viewpoint_cam.uid, pos_enc, torch.cat([shuffle_rgb,shuffle_depth],1).detach()) # kernel_weights: 1, 81, 400, 940; mask: 1, 1, 400, 940
            else:
                gt_image = viewpoint_cam.original_image.cuda().unsqueeze(0) # 1, 3, 400, 940
                gt_depth = viewpoint_cam.depth.cuda().unsqueeze(0).unsqueeze(0) # 1, 1, 400, 940
                gt_depth = gt_depth - gt_depth.min()
                gt_depth = 1 - (gt_depth / gt_depth.max())
                kernel_weights, mask = blur(viewpoint_cam.uid, pos_enc, torch.cat([shuffle_rgb.detach(),shuffle_depth.detach(),gt_image,gt_depth],1)) # kernel_weights: 1, 81, 400, 940; mask: 1, 1, 400, 940
            patches = unfold_ss(shuffle_rgb) # 1, 9 * 9 * 3, 400 * 940
            patches = patches.view(1, 3, dataset.kernel ** 2, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1]) # 1, 3, 81, 400, 940
            kernel_weights = kernel_weights.unsqueeze(1) # 1, 1, 81, 400, 940
            rgb = torch.sum(patches * kernel_weights, 2)[0] # 3, 400, 940
            mask = mask[0] # 1, 400, 940
            image = mask*rgb + (1-mask)*image
            # image = rgb

            maskloss = opt.zmask_loss_alpha * abs((mask.mean() - 0)) if (opt.use_mask_loss and (iteration > 0)) else 0
            maskloss_sparse = opt.mask_sparse_loss_alpha * torch.cat((mask,1.-mask),dim=0).min(0)[0].mean() if opt.use_mask_sparse_loss else 0
            # center = align_loss_center(iteration, init=1.0, final=0.5) # float
            # center = torch.clamp((1 - mask), 0.5, 1.0) # 1, 400, 940
            center = torch.sigmoid(5. * (1 - mask.detach())) if (opt.use_align_loss and iteration > opt.align_loss_iteration) else 0 # 1, 400, 940
            alignloss = opt.align_loss_alpha * align_loss(kernel_weights.squeeze(0), dataset.kernel, center) if (opt.use_align_loss and iteration > opt.align_loss_iteration) else 0

            depthloss = opt.depth_loss_alpha * tv_loss(shuffle_depth) if opt.use_depth_loss else 0
            # dynamic_tvloss = False
            dynamic_tvloss = True
            if dynamic_tvloss:
                motion_mask = viewpoint_cam.motion_mask.cuda()[None, None] # 1, 1, 400, 940
                tvloss = opt.rgbtv_loss_alpha * tv_loss(shuffle_rgb, motion_mask) if opt.use_rgbtv_loss else 0
            else:
                tvloss = opt.rgbtv_loss_alpha * tv_loss(shuffle_rgb) if opt.use_rgbtv_loss else 0

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.depth.cuda()
        if not viewpoint_cam.is_virtual:
            # blur_map = viewpoint_cam.blur_map.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + maskloss + depthloss + tvloss + alignloss + maskloss_sparse
        else:
            # blur_map = viewpoint_cam.blur_map.cuda()
            unseen_v_Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.virtual_lambda_dssim) * unseen_v_Ll1 + opt.virtual_lambda_dssim * (1.0 - ssim(image, gt_image)) + maskloss + depthloss + tvloss + alignloss + maskloss_sparse

        loss.backward() # retain_graph=True
        iter_end.record() # record end time

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')    # restore: move camera data to the CPU

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log # MARK: ema_loss
            if iteration % 10 == 0:
                postfix_info = {
                    "L": f"{ema_loss_for_log:.{4}f}",
                    "G": f"{gaussians.get_xyz.shape[0]}",
                    "S": f"{dataset.source_path.split('/')[-2] if dataset.source_path.split('/')[-1] == 'dense' else dataset.source_path.split('/')[-1]}",
                    "E": f"{dataset.experiment}"
                }
                progress_bar.set_postfix(postfix_info)
                progress_bar.update(10)
            if iteration == opt.iterations: # Termination iteration
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr_test, cur_psnr_train, cur_psnr_virtual, cur_ssim_test, cur_lpips_test = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, blur, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof, dataset, opt, lpips_fn, model, use_model=use_alex)
            if iteration in testing_iterations: # best_psnr_test only aims to testing_iterations
                if cur_psnr_test.item() > best_psnr_test:
                    best_psnr_test = cur_psnr_test.item()
                    best_ssim_test = cur_ssim_test.item()
                    best_lpips_test = cur_lpips_test.item()
                    best_iteration_test = iteration
                    scene.save(iteration, starttime, dataset.operate, True)                                   # save 3d scene (point cloud)
                    deform.save_weights(args.model_path, iteration, starttime, dataset.operate, True)         # save deformable filed
                    blur.save_weights(args.model_path, iteration, starttime, dataset.operate, True)           # save blurkerel
                if cur_psnr_train.item() > best_psnr_train:
                    best_psnr_train = cur_psnr_train.item()
                    best_iteration_train = iteration
                if cur_psnr_virtual.item() > best_psnr_virtual:
                    best_psnr_virtual = cur_psnr_virtual.item()
                    best_iteration_virtual = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, starttime, dataset.operate)                                   # save 3d scene (point cloud)
                deform.save_weights(args.model_path, iteration, starttime, dataset.operate)         # save deformable filed
                # blur.save_weights(args.model_path, iteration, starttime, dataset.operate)       # save blur filed

            # Densification
            if (iteration < opt.densify_until_iter): # and (iteration < 6000)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 10e9 if iteration > opt.opacity_reset_interval else None # set size_threshold
                    gs_num = gaussians.densify_and_prune(opt.densify_grad_threshold, opt.for_min_opacity, opt.prune_cameras_extent, size_threshold, iteration) # 0.005, scene.cameras_extent
                    gs_num['iteration'] = iteration
                    gs_texts.append(gs_num)
                    # print(iteration, gaussians.get_xyz.shape[0])

                if iteration > opt.opacity_reset_interval and gaussians.get_xyz.shape[0] < 1000:
                    gaussians.densify_from_pcd(dataset.source_path)
                    # print(iteration, gaussians.get_xyz.shape[0])
                    
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                # if iteration in [3000, 10000]:
                    gaussians.reset_opacity()

                # if (iteration - 1) % 25 == 0: # Near plane point clipping
                #     mask_near = None
                #     if iteration > 2000:
                #         for idx, view in enumerate(scene.getTrainCameras().copy()):
                #             mask_temp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True) < 5.
                #             mask_near = mask_near + mask_temp if mask_near is not None else mask_temp
                #         gaussians.prune_points(mask_near.squeeze())
                
                # clean views
                # clean_views(iteration, clean_iterations, scene, gaussians, pipe, background, deform, dataset.is_6dof)

            # Optimizer step
            if iteration < opt.iterations: # Termination iteration
                if iteration > opt.blur_iteration:
                    blur.optimizer.step()
                    blur.update_learning_rate(iteration)
                    blur.optimizer.zero_grad()
                
                gaussians.optimizer.step()
                deform.optimizer.step()
                
                gaussians.update_learning_rate(iteration)
                deform.update_learning_rate(iteration)
                # gaussians.update_learning_rate(max(iteration - opt.position_lr_start, 0))

                gaussians.optimizer.zero_grad(set_to_none=True) # clear gradient, every iteration clear gradient
                deform.optimizer.zero_grad() # deform also needs to clear gradient

    scene_name = dataset.model_path.split("/")[-1]
    print("{} : {}".format(scene_name, dataset.experiment))
    print("{} Best PSNR = {:.5f} in Iteration {}, gs_num = {}".format(scene_name, best_psnr_test, best_iteration_test, gaussians.get_xyz.shape[0]))
    print("{} Best SSIM = {:.5f} in Iteration {}, gs_num = {}".format(scene_name, best_ssim_test, best_iteration_test, gaussians.get_xyz.shape[0]))
    print("{} Best LPIPS = {:.5f} in Iteration {}, gs_num = {}".format(scene_name, best_lpips_test, best_iteration_test, gaussians.get_xyz.shape[0]))
    print("{} Best PSNR = {:.5f} in Iteration {}, gs_num = {}".format(scene_name, best_psnr_train, best_iteration_train, gaussians.get_xyz.shape[0]))
    print("{} Best PSNR = {:.5f} in Iteration {}, gs_num = {}".format(scene_name, best_psnr_virtual, best_iteration_virtual, gaussians.get_xyz.shape[0]))
    
    gs_num_path = os.path.join(dataset.model_path, dataset.operate, starttime[:16],'gs_num.json')
    with open(gs_num_path, "w") as file:
        json.dump(gs_texts, file, indent=4)
    loss_texts_path = os.path.join(dataset.model_path, dataset.operate, starttime[:16],'loss.json')
    with open(loss_texts_path, "w") as file:
        json.dump(loss_texts, file, indent=4)

    result = {}
    result['scene'], result['experiment'] = scene_name, starttime[5:16] + '_' + dataset.experiment
    result['PSNR'], result['SSIM'], result['LPIPS'] = best_psnr_test, best_ssim_test, best_lpips_test
    result['Iteration'], result['gs_num'] = best_iteration_test, gaussians.get_xyz.shape[0]
    current_file_path = os.path.abspath(__file__)
    results_path = os.path.join(current_file_path.split('/train')[0] , "output/dydeblur/results.json")
    with open(results_path, 'r+') as file:
        results = json.load(file)
        results.append(result)
        file.seek(0)
        json.dump(results, file, indent=4)


def prepare_output_and_logger(args, starttime): # TensorBoard writer
    if not args.model_path: # If no 'args.model_path' is provided
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path)) # 'output/dydeblur/D_NeRF/trex'
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f: # output; MARK: cfg_args
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        # tb_writer = SummaryWriter(os.path.join(args.model_path, "tb_logs", args.operate, starttime[:16]), comment=starttime[:13], flush_secs=60)
        tb_writer = SummaryWriter(os.path.join("output/dydeblur/tb_log", args.model_path.split('/')[-1], starttime[5:16]+'_'+args.experiment), comment=starttime[:13], flush_secs=60)
        tb_writer.add_text('d-3dgs', args.experiment)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def clean_views(iteration, clean_iterations, scene, gaussians, pipe, background, deform, is_6dof=False):
    if iteration in clean_iterations:
        visible_pnts = None
        for viewpoint_cam in scene.getTrainCameras().copy():
            fid = viewpoint_cam.fid
            xyz = scene.gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling, _ = deform.step(xyz.detach(), time_input)

            render_pkg = render(viewpoint_cam, scene.gaussians, pipe, background, d_xyz, d_rotation, d_scaling, is_6dof, train=0, lambda_s=0.01, lambda_p=0.01, max_clamp=1.1)   
            visibility_filter = render_pkg["visibility_filter"]
            if visible_pnts is None:
                visible_pnts = visibility_filter
            visible_pnts += visibility_filter
        unvisible_pnts = ~visible_pnts
        gaussians.prune_points(unvisible_pnts)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, blurFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, dataset=None, opt=None, lpips_fn=None, model=None, use_model=True): # renderFunc == render
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    test_ssim = 0.0
    test_lpips = 0.0
    train_psnr = 0.0
    virtual_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache() # clearing GPU memory
        validation_configs = ({'name': 'test', 'cameras': scene.getSortedTestCameras()}, # n
                              {'name': 'train','cameras': scene.getSortedTrainCameras()}, # n
                              {'name': 'virtual', 'cameras': scene.getSortedVirtualCameras()}) # NOTE 4n

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                gts = torch.tensor([], device="cuda")
                motion_masks = torch.tensor([], device="cuda")
                shuffle_rgbs = torch.tensor([], device="cuda")
                images = torch.tensor([], device="cuda")
                depths = torch.tensor([], device="cuda")
                masks = torch.tensor([], device="cuda")
                alignlosss = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    torch.cuda.empty_cache() # clearing GPU memory
                    if load2gpu_on_the_fly:
                        viewpoint.load2device() # accelerate: move camera data to the GPU
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling, _ = deform.step(xyz.detach(), time_input)

                    if config["name"] == 'train' or config["name"] == 'virtual': # NOTE
                        gt_depth = viewpoint.depth
                        gt_depth = gt_depth - gt_depth.min()
                        gt_depth = gt_depth / gt_depth.max()
                        gt_depth = 1 - gt_depth # reverse
                        motion_mask = viewpoint.motion_mask[None] # 1, 400, 940

                        ret = renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof, train=0, lambda_s=0.01, lambda_p=0.01, max_clamp=1.1)              
                        image, sharp_image, depth = ret["render"], ret["render"], ret["depth"]

                        if iteration > opt.blur_iteration: # blur process
                            shuffle_rgb = sharp_image.unsqueeze(0) # 1, 3, 400, 940
                            shuffle_depth = depth.unsqueeze(0) - depth.min()
                            shuffle_depth = shuffle_depth/shuffle_depth.max()
                            unfold_ss = torch.nn.Unfold(kernel_size=(dataset.kernel, dataset.kernel), padding=dataset.kernel // 2).cuda()
                            pos_enc = get_2d_emb(1, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1], 16, torch.device(0))     

                            if dataset.not_use_gt_rgbd:
                                kernel_weights, mask = blurFunc(viewpoint.uid, pos_enc, torch.cat([shuffle_rgb,shuffle_depth],1).detach()) # kernel_weights: 1, 81, 400, 940; mask: 1, 1, 400, 940
                            else:
                                gt_image = viewpoint.original_image.cuda().unsqueeze(0) # 1, 3, 400, 940
                                gt_depth_unsqueeze = gt_depth.unsqueeze(0).unsqueeze(0) # 1, 1, 400, 940
                                kernel_weights, mask = blurFunc(viewpoint.uid, pos_enc, torch.cat([shuffle_rgb.detach(),shuffle_depth.detach(),gt_image,gt_depth_unsqueeze],1)) # kernel_weights: 1, 81, 400, 940; mask: 1, 1, 400, 940

                            patches = unfold_ss(shuffle_rgb)
                            patches = patches.view(1, 3, dataset.kernel ** 2, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1])
                            kernel_weights = kernel_weights.unsqueeze(1) # 1, 1, 81, 400, 940
                            rgb = torch.sum(patches * kernel_weights, 2)[0]
                            mask = mask[0] # 1, 400, 940
                            image = mask*rgb + (1-mask)*sharp_image
                            # image = rgb

                            masks = torch.cat((masks, mask.mean().unsqueeze(0)), dim=0) # mask; 1, 1
                            motion_masks = torch.cat((motion_masks, motion_mask.unsqueeze(0)), dim=0) # motion_mask; 1, 1, 400, 940
                            shuffle_rgbs = torch.cat((shuffle_rgbs, shuffle_rgb), dim=0) # shuffle_rgb; 1, 3, 400, 940
                            
                            # center = align_loss_center(iteration, init=1.0, final=0.5)
                            # center = torch.clamp((1 - mask), 0.5, 1.0)
                            center = torch.sigmoid(5. * (1 - mask.detach()))
                            alignloss = align_loss(kernel_weights.squeeze(0), dataset.kernel, center) # FIXME center
                            alignlosss = torch.cat((alignlosss, alignloss.unsqueeze(0)), dim=0) # alignloss; 1, 1

                    else:
                        ret = renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof, train=0, lambda_s=0.01, lambda_p=0.01, max_clamp=1.1)              
                        image, depth = ret["render"], ret["depth"]

                    image = torch.clamp(image, 0.0, 1.0)
                    depth = depth - depth.min()
                    depth = depth / depth.max()

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)    # gt image; 1, 3, 400, 940                    
                    images = torch.cat((images, image.unsqueeze(0)), dim=0) # target image; 1, 3, 400, 940
                    depths = torch.cat((depths, depth.unsqueeze(0)), dim=0) # depth; 1, 1, 400, 940

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu') # restore: move camera data to the CPU

                    if tb_writer and (idx % 6 == 0): # Show only the first 5 images
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration, dataformats='NCHW')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                                             depth, global_step=iteration, dataformats='CHW')
                        if config["name"] == 'train'or config["name"] == 'virtual': # NOTE
                            tb_writer.add_images(config['name'] + "_view_{}/sharp".format(viewpoint.image_name),
                                             sharp_image[None], global_step=iteration)
                            if iteration > opt.blur_iteration:
                                tb_writer.add_images(config['name'] + "_view_{}/blur_map".format(viewpoint.image_name),
                                                mask, global_step=iteration, dataformats='CHW')
                                                   
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                            if config["name"] == 'train'or config["name"] == 'virtual': # NOTE
                                tb_writer.add_images(config['name'] + "_view_{}/gt_depth".format(viewpoint.image_name),
                                                    gt_depth, global_step=iteration, dataformats='HW')
                                # tb_writer.add_images(config['name'] + "_view_{}/motion_mask".format(viewpoint.image_name),
                                #                     motion_mask, global_step=iteration, dataformats='CHW')
                
                torch.cuda.empty_cache() # clearing GPU memory
                l1_test = l1_loss(images, gts) # batch, channel, height, width -> torch.Size([])
                psnr_test = psnr(images, gts).mean() # batch, 1 -> torch.Size([]) TODO SSIM, LPIPS
                depth_tv_loss = opt.depth_loss_alpha * tv_loss(depths) if opt.use_depth_loss else 0
                
                if config['name'] == 'test':
                    test_psnr = psnr_test

                    ssims = []
                    lpipss = []
                    for idx in tqdm(range(len(images)), desc="Metric evaluation progress"):
                        torch.cuda.empty_cache() # clearing GPU memory
                        ssims.append(ssim(images[idx].unsqueeze(0), gts[idx].unsqueeze(0))) # 1, 3, 400, 940
                        if use_model:
                            with torch.no_grad():
                                lpips_value = model.forward(images[idx].unsqueeze(0), gts[idx].unsqueeze(0))
                                lpipss.append(lpips_value.item())
                        else:
                            lpipss.append(lpips_fn(images[idx].unsqueeze(0), gts[idx].unsqueeze(0)).detach())
                    test_ssim = torch.tensor(ssims).mean()
                    test_lpips = torch.tensor(lpipss).mean()

                elif config['name'] == 'train':
                    train_psnr = psnr_test
                    maskloss = opt.zmask_loss_alpha * abs((mask.mean() - 0)) if (opt.use_mask_loss and (iteration > 0)) else 0 
                    maskloss += opt.mask_sparse_loss_alpha * torch.cat((mask,1.-mask),dim=0).min(0)[0].mean() if opt.use_mask_sparse_loss else 0
                    alignloss = opt.align_loss_alpha * alignlosss.mean() if (opt.use_align_loss and iteration > opt.align_loss_iteration) else 0
                    # dynamic_tvloss = False
                    dynamic_tvloss = True
                    if dynamic_tvloss:
                        tvloss = opt.rgbtv_loss_alpha * tv_loss(shuffle_rgbs, motion_masks) if opt.use_rgbtv_loss else 0
                    else:
                        tvloss = opt.rgbtv_loss_alpha * tv_loss(shuffle_rgbs) if opt.use_rgbtv_loss else 0
                else:
                    virtual_psnr = psnr_test
                    maskloss = opt.zmask_loss_alpha * abs((mask.mean() - 0)) if (opt.use_mask_loss and (iteration > 0)) else 0 
                    maskloss += opt.mask_sparse_loss_alpha * torch.cat((mask,1.-mask),dim=0).min(0)[0].mean() if opt.use_mask_sparse_loss else 0
                    alignloss = opt.align_loss_alpha * alignlosss.mean() if (opt.use_align_loss and iteration > opt.align_loss_iteration) else 0 
                    # dynamic_tvloss = False
                    dynamic_tvloss = True
                    if dynamic_tvloss:
                        tvloss = opt.rgbtv_loss_alpha * tv_loss(shuffle_rgbs, motion_masks) if opt.use_rgbtv_loss else 0
                    else:
                        tvloss = opt.rgbtv_loss_alpha * tv_loss(shuffle_rgbs) if opt.use_rgbtv_loss else 0
                
                if config['name'] == 'test':
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, test_ssim, test_lpips)) # MARK: console output
                else:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test)) # MARK: console output
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - depth_tv_loss', depth_tv_loss, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if config["name"] == 'train'or config["name"] == 'virtual':
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - mask_loss', maskloss, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - align_loss', alignloss, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - rgbtv_loss', tvloss, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/dynamic_histogram", scene.gaussians.get_dynamic, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache() # clearing GPU memory

    return test_psnr, train_psnr, virtual_psnr, test_ssim, test_lpips


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters") # ArgumentParser: Object for parsing command line strings into Python objects.
    lp = ModelParams(parser)            # MARK: model
    op = OptimizationParams(parser)     # MARK: optimization
    pp = PipelineParams(parser)         # MARK: pipeline
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[3600, 4000, 5000, 6000] + list(range(7000, 40001, 1000)))
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20000, 40000])
    parser.add_argument("--clean_iterations", nargs="+", type=int, default=list(range(10000, 25001, 5000)))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])          # Namespace
    args.save_iterations.append(args.iterations)    # save the last iteration

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet) # random seed

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.clean_iterations) # lp: dataset,  op: opt,  pp: pipe,

    # All done
    print("\nTraining complete.")

'''
train: deform, point_cloud, cameras.json, cfg_args, input.ply
render: train, test
metrics: results.json, per_view.json
'''
