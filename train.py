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
os.environ["CUDA_VISIBLE_DEVICES"] = '2' # MARK: GPU
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence # what is kl_divergence?
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True # MARK: switch tb_writer
    # TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations): # lp: dataset,  op: opt,  pp: pipe,
    tb_writer = prepare_output_and_logger(dataset) # TensorBoard
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof) # DeformModel contains DeformNetwork
    deform.train_setting(opt) # Initialize the optimizer and the learning rate scheduler

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt) # Initialize the optimizer and the learning rate scheduler

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)   # record start time
    iter_end = torch.cuda.Event(enable_timing=True)     # record end time

    viewpoint_stack = None
    ema_loss_for_log = 0.0  # exponential moving average loss
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000) # Novelty: learning rate decay function
    for iteration in range(1, opt.iterations + 1):
        # print('gs_num:',gaussians.get_xyz.shape[0])
        if network_gui.conn == None: # I guess visualization? GUI
            network_gui.try_connect()
        while network_gui.conn != None: # I guess visualization? GUI
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record() # record start time

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)  # Monocular Dynamic Scene, current total frame
        time_interval = 1 / total_frame     # time_interval is used for add noise

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))   # len(viewpoint_stack) deincrease
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()     # accelerate: move camera data to the GPU
        total_frame = len(viewpoint_stack)  # Monocular Dynamic Scene, current total frame
        fid = viewpoint_cam.fid             # input time
        
        if iteration < opt.warm_up:         # warm_up: maybe for static region
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]  # current gaussian quantity, eg: 10587
            # print(N)
            time_input = fid.unsqueeze(0).expand(N, -1) # Expand the input in the time dimension, eg:torch.Size([10587,1])

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration) # for time_input; why do this?
            d_xyz, d_rotation, d_scaling, _ = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise) # MARK: .detach()
            dynamic = torch.sum(d_xyz.detach().abs(), dim=1) # add
            dynamic = (dynamic - dynamic.mean()) * 3.0 / dynamic.std() # Z-Score normalize, scale-factor
            # dynamic = (dynamic - dynamic.min()) / (dynamic.max() - dynamic.min()) # Max-min normalize
            gaussians._dynamic = 0.4 * gaussians._dynamic.detach() + 0.6 * dynamic[..., None]
            # gaussians._dynamic = dynamic[..., None]

        # Render
        render_pkg_re_static = render(viewpoint_cam, gaussians, pipe, background, 0, 0, 0, dataset.is_6dof)                 # static
        image_static, dynamic_mask_static = render_pkg_re_static["render"], render_pkg_re_static["dynamic"] # 3, 800, 800
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)   # dynamic
        image_dynamic, viewspace_point_tensor, visibility_filter, radii, dynamic_mask = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["dynamic"]
        # dynamic_mask, _ = torch.max(torch.stack((dynamic_mask_static, dynamic_mask)), dim=0) # Combine static and dynamic 
        dynamic_mask = torch.clamp(dynamic_mask, 0.0, 1.0)
        image = dynamic_mask * image_dynamic + (1 - dynamic_mask) * image_static                                            # blend

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        Ll1_dynamic = l1_loss(image_dynamic, gt_image)
        # Ll1_static = l1_loss((1 - dynamic_mask.detach()) * image_static , (1 - dynamic_mask.detach()) * gt_image)
        loss = (1.0 - opt.lambda_dssim) * (Ll1 + Ll1_dynamic) + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # opt.lambda_dssim == 0.2
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()   # record end time

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')    # restore: move camera data to the CPU

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log # MARK: ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations: # Termination iteration
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations: # best_psnr only aims to testing_iterations
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)                               # save 3d scene (point cloud)
                deform.save_weights(args.model_path, iteration)     # save deformable filed

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # set size_threshold
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations: # Termination iteration
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True) # clear gradient, every iteration clear gradient
                deform.optimizer.zero_grad() # deform also needs to clear gradient
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args): # TensorBoard writer
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
        starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
        tb_writer = SummaryWriter(os.path.join(args.model_path, "tb_logs", args.operate, starttime[:16]), comment=starttime[:13], flush_secs=60)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False): # renderFunc == render
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache() # clearing GPU memory
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device() # accelerate: move camera data to the GPU
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling, dynamic = deform.step(xyz.detach(), time_input)

                    ret_static = renderFunc(viewpoint, scene.gaussians, *renderArgs, 0, 0, 0, is_6dof)
                    ret = renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)
                    image_static = torch.clamp(ret_static["render"],0.0, 1.0)
                    image_dynamic = torch.clamp(ret["render"],0.0, 1.0)
                    dynamic_mask = ret["dynamic"]
                    image = dynamic_mask * image_dynamic + (1 - dynamic_mask) * image_static

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)         

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu') # restore: move camera data to the CPU
                    if tb_writer and (idx < 5): # Show only the first 5 images
                        tb_writer.add_images(config['name'] + "_view_{}/dynamic".format(viewpoint.image_name),
                                             image_dynamic[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/dynamic_mask".format(viewpoint.image_name),
                                             dynamic_mask, global_step=iteration, dataformats='HW') 
                        tb_writer.add_images(config['name'] + "_view_{}/static".format(viewpoint.image_name),
                                             image_static[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/blend".format(viewpoint.image_name),
                                             image[None], global_step=iteration) 
                       
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test)) # MARK: console output
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/dynamic_histogram", scene.gaussians.get_dynamic, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache() # clearing GPU memory

    return test_psnr


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
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])          # Namespace
    args.save_iterations.append(args.iterations)    # save the last iteration

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations) # lp: dataset,  op: opt,  pp: pipe,

    # All done
    print("\nTraining complete.")

'''
train: deform, point_cloud, cameras.json, cfg_args, input.ply
render: train, test
metrics: results.json, per_view.json
'''
