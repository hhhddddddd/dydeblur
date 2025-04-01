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
import random
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.motion_model import MotionBases
from scene.mlp_model import MLP
from scene.blur_model import Blur
from scene.cameras import Camera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import getWorld2View, focal2fov
from utils.pose_utils import viewmatrix, posevisual
from utils.loss_utils import project_2d_tracks, masked_l1_loss, compute_se3_smoothness_loss, compute_accel_loss, compute_z_acc_loss

from simple_lama_inpainting import SimpleLama
from utils.warp_utils import Warper

class Scene:
    gaussians: GaussianModel # class attribute

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], train=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:  # initialization self.loaded_iter, according to load_iteration
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, args.operate, args.time, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.virtual_cameras = {} # unseen view virtual cams

        self.sorted_train_cameras = {}
        self.sorted_test_cameras = {}
        self.sorted_virtual_cameras = {}
        print(args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")): 
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval) # Colmap

        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval) # Blender; D_NeRF

        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz") # DTU

        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval) # nerfies, hyper-nerf, nerf-ds

        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path) # Dynamic-360

        elif os.path.exists(os.path.join(args.source_path, "background_mask")):
            print("Found background_mask, assuming D2RF data set!")
            scene_info = sceneLoadTypeCallbacks["D2RF"](args.source_path, args.camera_scale) # D2RF

        elif os.path.exists(os.path.join(args.source_path, "sharp_images")):
            print("Found sharp_images, assuming DyBluRF data set!")
            scene_info = sceneLoadTypeCallbacks["DyBluRF"](args.source_path, args.camera_scale) # DyBluRF

        elif os.path.exists(os.path.join(args.source_path, "flow3d_preprocessed")):
            print("Found flow3d_preprocessed, assuming Deblur4DGS data set!")
        #     scene_info = sceneLoadTypeCallbacks["Deblur4DGS"](args.source_path, args.camera_scale) # Deblur4DGS

        # elif os.path.exists(os.path.join(args.source_path, "DyDeblur")):
        #     print("DyDeblur, assuming DyDeblur data set!")
            scene_info = sceneLoadTypeCallbacks["DyDeblur"](args.source_path, args.camera_scale) # DyDeblur

        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24) # Neu3D, DyNeRF
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter: # Output "input.ply" and "cameras.json"
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file: # MARK: input.ply
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras_sc1.json"), 'w') as file: # MARK: cameras.json
                json.dump(json_cams, file, indent=True)


        self.cameras_extent = scene_info.nerf_normalization["radius"] # cameras_extent: the max distance between mean_camera_center and camera 
        print('Real Cameras Extent =', self.cameras_extent)
        for resolution_scale in resolution_scales:  # CameraInfo -> Camera; MARK: resize image
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
            # get unseen cameras
            if train:
                print("Generating Virtual Cameras")
                self.virtual_cameras[resolution_scale] = self.generateVirtualCams(self.train_cameras[resolution_scale], self.test_cameras[resolution_scale], scene_info.sc)
            torch.cuda.empty_cache()

        # shuffle train camera and test camera
        if shuffle: 
            for resolution_scale in resolution_scales:
                self.sorted_train_cameras[resolution_scale] = self.train_cameras[resolution_scale].copy()
                self.sorted_test_cameras[resolution_scale] = self.test_cameras[resolution_scale].copy()
                if train:
                    self.sorted_virtual_cameras[resolution_scale] = self.virtual_cameras[resolution_scale].copy()

                random.shuffle(self.train_cameras[resolution_scale])  # Multi-res consistent random shuffling
                random.shuffle(self.test_cameras[resolution_scale])  # Multi-res consistent random shuffling
                if train:
                    random.shuffle(self.virtual_cameras[resolution_scale])  # Multi-res consistent random shuffling
     
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, args.operate, args.time,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        # elif os.path.exists(os.path.join(args.source_path, "DyDeblur")):
        elif os.path.exists(os.path.join(args.source_path, "flow3d_preprocessed")):
            print("DyDeblur, assuming DyDeblur data set!")
            fg_params, motion_bases, bg_params, tracks_3d = self.gaussians.depth_track_point(scene_info.foreground_points, scene_info.background_points, args.canot)
            self.run_initial_optim(fg_params, motion_bases, tracks_3d, scene_info.Ks, scene_info.w2cs, num_iters=1000) # tracks_3d is gt; 1000
            self.motion_model = motion_bases # deform model
            # TODO add depth foregroud
            # init gaussian
            self.gaussians.create_from_track(fg_params, bg_params, args.gaussian_spatial_lr_scale) # self.cameras_extent
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, args.gaussian_spatial_lr_scale) # self.cameras_extent

    def run_initial_optim(self, fg, bases, tracks_3d, Ks, w2cs, num_iters, use_depth_range_loss=False):

        w2cs = w2cs[::2] # 24, 4, 4
        Ks = Ks[::2] # 24, 3, 3
        optimizer = torch.optim.Adam(
            [
                {"params": bases.params["rots"], "lr": 1e-2},
                {"params": bases.params["transls"], "lr": 3e-2},
                {"params": fg.params["motion_coefs"], "lr": 1e-2},
                {"params": fg.params["means"], "lr": 1e-3},
            ],
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.1 ** (1 / num_iters)
        )
        G = fg.params.means.shape[0] # 7392
        num_frames = bases.num_frames # 24
        device = bases.params["rots"].device

        w_smooth_func = lambda i, min_v, max_v, th: (
            min_v if i <= th else (max_v - min_v) * (i - th) / (num_iters - th) + min_v
        )

        gt_2d, gt_depth = project_2d_tracks(
            tracks_3d.xyz.swapaxes(0, 1), Ks, w2cs, return_depth=True
        )
        # (G, T, 2)
        gt_2d = gt_2d.swapaxes(0, 1)
        # (G, T)
        gt_depth = gt_depth.swapaxes(0, 1)

        ts = torch.arange(0, num_frames, device=device) # [0, 1, ..., 23]
        ts_clamped = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts_clamped - 1, ts_clamped, ts_clamped + 1))  # i (3B,)

        # def get_coefs(self) -> torch.Tensor:
        #     assert "motion_coefs" in self.params
        #     return self.motion_coef_activation(self.params["motion_coefs"])
        # self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)

        get_coefs = lambda x: F.softmax(x, dim=-1)
        
        pbar = tqdm(range(0, num_iters))
        for i in pbar:
            coefs = get_coefs(fg.params["motion_coefs"])
            transfms = bases.compute_transforms(ts, coefs)
            positions = torch.einsum(
                "pnij,pj->pni",
                transfms,
                F.pad(fg.params["means"], (0, 1), value=1.0),
            )

            loss = 0.0
            track_3d_loss = masked_l1_loss(
                positions,
                tracks_3d.xyz.cuda(),
                (tracks_3d.visibles.cuda().float() * tracks_3d.confidences.cuda())[..., None],
            )
            loss += track_3d_loss * 1.0

            pred_2d, pred_depth = project_2d_tracks(
                positions.swapaxes(0, 1), Ks.cuda(), w2cs.cuda(), return_depth=True
            )
            pred_2d = pred_2d.swapaxes(0, 1)
            pred_depth = pred_depth.swapaxes(0, 1)

            loss_2d = (
                masked_l1_loss(
                    pred_2d,
                    gt_2d.cuda(),
                    (tracks_3d.invisibles.cuda().float() * tracks_3d.confidences.cuda())[..., None],
                    quantile=0.95,
                )
                / Ks.cuda()[0, 0, 0]
            )
            loss += 0.5 * loss_2d

            if use_depth_range_loss:
                near_depths = torch.quantile(gt_depth, 0.0, dim=0, keepdim=True)
                far_depths = torch.quantile(gt_depth, 0.98, dim=0, keepdim=True)
                loss_depth_in_range = 0
                if (pred_depth < near_depths).any():
                    loss_depth_in_range += (near_depths - pred_depth)[
                        pred_depth < near_depths
                    ].mean()
                if (pred_depth > far_depths).any():
                    loss_depth_in_range += (pred_depth - far_depths)[
                        pred_depth > far_depths
                    ].mean()

                loss += loss_depth_in_range * w_smooth_func(i, 0.05, 0.5, 400)

            motion_coef_sparse_loss = 1 - (coefs**2).sum(dim=-1).mean()
            loss += motion_coef_sparse_loss * 0.01

            # motion basis should be smooth.
            w_smooth = w_smooth_func(i, 0.01, 0.1, 400)
            small_acc_loss = compute_se3_smoothness_loss(
                bases.params["rots"], bases.params["transls"]
            )
            loss += small_acc_loss * w_smooth

            small_acc_loss_tracks = compute_accel_loss(positions)
            loss += small_acc_loss_tracks * w_smooth * 0.5

            transfms_nbs = bases.compute_transforms(ts_neighbors, coefs)
            means_nbs = torch.einsum(
                "pnij,pj->pni", transfms_nbs, F.pad(fg.params["means"], (0, 1), value=1.0)
            )  # (G, 3n, 3)
            means_nbs = means_nbs.reshape(means_nbs.shape[0], 3, -1, 3)  # [G, 3, n, 3]
            z_accel_loss = compute_z_acc_loss(means_nbs, w2cs.cuda())
            loss += z_accel_loss * 0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(
                f"{loss.item():.3f} "
                f"{track_3d_loss.item():.3f} "
                f"{motion_coef_sparse_loss.item():.3f} "
                f"{small_acc_loss.item():.3f} "
                f"{small_acc_loss_tracks.item():.3f} "
                f"{z_accel_loss.item():.3f} "
            )
    
    def save(self, iteration, starttime, operate, best=False):
        if not best:
            point_cloud_path = os.path.join(self.model_path, operate, starttime[:16], "point_cloud/iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, operate, starttime[:16], "point_cloud/iteration_66666")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_learn(self, iteration, canonical):
        if canonical:
            point_cloud_path = os.path.join(self.model_path, "deformation/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "canonical.ply"))
        else:
            point_cloud_path = os.path.join(self.model_path, "deformation/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "deformation.ply"))            

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getVirtualCameras(self, scale=1.0):
        return self.virtual_cameras[scale]
    
    def getSortedTrainCameras(self, scale=1.0):
        return self.sorted_train_cameras[scale]

    def getSortedTestCameras(self, scale=1.0):
        return self.sorted_test_cameras[scale]
    
    def getSortedVirtualCameras(self, scale=1.0):
        return self.sorted_virtual_cameras[scale]

    def generateVirtualCams(self, input_cams, target_cams, sc, batch_size=4, inpaint=True):
        print('Train view num: ', len(input_cams))

        input_imgs, input_blur_maps, input_motion_masks, mvs_depths, input_extrs, input_intrs, target_extrs, target_intrs = self.prepare_data(input_cams, target_cams, sc) # mvs_depths: absolute depth

        # split into batches
        input_batches = create_batches(input_imgs, input_blur_maps, input_motion_masks, mvs_depths, input_extrs, input_intrs, batch_size=batch_size)
        target_batches = create_batches(target_intrs, target_extrs, batch_size=batch_size)

        warper = Warper()
        warped_frames, valid_masks, warped_depths = [], [], []
        warped_blur_maps, warped_blur_masks = [], []
        warped_motion_maps, warped_motion_masks = [], []
        with torch.no_grad():
            for (input_imgs_batch, input_blur_batch, input_motion_batch, mvs_depths_batch, input_extrs_batch, input_intrs_batch), (target_intrs_batch, target_extrs_batch) \
                in tqdm(zip(input_batches, target_batches), desc="Unseen view prior", unit="batch", total=int(len(input_imgs) / batch_size)):
                torch.cuda.empty_cache()
                masks_batch = None
                # get priors for unseen views by forward warping; MARK: mvs_depths: disparity
                warped_frame, valid_mask, warped_depth, _ = warper.forward_warp(input_imgs_batch, masks_batch, mvs_depths_batch, input_extrs_batch, 
                                                                                target_extrs_batch, input_intrs_batch, target_intrs_batch)
                warped_frames.append(warped_frame.cpu()) # batch_size, 3, 288, 512
                valid_masks.append(valid_mask.cpu()) # batch_size, 1, 288, 512
                warped_depths.append(warped_depth.cpu()) # batch_size, 1, 288, 512; very inaccurate
                
                warped_blur_map, warped_blur_mask, _, _ = warper.forward_warp(input_blur_batch, masks_batch, mvs_depths_batch, input_extrs_batch, 
                                                                                target_extrs_batch, input_intrs_batch, target_intrs_batch)
                warped_blur_maps.append(warped_blur_map.cpu()) # batch_size, 3, 288, 512; 3 is repeat
                warped_blur_masks.append(warped_blur_mask.cpu()) # batch_size, 1, 288, 512            

                warped_motion_map, warped_motion_mask, _, _ = warper.forward_warp(input_motion_batch, masks_batch, mvs_depths_batch, input_extrs_batch, 
                                                                                target_extrs_batch, input_intrs_batch, target_intrs_batch)
                warped_motion_maps.append(warped_motion_map.cpu()) # batch_size, 3, 288, 512; 3 is repeat
                warped_motion_masks.append(warped_motion_mask.cpu()) # batch_size, 1, 288, 512    

        warped_depths = torch.cat(warped_depths, dim=0) # N, 1, 288, 512
        valid_masks = torch.cat(valid_masks, dim=0) # N, 1, 288, 512
        warped_frames = torch.cat(warped_frames, dim=0) # N, 3, 288, 512

        warped_blur_maps = torch.cat(warped_blur_maps, dim=0) # N, 3, 288, 512
        warped_blur_masks = torch.cat(warped_blur_masks, dim=0) # N, 1, 288, 512

        warped_motion_maps = torch.cat(warped_motion_maps, dim=0) # N, 3, 288, 512
        warped_motion_masks = torch.cat(warped_motion_masks, dim=0) # N, 1, 288, 512

        if inpaint:
            simple_lama = SimpleLama() # use Lama for inpainting if needed

        print('Virtual view num: ', len(warped_frames))
        virtual_cams = []
        unit = len(warped_frames) // batch_size
        for i in range(len(warped_frames)):
            id = len(input_cams) + i # camera id

            if i < unit:
                uid = input_cams[i].uid
                fid = input_cams[i].fid.cpu().numpy().squeeze(0)
                image_name = input_cams[i].image_name + '_pr'
            elif i < 2*unit:
                uid = input_cams[i % unit + 1].uid
                fid = input_cams[i % unit + 1].fid.cpu().numpy().squeeze(0)
                image_name = input_cams[i % unit + 1].image_name + '_pl'
            elif i < 4*unit:
                uid = input_cams[i % unit].uid
                fid = input_cams[i % unit].fid.cpu().numpy().squeeze(0)
                image_name = input_cams[i % unit].image_name + '_vr'
            elif i < 6*unit:
                uid = input_cams[i % unit + 1].uid
                fid = input_cams[i % unit + 1].fid.cpu().numpy().squeeze(0)
                image_name = input_cams[i % unit + 1].image_name + '_vl'
            elif i < 8*unit:
                uid = input_cams[i % unit].uid
                fid = input_cams[i % unit].fid.cpu().numpy().squeeze(0)
                image_name = input_cams[i % unit].image_name + '_cr'
            else:
                uid = input_cams[i % unit + 1].uid
                fid = input_cams[i % unit + 1].fid.cpu().numpy().squeeze(0)
                image_name = input_cams[i % unit + 1].image_name + '_cl'

            C2W = target_extrs[i].inverse()
            C2W[:3, 3] = C2W[:3, 3] * sc # MARK: scale
            target_extr = C2W.inverse()
            R, T = target_extr[:3, :3].cpu().numpy().transpose(), target_extr[:3, 3].cpu().numpy()
            focal_length_x, focal_length_y = target_intrs[i][0,0], target_intrs[i][1,1]
            H, W = warped_frames.shape[2:4]
            K = np.array([
                [focal_length_x, 0., W/2],
                [0., focal_length_y, H/2],
                [0., 0., 1.]
            ], dtype=np.float32)
            FovY = focal2fov(focal_length_y, H)
            FovX = focal2fov(focal_length_x, W)
            depth = warped_depths[i].cpu().numpy().squeeze(0) * sc # depth scale

            warped_img = warped_frames[i] # 3, 288, 512
            mask = valid_masks[i].squeeze().to(torch.bool).detach().cpu().numpy() # 288, 512

            warped_blur_map = warped_blur_maps[i] # 3, 288, 512; 3 is repeat
            warped_blur_mask = warped_blur_masks[i].squeeze().to(torch.bool).detach().cpu().numpy() # 288, 512

            warped_motion_map = warped_motion_maps[i] # 3, 288, 512; 3 is repeat
            warped_motion_mask = warped_motion_masks[i].squeeze().to(torch.bool).detach().cpu().numpy() # 288, 512

            if inpaint:# inpaint
                warped_img = warped_img.permute(1,2,0).cpu().numpy() # 288, 512, 3
                warped_img = torch.from_numpy(np.array(simple_lama(warped_img*255, (~mask).astype(np.uint8)*255))[:H, :W]).permute(2,0,1)/255. # 3, 288, 512

                warped_blur_map = warped_blur_map.permute(1,2,0).cpu().numpy() # 288, 512, 3; 3 is repeat
                warped_blur_map = torch.from_numpy(np.array(simple_lama(warped_blur_map*255, (~warped_blur_mask).astype(np.uint8)*255))[:H, :W]).permute(2,0,1)/255. # 3, 288, 512; 3 is repeat
                warped_blur_map = warped_blur_map[0,:,:].squeeze(0) # 288, 512

                warped_motion_map = warped_motion_map.permute(1,2,0).cpu().numpy() # 288, 512, 3; 3 is repeat
                warped_motion_map = torch.from_numpy(np.array(simple_lama(warped_motion_map*255, (~warped_motion_mask).astype(np.uint8)*255))[:H, :W]).permute(2,0,1)/255. # 3, 288, 512; 3 is repeat
                warped_motion_map = warped_motion_map[0,:,:].squeeze(0) # 288, 512
                warped_motion_map = warped_motion_map > 0.9                
                
            virtual_cam = Camera(colmap_id=None, R=R, T=T, K=K, fid=fid,
                                FoVx=FovX, FoVy=FovY, 
                                image=warped_img, gt_alpha_mask=None,
                                image_name=image_name, uid=uid, data_device='cuda',
                                depth=depth, blur_map=warped_blur_map, motion_mask=warped_motion_map, is_virtual=True)
            virtual_cams.append(virtual_cam)

        return virtual_cams

    def getWorld2View2(self, R, t, scale=1.0): # world -> camera
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = cam_center / scale # camera center scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)
    
    def prepare_data(self, input_cams, target_cams, sc=1.):
        
        ids = range(len(target_cams))
        
        input_imgs = torch.stack([input_cams[id].original_image for id in ids]) # v_num, 3, 288, 512
        input_blur_maps = torch.stack([input_cams[id].blur_map for id in ids]).unsqueeze(1).repeat(1, 3, 1, 1) # v_num, 3, 288, 512
        input_motion_masks = torch.stack([input_cams[id].motion_mask for id in ids]).unsqueeze(1).repeat(1, 3, 1, 1) # v_num, 3, 288, 512
        if input_cams[0].depth_no_sc is None:
            mvs_depths = torch.from_numpy(np.stack([input_cams[id].depth.cpu() for id in ids])).unsqueeze(1) # v_num, 1, 288, 512
        else:
            mvs_depths = torch.from_numpy(np.stack([input_cams[id].depth_no_sc.cpu() for id in ids])).unsqueeze(1) # v_num, 1, 288, 512
        input_extrs = torch.from_numpy(np.stack([self.getWorld2View2(input_cams[id].R, input_cams[id].T, sc) for id in ids])) # v_num, 4, 4
        input_intrs = torch.from_numpy(np.stack([input_cams[id].K for id in ids])) # v_num, 3, 3

        # target pose
        parallel_poses = torch.from_numpy(parallel_trajectory_interpolation(input_extrs)) # c2w
        parallel_extrs = torch.from_numpy(np.stack([np.linalg.inv(parallel_poses[i]) for i in range(parallel_poses.shape[0])])) # w2c
        parallel_intrs = torch.from_numpy(np.stack([input_cams[0].K] * len(parallel_extrs)))  # same intrinsics
        vertical_poses = torch.from_numpy(vertical_trajectory_interpolation(input_extrs, right=0., left=0., radiu=0.5)) # c2w
        vertical_extrs = torch.from_numpy(np.stack([np.linalg.inv(vertical_poses[i]) for i in range(vertical_poses.shape[0])])) # w2c
        vertical_intrs = torch.from_numpy(np.stack([input_cams[0].K] * len(vertical_extrs)))  # same intrinsics
        # central_poses = torch.from_numpy(central_interpolation(input_extrs, right=0., left=0., radiu=1.0)) # c2w
        # central_extrs = torch.from_numpy(np.stack([np.linalg.inv(central_poses[i]) for i in range(central_poses.shape[0])])) # w2c
        # central_intrs = torch.from_numpy(np.stack([input_cams[0].K] * len(central_extrs)))  # same intrinsics

        # input_imgs = torch.cat([input_imgs[:-1,...], input_imgs[1:,...], input_imgs[:-1,...], input_imgs[:-1,...], input_imgs[1:,...], input_imgs[1:,...], \
        #                         input_imgs[:-1,...], input_imgs[:-1,...], input_imgs[1:,...], input_imgs[1:,...]], dim=0).cpu() # NOTE all 
        # input_blur_maps = torch.cat([input_blur_maps[:-1,...], input_blur_maps[1:,...], input_blur_maps[:-1,...], input_blur_maps[:-1,...], \
        #                              input_blur_maps[1:,...], input_blur_maps[1:,...], input_blur_maps[:-1,...], input_blur_maps[:-1,...], \
        #                              input_blur_maps[1:,...], input_blur_maps[1:,...]], dim=0).cpu()
        # input_motion_masks = torch.cat([input_motion_masks[:-1,...], input_motion_masks[1:,...], input_motion_masks[:-1,...], input_motion_masks[:-1,...], \
        #                                 input_motion_masks[1:,...], input_motion_masks[1:,...], input_motion_masks[:-1,...], input_motion_masks[:-1,...], \
        #                                 input_motion_masks[1:,...], input_motion_masks[1:,...]], dim=0).cpu()
        # mvs_depths = torch.cat([mvs_depths[:-1,...], mvs_depths[1:,...], mvs_depths[:-1,...], mvs_depths[:-1,...], mvs_depths[1:,...], mvs_depths[1:,...], \
        #                         mvs_depths[:-1,...], mvs_depths[:-1,...], mvs_depths[1:,...], mvs_depths[1:,...]], dim=0).cpu()
        # input_extrs = torch.cat([input_extrs[:-1,...], input_extrs[1:,...], input_extrs[:-1,...], input_extrs[:-1,...], input_extrs[1:,...], input_extrs[1:,...], \
        #                          input_extrs[:-1,...], input_extrs[:-1,...], input_extrs[1:,...], input_extrs[1:,...]], dim=0)
        # input_intrs = torch.cat([input_intrs[:-1,...], input_intrs[1:,...], input_intrs[:-1,...], input_intrs[:-1,...], input_intrs[1:,...], input_intrs[1:,...], \
        #                          input_intrs[:-1,...], input_intrs[:-1,...], input_intrs[1:,...], input_intrs[1:,...]], dim=0)
        # target_extrs = torch.cat([parallel_extrs.repeat(2, 1, 1), vertical_extrs, central_extrs], dim=0)
        # target_intrs = torch.cat([parallel_intrs.repeat(2, 1, 1), vertical_intrs, central_intrs], dim=0)

        # input_imgs = torch.cat([input_imgs[:-1,...], input_imgs[1:,...], input_imgs[:-1,...], input_imgs[:-1,...], input_imgs[1:,...], input_imgs[1:,...]], dim=0).cpu() # NOTE only_pv
        # input_blur_maps = torch.cat([input_blur_maps[:-1,...], input_blur_maps[1:,...], input_blur_maps[:-1,...], input_blur_maps[:-1,...], \
        #                              input_blur_maps[1:,...], input_blur_maps[1:,...]], dim=0).cpu()
        # input_motion_masks = torch.cat([input_motion_masks[:-1,...], input_motion_masks[1:,...], input_motion_masks[:-1,...], input_motion_masks[:-1,...], \
        #                                 input_motion_masks[1:,...], input_motion_masks[1:,...]], dim=0).cpu()
        # mvs_depths = torch.cat([mvs_depths[:-1,...], mvs_depths[1:,...], mvs_depths[:-1,...], mvs_depths[:-1,...], mvs_depths[1:,...], mvs_depths[1:,...]], dim=0).cpu()
        # input_extrs = torch.cat([input_extrs[:-1,...], input_extrs[1:,...], input_extrs[:-1,...], input_extrs[:-1,...], input_extrs[1:,...], input_extrs[1:,...]], dim=0)
        # input_intrs = torch.cat([input_intrs[:-1,...], input_intrs[1:,...], input_intrs[:-1,...], input_intrs[:-1,...], input_intrs[1:,...], input_intrs[1:,...]], dim=0)
        # target_extrs = torch.cat([parallel_extrs.repeat(2, 1, 1), vertical_extrs], dim=0)
        # target_intrs = torch.cat([parallel_intrs.repeat(2, 1, 1), vertical_intrs], dim=0)

        input_imgs = torch.cat([input_imgs[:-1,...], input_imgs[1:,...], input_imgs[:-1,...], input_imgs[:-1,...]], dim=0).cpu() # NOTE only_vertical_right
        input_blur_maps = torch.cat([input_blur_maps[:-1,...], input_blur_maps[1:,...], input_blur_maps[:-1,...], input_blur_maps[:-1,...]], dim=0).cpu()
        input_motion_masks = torch.cat([input_motion_masks[:-1,...], input_motion_masks[1:,...], input_motion_masks[:-1,...], input_motion_masks[:-1,...]], dim=0).cpu()
        mvs_depths = torch.cat([mvs_depths[:-1,...], mvs_depths[1:,...], mvs_depths[:-1,...], mvs_depths[:-1,...]], dim=0).cpu()
        input_extrs = torch.cat([input_extrs[:-1,...], input_extrs[1:,...], input_extrs[:-1,...], input_extrs[:-1,...]], dim=0)
        input_intrs = torch.cat([input_intrs[:-1,...], input_intrs[1:,...], input_intrs[:-1,...], input_intrs[:-1,...]], dim=0)
        target_extrs = torch.cat([parallel_extrs.repeat(2, 1, 1), vertical_extrs], dim=0)
        target_intrs = torch.cat([parallel_intrs.repeat(2, 1, 1), vertical_intrs], dim=0)

        # input_imgs = torch.cat([input_imgs[:-1,...], input_imgs[1:,...], input_imgs[:-1,...], input_imgs[:-1,...], input_imgs[1:,...], input_imgs[1:,...]], dim=0).cpu() # NOTE vright_cleft
        # input_blur_maps = torch.cat([input_blur_maps[:-1,...], input_blur_maps[1:,...], input_blur_maps[:-1,...], input_blur_maps[:-1,...], \
        #                              input_blur_maps[1:,...], input_blur_maps[1:,...]], dim=0).cpu()
        # input_motion_masks = torch.cat([input_motion_masks[:-1,...], input_motion_masks[1:,...], input_motion_masks[:-1,...], input_motion_masks[:-1,...], \
        #                                 input_motion_masks[1:,...], input_motion_masks[1:,...]], dim=0).cpu()
        # mvs_depths = torch.cat([mvs_depths[:-1,...], mvs_depths[1:,...], mvs_depths[:-1,...], mvs_depths[:-1,...], mvs_depths[1:,...], mvs_depths[1:,...]], dim=0).cpu()
        # input_extrs = torch.cat([input_extrs[:-1,...], input_extrs[1:,...], input_extrs[:-1,...], input_extrs[:-1,...], input_extrs[1:,...], input_extrs[1:,...]], dim=0)
        # input_intrs = torch.cat([input_intrs[:-1,...], input_intrs[1:,...], input_intrs[:-1,...], input_intrs[:-1,...], input_intrs[1:,...], input_intrs[1:,...]], dim=0)
        # target_extrs = torch.cat([parallel_extrs.repeat(2, 1, 1), vertical_extrs, central_extrs], dim=0)
        # target_intrs = torch.cat([parallel_intrs.repeat(2, 1, 1), vertical_intrs, central_intrs], dim=0)

        visual = False
        # visual = True
        if visual:
            save_path = "/home/xuankai/aimage/vitual_pose"
            save_path1 = "/home/xuankai/aimage/test_pose"
            train_c2ws = torch.from_numpy(np.stack([np.linalg.inv(self.getWorld2View2(input_cams[id].R, input_cams[id].T, sc)) for id in ids]))
            test_c2ws = torch.from_numpy(np.stack([np.linalg.inv(self.getWorld2View2(target_cams[id].R, target_cams[id].T, sc)) for id in ids]))
            virtual_c2ws = torch.from_numpy(np.stack([np.linalg.inv(target_extrs[id]) for id in range(len(target_extrs))]))
            # posevisual(save_path, train_c2ws, test_c2ws, parallel_poses, vertical_poses, central_poses)
            posevisual(save_path, train_c2ws, test_c2ws, virtual_c2ws)
            posevisual(save_path1, train_c2ws, test_c2ws)

        return input_imgs, input_blur_maps, input_motion_masks, mvs_depths, input_extrs, input_intrs, target_extrs, target_intrs

def create_batches(*tensors: torch.Tensor, batch_size: int):
    return list(zip(*[torch.split(tensor, batch_size) for tensor in tensors]))

def parallel_trajectory_interpolation(extrinsics):
    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])]) # c2w, v_num, 4, 4

    start_pos = poses[:-1,...]
    end_pos = poses[1:,...]
    mid_pos = (start_pos + end_pos) * 0.5

    pseudo_poses = []
    for pose in mid_pos:
        pseudo_pose = np.eye(4, dtype=np.float32)
        pseudo_pose[:3, :4] = viewmatrix(pose[:3,2], pose[:3,1], pose[:3,3])
        pseudo_poses.append(pseudo_pose)

    return np.stack(pseudo_poses, axis=0)

def vertical_trajectory_single_direction_interpolation(start_pos, end_pos, z_scale=0.5, radiu=0.5):
    '''
    known: (a1, a2, a3), (b1, b2, b3), a1c1 + a2c2 + a3c3 = 0, b1c1 + b2c2 + b3c3 = 0, c3=1
    ask: (c1, c2, c3)
    '''
    a = start_pos[:, :3, 2].copy() # z-axis; v_num-1, 3
    b = (end_pos - start_pos)[:, :3, 3] # camera trajectory; v_num-1, 3
    c1 = ((a[:,1]*b[:,2]-a[:,2]*b[:,1]) / (a[:,0]*b[:,1]-a[:,1]*b[:,0] + 1e-5))[...,np.newaxis] # v_num-1, 1
    c2 = ((a[:,2]*b[:,0]-a[:,0]*b[:,2]) / (a[:,0]*b[:,1]-a[:,1]*b[:,0] + 1e-5))[...,np.newaxis] # v_num-1, 1
    c3 = np.ones(c1.shape)
    c = np.concatenate([c1[:,0:1], c2[:,0:1], c3[:,0:1]], -1) # v_num-1, 3
    c = c / np.linalg.norm(c, axis=1)[...,np.newaxis] # v_num-1, 3
    a = a / np.linalg.norm(a, axis=1)[...,np.newaxis] # v_num-1, 3

    pseudo_poses1 = start_pos.copy() # MARK: in-place
    pseudo_poses1[:, :3, 3] = pseudo_poses1[:, :3, 3] + a * z_scale + c * radiu # v_num-1, 4, 4

    pseudo_poses2 = start_pos.copy()
    pseudo_poses2[:, :3, 3] = pseudo_poses2[:, :3, 3] + a * z_scale - c * radiu # v_num-1, 4, 4

    pseudo_poses = np.concatenate([pseudo_poses1[:,:], pseudo_poses2[:,:]], 0) # 2 * (v_num-1), 4, 4
    return pseudo_poses

def central_single_direction_interpolation(start_pos, end_pos, central_poses, z_scale=0.5, radiu=0.5):
    
    '''
    known: (a1, a2, a3), (b1, b2, b3), a1c1 + a2c2 + a3c3 = 0, b1c1 + b2c2 + b3c3 = 0, c3=1
    ask: (c1, c2, c3)
    '''
    a = start_pos[:, :3, 2].copy() # z-axis; v_num-1, 3
    b = (end_pos - start_pos)[:, :3, 3] # camera trajectory; v_num-1, 3
    c1 = ((a[:,1]*b[:,2]-a[:,2]*b[:,1]) / (a[:,0]*b[:,1]-a[:,1]*b[:,0] + 1e-5))[...,np.newaxis] # v_num-1, 1
    c2 = ((a[:,2]*b[:,0]-a[:,0]*b[:,2]) / (a[:,0]*b[:,1]-a[:,1]*b[:,0] + 1e-5))[...,np.newaxis] # v_num-1, 1
    c3 = np.ones(c1.shape)
    c = np.concatenate([c1[:,0:1], c2[:,0:1], c3[:,0:1]], -1) # v_num-1, 3
    c = c / np.linalg.norm(c, axis=1)[...,np.newaxis] # v_num-1, 3
    a = a / np.linalg.norm(a, axis=1)[...,np.newaxis] # v_num-1, 3

    pseudo_poses1 = central_poses.copy() # MARK: in-place
    start_center1 = start_pos[:, :3, 3].copy()
    pseudo_poses1[:, :3, 3] = start_center1 + b * 0.5 + a * z_scale + c * radiu # v_num-1, 4, 4

    pseudo_poses2 = central_poses.copy() # MARK: in-place
    start_center2 = start_pos[:, :3, 3].copy()
    pseudo_poses2[:, :3, 3] = start_center2 + b * 0.5 + a * z_scale - c * radiu # v_num-1, 4, 4

    pseudo_poses = np.concatenate([pseudo_poses1[:,:], pseudo_poses2[:,:]], 0) # 2 * (v_num-1), 4, 4
    return pseudo_poses

def vertical_trajectory_interpolation(extrinsics, right=0.5, left=0.5, radiu=0.5):

    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])]) # c2w, v_num, 4, 4

    start_pos = poses[:-1,...]
    end_pos = poses[1:,...]
    pseudo_poses_right = vertical_trajectory_single_direction_interpolation(start_pos, end_pos, right, radiu) # 2 * (v_num-1), 4, 4
    pseudo_poses_left = vertical_trajectory_single_direction_interpolation(end_pos, start_pos, left, radiu) # 2 * (v_num-1), 4, 4
    pseudo_poses = np.concatenate([pseudo_poses_right[:,:], pseudo_poses_left[:,:]], 0) # 4 * (v_num-1), 4, 4

    # return pseudo_poses
    return pseudo_poses_right

def central_interpolation(extrinsics, right=0.5, left=0.5, radiu=0.5):

    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])]) # c2w, v_num, 4, 4

    start_pos = poses[:-1,...]
    end_pos = poses[1:,...]
    mid_pos = (start_pos + end_pos) * 0.5

    pseudo_poses = []
    for pose in mid_pos:
        pseudo_pose = np.eye(4, dtype=np.float32)
        pseudo_pose[:3, :4] = viewmatrix(pose[:3,2], pose[:3,1], pose[:3,3]) # lookdir, up, position
        pseudo_poses.append(pseudo_pose)

    central_poses =  np.stack(pseudo_poses, axis=0) # v_num-1, 4, 4
    pseudo_poses_right = central_single_direction_interpolation(start_pos, end_pos, central_poses, right, radiu) # 2 * (v_num-1), 4, 4
    pseudo_poses_left = central_single_direction_interpolation(end_pos, start_pos, central_poses, left, radiu) # 2 * (v_num-1), 4, 4
    central_pseudo_poses = np.concatenate([pseudo_poses_right[:,:], pseudo_poses_left[:,:]], 0) # 4 * (v_num-1), 4, 4

    return central_pseudo_poses
    # return pseudo_poses_left