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
import numpy as np
from tqdm import tqdm
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.mlp_model import MLP
from scene.blur_model import Blur
from scene.cameras import Camera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import getWorld2View, focal2fov
from utils.pose_utils import viewmatrix, posevisual

from simple_lama_inpainting import SimpleLama
from utils.warp_utils import Warper

class Scene:
    gaussians: GaussianModel # class attribute

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
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
            scene_info = sceneLoadTypeCallbacks["D2RF"](args.source_path) # D2RF

        elif os.path.exists(os.path.join(args.source_path, "sharp_images")):
            print("Found sharp_images, assuming DyBluRF data set!")
            scene_info = sceneLoadTypeCallbacks["DyBluRF"](args.source_path) # DyBluRF

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
        print('cameras_extent =', self.cameras_extent)
        for resolution_scale in resolution_scales:  # CameraInfo -> Camera; MARK: resize image
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
            # get unseen cameras
            print("Generating Virtual Cameras")
            self.virtual_cameras[resolution_scale] = self.generateVirtualCams(self.train_cameras[resolution_scale], self.test_cameras[resolution_scale], scene_info.sc)
            torch.cuda.empty_cache()

        # shuffle train camera and test camera
        if shuffle: 
            for resolution_scale in resolution_scales:
                self.sorted_train_cameras[resolution_scale] = self.train_cameras[resolution_scale].copy()
                self.sorted_test_cameras[resolution_scale] = self.test_cameras[resolution_scale].copy()
                self.sorted_virtual_cameras[resolution_scale] = self.virtual_cameras[resolution_scale].copy()

                random.shuffle(self.train_cameras[resolution_scale])  # Multi-res consistent random shuffling
                random.shuffle(self.test_cameras[resolution_scale])  # Multi-res consistent random shuffling
                random.shuffle(self.virtual_cameras[resolution_scale])  # Multi-res consistent random shuffling
     
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, args.operate, args.time,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

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

    def generateVirtualCams(self, input_cams, target_cams, sc, batch_size=1, inpaint=True):
        print('Train view num: ', len(input_cams))

        only_right = False
        # only_right = True
        input_imgs, mvs_depths, input_extrs, input_intrs, target_extrs, target_intrs = self.prepare_data(input_cams, target_cams, sc, only_right) # mvs_depths: absolute depth

        # split into batches
        input_batches = create_batches(input_imgs, mvs_depths, input_extrs, input_intrs, batch_size=batch_size)
        target_batches = create_batches(target_intrs, target_extrs, batch_size=batch_size)

        warper = Warper()
        warped_frames, valid_masks, warped_depths = [], [], []
        with torch.no_grad():
            for (input_imgs_batch, mvs_depths_batch, input_extrs_batch, input_intrs_batch), (target_intrs_batch, target_extrs_batch) \
                in tqdm(zip(input_batches, target_batches), desc="Unseen view prior", unit="batch", total=int(len(input_imgs) / batch_size)):
                torch.cuda.empty_cache()
                masks_batch = None
                # get priors for unseen views by forward warping
                warped_frame, valid_mask, warped_depth, _ = warper.forward_warp(input_imgs_batch, masks_batch, mvs_depths_batch, input_extrs_batch, 
                                                                                target_extrs_batch, input_intrs_batch, target_intrs_batch)
                warped_frames.append(warped_frame.cpu())
                valid_masks.append(valid_mask.cpu())
                warped_depths.append(warped_depth.cpu()) # very inaccurate
                

        warped_depths = torch.cat(warped_depths, dim=0)
        valid_masks = torch.cat(valid_masks, dim=0)
        warped_frames = torch.cat(warped_frames, dim=0)

        print('Virtual view num: ', len(warped_frames))
        virtual_cams = []
        if inpaint:
            simple_lama = SimpleLama() # use Lama for inpainting if needed
        for i in range(len(warped_frames)):
            id = len(input_cams) + i # uid
            if i < (len(warped_frames) // 4):
                fid = input_cams[i].fid.cpu().numpy().squeeze(0)
                image_name = input_cams[i].image_name + '_p'
            elif i < (len(warped_frames) // 2):
                fid = input_cams[i-(len(warped_frames)//4)+1].fid.cpu().numpy().squeeze(0)
                image_name = input_cams[i-(len(warped_frames)//4)+1].image_name + '_p'
            else:
                fid = input_cams[i % (len(warped_frames)//4)].fid.cpu().numpy().squeeze(0)
                image_name = input_cams[i % (len(warped_frames)//4)].image_name + '_v'

            C2W = target_extrs[i].inverse()
            C2W[:3, 3] = C2W[:3, 3] * sc
            target_extr = C2W.inverse()
            R, T = target_extr[:3, :3].cpu().numpy().transpose(), target_extr[:3, 3].cpu().numpy()
            focal_length_x, focal_length_y = target_intrs[i][0,0], target_intrs[i][1,1]
            H, W = warped_frames.shape[2:4]
            K = np.array([
                [focal_length_x, 0., W],
                [0., focal_length_y, H],
                [0., 0., 1.]
            ], dtype=np.float32)
            FovY = focal2fov(focal_length_y, H)
            FovX = focal2fov(focal_length_x, W)
            warped_img = warped_frames[i]
            mask = valid_masks[i].squeeze().to(torch.bool).detach().cpu().numpy()
            depth = warped_depths[i].cpu().numpy().squeeze(0)
            if inpaint:# inpaint
                warped_img = warped_img.permute(1,2,0).cpu().numpy()
                warped_img = torch.from_numpy(np.array(simple_lama(warped_img*255, (~mask).astype(np.uint8)*255))[:H, :W]).permute(2,0,1)/255.
            
            virtual_cam = Camera(colmap_id=None, R=R, T=T, K=K, fid=fid,
                                FoVx=FovX, FoVy=FovY, 
                                image=warped_img, gt_alpha_mask=None,
                                image_name=image_name, uid=id, data_device='cuda',
                                depth=depth, is_virtual=True)
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
    
    def prepare_data(self, input_cams, target_cams, sc=1., only_right=True):
        
        ids = range(len(target_cams))
        
        input_imgs = torch.stack([input_cams[id].original_image for id in ids]) # v_num, 3, 288, 512
        mvs_depths = torch.from_numpy(np.stack([input_cams[id].depth.cpu() for id in ids])).unsqueeze(1) # v_num, 1, 288, 512
        input_extrs = torch.from_numpy(np.stack([self.getWorld2View2(input_cams[id].R, input_cams[id].T, sc) for id in ids])) # v_num, 4, 4
        input_intrs = torch.from_numpy(np.stack([input_cams[id].K for id in ids])) # v_num, 3, 3

        # target_extrs = torch.from_numpy(np.stack([getWorld2View(target_cams[id].R, target_cams[id].T) for id in ids])) # v_num, 4, 4
        # target_intrs = torch.from_numpy(np.stack([target_cams[id].K for id in ids])) # v_num, 3, 3

        parallel_poses = torch.from_numpy(parallel_trajectory_interpolation(input_extrs)) # c2w
        parallel_extrs = torch.from_numpy(np.stack([np.linalg.inv(parallel_poses[i]) for i in range(parallel_poses.shape[0])])) # w2c
        parallel_intrs = torch.from_numpy(np.stack([input_cams[0].K] * len(parallel_extrs)))  # same intrinsics
        vertical_poses = torch.from_numpy(vertical_trajectory_interpolation(input_extrs)) # c2w
        vertical_extrs = torch.from_numpy(np.stack([np.linalg.inv(vertical_poses[i]) for i in range(vertical_poses.shape[0])])) # w2c
        vertical_intrs = torch.from_numpy(np.stack([input_cams[0].K] * len(vertical_extrs)))  # same intrinsics


        input_imgs = torch.cat([input_imgs[:-1,...], input_imgs[1:,...], input_imgs[:-1,...], input_imgs[:-1,...]], dim=0).cpu()
        mvs_depths = torch.cat([mvs_depths[:-1,...], mvs_depths[1:,...], mvs_depths[:-1,...], mvs_depths[:-1,...]], dim=0).cpu()
        input_extrs = torch.cat([input_extrs[:-1,...], input_extrs[1:,...], input_extrs[:-1,...], input_extrs[:-1,...]], dim=0)
        input_intrs = torch.cat([input_intrs[:-1,...], input_intrs[1:,...], input_intrs[:-1,...], input_intrs[:-1,...]], dim=0)
        target_extrs = torch.cat([parallel_extrs.repeat(2, 1, 1), vertical_extrs], dim=0)
        target_intrs = torch.cat([parallel_intrs.repeat(2, 1, 1), vertical_intrs], dim=0)

        visual = False
        if visual:
            save_path = "/home/xuankai/aimage/pv_pose"
            train_c2ws = torch.from_numpy(np.stack([np.linalg.inv(self.getWorld2View2(input_cams[id].R, input_cams[id].T, sc)) for id in ids]))
            test_c2ws = torch.from_numpy(np.stack([np.linalg.inv(self.getWorld2View2(target_cams[id].R, target_cams[id].T, sc)) for id in ids]))
            virtual_c2ws = torch.from_numpy(np.stack([np.linalg.inv(target_extrs[id]) for id in range(len(target_extrs))]))
            posevisual(save_path, train_c2ws, test_c2ws, virtual_c2ws)

        return input_imgs, mvs_depths, input_extrs, input_intrs, target_extrs, target_intrs

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

def vertical_trajectory_interpolation(extrinsics):
    '''
    known: (a1, a2, a3), (b1, b2, b3), a1c1 + a2c2 + a3c3 = 0, b1c1 + b2c2 + b3c3 = 0, c3=1
    ask: (c1, c2, c3)
    '''
    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])]) # c2w, v_num, 4, 4

    start_pos = poses[:-1,...]
    end_pos = poses[1:,...]
    a = start_pos[:, :3, 2] # z-axis; v_num-1, 3
    b = (end_pos - start_pos)[:, :3, 3] # camera trajectory; v_num-1, 3
    c1 = ((a[:,1]*b[:,2]-a[:,2]*b[:,1]) / (a[:,0]*b[:,1]-a[:,1]*b[:,0] + 1e-5))[...,np.newaxis] # v_num-1, 1
    c2 = ((a[:,2]*b[:,0]-a[:,0]*b[:,2]) / (a[:,0]*b[:,1]-a[:,1]*b[:,0] + 1e-5))[...,np.newaxis] # v_num-1, 1
    c3 = np.ones(c1.shape)
    c = np.concatenate([c1[:,0:1], c2[:,0:1], c3[:,0:1]], -1) # v_num-1, 3
    c = c / np.linalg.norm(c, axis=1)[...,np.newaxis] # v_num-1, 3

    pseudo_poses1 = start_pos.copy() # MARK: in-place
    pseudo_poses1[:, :3, 3] = pseudo_poses1[:, :3, 3] + c * 0.5 # v_num-1, 4, 4

    pseudo_poses2 = start_pos.copy()
    pseudo_poses2[:, :3, 3] = pseudo_poses2[:, :3, 3] - c * 0.5 # v_num-1, 4, 4

    pseudo_poses = np.concatenate([pseudo_poses1[:,:], pseudo_poses2[:,:]], 0) # 2 * (v_num-1), 4, 4
    return pseudo_poses