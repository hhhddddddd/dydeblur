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
import cupy as cp
from utils.general_utils import inverse_sigmoid, gumbel_sigmoid, inverse_gumbel_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
import roma
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import  TrackObservations, StaticObservations, strip_symmetric, build_scaling_rotation, get_linear_noise_func, knn
from scene.blur_kernel import GTnet
from scene.motion_model import init_motion_params_with_procrustes
from random import randint

class GaussianParams(nn.Module):
    def __init__(self, means, quats, scales, colors, opacities, motion_coefs, dynamics):
        super().__init__()
        self.means = means
        self.quats = quats
        self.scales = scales
        self.colors = colors
        self.opacities = opacities
        self.motion_coefs = motion_coefs
        self.dynamics = dynamics

        params_dict = { # only for fg
            "means": nn.Parameter(means),
            "motion_coefs": nn.Parameter(motion_coefs),
        }
        self.params = nn.ParameterDict(params_dict)

class GaussianModel: # when initial, gaussians is already belong to scene
    def __init__(self, sh_degree: int): # contain setup_functions

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2) # RS(RS)^T
            symm = strip_symmetric(actual_covariance) # Only the upper triangle of symmetric matrix
            return symm

        self.active_sh_degree = 0       # the sh degree currently in use
        self.max_sh_degree = sh_degree  # maximum sh degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)      # direct color
        self._features_rest = torch.empty(0)    # rest color
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._motion_coefs = torch.empty(0)
        self._dynamics = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.gumbel_noise = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.motion_coefs_activation = lambda x: F.softmax(x, dim=-1)
        # self.inverse_motion_coefs_activation = None

        # self.dynamic_activation = gumbel_sigmoid
        # self.inverse_dynamic_activation = inverse_gumbel_sigmoid

        self.rotation_activation = torch.nn.functional.normalize # strange

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_dynamics(self):
        return self._dynamics

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest   

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_motion_coefs(self):
        ret = self.motion_coefs_activation(self._motion_coefs)
        return ret

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_GTnet(self, hidden=2, width=64, pos_delta=0, num_moments=4):
        self.GTnet = GTnet(num_hidden=hidden, width=width, pos_delta=pos_delta, num_moments=num_moments)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float): # 3d scene gaussians initialization
        self.spatial_lr_scale = spatial_lr_scale   # spatial_lr_scale: camera_extent
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()     # torch.Size([100000, 3])
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())   # torch.Size([100000, 3])
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # torch.Size([100000, 3, 16])
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)  # torch.Size([100000])
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1 # NOTE don't understand
        motion_coefs = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        # dynamics = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))     # direct color  torch.Size([100000, 1, 3])
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))    # rest color    torch.Size([100000, 15, 3])
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._motion_coefs = nn.Parameter(motion_coefs.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_track(self, fg_params, bg_params, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale   # spatial_lr_scale: camera_extent
        fused_point_cloud = torch.cat([fg_params.means.detach(), bg_params.means.detach()], dim=0).float().cuda()     # torch.Size([100000, 3])
        fused_color = RGB2SH(torch.cat([fg_params.colors, bg_params.colors], dim=0).float().cuda())   # torch.Size([100000, 3])
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # torch.Size([100000, 3, 16])
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = torch.cat([fg_params.scales, bg_params.scales], dim=0).cuda() 
        rots = torch.cat([fg_params.quats, bg_params.quats], dim=0).cuda() 
        rots[:, 0] = 1 # NOTE don't understand
        motion_coefs = torch.cat([fg_params.motion_coefs.detach(), bg_params.motion_coefs.detach()], dim=0).cuda() 
        dynamics = torch.cat([fg_params.dynamics, bg_params.dynamics], dim=0).cuda() 
        opacities = torch.cat([fg_params.opacities, bg_params.opacities], dim=0).cuda() 

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))     # direct color  torch.Size([100000, 1, 3])
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))    # rest color    torch.Size([100000, 15, 3])
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._motion_coefs = nn.Parameter(motion_coefs.requires_grad_(True))
        self._dynamics = nn.Parameter(dynamics.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def depth_track_point(self, foreground_points, background_points, cano_t, num_motion_bases=20): # origin: 20
        tracks_3d = TrackObservations(  xyz=foreground_points[0],
                                        visibles=foreground_points[1],
                                        invisibles=foreground_points[2],
                                        confidences=foreground_points[3],
                                        colors=foreground_points[4])

        rot_type = "6d"
        cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item()) # 10
        # cano_t = 20
        print("cano_t:", cano_t)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda

        motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
            tracks_3d, num_motion_bases, rot_type, cano_t)
        motion_bases = motion_bases.to(device)

        fg_params, idcs = self.init_fg(cano_t, tracks_3d, motion_coefs)
        fg_params = fg_params.to(device)

        if idcs is not None:
            xyz = tracks_3d.xyz[idcs] # 7392, 24, 3
            visibles = tracks_3d.visibles[idcs] # 7392, 24
            invisibles = tracks_3d.invisibles[idcs] # 7392, 24
            confidences = tracks_3d.confidences[idcs] # 7392, 24
            colors = tracks_3d.colors[idcs] # 7392, 3

            tracks_3d = TrackObservations( xyz=xyz, visibles=visibles, invisibles=invisibles, confidences=confidences, colors=colors)

        bg_points = StaticObservations( xyz=background_points[0],
                                        normals=background_points[1],
                                        colors=background_points[2])
        bg_params = self.init_bg(bg_points, num_motion_bases)
        bg_params = bg_params.to(device)

        # tracks_3d = tracks_3d.to(device)

        return fg_params, motion_bases, bg_params, tracks_3d # tracks_3d for 'run_initial_optim'
    
    def init_fg(self, cano_t, tracks_3d, motion_coefs):
        visibiles = True # NOTE visibiles in cano_t space
        # visibiles = False
        if visibiles:
            vis = tracks_3d.visibles[:, cano_t]
            idcs = torch.where(vis)[0]
            # Initialize gaussian means.
            means = tracks_3d.xyz[idcs, cano_t]

            num_fg = means.shape[0]

            # Initialize gaussian colors.
            colors = tracks_3d.colors[idcs]
            # Initialize gaussian scales: find the average of the three nearest
            # neighbors in the first frame for each point and use that as the
            # scale.
            dists, _ = knn(means, 3)
            dists = torch.from_numpy(dists)
            scales = dists.mean(dim=-1, keepdim=True)
            scales = scales.clamp(torch.quantile(scales, 0.05), torch.quantile(scales, 0.95))
            scales = torch.log(scales.repeat(1, 3))

            # Initialize gaussian orientations as random.
            quats = torch.rand(num_fg, 4)
            # Initialize gaussian opacities.
            opacities = torch.logit(torch.full((num_fg,), 0.7)).unsqueeze(1) # logit <=> sigmoid
            dynamics = torch.ones(num_fg, 1) # fg: torch.ones
            motion_coefs = motion_coefs[idcs]
        else:
            idcs = None
            num_fg = tracks_3d.xyz.shape[0]

            # Initialize gaussian colors.
            colors = tracks_3d.colors
            # Initialize gaussian scales: find the average of the three nearest
            # neighbors in the first frame for each point and use that as the
            # scale.
            dists, _ = knn(tracks_3d.xyz[:, cano_t], 3)
            dists = torch.from_numpy(dists)
            scales = dists.mean(dim=-1, keepdim=True)
            scales = scales.clamp(torch.quantile(scales, 0.05), torch.quantile(scales, 0.95))
            scales = torch.log(scales.repeat(1, 3))
            # Initialize gaussian means.
            means = tracks_3d.xyz[:, cano_t]
            # Initialize gaussian orientations as random.
            quats = torch.rand(num_fg, 4)
            # Initialize gaussian opacities.
            opacities = torch.logit(torch.full((num_fg,), 0.7)).unsqueeze(1) # logit <=> sigmoid
            dynamics = torch.ones(num_fg, 1) # fg: torch.ones
        gaussians = GaussianParams(means, quats, scales, colors, opacities, motion_coefs, dynamics) # MARK: fg
        return gaussians, idcs

    def init_bg(self, points, num_motion_bases):
        """
        using dataclasses instead of individual tensors so we know they're consistent
        and are always masked/filtered together
        """
        num_init_bg_gaussians = points.xyz.shape[0]
        bg_points_centered = points.xyz 
        bg_scene_center = points.xyz.mean(0)
        bg_points_centered = points.xyz - bg_scene_center
        bg_min_scale = bg_points_centered.quantile(0.05, dim=0)
        bg_max_scale = bg_points_centered.quantile(0.95, dim=0)
        bg_scene_scale = torch.max(bg_max_scale - bg_min_scale).item() / 2.0 # bg_scene_scale
        bkdg_colors = points.colors

        # Initialize gaussian scales: find the average of the three nearest
        # neighbors in the first frame for each point and use that as the
        # scale.
        dists, _ = knn(points.xyz, 3)
        dists = torch.from_numpy(dists)
        bg_scales = dists.mean(dim=-1, keepdim=True)
        bkdg_scales = torch.log(bg_scales.repeat(1, 3))

        bg_means = points.xyz

        # Initialize gaussian orientations by normals.
        local_normals = points.normals.new_tensor([[0.0, 0.0, 1.0]]).expand_as(
            points.normals
        )
        bg_quats = roma.rotvec_to_unitquat(
            F.normalize(local_normals.cross(points.normals, dim=-1), dim=-1)
            * (local_normals * points.normals).sum(-1, keepdim=True).acos_()
        ).roll(1, dims=-1)
        bg_opacities = torch.logit(torch.full((num_init_bg_gaussians,), 0.7)).unsqueeze(1)
        bg_motion_coefs = torch.ones(num_init_bg_gaussians, num_motion_bases).cuda() # 100000, 10; NOTE bg_motion_coefs
        dynamics = torch.zeros(num_init_bg_gaussians, 1) # bg: torch.zeros
        gaussians = GaussianParams(bg_means, bg_quats, bkdg_scales, bkdg_colors, bg_opacities, bg_motion_coefs, dynamics)

        return gaussians

    def training_setup(self, training_args): # Initialize the optimizer and the learning rate scheduler; training_args: opt
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # self.denom: self.denominator

        # self.spatial_lr_scale = 5. # FIXME
        print("Gaussian Cameras Extent =", self.spatial_lr_scale)
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._motion_coefs], 'lr': training_args.motion_coefs_lr, "name": "motion_coefs"},
            {'params': [self._dynamics], 'lr': 0.1, "name": "dynamics"},
            {'params': self.GTnet.parameters(), 'lr': training_args.gtnet_lr, "name": "GTnet"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


        # self.scale_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr * self.spatial_lr_scale,
        #                                             lr_final=0.00001 * self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            # if param_group["name"] == "scaling":
            #     lr = self.scale_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('dynamics')
        for i in range(self._motion_coefs.shape[1]):
            l.append('motion_coefs_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        dynamics = self._dynamics.detach().cpu().numpy()
        motion_coefs = self._motion_coefs.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, dynamics, motion_coefs, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self): # Densification
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1): # produce gaussian from point cloud data(.ply)
        self.og_number_points = og_number_points # the number of gaussian
        plydata = PlyData.read(path)    # read .ply file

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        dynamics = np.asarray(plydata.elements[0]["dynamics"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        motion_coefs_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion_coefs_")]
        motion_coefs = np.zeros((xyz.shape[0], len(motion_coefs_names)))
        for idx, attr_name in enumerate(motion_coefs_names):
            motion_coefs[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion_coefs = nn.Parameter(torch.tensor(motion_coefs, dtype=torch.float, device="cuda").requires_grad_(True))
        self._dynamics = nn.Parameter(torch.tensor(dynamics, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def fetchPly(self, path): # fetch points
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'],
                        vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        return BasicPointCloud(points=positions, colors=colors, normals=normals)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups: # xyz, r, s, density, f_dc, f_rest, motion_coefs, dynamics
            if group["name"] == name: # density
                stored_state = self.optimizer.state.get(group['params'][0], None) # get state dictionary
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]] # del state dictionary
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state # update state dictionary

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups: # xyz, r, s, density, f_dc, f_rest, motion_coefs, dynamics
            if group['name'] == "GTnet":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None) # get state dictionary
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]] # del state dictionary
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state # update state dictionary

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # prune gaussian 
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._motion_coefs = optimizable_tensors["motion_coefs"]
        self._dynamics = optimizable_tensors["dynamics"]

        # prune xyz_gradient_accm, denom, max_radii2D
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups: # xyz, r, s, density, f_dc, f_rest, motion_coefs, dynamics
            if group['name'] == "GTnet":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None) # get state dictionary
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]] # del state dictionary
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state # update state dictionary

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_motion_coefs, new_dynamics, selected_pts_mask=None, N=1, add_sfm_point=False):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "motion_coefs": new_motion_coefs,
             "dynamics": new_dynamics} # new gaussian attribute

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._motion_coefs = optimizable_tensors["motion_coefs"]
        self._dynamics = optimizable_tensors["dynamics"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # clear accumulating gradient
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if add_sfm_point:
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            return
        if selected_pts_mask is None: selected_pts_mask = torch.tensor([False]).repeat(self.max_radii2D.shape[0]).cuda()
        self.max_radii2D = torch.cat([self.max_radii2D, (self.max_radii2D[selected_pts_mask].repeat(N))/N], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2): # MARK: N=3
        n_init_points = self.get_xyz.shape[0] # number of point after clone
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda") # flatten gradient
        padded_grad[:grads.shape[0]] = grads.squeeze() # flatten gradient
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) # grad is larges
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent) # scale is large, 0.01 * 5
        # Increase the split Gaussian
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds) # N_point, 3
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1) # rotate quaternion -> rotation matrix
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)) # MARK: clever
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_motion_coefs = self._motion_coefs[selected_pts_mask].repeat(N, 1)
        new_dynamics = self._dynamics[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_motion_coefs, new_dynamics, selected_pts_mask, N=N)

        # Delete the original Gaussian
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) # grad is large
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent) # scale is small, 0.01 * camera_extent

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_motion_coefs = self._motion_coefs[selected_pts_mask]
        new_dynamics = self._dynamics[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_motion_coefs, new_dynamics, selected_pts_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration): # max_screen_size: size_threshold
        grads = self.xyz_gradient_accum / self.denom # [gs_num, 1]
        grads[grads.isnan()] = 0.0 # NaN -> 0.0

        gs_num = {}
        # densify
        clone = self.get_xyz.shape[0]
        self.densify_and_clone(grads, max_grad, extent) # under reconstruction
        gs_num['clone'] =  str(clone)+'->'+str(self.get_xyz.shape[0])
        split = self.get_xyz.shape[0]
        self.densify_and_split(grads, max_grad, extent) # over reconstruction
        gs_num['split'] =  str(split)+'->'+str(self.get_xyz.shape[0])

        # prune
        prune_mask = (self.get_opacity < min_opacity).squeeze() # min_opacity == 0.005
        gs_num['opacity_prune'] = str(prune_mask.sum().cpu().item())
        gs_num['radii2D'] = torch.all(self.max_radii2D == 0).cpu().item()
        print('max_radii2D: Mean={:.2f}, Max={:.2f}, Min={:.2f}'.format(self.max_radii2D.mean().item(), self.max_radii2D.max().item(), self.max_radii2D.min().item()))
        print('get_scale_x: Mean={:.2f}, Max={:.2f}, Min={:.2f}'.format(self.get_scaling[:,0].mean().item(), self.get_scaling[:,0].max().item(), self.get_scaling[:,0].min().item()))
        print('get_scale_y: Mean={:.2f}, Max={:.2f}, Min={:.2f}'.format(self.get_scaling[:,1].mean().item(), self.get_scaling[:,1].max().item(), self.get_scaling[:,1].min().item()))
        print('get_scale_z: Mean={:.2f}, Max={:.2f}, Min={:.2f}'.format(self.get_scaling[:,2].mean().item(), self.get_scaling[:,2].max().item(), self.get_scaling[:,2].min().item()))
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size                  # big points view sapce, strange: self.max_radii2D is always zero?
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent   # big points world space
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            gs_num['scale'] = str(big_points_ws.sum().cpu().item())
        self.prune_points(prune_mask)
        
        fg = self.get_dynamics.detach().bool().squeeze() # 107392, 1;
        bg = ~fg
        print('fg_gs={}, bg_gs={}'.format(fg.sum().item(), bg.sum().item()))

        torch.cuda.empty_cache()
        return gs_num

    def add_densification_stats(self, viewspace_point_tensor, update_filter): # update_filter: visibility_filter
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True) # viewspace_point_tensor: x, y
        self.denom[update_filter] += 1

    def densify_from_pcd(self, source_path): # I: uselsee
        ply_path = os.path.join(source_path, "sparse_/points3D.ply")
        try:
            pcd = self.fetchPly(ply_path)
        except:
            ply_path = os.path.join(source_path, "sparse_/0/points3D.ply")
            pcd = self.fetchPly(ply_path)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Re-import the scene point cloud:", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2 * 1e-7))[..., None].repeat(1, 3)
        scales =  torch.log(torch.ones_like(torch.from_numpy(np.asarray(pcd.points))) * 1e-4).cuda()
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        dynamics = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        new_xyz = fused_point_cloud
        new_features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        new_features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        new_opacities = opacities
        new_scaling = scales
        new_rotation = rots
        new_dynamic = dynamics

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_dynamic, add_sfm_point=True)

    def depth_reinit(self, scene, render_depth, iteration, num_depth, args, pipe, background, deform):

        out_pts_list = []
        gt_list = []
        views = scene.getTrainCameras().copy()
        smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        for view in views:
            gt = view.original_image[0:3, :, :]
            if iteration < 3000: # warm_up == 3000
                d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            else:
                N = self.get_xyz.shape[0]  # current gaussian quantity, eg: 10587
                time_input = view.fid.unsqueeze(0).expand(N, -1) # Expand the input in the time dimension, eg:torch.Size([10587,1])
                time_interval = 1 / (randint(0, len(views)-1) + 1)
                ast_noise = torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration) # for time_input; why do this?
                d_xyz, d_rotation, d_scaling, _ = deform.step(self.get_xyz.detach(), time_input + ast_noise) # MARK: .detach()

            render_depth_pkg = render_depth(view, self, pipe, background, d_xyz, d_rotation, d_scaling)

            out_pts = render_depth_pkg["out_pts"] # 3, 400, 940
            accum_alpha = render_depth_pkg["accum_alpha"] # 1, 400, 940

            prob = 1 - accum_alpha
            prob = prob / prob.sum()
            prob = prob.reshape(-1).cpu().numpy()

            factor = 1 / (gt.shape[1]*gt.shape[2]*len(views) / num_depth)

            N_xyz = prob.shape[0] # 400 * 940
            num_sampled = int(N_xyz*factor)

            indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)
            
            out_pts = out_pts.permute(1,2,0).reshape(-1,3)
            gt = gt.permute(1,2,0).reshape(-1,3)

            out_pts_list.append(out_pts[indices])
            gt_list.append(gt[indices])       

        out_pts_merged=torch.cat(out_pts_list) # 59670, 3
        gt_merged=torch.cat(gt_list) # 59670, 3

        return out_pts_merged, gt_merged

    def reinitial_pts(self, pts, rgb):

        fused_point_cloud = pts
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        dynamics = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._dynamic = nn.Parameter(dynamics.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  

    def add_depth_points(self, pts, rgb):

        fused_point_cloud = pts
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        scales =  torch.log(torch.ones_like(pts) * 0.1).cuda() # Empirically 0.1
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        dynamics = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")

        opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # large opacity 0.9

        new_xyz = fused_point_cloud
        new_features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        new_features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        new_opacities = opacities
        new_scaling = scales
        new_rotation = rots
        new_dynamic = dynamics

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_dynamic, add_sfm_point=True)
        
    def depth_reinit_som(self, scene, render_depth_som, iteration, num_depth, args, pipe, background, motion_model):

        out_pts_list = []
        gt_list = []
        views = scene.getTrainCameras().copy()
        
        with torch.no_grad():
            fg = self.get_dynamics.bool().squeeze() # 107392, 1;
            indices = torch.nonzero(fg).squeeze() 
            means = self.get_xyz.detach()          # 107392, 3
            quats = self.get_rotation.detach()     # 107392, 4
            coefs = self.get_motion_coefs.detach() # 107392, 10
            fg_means = means[indices]   # 7392, 3
            fg_quats = quats[indices]   # 7392, 4
            fg_coefs = coefs[indices]   # 7392, 10

            for view in views:
                gt = (view.original_image[0:3, :, :]).permute(1,2,0).reshape(-1,3)  # h*w, 3
                gt_mask = view.motion_mask.bool().reshape(-1) # h*w

                transfms = motion_model.compute_transforms(torch.tensor([view.uid]).cuda().float(), fg_coefs)  # 7392, 1, 3, 4 

                new_fg_means = torch.einsum( # 7392, 1, 3
                    "pnij,pj->pni",
                    transfms,
                    F.pad(fg_means, (0, 1), value=1.0),
                )
                new_fg_quats = roma.quat_xyzw_to_wxyz( # 7392, 1, 4 
                    (
                        roma.quat_product(
                            roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                            roma.quat_wxyz_to_xyzw(fg_quats[:, None]),
                        )
                    )
                )
                new_fg_quats = F.normalize(new_fg_quats, p=2, dim=-1) # 7392, 1, 4 

                means_clone = means.clone() # 107392, 3
                quats_clone = quats.clone() # 107392, 4
                means_clone[indices] = new_fg_means.squeeze(1) # update fg means
                quats_clone[indices] = new_fg_quats.squeeze(1) # update fg quats


                # render_depth_pkg = render_depth_som(view, self, pipe, background, motion_model, indices)
                render_depth_pkg = render_depth_som(view, self, pipe, background, means_clone, quats_clone)

                out_pts = render_depth_pkg["out_pts"].permute(1,2,0).reshape(-1,3) # 400 * 940, 3
                # out_pts = render_depth_pkg["render"].permute(1,2,0).reshape(-1,3) # 400 * 940, 3
                accum_alpha = render_depth_pkg["accum_alpha"].permute(1,2,0).reshape(-1,1) # 400 * 940, 1

                prob = 1 - accum_alpha
                dynamic_out_pts = out_pts[gt_mask] # n, 3
                dynamic_gt = gt[gt_mask]
                dynamic_prob = prob[gt_mask].cpu().numpy() # n, 1
                dynamic_prob = dynamic_prob / dynamic_prob.sum()
                dynamic_prob = dynamic_prob.reshape(-1)

                if dynamic_out_pts.shape[0] > 2000:
                    indices_ = np.random.choice(dynamic_out_pts.shape[0], size=2000, p=dynamic_prob, replace=False)
                    # indices = np.random.choice(dynamic_out_pts.shape[0], size=1000, replace=False)
                else:
                    indices_ = np.arange(dynamic_out_pts.shape[0])

                out_pts_list.append(dynamic_out_pts[indices_])
                gt_list.append(dynamic_gt[indices_])       

            out_pts_merged=torch.cat(out_pts_list) # 59670, 3
            gt_merged=torch.cat(gt_list) # 59670, 3

        return out_pts_merged, gt_merged 

    def depth_reinit_som_new(self, scene, render_depth_som, iteration, num_depth, args, pipe, background, motion_model):

        out_pts_list = []
        gt_list = []
        views = scene.getTrainCameras().copy()
        
        with torch.no_grad():
            fg = self.get_dynamics.bool().squeeze() # 107392, 1;
            indices = torch.nonzero(fg).squeeze() 
            means = self.get_xyz.detach()          # 107392, 3
            coefs = self.get_motion_coefs.detach() # 107392, 10
            fg_means = means[indices]   # 7392, 3
            fg_coefs = coefs[indices]   # 7392, 10

            for view in views:
                gt = (view.original_image[0:3, :, :]).permute(1,2,0)  # h, w, 3
                gt_mask = view.motion_mask.bool() # h, w
                gt_depth = view.depth # h, w

                H, W = gt_depth.shape
                grid = torch.stack(
                    torch.meshgrid(
                        torch.arange(W, dtype=torch.float32),
                        torch.arange(H, dtype=torch.float32),
                        indexing="xy",
                    ),
                    dim=-1,
                ).cuda()

                bool_mask = (gt_mask) * (gt_depth > 0).to(torch.bool) # valid dynamic mask
                R, T, K = view.R, view.T, view.K # R in c2w, T in w2c
                matrix = np.hstack((np.transpose(R),T.reshape(3,1)))
                w2c = np.vstack((matrix, np.array([0.,0.,0.,1.])))
                K, w2c = K.cuda(), torch.tensor(w2c).cuda()
                points = ( # n_i, 3
                    torch.einsum(
                        "ij,pj->pi",
                        torch.linalg.inv(K),
                        F.pad(grid[bool_mask], (0, 1), value=1.0),
                    )
                    * gt_depth[bool_mask][:, None]
                )
                points = torch.einsum( # n_i, 3
                    "ij,pj->pi", torch.linalg.inv(w2c.float())[:3], F.pad(points, (0, 1), value=1.0)
                )
                dynamic_gt = gt[bool_mask] # n_i, 3;

                # observe frame -> canonical frame
                transfms = motion_model.compute_transforms(torch.tensor([view.uid]).cuda().float(), fg_coefs)  # 7392, 1, 3, 4 

                new_fg_means = torch.einsum( # 7392, 1, 3
                    "pnij,pj->pni",
                    transfms,
                    F.pad(fg_means, (0, 1), value=1.0),
                )

                # nearest indices
                distances = torch.cdist(points, new_fg_means[:,0,:], p=2) # n_i, n_fg
                nearest_indices = torch.argmin(distances, dim=1) # n_i
                # transferm
                new_transfms = transfms[nearest_indices] # n_i, 1, 3, 4
                # real mean
                R_can2obs = new_transfms[:, 0, :, :3] # n_i, 3, 3
                t_can2obs = new_transfms[:, 0, :, 3] # n_i, 3
                R_obs2can = torch.inverse(R_can2obs)
                dynamic_out_pts = torch.matmul(R_obs2can, (points - t_can2obs).unsqueeze(-1)).squeeze(-1) # n_i, 3

                if dynamic_out_pts.shape[0] > 2000:
                    indices_ = np.random.choice(dynamic_out_pts.shape[0], size=1000, replace=False)
                else:
                    indices_ = np.arange(dynamic_out_pts.shape[0])

                out_pts_list.append(dynamic_out_pts[indices_])
                gt_list.append(dynamic_gt[indices_])       

            out_pts_merged=torch.cat(out_pts_list) # 59670, 3
            gt_merged=torch.cat(gt_list) # 59670, 3

        return out_pts_merged, gt_merged 

    def add_depth_points_som(self, pts, rgb):

        fused_point_cloud = pts
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        scales =  torch.log(torch.ones_like(pts) * 0.5).cuda() # Empirically 0.1
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        dynamics = torch.ones((fused_point_cloud.shape[0], 1), device="cuda")
        motion_coefs = torch.rand((fused_point_cloud.shape[0], 20), device="cuda")

        opacities = inverse_sigmoid(0.7 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # large opacity 0.9

        new_xyz = fused_point_cloud
        new_features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        new_features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        new_opacities = opacities
        new_scaling = scales
        new_rotation = rots
        new_dynamic = dynamics
        new_motion_coefs = motion_coefs

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_motion_coefs, new_dynamic, add_sfm_point=True)
