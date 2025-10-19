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

import roma
import time
import torch
import torch.nn.functional as F
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           train=1, lambda_s=0.01, lambda_p=0.01, max_clamp=1.1, scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    deformable_scale = 1.0
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0 # [gs_num, 3]; dtype == torch.float32
    try:
        screenspace_points.retain_grad() # retain gradient
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier, # defalut: 1.
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform, # MARK: projection matrix
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False: # judge whether warm_up
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz * deformable_scale # deformable_scale; [gs_num, 3]
    means2D = screenspace_points # [gs_num, 3]
    opacity = pc.get_opacity # [gs_num, 1]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier) # the only use of get_covariance
    else:
        scales = pc.get_scaling + d_scaling * deformable_scale # deformable_scale; [gs_num, 3]
        rotations = pc.get_rotation + d_rotation # [gs_num, 4]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)             # torch.Size([gs_num, 3, 16])
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))      # torch.Size([gs_num, 3])
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)                                   # torch.Size([gs_num, 3])
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)                              # torch.Size([gs_num, 3])
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)                                             # torch.Size([gs_num, 3])
        else:
            shs = pc.get_features   # torch.Size([gs_num, 16, 3])
    else:
        colors_precomp = override_color

    dynamic_precomp = torch.cat((pc.get_dynamics.detach(), torch.zeros_like(pc.get_dynamics.detach()).expand(-1, 2)), dim=-1) # (G, 3)
    # dynamic_means2D = torch.zeros_like(pc.get_dynamics, dtype=pc.get_dynamics, requires_grad=True, device="cuda") + 0 # [gs_num, 3]; dtype == torch.float32
    # dynamic_means2D.retain_grad() # retain gradient

    if not train:  # testing time
        rendered_image, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        dynamic_mask, *_ = rasterizer(
            means3D=means3D,
            means2D=means2D, # TODO: process
            shs=None,
            colors_precomp=dynamic_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        
        return {"render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth": depth,
        "sharp_image": rendered_image,
        "dynamic_mask": dynamic_mask[0, ...],}

    else: # training time
        with torch.no_grad():
            sharp_image, _, _ = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

        _pos = means3D.detach()
        _scales = scales.detach()
        _rotations = rotations.detach()
        _viewdirs = viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)

        scales_delta, rotations_delta, pos_delta = pc.GTnet(_pos, _scales, _rotations, _viewdirs)
        scales_delta = torch.clamp(lambda_s * scales_delta + (1-lambda_s), min=1.0, max=max_clamp)
        rotations_delta = torch.clamp(lambda_s * rotations_delta + (1-lambda_s), min=1.0, max=max_clamp)
        # print(scales_delta.mean().item(),scales_delta.std().item(),'s')
        # print(rotations_delta.mean().item(),rotations_delta.std().item(),'r')
        transformed_scales = scales * scales_delta
        transformed_rotations = rotations * rotations_delta

        rendered_image, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = transformed_scales,
            rotations = transformed_rotations,
            cov3D_precomp = cov3D_precomp)
        
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": depth,
                "sharp_image": sharp_image}

def render_depth(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, \
    d_xyz, d_rotation, d_scaling, scaling_modifier = 1.0, override_color = None, culling=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    res  = rasterizer.render_depth(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    return res

def render_som(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, motion_model,
           train=1, lambda_s=0.01, lambda_p=0.01, max_clamp=1.1, scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0 # [gs_num, 3]; dtype == torch.float32
    try:
        screenspace_points.retain_grad() # retain gradient
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier, # defalut: 1.
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform, # MARK: projection matrix
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # calculate means3D
    fg = pc.get_dynamics.detach().bool().squeeze() # 107392, 1;
    indices = torch.nonzero(fg).squeeze() # fg index
    # print("Foreground Gaussians:", len(indices)) # 7392

    deformation_t_start = time.time()
    means = pc.get_xyz          # 107392, 3
    quats = pc.get_rotation     # 107392, 4
    coefs = pc.get_motion_coefs # 107392, 10
    fg_means = means[indices]   # 7392, 3
    fg_quats = quats[indices]   # 7392, 4
    fg_coefs = coefs[indices]   # 7392, 10
    # transfms = motion_model.compute_transforms(viewpoint_camera.fid, fg_coefs)  # 7392, 1, 3, 4 
    transfms = motion_model.compute_transforms(torch.tensor([viewpoint_camera.uid]).cuda().float(), fg_coefs)  # 7392, 1, 3, 4 

    fg_means = torch.einsum( # 7392, 1, 3
        "pnij,pj->pni",
        transfms,
        F.pad(fg_means, (0, 1), value=1.0),
    )
    fg_quats = roma.quat_xyzw_to_wxyz( # 7392, 1, 4 
        (
            roma.quat_product(
                roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                roma.quat_wxyz_to_xyzw(fg_quats[:, None]),
            )
        )
    )
    fg_quats = F.normalize(fg_quats, p=2, dim=-1) # 7392, 1, 4 

    means_clone = means.clone() # 107392, 3
    quats_clone = quats.clone() # 107392, 4
    means_clone[indices] = fg_means.squeeze(1) # update fg means
    quats_clone[indices] = fg_quats.squeeze(1) # update fg quats
    deformation_t_end = time.time()
    deformation_t = (deformation_t_end - deformation_t_start)

    means3D = means_clone
    # means3D = means
    means2D = screenspace_points # 107392, 3
    opacity = pc.get_opacity # 107392, 1

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier) # the only use of get_covariance
    else:
        scales = pc.get_scaling # 107392, 3
        rotations = quats_clone # 107392, 4
        # rotations = quats

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)             # torch.Size([gs_num, 3, 16])
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))      # torch.Size([gs_num, 3])
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)                                   # torch.Size([gs_num, 3])
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)                              # torch.Size([gs_num, 3])
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)                                             # torch.Size([gs_num, 3])
        else:
            shs = pc.get_features # 107392, 16, 3
    else:
        colors_precomp = override_color

    dynamic_precomp = torch.cat((pc.get_dynamics.detach(), torch.zeros_like(pc.get_dynamics.detach()).expand(-1, 2)), dim=-1) # (G, 3)
    # dynamic_means2D = torch.zeros_like(pc.get_dynamics, dtype=pc.get_dynamics, requires_grad=True, device="cuda") + 0 # [gs_num, 3]; dtype == torch.float32
    # dynamic_means2D.retain_grad() # retain gradient

    if not train:  # testing time
        rasterizer_t_start = time.time()
        rendered_image, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        dynamic_mask, *_ = rasterizer(
            means3D=means3D,
            means2D=means2D, # TODO: process
            shs=None,
            colors_precomp=dynamic_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        rasterizer_t_end = time.time()
        rasterizer_t = (rasterizer_t_end - rasterizer_t_start)
        
        return {"render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth": depth,
        "sharp_image": rendered_image,
        "dynamic_mask": dynamic_mask[0, ...],
        "deformation_time": deformation_t,
        "rasterizer_time": rasterizer_t,}
        # "dynamic_mask": depth,}

    else: # training time
        with torch.no_grad():
            sharp_image, _, _ = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

        _pos = means3D.detach()
        _scales = scales.detach()
        _rotations = rotations.detach()
        _viewdirs = viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)

        scales_delta, rotations_delta, pos_delta = pc.GTnet(_pos, _scales, _rotations, _viewdirs)
        scales_delta = torch.clamp(lambda_s * scales_delta + (1-lambda_s), min=1.0, max=max_clamp)
        rotations_delta = torch.clamp(lambda_s * rotations_delta + (1-lambda_s), min=1.0, max=max_clamp)
        # print(scales_delta.mean().item(),scales_delta.std().item(),'s')
        # print(rotations_delta.mean().item(),rotations_delta.std().item(),'r')
        transformed_scales = scales * scales_delta
        transformed_rotations = rotations * rotations_delta

        rendered_image, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = transformed_scales,
            rotations = transformed_rotations,
            cov3D_precomp = cov3D_precomp)
        
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": depth,
                "sharp_image": sharp_image}

def render_depth_som(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, \
    means_clone, quats_clone, scaling_modifier = 1.0, override_color = None, culling=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = means_clone
    # means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling + d_scaling
        # rotations = pc.get_rotation + d_rotation
        scales = pc.get_scaling # 107392, 3
        rotations = quats_clone # 107392, 4

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    res  = rasterizer.render_depth(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    return res