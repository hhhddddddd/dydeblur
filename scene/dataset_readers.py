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
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
import open3d as o3d
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
from utils.normal_utils import normal_from_depth_image
from utils.colmap_utils import get_colmap_camera_params, parse_tapir_track_info, normalize_coords
from mvs_modules.mvs_estimator import MvsEstimator

class CameraInfo(NamedTuple):
    uid: int            # index, Intrinsics
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float          # frame time
    depth: Optional[np.array] = None
    depth_no_sc: Optional[np.array] = None
    depthv2: Optional[np.array] = None
    blur_map: Optional[np.array] = None
    motion_mask: Optional[np.array] = None

    K: np.array = None
    bounds: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    sc: float = 1.
    foreground_points: tuple = ()
    background_points: tuple = ()
    Ks: torch.Tensor = None
    w2cs: torch.Tensor = None


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose

def farthest_point_sample(xyz, npoint):
    """
    Farthest point sampling

    Input:
        xyz: pointcloud data, [B, N, C], B: batchsize, N: the number of point in one bitchsize, C: the number of characteristic in one point
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # 1, 512
    distance = torch.ones(B, N).to(device).to(torch.float64) * 1e10 # 1, 100000
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1) # B N
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers) # stack
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T) # 4, 4
        C2W = np.linalg.inv(W2C) # Computed inverse matrix; 4, 4
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers) # center: mean_camera_center, diagonal: max distance between mean_camera_center and camera
    radius = diagonal * 1.1

    translate = -center # NOTE "-center"

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]              # Extrinsics
        intr = cam_intrinsics[extr.camera_id]   # Intrinsics 
        height = intr.height
        width = intr.width

        uid = intr.id                           # Intrinsics 
        R = np.transpose(qvec2rotmat(extr.qvec))# NOTE
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0] # IMG_4026
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path, sc=1.): # fetch points
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    positions *= sc
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0 # NOTE 255 in .ply, raw rgb is zero!
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetch_slm_Ply(path, sc=1.): # fetch points
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    positions *= sc
    colors = np.vstack([vertices['f_dc_0'], vertices['f_dc_1'],
                   vertices['f_dc_2']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None): # store points
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), # f4: np.float32, u1: np.uint8
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval: # divide the training set and test set
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos) # average camera center, diagonal

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"): # path: args.source_path
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"])) # frame["transform_matrix"]: c2w (OpenGL/Blender), matrix: w2c
            R = -np.transpose(matrix[:3, :3]) # OpenGL/Blender -> Colmap
            R[:, 0] = -R[:, 0] # OpenGL/Blender -> Colmap
            T = -matrix[:3, 3] # why negative?

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem # image_name: r_000
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA")) # 800, 800, 4

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4] # mask

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4]) # change background
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx # NOTE Why? H = W = 800 ?
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"): # nerf_synthetic, D_NeRF
    print("Reading Training Transforms") # MARK: transforms_train.json
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms") # MARK: transforms_test.json
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)

    if not eval: # No evaluation
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos) # average camera center, diagonal

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path): # MARK: random points
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path) # MARK: points3d.ply
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:      # MARK: scene.json
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:   # MARK: metadata.json
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:    # MARK: dataset.json
        dataset_json = json.load(f)

    coord_scale = scene_json['scale'] # 0.04
    scene_center = scene_json['center'] # 0.7335441453210703, 5.389038591423233, 10.511067242424215

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        all_img = train_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center # camera['position']: -2.001690149307251, 1.6647430658340454, -1.7718678712844849
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5) # n_cameras, 3, 5
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1) # llff coordinate (DRB) -> opengl nerf coordinate (RUB); n_cameras, 3, 4
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1) # homogeneous coordinates
    poses = poses @ np.diag([1, -1, -1, 1]) # opengl nerf coordinate (RUB) -> opencv colmap (RDF); n_cameras, 4, 4

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w)) # w2c
        R = np.transpose(matrix[:3, :3]) # R in c2w
        T = matrix[:3, 3] # T in w2c

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0]):
    print("Reading Training Camera")
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
                                         num_images=num_images)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(
        path, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0) # camera center
    vec2 = normalize(poses[:, :3, 2].sum(0)) # Z-axis
    up = poses[:, :3, 1].sum(0) # Up-axis
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def recenter_poses(poses): # poses: 48, 3, 5

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses) # 3, 5
    c2w = np.concatenate([c2w[:3,:4], bottom], -2) # 4, 4
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1]) # 48, 1, 4
    poses = np.concatenate([poses[:,:3,:4], bottom], -2) # 48, 4, 4

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def readD2RFCameras(path, camera_scale=-1):

    # high = True
    high = False
    poses_arr = np.load(os.path.join(path, 'poses_bounds.npy')) # MARK: load camera pose
    
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3, 5, 34; poses[:,4,0]: 800., 1880., 723.199
    bds = poses_arr[:, -2:].transpose([1,0]) # 2, 34
    
    if high:
        imgdir = os.path.join(path, 'images_bokeh') # './data/D2RF/Car/images_bokeh'    
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')] # image path (left camera blur image & right camera blur image)
    else:
        imgdir = os.path.join(path, 'images_2') # './data/D2RF/Car/images_2'    
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')] # image path (left camera blur image & right camera blur image)
    imgdir_sharp = os.path.join(path, 'images') # './data/D2RF/Car/images'    
    imgfiles_sharp = [os.path.join(imgdir_sharp, f) for f in sorted(os.listdir(imgdir_sharp)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')] # image path (left camera sharp image & right camera sharp image)  
    depth_dir = os.path.join(path, 'dpt') # './data/D2RF/Car/dpt'       
    depth_files = [os.path.join(depth_dir, f) for f in sorted(os.listdir(depth_dir)) if f.endswith('npy')] # image depth 
    depth_fine_dir = os.path.join(path, 'dpt') # depth anything v2
    depth_fine_files = [os.path.join(depth_fine_dir, f) for f in sorted(os.listdir(depth_fine_dir)) if f.endswith('npy')] # depth map path
    blur_map_dir = os.path.join(path, 'blur_masks_npy') # DMENet
    blur_map_files = [os.path.join(blur_map_dir, f) for f in sorted(os.listdir(blur_map_dir)) if f.endswith('npy')] # blur map path
    motion_mask_dir = os.path.join(path, 'motion_masks') # motion mask
    motion_mask_files = [os.path.join(motion_mask_dir, f) for f in sorted(os.listdir(motion_mask_dir)) if f.endswith('png')] # motion mask path

    sh = imageio.imread(imgfiles[0]).shape # 400, 940, 3
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # change height and width in poses; poses[:,4,0]: 400., 940., 723.199
    if high:
        poses[2, 4, :] = poses[2, 4, :] * 1./ 1. # change focal, image -> image_2; poses[:,4,0]: 800., 1280., 723.199
    else:
        poses[2, 4, :] = poses[2, 4, :] * 1./ 2. # change focal, image -> image_2; poses[:,4,0]: 400., 940., 361.599
    
    poses = np.concatenate([poses[:, 1:2, :], 
                            -poses[:, 0:1, :], 
                            poses[:, 2:, :]], 1) # llff (DRB) -> nerf (RUB)
    poses = np.concatenate([poses[:, 0:1, :], 
                            -poses[:, 1:3, :], 
                            poses[:, 3:, :]], 1) # nerf (RUB) -> colmap (RDF)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # 34, 3, 5
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)     # 34, 2

    if camera_scale == -1.:
        bd_factor = 0.9
        sc =  1./(np.percentile(bds[:, 0], 5) * bd_factor)
    else:
        sc = camera_scale
    poses[:,:3,3] *= sc # change camera center
    bds *= sc
    print('sc =',sc)

    # poses = recenter_poses(poses) # FIXME Is it necessary?

    height = poses[0,0,-1]     # height: 400
    width = poses[0,1,-1]      # width: 940
    focal = poses[0,2,-1]      # focal: 361.599
                 
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    K = np.array([
        [focal, 0., width/2],
        [0., focal, height/2],
        [0., 0., 1.]
    ], dtype=np.float32)

    cam_infos = []
    for idx, img_path in enumerate(imgfiles):
        # img = imageio.imread(img_path)[...,:3]/255.  # 400, 940, 3; MARK: load image
        if idx % 2 == 0:
            img = Image.open(img_path)
        else: 
            img_path = imgfiles_sharp[idx]
            img = Image.open(img_path) # 0~255
            img = img.resize((sh[1],sh[0])) # resize sharp
        
        image_name = os.path.basename(img_path).split(".")[0] # 000000_left

        depth = np.load(depth_files[idx])
        depth = cv.resize(depth, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # Disparity, 400, 940
        depthv2 = np.load(depth_fine_files[idx])
        depthv2 = cv.resize(depthv2, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # depth anything v2
        blur_map = np.load(blur_map_files[idx]) # small is sharp, 400, 940
        blur_map = cv.resize(blur_map, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST)
        blur_map = 1 - (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min()) # large is sharp
        motion_mask = np.array(Image.open(motion_mask_files[idx]))
        motion_mask = cv.resize(motion_mask, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST)
        motion_mask = np.clip(motion_mask, 0.0, 1.0) # large is dynamic

        uid =  idx // 2             # colmap_id,    unsureness
        fid = (idx // 2) / (len(poses) // 2)

        c2w = poses[idx, :3, :4]          
        bottom = np.reshape([0,0,0,1.], [1,4])
        c2w = np.concatenate([c2w[:3,:4], bottom], -2)  # c2w, homogeneous coordinates
        matrix = np.linalg.inv(np.array(c2w))           # w2c, homogeneous coordinates
        R = np.transpose(matrix[:3, :3])            # R in c2w
        T = matrix[:3, 3]                           # T in w2c
        
        bounds = bds[idx]

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, bounds=bounds, FovY=np.array(FovY), FovX=np.array(FovX), image=img, depth=depth, depthv2=depthv2, blur_map=blur_map,
                              motion_mask=motion_mask, image_path=img_path, image_name=image_name, width=int(width), height=int(height), fid=fid)
        cam_infos.append(cam_info)
    return cam_infos, sc
    
def readD2RFDataset(path, camera_scale=-1, eval = True, llffhold = 2):

    cam_infos_unsorted, sc = readD2RFCameras(path, camera_scale)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval: # divide the training set and test set
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos) # average camera center, diagonal
 
    depth_point = False
    # depth_point =True
    if depth_point:
        print("depth point!")
        ply_path = os.path.join(path, "depth_rgb_True.ply")
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        print("sfm point!")
        ply_path = os.path.join(path, "sparse_/0/points3D.ply")
        bin_path = os.path.join(path, "sparse_/0/points3D.bin")
        txt_path = os.path.join(path, "sparse_/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    pcd_max = pcd.points.max(0)
    pcd_min = pcd.points.min(0)
    pcd_num = pcd.points.shape[0]
    print('final_pcd_max =', pcd_max)
    print('final_pcd_min =', pcd_min)
    print('final_pcd_num =', pcd_num)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           sc=sc)
    return scene_info

def readDyBluRFCameras(path, camera_scale=-1):

    # high = True
    high = False
    poses_arr = np.load(os.path.join(path, 'poses_bounds.npy')) # 48, 17; MARK: camera pose

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0]) # 3, 5, 48; poses[:,4,0]: 696., 1239., 716.985
    bds = poses_arr[:, -2:].transpose([1, 0]) # 2, 48
    
    if high:
        imgdir = os.path.join(path, 'images') # blur image path, MARK: train image
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    else:
        imgdir = os.path.join(path, 'images_512x288') # blur image path, MARK: train image
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgdir_inference = os.path.join(path, 'inference_images') # sharp image path, MARK: test image
    imgfiles_inference = [os.path.join(imgdir_inference, f) for f in sorted(os.listdir(imgdir_inference)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    depth_dir = os.path.join(path, 'disp') # disp
    depth_files = [os.path.join(depth_dir, f) for f in sorted(os.listdir(depth_dir)) if f.endswith('npy')] # depth map path
    depth_fine_dir = os.path.join(path, 'disp') # depth anything v2
    depth_fine_files = [os.path.join(depth_fine_dir, f) for f in sorted(os.listdir(depth_fine_dir)) if f.endswith('npy')] # depth map path
    blur_map_dir = os.path.join(path, 'blur_masks_npy') # DMENet
    blur_map_files = [os.path.join(blur_map_dir, f) for f in sorted(os.listdir(blur_map_dir)) if f.endswith('npy')] # blur map path
    motion_mask_dir = os.path.join(path, 'motion_masks') # motion mask
    motion_mask_files = [os.path.join(motion_mask_dir, f) for f in sorted(os.listdir(motion_mask_dir)) if f.endswith('png')] # motion mask path

    sh = imageio.imread(imgfiles[0]).shape # 288, 512, 3
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # change height and width; poses[:,4,0]: 288., 512., 716.985
    if high:
        poses[2, 4, :] = poses[2, 4, :] * 1. / 1.0 # change focal; poses[:,4,0]: 720., 1280., 716.985
    else:
        poses[2, 4, :] = poses[2, 4, :] * 1. / 2.5 # change focal; poses[:,4,0]: 288., 512., 286.794
    
    poses = np.concatenate([poses[:, 1:2, :], 
                            -poses[:, 0:1, :], 
                            poses[:, 2:, :]], 1)            # llff (DRB) -> nerf (RUB)
    poses = np.concatenate([poses[:, 0:1, :], 
                            -poses[:, 1:3, :], 
                            poses[:, 3:, :]], 1)            # nerf (RUB) -> colmap (RDF)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)    # 48, 3, 5
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)        # 48, 2

    if camera_scale == -1.:
        bd_factor = 0.9
        sc =  1./(np.percentile(bds[:, 0], 5) * bd_factor)
    else:
        sc = camera_scale
    poses[:,:3,3] *= sc # change camera center
    bds *= sc
    print('sc =',sc)

    # poses = recenter_poses(poses) # FIXME Is it necessary?

    height = poses[0,0,-1]     # height: 288
    width = poses[0,1,-1]      # width: 512 
    focal = poses[0,2,-1]      # focal: 286.794
                 
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    K = np.array([
        [focal, 0., width/2],
        [0., focal, height/2],
        [0., 0., 1.]
    ], dtype=np.float32)

    cam_infos = []
    for idx in range(len(poses)): 
        if idx % 2 == 0:
            img_path = imgfiles[idx // 2] # MARK: train image
            depth = np.load(depth_files[idx // 2])
            depth = cv.resize(depth, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # disparity, 288, 512
            depthv2 = np.load(depth_fine_files[idx // 2])
            depthv2 = cv.resize(depthv2, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # depth anything v2
            blur_map = np.load(blur_map_files[idx // 2]) # small is sharp, 288, 512
            blur_map = cv.resize(blur_map, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST)
            blur_map = 1 - (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min()) # large is sharp
            motion_mask = np.array(Image.open(motion_mask_files[idx // 2]))
            motion_mask = cv.resize(motion_mask, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST)
            motion_mask = np.clip(motion_mask, 0.0, 1.0) # large is dynamic
        else:
            img_path = imgfiles_inference[idx // 2] # MARK: test image
            depth = None
            depthv2 = None
            blur_map = None
            motion_mask = None
        img = Image.open(img_path) # 288, 512, 3
        if img.size[0] != sh[1]:
            img = img.resize((sh[1],sh[0])) # resize sharp
        image_name = os.path.basename(img_path).split(".")[0] # 00000
        
        uid =  idx // 2             # colmap_id,    unsureness
        fid = (idx // 2) / (len(poses) // 2)

        c2w = poses[idx, :3, :4]                
        bottom = np.reshape([0,0,0,1.], [1,4])
        c2w = np.concatenate([c2w[:3,:4], bottom], -2)  # c2w, homogeneous coordinates
        matrix = np.linalg.inv(np.array(c2w))           # w2c, homogeneous coordinates
        R = np.transpose(matrix[:3, :3])            # R in c2w, np.transpose(matrix[:3, :3]) == c2w[:3, :3]
        T = matrix[:3, 3]                           # T in w2c
        
        bounds = bds[idx] # min: ~1.0, max: ~2.0

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, bounds=bounds, FovY=np.array(FovY), FovX=np.array(FovX), image=img, depth=depth, depthv2=depthv2, blur_map=blur_map,
                              motion_mask=motion_mask, image_path=img_path, image_name=image_name, width=int(width), height=int(height), fid=fid)
        cam_infos.append(cam_info)

    return cam_infos, sc

def readDyBluRFDataset(path, camera_scale=-1, eval = True, llffhold = 2):

    cam_infos_unsorted, sc = readDyBluRFCameras(path, camera_scale)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval: # divide the training set and test set
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos) # average camera center, diagonal

    depth_point = False
    # depth_point =True
    if depth_point:
        print("depth point!")
        ply_path = os.path.join(path, "depth_rgb_True.ply")
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        print("sfm point!")
        ply_path = os.path.join(path, "sparse_/points3D.ply") # NOTE sfm point cloud
        bin_path = os.path.join(path, "sparse_/points3D.bin")
        txt_path = os.path.join(path, "sparse_/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    '''
    # if not os.path.exists(os.path.join(path, "mvs.ply")): # NOTE MVS point cloud
    #     # Generate mvs point cloud
    #     mvs_estimator = MvsEstimator("./mvs_modules/configs/config_mvsformer.json")
    #     vertices, mvs_depths, masks = mvs_estimator.get_mvs_pts(train_cam_infos)
    #     # vertices, mvs_depths, masks = mvs_estimator.get_mvs_pts(test_cam_infos)
    #     torch.cuda.empty_cache()
    #     # for i, cam in enumerate(train_cam_infos):
    #     #     cam.mvs_depth = mvs_depths[i]
    #     #     cam.mvs_mask = masks[i]

    #     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    #     colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    #     normals = np.zeros_like(positions)

    #     # random down sample
    #     print('Points num: ', len(positions))
    #     pc_downsample = 0.1
    #     if pc_downsample < 1.0:
    #         random_idx = np.random.choice(positions.shape[0], int(positions.shape[0] * pc_downsample), replace=False)
    #         positions = positions[random_idx]
    #         colors = colors[random_idx]
    #         normals = normals[random_idx]

    #     # save points
    #     ply_path = os.path.join(path, "mvs.ply")
    #     storePly(ply_path, positions, colors)
    #     pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
    #     print(f"Initial points num: {positions.shape[0]}")
    #     del mvs_estimator
    # else: 
    #     ply_path = os.path.join(path, "mvs.ply")
    #     pcd = fetchPly(ply_path)

    # ply_path = os.path.join(path, "random.ply") # NOTE random point cloud
    # if not os.path.exists(ply_path):
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")

    #     ply_sfm_path = os.path.join(path, "sparse_/points3D.ply")
    #     ply_sfm_data = PlyData.read(ply_sfm_path)
    #     vertices = ply_sfm_data['vertex']
    #     points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    #     pcd_max = points.max(0) # points: n_points, 3
    #     pcd_min = points.min(0)
    #     dist_max = pcd_max - pcd_min
    #     print('sfm_pcd_max =', pcd_max)
    #     print('sfm_pcd_min =', pcd_min)
    #     print('sfm_pcd_dist =', dist_max)

    #     xyz = np.random.random((num_pts, 3)) * dist_max * 1.1 + pcd_min * 1.1      
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
    #         shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    
    
    # ply_path = os.path.join(path, "lsm.ply") # NOTE lsm point cloud
    # if not os.path.exists(ply_path):
    #     ply_sfm_path = os.path.join(path, "sparse_/points3D.ply")
    #     ply_sfm_data = PlyData.read(ply_sfm_path)
    #     vertices = ply_sfm_data['vertex']
    #     points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    #     pcd_sfm_max = points.max(0) # points: n_points, 3
    #     pcd_sfm_min = points.min(0)
    #     dist_sfm_max = pcd_sfm_max - pcd_sfm_min
    #     print('sfm_pcd_max =', pcd_sfm_max)
    #     print('sfm_pcd_min =', pcd_sfm_min)
    #     print('sfm_pcd_dist =', dist_sfm_max)
    
    #     ply_lsm_path = os.path.join(path, "lsm_raw.ply")
    #     ply_lsm_data = PlyData.read(ply_lsm_path)
    #     vertices = ply_lsm_data['vertex']
    #     points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    #     pcd_lsm_max = points.max(0) # points: n_points, 3
    #     pcd_lsm_min = points.min(0)
    #     dist_lsm_max = pcd_lsm_max - pcd_lsm_min
    #     print('lsm_pcd_max =', pcd_lsm_max)
    #     print('lsm_pcd_min =', pcd_lsm_min)
    #     print('lsm_pcd_dist =', dist_lsm_max)

    #     points = (points - pcd_lsm_min) / dist_lsm_max
    #     xyz = points * dist_sfm_max * 1.1 + pcd_sfm_min * 1.1 # xyz: n_points, 3
    #     colors = np.vstack([vertices['f_dc_0'], vertices['f_dc_1'],
    #                    vertices['f_dc_2']]).T / 255.0
    #     normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    #     # NOTE Farthest Point Sampling
    #     random_idx = farthest_point_sample(torch.tensor(xyz[np.newaxis,...]).cuda(), 100_000).squeeze(0).cpu().numpy()
    #     xyz = xyz[random_idx]
    #     colors = colors[random_idx]
    #     normals = normals[random_idx]

    #     pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)
        
    #     storePly(ply_path, xyz, colors * 255, normals)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    '''
    
    pcd_max = pcd.points.max(0)
    pcd_min = pcd.points.min(0)
    pcd_num = pcd.points.shape[0]
    print('final_pcd_max =', pcd_max)
    print('final_pcd_min =', pcd_min)
    print('final_pcd_num =', pcd_num)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           sc=sc)
    return scene_info

def readDeblur4DGSCameras(path, camera_scale=-1):

    imgdir = os.path.join(path, 'images') # blur image path, MARK: train image
    imgfiles_all = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles = imgfiles_all[::2]

    imgdir_inference = os.path.join(path, 'images_test') # sharp image path, MARK: test image
    imgfiles_inference_all = [os.path.join(imgdir_inference, f) for f in sorted(os.listdir(imgdir_inference)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles_inference = imgfiles_inference_all[1::2]

    depth_dir = os.path.join(path, 'flow3d_preprocessed/aligned_depth_anything_colmap') # disparity
    depth_files = [os.path.join(depth_dir, f) for f in sorted(os.listdir(depth_dir)) if f.endswith('npy')] # depth map path
    depth_files = depth_files[::2]
    # depth_fine_dir = os.path.join(path, 'disp') # depth anything v2
    # depth_fine_files = [os.path.join(depth_fine_dir, f) for f in sorted(os.listdir(depth_fine_dir)) if f.endswith('npy')] # depth map path
    # blur_map_dir = os.path.join(path, 'blur_masks_npy') # DMENet
    # blur_map_files = [os.path.join(blur_map_dir, f) for f in sorted(os.listdir(blur_map_dir)) if f.endswith('npy')] # blur map path
    # motion_mask_dir = os.path.join(path, 'motion_masks') # motion mask
    # motion_mask_files = [os.path.join(motion_mask_dir, f) for f in sorted(os.listdir(motion_mask_dir)) if f.endswith('png')] # motion mask path
    blur_map_files = depth_files
    motion_mask_files = depth_files

    frame_names_all = [f.split('/')[-1].split('.')[0] for f in imgfiles_all] # '00000', '00001', '00002'
    Ks, w2cs = get_colmap_camera_params( # 48, 4, 4
        os.path.join(path, "flow3d_preprocessed/colmap/sparse/"),
        [frame_name + ".png" for frame_name in frame_names_all],
    )
    # if path.split('/')[-2] == 'man':
    #     Ks[[10,11,22,23,34,35,46,47]] = Ks[[11,10,23,22,35,34,47,46]]
    #     w2cs[[10,11,22,23,34,35,46,47]] = w2cs[[11,10,23,22,35,34,47,46]]
    Ks = torch.from_numpy(Ks[:, :3, :3].astype(np.float32)) # 48, 3, 3; f_x=f_y=666.057, w/2=640, h/2=360
    if imageio.imread(imgfiles[0]).shape[0] == 720:
        Ks[:, :2] /= 1.0 # f_x=f_y=666.057, w/2=640, h/2=360; MARK: high resolution
    else:
        Ks[:, :2] /= 2.5 # f_x=f_y=266.423, w/2=256, h/2=144; MARK: low resolution
    w2cs = torch.from_numpy(w2cs.astype(np.float32)) # 48, 4, 4
    c2ws = w2cs.inverse() # 48, 4, 4

    if camera_scale == -1.:
        bd_factor = 0.9
        sc =  1./(10 * bd_factor)
    else:
        sc = camera_scale
    c2ws[:,:3,3] *= sc # change camera center
    print('sc =',sc)

    sh = imageio.imread(imgfiles[0]).shape # 288, 512, 3

    height = sh[0]      # height: 288
    width = sh[1]       # width: 512
    focal = Ks[0,0,0]   # focal: 266.423
                 
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    cam_infos = []
    for idx in range(len(c2ws)): 
        if idx % 2 == 0:
            img_path = imgfiles[idx // 2] # MARK: train image
            depth = np.load(depth_files[idx // 2])
            depth = cv.resize(depth, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # disparity; 288, 512
            depth[depth < 1e-3] = 1e-3
            depth = 1.0 / depth # disparity -> depth
            # depthv2 = np.load(depth_fine_files[idx // 2])
            # depthv2 = cv.resize(depthv2, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # depth anything v2
            # blur_map = np.load(blur_map_files[idx // 2]) # small is sharp, 288, 512
            # blur_map = 1 - (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min()) # large is sharp
            # motion_mask = np.array(Image.open(motion_mask_files[idx // 2]))
            # motion_mask = np.clip(motion_mask, 0.0, 1.0) # large is dynamic
            blur_map = np.ones(depth.shape) # 288, 512
            motion_mask = np.ones(depth.shape) # 288, 512
        else:
            img_path = imgfiles_inference[idx // 2] # MARK: test image
            # depth = None
            # depthv2 = None
            # blur_map = None
            # motion_mask = None
        img = Image.open(img_path) # 288, 512, 3
        if img.size[0] != sh[1]:
            img = img.resize((sh[1],sh[0])) # resize sharp
        image_name = os.path.basename(img_path).split(".")[0] # 00000
        
        uid =  idx // 2             # colmap_id,    unsureness
        fid = (idx // 2) / (len(c2ws) // 2)

        c2w = c2ws[idx]                         # c2w, homogeneous coordinates               
        matrix = np.linalg.inv(np.array(c2w))   # w2c, homogeneous coordinates
        R = np.transpose(matrix[:3, :3])        # R in c2w, np.transpose(matrix[:3, :3]) == c2w[:3, :3]
        T = matrix[:3, 3]                       # T in w2c
        K = Ks[idx]

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=np.array(FovY), FovX=np.array(FovX), image=img, depth=depth, blur_map=blur_map,
                              motion_mask=motion_mask, image_path=img_path, image_name=image_name, width=int(width), height=int(height), fid=fid)
        cam_infos.append(cam_info)

    return cam_infos, sc

def readDeblur4DGSCameras_depth(path, camera_scale=-1):

    imgdir = os.path.join(path, 'images') # blur image path, MARK: train image
    imgfiles_all = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles = imgfiles_all[::2]

    imgdir_inference = os.path.join(path, 'images_test') # sharp image path, MARK: test image
    imgfiles_inference_all = [os.path.join(imgdir_inference, f) for f in sorted(os.listdir(imgdir_inference)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles_inference = imgfiles_inference_all[1::2]

    depth_dir = "/home/xuankai/code/dydeblur/data/DyBluRF/stereo_blur_dataset/seesaw/dense/disp" # disparity
    depth_files = [os.path.join(depth_dir, f) for f in sorted(os.listdir(depth_dir)) if f.endswith('npy')] # depth map path

    blur_map_files = depth_files
    motion_mask_files = depth_files

    frame_names_all = [f.split('/')[-1].split('.')[0] for f in imgfiles_all] # '00000', '00001', '00002'
    Ks, w2cs = get_colmap_camera_params( # 48, 4, 4
        os.path.join(path, "flow3d_preprocessed/colmap/sparse/"),
        [frame_name + ".png" for frame_name in frame_names_all],
    )

    Ks = torch.from_numpy(Ks[:, :3, :3].astype(np.float32)) # 48, 3, 3; f_x=f_y=666.057, w/2=640, h/2=360
    if imageio.imread(imgfiles[0]).shape[0] == 720:
        Ks[:, :2] /= 1.0 # f_x=f_y=666.057, w/2=640, h/2=360; MARK: high resolution
    else:
        Ks[:, :2] /= 2.5 # f_x=f_y=266.423, w/2=256, h/2=144; MARK: low resolution
    w2cs = torch.from_numpy(w2cs.astype(np.float32)) # 48, 4, 4
    c2ws = w2cs.inverse() # 48, 4, 4

    if camera_scale == -1.:
        bd_factor = 0.9
        sc =  1./(10 * bd_factor)
    else:
        sc = camera_scale
    c2ws[:,:3,3] *= sc # change camera center
    print('sc =',sc)

    sh = imageio.imread(imgfiles[0]).shape # 288, 512, 3

    height = sh[0]      # height: 288
    width = sh[1]       # width: 512
    focal = Ks[0,0,0]   # focal: 266.423
                 
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    cam_infos = []
    for idx in range(len(c2ws)): 
        if idx % 2 == 0:
            img_path = imgfiles[idx // 2] # MARK: train image
            depth = np.load(depth_files[idx // 2])
            depth = cv.resize(depth, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # disparity; 288, 512

            blur_map = np.ones(depth.shape) # 288, 512
            motion_mask = np.ones(depth.shape) # 288, 512
        else:
            img_path = imgfiles_inference[idx // 2] # MARK: test image
        img = Image.open(img_path) # 288, 512, 3
        if img.size[0] != sh[1]:
            img = img.resize((sh[1],sh[0])) # resize sharp
        image_name = os.path.basename(img_path).split(".")[0] # 00000
        
        uid =  idx // 2             # colmap_id,    unsureness
        fid = (idx // 2) / (len(c2ws) // 2)

        c2w = c2ws[idx]                         # c2w, homogeneous coordinates               
        matrix = np.linalg.inv(np.array(c2w))   # w2c, homogeneous coordinates
        R = np.transpose(matrix[:3, :3])        # R in c2w, np.transpose(matrix[:3, :3]) == c2w[:3, :3]
        T = matrix[:3, 3]                       # T in w2c
        K = Ks[idx]

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=np.array(FovY), FovX=np.array(FovX), image=img, depth=depth, blur_map=blur_map,
                              motion_mask=motion_mask, image_path=img_path, image_name=image_name, width=int(width), height=int(height), fid=fid)
        cam_infos.append(cam_info)

    return cam_infos, sc

def readDeblur4DGSCameras_pose(path, camera_scale=-1):

    imgdir = os.path.join(path, 'images') # blur image path, MARK: train image
    imgfiles_all = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles = imgfiles_all[::2]

    imgdir_inference = os.path.join(path, 'images_test') # sharp image path, MARK: test image
    imgfiles_inference_all = [os.path.join(imgdir_inference, f) for f in sorted(os.listdir(imgdir_inference)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles_inference = imgfiles_inference_all[1::2]

    depth_dir = os.path.join(path, 'flow3d_preprocessed/aligned_depth_anything_colmap') # disparity
    depth_files = [os.path.join(depth_dir, f) for f in sorted(os.listdir(depth_dir)) if f.endswith('npy')] # depth map path
    depth_files = depth_files[::2]

    poses_arr = np.load("/home/xuankai/code/dydeblur/data/DyBluRF/stereo_blur_dataset/seesaw/dense/poses_bounds.npy") # 48, 17; MARK: camera pose
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0]) # 3, 5, 48; poses[:,4,0]: 696., 1239., 716.985
    bds = poses_arr[:, -2:].transpose([1, 0]) # 2, 48

    sh = imageio.imread(imgfiles[0]).shape # 288, 512, 3
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # change height and width; poses[:,4,0]: 288., 512., 716.985
    poses[2, 4, :] = poses[2, 4, :] * 1. / 2.5 # change focal; poses[:,4,0]: 288., 512., 286.794
    
    poses = np.concatenate([poses[:, 1:2, :], 
                            -poses[:, 0:1, :], 
                            poses[:, 2:, :]], 1)            # llff (DRB) -> nerf (RUB)
    poses = np.concatenate([poses[:, 0:1, :], 
                            -poses[:, 1:3, :], 
                            poses[:, 3:, :]], 1)            # nerf (RUB) -> colmap (RDF)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)    # 48, 3, 5
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)        # 48, 2

    if camera_scale == -1.:
        bd_factor = 0.9
        sc =  1./(10 * bd_factor)
    else:
        sc = camera_scale
    poses[:,:3,3] *= sc # change camera center
    bds *= sc
    print('sc =',sc)

    height = poses[0,0,-1]     # height: 288
    width = poses[0,1,-1]      # width: 512 
    focal = poses[0,2,-1]      # focal: 286.794
                 
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    K = np.array([
        [focal, 0., width/2],
        [0., focal, height/2],
        [0., 0., 1.]
    ], dtype=np.float32)

    cam_infos = []
    for idx in range(len(poses)): 
        if idx % 2 == 0:
            img_path = imgfiles[idx // 2] # MARK: train image
            depth = np.load(depth_files[idx // 2])
            depth = cv.resize(depth, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # disparity; 288, 512
            depth[depth < 1e-3] = 1e-3
            depth = 1.0 / depth # disparity -> depth
            blur_map = np.ones(depth.shape) # 288, 512
            motion_mask = np.ones(depth.shape) # 288, 512
        else:
            img_path = imgfiles_inference[idx // 2] # MARK: test image
        img = Image.open(img_path) # 288, 512, 3
        if img.size[0] != sh[1]:
            img = img.resize((sh[1],sh[0])) # resize sharp
        image_name = os.path.basename(img_path).split(".")[0] # 00000
        
        uid =  idx // 2             # colmap_id,    unsureness
        fid = (idx // 2) / (len(poses) // 2)

        c2w = poses[idx, :3, :4]                
        bottom = np.reshape([0,0,0,1.], [1,4])
        c2w = np.concatenate([c2w[:3,:4], bottom], -2)  # c2w, homogeneous coordinates
        matrix = np.linalg.inv(np.array(c2w))           # w2c, homogeneous coordinates
        R = np.transpose(matrix[:3, :3])            # R in c2w, np.transpose(matrix[:3, :3]) == c2w[:3, :3]
        T = matrix[:3, 3]                           # T in w2c

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=np.array(FovY), FovX=np.array(FovX), image=img, depth=depth, blur_map=blur_map,
                              motion_mask=motion_mask, image_path=img_path, image_name=image_name, width=int(width), height=int(height), fid=fid)
        cam_infos.append(cam_info)

    return cam_infos, sc

def readDeblur4DGSDataset(path, camera_scale=-1, eval=True, llffhold=2):
    
    cam_infos_unsorted, sc = readDeblur4DGSCameras(path, camera_scale)
    # cam_infos_unsorted, sc = readDeblur4DGSCameras_depth(path, camera_scale)
    # cam_infos_unsorted, sc = readDeblur4DGSCameras_pose(path, camera_scale)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval: # divide the training set and test set
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos) # average camera center, diagonal

    depth_point = False
    # depth_point =True
    if depth_point:
        print("depth point!")
        ply_path = os.path.join(path, "depth_rgb_True.ply")
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        print("sfm point!")
        ply_path = os.path.join(path, "flow3d_preprocessed/colmap/sparse/points3D.ply") # NOTE sfm point cloud
        # ply_path = "/home/xuankai/code/dydeblur/data/DyBluRF/stereo_blur_dataset/seesaw/dense/sparse_/points3D.ply" # NOTE sfm point cloud
        bin_path = os.path.join(path, "flow3d_preprocessed/colmap/sparse/points3D.bin")
        txt_path = os.path.join(path, "flow3d_preprocessed/colmap/sparse/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    pcd_max = pcd.points.max(0)
    pcd_min = pcd.points.min(0)
    pcd_num = pcd.points.shape[0]
    print('final_pcd_max =', pcd_max)
    print('final_pcd_min =', pcd_min)
    print('final_pcd_num =', pcd_num)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           sc=sc)
    return scene_info

def readDyDeblurCameras(path, camera_scale=-1):
    
    imgdir = os.path.join(path, 'images') # blur image path, train image
    imgfiles_all = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    # imgfiles = imgfiles_all[::2] # only train
    imgfiles = imgfiles_all # only train

    imgdir_inference = os.path.join(path, 'images_test') # sharp image path, test image
    imgfiles_inference_all = [os.path.join(imgdir_inference, f) for f in sorted(os.listdir(imgdir_inference)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    # imgfiles_inference = imgfiles_inference_all[1::2] # only test
    imgfiles_inference = imgfiles_inference_all # only test

    maskdir = os.path.join(path, 'masks') # mask path
    maskfiles_all = [os.path.join(maskdir, f) for f in sorted(os.listdir(maskdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    # maskfiles = maskfiles_all[::2] # only train
    maskfiles = maskfiles_all # only train

    depth_dir = os.path.join(path, 'flow3d_preprocessed/aligned_depth_anything_colmap') # disparity
    depth_files = [os.path.join(depth_dir, f) for f in sorted(os.listdir(depth_dir)) if f.endswith('npy')] # depth map path
    # depth_files = depth_files[::2] # only train
    depth_files = depth_files # only train

    # frame_names_all = [f.split('/')[-1].split('.')[0] for f in imgfiles_all] # '00000', '00001', '00002'
    frame_names_all = ["%05d"%i for i in range(2*len(imgfiles))] # dyblurf; '00000', '00001', '00002'
    Ks, w2cs = get_colmap_camera_params( # 48, 4, 4
        os.path.join(path, "flow3d_preprocessed/colmap/sparse/"),
        [frame_name + ".png" for frame_name in frame_names_all],
    )
    Ks = torch.from_numpy(Ks[:, :3, :3].astype(np.float32)) # 48, 3, 3; f_x=f_y=666.057, w/2=640, h/2=360
    # optimization Ks
    h, w = imageio.imread(imgfiles[0]).shape[:2]
    hx2, wx2 = h/2, w/2
    virtual_hx2, virtual_wx2, virtual_fx2 = Ks[0,1,2], Ks[0,0,2], Ks[0,0,0]
    real_fx2 = virtual_fx2 * ((hx2/virtual_hx2)+(wx2/virtual_wx2)) / 2
    Ks[:,1,2], Ks[:,0,2], Ks[:,0,0], Ks[:,1,1] = hx2, wx2, real_fx2, real_fx2

    # if (imageio.imread(imgfiles[0]).shape[0] == 720) or (imageio.imread(imgfiles[0]).shape[0] == 800):
    #     Ks[:, :2] /= 1.0 # f_x=f_y=666.057, w/2=640, h/2=360; MARK: high resolution
    # elif (imageio.imread(imgfiles[0]).shape[1] == 940): # d2rf
    #     Ks[:, :2] /= 2.0        
    # else:
    #     Ks[:, :2] /= 2.5 # f_x=f_y=266.423, w/2=256, h/2=144; MARK: low resolution
    w2cs = torch.from_numpy(w2cs.astype(np.float32)) # 48, 4, 4
    c2ws = w2cs.inverse() # 48, 4, 4

    if camera_scale == -1.:
        bd_factor = 0.9
        sc =  1./(10 * bd_factor)
    else:
        sc = camera_scale
    c2ws[:,:3,3] *= sc # change camera center
    w2cs = c2ws.inverse() # 48, 4, 4; MARK: new w2cs
    print('sc =',sc)

    sh = imageio.imread(imgfiles[0]).shape # 288, 512, 3

    height = sh[0]      # height: 288
    width = sh[1]       # width: 512
    focal = Ks[0,0,0]   # focal: 266.423
                 
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    depth_no_scale = []
    for idx in range(len(c2ws)): 
        if idx % 2 == 0:
            depth = np.load(depth_files[idx // 2])
            depth = cv.resize(depth, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # disparity; 288, 512
            depth[depth < 1e-3] = 1e-3
            depth = 1.0 / depth # disparity -> depth 
            depth_no_scale.append(depth)
    depth_no_scale = np.array(depth_no_scale) # 24, 288, 512
    max_depth_values_per_frame = depth_no_scale.reshape(depth_no_scale.shape[0], -1).max(1) # (24,)
    max_depth_value = np.median(max_depth_values_per_frame) * 2.5 # 806.776
    depth_no_scale = np.clip(depth_no_scale, 0, max_depth_value) # 24, 288, 512; limit depth range
    depth_scale = depth_no_scale * sc # 24, 288, 512; change depth
    # depth_scale = depth_no_scale * 0.01 # 24, 288, 512; change depth


    cam_infos = []
    images, depths, masks, image_names = [], [], [], []
    for idx in range(len(c2ws)): 
        if idx % 2 == 0:
            img_path = imgfiles[idx // 2] # MARK: train image
            mask = np.array(Image.open(maskfiles[idx // 2])) / 255.
            if mask.shape[-1] == 3: # 288, 512, 3
                mask = mask[:, :, 0] # 288, 512
            depth = depth_scale[idx // 2]
            depth_no_sc = depth_no_scale[idx // 2]

            blur_map = np.ones(depth.shape) # 288, 512
            motion_mask = np.ones(depth.shape) # 288, 512
        else:
            img_path = imgfiles_inference[idx // 2] # MARK: test image

        img = Image.open(img_path) # 288, 512, 3
        if img.size[0] != sh[1]:
            img = img.resize((sh[1],sh[0])) # resize sharp
        image_name = os.path.basename(img_path).split(".")[0] # 00000
        
        uid =  idx // 2             # colmap_id,    unsureness
        fid = (idx // 2) / (len(c2ws) // 2)

        c2w = c2ws[idx]                         # c2w, homogeneous coordinates               
        matrix = np.linalg.inv(np.array(c2w))   # w2c, homogeneous coordinates
        R = np.transpose(matrix[:3, :3])        # R in c2w, np.transpose(matrix[:3, :3]) == c2w[:3, :3]
        T = matrix[:3, 3]                       # T in w2c
        K = Ks[idx]

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=np.array(FovY), FovX=np.array(FovX), image=img, depth=depth, blur_map=blur_map, depth_no_sc=depth_no_sc, 
                              motion_mask=mask, image_path=img_path, image_name=image_name, width=int(width), height=int(height), fid=fid)
        cam_infos.append(cam_info)

        if idx % 2 == 0: # train
            images.append(np.array(img)) # np.array(img).shape: 288, 512, 3
            depths.append(depth)
            masks.append(mask)
            image_names.append(image_name)
    
    track_dir = os.path.join(path, 'flow3d_preprocessed/2d_tracks') # 2d_track
    
    query_tracks_2d, foreground_points = get_foreground(track_dir, images, depths, masks, Ks, c2ws, image_names, num_samples=40_000) # foreground & background
    background_points = get_background(images, depths, masks, Ks, w2cs, num_samples=100_000)
    # static_masks = [1 - x for x in masks]
    # foreground_points = get_background(images, depths, static_masks, Ks, w2cs, num_samples=100_000)
    # tracks_3d = TrackObservations()
                                  
    return cam_infos, foreground_points, background_points, Ks, w2cs, sc

def readDyDeblurDataset(path, camera_scale=-1, eval=True, llffhold=2):
    
    cam_infos_unsorted, foreground_points, background_points, Ks, w2cs, sc = readDyDeblurCameras(path, camera_scale) # image, image_test, camera, depth, track_3d
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval: # divide the training set and test set
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos) # average camera center, diagonal

    ply_path = os.path.join(path, "flow3d_preprocessed/colmap/sparse/points3D.ply") # NOTE sfm point cloud
    bin_path = os.path.join(path, "flow3d_preprocessed/colmap/sparse/points3D.bin")
    txt_path = os.path.join(path, "flow3d_preprocessed/colmap/sparse/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, 
                           sc=sc,
                           foreground_points=foreground_points,
                           background_points=background_points,
                           Ks=Ks, 
                           w2cs=w2cs)
    return scene_info

def get_foreground(track_dir, images, depths, masks, Ks, c2ws, image_names, num_samples: int, step: int = 1):

    # load checkpoint
    # cached_track_3d_path = osp.join(self.cache_dir, f"tracks_3d_{num_samples}.pth")
    # if osp.exists(cached_track_3d_path) and step == 1 and self.load_from_cache: # when 'step == 1', load cache 3d track data
    #     print("loading cached 3d tracks data...")
    #     start, end = self.start, self.end
    #     cached_track_3d_data = torch.load(cached_track_3d_path)
    #     tracks_3d, visibles, invisibles, confidences, track_colors = (
    #         cached_track_3d_data["tracks_3d"][:, start:end],
    #         cached_track_3d_data["visibles"][:, start:end],
    #         cached_track_3d_data["invisibles"][:, start:end],
    #         cached_track_3d_data["confidences"][:, start:end],
    #         cached_track_3d_data["track_colors"],
    #     )
    #     return tracks_3d, visibles, invisibles, confidences, track_colors

    # load query_tracks_2d
    c2ws = c2ws[::2] # only train
    Ks = Ks[::2] # only train
    track_files = [os.path.join(track_dir, f) for f in sorted(os.listdir(track_dir)) if f.endswith('npy')]
    query_tracks_2d_files = [f"{frame_name}_{frame_name}.npy" for frame_name in image_names] # frame i -> frame i
    query_tracks_2d = [np.load(os.path.join(track_dir, f)).astype(np.float32) for f in query_tracks_2d_files] # 24, n_j, 4

    # Load 2D tracks.
    raw_tracks_2d = []
    num_frames = len(c2ws) # 24
    candidate_frames = list(range(0, num_frames, step)) # [0, 1, ..., 23]
    num_sampled_frames = len(candidate_frames) # 24
    for i in candidate_frames:
        curr_num_samples = query_tracks_2d[i].shape[0] # reference frame
        num_samples_per_frame = (
            int(np.floor(num_samples / num_sampled_frames)) # 416
            if i != candidate_frames[-1]
            else num_samples
            - (num_sampled_frames - 1)
            * int(np.floor(num_samples / num_sampled_frames))
        )
        if num_samples_per_frame < curr_num_samples:
            track_sels = np.random.choice(
                curr_num_samples, (num_samples_per_frame,), replace=False
            ) # track_sels is index
        else:
            track_sels = np.arange(0, curr_num_samples) # track_sels is index
        curr_tracks_2d = [] # 24, 416, 4
        for j in range(0, num_frames, step): # [0, 1, ..., 23]
            if i == j:
                target_tracks_2d = query_tracks_2d[i] # 3113, 4
            else:
                # target_tracks_2d = np.load(track_files[i*num_frames*2+j]).astype(np.float32)
                target_tracks_2d = np.load(track_files[i*num_frames+j]).astype(np.float32)
            curr_tracks_2d.append(target_tracks_2d[track_sels]) # 24, n_j, 4
        raw_tracks_2d.append(np.stack(curr_tracks_2d, axis=1)) # 24, n_j, 24, 4

    # Process 3D tracks.
    inv_Ks = torch.linalg.inv(Ks)[::step] # 24, 3, 3
    H, W, _ = images[0].shape # 288, 512
    filtered_tracks_3d, filtered_visibles, filtered_track_colors = [], [], []
    filtered_invisibles, filtered_confidences = [], []
    masks = np.array(masks) * (np.array(depths) > 0) # 24, 288, 512; refine mask
    masks = (masks > 0.5).astype(float) # Binarization
    for i, tracks_2d in enumerate(raw_tracks_2d): # raw_tracks_2d: 24, n_j, 24, 4
        tracks_2d = torch.tensor(tracks_2d).swapdims(0, 1) # n_j, 24, 4 -> 24, n_j, 4
        tracks_2d, occs, dists = ( # tracks_2d: 24, n_j, 2; occs: 24, n_j; dists: 24, n_j;
            tracks_2d[..., :2],
            tracks_2d[..., 2],
            tracks_2d[..., 3],
        )
        # visibles = postprocess_occlusions(occs, dists)
        visibles, invisibles, confidences = parse_tapir_track_info(occs, dists) # 24, n_j; parse TAPIR track information
        # Unproject 2D tracks to 3D.
        track_depths = F.grid_sample( # 24, n_j, 1; Sampling from the depth map
            torch.tensor(np.array(depths))[::step, None].float(), # 24, 1, 288, 512
            normalize_coords(tracks_2d[..., None, :], H, W), # tracks_2d[..., None, :]: 24, n_j, 2 -> 24, n_j, 1, 2
            align_corners=True,
            padding_mode="border",
        )[:, 0] # 24, 1, n_j, 1 -> 24, n_j, 1
        tracks_3d = ( # 24, n_j, 3; 2D pixel system -> 3D camera system
            torch.einsum(
                "nij,npj->npi",
                inv_Ks,
                F.pad(tracks_2d, (0, 1), value=1.0), # 24, n_j, 2 -> 24, n_j, 3
            )
            * track_depths
        )
        tracks_3d = torch.einsum( # 24, n_j, 3; 3D camera system -> 3D world system
            "nij,npj->npi", c2ws, F.pad(tracks_3d, (0, 1), value=1.0) # 24, n_j, 3 -> 24, n_j, 4
        )[..., :3] # 24, n_j, 4 -> 24, n_j, 3
        # Filter out out-of-mask tracks.
        is_in_masks = ( # 24, n_j; valid dynamic mask
            F.grid_sample(
                torch.tensor(np.array(masks))[::step, None].float(), # 24, 1, 288, 512; mask = dynamic_mask && valid_mask
                normalize_coords(tracks_2d[..., None, :], H, W), # tracks_2d[..., None, :]: 10, 1000, 2 -> 10, 1000, 1, 2
                align_corners=True,
            ).squeeze() # squeeze(): 24, 1, n_j, 1 -> 24, n_j
            == 1
        )
        visibles *= is_in_masks             # 24, n_j; Eliminate static area tracking points, Keep dynamic area tracking points
        invisibles *= is_in_masks           # 24, n_j
        confidences *= is_in_masks.float()  # 24, n_j
        # Get track's color from the query frame.
        track_colors = ( # n_j, 3; Sampling from the color map
            F.grid_sample( # 1, 3, 1, n_j
                torch.tensor(np.array(images[i * step : i * step + 1])).permute(0, 3, 1, 2).float() / 255., # 1, 3, 288, 512; NOTE image 255 process
                normalize_coords(tracks_2d[i : i + 1, None, :], H, W), # tracks_2d[i : i + 1, None, :]: 24, n_j, 2 -> 1, 1, n_j, 2
                align_corners=True,
                padding_mode="border",
            )
            .squeeze() # 1, 3, 1, n_j -> 3, n_j 
            .T # 3, n_j -> n_j, 3
        )
        # at least visible 5% of the time, otherwise discard
        visible_counts = visibles.sum(0) # (n_j,); num of subframes of visible track
        valid = visible_counts >= min( # valid track trace
            int(0.05 * num_frames),
            visible_counts.float().quantile(0.1).item(), # strange: 0
        )

        filtered_tracks_3d.append(tracks_3d[:, valid])      # 24, 24, n_j_new, 3
        filtered_visibles.append(visibles[:, valid])        # 24, 24, n_j_new
        filtered_invisibles.append(invisibles[:, valid])    # 24, 24, n_j_new
        filtered_confidences.append(confidences[:, valid])  # 24, 24, n_j_new
        filtered_track_colors.append(track_colors[valid])   # 24, n_j_new, 3

    filtered_tracks_3d = torch.cat(filtered_tracks_3d, dim=1).swapdims(0, 1)        # n_all, 24, 3
    filtered_visibles = torch.cat(filtered_visibles, dim=1).swapdims(0, 1)          # n_all, 24
    filtered_invisibles = torch.cat(filtered_invisibles, dim=1).swapdims(0, 1)      # n_all, 24
    filtered_confidences = torch.cat(filtered_confidences, dim=1).swapdims(0, 1)    # n_all, 24
    filtered_track_colors = torch.cat(filtered_track_colors, dim=0)                 # n_all, 3

    return query_tracks_2d, (
        filtered_tracks_3d,
        filtered_visibles,
        filtered_invisibles,
        filtered_confidences,
        filtered_track_colors,
    )

def get_background(images, depths, masks, Ks, w2cs, num_samples) :
    H, W, _ = images[0].shape
    w2cs = w2cs[::2] # 24, 4, 4; only train
    Ks = Ks[::2] # 24, 3, 3; only train
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(W, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            indexing="xy",
        ),
        dim=-1,
    )
    num_frames = len(images) # 24
    candidate_frames = list(range(num_frames)) # [0, 1, ..., 23]
    num_sampled_frames = len(candidate_frames)
    bkgd_points, bkgd_point_normals, bkgd_point_colors = [], [], []
    for i in candidate_frames:
        img = torch.tensor(images[i]) # 288, 512, 3
        depth = torch.tensor(depths[i]) # 288, 512
        bool_mask = ((1.0 - torch.tensor(masks[i])) * (depth > 0)).to(torch.bool) # valid static mask
        w2c = w2cs[i]
        K = Ks[i]
        points = ( # n_i, 3
            torch.einsum(
                "ij,pj->pi",
                torch.linalg.inv(K),
                F.pad(grid[bool_mask], (0, 1), value=1.0),
            )
            * depth[bool_mask][:, None]
        )
        points = torch.einsum( # n_i, 3
            "ij,pj->pi", torch.linalg.inv(w2c.double())[:3], F.pad(points, (0, 1), value=1.0)
        )
        point_normals = normal_from_depth_image(depth, K, w2c)[bool_mask] # n_i, 3
        point_colors = img[bool_mask] / 255. # n_i, 3; NOTE image 255 process
        curr_num_samples = points.shape[0]
        num_samples_per_frame = (
            int(np.floor(num_samples / num_sampled_frames))
            if i != candidate_frames[-1]
            else num_samples
            - (num_sampled_frames - 1)
            * int(np.floor(num_samples / num_sampled_frames))
        )
        if num_samples_per_frame < curr_num_samples:
            point_sels = np.random.choice(
                curr_num_samples, (num_samples_per_frame,), replace=False
            )
        else:
            point_sels = np.arange(0, curr_num_samples)
        bkgd_points.append(points[point_sels])
        bkgd_point_normals.append(point_normals[point_sels])
        bkgd_point_colors.append(point_colors[point_sels])
    bkgd_points = torch.cat(bkgd_points, dim=0)
    bkgd_point_normals = torch.cat(bkgd_point_normals, dim=0)
    bkgd_point_colors = torch.cat(bkgd_point_colors, dim=0)

    return (bkgd_points, bkgd_point_normals, bkgd_point_colors)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "D2RF": readD2RFDataset, # D2RF dataset in [https://github.com/xianrui-luo/D2RF]
    "DyBluRF": readDyBluRFDataset, # DyBluRF dataset in [https://github.com/huiqiang-sun/DyBluRF]
    "Deblur4DGS": readDeblur4DGSDataset, # Deblur4DGS dataset in [https://github.com/ZcsrenlongZ/Deblur4DGS]
    "DyDeblur": readDyDeblurDataset,
}
