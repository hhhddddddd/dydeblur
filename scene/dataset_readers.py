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
    poses_arr = np.load(os.path.join(path, 'poses_bounds.npy')) # MARK: load camera pose
    
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3, 5, 34
    bds = poses_arr[:, -2:].transpose([1,0]) # 2, 34
    
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
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # change height and width in poses
    poses[2, 4, :] = poses[2, 4, :] * 1./ 2 # change focal, image -> image_2
    
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

    height = poses[0,0,-1]     # height
    width = poses[0,1,-1]      # width 
    focal = poses[0,2,-1]      # focal
                 
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    K = np.array([
        [focal, 0., width],
        [0., focal, height],
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
        depth = cv.resize(depth, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # absolute depth, 400, 940
        depthv2 = np.load(depth_fine_files[idx])
        depthv2 = cv.resize(depthv2, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # depth anything v2
        blur_map = np.load(blur_map_files[idx]) # small is sharp, 400, 940
        blur_map = 1 - (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min()) # large is sharp
        motion_mask = np.array(Image.open(motion_mask_files[idx]))
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
 
    # depth_point = False
    depth_point =True
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
    print('final_pcd_max =', pcd_max)
    print('final_pcd_min =', pcd_min)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           sc=sc)
    return scene_info

def readDyBluRFCameras(path, camera_scale=-1):

    poses_arr = np.load(os.path.join(path, 'poses_bounds.npy')) # 48, 17; MARK: camera pose

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0]) # 3, 5, 48
    bds = poses_arr[:, -2:].transpose([1, 0]) # 2, 48
   
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
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # change height and width
    poses[2, 4, :] = poses[2, 4, :] * 1. / 2.5 # change focal
    
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

    height = poses[0,0,-1]     # height
    width = poses[0,1,-1]      # width 
    focal = poses[0,2,-1]      # focal
                 
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    K = np.array([
        [focal, 0., width],
        [0., focal, height],
        [0., 0., 1.]
    ], dtype=np.float32)

    cam_infos = []
    for idx in range(len(poses)): 
        if idx % 2 == 0:
            img_path = imgfiles[idx // 2] # MARK: train image
            depth = np.load(depth_files[idx // 2])
            depth = cv.resize(depth, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # absolute depth, 288, 512
            depthv2 = np.load(depth_fine_files[idx // 2])
            depthv2 = cv.resize(depthv2, (sh[1],sh[0]), interpolation=cv.INTER_NEAREST) # depth anything v2
            blur_map = np.load(blur_map_files[idx // 2]) # small is sharp, 288, 512
            blur_map = 1 - (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min()) # large is sharp
            motion_mask = np.array(Image.open(motion_mask_files[idx // 2]))
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
    print('final_pcd_max =', pcd_max)
    print('final_pcd_min =', pcd_min)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           sc=sc)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "D2RF": readD2RFDataset, # D2RF dataset in [https://github.com/xianrui-luo/D2RF]
    "DyBluRF": readDyBluRFDataset, # DyBluRF dataset in [https://github.com/huiqiang-sun/DyBluRF]
}
