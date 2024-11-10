import os
import torch
import subprocess
import numpy as np
from plyfile import PlyData
# from utils.graphics_utils import fov2focal
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def rodrigues_mat_to_rot(R):
    eps = 1e-16
    trc = np.trace(R)
    trc2 = (trc - 1.) / 2.
    # sinacostrc2 = np.sqrt(1 - trc2 * trc2)
    s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if (1 - trc2 * trc2) >= eps:
        tHeta = np.arccos(trc2)
        tHetaf = tHeta / (2 * (np.sin(tHeta)))
    else:
        tHeta = np.real(np.arccos(trc2))
        tHetaf = 0.5 / (1 - tHeta / 6)
    omega = tHetaf * s
    return omega


def rodrigues_rot_to_mat(r):
    wx, wy, wz = r
    theta = np.sqrt(wx * wx + wy * wy + wz * wz)
    a = np.cos(theta)
    b = (1 - np.cos(theta)) / (theta * theta)
    c = np.sin(theta) / theta
    R = np.zeros([3, 3])
    R[0, 0] = a + b * (wx * wx)
    R[0, 1] = b * wx * wy - c * wz
    R[0, 2] = b * wx * wz + c * wy
    R[1, 0] = b * wx * wy + c * wz
    R[1, 1] = a + b * (wy * wy)
    R[1, 2] = b * wy * wz - c * wx
    R[2, 0] = b * wx * wz - c * wy
    R[2, 1] = b * wz * wy + c * wx
    R[2, 2] = a + b * (wz * wz)
    return R


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def render_wander_path(view):
    focal_length = fov2focal(view.FoVy, view.image_height)
    R = view.R
    R[:, 1] = -R[:, 1]
    R[:, 2] = -R[:, 2]
    T = -view.T.reshape(-1, 1)
    pose = np.concatenate([R, T], -1)

    num_frames = 60
    max_disp = 5000.0  # 64 , 48

    max_trans = max_disp / focal_length  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0  # * 3.0 / 4.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ], axis=0)  # [np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose)  # torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate([pose, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        output_poses.append(torch.Tensor(render_pose))

    return output_poses

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        # self.ax = self.fig.gca(projection='3d')       
        self.ax = self.fig.add_subplot(projection = '3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def pointvisual(self, points, color):
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, marker='o', s=1)
    
    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()

    def save(self, path):
        plt.savefig(path)


def posevisual(save_path, train_c2ws, test_c2ws=None, virtual_c2ws=None): # c2w: n_poses, 3, 4 
    visualizer = CameraPoseVisualizer([-20, 20], [-20, 20], [0, 15])
    
    if train_c2ws.shape[1] == 3:
        bottom = np.array([[0,0,0,1]]).repeat(train_c2ws.shape[0], axis=0) # n_poses, 4
        train_c2ws = np.concatenate([train_c2ws[..., :4], bottom[:,np.newaxis,:]], 1) # n_poses, 4, 4

    for idx in range(train_c2ws.shape[0]):
        c2w = train_c2ws[idx].numpy()
        # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
        visualizer.extrinsic2pyramid(c2w, 'r', 2, 0.3)

    if test_c2ws != None:
        if test_c2ws.shape[1] == 3:
            bottom = np.array([[0,0,0,1]]).repeat(test_c2ws.shape[0], axis=0) # n_poses, 4
            test_c2ws = np.concatenate([test_c2ws[..., :4], bottom[:,np.newaxis,:]], 1) # n_poses, 4, 4

        for idx in range(test_c2ws.shape[0]):
            c2w = test_c2ws[idx].numpy()
            # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
            visualizer.extrinsic2pyramid(c2w, 'g', 1, 0.6)

    if virtual_c2ws != None:
        if virtual_c2ws.shape[1] == 3:
            bottom = np.array([[0,0,0,1]]).repeat(virtual_c2ws.shape[0], axis=0) # n_poses, 4
            virtual_c2ws = np.concatenate([virtual_c2ws[..., :4], bottom[:,np.newaxis,:]], 1) # n_poses, 4, 4

        for idx in range(virtual_c2ws.shape[0]):
            c2w = virtual_c2ws[idx].numpy()
            # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
            visualizer.extrinsic2pyramid(c2w, 'b', 1, 0.6)

    visualizer.save(save_path)
    print("Camera Pose is OK!")

def fetchPly(path): # fetch points
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return positions

if __name__ == "__main__": # TODO point visual
    # path = "/home/xuankai/code/d-3dgs/data/D2RF/"
    path = "/home/xuankai/code/d-3dgs/data/DyBluRF/stereo_blur_dataset/"
    for scene in sorted(os.listdir(path)):
        # pose_path = path + scene + "/poses_bounds.npy"
        # point_path = path + scene + "/sparse_/points3D.ply"
        # save_path = "/home/xuankai/code/d-3dgs/assets/camera/D2RF/" + scene + ".png"

        pose_path = path + scene + "/dense/poses_bounds.npy"
        point_path = path + scene + "/dense/sparse_/points3D.ply"
        save_path = "/home/xuankai/code/d-3dgs/assets/points/DyBluRF/" + scene + ".png"

        poses = np.load(pose_path)
        poses = poses[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3, 5, 34

        poses = np.concatenate([poses[:, 1:2, :], 
                                -poses[:, 0:1, :], 
                                poses[:, 2:, :]], 1) # llff (DRB) -> nerf (RUB)
        poses = np.concatenate([poses[:, 0:1, :], 
                                -poses[:, 1:3, :], 
                                poses[:, 3:, :]], 1) # nerf (RUB) -> colmap (RDF)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32) # 34, 3, 5
        bottom = np.array([[0,0,0,1]]).repeat(poses.shape[0], axis=0) # 34, 4
        c2ws = np.concatenate([poses[..., :4], bottom[:,np.newaxis,:]], 1) # 34, 4, 4

        # visualizer = CameraPoseVisualizer([-20, 20], [-20, 20], [0, 15])
        visualizer = CameraPoseVisualizer([-200, 200], [-100, 100], [-20, 400])
        for idx in range(c2ws.shape[0]):
            if idx % 2 == 0: # train
                color = 'r'
            else: # test
                color = 'g'

            c2w = c2ws[idx]
            # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
            visualizer.extrinsic2pyramid(c2w, color, 15)
    
        points = fetchPly(point_path)
        visualizer.pointvisual(points, color='b')
        visualizer.save(save_path)
