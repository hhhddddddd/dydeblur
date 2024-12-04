import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2' # MARK: GPU
import sys
import pdbr
import time
import json
import imageio
import random
from random import randint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from sklearn.neighbors import NearestNeighbors

# def handle_exception(exc_type, exc_value, exc_traceback):
#     # if exc_type is not KeyboardInterrupt:
#     print(f"Exception: {exc_value}")  
#     pdbr.post_mortem(exc_traceback)
         
# sys.excepthook = handle_exception

def torch_test():

    print(torch.cuda.is_available()) # Check whether CUDA is available
    print(torch.cuda.device_count()) # Check the available CUDA quantity
    print(torch.version.cuda) # Check the version number of CUDA
    print(torch.__version__) # Check the version number of torch

def find_nearest_neighbors(points, k): # CPU is faster for this task

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)  
    distances, indices = nbrs.kneighbors(points)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    return distances, indices
    
def find_nearest_neighbors_gpu(points, k): # GPU is slower for this task
    
    distances = torch.cdist(points, points)  
    distance, indices = torch.topk(distances, k + 1, dim=1, largest=False)  
    distance = distance[:, 1:]
    indices = indices[:, 1:]
    return distance, indices

def fetchPly(path): # fetch points
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                    vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def knn_chunk_mine(reference, query, chunk=30000): # B, gs_num, 3; MARK: time = 30.30
    
    fine_references = torch.tensor([], device="cuda") # B, gs_num, k * j_num, 3
    for i in range(0, query.shape[1], chunk):
        indices = torch.tensor([], device="cuda") # B, chunk, k * j_num
        for j in range(0, reference.shape[1], chunk):
            _, index = knn(reference[:,j:j+chunk,:], query[:,i:i+chunk,:]) # B, chunk, k
            indices = torch.cat((indices, index + j), dim=-1) # B, chunk, k * j_num; float32
        fine_references = torch.cat((fine_references, reference.squeeze()[indices.int()]), dim=1) # B, chunk, k * j_num, 3 -> B, gs_num, k * j_num, 3

    distances = torch.tensor([], device="cuda") # B, gs_num, k * j_num, 3
    indices = torch.tensor([], device="cuda") # B, gs_num, k * j_num, 3
    for i in range(0, query.shape[1]): # gs_num
        reference = fine_references[:,i,:,:] # B, k * j_num, 3
        fine_query = query[:,i,:].unsqueeze(0) # B, 1, 3
        distance, index = knn(reference, fine_query) # B, 1, k
        indices = torch.cat((indices, index), dim=1) # B, gs_num, k
        distances = torch.cat((distances, distance), dim=1) # B, gs_num, k

    return distances, indices

def knn_chunk_change1(reference, query, chunk=30000): # B, gs_num, 3; MARK: time = 30.03
    
    fine_indices = torch.tensor([], device="cuda") # B, gs_num, k * j_num
    for i in range(0, query.shape[1], chunk):
        indices = torch.tensor([], device="cuda") # B, chunk, k * j_num
        for j in range(0, reference.shape[1], chunk):
            _, index = knn(reference[:,j:j+chunk,:], query[:,i:i+chunk,:]) # B, chunk, k
            indices = torch.cat((indices, index + j), dim=-1) # B, chunk, k * j_num; int64 -> float32
        fine_indices = torch.cat((fine_indices, indices), dim=1) # B, gs_num, k * j_num
    references = reference.squeeze()[fine_indices.int()] # B, gs_num, k * j_num, 3

    distances = torch.tensor([], device="cuda") # B, gs_num, k
    indices = torch.tensor([], device="cuda") # B, gs_num, k
    for i in range(0, query.shape[1]): # gs_num
        reference = references[:,i,:,:] # B, k * j_num, 3
        fine_query = query[:,i,:].unsqueeze(0) # B, 1, 3
        distance, index = knn(reference, fine_query) # B, 1, k
        indices = torch.cat((indices, index), dim=1) # B, gs_num, k
        distances = torch.cat((distances, distance), dim=1) # B, gs_num, k

    return distances, indices

def knn_chunk_change2(reference, query, chunk=30000, k=5): # B, gs_num, 3; MARK: time = 29.94
    
    num = query.shape[1] // chunk
    # remain = query.shape[1] % chunk
    # if remain != 0:
    #     num += 1
    # elif remain >= k:
    #     n_fine = num * k
    # else:
    #     n_fine = (num - 1) * k + remain
    n_fine = num * k
        
    indices = torch.empty((1, query.shape[1], n_fine), device="cuda") # B, gs_num, k * j_num
    for i in range(0, query.shape[1], chunk):
        for j in range(0, reference.shape[1], chunk):
            _, index = knn(reference[:,j:j+chunk,:], query[:,i:i+chunk,:]) # B, chunk, k
            indices[:, i:i+chunk, (j//chunk)*k:(j//chunk)*k+k] = index # B, chunk, k * j_num -> B, gs_num, k * j_num

    references = reference.squeeze()[indices.int()] # B, gs_num, k * j_num, 3

    distances = torch.empty((1, query.shape[1], k), device="cuda") # B, gs_num, k
    indices = torch.empty((1, query.shape[1], k), device="cuda") # B, gs_num, k
    for i in range(0, query.shape[1]): # gs_num
        reference = references[:,i,:,:] # B, k * j_num, 3
        fine_query = query[:,i,:].unsqueeze(0) # B, 1, 3
        distance, index = knn(reference, fine_query) # B, 1, k
        indices[:,i,:] = index # B, gs_num, k
        distances[:,i,:] = distance # B, gs_num, k

    return distances, indices

# ######################## point bounds ########################
# path = "/home/xuankai/code/d-3dgs/data/D2RF/"
# # path = "/home/xuankai/code/d-3dgs/data/DyBluRF/stereo_blur_dataset/"
# for scene in sorted(os.listdir(path)):

#     point_path = path + scene + "/sparse_/0/points3D.ply"
#     # point_path = path + scene + "/dense/sparse_/points3D.ply"
    
#     plydata = PlyData.read(point_path)
#     vertices = plydata['vertex']
#     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

#     print(scene, "pcd_max:", positions.max(0))
#     print(scene, "pcd_min:", positions.min(0))
# ######################## point bounds ########################

######################## train test camera distance ########################
# path = "/home/xuankai/code/d-3dgs/data/D2RF/"
# path = "/home/xuankai/code/d-3dgs/data/DyBluRF/stereo_blur_dataset/"
# for scene in sorted(os.listdir(path)):

#     # pose_path = path + scene + "/poses_bounds.npy"
#     pose_path = path + scene + "/dense/poses_bounds.npy"
    
#     poses_arr = np.load(pose_path)
#     poses = poses_arr[:, :-2].reshape([-1, 3, 5]) # 34, 3, 5

#     train_poses = []
#     test_poses = []
#     for i in range(len(poses)):
#         if i % 2 == 0:
#             train_pose = poses[i]
#             train_poses.append(train_pose)
#         else: 
#             test_pose = poses[i]
#             test_poses.append(test_pose)
#     # train_center = np.stack(train_poses)[:, :, 3] # 17, 3
#     # test_center = np.stack(test_poses)[:, :, 3] # 17, 3
#     # distance = np.linalg.norm((train_center - test_center),axis=1)
#     # distance_mean = distance.mean()

#     train_center_z = np.stack(train_poses)[:, -1, 3] # 17, 3
#     test_center_z = np.stack(test_poses)[:, -1, 3] # 17, 3
#     distance = abs(train_center_z - test_center_z)
#     # distance = train_center_z - test_center_z
#     # print(scene)
#     distance_mean = distance.mean()

#     print(scene, "z-distance mean:", distance_mean)
######################## train test camera distance ########################

path = "/home/xuankai/LSM/D2RF/"
# path = "/home/xuankai/code/d-3dgs/data/DyBluRF/stereo_blur_dataset/"
for scene in sorted(os.listdir(path)):

    point_path = path + scene + "/point_cloud.ply"
    # point_path = path + scene + "/dense/sparse_/points3D.ply"
    
    plydata = PlyData.read(point_path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    print(scene, "pcd_max:", positions.max(0))
    print(scene, "pcd_min:", positions.min(0))



