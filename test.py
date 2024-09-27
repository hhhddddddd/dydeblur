import torch
import torch.nn as nn
import torch.optim as optim

import pdbr
from sklearn.neighbors import NearestNeighbors

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


scene_name = '/home/xuankai/code/d-3dgs/utils'.split("/")[-1]
print(scene_name)
