import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType

# K-nearest neighbor
import pdbr
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors


def find_nearest_neighbors(points, k): # CPU is faster for this task

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)  
    distances, indices = nbrs.kneighbors(points)
    indices = indices[:, 1:]
    return indices
    
def find_nearest_neighbors_gpu(points, k): # GPU is slower for this task
    
    distances = torch.cdist(points, points)  
    _, indices = torch.topk(distances, k + 1, dim=1, largest=False)  
    indices = indices[:, 1:]
    return indices
         
def color_blend(position, xiefangcha, opicty, rgb):
    # blend weight
    
    # blend gaussian
    
    # blend
    print('ok')


    

