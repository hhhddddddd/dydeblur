import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.time_utils import get_embedder, Embedder

# K-nearest neighbor
import time
from sklearn.neighbors import NearestNeighbors

'''
def get_embedder(multires, i=1):
    if i == -1: # no Positional Encoding is performed
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
'''
class Blur(nn.Module): # xyz, 
    def __init__(self, D=8, W=256, multires=10, n_pts = 5): # potision(3) + rotation(4) + scale(3)
        super(Blur, self).__init__()
        self.n_pts = n_pts
        self.D = D 
        self.W = W 
        self.input_ch = n_pts * 2   # 10
        self.output_ch = n_pts      # 1
        self.skips = [3]            # skip connection
        self.spatial_lr_scale = 5

        # self.embed_fn, pos_encode_input_ch = get_embedder(multires, input_ch) # positional encoding function

        blur_net = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 2)] + [nn.Linear(W, self.output_ch)])
        
        self.blur = blur_net.cuda()

    def find_nearest_neighbors(self, points, k): # CPU is faster for this task

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)  
        distances, indices = nbrs.kneighbors(points) # distances: near -> far
        return distances, indices

    def forward(self, xyz, opacity, d_xyz=None, d_rotation=None, d_scaling=None):

        if d_xyz is not None:       
            points = xyz + d_xyz
        else:
            points = xyz # N, 3; N == 609

        points = points.to("cpu")
        # blend select: k_neighbors
        distances, indices = self.find_nearest_neighbors(points, self.n_pts) # N, k
        
        # prepare input: density, distance
        distances = torch.tensor(distances, dtype=torch.float).to("cuda")
        indices = torch.tensor(indices).to("cuda") # IndexError: too many indices for tensor of dimension 1
        densities = opacity.squeeze() # N;
        densities = densities[indices] # N, k
        input = torch.cat((distances, densities), dim=-1) # N, 2k

        # learn blend weight
        # input_emb = self.embed_fn(input) # positional encoding; 15006, 63
        h = input
        for i, l in enumerate(self.blur):
            h = self.blur[i](h)
            if i != self.D - 1:
                h = F.relu(h) # MARK: RELU
            else: 
                h = F.softmax(h, dim=-1)
            if i in self.skips:
                h = torch.cat([input, h], dim=-1)

        torch.cuda.empty_cache() # MARK: GPU
        return h, indices

    def train_setting(self, training_args): # Initialize the optimizer and the learning rate scheduler; training_args: opt
        l = [
            {'params': list(self.blur.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "blur"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.blur_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.dynamic_lr_max_steps)

    def save_weights(self, model_path, iteration, starttime, operate):
        out_weights_path = os.path.join(model_path, operate, starttime[:16], "blur/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.blur.state_dict(), os.path.join(out_weights_path, 'blur.pth'))

    def load_weights(self, model_path, operate, time, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, operate, time, "blur"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, operate, time, "blur/iteration_{}/blur.pth".format(loaded_iter))
        self.blur.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "blur":
                lr = self.blur_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    

