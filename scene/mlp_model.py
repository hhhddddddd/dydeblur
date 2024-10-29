import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.time_utils import get_embedder, Embedder


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


class MLP(nn.Module): 
    def __init__(self, D=8, W=256, input_ch=10, output_ch=1, multires=10): # potision(3) + rotation(4) + scale(3)
        super(MLP, self).__init__()
        self.D = D 
        self.W = W 
        self.input_ch = input_ch    # 10
        self.output_ch = output_ch  # 1
        self.skips = [3]            # skip connection
        self.spatial_lr_scale = 5

        self.embed_fn, pos_encode_input_ch = get_embedder(multires, input_ch) # positional encoding function

        dynamic_net = nn.ModuleList(
            [nn.Linear(pos_encode_input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + pos_encode_input_ch, W)
                    for i in range(D - 2)] + [nn.Linear(W, output_ch)])
        
        self.dynamic = dynamic_net.cuda()

    def forward(self, input):

        input_emb = self.embed_fn(input) # positional encoding; 15006, 63

        ''' normalize
        mean = x.mean(dim=0) # torch.Size([10])
        std = x.std(dim=0)   # torch.Size([10])
        input = (x - mean) / std # Z-score normalize; torch.Size([100000, 10])

        min, _ = x.min(dim=0) # torch.Size([10])
        max, _ = x.max(dim=0) # torch.Size([10])
        input = (x - min) / (max - min) # Min-max normalize; torch.Size([100000, 10])
        '''

        h = input_emb
        for i, l in enumerate(self.dynamic):
            h = self.dynamic[i](h)
            if i != self.D - 1:
                h = F.relu(h) # MARK: detial
            if i in self.skips:
                h = torch.cat([input_emb, h], dim=-1)

        return h

    def train_setting(self, training_args): # Initialize the optimizer and the learning rate scheduler; training_args: opt
        l = [
            {'params': list(self.dynamic.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "dynamic"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.dynamic_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.dynamic_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "dynamic/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.dynamic.state_dict(), os.path.join(out_weights_path, 'dynamic.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "dynamic"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "dynamic/iteration_{}/dynamic.pth".format(loaded_iter))
        self.dynamic.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "dynamic":
                lr = self.dynamic_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    

