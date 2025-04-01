# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

import numpy as np
import os
# from typing import Dict, Literal, Optional, Tuple

import torch
from torch import nn
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func

class Blur(nn.Module):
    def __init__(self, num_img, H=400, W=600, img_embed=32, ks=17, not_use_rgbd=False, not_use_pe=False, not_use_dynamic_mask=True, skip_connect=True):
        super().__init__()
        self.num_img = num_img
        self.W, self.H = W, H

        self.img_embed_cnl = img_embed

        self.min_freq, self.max_freq, self.num_frequencies = 0.0, 3.0, 4

        self.embedding_camera = nn.Embedding(self.num_img, self.img_embed_cnl) # learnable function

        print('this is single res kernel', ks)
        
        self.not_use_rgbd = not_use_rgbd
        self.not_use_pe = not_use_pe
        self.not_use_dynamic_mask = not_use_dynamic_mask
        self.skip_connect = skip_connect
        print('single res: not_use_rgbd', self.not_use_rgbd, 'not_use_pe', self.not_use_pe)
        rgd_dim = 0 if self.not_use_rgbd else 32
        pe_dim = 0 if self.not_use_pe else 16

        self.mlp_base_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(32+pe_dim+rgd_dim, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            )
        
        # self.mlp_head1 = torch.nn.Conv2d(64, ks**2, 1, bias=False) # NOTE
        if self.skip_connect:
            if not_use_dynamic_mask:
                self.mlp_mask1 = torch.nn.Conv2d(64+4, 1, 1, bias=False)
                self.mlp_head1 = torch.nn.Conv2d(64+4, ks**2, 1, bias=False) # NOTE
            else:
                self.mlp_mask1 = torch.nn.Conv2d(64+5, 1, 1, bias=False)
                self.mlp_head1 = torch.nn.Conv2d(64+5, ks**2, 1, bias=False) # NOTE
        else:
            self.mlp_mask1 = torch.nn.Conv2d(64, 1, 1, bias=False)
            self.mlp_head1 = torch.nn.Conv2d(64, ks**2, 1, bias=False) # NOTE

        if not not_use_rgbd and not_use_dynamic_mask:
            self.conv_rgbd = torch.nn.Sequential(
                torch.nn.Conv2d(4, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
                torch.nn.Conv2d(64, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
                torch.nn.Conv2d(64, 32, 3,padding=1)
                )
        if not not_use_rgbd and not not_use_dynamic_mask:
            self.conv_rgbd = torch.nn.Sequential(
                torch.nn.Conv2d(5, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
                torch.nn.Conv2d(64, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
                torch.nn.Conv2d(64, 32, 3,padding=1)
                )

    def forward(self, img_idx, pos_enc, img):
        img_embed = self.embedding_camera(torch.LongTensor([img_idx]).cuda())[None, None] # 1, 1, 1, 32
        img_embed = img_embed.expand(pos_enc.shape[0],pos_enc.shape[1],pos_enc.shape[2],img_embed.shape[-1]) # 1, 400, 940, 32

        if self.not_use_pe:
            inp = img_embed.permute(0,3,1,2)
        else:
            inp = torch.cat([img_embed,pos_enc],-1).permute(0,3,1,2) # 1, 48, 400, 940

        if self.not_use_rgbd:
            feat = self.mlp_base_mlp(inp)
        else:
            rgbd_feat = self.conv_rgbd(img) # 1, 32, 400, 940
            feat = self.mlp_base_mlp(torch.cat([inp,rgbd_feat],1)) # 1, 64, 400, 940

        # weight = self.mlp_head1(feat) # 1, 9 * 9, 400, 940 # NOTE
        if self.skip_connect:
            mask = self.mlp_mask1(torch.cat([feat,img],1)) # 1, 1, 400, 940
            weight = self.mlp_head1(torch.cat([feat,img],1)) # 1, 9 * 9, 400, 940 # NOTE
        else:
            mask = self.mlp_mask1(feat) # 1, 1, 400, 940
            weight = self.mlp_head1(feat) # 1, 9 * 9, 400, 940 # NOTE


        weight = torch.softmax(weight, dim=1)
        mask = torch.sigmoid(mask)

        return weight, mask


    def train_setting(self): # Initialize the optimizer
        l = [
            {'params':  list(self.embedding_camera.parameters()) +
                        list(self.mlp_base_mlp.parameters()) +
                        list(self.mlp_head1.parameters()) + 
                        list(self.mlp_mask1.parameters()),
             'lr': 5e-4, "name": "blur"} # 5e-4
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.blur_scheduler_args = get_expon_lr_func(lr_init=5e-4, lr_final=1e-5, lr_delay_mult=0.01, max_steps=35000)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "blur":
                lr = self.blur_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
   
    def save_weights(self, model_path, iteration, starttime, operate, best=False):   
        if not best:
            out_weights_path = os.path.join(model_path, operate, starttime[:16], "blur/iteration_{}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, operate, starttime[:16], "blur/iteration_66666")
        os.makedirs(out_weights_path, exist_ok=True)
        combined_state_dict = {
            'embed': self.embedding_camera.state_dict(),
            'base': self.mlp_base_mlp.state_dict(),
            'head': self.mlp_head1.state_dict(),
            'mask': self.mlp_mask1.state_dict()
        }
        torch.save(combined_state_dict, os.path.join(out_weights_path, 'blur.pth'))

    def load_weights(self, model_path, operate, time, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, operate, time, "blur"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, operate, time, "blur/iteration_{}/blur.pth".format(loaded_iter))
        combined_state_dict = torch.load(weights_path)
        self.embedding_camera.load_state_dict(combined_state_dict['embed'])
        self.mlp_base_mlp.load_state_dict(combined_state_dict['base'])
        self.mlp_head1.load_state_dict(combined_state_dict['head'])
        self.mlp_mask1.load_state_dict(combined_state_dict['mask'])

    

