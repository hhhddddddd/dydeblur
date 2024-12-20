import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False, spatial_lr_scale=5.):
        self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof).cuda() # MARK: cuda
        self.optimizer = None
        # self.spatial_lr_scale = 0.5
        # self.spatial_lr_scale = 5.0
        self.spatial_lr_scale = spatial_lr_scale   # scale factor: Adjusts the learning rate scaling for different spatial locations
        print('DeforModel Cameras Extent =', self.spatial_lr_scale) # FIXME 

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb) # DeformNetwork.forward

    def train_setting(self, training_args): # Initialize the optimizer and the learning rate scheduler; training_args: opt
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final, # * self.spatial_lr_scale, # NOTE
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration, starttime, operate, best=False):
        if not best:
            out_weights_path = os.path.join(model_path, operate, starttime[:16], "deform/iteration_{}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, operate, starttime[:16], "deform/iteration_66666")
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, operate, time, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, operate, time, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, operate, time, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
