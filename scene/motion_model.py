import random
from typing import Literal
import os
import cupy as cp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from cuml import HDBSCAN, KMeans

from tqdm import tqdm

from utils.general_utils import  TrackObservations, get_expon_lr_func
from utils.transforms_utils import cont_6d_to_rmat, solve_procrustes, get_weights_for_procrustes
from utils.system_utils import searchForMaxIteration

class MotionBases(nn.Module):
    def __init__(self, rots, transls):
        super().__init__()
        self.num_frames = rots.shape[1]
        self.num_bases = rots.shape[0]
        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots),
                "transls": nn.Parameter(transls),
            }
        )

    def train_setting(self, training_args): # Initialize the optimizer and the learning rate scheduler; training_args: opt
        l = [
            {'params': self.params["rots"], 'lr': 1.6e-4, "name": "rots"},
            {'params': self.params["transls"], 'lr': 1.6e-4, "name": "transls"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.motion_scheduler_args = get_expon_lr_func(lr_init=1.6e-4,lr_final=1.6e-4, 
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration, starttime, operate, best=False):
        if not best:
            out_weights_path = os.path.join(model_path, operate, starttime[:16], "motion/iteration_{}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, operate, starttime[:16], "motion/iteration_66666")
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.params.state_dict(), os.path.join(out_weights_path, 'motion.pth'))

    def load_weights(self, model_path, operate, time, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, operate, time, "motion"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, operate, time, "motion/iteration_{}/motion.pth".format(loaded_iter))
        self.params.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "rots":
                lr = self.motion_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            if param_group["name"] == "transls":
                lr = self.motion_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        param_keys = ["rots", "transls"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return MotionBases(**args)

    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        returns transforms (G, B, 3, 4)
        """
        # Local Time interpolation in Points-based Motion
        # ts (num_fg, B)
        if len(ts.shape) == 1:
            ts = ts.unsqueeze(0)
        ts_pre = torch.floor(ts.clone()).clamp(0., self.params["transls"].shape[1]-1).int()
        ts_next = torch.ceil(ts.clone()).clamp(0., self.params["transls"].shape[1]-1).int()

        #              (20, 8, 3)             ()
        #  the same pre and next for all gaussians
        transls_pre = self.params["transls"][:, ts_pre[0, :]]
        rots_pre = self.params["rots"][:, ts_pre[0, :]]

        transls_next = self.params["transls"][:, ts_next[0, :]]
        rots_next = self.params["rots"][:, ts_next[0, :]]

        transls_pre = torch.einsum("pk,kni->pni", coefs, transls_pre)  # (G, B, 3)  torch.Size([22200, 1, 3])
        rots_pre = torch.einsum("pk,kni->pni", coefs, rots_pre)  # (G, B, 6)  torch.Size([22200, 1, 6])
        transls_next = torch.einsum("pk,kni->pni", coefs, transls_next)  # (G, B, 3)  torch.Size([22200, 1, 3])
        rots_next = torch.einsum("pk,kni->pni", coefs, rots_next)  # (G, B, 6)  torch.Size([22200, 1, 6])
        
        num_fg = transls_pre.shape[0]
        if len(ts.shape) == 2 and ts.shape[0] == 1:  # (G, B, 1)
            ts = ts.repeat(num_fg, 1)
            ts_pre = ts_pre.repeat(num_fg, 1)
            ts_next = ts_next.repeat(num_fg, 1)
        w = (ts - ts_pre)  # next motion的权重
        w = w.unsqueeze(-1)
        # print(ts.shape, w.shape, transls_pre.shape, transls_next.shape)
        transls = (1. - w) * transls_pre + w * transls_next
        rots = (1. - w) * rots_pre+ w * rots_next
        rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
        # print(rotmats.shape, transls[..., None].shape)
        return torch.cat([rotmats, transls[..., None]], dim=-1)

def init_motion_params_with_procrustes(
    tracks_3d: TrackObservations,
    num_bases: int,
    rot_type: Literal["quat", "6d"],
    cano_t: int,
    cluster_init_method: str = "kmeans",
    min_mean_weight: float = 0.1,
) -> tuple[MotionBases, torch.Tensor, TrackObservations]:
    device = tracks_3d.xyz.device
    num_frames = tracks_3d.xyz.shape[1] # 24
    # sample centers and get initial se3 motion bases by solving procrustes
    means_cano = tracks_3d.xyz[:, cano_t].clone()  # [n_fg_gs, 3]

    # remove outliers
    scene_center = means_cano.median(dim=0).values # 3,
    print(f"{scene_center=}")
    dists = torch.norm(means_cano - scene_center, dim=-1) # n_fg_gs, 
    dists_th = torch.quantile(dists, 0.95)
    valid_mask = dists < dists_th # n_fg_gs,

    # remove tracks that are not visible in all frame
    valid_mask = valid_mask & tracks_3d.visibles.any(dim=1) # 7392

    xyz = tracks_3d.xyz[valid_mask] # 7392, 24, 3
    visibles = tracks_3d.visibles[valid_mask] # 7392, 24
    invisibles = tracks_3d.invisibles[valid_mask] # 7392, 24
    confidences = tracks_3d.confidences[valid_mask] # 7392, 24
    colors = tracks_3d.colors[valid_mask] # 7392, 3

    tracks_3d = TrackObservations( xyz=xyz, visibles=visibles, invisibles=invisibles, confidences=confidences, colors=colors)
    
    means_cano = means_cano[valid_mask] # init can_t space

    sampled_centers, num_bases, labels = sample_initial_bases_centers(
        cluster_init_method, cano_t, tracks_3d, num_bases
    ) # sampled_centers: 1, 10, 3; num_bases: 10; labels: 7392,

    # assign each point to the label to compute the cluster weight
    ids, counts = labels.unique(return_counts=True)
    num_bases = len(ids)
    sampled_centers = sampled_centers.cuda()[:, ids] # 1, 10, 3

    # compute basis weights from the distance to the cluster centers
    dists2centers = torch.norm(means_cano.cuda()[:, None] - sampled_centers, dim=-1) # 7392, 10
    motion_coefs = 10 * torch.exp(-dists2centers)

    init_rots, init_ts = [], []

    if rot_type == "quat":
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        rot_dim = 4
    else:
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
        rot_dim = 6

    init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)
    init_ts = torch.zeros(num_bases, num_frames, 3, device=device)
    errs_before = np.full((num_bases, num_frames), -1.0)
    errs_after = np.full((num_bases, num_frames), -1.0)

    tgt_ts = list(range(cano_t - 1, -1, -1)) + list(range(cano_t, num_frames)) # [9, ..., 0, 10, ..., 23]; cano_t: 10
    skipped_ts = {}
    for n, cluster_id in enumerate(ids):
        mask_in_cluster = (labels == cluster_id).cpu()
        cluster = tracks_3d.xyz[mask_in_cluster].transpose(
            0, 1
        )  # [num_frames, n_pts, 3]
        visibilities = tracks_3d.visibles[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        confidences = tracks_3d.confidences[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        weights = get_weights_for_procrustes(cluster, visibilities) # [num_frames, n_pts]
        prev_t = cano_t
        cluster_skip_ts = []
        for cur_t in tgt_ts:
            # compute pairwise transform from cano_t
            procrustes_weights = ( # [n_pts,]
                weights[cano_t]
                * weights[cur_t]
                * (confidences[cano_t] + confidences[cur_t])
                / 2
            )
            if procrustes_weights.sum() < min_mean_weight * num_frames:
                init_rots[n, cur_t] = init_rots[n, prev_t]
                init_ts[n, cur_t] = init_ts[n, prev_t]
                cluster_skip_ts.append(cur_t)
            else:
                se3, (err, err_before) = solve_procrustes(
                    cluster[cano_t],
                    cluster[cur_t],
                    weights=procrustes_weights,
                    enforce_se3=True,
                    rot_type=rot_type,
                )
                init_rot, init_t, _ = se3
                assert init_rot.shape[-1] == rot_dim
                # double cover
                if rot_type == "quat" and torch.linalg.norm( # special process for "quat"
                    init_rot - init_rots[n][prev_t]
                ) > torch.linalg.norm(-init_rot - init_rots[n][prev_t]):
                    init_rot = -init_rot
                init_rots[n, cur_t] = init_rot
                init_ts[n, cur_t] = init_t
                if err == np.nan:
                    print(f"{cur_t=} {err=}")
                    print(f"{procrustes_weights.isnan().sum()=}")
                if err_before == np.nan:
                    print(f"{cur_t=} {err_before=}")
                    print(f"{procrustes_weights.isnan().sum()=}")
                errs_after[n, cur_t] = err
                errs_before[n, cur_t] = err_before
            prev_t = cur_t
        skipped_ts[cluster_id.item()] = cluster_skip_ts

    bases = MotionBases(init_rots, init_ts)
    return bases, motion_coefs, tracks_3d

def sample_initial_bases_centers(
    mode: str, cano_t: int, tracks_3d: TrackObservations, num_bases: int
):
    """
    :param mode: "farthest" | "hdbscan" | "kmeans"
    :param tracks_3d: [G, T, 3]
    :param cano_t: canonical index
    :param num_bases: number of SE3 bases
    """
    assert mode in ["farthest", "hdbscan", "kmeans"]
    means_canonical = tracks_3d.xyz[:, cano_t].clone() # 7392, 3

    xyz = cp.asarray(tracks_3d.xyz) # 7392, 24, 3
    visibles = cp.asarray(tracks_3d.visibles)

    num_tracks = xyz.shape[0]
    xyz_interp = batched_interp_masked(xyz, visibles) # 7392, 24, 3; NOTE process invisibles

    velocities = xyz_interp[:, 1:] - xyz_interp[:, :-1] # 7392, 23, 3;
    vel_dirs = (
        velocities / (cp.linalg.norm(velocities, axis=-1, keepdims=True) + 1e-5)
    ).reshape((num_tracks, -1)) # 7392, 23*3

    # [num_bases, num_gaussians]
    if mode == "kmeans":
        model = KMeans(n_clusters=num_bases)
    else:
        model = HDBSCAN(min_cluster_size=20, max_cluster_size=num_tracks // 4)
    model.fit(vel_dirs)
    labels = model.labels_
    num_bases = labels.max().item() + 1 # update num_bases for HDBSCAN
    sampled_centers = torch.stack( # why xyz_interp
        [
            means_canonical[torch.tensor(labels == i).cpu()].median(dim=0).values # NOTE
            if torch.tensor(labels == i).sum() !=0 else means_canonical[random.randint(0, num_bases)]
            for i in range(num_bases)
        ]
    )[None]
    print("number of {} clusters: ".format(mode), num_bases)
    return sampled_centers, num_bases, torch.tensor(labels)

def interp_masked(vals: cp.ndarray, mask: cp.ndarray, pad: int = 1) -> cp.ndarray:
    """
    hacky way to interpolate batched with cupy
    by concatenating the batches and pad with dummy values
    :param vals: [B, M, *]
    :param mask: [B, M]
    """
    assert mask.ndim == 2
    assert vals.shape[:2] == mask.shape

    B, M = mask.shape # B: 4096, M: 24

    # get the first and last valid values for each track
    sh = vals.shape[2:] # 3
    vals = vals.reshape((B, M, -1))
    D = vals.shape[-1] # 3
    first_val_idcs = cp.argmax(mask, axis=-1) # 4096
    last_val_idcs = M - 1 - cp.argmax(cp.flip(mask, axis=-1), axis=-1) # 4096
    bidcs = cp.arange(B)

    v0 = vals[bidcs, first_val_idcs][:, None] # 4096, 1, 3
    v1 = vals[bidcs, last_val_idcs][:, None]
    m0 = mask[bidcs, first_val_idcs][:, None] # 4096, 1
    m1 = mask[bidcs, last_val_idcs][:, None]
    if pad > 1:
        v0 = cp.tile(v0, [1, pad, 1])
        v1 = cp.tile(v1, [1, pad, 1])
        m0 = cp.tile(m0, [1, pad])
        m1 = cp.tile(m1, [1, pad])

    vals_pad = cp.concatenate([v0, vals, v1], axis=1) # 4096, 26, 3
    mask_pad = cp.concatenate([m0, mask, m1], axis=1) # 4096, 26

    M_pad = vals_pad.shape[1] # 26
    vals_flat = vals_pad.reshape((B * M_pad, -1)) # 4096*26, 3
    mask_flat = mask_pad.reshape((B * M_pad,)) # 4096*26
    idcs = cp.where(mask_flat)[0] # True's index

    cx = cp.arange(B * M_pad)
    out = cp.zeros((B * M_pad, D), dtype=vals_flat.dtype)
    for d in range(D):
        out[:, d] = cp.interp(cx, idcs, vals_flat[idcs, d])

    out = out.reshape((B, M_pad, *sh))[:, pad:-pad]
    return out

def batched_interp_masked(
    vals: cp.ndarray, mask: cp.ndarray, batch_num: int = 4096, batch_time: int = 64
):
    assert mask.ndim == 2
    B, M = mask.shape # B: n_tracks, M: n_frames
    out = cp.zeros_like(vals)
    for b in tqdm(range(0, B, batch_num), leave=False):
        for m in tqdm(range(0, M, batch_time), leave=False):
            x = interp_masked(
                vals[b : b + batch_num, m : m + batch_time],
                mask[b : b + batch_num, m : m + batch_time],
            )  # (batch_num, batch_time, *)
            out[b : b + batch_num, m : m + batch_time] = x
    return out