import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np
import cv2
import re

from PIL import Image
from kornia.utils import create_meshgrid
from utils.klevr_utils import scale_rays, calculate_near_and_far

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).to(x.device))

def seed_everything(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_ckpt(network, ckpt_file, key_prefix, strict=True):
    print("loading ckpt from ", ckpt_file, " ...")
    ckpt_dict = torch.load(ckpt_file)

    if "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]

    state_dict = {}
    for key, val in ckpt_dict.items():
        if key_prefix in key:
            state_dict[key[len(key_prefix) + 1 :]] = val
    network.load_state_dict(state_dict, strict)


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction="mean")
        self.loss_ray = nn.SmoothL1Loss(reduction="none")

    def forward(self, inputs, targets):
        loss = 0
        if isinstance(inputs, dict):
            for l in range(self.levels):
                depth_pred_l = inputs[f"level_{l}"]
                V = depth_pred_l.shape[1]

                depth_gt_l = targets[f"level_{l}"]
                depth_gt_l = depth_gt_l[:, :V]
                mask_l = depth_gt_l > 0

                loss = loss + self.loss(
                    depth_pred_l[mask_l], depth_gt_l[mask_l]
                ) * 2 ** (1 - l)
        else:
            mask = targets > 0
            loss = loss + (self.loss_ray(inputs, targets) * mask).sum() / len(mask)

        return loss


def self_supervision_loss(
    loss_fn,
    rays_pixs,
    rendered_depth,
    depth_map,
    rays_gt_rgb,
    unpre_imgs,
    rendered_rgb,
    intrinsics,
    c2ws,
    w2cs,
):
    loss = 0
    target_points = torch.stack(
        [rays_pixs[1], rays_pixs[0], torch.ones(rays_pixs[0].shape[0]).to(rays_pixs.device)], dim=-1
    )
    target_points = rendered_depth.view(-1, 1) * (
        target_points @ torch.inverse(intrinsics[0, -1]).t()
    )
    target_points = target_points @ c2ws[0, -1][:3, :3].t() + c2ws[0, -1][:3, 3]

    rgb_mask = (rendered_rgb - rays_gt_rgb).abs().mean(dim=-1) < 0.02

    for v in range(len(w2cs[0]) - 1):
        points_v = target_points @ w2cs[0, v][:3, :3].t() + w2cs[0, v][:3, 3]
        points_v = points_v @ intrinsics[0, v].t()
        z_pred = points_v[:, -1].clone()
        points_v = points_v[:, :2] / points_v[:, -1:]

        points_unit = points_v.clone()
        H, W = depth_map["level_0"].shape[-2:]
        points_unit[:, 0] = points_unit[:, 0] / W
        points_unit[:, 1] = points_unit[:, 1] / H
        grid = 2 * points_unit - 1

        warped_rgbs = F.grid_sample(
            unpre_imgs[:, v],
            grid.view(1, -1, 1, 2),
            align_corners=True,
            mode="bilinear",
            padding_mode="zeros",
        ).squeeze()
        photo_mask = (warped_rgbs.t() - rays_gt_rgb).abs().mean(dim=-1) < 0.02

        pixel_coor = points_v.round().long()
        k = 5
        pixel_coor[:, 0] = pixel_coor[:, 0].clip(k // 2, W - (k // 2) - 1)
        pixel_coor[:, 1] = pixel_coor[:, 1].clip(2, H - (k // 2) - 1)
        lower_b = pixel_coor - (k // 2)
        higher_b = pixel_coor + (k // 2)

        ind_h = (
            lower_b[:, 1:] * torch.arange(k - 1, -1, -1).view(1, -1).to(lower_b.device)
            + higher_b[:, 1:] * torch.arange(0, k).view(1, -1).to(lower_b.device)
        ) // (k - 1)
        ind_w = (
            lower_b[:, 0:1] * torch.arange(k - 1, -1, -1).view(1, -1).to(lower_b.device)
            + higher_b[:, 0:1] * torch.arange(0, k).view(1, -1).to(lower_b.device)
        ) // (k - 1)

        patches_h = torch.gather(
            unpre_imgs[:, v].mean(dim=1).expand(ind_h.shape[0], -1, -1),
            1,
            ind_h.unsqueeze(-1).expand(-1, -1, W),
        )
        patches = torch.gather(patches_h, 2, ind_w.unsqueeze(1).expand(-1, k, -1))
        ent_mask = patches.view(-1, k * k).std(dim=-1) > 0.05

        for l in range(3):
            depth = F.grid_sample(
                depth_map[f"level_{l}"][:, v : v + 1],
                grid.view(1, -1, 1, 2),
                align_corners=True,
                mode="bilinear",
                padding_mode="zeros",
            ).squeeze()
            in_mask = (grid > -1.0) * (grid < 1.0)
            in_mask = (in_mask[..., 0] * in_mask[..., 1]).float()
            loss = loss + loss_fn(
                depth, z_pred * in_mask * photo_mask * ent_mask * rgb_mask
            ) * 2 ** (1 - l)
    loss = loss / (len(w2cs[0]) - 1)

    return loss


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]


def abs_error(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    err = depth_pred - depth_gt
    return np.abs(err) if type(depth_pred) is np.ndarray else err.abs()


def acc_threshold(depth_pred, depth_gt, mask, threshold):
    errors = abs_error(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return (
        acc_mask.astype("float") if type(depth_pred) is np.ndarray else acc_mask.float()
    )


# Ray helpers
def get_rays(
    H,
    W,
    intrinsics_target,
    c2w_target,
    chunk=-1,
    chunk_id=-1,
    train=True,
    train_batch_size=-1,
    mask=None,
):
    if train:
        # if mask is None:
        # not using gt target depth map as mask
        if True:
            xs, ys = (
                torch.randint(0, W, (train_batch_size,)).float().to(intrinsics_target.device),
                torch.randint(0, H, (train_batch_size,)).float().to(intrinsics_target.device),
            )
        else:  # Sample 8 times more points to get mask points as much as possible
            xs, ys = (
                torch.randint(0, W, (8 * train_batch_size,)).float().to(intrinsics_target.device),
                torch.randint(0, H, (8 * train_batch_size,)).float().to(intrinsics_target.device),
            )
            masked_points = mask[ys.long(), xs.long()]
            xs_, ys_ = xs[~masked_points], ys[~masked_points]
            xs, ys = xs[masked_points], ys[masked_points]
            xs, ys = torch.cat([xs, xs_]), torch.cat([ys, ys_])
            xs, ys = xs[:train_batch_size], ys[:train_batch_size]
    else:
        ys, xs = torch.meshgrid(
            torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing="ij"
        )  # pytorch's meshgrid has indexing='ij'
        ys, xs = ys.to(intrinsics_target.device).reshape(-1), xs.to(intrinsics_target.device).reshape(-1)
        if chunk > 0:
            ys, xs = (
                ys[chunk_id * chunk : (chunk_id + 1) * chunk],
                xs[chunk_id * chunk : (chunk_id + 1) * chunk],
            )
    dirs = torch.stack(
        [
            (xs - intrinsics_target[0, 2]) / intrinsics_target[0, 0],
            (ys - intrinsics_target[1, 2]) / intrinsics_target[1, 1],
            torch.ones_like(xs),
        ],
        -1,
    )  # use 1 instead of -1
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_dir = (
        dirs @ c2w_target[:3, :3].t()
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # the rays_dir might need to be normalized to 1
    # rays_dir = rays_dir / torch.norm(rays_dir, dim=-1, keepdim=True)
    rays_orig = c2w_target[:3, -1].clone().reshape(1, 3).expand(rays_dir.shape[0], -1)

    rays_pixs = torch.stack((ys, xs))  # row col

    return rays_orig, rays_dir, rays_pixs


def conver_to_ndc(ray_pts, w2c_ref, intrinsics_ref, W_H, depth_values):
    nb_rays, nb_samples = ray_pts.shape[:2]
    ray_pts = ray_pts.reshape(-1, 3)

    R = w2c_ref[:3, :3]  # (3, 3)
    T = w2c_ref[:3, 3:]  # (3, 1)
    ray_pts = torch.matmul(ray_pts, R.t()) + T.reshape(1, 3)

    ray_pts_ndc = ray_pts @ intrinsics_ref.t()

    # pts_depth_source_view = ray_pts_ndc[:, 2:]
    ray_pts_ndc[:, :2] = ray_pts_ndc[:, :2] / (
        ray_pts_ndc[:, -1:] * W_H.reshape(1, 2)
    )  # normalize x,y to 0~1

    grid = ray_pts_ndc[None, None, :, :2] * 2 - 1
    near = F.grid_sample(
        depth_values[:, :1],
        grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    ).squeeze()
    far = F.grid_sample(
        depth_values[:, -1:],
        grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    ).squeeze()
    ray_pts_ndc[:, 2] = (ray_pts_ndc[:, 2] - near) / (far - near)  # normalize z to 0~1

    ray_pts_ndc = ray_pts_ndc.view(nb_rays, nb_samples, 3)
    # pts_depth_source_view = pts_depth_source_view.view(nb_rays, nb_samples)
    # pts_depth_source_view = torch.abs(pts_depth_source_view.view(nb_rays, nb_samples))

    # return ray_pts_ndc, pts_depth_source_view
    return ray_pts_ndc


def get_sample_points(
    nb_coarse,
    nb_fine,
    near,
    far,
    rays_o,
    rays_d,
    nb_views,
    w2cs,
    intrinsics,
    depth_values,
    W_H,
    with_noise=False,
    rays_depth=None,
):
    # This is the option for using target depth to sample points, 
    # to use original geonerf sampling set this falses
    if rays_depth is not None:
        target_depth_guidance = True  
    else:
        target_depth_guidance = False

    device = rays_o.device
    nb_rays = rays_o.shape[0]

    with torch.no_grad():

        if target_depth_guidance:

            N_samples = nb_coarse + nb_fine
            t_vals = torch.linspace(0.0, 1.0, steps=nb_coarse).view(nb_coarse).to(device)
            pts_depth = near * (1.0 - t_vals) + far * (t_vals)
            pts_depth = pts_depth.expand([nb_rays, nb_coarse])
            expand_rays_depth = rays_depth.unsqueeze(1).repeat(1, nb_fine-1)
            x = torch.normal(mean=expand_rays_depth, std=0.5)
            x = torch.cat([pts_depth, x, rays_depth.unsqueeze(1)], dim=1)
            x1, indice = torch.sort(x, dim=1)
            pts_depth = torch.clamp(x1, min=near, max=far)

            # The last one is the predicted depth
            middle_pts_mask = (indice == (N_samples - 1))

            ray_pts = rays_o.unsqueeze(1) + pts_depth.unsqueeze(-1) * rays_d.unsqueeze(1)   #  [ray_samples 1+N_samples 3 ]

        else:
            ## Counting the number of source views for which the points are valid
            t_vals = torch.linspace(0.0, 1.0, steps=nb_coarse).view(1, nb_coarse).to(device)
            pts_depth = near * (1.0 - t_vals) + far * (t_vals)
            pts_depth = pts_depth.expand([nb_rays, nb_coarse])
            ray_pts = rays_o.unsqueeze(1) + pts_depth.unsqueeze(-1) * rays_d.unsqueeze(1)
            
            valid_points = torch.zeros([nb_rays, nb_coarse]).to(device)
            for idx in range(nb_views):
                w2c_ref, intrinsic_ref = w2cs[0, idx], intrinsics[0, idx]
                ray_pts_ndc = conver_to_ndc(
                    ray_pts,
                    w2c_ref,
                    intrinsic_ref,
                    W_H,
                    depth_values=depth_values[f"level_0"][:, idx],
                )
                valid_points += (
                    ((ray_pts_ndc >= 0) & (ray_pts_ndc <= 1)).sum(dim=-1) == 3
                ).float()

            ## Creating a distribution based on the counted values and sample more points
            if nb_fine > 0:
                point_distr = torch.distributions.categorical.Categorical(
                    logits=valid_points
                )
                t_vals = (
                    point_distr.sample([nb_fine]).t()
                    - torch.rand([nb_rays, nb_fine]).to(device)
                ) / (nb_coarse - 1)
                pts_depth_fine = near * (1.0 - t_vals) + far * (t_vals)

                pts_depth = torch.cat([pts_depth, pts_depth_fine], dim=-1)
                pts_depth, _ = torch.sort(pts_depth)

            if with_noise:  ## Add noise to sample points during training
                # get intervals between samples
                mids = 0.5 * (pts_depth[..., 1:] + pts_depth[..., :-1])
                upper = torch.cat([mids, pts_depth[..., -1:]], -1)
                lower = torch.cat([pts_depth[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(pts_depth.shape, device=device)
                pts_depth = lower + (upper - lower) * t_rand

            ray_pts = rays_o.unsqueeze(1) + pts_depth.unsqueeze(-1) * rays_d.unsqueeze(1)

        # middle_ray_pts_ndc = {"level_0": [], "level_1": [], "level_2": []}
        all_ray_pts_ndc = {"level_0": [], "level_1": [], "level_2": []}
        for idx in range(nb_views):
            w2c_ref, intrinsic_ref = w2cs[0, idx], intrinsics[0, idx]
            for l in range(3):
                    ray_pts_ndc = conver_to_ndc(
                        ray_pts,
                        w2c_ref,
                        intrinsic_ref,
                        W_H,
                        depth_values=depth_values[f"level_{l}"][:, idx],
                    )
                    # middle_ray_pts_ndc[f"level_{l}"].append(ray_pts_ndc[:,:1])
                    all_ray_pts_ndc[f"level_{l}"].append(ray_pts_ndc)
        for l in range(3):
            # middle_ray_pts_ndc[f"level_{l}"] = torch.stack(middle_ray_pts_ndc[f"level_{l}"], dim=2)
            all_ray_pts_ndc[f"level_{l}"] = torch.stack(all_ray_pts_ndc[f"level_{l}"], dim=2)
        if torch.isnan(all_ray_pts_ndc["level_0"]).any():
            print("NAN in all_ray_pts_ndc")

        return pts_depth, ray_pts, all_ray_pts_ndc, middle_pts_mask


def get_rays_pts(
    H,
    W,
    c2ws,
    w2cs,
    intrinsics,
    near_fars,
    depth_values,
    nb_coarse,
    nb_fine,
    nb_views,
    chunk=-1,
    chunk_idx=-1,
    train=False,
    train_batch_size=-1,
    target_img=None,
    target_depth=None,
    target_segmentation=None,
    depth_map=None,
):
    '''
    # TODO: Add documentation
    '''
    if train:
        if target_depth.sum() > 0:
            depth_mask = target_depth > 0
        else:
            depth_mask = None
    else:
        depth_mask = None

    rays_orig, rays_dir, rays_pixs = get_rays(
        H,
        W,
        intrinsics[0, -1],
        c2ws[0, -1],
        chunk=chunk,
        chunk_id=chunk_idx,
        train=train,
        train_batch_size=train_batch_size,
        mask=depth_mask,
    )

    rays_pixs_int = rays_pixs.long()

    ## Extracting estimated depth of target view
    if depth_map is not None:
        rays_depth = depth_map[rays_pixs_int[0], rays_pixs_int[1]]
    else:
        rays_depth = None

    ## Extracting ground truth color and depth of target view
    if train:
        rays_gt_rgb = target_img[:, rays_pixs_int[0], rays_pixs_int[1]].permute(1, 0)
        if target_segmentation is not None:
            rays_gt_semantic = target_segmentation[rays_pixs_int[0], rays_pixs_int[1]]
        else:
            rays_gt_semantic = None
        rays_gt_depth = target_depth[rays_pixs_int[0], rays_pixs_int[1]]
    else:
        rays_gt_rgb = None
        rays_gt_depth = None
        rays_gt_semantic = None

    # klevr dataset have scene boundaries, se can scale the rays and calculate the near/far
    # all klevr dataset have the same scene boundaries: "min": [-3.1, -3.1, -0.1], "max": [3.1, 3.1, 3.1]
    if near_fars.sum()==0:
    # if False:
        # rays_orig, rays_dir = scale_rays(rays_orig, rays_dir, np.array([[-3.1, -3.1, -0.1], [3.1, 3.1, 3.1]]), (H, W))
        print("calculate near/far")
        near, far = calculate_near_and_far(rays_orig, rays_dir, bbox_min=[-3.1, -3.1, -0.1], bbox_max=[3.1, 3.1, 3.1])
    else:
        near, far = near_fars[0, -1, 0], near_fars[0, -1, 1]  ## near/far of the target view
        # near, far = 3, 17  ## near/far of the target view
    assert near < far, "near should be smaller than far"
    
    # travel along the rays
    W_H = torch.tensor([W - 1, H - 1]).to(rays_orig.device)
    pts_depth, ray_pts, all_ray_pts_ndc, middle_pts_mask = get_sample_points(
        nb_coarse,
        nb_fine,
        near,
        far,
        rays_orig,
        rays_dir,
        nb_views,
        w2cs,
        intrinsics,
        depth_values,
        W_H,
        with_noise=train,
        rays_depth=rays_depth,
    )

    return (
        middle_pts_mask,
        pts_depth,
        ray_pts,
        all_ray_pts_ndc,
        rays_dir,
        rays_gt_rgb,
        rays_gt_semantic,
        rays_gt_depth,
        rays_pixs,
        rays_depth,
    )


def normal_vect(vect, dim=-1):
    return vect / (torch.sqrt(torch.sum(vect**2, dim=dim, keepdim=True)) + 1e-7)


def interpolate_3D(feats, pts_ndc, padding_mode='border'):
    H, W = pts_ndc.shape[-3:-1]
    grid = pts_ndc.view(-1, 1, H, W, 3) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
    # grid should be restrict to [-1,1], normal case don't matters, but when the grid is infinite, it will cause error
    eps = 1e-3
    grid = torch.clamp(grid, -1-eps, 1+eps)
    features = (
        F.grid_sample(
            feats, grid, align_corners=True, mode="bilinear", padding_mode=padding_mode
        )[:, :, 0]
        .permute(2, 3, 0, 1)
        .squeeze()
    )

    return features


def interpolate_2D(feats, imgs, semantic_feats, pts_ndc, padding_mode='border', use_batch_semantic_features=False):
    '''
    getting the features and images at the points / a image and feature
    feats: (1,8,256,256)
    semant: (1,21(number_class),256,256),
    imgs: (1,3,256,256)
    pts_ndc: (4096,128,3)
    return: (4096,128,8), (4096,128,3)
    '''
    H, W = pts_ndc.shape[-3:-1] 
    grid = pts_ndc[..., :2].view(-1, H, W, 2) * 2 - 1.0  # [1 H W 2] (x,y) (1,4096,128,2)
    # grid should be restrict to [-1,1], normal case don't matters, but when the grid is infinite, it will cause error
    eps = 1e-3
    grid = torch.clamp(grid, -1-eps, 1+eps)
    features = ( 
        F.grid_sample(
            feats, grid, align_corners=True, mode="bilinear", padding_mode=padding_mode
        )
        .permute(2, 3, 1, 0)
        .squeeze()
    )

    images = (
        F.grid_sample(
            imgs, grid, align_corners=True, mode="bilinear", padding_mode=padding_mode
        )
        .permute(2, 3, 1, 0)
        .squeeze()
    )
    if use_batch_semantic_features:
        feats_size = np.array(feats.shape[-2:], dtype=np.int32)
        semantic_feature_batch_size = 1
        semantic_feature_batch_size_ = (semantic_feature_batch_size / feats_size) * 2
        ys, xs = torch.meshgrid(
            semantic_feature_batch_size_[0]*torch.linspace(-semantic_feature_batch_size, semantic_feature_batch_size, 2*semantic_feature_batch_size+1), 
            semantic_feature_batch_size_[1]*torch.linspace(-semantic_feature_batch_size, semantic_feature_batch_size, 2*semantic_feature_batch_size+1), 
            indexing="ij"
        )
        batch_shift = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(grid.device)
        batch_grid = grid.unsqueeze(-2).repeat(1,1,1,9,1) + batch_shift
        # grid size: (1, 4096, 128*9, 2)
        batch_grid = batch_grid.view(grid.shape[0], grid.shape[1], -1, 2)
    else:
        batch_grid = grid

    semantic_features = (
        F.grid_sample(
            semantic_feats, batch_grid, align_corners=True, mode="bilinear", padding_mode=padding_mode
        )
        .permute(2, 3, 1, 0)
        .squeeze()
    )
    if use_batch_semantic_features:
        # semantic_features: (4096, 128, 9*21)
        semantic_features = semantic_features.reshape(pts_ndc.shape[0], pts_ndc.shape[1], -1)
        # semantic_features = semantic_features.mean(dim=2)

    with torch.no_grad():
        in_mask = (grid > -1.0) * (grid < 1.0) # (1,4096,128,2)
        in_mask = (in_mask[..., 0] * in_mask[..., 1]).float().permute(1, 2, 0) # (4096,128,1)

    return features, images, semantic_features, in_mask


def read_pfm(filename):
    file = open(filename, "rb")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    if src_grid == None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad > 0:
            H_pad, W_pad = H + pad * 2, W + pad * 2
        else:
            H_pad, W_pad = H, W

        if depth_values.dim() != 4:
            depth_values = depth_values[..., None, None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(
            H_pad, W_pad, normalized_coordinates=False, device=device
        )  # (1, H, W, 2)
        if pad > 0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat(
            (ref_grid, torch.ones_like(ref_grid[:, :1])), 1
        )  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.reshape(B, 1, D * W_pad * H_pad)
        # del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory

        src_grid = (
            src_grid_d[:, :2] / src_grid_d[:, 2:]
        )  # divide by depth (B, 2, D*H*W)
        # del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)
        # Important, if the grid value too small or too large cause NAN in grid_sample
        margin = 1e-1
        src_grid = torch.clamp(src_grid, -1.0-margin, 1.0+margin)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(
        src_feat,
        src_grid.view(B, D, W_pad * H_pad, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    if torch.isnan(warped_src_feat).any():
        print("nan in warped_src_feat")
    return warped_src_feat, src_grid

##### Functions for view selection
TINY_NUMBER = 1e-5  # float32 only has 7 decimal digits precision

def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(
        np.clip(np.sum(vec1_unit * vec2_unit, axis=-1), -1.0, 1.0)
    )
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    assert (
        R1.shape[-1] == 3
        and R2.shape[-1] == 3
        and R1.shape[-2] == 3
        and R2.shape[-2] == 3
    )
    return np.arccos(
        np.clip(
            (np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1)
            / 2.0,
            a_min=-1 + TINY_NUMBER,
            a_max=1 - TINY_NUMBER,
        )
    )

def compute_nearest_camera_indices(database, que_ids, ref_ids=None):
    if ref_ids is None: ref_ids = que_ids
    ref_poses = [database.get_pose(ref_id) for ref_id in ref_ids]
    ref_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in ref_poses])
    que_poses = [database.get_pose(que_id) for que_id in que_ids]
    que_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in que_poses])

    dists = np.linalg.norm(ref_cam_pts[None, :, :] - que_cam_pts[:, None, :], 2, 2)
    dists_idx = np.argsort(dists, 1)
    return dists_idx

def get_nearest_pose_ids(
    tar_pose,
    ref_poses,
    num_select,
    tar_id=-1,
    angular_dist_method="dist",
    scene_center=(0, 0, 0),
):
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams - 1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == "matrix":
        dists = batched_angular_dist_rot_matrix(
            batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3]
        )
    elif angular_dist_method == "vector":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == "dist":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception("unknown angular distance calculation method!")

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]

    return selected_ids

def downsample_gaussian_blur(img, ratio):
    sigma = (1 / ratio) / 3
    # ksize=np.ceil(2*sigma)
    ksize = int(np.ceil(((sigma - 0.8) / 0.3 + 1) * 2 + 1))
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT101)
    return img

def pose_inverse(pose):
    R = pose[:, :3].T
    t = - R @ pose[:, 3:]
    return np.concatenate([R, t], -1)


lable_color_map = np.array([
    [174, 199, 232],  # wall
    [152, 223, 138],  # floor
    [31, 119, 180],   # cabinet
    [255, 187, 120],  # bed
    [188, 189, 34],   # chair
    [140, 86, 75],    # sofa
    [255, 152, 150],  # table
    [214, 39, 40],    # door
    [197, 176, 213],  # window
    [148, 103, 189],  # bookshelf
    [196, 156, 148],  # picture
    [23, 190, 207],   # counter
    [247, 182, 210],  # desk
    [219, 219, 141],  # curtain
    [255, 127, 14],   # refrigerator
    [91, 163, 138],   # shower curtain
    [44, 160, 44],    # toilet
    [112, 128, 144],  # sink
    [227, 119, 194],  # bathtub
    [82, 84, 163],    # otherfurn
    [248, 166, 116],  # invalid
    ])/255.0
