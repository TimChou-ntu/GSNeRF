import torch
import torch.nn.functional as F

from utils.utils import normal_vect, interpolate_3D, interpolate_2D


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []

        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)
        self.freq_bands = freq_bands.reshape(1, -1, 1)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))

        self.embed_fns = embed_fns

    def embed(self, inputs):
        repeat = inputs.dim() - 1
        self.freq_bands = self.freq_bands.to(inputs.device)
        inputs_scaled = (
            inputs.unsqueeze(-2) * self.freq_bands.view(*[1] * repeat, -1, 1)
        ).reshape(*inputs.shape[:-1], -1)
        inputs_scaled = torch.cat(
            (inputs, torch.sin(inputs_scaled), torch.cos(inputs_scaled)), dim=-1
        )
        return inputs_scaled


def get_embedder(multires=4):

    embed_kwargs = {
        "include_input": True,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed


def sigma2weights(sigma):
    alpha = 1.0 - torch.exp(-sigma)
    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )[:, :-1]
    weights = alpha * T

    return weights


def volume_rendering(rgb_sigma, pts_depth):
    rgb = rgb_sigma[..., :3]
    weights = sigma2weights(rgb_sigma[..., 3])

    rendered_rgb = torch.sum(weights[..., None] * rgb, -2)
    rendered_depth = torch.sum(weights * pts_depth, -1)

    return rendered_rgb, rendered_depth


def get_angle_wrt_src_cams(c2ws, rays_pts, rays_dir_unit):
    nb_rays = rays_pts.shape[0]
    ## Unit vectors from source cameras to the points on the ray
    dirs = normal_vect(rays_pts.unsqueeze(2) - c2ws[:, :3, 3][None, None])
    ## Cosine of the angle between two directions
    angle_cos = torch.sum(
        dirs * rays_dir_unit.reshape(nb_rays, 1, 1, 3), dim=-1, keepdim=True
    )
    # Cosine to Sine and approximating it as the angle (angle << 1 => sin(angle) = angle)
    angle = (1 - (angle_cos**2)).abs().sqrt()

    return angle


def interpolate_pts_feats(imgs, feats_fpn, semantic_feat, feats_vol, rays_pts_ndc, padding_mode='border', use_batch_semantic_features=False):
    nb_views = feats_fpn.shape[1]
    interpolated_feats = []

    for i in range(nb_views):
        ray_feats_0 = interpolate_3D(feats_vol[f"level_0"][:, i], rays_pts_ndc[f"level_0"][:, :, i], padding_mode=padding_mode)
        ray_feats_1 = interpolate_3D(
            feats_vol[f"level_1"][:, i], rays_pts_ndc[f"level_1"][:, :, i], padding_mode=padding_mode
        )
        ray_feats_2 = interpolate_3D(
            feats_vol[f"level_2"][:, i], rays_pts_ndc[f"level_2"][:, :, i], padding_mode=padding_mode
        )
        ray_feats_fpn, ray_colors, ray_semantic_feats_fpn, ray_masks = interpolate_2D(
            feats_fpn[:, i], imgs[:, i], semantic_feat[:, i], rays_pts_ndc[f"level_0"][:, :, i], padding_mode=padding_mode, use_batch_semantic_features=use_batch_semantic_features
        )
        # When using only one point per ray, all features except ray masks are 2D (N_rays, C), while ray masks are 3D (N_rays, C, 1), so we need to squeeze it
        if ray_colors.dim() == 2 and ray_masks.dim() == 3:
            ray_masks = ray_masks.squeeze(-1)

        interpolated_feats.append(
            torch.cat(
                [
                    ray_feats_0,                # (N_rays, N_samples, 8)
                    ray_feats_1,                # (N_rays, N_samples, 8)
                    ray_feats_2,                # (N_rays, N_samples, 8)
                    ray_feats_fpn,              # (N_rays, N_samples, 8)
                    ray_colors,                 # (N_rays, N_samples, 3)
                    ray_semantic_feats_fpn,     # (N_rays, N_samples, nb_classes(21)) / if use_batch_semantic_features, (N_rays, N_samples, 9*nb_classes(21))
                    ray_masks,                  # (N_rays, N_samples, 1)
                ],
                dim=-1,
            )
        )
    interpolated_feats = torch.stack(interpolated_feats, dim=2)
    if torch.isnan(interpolated_feats).any():
        print("interpolated_feats has nan values")
    return interpolated_feats


def get_occ_masks(depth_map_norm, rays_pts_ndc, visibility_thr=0.2):
    nb_views = depth_map_norm["level_0"].shape[1]
    z_diff = []
    for i in range(nb_views):
        ## Interpolate depth maps corresponding to each sample point
        # [1 H W 3] (x,y,z)
        grid = rays_pts_ndc[f"level_0"][None, :, :, i, :2] * 2 - 1.0
        rays_depths = F.grid_sample(
            depth_map_norm["level_0"][:, i : i + 1],
            grid,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )[0, 0]
        z_diff.append(rays_pts_ndc["level_0"][:, :, i, 2] - rays_depths)
    z_diff = torch.stack(z_diff, dim=2)

    occ_masks = z_diff.unsqueeze(-1) < visibility_thr

    return occ_masks


def render_rays(
    c2ws,
    rays_pts,
    rays_pts_ndc,
    pts_depth,
    rays_dir,
    feats_vol,
    feats_fpn,
    imgs,
    depth_map_norm,
    renderer_net,
    middle_pts_mask,
    semantic_feat=None,
    use_batch_semantic_features=False,
):
    ## The angles between the ray and source camera vectors
    rays_dir_unit = rays_dir / torch.norm(rays_dir, dim=-1, keepdim=True)
    angles = get_angle_wrt_src_cams(c2ws, rays_pts, rays_dir_unit)

    ## Positional encoding
    embedded_angles = get_embedder()(angles)

    ## Interpolate all features for sample points (N_rays, N_samples, source_view_num, 8+8+8+8+3+21+1)
    pts_feat = interpolate_pts_feats(imgs, feats_fpn, semantic_feat, feats_vol, rays_pts_ndc, use_batch_semantic_features=use_batch_semantic_features)

    ## Getting Occlusion Masks based on predicted depths (N_rays, N_samples, source_view_num, 1)
    occ_masks = get_occ_masks(depth_map_norm, rays_pts_ndc)

    ## rendering sigma and RGB values
    rgb_sigma, rendered_semantic = renderer_net(embedded_angles, pts_feat, occ_masks, middle_pts_mask)

    rendered_rgb, rendered_depth = volume_rendering(rgb_sigma, pts_depth)

    if torch.isnan(rendered_semantic).sum() > 0:
        print("NaN in rendered_semantic")

    return rendered_rgb, rendered_semantic, rendered_depth
