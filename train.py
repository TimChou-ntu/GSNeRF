import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kornia

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers
from pytorch_lightning.loggers import WandbLogger

import os
import numpy as np
import sys
import imageio
import lpips
from skimage.metrics import structural_similarity as ssim
# from torchmetrics import JaccardIndex
from utils.metrics import IoU

from model.geo_reasoner import CasMVSNet
from model.self_attn_renderer import Renderer
from model.UNet import UNet
from utils.rendering import render_rays
from utils.utils import (
    load_ckpt,
    init_log,
    get_rays_pts,
    SL1Loss,
    self_supervision_loss,
    img2mse,
    mse2psnr,
    acc_threshold,
    abs_error,
    visualize_depth,
    seed_everything,
    lable_color_map,
)
from utils.optimizer import get_optimizer, get_scheduler
from utils.options import config_parser
from utils.depth_map import get_target_view_depth
from data.get_datasets import (
    get_training_dataset,
    get_finetuning_dataset,
    get_validation_dataset,
)
from utils.loss import SemanticLoss
from utils.depth_loss import UnsupLossMultiStage

lpips_fn = lpips.LPIPS(net="vgg")

def compute_depth_errors(gt, pred, var=None):
    thresh = np.maximum((gt / pred), (pred / gt))

    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_diff = np.mean(np.abs(gt - pred))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    irmse = (1/gt - 1/pred) ** 2
    irmse = np.sqrt(irmse.mean())

    if var is not None:
        var[var < 1e-6] = 1e-6
        nll = 0.5 * (np.log(var) + np.log(2*np.pi) + (np.square(gt - pred) / var))
        nll = np.mean(nll)
    else:
        nll = 0.0

    return dict(a1=a1, a2=a2, a3=a3,
                abs_diff=abs_diff,
                abs_rel=abs_rel, sq_rel=sq_rel,
                rmse=rmse, log_10=log_10, irmse=irmse,
                rmse_log=rmse_log, silog=silog,
                nll=nll)
    

class GSNeRF(LightningModule):
    def __init__(self, hparams):
        super(GSNeRF, self).__init__()
        self.validation_step_outputs = []
        self.hparams.update(vars(hparams))
        self.wr_cntr = 0

        self.depth_loss = SL1Loss()
        self.unsup_depth_loss = UnsupLossMultiStage()
        # (0: wall, 1: floor, 20: invalid)
        # 20 + 1 classes cause unbalanced data, minor
        weights = [hparams.background_weight, hparams.background_weight] + [1.0] * (hparams.nb_class - 3) + [hparams.background_weight]
        class_weights = torch.FloatTensor(weights)
        self.semantic_loss = SemanticLoss(nb_class=hparams.nb_class, ignore_label=hparams.ignore_label, weight=class_weights)
        self.semantic_feat_loss = nn.CrossEntropyLoss(ignore_index=hparams.ignore_label, weight=class_weights)
        self.target_depth_loss = nn.SmoothL1Loss(reduction="mean")
        self.learning_rate = hparams.lrate

        # Create geometry_reasoner and renderer models
        self.geo_reasoner = CasMVSNet(use_depth=hparams.use_depth, nb_class=hparams.nb_class, feat_net=hparams.feat_net)
        self.renderer = Renderer(nb_samples_per_ray=hparams.nb_coarse + hparams.nb_fine, 
                                 nb_view=hparams.nb_views, nb_class=hparams.nb_class,
                                 using_semantic_global_tokens=hparams.using_semantic_global_tokens,
                                 only_using_semantic_global_tokens=hparams.only_using_semantic_global_tokens,
                                 use_batch_semantic_feature=hparams.use_batch_semantic_feature,
                                )
        if hparams.target_depth_estimation & hparams.use_depth_refine_net:
            self.depth_refine_net = UNet(n_channels=1, n_classes=1)

        self.eval_metric = [0.01, 0.05, 0.1]

        self.miou = IoU(num_classes=hparams.nb_class-1, ignore_label=hparams.ignore_label)
        self.automatic_optimization = False
        self.save_hyperparameters()

    def unpreprocess(self, data, shape=(1, 1, 3, 1, 1)):
        # to unnormalize image for visualization
        device = data.device
        mean = (
            torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
            .view(*shape)
            .to(device)
        )
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

    def setup(self, stage=None):
        if self.hparams.scene == "None":  ## Generalizable
            self.train_dataset, self.train_sampler = get_training_dataset(self.hparams)
            self.val_dataset = get_validation_dataset(self.hparams)
        else:  ## Fine-tune
            self.train_dataset, self.train_sampler = get_finetuning_dataset(
                self.hparams
            )
            self.val_dataset = get_validation_dataset(self.hparams)

    def configure_optimizers(self):
        if self.hparams.target_depth_estimation & self.hparams.use_depth_refine_net:
            opt = get_optimizer(self.hparams, [self.geo_reasoner, self.renderer, self.depth_refine_net])
            
        else:
            opt = get_optimizer(self.hparams, [self.geo_reasoner, self.renderer])
            
        sch = get_scheduler(self.hparams, optimizer=opt)

        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            shuffle=True if self.train_sampler is None else False,
            num_workers=0,
            batch_size=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=1,
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        loss = 0
        nb_views = self.hparams.nb_views
        H, W = batch["images"].shape[-2:]
        H, W = int(H), int(W)

        ## Inferring Geometry Reasoner
        feats_vol, feats_fpn, depth_map, depth_values, semantic_logits, semantic_feats = self.geo_reasoner(
            imgs=batch["images"][:, :nb_views],
            affine_mats=batch["affine_mats"][:, :nb_views],
            affine_mats_inv=batch["affine_mats_inv"][:, :nb_views],
            near_far=batch["near_fars"][:, :nb_views],
            closest_idxs=batch["closest_idxs"][:, :nb_views],
            gt_depths=batch["depths_aug"][:, :nb_views],
        )

        ## Normalizing depth maps in NDC coordinate
        depth_map_norm = {}
        for l in range(3):
            depth_map_norm[f"level_{l}"] = (
                depth_map[f"level_{l}"].detach() - depth_values[f"level_{l}"][:, :, 0]
            ) / (
                depth_values[f"level_{l}"][:, :, -1]
                - depth_values[f"level_{l}"][:, :, 0]
            )

        ## currently scannet dataset no normalizing
        if self.hparams.dataset_name != 'scannet' and self.hparams.dataset_name != 'replica':
            unpre_imgs = self.unpreprocess(batch["images"])
        else:
            unpre_imgs = batch["images"]

        # Get the target depth map
        if self.hparams.target_depth_estimation:
            target_depth_estimation = get_target_view_depth(
                depth_map['level_0'][0, :nb_views],
                batch['intrinsics'][0, :nb_views],
                batch['c2ws'][0, :nb_views],
                batch['intrinsics'][0, nb_views],
                batch['w2cs'][0, nb_views],
                (W, H),
                3, # grid size set to 1 for now, can be changed to 2 or 4
            )
            target_depth_estimation = kornia.filters.bilateral_blur(target_depth_estimation.unsqueeze(0).unsqueeze(0), (3, 3), 1, (2, 2)).squeeze()
            
            if self.hparams.target_depth_estimation & self.hparams.use_depth_refine_net:
                target_depth_estimation = self.depth_refine_net(target_depth_estimation.unsqueeze(0).unsqueeze(0))['logits'].squeeze()
                near, far = batch["near_fars"][0, -1, 0], batch["near_fars"][0, -1, 1]  ## near/far of the target view
                target_depth_estimation = torch.nn.functional.sigmoid(target_depth_estimation)*(far-near)+near
                
            # target_depth_estimation = torch.ones_like(depth_map["level_0"][0, 0]).to("cuda")
            if torch.isnan(target_depth_estimation).any():
                print("nan in target depth estimation")
        else:
            target_depth_estimation = None

        (
            middle_pts_mask,
            pts_depth,
            rays_pts,
            rays_pts_ndc,
            rays_dir,
            rays_gt_rgb,
            rays_gt_semantic,
            rays_gt_depth,
            rays_pixs,
            rays_depth,
        ) = get_rays_pts(
            H,
            W,
            batch["c2ws"],
            batch["w2cs"],
            batch["intrinsics"],
            batch["near_fars"],
            depth_values,
            self.hparams.nb_coarse,
            self.hparams.nb_fine,
            nb_views=nb_views,
            train=True,
            train_batch_size=self.hparams.batch_size,
            target_img=unpre_imgs[0, -1],
            target_segmentation=batch["semantics"][0, -1], # (B, S, H, W)
            # This is GT
            target_depth=batch["depths_h"][0, -1],
            # This is estimation based on source view depth prediction
            depth_map=target_depth_estimation,
        )
        # debug
        # torch.save(rays_pts, f"../visualize_train_tmp_scale/rays_pts{batch_nb}.pt")

        ## Rendering
        rendered_rgb, rendered_semantic, rendered_depth = render_rays(
            c2ws=batch["c2ws"][0, :nb_views],
            rays_pts=rays_pts,
            rays_pts_ndc=rays_pts_ndc,
            pts_depth=pts_depth,
            rays_dir=rays_dir,
            feats_vol=feats_vol,
            feats_fpn=feats_fpn[:, :nb_views],
            imgs=unpre_imgs[:, :nb_views],
            depth_map_norm=depth_map_norm,
            renderer_net=self.renderer,
            middle_pts_mask=middle_pts_mask,
            # middle_ray_pts=middle_ray_pts,
            # middle_ray_pts_ndc=middle_ray_pts_ndc,
            # middle_pts_depth=middle_pts_depth,
            # semantic_net=self.semantic_net,
            semantic_feat=semantic_feats,
            use_batch_semantic_features=self.hparams.use_batch_semantic_feature,
        )

        # semantic_logits might be supervised by ground truth segmentation, batch size=1
        semantic_logits_loss = self.semantic_feat_loss(semantic_logits.squeeze(), batch["semantics"][0, :nb_views])

        # target_depth might be supervised by ground truth depth, batch size=1
        # target_depth_loss = self.target_depth_loss(target_depth_estimation, batch["depths_h"][0, -1])

        # Supervising depth maps with either ground truth depth or self-supervision loss
        # This loss is only used in the generalizable model
        # Not using right now
        if self.hparams.scene == "None":
        # if False:
            ## if ground truth is available
            if isinstance(batch["depths"], dict) and (not self.hparams.self_supervised_depth_loss):
            # if False:
                loss = loss + 1 * self.depth_loss(depth_map, batch["depths"])
                if loss != 0:
                    self.log("train/dlossgt", loss.item(), prog_bar=False,sync_dist =True)
            else:
                loss = loss + 0.1 * self_supervision_loss(
                    self.depth_loss,
                    rays_pixs,
                    rendered_depth.detach(),
                    depth_map,
                    rays_gt_rgb,
                    unpre_imgs,
                    rendered_rgb.detach(),
                    batch["intrinsics"],
                    batch["c2ws"],
                    batch["w2cs"],
                )
                if loss != 0:
                    self.log("train/dlosspgt", loss.item(), prog_bar=False,sync_dist =True)

                ## Supervising each depth_map
                RC_MVS_loss = 0.1 * self.unsup_depth_loss(
                    inputs=depth_map, 
                    imgs=unpre_imgs[:, :nb_views], 
                    cams=batch["project_mats"][:, :nb_views],
                )[0]
                loss = loss + RC_MVS_loss
                if RC_MVS_loss != 0:
                    self.log("train/RC_MVS_loss", RC_MVS_loss.item(), prog_bar=False,sync_dist =True)

        mask = rays_gt_depth > 0
        depth_available = mask.sum() > 0

        ## Supervising ray depths
        # if False:
        ## This loss is only used in the generalizable model
        if self.hparams.scene == "None":
            if self.hparams.self_supervised_depth_loss:
                loss = loss + 0.05 * self.depth_loss(rendered_depth, rays_depth.detach())
            else:
                loss = loss + 0.1 * self.depth_loss(rendered_depth, rays_gt_depth)

        self.log(
            f"train/acc_l_{self.eval_metric[0]}mm",
            acc_threshold(
                rendered_depth, rays_gt_depth, mask, self.eval_metric[0]
            ).mean(),
            prog_bar=False,sync_dist =True
        )
        self.log(
            f"train/acc_l_{self.eval_metric[1]}mm",
            acc_threshold(
                rendered_depth, rays_gt_depth, mask, self.eval_metric[1]
            ).mean(),
            prog_bar=False,sync_dist =True
        )
        self.log(
            f"train/acc_l_{self.eval_metric[2]}mm",
            acc_threshold(
                rendered_depth, rays_gt_depth, mask, self.eval_metric[2]
            ).mean(),
            prog_bar=False,sync_dist =True
        )

        abs_err = abs_error(rendered_depth, rays_gt_depth, mask).mean()
        self.log("train/abs_err", abs_err, prog_bar=False,sync_dist =True)

        ## Reconstruction loss
        if self.hparams.segmentation:
            # originally (N_rays, 1, nb_classes), view as (N_rays, nb_classes)
            rendered_semantic = rendered_semantic.view(rays_gt_semantic.shape[0], -1)
            croos_entropy_loss = self.semantic_loss(rendered_semantic, rays_gt_semantic)
        if torch.isnan(loss):
            print("depth loss is nan, skipping batch...")
        if torch.isnan(croos_entropy_loss):
            print("Nan semantic loss encountered, skipping batch...")

        mse_loss = img2mse(rendered_rgb, rays_gt_rgb)
        if torch.isnan(mse_loss):
            print("Nan mse loss encountered, skipping batch...")
        # loss = loss + mse_loss + croos_entropy_loss*0.1
        if self.hparams.segmentation:
            if self.global_step < self.hparams.two_stage_training_steps:
                loss = loss + mse_loss + semantic_logits_loss*self.hparams["cross_entropy_weight"]
            else:
                loss = loss + mse_loss + croos_entropy_loss*self.hparams["cross_entropy_weight"] + semantic_logits_loss*self.hparams["cross_entropy_weight"]
        else:
            loss = loss + mse_loss
        # loss = loss + mse_loss + croos_entropy_loss*0.1 + semantic_logits_loss*0.1 + target_depth_loss
        # loss = mse_loss + croos_entropy_loss*0.01
        if self.hparams.ddp:
            losses = self.all_gather(loss)
            if any(torch.isnan(loss) for loss in losses):
                print("skip nan loss")
                return None  # skip training step
        if torch.isnan(loss):
            print("Nan loss encountered, skipping batch...")
            img_vis = (
                torch.cat(
                    [unpre_imgs[:,i] for i in range(unpre_imgs.shape[1])],
                    dim=0,
                )
                .clip(0, 1)
                .permute(2, 0, 3, 1)
                .reshape(H, -1, 3)
                .cpu()
                .numpy()
            )
            os.makedirs(
                f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/nan_batch/",
                exist_ok=True,
            )
            imageio.imwrite(
                f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/nan_batch/{self.global_step:08d}_{batch_nb:04d}.png",
                (img_vis * 255).astype("uint8"),
            )
            return {"loss": torch.nan_to_num(loss, nan=0.0)}
            return None
        else:
            with torch.no_grad():
                self.log("train/loss", loss.item(), prog_bar=True,sync_dist =True)
                psnr = mse2psnr(mse_loss.detach())
                self.log("train/PSNR", psnr.item(), prog_bar=False,sync_dist =True)
                self.log("train/img_mse_loss", mse_loss.item(), prog_bar=False,sync_dist =True)
                self.log("train/semantic_loss", croos_entropy_loss.item(), prog_bar=False,sync_dist =True)
                self.log("train/semantic_feats_loss", semantic_logits_loss.item(), prog_bar=False,sync_dist =True)

            # Manual Optimization
            opt = self.optimizers()
            sch = self.lr_schedulers()

            self.manual_backward(loss)
            # clip gradients, not sure whether gradient explosion will happen
            self.clip_gradients(opt, gradient_clip_val=2, gradient_clip_algorithm="value")

            # Warming up the learning rate
            if self.trainer.global_step < self.hparams.warmup_steps:
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
                )
                for pg in opt.param_groups:
                    pg["lr"] = lr_scale * self.learning_rate

            self.log("train/lr", opt.param_groups[0]["lr"], prog_bar=False,sync_dist =True)
            
            opt.step()
            sch.step()
            opt.zero_grad()

            del target_depth_estimation, rendered_depth, rendered_rgb, rendered_semantic, rays_gt_depth, rays_gt_rgb, rays_gt_semantic, rays_pixs

            return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        ## This makes Batchnorm to behave like InstanceNorm
        self.geo_reasoner.train()

        log_keys = [
            "val_psnr",
            "val_ssim",
            "val_lpips",
            "val_depth_loss_r",
            "val_abs_err",
            "mask_sum",
        ] + [f"val_acc_{i}mm" for i in self.eval_metric]
        loss = {}
        loss = init_log(loss, log_keys)

        H, W = batch["images"].shape[-2:]
        H, W = int(H), int(W)

        nb_views = self.hparams.nb_views

        with torch.no_grad():
            ## Inferring Geometry Reasoner
            feats_vol, feats_fpn, depth_map, depth_values, semantic_logits, semantic_feats = self.geo_reasoner(
                imgs=batch["images"][:, :nb_views],
                affine_mats=batch["affine_mats"][:, :nb_views],
                affine_mats_inv=batch["affine_mats_inv"][:, :nb_views],
                near_far=batch["near_fars"][:, :nb_views],
                closest_idxs=batch["closest_idxs"][:, :nb_views],
                gt_depths=batch["depths_aug"][:, :nb_views],
            )

            ## Normalizing depth maps in NDC coordinate
            depth_map_norm = {}
            for l in range(3):
                depth_map_norm[f"level_{l}"] = (
                    depth_map[f"level_{l}"] - depth_values[f"level_{l}"][:, :, 0]
                ) / (
                    depth_values[f"level_{l}"][:, :, -1]
                    - depth_values[f"level_{l}"][:, :, 0]
                )

            ## currently scannet dataset no normalizing
            if self.hparams.dataset_name != 'scannet' and self.hparams.dataset_name != 'replica':
                unpre_imgs = self.unpreprocess(batch["images"])
            else:
                unpre_imgs = batch["images"]
                
            # Get the target depth map
            if self.hparams.target_depth_estimation:
                target_depth_estimation = get_target_view_depth(
                    depth_map['level_0'][0, :nb_views],
                    batch['intrinsics'][0, :nb_views],
                    batch['c2ws'][0, :nb_views],
                    batch['intrinsics'][0, nb_views],
                    batch['w2cs'][0, nb_views],
                    (W, H),
                    3, # grid size set to 1 for now, can be changed to 2 or 4
                )
                # Bilateral filtering
                target_depth_estimation = kornia.filters.bilateral_blur(target_depth_estimation.unsqueeze(0).unsqueeze(0), (3, 3), 1, (2, 2)).squeeze()

                if self.hparams.target_depth_estimation & self.hparams.use_depth_refine_net:
                    target_depth_estimation = self.depth_refine_net(target_depth_estimation.unsqueeze(0).unsqueeze(0))['logits'].squeeze()
                    near, far = batch["near_fars"][0, -1, 0], batch["near_fars"][0, -1, 1]  ## near/far of the target view
                    target_depth_estimation = torch.nn.functional.sigmoid(target_depth_estimation)*(far-near)+near
            else:
                target_depth_estimation = None

            rendered_rgb, rendered_semantic, rendered_depth = [], [], []
            for chunk_idx in range(
                H * W // self.hparams.chunk + int(H * W % self.hparams.chunk > 0)
            ):
                middle_pts_mask, pts_depth, rays_pts, rays_pts_ndc, rays_dir, _, _, _, _, _ = get_rays_pts(
                    H,
                    W,
                    batch["c2ws"],
                    batch["w2cs"],
                    batch["intrinsics"],
                    batch["near_fars"],
                    depth_values,
                    self.hparams.nb_coarse,
                    # 256,
                    self.hparams.nb_fine,
                    nb_views=nb_views,
                    chunk=self.hparams.chunk,
                    chunk_idx=chunk_idx,
                    depth_map=target_depth_estimation,
                )

                # Rendering
                rend_rgb, ren_semantic, rend_depth = render_rays(
                    c2ws=batch["c2ws"][0, :nb_views],
                    rays_pts=rays_pts,
                    rays_pts_ndc=rays_pts_ndc,
                    pts_depth=pts_depth,
                    rays_dir=rays_dir,
                    feats_vol=feats_vol,
                    feats_fpn=feats_fpn[:, :nb_views],
                    imgs=unpre_imgs[:, :nb_views],
                    depth_map_norm=depth_map_norm,
                    renderer_net=self.renderer,
                    middle_pts_mask=middle_pts_mask,
                    semantic_feat=semantic_feats,
                    use_batch_semantic_features=self.hparams.use_batch_semantic_feature,
                )
                rendered_rgb.append(rend_rgb)
                rendered_semantic.append(ren_semantic)
                rendered_depth.append(rend_depth)

            rendered_rgb = torch.clamp(
                torch.cat(rendered_rgb).reshape(H, W, 3).permute(2, 0, 1), 0, 1
            )
            rendered_semantic = torch.cat(rendered_semantic).reshape(H, W, -1).permute(2, 0, 1)
            rendered_depth = torch.cat(rendered_depth).reshape(H, W)

            ## Check if there is any ground truth depth information for the dataset
            depth_available = batch["depths_h"].sum() > 0

            ## Evaluate only on pixels with meaningful ground truth depths
            if depth_available:
                mask = batch["depths_h"] > 0
                img_gt_masked = (unpre_imgs[0, -1] * mask[0, -1][None]).cpu()
                rendered_rgb_masked = (rendered_rgb * mask[0, -1][None]).cpu()
            else:
                img_gt_masked = unpre_imgs[0, -1].cpu()
                rendered_rgb_masked = rendered_rgb.cpu()

            unpre_imgs = unpre_imgs.cpu()
            rendered_rgb, rendered_depth = rendered_rgb.cpu(), rendered_depth.cpu()
            rendered_semantic_pred = torch.argmax(rendered_semantic, dim=0).cpu()
            pred_imgs = torch.from_numpy(lable_color_map[rendered_semantic_pred]).permute(2, 0, 1)
            gt_img = torch.from_numpy(lable_color_map[batch["semantics"][0, -1].cpu()]).permute(2, 0, 1)
            img_err_abs = (rendered_rgb_masked - img_gt_masked).abs()
            semantic_logits_img = torch.from_numpy(lable_color_map[torch.argmax(semantic_logits, dim=2).cpu()])[0].permute(0, 3, 1, 2)
            semantic_gt_img = torch.from_numpy(lable_color_map[batch["semantics"][0, :nb_views].cpu()]).permute(0, 3, 1, 2)
            ## Compute miou
            if self.hparams.segmentation:
                iou_score = self.miou(
                    true_labels=batch["semantics"][0, -1].reshape(-1),
                    predicted_labels=rendered_semantic_pred.reshape(-1),
                )
                loss["val_miou"] = iou_score["miou"].clone().detach()
                loss["val_acc"] = iou_score["total_accuracy"].clone().detach()
                loss["val_class_acc"] = iou_score["class_average_accuracy"].clone().detach()
            depth_target = batch["depths_h"][0, -1].cpu()
            mask_target = depth_target > 0

            if depth_available:
                loss["val_psnr"] = mse2psnr(torch.mean(img_err_abs[:, mask_target] ** 2))
            else:
                loss["val_psnr"] = mse2psnr(torch.mean(img_err_abs**2))
            loss["val_ssim"] = ssim(
                rendered_rgb_masked.permute(1, 2, 0).numpy(),
                img_gt_masked.permute(1, 2, 0).numpy(),
                data_range=1,
                channel_axis=2,
            )
            loss["val_lpips"] = lpips_fn(
                rendered_rgb_masked[None] * 2 - 1, img_gt_masked[None] * 2 - 1
            ).item()  # Normalize to [-1,1]

            if depth_available:
                loss["val_abs_err"] = abs_error(
                    rendered_depth, depth_target, mask_target
                ).sum()
                loss[f"val_acc_{self.eval_metric[0]}mm"] = acc_threshold(
                    rendered_depth, depth_target, mask_target, self.eval_metric[0]
                ).sum()
                loss[f"val_acc_{self.eval_metric[1]}mm"] = acc_threshold(
                    rendered_depth, depth_target, mask_target, self.eval_metric[1]
                ).sum()
                loss[f"val_acc_{self.eval_metric[2]}mm"] = acc_threshold(
                    rendered_depth, depth_target, mask_target, self.eval_metric[2]
                ).sum()
                loss["mask_sum"] = mask_target.float().sum()
                
                d1 = compute_depth_errors(depth_target[mask_target].numpy(), target_depth_estimation[mask_target].cpu().numpy())
                d2 = compute_depth_errors(depth_target[mask_target].numpy(), rendered_depth[mask_target].numpy())
                for key in d1.keys():
                    loss[f"val_{key}_estimate"] = d1[key]
                    loss[f"val_{key}_render"] = d2[key]
            
            depth_minmax = [
                0.9 * batch["near_fars"].min().detach().cpu().numpy(),
                1.1 * batch["near_fars"].max().detach().cpu().numpy(),
            ]
            # make sure the folder exists
            if self.hparams.eval:
                folder = "evaluation_"
            else:
                folder = "prediction_"
            os.makedirs(
                f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/{folder}/{self.global_step:08d}/",
                exist_ok=True,
            )

            # Visualize target comparison
            if "target" in self.hparams.val_save_img_type:

                semantic_vis = (
                    semantic_logits_img
                    .clip(0, 1)
                    .permute(2, 0, 3, 1)
                    .reshape(H, -1, 3)
                    .numpy()
                )
                semantic_gt_vis = (
                    semantic_gt_img
                    .clip(0, 1)
                    .permute(2, 0, 3, 1)
                    .reshape(H, -1, 3)
                    .numpy()
                )
                imageio.imwrite(
                    f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/{folder}/{self.global_step:08d}/{self.wr_cntr:02d}_semantic.png",
                    (semantic_vis * 255).astype("uint8"),
                )
                imageio.imwrite(
                    f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/{folder}/{self.global_step:08d}/{self.wr_cntr:02d}_semantic_gt.png",
                    (semantic_gt_vis * 255).astype("uint8"),
                )
                rendered_depth_vis, _ = visualize_depth(rendered_depth)

                img_vis = (
                    torch.cat(
                        (
                            unpre_imgs[:, -1],
                            torch.stack([rendered_rgb, pred_imgs, gt_img, img_err_abs * 5]),
                            rendered_depth_vis[None],
                        ),
                        dim=0,
                    )
                    .clip(0, 1)
                    .permute(2, 0, 3, 1)
                    .reshape(H, -1, 3)
                    .numpy()
                )


                imageio.imwrite(
                    f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/{folder}/{self.global_step:08d}/{self.wr_cntr:02d}.png",
                    (img_vis * 255).astype("uint8"),
                )
            
            if "depth" in self.hparams.val_save_img_type:
                depth_map_vis = []
                gt_depth_vis = []
                target_depth = []
                filtered_depth = []
                
                # all images (source and target)
                for i in range(batch["depths_h"].shape[1]):
                    gt_depth_vis.append(visualize_depth(batch["depths_h"][0, i])[0])
               
                # only source image
                for i in range(depth_map['level_0'].shape[1]):
                    depth_map_vis.append(visualize_depth(depth_map['level_0'][0, i])[0])

                # only target image
                target_depth.append(visualize_depth(target_depth_estimation)[0])
                
                # uncomment this for visualization of GT depth maps
                depth_vis = (
                    torch.cat(
                        (
                            torch.stack(gt_depth_vis),
                        ),
                        dim=0,
                    )
                    .clip(0, 1)
                    .permute(2, 0, 3, 1)
                    .reshape(H, -1, 3)
                    .numpy()
                )

                imageio.imwrite(
                    f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/{folder}/{self.global_step:08d}/depth_{self.wr_cntr:02d}.png",
                    (depth_vis * 255).astype("uint8"),
                )

                depth_map_vis_ = (
                    torch.cat(
                        (
                            torch.stack(depth_map_vis),
                        ),
                        dim=0,
                    )
                    .clip(0, 1)
                    .permute(2, 0, 3, 1)
                    .reshape(H, -1, 3)
                    .numpy()
                )

                imageio.imwrite(
                    f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/{folder}/{self.global_step:08d}/depthmap_{self.wr_cntr:02d}.png",
                    (depth_map_vis_ * 255).astype("uint8"),
                )

                target_depth_vis = (
                    torch.cat(
                        (
                            torch.stack(target_depth),
                        ),
                        dim=0,
                    )
                    .clip(0, 1)
                    .permute(2, 0, 3, 1)
                    .reshape(H, -1, 3)
                    .numpy()
                )

                imageio.imwrite(
                    f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/{folder}/{self.global_step:08d}/target_depth_{self.wr_cntr:02d}.png",
                    (target_depth_vis * 255).astype("uint8"),
                )


            if "source" in self.hparams.val_save_img_type:
                original_img_vis = []
                for i in range(unpre_imgs.shape[1]):
                    original_img_vis.append(unpre_imgs[0, i])
                unpre_imgs_vis = (
                    torch.cat(
                        (
                            torch.stack(original_img_vis),
                        ),
                        dim=0,
                    )
                    .clip(0, 1)
                    .permute(2, 0, 3, 1)
                    .reshape(H, -1, 3)
                    .numpy()
                )


                imageio.imwrite(
                    f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/prediction_/{self.global_step:08d}/source_{self.wr_cntr:02d}.png",
                    (unpre_imgs_vis * 255).astype("uint8"),
                )

            self.wr_cntr += 1
        self.validation_step_outputs.append(loss)
        del target_depth_estimation, rendered_depth, rendered_rgb, rendered_semantic
        return loss

    def on_validation_epoch_end(self):
        # recount the number of rendered images
        print(f"Image {self.wr_cntr:02d} rendered.")
        self.wr_cntr = 0
        outputs = self.validation_step_outputs
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()
        if self.hparams.segmentation:
            val_results = {}
            mean_miou = torch.stack([x["val_miou"] for x in outputs]).mean()
            mean_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
            mean_class_acc = torch.stack([x["val_class_acc"] for x in outputs]).mean()
            try:
                val_results["val_miou"] = torch.stack([x["val_miou"] for x in outputs]).reshape(-1, 10).mean(1).tolist()
                val_results["val_acc"] = torch.stack([x["val_acc"] for x in outputs]).reshape(-1, 10).mean(1).tolist()
                val_results["val_class_acc"] = torch.stack([x["val_class_acc"] for x in outputs]).reshape(-1, 10).mean(1).tolist()
                with open(os.path.join(self.hparams.logdir,self.hparams.dataset_name,self.hparams.expname,'val_results.json'), 'w') as f:
                    f.write(str(val_results).replace('\'', '\"'))
            except:
                pass
        mean_ssim = np.stack([x["val_ssim"] for x in outputs]).mean()
        mean_lpips = np.stack([x["val_lpips"] for x in outputs]).mean()
        mask_sum = torch.stack([x["mask_sum"] for x in outputs]).sum()
        mean_d_loss_r = torch.stack([x["val_depth_loss_r"] for x in outputs]).mean()
        mean_abs_err = torch.stack([x["val_abs_err"] for x in outputs]).sum() / mask_sum
        mean_acc_1mm = (
            torch.stack([x[f"val_acc_{self.eval_metric[0]}mm"] for x in outputs]).sum()
            / mask_sum
        )
        mean_acc_2mm = (
            torch.stack([x[f"val_acc_{self.eval_metric[1]}mm"] for x in outputs]).sum()
            / mask_sum
        )
        mean_acc_4mm = (
            torch.stack([x[f"val_acc_{self.eval_metric[2]}mm"] for x in outputs]).sum()
            / mask_sum
        )
        self.log("val/PSNR", mean_psnr, prog_bar=False,sync_dist =True)
        self.log("val/SSIM", mean_ssim, prog_bar=False,sync_dist =True)
        self.log("val/LPIPS", mean_lpips, prog_bar=False,sync_dist =True)
        if self.hparams.segmentation:
            self.log("val/mIoU", mean_miou, prog_bar=False,sync_dist =True)
            self.log("val/m_acc", mean_acc, prog_bar=False,sync_dist =True)
            self.log("val/m_class_acc", mean_class_acc, prog_bar=False,sync_dist =True)
        if mask_sum > 0:
            self.log("val/d_loss_r", mean_d_loss_r, prog_bar=False,sync_dist =True)
            self.log("val/abs_err", mean_abs_err, prog_bar=False,sync_dist =True)
            self.log(f"val/acc_{self.eval_metric[0]}mm", mean_acc_1mm, prog_bar=False,sync_dist =True)
            self.log(f"val/acc_{self.eval_metric[1]}mm", mean_acc_2mm, prog_bar=False,sync_dist =True)
            self.log(f"val/acc_{self.eval_metric[2]}mm", mean_acc_4mm, prog_bar=False,sync_dist =True)

        with open(
            f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/{self.hparams.expname}_metrics.txt",
            "w",
        ) as metric_file:
            metric_file.write(f"PSNR: {mean_psnr}\n")
            metric_file.write(f"SSIM: {mean_ssim}\n")
            metric_file.write(f"LPIPS: {mean_lpips}")

        self.validation_step_outputs.clear()  # free memory
        del outputs
        return


if __name__ == "__main__":
    # torch.set_default_dtype(torch.float32)
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_float32_matmul_precision(precision="high")
    args = config_parser()

    ## Setting seeds
    print(f"Setting seeds to {args.seed}")
    seed_everything(args.seed)
    pl.seed_everything(args.seed)
    
    gsnerf = GSNeRF(args)

    ckpt_file = None


    ## Setting a callback to automatically save checkpoints
    checkpoint_callback = ModelCheckpoint(
        f"{args.logdir}/{args.dataset_name}/{args.expname}/ckpts",
        filename="ckpt_step-{step:06d}",
        auto_insert_metric_name=False,
        monitor="val/mIoU",
        save_top_k=2,
    )

    ## Setting up a logger
    if args.logger == "wandb":
        logger = WandbLogger(
            name=args.expname,
            project="gsnerf",
            save_dir=f"{args.logdir}/{args.dataset_name}",
            resume="allow",
            id=args.expname,
        )
    elif args.logger == "tensorboard":
        logger = loggers.TensorBoardLogger(
            save_dir=f"{args.logdir}/{args.dataset_name}/{args.expname}",
            name=args.expname + "_logs",
        )
    else:
        logger = None

    # args.use_amp = False if args.eval else True
    args.use_amp = False 
    trainer = Trainer(
        accelerator="auto",
        max_steps=args.num_steps,
        callbacks=[checkpoint_callback],
        logger=logger,
        num_sanity_val_steps=2,
        # val_check_interval=0.3,
        val_check_interval=0.005,
        benchmark=True,
        precision='16-mixed' if args.use_amp else 32,
        strategy= 'ddp' if args.ddp else 'auto',
    )

    if not args.eval:  ## Train
        if args.finetune:
            # specify the path of your generalized nerf checkpoint
            ckpt_file = "pretrained_weights/pretrained.ckpt"
            print(f"load pretrained weights from scannet_{ckpt_file}")
            load_ckpt(gsnerf.geo_reasoner, ckpt_file, "geo_reasoner")
            load_ckpt(gsnerf.renderer, ckpt_file, "renderer")
        else:  ## Generalizable
            print("!!! NOT Loading pretrained weights from Cascade MVSNet!!!")

        trainer.fit(gsnerf)
    else:  ## Eval
        gsnerf = GSNeRF(args)

        if ckpt_file is None:
            if args.use_depth:
                ckpt_file = "pretrained_weights/pretrained_w_depth.ckpt"
            else:
                ckpt_file = args.ckpt_path

        trainer.validate(gsnerf, ckpt_path=ckpt_file)