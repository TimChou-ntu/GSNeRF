import torch
from torch.utils.data import Dataset
import json
import numpy as np
import random
import time
import os
import cv2
from glob import glob as glob
from PIL import Image
from torchvision import transforms as T

from utils.utils import read_pfm, get_nearest_pose_ids, get_rays, compute_nearest_camera_indices
from utils.scannet_utils import parse_database_name, get_database_split, get_coords_mask, random_crop, random_flip, build_imgs_info, pad_imgs_info, imgs_info_to_torch, imgs_info_slice

def set_seed(index,is_train):
    if is_train:
        np.random.seed((index+int(time.time()))%(2**16))
        random.seed((index+int(time.time()))%(2**16)+1)
        torch.random.manual_seed((index+int(time.time()))%(2**16)+1)
    else:
        np.random.seed(index % (2 ** 16))
        random.seed(index % (2 ** 16) + 1)
        torch.random.manual_seed(index % (2 ** 16) + 1)

def add_depth_offset(depth,mask,region_min,region_max,offset_min,offset_max,noise_ratio,depth_length):
    coords = np.stack(np.nonzero(mask), -1)[:, (1, 0)]
    length = np.max(coords, 0) - np.min(coords, 0)
    center = coords[np.random.randint(0, coords.shape[0])]
    lx, ly = np.random.uniform(region_min, region_max, 2) * length
    diff = coords - center[None, :]
    mask0 = np.abs(diff[:, 0]) < lx
    mask1 = np.abs(diff[:, 1]) < ly
    masked_coords = coords[mask0 & mask1]
    global_offset = np.random.uniform(offset_min, offset_max) * depth_length
    if np.random.random() < 0.5:
        global_offset = -global_offset
    local_offset = np.random.uniform(-noise_ratio, noise_ratio, masked_coords.shape[0]) * depth_length + global_offset
    depth[masked_coords[:, 1], masked_coords[:, 0]] += local_offset

def build_src_imgs_info_select(database, ref_ids, ref_ids_all, cost_volume_nn_num, pad_interval=-1):
    # ref_ids - selected ref ids for rendering
    ref_idx_exp = compute_nearest_camera_indices(database, ref_ids, ref_ids_all)
    ref_idx_exp = ref_idx_exp[:, 1:1 + cost_volume_nn_num]
    ref_ids_all = np.asarray(ref_ids_all)
    ref_ids_exp = ref_ids_all[ref_idx_exp]  # rfn,nn
    ref_ids_exp_ = ref_ids_exp.flatten()
    ref_ids = np.asarray(ref_ids)
    ref_ids_in = np.unique(np.concatenate([ref_ids_exp_, ref_ids]))  # rfn'
    mask0 = ref_ids_in[None, :] == ref_ids[:, None]  # rfn,rfn'
    ref_idx_, ref_idx = np.nonzero(mask0)
    ref_real_idx = ref_idx[np.argsort(ref_idx_)]  # sort

    rfn, nn = ref_ids_exp.shape
    mask1 = ref_ids_in[None, :] == ref_ids_exp.flatten()[:, None]  # nn*rfn,rfn'
    ref_cv_idx_, ref_cv_idx = np.nonzero(mask1)
    ref_cv_idx = ref_cv_idx[np.argsort(ref_cv_idx_)]  # sort
    ref_cv_idx = ref_cv_idx.reshape([rfn, nn])
    is_aligned = not database.database_name.startswith('space')
    ref_imgs_info = build_imgs_info(database, ref_ids_in, pad_interval, is_aligned)
    return ref_imgs_info, ref_cv_idx, ref_real_idx


class RendererDataset(Dataset):
    default_cfg={
        'train_database_types':['scannet'],
        'type2sample_weights': {'scannet': 1},
        'val_database_name': 'scannet/scene0200_00/black_320',
        'val_database_split_type': 'val',

        'min_wn': 8,
        'max_wn': 9,
        'ref_pad_interval': 16,
        'train_ray_num': 512,
        'foreground_ratio': 0.5,
        'resolution_type': 'lr',
        "use_consistent_depth_range": True,
        'use_depth_loss_for_all': False,
        "use_depth": True,
        "use_src_imgs": False,
        "cost_volume_nn_num": 3,

        "aug_depth_range_prob": 0.05,
        'aug_depth_range_min': 0.95,
        'aug_depth_range_max': 1.05,
        "aug_use_depth_offset": True,
        "aug_depth_offset_prob": 0.25,
        "aug_depth_offset_region_min": 0.05,
        "aug_depth_offset_region_max": 0.1,
        'aug_depth_offset_min': 0.5,
        'aug_depth_offset_max': 1.0,
        'aug_depth_offset_local': 0.1,
        "aug_use_depth_small_offset": True,
        "aug_use_global_noise": True,
        "aug_global_noise_prob": 0.5,
        "aug_depth_small_offset_prob": 0.5,
        "aug_forward_crop_size": (400,600),
        "aug_pixel_center_sample": True,
        "aug_view_select_type": "easy",

        "use_consistent_min_max": False,
        "revise_depth_range": False,
    }
    def __init__(self, root_dir, is_train, cfg=None):
        if cfg is not None:
            self.cfg={**self.default_cfg,**cfg}
        else:
            self.cfg={**self.default_cfg}
        self.root_dir = root_dir
        self.is_train = is_train
        if is_train:
            self.num=999999
            self.type2scene_names,self.database_types,self.database_weights = {}, [], []
            if self.cfg['resolution_type']=='hr':
                type2scene_names={}
            elif self.cfg['resolution_type']=='lr':
                type2scene_names={
                    'scannet': np.loadtxt('configs/lists/scannetv2_train_split.txt',dtype=str).tolist(),
                }
            else:
                raise NotImplementedError

            for database_type in self.cfg['train_database_types']:
                self.type2scene_names[database_type] = type2scene_names[database_type]
                self.database_types.append(database_type)
                self.database_weights.append(self.cfg['type2sample_weights'][database_type])
            assert(len(self.database_types)>0)
            # normalize weights
            self.database_weights=np.asarray(self.database_weights)
            self.database_weights=self.database_weights/np.sum(self.database_weights)
        else:
            self.database = parse_database_name(self.cfg['val_database_name'], root_dir=self.root_dir)
            self.ref_ids, self.que_ids = get_database_split(self.database,self.cfg['val_database_split_type'])
            self.num=len(self.que_ids)
        self.database_statistics = {}

        self.blender2opencv = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).to(torch.float32)

    def get_database_ref_que_ids(self, index):
        if self.is_train:
            database_type = np.random.choice(self.database_types,1,False,p=self.database_weights)[0]
            database_scene_name = np.random.choice(self.type2scene_names[database_type])
            database = parse_database_name(database_scene_name, root_dir=self.root_dir)
            # if there is no depth for all views, we repeat random sample until find a scene with depth
            while True:
                ref_ids = database.get_img_ids(check_depth_exist=True)
                if len(ref_ids)==0:
                    database_type = np.random.choice(self.database_types, 1, False, self.database_weights)[0]
                    database_scene_name = np.random.choice(self.type2scene_names[database_type])
                    database = parse_database_name(database_scene_name, root_dir=self.root_dir)
                else: break
            que_id = np.random.choice(ref_ids)
            # if database.database_name.startswith('real_estate'):
            #     que_id, ref_ids = select_train_ids_for_real_estate(ref_ids)
        else:
            database = self.database
            que_id, ref_ids = self.que_ids[index], self.ref_ids
        return database, que_id, np.asarray(ref_ids)

    def select_working_views_impl(self, database_name, dist_idx, ref_num):
        if self.cfg['aug_view_select_type']=='default':
            if database_name.startswith('space') or database_name.startswith('real_estate'):
                pass
            elif database_name.startswith('gso'):
                pool_ratio = np.random.randint(1, 5)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('real_iconic'):
                pool_ratio = np.random.randint(1, 5)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('dtu_train'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            elif database_name.startswith('scannet'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            else:
                raise NotImplementedError
        elif self.cfg['aug_view_select_type']=='easy':
            if database_name.startswith('space') or database_name.startswith('real_estate'):
                pass
            elif database_name.startswith('gso'):
                pool_ratio = 3
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 24)]
            elif database_name.startswith('real_iconic'):
                pool_ratio = np.random.randint(1, 4)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 20)]
            elif database_name.startswith('dtu_train'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            elif database_name.startswith('scannet'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            else:
                raise NotImplementedError

        return dist_idx

    def select_working_views(self, database, que_id, ref_ids):
        database_name = database.database_name
        dist_idx = compute_nearest_camera_indices(database, [que_id], ref_ids)[0]
        if self.is_train:
            if np.random.random()>0.02: # 2% chance to include que image
                dist_idx = dist_idx[ref_ids[dist_idx]!=que_id]
            ref_num = np.random.randint(self.cfg['min_wn'], self.cfg['max_wn'])
            dist_idx = self.select_working_views_impl(database_name,dist_idx,ref_num)
            if not database_name.startswith('real_estate'):
                # we already select working views for real estate dataset
                np.random.shuffle(dist_idx)
                dist_idx = dist_idx[:ref_num]
                ref_ids = ref_ids[dist_idx]
            else:
                ref_ids = ref_ids[:ref_num]
        else:
            dist_idx = dist_idx[:self.cfg['min_wn']]
            ref_ids = ref_ids[dist_idx]
        return ref_ids

    def random_change_depth_range(self, depth_range):
        depth_range_new = depth_range.copy()
        if np.random.random()<self.cfg['aug_depth_range_prob']:
            depth_range_new[:,0] *= np.random.uniform(self.cfg['aug_depth_range_min'],1.0)
            depth_range_new[:,1] *= np.random.uniform(1.0,self.cfg['aug_depth_range_max'])
        return depth_range_new


    def add_depth_noise(self,depths,masks,depth_ranges):
        rfn = depths.shape[0]
        depths_output = []
        for rfi in range(rfn):
            depth, mask, depth_range = depths[rfi,0], masks[rfi,0], depth_ranges[rfi]

            depth = depth.copy()
            near, far = depth_range
            depth_length = far - near
            if self.cfg['aug_use_depth_offset'] and np.random.random() < self.cfg['aug_depth_offset_prob']:
                add_depth_offset(depth, mask,self.cfg['aug_depth_offset_region_min'],
                                 self.cfg['aug_depth_offset_region_max'],
                                 self.cfg['aug_depth_offset_min'],
                                 self.cfg['aug_depth_offset_max'],
                                 self.cfg['aug_depth_offset_local'], depth_length)
            if self.cfg['aug_use_depth_small_offset'] and np.random.random() < self.cfg['aug_depth_small_offset_prob']:
                add_depth_offset(depth, mask, 0.1, 0.2, 0.01, 0.05, 0.005, depth_length)
            if self.cfg['aug_use_global_noise'] and np.random.random() < self.cfg['aug_global_noise_prob']:
                depth += np.random.uniform(-0.005,0.005,depth.shape).astype(np.float32)*depth_length
            depths_output.append(depth)
        return np.asarray(depths_output)[:,None,:,:]

    def generate_coords_for_training(self, database, que_imgs_info):
        if (database.database_name.startswith('real_estate') \
                or database.database_name.startswith('real_iconic') \
                or database.database_name.startswith('space')) and self.cfg['aug_pixel_center_sample']:
                que_mask_cur = np.zeros_like(que_imgs_info['masks'][0, 0]).astype(np.bool)
                h, w = que_mask_cur.shape
                center_ratio = 0.8
                begin_ratio = (1-center_ratio)/2
                hb, he = int(h*begin_ratio), int(h*(center_ratio+begin_ratio))
                wb, we = int(w*begin_ratio), int(w*(center_ratio+begin_ratio))
                que_mask_cur[hb:he,wb:we] = True
                coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], 0.9).reshape([1, -1, 2])
        else:
            que_mask_cur = que_imgs_info['masks'][0,0]>0
            coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], self.cfg['foreground_ratio']).reshape([1,-1,2])
        return coords

    def consistent_depth_range(self, ref_imgs_info, que_imgs_info):
        depth_range_all = np.concatenate([ref_imgs_info['depth_range'], que_imgs_info['depth_range']], 0)
        if self.cfg['use_consistent_min_max']:
            depth_range_all[:, 0] = np.min(depth_range_all)
            depth_range_all[:, 1] = np.max(depth_range_all)
        else:
            range_len = depth_range_all[:, 1] - depth_range_all[:, 0]
            max_len = np.max(range_len)
            range_margin = (max_len - range_len) / 2
            ref_near = depth_range_all[:, 0] - range_margin
            ref_near = np.max(np.stack([ref_near, depth_range_all[:, 0] * 0.5], -1), 1)
            depth_range_all[:, 0] = ref_near
            depth_range_all[:, 1] = ref_near + max_len
        ref_imgs_info['depth_range'] = depth_range_all[:-1]
        que_imgs_info['depth_range'] = depth_range_all[-1:]


    def multi_scale_depth(self, depth_h):
        '''
        This is the implementation of Klevr dataset and move here to make dataset format the same
        '''
        
        depth = {}
        for l in range(3):

            depth[f"level_{l}"] = cv2.resize(
                depth_h,
                None,
                fx=1.0 / (2**l),
                fy=1.0 / (2**l),
                interpolation=cv2.INTER_NEAREST,
            )
            # depth[f"level_{l}"][depth[f"level_{l}"] > far_bound * 0.95] = 0.0

        if self.is_train:
            cutout = np.ones_like(depth[f"level_2"])
            h0 = int(np.random.randint(0, high=cutout.shape[0] // 5, size=1))
            h1 = int(
                np.random.randint(
                    4 * cutout.shape[0] // 5, high=cutout.shape[0], size=1
                )
            )
            w0 = int(np.random.randint(0, high=cutout.shape[1] // 5, size=1))
            w1 = int(
                np.random.randint(
                    4 * cutout.shape[1] // 5, high=cutout.shape[1], size=1
                )
            )
            cutout[h0:h1, w0:w1] = 0
            depth_aug = depth[f"level_2"] * cutout
        else:
            depth_aug = depth[f"level_2"].copy()

        return depth, depth_aug




    def __getitem__(self, index):
        set_seed(index, self.is_train)
        database, que_id, ref_ids_all = self.get_database_ref_que_ids(index)
        ref_ids = self.select_working_views(database, que_id, ref_ids_all)
        if self.cfg['use_src_imgs']:
            # src_imgs_info used in construction of cost volume
            ref_imgs_info, ref_cv_idx, ref_real_idx = build_src_imgs_info_select(database,ref_ids,ref_ids_all,self.cfg['cost_volume_nn_num'])
        else:
            ref_idx = compute_nearest_camera_indices(database, ref_ids)[:,0:4] # used in cost volume construction
            is_aligned = not database.database_name.startswith('space')
            ref_imgs_info = build_imgs_info(database, ref_ids, -1, is_aligned)
        # semray's implementation query image cannot access to depth, we use depth here but not as input nor supervision
        # que_imgs_info = build_imgs_info(database, [que_id], has_depth=self.is_train)
        que_imgs_info = build_imgs_info(database, [que_id])

        if self.is_train:
            # data augmentation
            depth_range_all = np.concatenate([ref_imgs_info['depth_range'],que_imgs_info['depth_range']],0)

            depth_range_all = self.random_change_depth_range(depth_range_all)
            ref_imgs_info['depth_range'] = depth_range_all[:-1]
            que_imgs_info['depth_range'] = depth_range_all[-1:]



            if database.database_name.startswith('real_estate') \
                or database.database_name.startswith('real_iconic') \
                or database.database_name.startswith('space'):
                # crop all datasets
                ref_imgs_info, que_imgs_info = random_crop(ref_imgs_info, que_imgs_info, self.cfg['aug_forward_crop_size'])
                if np.random.random()<0.5:
                    ref_imgs_info, que_imgs_info = random_flip(ref_imgs_info, que_imgs_info)

            if self.cfg['use_depth_loss_for_all'] and self.cfg['use_depth']:
                if not database.database_name.startswith('gso'):
                    ref_imgs_info['true_depth'] = ref_imgs_info['depth']

        if self.cfg['use_consistent_depth_range']:
            self.consistent_depth_range(ref_imgs_info, que_imgs_info)


        ref_imgs_info = pad_imgs_info(ref_imgs_info,self.cfg['ref_pad_interval'])

        # don't feed depth to gpu
        if not self.cfg['use_depth']:
            if 'depth' in ref_imgs_info: ref_imgs_info.pop('depth')
            if 'depth' in que_imgs_info: que_imgs_info.pop('depth')
            if 'true_depth' in ref_imgs_info: ref_imgs_info.pop('true_depth')

        if self.cfg['use_src_imgs']:
            src_imgs_info = ref_imgs_info.copy()
            ref_imgs_info = imgs_info_slice(ref_imgs_info, ref_real_idx)
            ref_imgs_info['nn_ids'] = ref_cv_idx
        else:
            # 'nn_ids' used in constructing cost volume (specify source image ids)
            ref_imgs_info['nn_ids'] = ref_idx.astype(np.int64)

        # ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
        # que_imgs_info = imgs_info_to_torch(que_imgs_info)

        # outputs = {'ref_imgs_info': ref_imgs_info, 'que_imgs_info': que_imgs_info, 'scene_name': database.database_name}
        # if self.cfg['use_src_imgs']: outputs['src_imgs_info'] = imgs_info_to_torch(src_imgs_info)

        same_format_as_klevr = True
        if same_format_as_klevr:
            sample = {}
            sample['images'] = np.concatenate((ref_imgs_info['imgs'], que_imgs_info['imgs']), 0)
            sample['semantics'] = np.concatenate((ref_imgs_info['labels'], que_imgs_info['labels']), 0).squeeze(1)


            # sample['w2cs'] = np.concatenate((ref_imgs_info['poses'], que_imgs_info['poses']), 0) # (1+nb_views, 3, 4)
            # sample['w2cs'] = np.concatenate((sample['w2cs'], torch.ones_like(sample['w2cs'])[:,0:1,:]), 1) # (1+nb_views, 4, 4)

            sample['c2ws'] = np.concatenate((ref_imgs_info['poses'], que_imgs_info['poses']), 0) # (1+nb_views, 4, 4)
            # sample['c2ws'] = np.concatenate((sample['c2ws'], torch.ones_like(sample['c2ws'])[:,0:1,:]), 1) # (1+nb_views, 4, 4)
            # sample['c2ws'] = sample['c2ws'] @ self.blender2opencv


            sample['intrinsics'] = np.concatenate((ref_imgs_info['Ks'], que_imgs_info['Ks']), 0)
            sample['near_fars'] = np.concatenate((ref_imgs_info['depth_range'], que_imgs_info['depth_range']), 0)
            sample['depths_h'] = np.concatenate((ref_imgs_info['depth'], que_imgs_info['depth']), 0).squeeze(1)
            sample['closest_idxs'] = ref_imgs_info['nn_ids'] # (nb_view, 4) # used in cost volume construction # hard code to [0:4]

            # affine_mats (1+nb_views, 4, 4, 3)
            # affine_mats_inv (1+nb_views, 4, 4, 3)
            # depths_aug (1+nb_views, 1, H/4, W/4)
            # depths {dict} {'level_0': (1+nb_views, 1, H, W), 'level_1': (1+nb_views, 1, H/2, W/2), 'level_2': (1+nb_views, 1, H/4, W/4)}

            sample['w2cs'] = []
            affine_mats, affine_mats_inv, depths_aug = [], [], []
            depths = {"level_0": [], "level_1": [], "level_2": []}

            for i in range(sample['c2ws'].shape[0]):
                sample['w2cs'].append(np.linalg.inv(sample['c2ws'][i]))
                # sample['w2cs'].append(torch.asarray(np.linalg.inv(np.asarray(sample['c2ws'][i]))))

                aff = []
                aff_inv = []

                for l in range(3):
                    proj_mat_l = np.eye(4)
                    intrinsic_temp = sample['intrinsics'][i].copy()
                    intrinsic_temp[:2] = intrinsic_temp[:2]/(2**l)
                    proj_mat_l[:3,:4] = intrinsic_temp @ sample['w2cs'][i][:3,:4]
                    aff.append(proj_mat_l)
                    aff_inv.append(np.linalg.inv(proj_mat_l))

                aff = np.stack(aff, axis=-1)
                aff_inv = np.stack(aff_inv, axis=-1)

                affine_mats.append(aff)
                affine_mats_inv.append(aff_inv)

                depth, depth_aug = self.multi_scale_depth(np.asarray(sample['depths_h'][i]))
                depths["level_0"].append(depth["level_0"])
                depths["level_1"].append(depth["level_1"])
                depths["level_2"].append(depth["level_2"])
                depths_aug.append(depth_aug)

            affine_mats = np.stack(affine_mats)
            affine_mats_inv = np.stack(affine_mats_inv)
            depths_aug = np.stack(depths_aug)
            depths["level_0"] = np.stack(depths["level_0"])
            depths["level_1"] = np.stack(depths["level_1"])
            depths["level_2"] = np.stack(depths["level_2"])


            sample['w2cs'] = np.stack(sample['w2cs'], 0) # (1+nb_views, 4, 4)
            sample['affine_mats'] = affine_mats
            sample['affine_mats_inv'] = affine_mats_inv
            sample['depths_aug'] = depths_aug
            sample['depths'] = depths

            return sample
        
        # if same_format_as_klevr == False:
        # return outputs

    def __len__(self):
        return self.num




# class ScannetDataset(Dataset):
#     def __init__(
#             self,
#             root_dir,
#             nb_views,
#             split='train',
#             get_rgb=True,
#             get_semantic=False,
#             max_len=-1,
#             scene=None,
#             downSample=1.0,
#     ):
#         super().__init__()
#         if split == 'train':
#             self.num=999999
#         else:
#             self.num =
#         self.root_dir = root_dir
#         self.split = split
#         self.get_rgb = get_rgb
#         self.get_semantic = get_semantic
#         self.max_len = max_len
#         self.scene = scene
#         self.downSample = downSample
#         self.nb_views = nb_views
                        
#         self.define_transforms()
#         self.read_meta()
#         self.white_back = False
#         self.buid_proj_mats()

#     def define_transforms(self):
#         # this normalize is for imagenet pretrained resnet for CasMVSNet pretrained weights
#         self.transform = T.Compose(
#             [
#                 T.ToTensor(),
#                 # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ]
#         )
        
#     def read_meta(self):
#         '''
#         scannet's images are not using same trajectory, so we need to distinguish them by scene. Here only read the scene names.
#         '''
#         with open(f"configs/lists/scannetv2_{self.split}_split.txt") as f:
#             self.scans = [line.rstrip() for line in f.readlines()]
#             if self.scene != "None":
#                 self.scans = [self.scene]

        
#     def buid_proj_mats(self):
#         return None