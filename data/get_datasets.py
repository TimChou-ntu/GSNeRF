# GeoNeRF is a generalizable NeRF model that renders novel views
# without requiring per-scene optimization. This software is the 
# implementation of the paper "GeoNeRF: Generalizing NeRF with 
# Geometry Priors" by Mohammad Mahdi Johari, Yann Lepoittevin,
# and Francois Fleuret.

# Copyright (c) 2022 ams International AG

# This file is part of GeoNeRF.
# GeoNeRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# GeoNeRF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GeoNeRF. If not, see <http://www.gnu.org/licenses/>.

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import numpy as np

from data.llff import LLFF_Dataset
from data.dtu import DTU_Dataset
from data.nerf import NeRF_Dataset
from data.klevr import KlevrDataset
from data.scannet import RendererDataset

def get_training_dataset(args, downsample=1.0):
    
    if args.dataset_name == "klevr":
        train_dataset = KlevrDataset(
            root_dir=args.klevr_path,
            split="train",
            max_len=-1,
            downSample=downsample,
            nb_views=args.nb_views,
            get_semantic=args.segmentation,
        )
        train_sampler = None
        return train_dataset, train_sampler
    
    elif args.dataset_name == "scannet":
        train_dataset = RendererDataset(
            root_dir=args.scannet_path,
            is_train=True
        )
        train_sampler = None
        return train_dataset, train_sampler
    
    train_datasets = [
        DTU_Dataset(
            original_root_dir=args.dtu_path,
            preprocessed_root_dir=args.dtu_pre_path,
            split="train",
            max_len=-1,
            downSample=downsample,
            nb_views=args.nb_views,
        ),
        LLFF_Dataset(
            root_dir=args.ibrnet1_path,
            split="train",
            max_len=-1,
            downSample=downsample,
            nb_views=args.nb_views,
            imgs_folder_name="images",
        ),
        LLFF_Dataset(
            root_dir=args.ibrnet2_path,
            split="train",
            max_len=-1,
            downSample=downsample,
            nb_views=args.nb_views,
            imgs_folder_name="images",
        ),
        LLFF_Dataset(
            root_dir=args.llff_path,
            split="train",
            max_len=-1,
            downSample=downsample,
            nb_views=args.nb_views,
            imgs_folder_name="images_4",
        ),
    ]
    weights = [0.5, 0.22, 0.12, 0.16]

    train_weights_samples = []
    for dataset, weight in zip(train_datasets, weights):
        num_samples = len(dataset)
        weight_each_sample = weight / num_samples
        train_weights_samples.extend([weight_each_sample] * num_samples)

    train_dataset = ConcatDataset(train_datasets)
    train_weights = torch.from_numpy(np.array(train_weights_samples))
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    return train_dataset, train_sampler


def get_finetuning_dataset(args, downsample=1.0):
    if args.dataset_name == "dtu":
        train_dataset = DTU_Dataset(
            original_root_dir=args.dtu_path,
            preprocessed_root_dir=args.dtu_pre_path,
            split="train",
            max_len=-1,
            downSample=downsample,
            nb_views=args.nb_views,
            scene=args.scene,
        )
    elif args.dataset_name == "llff":
        train_dataset = LLFF_Dataset(
            root_dir=args.llff_path,
            split="train",
            max_len=-1,
            downSample=downsample,
            nb_views=args.nb_views,
            scene=args.scene,
            imgs_folder_name="images_4",
        )
    elif args.dataset_name == "nerf":
        train_dataset = NeRF_Dataset(
            root_dir=args.nerf_path,
            split="train",
            max_len=-1,
            downSample=downsample,
            nb_views=args.nb_views,
            scene=args.scene,
        )

    train_sampler = None

    return train_dataset, train_sampler


def get_validation_dataset(args, downsample=1.0):
    if args.scene == "None":
        max_len = 10
    else:
        max_len = -1

    if args.dataset_name == "dtu":
        val_dataset = DTU_Dataset(
            original_root_dir=args.dtu_path,
            preprocessed_root_dir=args.dtu_pre_path,
            split="val",
            max_len=max_len,
            downSample=downsample,
            nb_views=args.nb_views,
            scene=args.scene,
        )
    elif args.dataset_name == "llff":
        val_dataset = LLFF_Dataset(
            root_dir=args.llff_test_path if not args.llff_test_path is None else args.llff_path,
            split="val",
            max_len=max_len,
            downSample=downsample,
            nb_views=args.nb_views,
            scene=args.scene,
            imgs_folder_name="images_4",
        )
    elif args.dataset_name == "nerf":
        val_dataset = NeRF_Dataset(
            root_dir=args.nerf_path,
            split="val",
            max_len=max_len,
            downSample=downsample,
            nb_views=args.nb_views,
            scene=args.scene,
        )
    elif args.dataset_name == "klevr":
        val_dataset = KlevrDataset(
            root_dir=args.klevr_path,
            split="val",
            max_len=max_len,
            downSample=downsample,
            nb_views=args.nb_views,
            get_semantic=args.segmentation,
        )
    elif args.dataset_name == "scannet":
        val_set_list, val_set_names = [], []
        if isinstance(args.val_set_list, str):
            val_scenes = np.loadtxt(args.val_set_list, dtype=str).tolist()
            for name in val_scenes:
                val_cfg = {'val_database_name': name}
                val_set = RendererDataset(cfg=val_cfg, is_train=False, root_dir=args.scannet_path)
                val_set_list.append(val_set)
                val_set_names.append(name)
                print(f'{name} val set len {len(val_set)}')
        val_dataset = ConcatDataset(val_set_list)
    else:
        raise NotImplementedError
    
    return val_dataset
