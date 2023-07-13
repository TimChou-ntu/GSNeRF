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

import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="Config file path")

    # Task options
    parser.add_argument("--segmentation", action="store_true", help="Use segmentation mask for training")
    parser.add_argument("--nb_class", type=int, default=21, help="Number of classes for segmentation")
    parser.add_argument("--ignore_label", type=int, default=20, help="Ignore label for segmentation")

    # Datasets options
    parser.add_argument("--dataset_name", type=str, default="llff", choices=["llff", "nerf", "dtu", "klevr", "scannet"],)
    parser.add_argument("--llff_path", type=str, help="Path to llff dataset")
    parser.add_argument("--llff_test_path", type=str, help="Path to llff dataset")
    parser.add_argument("--dtu_path", type=str, help="Path to dtu dataset")
    parser.add_argument("--dtu_pre_path", type=str, help="Path to preprocessed dtu dataset")
    parser.add_argument("--nerf_path", type=str, help="Path to nerf dataset")
    parser.add_argument("--ams_path", type=str, help="Path to ams dataset")
    parser.add_argument("--ibrnet1_path", type=str, help="Path to ibrnet1 dataset")
    parser.add_argument("--ibrnet2_path", type=str, help="Path to ibrnet2 dataset")
    parser.add_argument("--klevr_path", type=str, help="Path to klevr dataset")
    parser.add_argument("--scannet_path", type=str, help="Path to scannet dataset")

    # for scannet dataset
    parser.add_argument("--val_set_list", type=str, help="Path to scannet val dataset list")

    # Training options
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--nb_views", type=int, default=3)
    parser.add_argument("--lrate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Gradually warm-up learning rate in optimizer")
    parser.add_argument("--scene", type=str, default="None", help="Scene for fine-tuning")
    parser.add_argument("--cross_entropy_weight", type=float, default=0.1, help="Weight for cross entropy loss")
    parser.add_argument("--optimizer", type=str, default="adam", help="select optimizer: adam / sgd")

    # Rendering options
    parser.add_argument("--chunk", type=int, default=4096, help="Number of rays rendered in parallel")
    parser.add_argument("--nb_coarse", type=int, default=96, help="Number of coarse samples per ray")
    parser.add_argument("--nb_fine", type=int, default=32, help="Number of additional fine samples per ray",)
    # parser.add_argument("--nb_coarse", type=int, default=48, help="Number of coarse samples per ray")
    # parser.add_argument("--nb_fine", type=int, default=16, help="Number of additional fine samples per ray",)

    # Other options
    parser.add_argument("--expname", type=str, help="Experiment name")
    parser.add_argument("--logger", type=str, default="wandb", choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--logdir", type=str, default="./logs/", help="Where to store ckpts and logs")
    parser.add_argument("--eval", action="store_true", help="Render and evaluate the test set")
    parser.add_argument("--use_depth", action="store_true", help="Use ground truth low-res depth maps in rendering process")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--val_save_img_type", default=["target"], action="append", help="choices=[target, depth, source], Save target comparison images or depth maps or source images")
    parser.add_argument("--target_depth_estimation", action="store_true", help="Use target depth estimation in rendering process")
    parser.add_argument("--use_depth_refine_net", action="store_true", help="Use depth refine net before rendering process")

    # resume options
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to a checkpoint to resume training")
    return parser.parse_args()
