<!-- To train:
python train.py --config configs/config_general_scannet.txt --segmentation --logger wandb --val_save_img_type depth --target_depth_estimation

To validation:
python train.py --config configs/config_general_scannet.txt --segmentation --logger none --val_save_img_type depth --target_depth_estimation --ckpt_path /mnt/sdb/timothy/Desktop/2023Spring/generalized_nerf/logs_scannet/scannet/0625_scannet_withdepthloss_withsemanticfeatloss_semanticnetwasntinoptimizer/ckpts/ckpt_step-324895.ckpt --eval



 -->

# GSNeRF: Enhancing 3D Scene Understanding with Generalizable Semantic Neural Radiance Fields <br>
> Zi-Ting Chou, Sheng-Yu Huang, I-Jieh Liu, Yu-Chiang Wang <br>
> [Project Page](https://timchou-ntu.github.io/gsnerf/) | [Paper](https://arxiv.org/abs/2403.03608)

This repository contains a official PyTorch Lightning implementation of our paper, GSNeRF (CVPR 2024).

## Installation

#### Tested on NVIDIA GeForce RTX 3090 GPUs with PyTorch 2.0.1 and PyTorch Lightning 2.0.4

To install the dependencies, in addition to PyTorch, run:

```
pip install -r requirements.txt
```

## Evaluation and Training
Following [Semantic Nerf](https://github.com/Harry-Zhi/semantic_nerf) and [Semantic-Ray](https://github.com/liuff19/Semantic-Ray), we conduct experiment on [ScanNet](#scannet-real-world-indoor-scene-dataset) and [Replica](#replica-synthetic-indoor-scene-dataset) respectively.
<!-- To reproduce our results, download pretrained weights from [here](https://drive.google.com/drive/folders/1ZtAc7VYvltcdodT_BrUrQ_4IAhz_L-Rf?usp=sharing) and put them in [pretrained_weights](./pretrained_weights) folder. Then, follow the instructions for each of the [LLFF (Real Forward-Facing)](#llff-real-forward-facing-dataset), [NeRF (Realistic Synthetic)](#nerf-realistic-synthetic-dataset), and [DTU](#dtu-dataset) datasets. -->

## ScanNet (real-world indoor scene) Dataset
Download `scannet` from [here](https://github.com/ScanNet/ScanNet) and set its path as `scannet_path` in the [scannet.txt](./configs/scannet.txt) file.

For training a generalizable model, set the number of source views to 8 (nb_views = 8) in the [scannet.txt](./configs/scannet.txt) file and run the following command:

```
python train.py --config configs/scannet.txt --segmentation --logger wandb --val_save_img_type depth --target_depth_estimation
```

For evaluation on a novel scene, run the following command: (replace [ckpt path] with your trained checkpoint path.)

```
python train.py --config configs/scannet.txt --segmentation --logger none --val_save_img_type depth --target_depth_estimation --ckpt_path [ckpt path] --eval
```


## Replica (Synthetic indoor scene) Dataset
Download `Replica` from [here](https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0) and set its path as `replica_path` in the [replica.txt](configs/replica.txt) file. (Thanks [Semantic Nerf](https://github.com/Harry-Zhi/semantic_nerf) for rendering 2D image and semantic map.)

For training a generalizable model, set the number of source views to 8 (nb_views = 8) in the [replica.txt](./configs/replica.txt) file and run the following command:

```
python train.py --config configs/replica.txt --segmentation --logger wandb --val_save_img_type depth --target_depth_estimation
```

For evaluation on a novel scene, run the following command: (replace [ckpt path] with your trained checkpoint path.)

```
python train.py --config configs/replica.txt --segmentation --logger none --val_save_img_type depth --target_depth_estimation --ckpt_path [ckpt path] --eval
```

### Self-supervised depth model
Simply add --self_supervised_depth_loss at the end of command

### Contact
You can contact the author through email: A88551212@gmail.com

## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{Chou2024gsnerf,
      author    = {Zi‑Ting Chou* and Sheng‑Yu Huang* and I‑Jieh Liu and Yu‑Chiang Frank Wang},
      title     = {GSNeRF: Generalizable Semantic Neural Radiance Fields with Enhanced 3D Scene Understanding},
      booktitle = CVPR,
      year      = {2024},
      arxiv     = {2403.03608},
    }
```

### Acknowledgement

Some portions of the code were derived from [GeoNeRF](https://github.com/idiap/GeoNeRF).

Additionally, the well-structured codebases of [nerf_pl](https://github.com/kwea123/nerf_pl), [nesf](https://nesf3d.github.io/), and [RC-MVSNet](https://github.com/Boese0601/RC-MVSNet) were extremely helpful during the experiment. Shout out to them for their contributions.