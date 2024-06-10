# GSNeRF: Enhancing 3D Scene Understanding with Generalizable Semantic Neural Radiance Fields <br>
> Zi-Ting Chou, Sheng-Yu Huang, I-Jieh Liu, Yu-Chiang Wang <br>
> [Project Page](https://timchou-ntu.github.io/gsnerf/) | [Paper](https://arxiv.org/abs/2403.03608)

This repository contains a official PyTorch Lightning implementation of our paper, GSNeRF (CVPR 2024).

<div align="center">
  <img src="img/cvpr_poster_final_5120.png"/>
</div>

## Installation

#### Tested on NVIDIA GeForce RTX 3090 GPUs with cuda 11.7, PyTorch 2.0.1 and PyTorch Lightning 2.0.4

To install the dependencies, in addition to PyTorch, run:

```
git clone --recursive https://github.com/TimChou-ntu/GSNeRF.git
conda create -n gsnerf python=3.9
conda activate gsnerf
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
cd GSNeRF
pip install -r requirements.txt
```

## Evaluation and Training
Following [Semantic Nerf](https://github.com/Harry-Zhi/semantic_nerf) and [Semantic-Ray](https://github.com/liuff19/Semantic-Ray), we conduct experiment on [ScanNet](#scannet-real-world-indoor-scene-dataset) and [Replica](#replica-synthetic-indoor-scene-dataset) respectively.

Download `scannet` from [here](https://github.com/ScanNet/ScanNet) and set its path as `scannet_path` in the [scannet.txt](./configs/scannet.txt) file.

Download `Replica` from [here](https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0) and set its path as `replica_path` in the [replica.txt](configs/replica.txt) file. (Thanks [Semantic Nerf](https://github.com/Harry-Zhi/semantic_nerf) for rendering 2D image and semantic map.)

Organize the data in the following structure:
```
├── data
│   ├── scannet
│   │   ├── scene0000_00
│   │   │   ├── color
│   │   │   │   ├── 0.jpg
│   │   │   │   ├── ...
│   │   │   ├── depth
│   │   │   │   ├── 0.png
│   │   │   │   ├── ...
│   │   │   ├── label-filt
│   │   │   │   ├── 0.png
│   │   │   │   ├── ...
│   │   │   ├── pose
│   │   │   │   ├── 0.txt
│   │   │   │   ├── ...
│   │   │   ├── intrinsic
│   │   │   │   ├── extrinsic_color.txt
│   │   │   │   ├── intrinsic_color.txt
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── ...
│   │   ├── scannetv2-labels.combined.tsv
|   |
│   ├── replica
│   │   ├── office_0
│   │   │   ├── Sequence_1
│   │   │   │   ├── depth
|   │   │   │   │   ├── depth_0.png
|   │   │   │   │   ├── ...
│   │   │   │   ├── rgb
|   │   │   │   │   ├── rgb_0.png
|   │   │   │   │   ├── ...
│   │   │   │   ├── semantic_class
|   │   │   │   │   ├── semantic_class_0.png
|   │   │   │   │   ├── ...
│   │   │   │   ├── traj_w_c.txt
│   │   ├── ...
│   │   ├── semantic_info
```
## ScanNet (real-world indoor scene) Dataset

For training a generalizable model, set the number of source views to 8 (nb_views = 8) in the [scannet.txt](./configs/scannet.txt) file and run the following command:

```
python train.py --config configs/scannet.txt --segmentation --logger wandb --target_depth_estimation
```

For evaluation on a novel scene, run the following command: (replace [ckpt path] with your trained checkpoint path.)

```
python train.py --config configs/scannet.txt --segmentation --logger none --target_depth_estimation --ckpt_path [ckpt path] --eval
```

## Replica (Synthetic indoor scene) Dataset

For training a generalizable model, set the number of source views to 8 (nb_views = 8) in the [replica.txt](./configs/replica.txt) file and run the following command:

```
python train.py --config configs/replica.txt --segmentation --logger wandb --target_depth_estimation
```

For evaluation on a novel scene, run the following command: (replace [ckpt path] with your trained checkpoint path.)

```
python train.py --config configs/replica.txt --segmentation --logger none --target_depth_estimation --ckpt_path [ckpt path] --eval
```

### Self-supervised depth model
Simply add --self_supervised_depth_loss at the end of command.

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