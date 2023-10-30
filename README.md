<!-- To train:
python run_geo_nerf.py --config configs/config_general_scannet.txt --segmentation --logger wandb --val_save_img_type depth --target_depth_estimation

To validation:
python run_geo_nerf.py --config configs/config_general_scannet.txt --segmentation --logger none --val_save_img_type depth --target_depth_estimation --ckpt_path /mnt/sdb/timothy/Desktop/2023Spring/generalized_nerf/logs_scannet/scannet/0625_scannet_withdepthloss_withsemanticfeatloss_semanticnetwasntinoptimizer/ckpts/ckpt_step-324895.ckpt --eval



 -->

> # GSNeRF: Enhancing 3D Scene Understanding with Generalizable Semantic Neural Radiance Fields <br>
> Zi-Ting Chou, Sheng-Yu Huang, I-Jieh Liu, Yu-Chiang Wang <br>
> [Project Page (TBD)]() | [Paper (TBD)]()

This repository contains a PyTorch Lightning implementation of our paper, GSNeRF.

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
Download `scannet` from [here](https://github.com/ScanNet/ScanNet) and set its path as `scannet_path` in the [1003_scannet.txt](./configs/1003_scannet) file.

For training a generalizable model, set the number of source views to 8 (nb_views = 8) in the [1003_scannet.txt](./configs/1003_scannet.txt) file and run the following command:

```
python run_geo_nerf.py --config configs/1003_scannet.txt --segmentation --logger wandb --val_save_img_type depth --target_depth_estimation
```

For evaluation on a novel scene, run the following command: (replace [ckpt path] with your trained checkpoint path.)

```
python run_geo_nerf.py --config configs/1003_scannet.txt --segmentation --logger none --val_save_img_type depth --target_depth_estimation --ckpt_path [ckpt path] --eval
```


## Replica (Synthetic indoor scene) Dataset
Download `Replica` from [here](https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0) and set its path as `replica_path` in the [0727_replica.txt](configs/0727_replica.txt) file. (Thanks [Semantic Nerf](https://github.com/Harry-Zhi/semantic_nerf) for rendering 2D image and semantic map.)

For training a generalizable model, set the number of source views to 8 (nb_views = 8) in the [0727_replica.txt](./configs/0727_replica.txt) file and run the following command:

```
python run_geo_nerf.py --config configs/0727_replica.txt --segmentation --logger wandb --val_save_img_type depth --target_depth_estimation
```

For evaluation on a novel scene, run the following command: (replace [ckpt path] with your trained checkpoint path.)

```
python run_geo_nerf.py --config configs/0727_replica.txt --segmentation --logger none --val_save_img_type depth --target_depth_estimation --ckpt_path [ckpt path] --eval
```

### Self-supervised depth model
TODO.

### Contact
You can contact the author through email: A88551212@gmail.com

<!-- ## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{johari-et-al-2022,
  author = {Johari, M. and Lepoittevin, Y. and Fleuret, F.},
  title = {GeoNeRF: Generalizing NeRF with Geometry Priors},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
``` -->

### Acknowledgement
This work was supported by National Center for High-performance Computing (NCHC) for providing computational and storage resources.