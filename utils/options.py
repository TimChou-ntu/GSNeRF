import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="Config file path")

    # Task options
    parser.add_argument("--segmentation", action="store_true", help="Use segmentation mask for training")
    parser.add_argument("--nb_class", type=int, default=21, help="Number of classes for segmentation")
    parser.add_argument("--ignore_label", type=int, default=20, help="Ignore label for segmentation")

    # Datasets options
    parser.add_argument("--dataset_name", type=str, default="llff", choices=["llff", "nerf", "dtu", "klevr", "scannet", "replica"],)
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
    parser.add_argument("--replica_path", type=str, help="Path to replica dataset")

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
    parser.add_argument("--background_weight", type=float, default=1, help="Weight for background class in cross entropy loss")
    parser.add_argument("--two_stage_training_steps", type=int, default=60000, help="Use two stage training, indicating how many steps for first stage")
    parser.add_argument("--self_supervised_depth_loss", action="store_true", help="Use self supervised depth loss")

    # Rendering options
    parser.add_argument("--chunk", type=int, default=4096, help="Number of rays rendered in parallel")
    parser.add_argument("--nb_coarse", type=int, default=96, help="Number of coarse samples per ray")
    parser.add_argument("--nb_fine", type=int, default=32, help="Number of additional fine samples per ray",)

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
    parser.add_argument("--using_semantic_global_tokens", type=int, default=0, help="Use only semantic global tokens in rendering process. 0: not use, 1: use")
    parser.add_argument("--only_using_semantic_global_tokens", type=int, default=0, help="Use only semantic global tokens in rendering process. 0: not use, 1: use")
    parser.add_argument("--use_batch_semantic_feature", action="store_true", help="Use batch semantic feature in rendering process")
    parser.add_argument("--ddp", action="store_true", help="Use distributed data parallel")
    parser.add_argument("--feat_net", type=str, default="UNet", choices=["UNet", "smp_UNet"], help="FeatureNet used in depth estimation")
    # resume options
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to a checkpoint to resume training")
    parser.add_argument("--finetune", action="store_true", help="Finetune the model with a checkpoint")
    parser.add_argument("--fintune_scene", type=str, default="None", help="Scene for fine-tuning")
    return parser.parse_args()
