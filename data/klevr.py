import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from glob import glob as glob
from PIL import Image
from torchvision import transforms as T

from utils.klevr_utils import from_position_and_quaternion, scale_rays, calculate_near_and_far
from utils.utils import read_pfm, get_nearest_pose_ids, get_rays

# Nesf Klevr
class KlevrDataset(Dataset):
    def __init__(
            self, 
            root_dir, 
            nb_views, 
            split='train', 
            get_rgb=True, 
            get_semantic=False, 
            max_len=-1, 
            scene=None, 
            downSample=1.0
            ):

        # super().__init__()
        self.root_dir = root_dir
        self.get_rgb = get_rgb
        self.get_semantic = get_semantic
        self.nb_views = nb_views
        self.max_len = max_len
        self.scene = scene
        self.downSample = downSample
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        # This is hard coded for Klevr as all scans are in the same range (according to the metadata)
        self.scene_boundaries = np.array([[-3.1,-3.1,-0.1],[3.1,3.1,3.1]])
        if split == 'train':
            self.split = split
        elif split =='val':
            self.split = 'test'
        else:
            raise KeyError("only train/val split works")
        
        self.define_transforms()
        self.read_meta()
        self.white_back = True
        self.buid_proj_mats()

    def define_transforms(self):
        # this normalize is for imagenet pretrained resnet for CasMVSNet pretrained weights
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def read_meta(self):
        self.metas = []
        # for now, we only use the first 10 scans to train, 11~20 scans to test
        if self.split == 'train':
            self.scans = sorted(glob(os.path.join(self.root_dir, '*')))[:100]
        else:
            self.scans = sorted(glob(os.path.join(self.root_dir, '*')))[100:120]

        # remap the scan_idx to the scan name
        self.scan_idx_to_name = {}

        # record ids that being used of each scan
        self.id_list = []

        # read the pair file, here use the same pair list as dtu dataset; 
        # could be 6x more pairs since each klevr scene have 300 views, dtu have 50 views
        if self.split == "train":
            if self.scene == "None":
                pair_file = f"configs/lists/dtu_pairs.txt"
            else:
                pair_file = f"configs/lists/dtu_pairs_ft.txt"
        else:
            pair_file = f"configs/lists/dtu_pairs_val.txt"

        for scan_idx, meta_filename in enumerate(self.scans):
            with open(pair_file) as f:
                scan = meta_filename.split('/')[-1]
                self.scan_idx_to_name[scan_idx] = scan
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    self.metas += [(scan_idx, ref_view, src_views[:self.nb_views])]
                    self.id_list.append([ref_view] + src_views)

        self.id_list = np.unique(self.id_list)
        self.build_remap()

    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype("int")
        for i, item in enumerate(self.id_list):
            self.remap[item] = i

    def buid_proj_mats(self):
        # maybe not calculate near_far now. Do it when creating the rays
        self.near_fars, self.intrinsics, self.world2cams, self.cam2worlds = None, {}, {}, {}
        for scan_idx, meta_fileprefix in enumerate(self.scans):
            meta_filename = os.path.join(meta_fileprefix, 'metadata.json')
            intrinsic, world2cam, cam2world = self.read_cam_file(meta_filename, scan_idx)
            self.intrinsics[scan_idx], self.world2cams[scan_idx], self.cam2worlds[scan_idx] = np.array(intrinsic), np.array(world2cam), np.array(cam2world)
            
    def read_cam_file(self, filename, scan_idx):
        '''
        read the metadata file and return the near/far, intrinsic, world2cam, cam2world
        filename(str): the metadata file
        scan_idx(int): the index of the scan

        return:
            intrinsic: the intrinsic of the scan                                              [N,3,3]
            world2cam: the world2cam of the scan                                              [N,4,4]
            cam2world: the cam2world of the scan                                              [N,4,4]
        '''
        intrinsic, world2cam, cam2world = [], [], []
        with open(filename, 'r') as f:
            meta = json.load(f)
        w, h = meta['metadata']['width'], meta['metadata']['height']
        focal = meta['camera']['focal_length']*w/meta['camera']['sensor_width']

        camera_positions = np.array(meta['camera']['positions'])
        camera_quaternions = np.array(meta['camera']['quaternions'])
        # calculate camera pose of each frame that will be used (in this scan idx)
        for frame_id in self.id_list:
            c2w = from_position_and_quaternion(camera_positions[frame_id], camera_quaternions[frame_id], False).tolist()
            # not sure
            c2w = np.array(c2w) @ self.blender2opencv
            cam2world += [c2w.tolist()]
            world2cam += [np.linalg.inv(c2w).tolist()]
            intrinsic += [[[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]]]

        return intrinsic, world2cam, cam2world

    def read_depth(self, filename, far_bound, noisy_factor=1.0):
        # read depth image, currently not using it
        return

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        # haven't used the depth image and noisy factor yet
        if self.split == "train" and self.scene == "None":
            noisy_factor = float(np.random.choice([1.0, 0.5], 1))
            close_views = int(np.random.choice([3, 4, 5], 1))
        else:
            noisy_factor = 1.0
            close_views = 5

        scan_idx, ref_id, src_ids = self.metas[idx]

        # notice that the ref_id is in the last position
        view_ids = src_ids + [ref_id]
        
        affine_mats, affine_mats_inv = [], []
        imgs = []
        intrinsics, w2cs, c2ws = [], [], []

        # intrinsic now every frame has its own intrinsic, but actually it is the same for all frames in a scan
        # # every scan has only one intrinsic, here actually is focal
        # intrinsic = self.intrinsics[scan_idx]
        
        for vid in view_ids:
            img_filename = os.path.join(self.root_dir, self.scan_idx_to_name[scan_idx],f'rgba_{vid:05d}.png')
            img = Image.open(img_filename)
            img_wh = np.round(np.array(img.size)*self.downSample).astype(np.int32)
            
            # originally NeRF use Image.Resampling.LANCZOS, not sure if BICUBIC is better
            img = img.resize(img_wh, Image.BICUBIC)
            # discard the alpha channel, only use rgb. Maybe need "valid_mask = img[-1]>0"
            img = self.transform(np.array(img)[:,:,:3])
            imgs += [img]

            index = self.remap[vid]
            # # debug
            # print("vid: ", vid, "index: ", index, "scan_idx: ", scan_idx)
            # print("self.remap[scan_idx]: ", self.remap[scan_idx])
            # print("self.cam2worlds[scan_idx]: ", np.array(self.cam2worlds[scan_idx]).shape)
            # raise Exception("debug")
            c2ws.append(self.cam2worlds[scan_idx][index])

            w2c = self.world2cams[scan_idx][index]
            w2cs.append(w2c)

            intrinsic = self.intrinsics[scan_idx][index]
            intrinsics.append(intrinsic)

            aff = []
            aff_inv = []
            # if using the depth image,  there should be for l in range(3)
            for l in range(3):
                proj_mat_l = np.eye(4)
                intrinsic_temp = intrinsic.copy()
                intrinsic_temp[:2] = intrinsic_temp[:2]/(2**l)
                proj_mat_l[:3,:4] = intrinsic_temp @ w2c[:3,:4]
                aff.append(proj_mat_l)
                aff_inv.append(np.linalg.inv(proj_mat_l))

            aff = np.stack(aff, axis=-1)
            aff_inv = np.stack(aff_inv, axis=-1)

            affine_mats.append(aff)
            affine_mats_inv.append(aff_inv)

        imgs = np.stack(imgs)
        affine_mats = np.stack(affine_mats)
        affine_mats_inv = np.stack(affine_mats_inv)
        intrinsics = np.stack(intrinsics)
        w2cs = np.stack(w2cs)
        c2ws = np.stack(c2ws)

        close_idxs = []
        for pose in c2ws[:-1]:
            close_idxs.append(
                get_nearest_pose_ids(
                    pose, 
                    c2ws[:-1], 
                    close_views, 
                    angular_dist_method="dist"
                )
            )
        close_idxs = np.stack(close_idxs, axis=0)
        self.near_fars = []
        for i in range(imgs.shape[0]):
            rays_orig, rays_dir, _ = get_rays(
                H=imgs.shape[2], 
                W=imgs.shape[3],
                # hard code to cuda
                intrinsics_target=torch.tensor(intrinsics[i].astype(imgs.dtype)),
                c2w_target=torch.tensor(c2ws[i].astype(imgs.dtype)),
                # hard code to 1
                train_batch_size=1,
            )
            near, far = calculate_near_and_far(rays_orig, rays_dir, bbox_min=[-3.1, -3.1, -0.1], bbox_max=[3.1, 3.1, 3.1])
            near = near.min().item()
            far = far.max().item()
            self.near_fars.append([near, far])

        sample = {}
        sample['images'] = imgs
        sample['w2cs'] = w2cs.astype(imgs.dtype)
        sample['c2ws'] = c2ws.astype(imgs.dtype)
        sample['intrinsics'] = intrinsics.astype(imgs.dtype)
        sample['affine_mats'] = affine_mats.astype(imgs.dtype)
        sample['affine_mats_inv'] = affine_mats_inv.astype(imgs.dtype)
        sample['closest_idxs'] = close_idxs
        # depth aug seems to be a must to give, but if set to None doesn't matter (the use_depth is False)
        sample['depths_aug'] = np.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3]))
        sample['depths_h'] = np.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3]))
        # depth should be a dict, but if set to None doesn't matter (the use_depth is False) (original code still use depth loss)
        sample['depths'] = np.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3]))
        # near_fars is now just using constant value
        sample['near_fars'] = np.array(self.near_fars).astype(imgs.dtype)
        return sample