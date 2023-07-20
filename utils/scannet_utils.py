import abc
import glob
import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread
from natsort import natsorted

from utils.utils import downsample_gaussian_blur, pose_inverse


# From https://github.com/open-mmlab/mmdetection3d/blob/fcb4545ce719ac121348cab59bac9b69dd1b1b59/mmdet3d/datasets/scannet_dataset.py
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.
    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, seg_label):
        """Call function to map original semantic class to valid category ids.
        Args:
            results (dict): Result dict containing point semantic masks.
        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.
                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        seg_label = np.clip(seg_label, 0, self.max_cat_id)
        return self.cat_id2class[seg_label]


class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self, check_depth_exist=False):
        pass

    @abc.abstractmethod
    def get_bbox(self, img_id):
        pass

    @abc.abstractmethod
    def get_depth(self, img_id):
        pass

    @abc.abstractmethod
    def get_mask(self, img_id):
        pass

    @abc.abstractmethod
    def get_depth_range(self, img_id):
        pass



class ScannetDatabase(BaseDatabase):
    def __init__(self, database_name, root_dir='data/scannet'):
        super().__init__(database_name)
        _, self.scene_name, background_size = database_name.split('/')
        background, image_size = background_size.split('_')
        image_size = int(image_size)
        self.image_size = image_size
        self.background = background
        self.root_dir = f'{root_dir}/{self.scene_name}'
        self.ratio = image_size / 1296
        self.h, self.w = int(self.ratio*972), int(image_size)

        rgb_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "color", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        rgb_paths = sorted(rgb_paths)

        K = np.loadtxt(
            f'{self.root_dir}/intrinsic/intrinsic_color.txt').reshape([4, 4])[:3, :3]
        # After resize, we need to change the intrinsic matrix
        K[:2, :] *= self.ratio
        self.K = K

        self.img_ids = []
        for i, rgb_path in enumerate(rgb_paths):
            pose = self.get_pose(i)
            if np.isinf(pose).any() or np.isnan(pose).any():
                continue
            self.img_ids.append(f'{i}')

        self.img_id2imgs = {}
        # mapping from scanntet class id to nyu40 class id
        # mapping_file = 'data/scannet/scannetv2-labels.combined.tsv'
        mapping_file = os.path.join(root_dir, 'scannetv2-labels.combined.tsv')
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
        )

    def get_image(self, img_id):
        if img_id in self.img_id2imgs:
            return self.img_id2imgs[img_id]
        img = imread(os.path.join(
            self.root_dir, 'color', f'{int(img_id)}.jpg'))
        if self.w != 1296:
            img = cv2.resize(downsample_gaussian_blur(
                img, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        return img

    def get_K(self, img_id):
        return self.K.astype(np.float32)

    def get_pose(self, img_id):
        transf = np.diag(np.asarray([1, -1, -1, 1]))
        pose = np.loadtxt(
            f'{self.root_dir}/pose/{int(img_id)}.txt').reshape([4, 4])
        pose = transf @ pose
        # c2w in files, change to w2c
        # pose = pose_inverse(pose)
        return pose.copy()

    def get_img_ids(self, check_depth_exist=False):
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def get_depth(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        img = Image.open(f'{self.root_dir}/depth/{int(img_id)}.png')
        depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
        # depth = np.asarray(img, dtype=np.float32)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_mask(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        return np.ones([h, w], bool)

    def get_depth_range(self, img_id):
        return np.asarray((0.1, 10.0), np.float32)

    def get_label(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        img = Image.open(f'{self.root_dir}/label-filt/{int(img_id)}.png')
        label = np.asarray(img, dtype=np.int32)
        label = np.ascontiguousarray(label)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        label = self.scan2nyu[label]
        return self.label_mapping(label)

def parse_database_name(database_name, root_dir) -> BaseDatabase:
    name2database = {
        'scannet': ScannetDatabase,
        'replica': ReplicaDatabase,
    }
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name, root_dir)
    else:
        raise NotImplementedError
    
def get_database_split(database: BaseDatabase, split_type='val'):
    database_name = database.database_name
    if split_type.startswith('val'):
        splits = split_type.split('_')
        depth_valid = not(len(splits) > 1 and splits[1] == 'all')
        if database_name.startswith('scannet'):
            img_ids = database.get_img_ids()
            train_ids = img_ids[:700:5]
            val_ids = img_ids[2:700:20]
            if len(val_ids) > 10:
                val_ids = val_ids[:10]
        else:
            raise NotImplementedError
    elif split_type.startswith('test'):
        splits = split_type.split('_')
        depth_valid = not(len(splits) > 1 and splits[1] == 'all')
        if database_name.startswith('scannet'):
            img_ids = database.get_img_ids()
            train_ids = img_ids[:700:5]
            val_ids = img_ids[2:700:20]
            if len(val_ids) > 10:
                val_ids = val_ids[:10]
        else:
            raise NotImplementedError
    elif split_type.startswith('video'):
        img_ids = database.get_img_ids()
        train_ids = img_ids[::2]
        val_ids = img_ids[25:-25:2]
    else:
        raise NotImplementedError
    print('train_ids:\n', train_ids)
    print('val_ids:\n', val_ids)
    return train_ids, val_ids


def get_coords_mask(que_mask, train_ray_num, foreground_ratio):
    min_pos_num = int(train_ray_num * foreground_ratio)
    y0, x0 = np.nonzero(que_mask)
    y1, x1 = np.nonzero(~que_mask)
    xy0 = np.stack([x0, y0], 1).astype(np.float32)
    xy1 = np.stack([x1, y1], 1).astype(np.float32)
    idx = np.arange(xy0.shape[0])
    np.random.shuffle(idx)
    xy0 = xy0[idx]
    coords0 = xy0[:min_pos_num]
    # still remain pixels
    if min_pos_num < train_ray_num:
        xy1 = np.concatenate([xy1, xy0[min_pos_num:]], 0)
        idx = np.arange(xy1.shape[0])
        np.random.shuffle(idx)
        coords1 = xy1[idx[:(train_ray_num - min_pos_num)]]
        coords = np.concatenate([coords0, coords1], 0)
    else:
        coords = coords0
    return coords


def color_map_forward(rgb):
    return rgb.astype(np.float32) / 255


def pad_img_end(img, th, tw, padding_mode='edge', constant_values=0):
    h, w = img.shape[:2]
    hp = th - h
    wp = tw - w
    if hp != 0 or wp != 0:
        if padding_mode == 'constant':
            img = np.pad(img, ((0, hp), (0, wp), (0, 0)), padding_mode, constant_values=constant_values)
        else:
            img = np.pad(img, ((0, hp), (0, wp), (0, 0)), padding_mode)
    return img

def random_crop(ref_imgs_info, que_imgs_info, target_size):
    imgs = ref_imgs_info['imgs']
    n, _, h, w = imgs.shape
    out_h, out_w = target_size[0], target_size[1]
    if out_w >= w or out_h >= h:
        return ref_imgs_info

    center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
    center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)

    def crop(tensor):
        tensor = tensor[:, :, center_h - out_h // 2:center_h + out_h // 2,
                              center_w - out_w // 2:center_w + out_w // 2]
        return tensor

    def crop_imgs_info(imgs_info):
        imgs_info['imgs'] = crop(imgs_info['imgs'])
        if 'depth' in imgs_info: imgs_info['depth'] = crop(imgs_info['depth'])
        if 'true_depth' in imgs_info: imgs_info['true_depth'] = crop(imgs_info['true_depth'])
        if 'masks' in imgs_info: imgs_info['masks'] = crop(imgs_info['masks'])

        Ks = imgs_info['Ks'] # n, 3, 3
        h_init = center_h - out_h // 2
        w_init = center_w - out_w // 2
        Ks[:,0,2]-=w_init
        Ks[:,1,2]-=h_init
        imgs_info['Ks']=Ks
        return imgs_info

    return crop_imgs_info(ref_imgs_info), crop_imgs_info(que_imgs_info)

def random_flip(ref_imgs_info,que_imgs_info):
    def flip(tensor):
        tensor = np.flip(tensor.transpose([0, 2, 3, 1]), 2)  # n,h,w,3
        tensor = np.ascontiguousarray(tensor.transpose([0, 3, 1, 2]))
        return tensor

    def flip_imgs_info(imgs_info):
        imgs_info['imgs'] = flip(imgs_info['imgs'])
        if 'depth' in imgs_info: imgs_info['depth'] = flip(imgs_info['depth'])
        if 'true_depth' in imgs_info: imgs_info['true_depth'] = flip(imgs_info['true_depth'])
        if 'masks' in imgs_info: imgs_info['masks'] = flip(imgs_info['masks'])

        Ks = imgs_info['Ks']  # n, 3, 3
        Ks[:, 0, :] *= -1
        w = imgs_info['imgs'].shape[-1]
        Ks[:, 0, 2] += w - 1
        imgs_info['Ks'] = Ks
        return imgs_info

    ref_imgs_info = flip_imgs_info(ref_imgs_info)
    que_imgs_info = flip_imgs_info(que_imgs_info)
    return ref_imgs_info, que_imgs_info

def pad_imgs_info(ref_imgs_info,pad_interval):
    ref_imgs, ref_depths, ref_masks = ref_imgs_info['imgs'], ref_imgs_info['depth'], ref_imgs_info['masks']
    ref_depth_gt = ref_imgs_info['true_depth'] if 'true_depth' in ref_imgs_info else None
    rfn, _, h, w = ref_imgs.shape
    ph = (pad_interval - (h % pad_interval)) % pad_interval
    pw = (pad_interval - (w % pad_interval)) % pad_interval
    if ph != 0 or pw != 0:
        ref_imgs = np.pad(ref_imgs, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
        ref_depths = np.pad(ref_depths, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
        ref_masks = np.pad(ref_masks, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
        if ref_depth_gt is not None:
            ref_depth_gt = np.pad(ref_depth_gt, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
    ref_imgs_info['imgs'], ref_imgs_info['depth'], ref_imgs_info['masks'] = ref_imgs, ref_depths, ref_masks
    if ref_depth_gt is not None:
        ref_imgs_info['true_depth'] = ref_depth_gt
    return ref_imgs_info

def build_imgs_info(database, ref_ids, pad_interval=-1, is_aligned=True, align_depth_range=False, has_depth=True, replace_none_depth=False, add_label=True, num_classes=0):
    if not is_aligned:
        assert has_depth
        rfn = len(ref_ids)
        ref_imgs, ref_labels, ref_masks, ref_depths, shapes = [], [], [], [], []
        for ref_id in ref_ids:
            img = database.get_image(ref_id)
            if add_label:
                label = database.get_label(ref_id)
                ref_labels.append(label)
            shapes.append([img.shape[0], img.shape[1]])
            ref_imgs.append(img)
            ref_masks.append(database.get_mask(ref_id))
            ref_depths.append(database.get_depth(ref_id))

        shapes = np.asarray(shapes)
        th, tw = np.max(shapes, 0)
        for rfi in range(rfn):
            ref_imgs[rfi] = pad_img_end(ref_imgs[rfi], th, tw, 'reflect')
            ref_labels[rfi] = pad_img_end(ref_labels[rfi], th, tw, 'reflect')
            ref_masks[rfi] = pad_img_end(ref_masks[rfi][:, :, None], th, tw, 'constant', 0)[..., 0]
            ref_depths[rfi] = pad_img_end(ref_depths[rfi][:, :, None], th, tw, 'constant', 0)[..., 0]
        ref_imgs = color_map_forward(np.stack(ref_imgs, 0)).transpose([0, 3, 1, 2])
        ref_labels = np.stack(ref_labels, 0).transpose([0, 3, 1, 2])
        ref_masks = np.stack(ref_masks, 0)[:, None, :, :]
        ref_depths = np.stack(ref_depths, 0)[:, None, :, :]
    else:
        ref_imgs = color_map_forward(np.asarray([database.get_image(ref_id) for ref_id in ref_ids])).transpose([0, 3, 1, 2])
        ref_labels = np.asarray([database.get_label(ref_id) for ref_id in ref_ids])[:, None, :, :]
        ref_masks =  np.asarray([database.get_mask(ref_id) for ref_id in ref_ids], dtype=np.float32)[:, None, :, :]
        if has_depth:
            ref_depths = [database.get_depth(ref_id) for ref_id in ref_ids]
            if replace_none_depth:
                b, _, h, w = ref_imgs.shape
                for i, depth in enumerate(ref_depths):
                    if depth is None: ref_depths[i] = np.zeros([h, w], dtype=np.float32)
            ref_depths = np.asarray(ref_depths, dtype=np.float32)[:, None, :, :]
        else: ref_depths = None

    ref_poses = np.asarray([database.get_pose(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_Ks = np.asarray([database.get_K(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_depth_range = np.asarray([database.get_depth_range(ref_id) for ref_id in ref_ids], dtype=np.float32)
    if align_depth_range:
        ref_depth_range[:,0]=np.min(ref_depth_range[:,0])
        ref_depth_range[:,1]=np.max(ref_depth_range[:,1])
    ref_imgs_info = {'imgs': ref_imgs, 'poses': ref_poses, 'Ks': ref_Ks, 'depth_range': ref_depth_range, 'masks': ref_masks, 'labels': ref_labels}
    if has_depth: ref_imgs_info['depth'] = ref_depths
    if pad_interval!=-1:
        ref_imgs_info = pad_imgs_info(ref_imgs_info, pad_interval)
    return ref_imgs_info

def build_render_imgs_info(que_pose,que_K,que_shape,que_depth_range):
    h, w = que_shape
    h, w = int(h), int(w)
    que_coords = np.stack(np.meshgrid(np.arange(w), np.arange(h), indexing='xy'), -1)
    que_coords = que_coords.reshape([1, -1, 2]).astype(np.float32)
    return {'poses': que_pose.astype(np.float32)[None,:,:],  # 1,3,4
            'Ks': que_K.astype(np.float32)[None,:,:],  # 1,3,3
            'coords': que_coords,
            'depth_range': np.asarray(que_depth_range, np.float32)[None, :],
            'shape': (h,w)}

def imgs_info_to_torch(imgs_info):
    for k, v in imgs_info.items():
        if isinstance(v,np.ndarray):
            imgs_info[k] = torch.from_numpy(v)
    return imgs_info

def imgs_info_slice(imgs_info, indices):
    imgs_info_out={}
    for k, v in imgs_info.items():
        imgs_info_out[k] = v[indices]
    return imgs_info_out


class ReplicaDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, self.scene_name, self.seq_id, background_size = database_name.split('/')
        background, image_size = background_size.split('_')
        self.image_size = int(image_size)
        self.background = background
        self.root_dir = f'data/replica/{self.scene_name}/{self.seq_id}'
        self.ratio = self.image_size / 640
        self.h, self.w = int(self.ratio*480), int(self.image_size)

        rgb_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "rgb", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.rgb_paths = natsorted(rgb_paths)
        # DO NOT use sorted() here!!! it will sort the name in a wrong way since the name is like rgb_1.jpg

        depth_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "depth", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.depth_paths = natsorted(depth_paths)

        label_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "semantic_class", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.label_paths = natsorted(label_paths)

        # Replica camera intrinsics
        # Pinhole Camera Model
        fx, fy, cx, cy, s = 320.0, 320.0, 319.5, 229.5, 0.0
        if self.ratio != 1.0:
            fx, fy, cx, cy = fx * self.ratio, fy * self.ratio, cx * self.ratio, cy * self.ratio
        self.K = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        c2ws = np.loadtxt(f'{self.root_dir}/traj_w_c.txt',
                          delimiter=' ').reshape(-1, 4, 4).astype(np.float32)
        self.poses = []
        transf = np.diag(np.asarray([1, -1, -1]))
        num_poses = c2ws.shape[0]
        for i in range(num_poses):
            pose = c2ws[i][:3]
            # Change the pose to OpenGL coordinate system
            # TODO: check if this is correct, our code is using OpenCV coordinate system
            # pose = transf @ pose
            pose = pose_inverse(pose)
            self.poses.append(pose)

        self.img_ids = []
        for i, rgb_path in enumerate(self.rgb_paths):
            self.img_ids.append(i)

        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[
                3, 7, 8, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 22, 23, 26, 29, 31,
                34, 35, 37, 40, 44, 47, 52, 54, 56,
                59, 60, 61, 62, 63, 64, 65, 70, 71,
                76, 78, 79, 80, 82, 83, 87, 88, 91,
                92, 93, 95, 97, 98
            ],
            max_cat_id=101
        )

    def get_image(self, img_id):
        img = imread(self.rgb_paths[img_id])
        if self.w != 640:
            img = cv2.resize(downsample_gaussian_blur(
                img, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return img

    def get_K(self, img_id):
        return self.K

    def get_pose(self, img_id):
        pose = self.poses[img_id]
        return pose.copy()

    def get_img_ids(self, check_depth_exist=False):
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def get_depth(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        img = Image.open(self.depth_paths[img_id])
        depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm to m
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_mask(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        return np.ones([h, w], bool)

    def get_depth_range(self, img_id):
        return np.asarray((0.1, 6.0), np.float32)

    def get_label(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        img = Image.open(self.label_paths[img_id])
        label = np.asarray(img, dtype=np.int32)
        label = np.ascontiguousarray(label)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        return self.label_mapping(label)

