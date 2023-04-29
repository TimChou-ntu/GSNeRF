from scipy.spatial import transform
import numpy as np
import torch.nn.functional as F
import torch
import json
import os

def blender_quat2rot(quaternion):
    """Convert quaternion to rotation matrix.
    Equivalent to, but support batched case:
    ```python
    rot3x3 = mathutils.Quaternion(quaternion).to_matrix()
    ```
    Args:
    quaternion:
    Returns:
    rotation matrix
    """

    # Note: Blender first cast to double values for numerical precision while
    # we're using float32.
    q = np.sqrt(2) * quaternion

    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    qda = q0 * q1
    qdb = q0 * q2
    qdc = q0 * q3
    qaa = q1 * q1
    qab = q1 * q2
    qac = q1 * q3
    qbb = q2 * q2
    qbc = q2 * q3
    qcc = q3 * q3

    # Note: idx are inverted as blender and numpy convensions do not
    # match (x, y) -> (y, x)
    rotation = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float32)
    rotation[..., 0, 0] = 1.0 - qbb - qcc
    rotation[..., 1, 0] = qdc + qab
    rotation[..., 2, 0] = -qdb + qac

    rotation[..., 0, 1] = -qdc + qab
    rotation[..., 1, 1] = 1.0 - qaa - qcc
    rotation[..., 2, 1] = qda + qbc

    rotation[..., 0, 2] = qdb + qac
    rotation[..., 1, 2] = -qda + qbc
    rotation[..., 2, 2] = 1.0 - qaa - qbb
    return rotation

def make_transform_matrix(positions,rotations,):
    """Create the 4x4 transformation matrix.
    Note: This function uses numpy.
    Args:
    positions: Translation applied after the rotation.
        Last column of the transformation matrix
    rotations: Rotation. Top-left 3x3 matrix of the transformation matrix.
    Returns:
    transformation_matrix:
    """
    # Create the 4x4 transformation matrix
    rot_pos = np.broadcast_to(np.eye(4), (*positions.shape[:-1], 4, 4)).copy()
    rot_pos[..., :3, :3] = rotations
    # Note: Blender and numpy use different convensions for the translation
    rot_pos[..., :3, 3] = positions
    return rot_pos

def from_position_and_quaternion(positions, quaternions, use_unreal_axes):
    if use_unreal_axes:
        rotations = transform.Rotation.from_quat(quaternions).as_matrix()
    else:
        # Rotation matrix that rotates from world to object coordinates.
        # Warning: Rotations should be given in blender convensions as
        # scipy.transform uses different convensions.
        rotations = blender_quat2rot(quaternions)
    px2world_transform = make_transform_matrix(positions=positions,rotations=rotations)
    return px2world_transform

def scale_rays(all_rays_o, all_rays_d, scene_boundaries, img_wh):
    """Rescale scene boundaries.
    rays_o: (len(image_paths)*h*w, 3)
    rays_d: (len(image_paths)*h*w, 3)
    scene_boundaries: np.array(2 ,3), [min, max]
    img_wh: (2)
    """
    # Rescale (x, y, z) from [min, max] -> [-1, 1]
    # all_rays_o = all_rays_o.reshape(-1, img_wh[0], img_wh[1], 3) # (len(image_paths), h, w, 3))
    # all_rays_d = all_rays_d.reshape(-1, img_wh[0], img_wh[1], 3)
    assert all_rays_o.shape[-1] == 3, "all_rays_o should be (chunk, 3)"
    assert all_rays_d.shape[-1] == 3, "all_rays_d should be (chunk, 3)"
    old_min = torch.from_numpy(scene_boundaries[0]).to(all_rays_o.dtype).to(all_rays_o.device)
    old_max = torch.from_numpy(scene_boundaries[1]).to(all_rays_o.dtype).to(all_rays_o.device)
    new_min = torch.tensor([-1,-1,-1]).to(all_rays_o.dtype).to(all_rays_o.device)
    new_max = torch.tensor([1,1,1]).to(all_rays_o.dtype).to(all_rays_o.device)
    # scale = max(scene_boundaries[1] - scene_boundaries[0])/2
    # all_rays_o = (all_rays_o - torch.mean(all_rays_o, dim=-1, keepdim=True)) / scale
    # This is from jax3d.interp, kind of weird but true
    all_rays_o = ((new_min - new_max) / (old_min - old_max))*all_rays_o + (old_min * new_max - new_min * old_max) / (old_min - old_max)
    
    # We also need to rescale the camera direction by bbox.size.
    # The direction can be though of a ray from a point in space (the camera
    # origin) to another point in space (say the red light on the lego
    # bulldozer). When we scale the scene in a certain way, this direction
    # also needs to be scaled in the same way.
    all_rays_d = all_rays_d * 2 / (old_max - old_min)
    # (re)-normalize the rays
    all_rays_d = all_rays_d / torch.linalg.norm(all_rays_d, dim=-1, keepdims=True)
    return all_rays_o.reshape(-1, 3), all_rays_d.reshape(-1, 3)


def calculate_near_and_far(rays_o, rays_d, bbox_min=[-1.,-1.,-1.], bbox_max=[1.,1.,1.]):
    '''
    rays_o, (len(self.split_ids)*h*w, 3)
    rays_d, (len(self.split_ids)*h*w, 3)
    bbox_min=[-1,-1,-1], 
    bbox_max=[1,1,1]
    '''
    # map all shape to same (len(self.split_ids)*h*w, 3, 2)
    corners = torch.stack((torch.tensor(bbox_min),torch.tensor(bbox_max)), dim=-1).to(rays_o.dtype).to(rays_o.device)
    corners = corners.unsqueeze(0).repeat(rays_o.shape[0],1,1) # (len(self.split_ids)*h*w, 3, 2)
    corners -= torch.unsqueeze(rays_o, -1).repeat(1,1,2)
    intersections = (corners / (torch.unsqueeze(rays_d, -1).repeat(1,1,2)))

    min_intersections = torch.amax(torch.amin(intersections, dim=-1), dim=-1, keepdim=True)
    max_intersections = torch.amin(torch.amax(intersections, dim=-1), dim=-1, keepdim=True)
    epsilon = 1e-3*torch.ones_like(min_intersections)
    near = torch.maximum(epsilon, min_intersections)
    # tmp = near
    near = torch.where((near > max_intersections), epsilon, near)
    far = torch.where(near < max_intersections, max_intersections, near+epsilon)

    return near, far