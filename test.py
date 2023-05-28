import torch
import matplotlib.pyplot as plt
from data.klevr import KlevrDataset
import cv2
import numpy as np
import imageio
import open3d as o3d


dataset = KlevrDataset(
    root_dir="/home/timothy/Desktop/2023Spring/GeoNeRF/data/data/nesf_data/klevr/",
    split="val",
    nb_views=6,
    get_semantic=True,
    )

data = dataset[0]
print(data.keys())
pcd = o3d.geometry.PointCloud()

# From Github https://github.com/balcilar/DenseDepthMap
def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    
    mX = torch.zeros((m,n)) + torch.tensor(100000).to(torch.float32)
    mY = torch.zeros((m,n)) + torch.tensor(100000).to(torch.float32)
    mD = torch.zeros((m,n))
    y_ = Pts[1].to(torch.int32)
    x_ = Pts[0].to(torch.int32)
    mX[y_,x_] = Pts[0] - torch.round(Pts[0])
    mY[y_,x_] = Pts[1] - torch.round(Pts[1])
    mD[y_,x_] = Pts[2]
    
    KmX = torch.zeros((ng, ng, m - ng, n - ng))
    KmY = torch.zeros((ng, ng, m - ng, n - ng))
    KmD = torch.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    S = torch.zeros_like(KmD[0,0])
    Y = torch.zeros_like(KmD[0,0])
    
    for i in range(ng):
        for j in range(ng):
            s = 1/torch.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s
    
    S[S == 0] = 1
    out = torch.zeros((m,n))
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    return out

source_depths = data["depths_h"][:6]
target_depths = data["depths_h"][6:]
points = []
H, W = 256, 256

ys, xs = torch.meshgrid(
    torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing="ij"
)  # pytorch's meshgrid has indexing='ij'
for num in range(source_depths.shape[0]):
    mask = source_depths[num] > 0
    # print(mask.shape)
    ys, xs = ys.reshape(-1), xs.reshape(-1)

    dirs = torch.stack(
    [
        (xs - data["intrinsics"][num][0, 2]) / data["intrinsics"][num][0, 0],
        (ys - data["intrinsics"][num][1, 2]) / data["intrinsics"][num][1, 1],
        torch.ones_like(xs),
    ],
    -1,
    )
    rays_dir = (
        dirs @ torch.asarray(data["c2ws"][num][:3, :3]).t()
    )
    rays_orig = torch.asarray(data["c2ws"][num][:3, -1]).clone().reshape(1, 3).expand(rays_dir.shape[0], -1)
    rays_orig = rays_orig.reshape(H,W,-1)[mask]
    rays_depth = torch.asarray(source_depths[num]).reshape(H,W,-1)[mask]
    rays_dir = rays_dir.reshape(H,W,-1)[mask]
    print(rays_orig.shape)
    print(rays_dir.shape)
    print(rays_depth.shape)
    ray_pts = rays_orig + rays_depth * rays_dir
    points.append(ray_pts.reshape(-1,3))

points = torch.cat(points,0).reshape(-1,3)
print(points.shape)

w2c_ref = torch.asarray(data["w2cs"][6])
intrinsics_ref = torch.asarray(data["intrinsics"][6])
img_width = 256
img_height = 256

R = w2c_ref[:3, :3]  # (3, 3)
T = w2c_ref[:3, 3:]  # (3, 1)
ray_pts = torch.matmul(points, R.t()) + T.reshape(1, 3)

ray_pts_ndc = ray_pts @ intrinsics_ref.t()
ray_pts_ndc[:, 0] /= ray_pts_ndc[:, 2]
ray_pts_ndc[:, 1] /= ray_pts_ndc[:, 2]
mask = (ray_pts_ndc[:, 0] >= 0) & (ray_pts_ndc[:, 0] <= img_width) & (ray_pts_ndc[:, 1] >= 0) & (ray_pts_ndc[:, 1] <= img_height)
mask = mask & (ray_pts[:, 2] > 2)
points_2d = ray_pts_ndc[mask, 0:2]

lidarOnImage = torch.cat((points_2d, ray_pts[mask,2].reshape(-1,1)), 1)

out = []
for i in range(5):
    out.append(dense_map(lidarOnImage.T, 256, 256, i+1))
    print(out[i].shape)
fig, axs = plt.subplots(5, 3)
d_gt = np.asarray(data["depths_h"][6])
axs[0,0].imshow(d_gt)
d_merge = np.asarray(out[0])
axs[0,1].imshow(d_merge)
d_error = np.abs(d_gt - d_merge)
print("grid0 error:",d_error.sum())
axs[0,2].imshow(d_error)
axs[1,0].imshow(d_gt)
d_merge = np.asarray(out[1])
axs[1,1].imshow(d_merge)
d_error = np.abs(d_gt - d_merge)
print("grid1 error:",d_error.sum())
axs[1,2].imshow(d_error)
axs[2,0].imshow(d_gt)
d_merge = np.asarray(out[2])
axs[2,1].imshow(d_merge)
d_error = np.abs(d_gt - d_merge)
print("grid2 error:",d_error.sum())
axs[2,2].imshow(d_error)
axs[3,0].imshow(d_gt)
d_merge = np.asarray(out[3])
axs[3,1].imshow(d_merge)
d_error = np.abs(d_gt - d_merge)
print("grid3 error:",d_error.sum())
axs[3,2].imshow(d_error)
axs[4,0].imshow(d_gt)
d_merge = np.asarray(out[4])
axs[4,1].imshow(d_merge)
d_error = np.abs(d_gt - d_merge)
print("grid4 error:",d_error.sum())
axs[4,2].imshow(d_error)
plt.show()
# pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd])

pcd = o3d.t.geometry.PointCloud()
for i in range(6):
    d = o3d.geometry.Image(data["depths_h"][i])
    print(np.asarray(d).max())
    print(np.asarray(d).min())
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = data["intrinsics"][i]
    pcd_i = o3d.geometry.PointCloud.create_from_depth_image(d,cam,data["w2cs"][i])
    point_array = np.asarray(pcd_i.points)
    if i == 0:
        total_point = point_array
    else:
        total_point = np.concatenate((total_point,point_array),axis=0)
pcd.point.positions = o3d.core.Tensor(total_point, dtype=o3d.core.Dtype.Float32)
print("Total point number: ",total_point.shape)
# pcd.points = o3d.utility.Vector3dVector(total_point)
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw([pcd])
# print(np.asarray(pcd.points)[:,2].max())
# print(np.asarray(pcd.points)[:,2].min())
import sys
np.set_printoptions(threshold=sys.maxsize)
# depth_reproj = pcd.project_to_depth_image(width=256,height=256,intrinsic=o3d.core.Tensor(data["intrinsics"][6]),extrinsic=o3d.core.Tensor(data["w2cs"][6]))
print(type(pcd))
print(type(o3d.core.Tensor(data["intrinsics"][6])))
# depth_reproj = pcd.project_to_depth_image(width=256,height=256,intrinsic=o3d.core.Tensor(data["intrinsics"][6]))
print(data["intrinsics"][6])
print(data["w2cs"][6])
intrinsic = o3d.core.Tensor(data["intrinsics"][6])
extrinsic = o3d.core.Tensor(data["w2cs"][6])
depth_reproj = pcd.project_to_depth_image(256,
                                        256,
                                        intrinsic,
                                        extrinsic,
                                        depth_scale=1.0,
                                        depth_max=17.0)
fig, axs = plt.subplots(1, 3)
d_gt = np.asarray(data["depths_h"][6])
axs[0].imshow(d_gt)
d_merge = np.asarray(depth_reproj.to_legacy())
# print(d_merge.shape)
# print(d_merge)
# print(d_merge.max())
# print(d_merge.min())
axs[1].imshow(d_merge)
d_error = np.abs(d_gt - d_merge)
axs[2].imshow(d_error)
print("error:",d_error.sum())
plt.show()
print("Different: ",(torch.asarray(d_merge) - out[0]).sum().abs())
# print(data["images"][0])
# img_vis = (
#     data["images"]
#     .clip(0, 1)
#     .transpose((0, 2, 3, 1))
#     # .reshape((data["images"].shape[-1], -1, 3))
# )
# img_vis = np.concatenate((img_vis[0], img_vis[1],img_vis[2], img_vis[3],img_vis[4], img_vis[5], img_vis[6]), axis=1)
# print(img_vis.shape)

# color = ['b','g','r','c','m','y','k','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink']

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# print(data['w2cs'].shape)
# for id in range(7):
#     print(id)
#     ray_points = data["w2cs"][int(id)]
#     print(id, ray_points.shape)
#     data_ = ray_points
#     x = data_[1, -1]
#     y = data_[0, -1]
#     z = data_[2, -1]
#     ax.scatter(x, y, z,s=100, c=color[id%14])
# ax.scatter(0, 0, 0,s=100, c='r')
# fig2 = plt.figure()
# imgplot = plt.imshow(img_vis)
# plt.show()

# os.makedirs(
#     f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/nan_batch/",
#     exist_ok=True,
# )
# imageio.imwrite(
#     f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/nan_batch/{self.global_step:08d}_{batch_nb:04d}.png",
#     (img_vis * 255).astype("uint8"),
# )
# cv.imshow("img_vis", img_vis)
# cv.waitKey()

# # Load the two images
# img1 = data["images"][0]
# img2 = data["images"][1]

# # Load the camera matrices
# K1 = data["intrinsics"][0]
# K2 = data["intrinsics"][1]
# print(K1.shape)

# R1 = data["w2cs"][0][:3, :3]
# R2 = data["w2cs"][1][:3, :3]
# t1 = data["w2cs"][0][:3, 3:]
# t2 = data["w2cs"][1][:3, 3:]
# print(R1.shape)
# print(t1.shape)

# # Compute the essential matrix
# # E, _ = cv2.findEssentialMat(t1, t2, K1, cv2.RANSAC)
# E = cv2.essentialFromRt(R1,t1,R2,t2)
# # Compute the fundamental matrix
# F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

# # Select some points in the first image
# points1 = np.array([[100, 100],
#                     [200, 200],
#                     [150, 150]])


# # Compute the projection matrices
# P1 = np.dot(K1, data["w2cs"][0][:3,:])
# P2 = np.dot(K2, data["w2cs"][1][:3,:])


# # Compute the corresponding epipolar lines in the second image
# lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)

# # Draw the epipolar lines in the second image
# img2_lines = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# for r, pt1 in zip(lines2, points1):
#     a, b, c = r[0]
#     x0, y0 = 0, int(-c/b)
#     x1, y1 = img2.shape[1], int(-(c+a*x1)/b)
#     img2_lines = cv2.line(img2_lines, (x0, y0), (x1, y1), (0, 255, 0), 1)

# # Show the images with the epipolar lines
# cv2.imshow("Image 1", img1)
# cv2.imshow("Image 2 with epipolar lines", img2_lines)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

