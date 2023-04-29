import torch
import matplotlib.pyplot as plt
from data.klevr import KlevrDataset
import cv2
import numpy as np
import imageio

dataset = KlevrDataset(
    root_dir="/home/timothy/Desktop/2023Spring/GeoNeRF/data/data/nesf_data/klevr/",
    split="train",
    nb_views=6
    )

data = dataset[0]
print(data.keys())
print(data["images"].shape)
print(data["images"][0])
img_vis = (
    data["images"]
    .clip(0, 1)
    .transpose((0, 2, 3, 1))
    # .reshape((data["images"].shape[-1], -1, 3))
)
img_vis = np.concatenate((img_vis[0], img_vis[1],img_vis[2], img_vis[3],img_vis[4], img_vis[5], img_vis[6]), axis=1)
print(img_vis.shape)

color = ['b','g','r','c','m','y','k','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink']

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
print(data['w2cs'].shape)
for id in range(7):
    print(id)
    ray_points = data["w2cs"][int(id)]
    print(id, ray_points.shape)
    data_ = ray_points
    x = data_[1, -1]
    y = data_[0, -1]
    z = data_[2, -1]
    ax.scatter(x, y, z,s=100, c=color[id%14])
ax.scatter(0, 0, 0,s=100, c='r')
fig2 = plt.figure()
imgplot = plt.imshow(img_vis)
plt.show()

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

