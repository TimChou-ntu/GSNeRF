import torch
# This depth map seems to be slow and inaccurate
def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    
    mX = 100000*torch.ones((m,n)).to(Pts.dtype).to(Pts.device)
    mY = 100000*torch.ones((m,n)).to(Pts.dtype).to(Pts.device)
    mD = torch.zeros((m,n)).to(Pts.device)
    y_ = Pts[1].to(torch.int32)
    x_ = Pts[0].to(torch.int32)
    int_x = torch.round(Pts[0]).to(Pts.device)
    int_y = torch.round(Pts[1]).to(Pts.device)
    mX[y_,x_] = Pts[0] - int_x
    mY[y_,x_] = Pts[1] - int_y
    mD[y_,x_] = Pts[2]
    
    KmX = torch.zeros((ng, ng, m - ng, n - ng)).to(Pts.device)
    KmY = torch.zeros((ng, ng, m - ng, n - ng)).to(Pts.device)
    KmD = torch.zeros((ng, ng, m - ng, n - ng)).to(Pts.device)
    
    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    S = torch.zeros_like(KmD[0,0]).to(Pts.device)
    Y = torch.zeros_like(KmD[0,0]).to(Pts.device)
    
    for i in range(ng):
        for j in range(ng):
            s = 1/torch.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s
            del s
    
    S[S == 0] = 1
    out = torch.zeros((m,n)).to(Pts.device)
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    del mX, mY, mD, KmX, KmY, KmD, S, Y, y_, x_, int_x, int_y
    return out


def get_target_view_depth(source_depths, source_intrinsics, source_c2ws, target_intrinsics, target_w2c, img_wh, grid_size):
    ''' 
    source_depth: [N, H, W]
    source_intrinsics: [N, 3, 3]
    source_c2ws: [N, 4, 4]
    target_intrinsics: [3, 3]
    target_w2c: [4, 4]
    img_wh: [2]
    grid_size: int
    return: depth map [H, W]
    '''
    W, H = img_wh
    N = source_depths.shape[0]
    points = []

    ys, xs = torch.meshgrid(
        torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing="ij"
    )  # pytorch's meshgrid has indexing='ij'
    ys, xs = ys.reshape(-1).to(source_intrinsics.device), xs.reshape(-1).to(source_intrinsics.device)

    for num in range(N):
        # Might need to change this to be more general (too small or too big value are not good)
        mask = source_depths[num] > 0

        dirs = torch.stack(
        [
            (xs - source_intrinsics[num][0, 2]) / source_intrinsics[num][0, 0],
            (ys - source_intrinsics[num][1, 2]) / source_intrinsics[num][1, 1],
            torch.ones_like(xs),
        ],
        -1,
        )
        rays_dir = (
            dirs @ source_c2ws[num][:3, :3].t()
        )
        rays_orig = source_c2ws[num][:3, -1].clone().reshape(1, 3).expand(rays_dir.shape[0], -1)
        rays_orig = rays_orig.reshape(H,W,-1)[mask]
        rays_depth = source_depths[num].reshape(H,W,-1)[mask]
        rays_dir = rays_dir.reshape(H,W,-1)[mask]
        ray_pts = rays_orig + rays_depth * rays_dir
        points.append(ray_pts.reshape(-1,3))

        del rays_orig, rays_depth, rays_dir, ray_pts, dirs, mask

    points = torch.cat(points,0).reshape(-1,3)

    R = target_w2c[:3, :3]  # (3, 3)
    T = target_w2c[:3, 3:]  # (3, 1)
    ray_pts_transformed = torch.matmul(points, R.t()) + T.reshape(1, 3)

    ray_pts_ndc = ray_pts_transformed @ target_intrinsics.t()
    ndc = ray_pts_ndc[:, :2] / ray_pts_ndc[:, -1:]
    # ray_pts_ndc[:, 0] = ray_pts_ndc[:, 0] / ray_pts_ndc[:, 2]
    # ray_pts_ndc[:, 1] = ray_pts_ndc[:, 1] / ray_pts_ndc[:, 2]
    mask = (ndc[:, 0] >= 0) & (ndc[:, 0] <= W-1) & (ndc[:, 1] >= 0) & (ndc[:, 1] <= H-1)
    mask = mask & (ray_pts_transformed[:, 2] > 2)
    points_2d = ndc[mask, 0:2]

    lidarOnImage = torch.cat((points_2d, ray_pts_transformed[mask,2].reshape(-1,1)), 1)
    depth_map = dense_map(lidarOnImage.t(), H, W, grid_size)
    # del ray_pts_ndc, target_intrinsics, ray_pts_transformed
    # del ys, xs, ray_pts_transformed, points, points_2d, ray_pts_ndc, mask, lidarOnImage, R, T
    # depth_map = torch.ones_like(source_depths[0]).to("cuda")

    return depth_map
