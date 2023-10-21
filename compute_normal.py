import sys, os, cv2
import torch
import torch.nn.functional as F
import numpy as np

def get_points_coordinate(depth, instrinsic_inv, device="cuda"):
    B, height, width, C = depth.size()
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                           torch.arange(0, width, dtype=torch.float32, device=device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(instrinsic_inv, xyz)  # [B, 3, H*W]
    depth_xyz = xyz * depth.view(B, 1, -1)  # [B, 3, Ndepth, H*W]

    return depth_xyz.view(B, 3, height, width)


if __name__ == '__main__':
    ## step.1 get K
    K_np = np.array([[0.78913, 0, 0.49802, 0],
                     [0, 1.40643, 0.45859, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)
    intrinsic_inv_np = np.linalg.inv(K_np)
    intrinsic_inv_torch = torch.from_numpy(intrinsic_inv_np).unsqueeze(0)  # (B, 4, 4)

    ## step.2 get depth map
    # depth_path = r"F:\Datasets\wrj\4_20_depth_test\DJI_0190\DJI_0190_output_small.npy"
    depth_path = r"output_small_normal.npy"
    # output_normal_path = r"output_small_normal.npy"
    depth_np = np.load(depth_path)

    depth_np = np.squeeze(depth_np)
    print(depth_np, depth_np.shape)
    H, W = depth_np.shape
    depth_torch = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(-1)  # (B, h, w, 1)
    valid_depth = depth_np > 0.0

    ## step.3 compute matrix A
    # compute 3D points xyz
    points = get_points_coordinate(depth_torch, intrinsic_inv_torch[:, :3, :3], "cpu")
    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)

    # An = b
    matrix_a = point_matrix.view(1, 3, 25, H, W)  # (B, 3, 25, HxW)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1)  # (B, HxW, 25, 3)
    matrix_a_trans = matrix_a.transpose(3, 4)
    matrix_b = torch.ones([1, H, W, 25, 1])  # ([1, 192, 640, 25, 1])

    # dot(A.T, A)
    point_multi = torch.matmul(matrix_a_trans, matrix_a)
    matrix_deter = torch.det(point_multi.to("cpu"))
    # make inversible
    inverse_condition = torch.ge(matrix_deter, 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
    # diag matrix to update uninverse
    diag_constant = torch.ones([3], dtype=torch.float32)
    diag_element = torch.diag(diag_constant)
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(1, H, W, 1, 1)
    # inversible matrix
    inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
    inv_matrix = torch.inverse(inversible_matrix.to("cpu"))

    ## step.4 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
    norm_normalize = F.normalize(generated_norm, p=2, dim=3)
    norm_normalize_np = norm_normalize.squeeze().cpu().numpy()
    # print(norm_normalize_np)

    ## step.4
    np.save(output_normal_path, norm_normalize_np)
    norm_normalize_draw = (((norm_normalize_np + 1) / 2) * 255).astype(np.uint8)
    cv2.imwrite(output_normal_path.replace("_normal.npy", "_normal.png"), norm_normalize_draw)









