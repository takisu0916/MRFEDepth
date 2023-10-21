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


def main(depth, intrinsic_np):
    H, W = depth.shape
    intrinsic_np[0, :] *= W
    intrinsic_np[1, :] *= H

    depth_torch = torch.from_numpy(depth).unsqueeze(0).unsqueeze(-1)  # (B, h, w, 1)

    valid_depth = depth > 0.0

    intrinsic_inv_np = np.linalg.inv(intrinsic_np)
    intrinsic_inv_torch = torch.from_numpy(intrinsic_inv_np).unsqueeze(0)  # (B, 4, 4)

    ## step.2 compute matrix A
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

    ## step.3 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
    norm_normalize = F.normalize(generated_norm, p=2, dim=3)
    norm_normalize_np = norm_normalize.squeeze().cpu().numpy()
    # print(norm_normalize_np.shape)

    norm_normalize_draw = (((norm_normalize_np + 1) / 2) * 255).astype(np.uint8)

    # cv2.imshow('aa', norm_normalize_draw)
    # cv2.waitKey(0)

    # cv2.imwrite('2420000000007.jpg', norm_normalize_draw)
    # sys.exit()




    return norm_normalize_np


if __name__ == '__main__':
    ## step.1 input
    # depth & intrinsic path
    depth_path = r"F:\Datasets\wrj\4_20_depth_test\DJI_0190\DJI_0190_output_small.npy"
    intrinsic_path = r"F:\Datasets\cam_intrinsic_matrix\intrinsic_wrj_normed.npy"

    depth_np = np.load(depth_path)
    print(depth_np)

    depth_np = np.squeeze(depth_np)
    print(depth_np, depth_np.shape)


    # load depth & intrinsic

    H, W = depth_np.shape

    depth_torch = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(-1)  # (B, h, w, 1)
    valid_depth = depth_np > 0.0

    intrinsic_np = np.load(intrinsic_path)

    intrinsic_inv_np = np.linalg.inv(intrinsic_np)
    intrinsic_inv_torch = torch.from_numpy(intrinsic_inv_np).unsqueeze(0)  # (B, 4, 4)

    print(depth_torch.shape, intrinsic_inv_torch[:, :3, :3].shape)

    ## step.2 compute matrix A
    # compute 3D points xyz
    points = get_points_coordinate(depth_torch, intrinsic_inv_torch[:, :3, :3], "cpu")
    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)

    # An = b
    matrix_a = point_matrix.view(1, 3, 25, H, W)  # (B, 3, 25, HxW)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1)  # (B, HxW, 25, 3)
    matrix_a_trans = matrix_a.transpose(3, 4)
    matrix_b = torch.ones([1, H, W, 25, 1])  #([1, 192, 640, 25, 1])

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

    ## step.3 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
    norm_normalize = F.normalize(generated_norm, p=2, dim=3)
    norm_normalize_np = norm_normalize.squeeze().cpu().numpy()
    print(norm_normalize_np)
    # sys.exit()

    # ## step.4 save normal vector
    # #对norm_normalize_np
    fname ,fename = os.path.splitext(depth_path)
    fname+= '_normal' +fename
    #
    np.save(fname, norm_normalize_np)
    norm_normalize_draw = (((norm_normalize_np + 1) / 2) * 255).astype(np.uint8)
    cv2.imwrite(fname.replace("_normal.npy", "_normal.png"), norm_normalize_draw)
    print('saved!')
