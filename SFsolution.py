import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.measure import label
import cv2

"""
    FUNC：Obtain the scaling factor and obtain the absolute depth.
    INPUT: relative_depthmap:torch.Size([1, 1, h, w]),
         relative_normalmap:Relative surface normal vector.torch.Size([1, 3, h, w])
         k:Normalized intrinsic matrix,np(4,4)
         UAV_H:The flight altitude of the drone.int
         angle:The drone angle, int, ranging from 0 to 90 (0 being the level view).
"""


class ScaleFactorSolution():
    def __init__(self, relative_depthmap, relative_normalmap, k, UAV_H, angle):
        self.relative_depthmap_tensor = relative_depthmap if type(
            relative_depthmap) == 'torch.Tensor' else torch.from_numpy(relative_depthmap).unsqueeze(0).unsqueeze(1).cuda()
        self.relative_normalmap_tensor = relative_normalmap if type(
            relative_normalmap) == 'torch.Tensor' else torch.from_numpy(relative_normalmap).permute(2, 0, 1).unsqueeze(0).cuda()
        print(
            f"self.relative_depthmap_tensor:{self.relative_depthmap_tensor.shape},self.relative_normalmap_tensor:{self.relative_normalmap_tensor.shape}")
        assert self.relative_depthmap_tensor.shape[2:] == self.relative_normalmap_tensor.shape[2:], print('The depth map and the normal map have inconsistent formats.')

        self.relative_height = self.relative_depthmap_tensor.shape[2]  # H in relative_depthmap
        self.relative_width = self.relative_depthmap_tensor.shape[3]  # W in relative_depthmap
        self.k = k  # Normalized intrinsic matrix K
        self.inv_k = k_to_inv_k(self.k, self.relative_height, self.relative_width)  # inv_k tensor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backproject_depth = BackprojectDepth(1, self.relative_height, self.relative_width).to(self.device)  # Create a projected point cloud object
        self.UAV_H = UAV_H
        self.angle = angle

    def forward(self):
        cam_points = self.backproject_depth(self.relative_depthmap_tensor, self.inv_k)
        # calculate the cosine similarity
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # Process: Choose an angle of 5 degrees, convert the angle to radians using math.radians(5), and calculate the cosine using radians with math.cos().
        threshold = math.cos(math.radians(5))  # Angles within 5 degrees are considered as perpendicular to the ground.
        # mask[1, 1, 375, 1242]
        ones, zeros = torch.ones(1, 1, self.relative_height, self.relative_width).cuda(), torch.zeros(1, 1,
                                                                                                      self.relative_height,
                                                                                                      self.relative_width).cuda()
        # "0, 1, 1" represents the unit vector perpendicular to the ground in the XYZ coordinate system.
        vertical_new = torch.cat((zeros, ones* math.tan(math.radians(self.angle)) ,ones* math.tan(math.radians(self.angle)) ), dim=1)  #
        # vertical_new = torch.cat((zeros, ones , zeros), dim=1)

        # Calculate the angle between the normal vector map and the vector perpendicular to the ground.
        cosine_sim = cos(self.relative_normalmap_tensor, vertical_new).unsqueeze(1)
        # Norm mask
        ground_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)
        c = []


        show_gm = ground_mask.squeeze().cpu().numpy()  # tensor--->np
        show_gm_to255 = (show_gm + 0) * 255


        # # display the predicted ground area
        cv2.imwrite("ground_point_source.jpg", show_gm_to255)
        print('save the predicted ground area.')



        # reduce surrounding noise
        main_connect_component = large_connect_component(show_gm_to255)
        # main_connect_component = ground_mask



        # The predicted height of the drone's camera lens.

        cam_heights = (cam_points[:, :-1, :, :] * self.relative_normalmap_tensor).sum(1).abs().unsqueeze(1)
        # for i in range(0, len(index)):
        #     print(index[i], cosine_sim[0][0][index[i][0]][index[i][1]],cam_heights[0][0][index[i][0]][index[i][1]])




        cam_heights_masked = torch.masked_select(cam_heights, main_connect_component)
        print("cam_heights_masked:",cam_heights_masked)
        cam_heights_result = torch.median(cam_heights_masked).unsqueeze(0)  # calculate the median of a Gaussian distribution
        # cam_heights_result = torch.from_numpy(np.array( 1.857)).unsqueeze(0).cuda()
        print('predicted height:', cam_heights_result)



        # test1：Select the absolute depth of a certain point
        # select_point = self.relative_depthmap_tensor[0][0][54][459]
        # print("select_point_relative_depth:", select_point)  # Obtain the relative depth of a certain point.
        # absolute_depth = self.UAV_H / cam_heights_result * select_point
        # print('select_point_absolute_depth:', absolute_depth)  # Convert it to the absolute depth of this point.



        #test2：Obtain the depth within a 9-neighborhood.
        y =396
        x =860
        relative_depthmap_np = self.relative_depthmap_tensor.squeeze().cpu().numpy()
        select_point_list = [relative_depthmap_np[y][x]]
        for i in range(1,3):
            select_point_list.append(relative_depthmap_np[y-i][x])  # y-1
            select_point_list.append(relative_depthmap_np[y][x+i])  # x+1
            select_point_list.append(relative_depthmap_np[y+i][x])  # y+1
            select_point_list.append(relative_depthmap_np[y][x-i])  # x-1
        print('Depth of 9-neighborhood：',select_point_list)
        cam_heights_result = cam_heights_result.squeeze().cpu().numpy()

        absolute_depth = np.array(select_point_list) * self.UAV_H * 1.0 / cam_heights_result
        avg_absolute_depth = np.sum(absolute_depth)/9
        print('select_point_absolute_depth:', absolute_depth)  # Convert it to the absolute depth of this point
        print('select_point_absolute_depth:', avg_absolute_depth)  # avg
        print('scale_factor', self.UAV_H * 1.0 / cam_heights_result)  # obtain the scale_factor(SF)




        # test3：The absolute depth of the entire image.
        # relative_depthmap_np = self.relative_depthmap_tensor = self.relative_depthmap_tensor.squeeze().cpu().numpy()
        # absolute_depth = self.UAV_H / cam_heights_result * relative_depthmap_np


# np --> inv_k tensor
def k_to_inv_k(k, relative_height, relative_width):
    tensor_K = k.copy()
    # tensor_K[0, :] *= relative_width
    # tensor_K[1, :] *= relative_height
    tensor_K = torch.from_numpy(tensor_K).unsqueeze(0).cuda()  # np-->tensor
    # print(tensor_K)

    inv_K = torch.inverse(tensor_K)  # K-->K逆
    return inv_K
    # print("tensor_K:", tensor_K)
    # print("inv_K:", inv_K)


def large_connect_component(bw_img):
    place = {}
    lcc = np.zeros((bw_img.shape[0], bw_img.shape[1]))
    # Obtain the labeled image and its corresponding label number.
    labeled_img, num = label(bw_img, background=0, return_num=True)

    for i in range(1, num + 1):
        place[i] = np.sum(labeled_img == i)  # Store the region labels and their counts in a tuple.
    place1 = sorted(place.items(), key=lambda x: -x[1])  # sort（In descending order）
    for i in range(int(0 * (num + 1)),int(0.2 * (num + 1))):  # Select the top 50% largest regions.
        print(place1[i])
        lcc += (labeled_img == place1[i][0])


    lcc = np.array(lcc, dtype=bool)  #int-->bool


    show_gm_to255 = (lcc + 0) * 255  # bool, np 转为[0,255]
    # cv2.imwrite("ground_point_50percent.jpg", show_gm_to255)

    lcc = torch.from_numpy(lcc).unsqueeze(0).cuda()
    return lcc

    # Select the largest connected component.
    # for i in range(1, num + 1):
    #     if np.sum(labeled_img == i) > max_num:
    #         max_num = np.sum(labeled_img == i)
    #         max_label = i
    # lcc = (labeled_img == max_label)
    # return lcc


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud深度图转为点云
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        # Generate a matrix of grid point coordinates
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')

        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):  # depth:[1,1,8294400]
        print(depth, depth.shape, type(depth))

        print(type(self.pix_coords), self.pix_coords.shape)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        print(cam_points.shape)  # [1, 3, 8294400]

        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        print(cam_points.shape)
        cam_points = torch.cat([cam_points, self.ones], 1).reshape(self.batch_size, 4, self.height,
                                                                   self.width)  # [cam_points, self.ones]-->（1，4，375, 1242）
        print(cam_points, cam_points.shape)
        return cam_points



if __name__ == '__main__':
    depthmap = np.load(r'F:\Datasets\wrj\4_20_depth_test\DJI_0190\DJI_0190_output_small.npy')
    normalmap = np.load(r'F:\Datasets\wrj\4_20_depth_test\DJI_0190\DJI_0190_output_small_normal.npy')

    K = np.array([[0.8211, 0, 0.4919, 0],
                  [0, 1.2315, 0.4650, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)

    test = ScaleFactorSolution(depthmap, normalmap, K, 30, 45)   #Hight=30,Angle=45
    test.forward()
