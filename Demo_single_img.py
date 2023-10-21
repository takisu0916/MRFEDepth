from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import *
import torch
from torchvision import transforms, datasets
from cv2 import imwrite
import cv2
import networks
from layers import disp_to_depth

import ComputeNormal
import SFsolution


from utils import download_model_if_doesnt_exist
cv2.setNumThreads(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        # default=r"F:\Datasets\depth_map_430to506",
                        # default=r"F:\Git\yolov7-main\inference\latest\5040000000045.jpg",
                        default=r"G:\dataset\MOT\MOT17\train\MOT17-09-DPM\img1\000146.jpg",
                        type=str,

                        help='测试图片路径或者文件夹路径')
    parser.add_argument('--model_folder',
                        default=r'logs\eia_k5_dl_lr20',
                        type=str,
                        help='深度估计模型存放目录')
    parser.add_argument('--model_name',
                        default='w19',
                        type=str,
                        help='深度估计模型名称')
    parser.add_argument('--ext',
                        default='jpg',
                        type=str,
                        help='检测试图片格式')
    parser.add_argument("--GT_min_depth_percentile",  # GT min %
                        help="minimum visualization depth percentile",
                        type=float, default=5)  # 5
    parser.add_argument("--GT_max_depth_percentile",  # GT max %
                        help="maximum visualization depth percentile",
                        type=float, default=95)  # 95

    return parser.parse_args()




def test_simple(args):
    assert args.model_name is not None, "You must specify the --model_name parameter; see README.md for an example"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = os.path.join(args.model_folder, args.model_name)    #'...model_folder/model_name'
    print("Loading model from ", model_path)

    depth_encoder_path = os.path.join(model_path, "DepthEncoder.pth")    #'...model_folder/model_name/DepthEncoder.pth'
    depth_decoder_path = os.path.join(model_path, "DepthDecoder.pth")    #'...model_folder/model_name/DepthDecoder.pth'

    # DepthEncoder
    print("Loading pretrained encoder")
    encoder = networks.DepthEncoder.hrnet18(False)
    encoder.num_ch_enc = [64, 18, 36, 72, 144]
    # print('DepthEncoder:', encoder)


    loaded_dict_enc = torch.load(depth_encoder_path, map_location=device)

    feed_height,feed_width = loaded_dict_enc['height'],loaded_dict_enc['width']
    print(f'The height:{feed_height} and width:{feed_width} of the network input')

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    encoder.to(device)
    encoder.eval()
    para_sum_encoder = sum(p.numel() for p in encoder.parameters())

    # DepthDecoder
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)


    depth_decoder.to(device)
    depth_decoder.eval()
    para_sum_decoder = sum(p.numel() for p in depth_decoder.parameters())

    para_sum = para_sum_decoder + para_sum_encoder
    print(f"encoder has {para_sum_encoder} parameters")
    print(f"depth_decoder has {para_sum_decoder} parameters")
    print(f"encoder and depth_ decoder have  total {para_sum} parameters")


    if os.path.isfile(args.image_path):
        paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.jpg'))
    print(f"-> Predicting on {len(paths):d} test images")





    # start
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"): continue
            input_image = cv2.imread(image_path)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            rgb = input_image
            original_height,original_width = input_image.shape[0],input_image.shape[1] # original_height,original_width
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f'Predicting file: {image_name}')
            image_pre_path = os.path.split(image_path)[0]
            # image_class_name = os.path.basename(os.path.abspath(os.path.join(image_pre_path, '..\..\..'))) get e.g. DJI_0166ok
            input_image = cv2.resize(input_image,(feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            input_image = input_image.to('cuda')  # GPU shape[1,3,feed_height,feed_width]

            # net process
            features = encoder(input_image)  #
            outputs = depth_decoder(features)  #
            disp = outputs[("disp", 0)]  #  #shape[1,1,feed_height,feed_width]

            disp_resized = torch.nn.functional.interpolate(disp, (576, 1024), mode="bilinear",
                                                           align_corners=False)  #  shape[1,1,original_height,original_width]

            _, depth = disp_to_depth(disp_resized, 0.1,100)  # disp_to_depth function is in layers.py

            # disp_resized = disp_resized.squeeze().cpu().numpy()

            # plt process
            depth = depth.squeeze().cpu().numpy()
            vmin,vmax = np.percentile(depth,[5,95] )
            print(rf'min_depth{vmin}, max_depth {vmax},med:{np.median(depth)}')


            # bin
            depth[depth < vmin] = vmin
            depth[depth > vmax] = vmax

            normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
            colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)  # [0,1]-->[0,255]
            #
            # #
            # output_dir = os.path.join(rf'outputs_jvji',
            #                           rf'{args.model_name}',
            #                           rf'{image_class_name}',
            #                           f'{image_name}_{feed_height}_{feed_width}_disp_rgb.jpg')  #
            # if not os.path.exists(os.path.dirname(output_dir)): os.makedirs(os.path.dirname(output_dir))

            # image = np.concatenate([rgb, colormapped_im], 0)


            # #show
            image = pil.fromarray(colormapped_im)
            image.show()


            # save
            # cv2.imwrite('disp_rgb.jpg', colormapped_im[:,:,::-1])

            # name = os.path.splitext(os.path.basename(image_path))[0]
            # np.save(f'{name}_depth.npy',depth)














            # [rel --> abs]
            # intrinsic_np = np.array([[0.78913, 0, 0.49802, 0],
            #                [0, 1.40643, 0.45859, 0],
            #                [0, 0, 1, 0],
            #                [0, 0, 0, 1]], dtype=np.float32)
            #
            #
            #
            # norm_normalize_np = ComputeNormal.main(depth*10, intrinsic_np)  # 获得图片法向量
            # # print(norm_normalize_np.shape)
            #
            # test = SFsolution.ScaleFactorSolution(depth*10, norm_normalize_np, intrinsic_np, 30, 45)
            # test.forward()






















        print('-> Done!')



if __name__ == '__main__':
    args = parse_args()
    test_simple(args)



    # id_list = os.listdir(args.image_path)
    # for id in tqdm(id_list):
    #     temp = args.image_path
    #     args.image_path = os.path.join(args.image_path, id, 'image_02', 'data', 'images')
    #     test_simple(args)
    #     args.image_path = temp