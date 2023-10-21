from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import struct
import torch
from torchvision import transforms, datasets
from cv2 import imwrite
import cv2
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
import xlwt
from tqdm import *
# sys.stdout = open(os.devnull, 'w')


cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        default=r"test\wrj_img",
                        type=str,
                        help='测试图片路径或者文件夹路径')
    parser.add_argument('--model_folder',
                        default=r'logs\eia_k5_dl_lr15',
                        # default=r'F:\ablation study\results',
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
    parser.add_argument("--GT_min_depth_percentile",
                        help="minimum visualization depth percentile",
                        type=float, default=5)  # 5
    parser.add_argument("--GT_max_depth_percentile",
                        help="maximum visualization depth percentile",
                        type=float, default=95)  # 95
    return parser.parse_args()


#read bin
def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3





def evaluate(args):

    new_excel = xlwt.Workbook(encoding='ascii')
    workshell = new_excel.add_sheet('model')


    workshell.write(0, 0, 'model')
    workshell.write(0, 1, 'images')
    workshell.write(0, 2, 'abs_rel')
    workshell.write(0, 3, 'sq_rel')
    workshell.write(0, 4, 'rmse')
    workshell.write(0, 5, 'rmse_log')
    workshell.write(0, 6, 'a1')
    workshell.write(0, 7, 'a2')
    workshell.write(0, 8, 'a3')


    errors = []
    ratios = []
    pred_min_depth_m = 1e-3
    pred_max_depth_m = 80


    assert args.model_name is not None, "You must specify the --model_name parameter; see README.md for an example"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = os.path.join(args.model_folder, args.model_name)    #'...model_folder/model_name'
    print("Loading model from ", model_path)

    depth_encoder_path = os.path.join(model_path, "DepthEncoder.pth")    #'...model_folder/model_name/DepthEncoder.pth'
    depth_decoder_path = os.path.join(model_path, "DepthDecoder.pth")    #'...model_folder/model_name/DepthDecoder.pth'


    #DepthEncoder
    print("Loading pretrained encoder")
    encoder = networks.DepthEncoder.hrnet18(False)
    encoder.num_ch_enc = [64, 18, 36, 72, 144]
    loaded_dict_enc = torch.load(depth_encoder_path, map_location=device)

    feed_height,feed_width = loaded_dict_enc['height'],loaded_dict_enc['width']
    print(f'The height:{feed_height} and width:{feed_width} of the network input')

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    encoder.to(device)
    encoder.eval()
    para_sum_encoder = sum(p.numel() for p in encoder.parameters())


    #DepthDecoder
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    para_sum_decoder = sum(p.numel() for p in depth_decoder.parameters())

    para_sum = para_sum_decoder + para_sum_encoder
    print(f"depth_encoder has {para_sum_encoder} parameters.")
    print(f"depth_decoder has {para_sum_decoder} parameters.")
    print(f"encoder and depth_decoder have total {para_sum} parameters.")



    if os.path.isfile(args.image_path):
        paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.jpg'))
    print(f"-> Predicting on {len(paths):d} test images")




    with torch.no_grad():
        for idx, image_abs_path in enumerate(paths):

            if image_abs_path.endswith("_disp.jpg"): continue
            input_image = cv2.imread(image_abs_path)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


            image_pre_path = os.path.split(image_abs_path)[0]
            image_class_name = os.path.basename(os.path.abspath(os.path.join(image_pre_path, '..\..\..'))) #get e.g. DJI_0166ok
            image_name = os.path.splitext(os.path.basename(image_abs_path))[0]
            GT_image_depth_path = os.path.join(image_pre_path,rf'stereo\depth_maps\{image_name}.jpg.geometric.bin')



            GT_depth_map = read_array(GT_image_depth_path)

            print(rf'GT maps height:{GT_depth_map.shape[0]} width:{GT_depth_map.shape[1]} ')

            min_depth, max_depth = np.percentile(GT_depth_map, [args.GT_min_depth_percentile, args.GT_max_depth_percentile])
            print(rf'GT 5%min_depth{min_depth}, 95%max_depth {max_depth}')

            GT_depth_map[GT_depth_map < min_depth] = min_depth
            GT_depth_map[GT_depth_map > max_depth] = max_depth
            print('已限制GT深度范围')



            print(f'Predicting file: {image_name}')
            input_image = cv2.resize(input_image, (feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            input_image = input_image.to('cuda')  # GPU shape[1,3,feed_height,feed_width]

            # net process
            features = encoder(input_image)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)]  #shape[1,1,feed_height,feed_width]


            disp_resized = torch.nn.functional.interpolate(disp, (GT_depth_map.shape[0], GT_depth_map.shape[1]),
                                                           mode="bilinear",align_corners=False)
            _, pred_depth = disp_to_depth(disp_resized, 0.1, 100)
            pred_depth = pred_depth.squeeze().cpu().numpy()
            # pred_depth *= STEREO_SCALE_FACTOR # rel-->abs



            # compare: pred--gt
            mask = GT_depth_map > 0
            pred_depth = pred_depth[mask]
            gt_depth = GT_depth_map[mask]

            ratio = np.median(gt_depth) / np.median(pred_depth) #mid
            ratios.append(ratio)
            pred_depth *= ratio


            pred_depth[pred_depth < pred_min_depth_m] = pred_min_depth_m
            pred_depth[pred_depth > pred_max_depth_m] = pred_max_depth_m

            # abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt_depth, pred_depth)
            errors.append(compute_errors(gt_depth, pred_depth))
            # print(f'abs_rel:{abs_rel}, sq_rel:{sq_rel}, RMSE:{rmse}, rmse_log:{rmse_log}, a1:{a1}, a2:{a2}, a3:{a3}')





            workshell.write(idx + 1, 0, 'MRFEDepth')
            workshell.write(idx + 1, 1, image_name)
            workshell.write(idx + 1, 2, str(errors[idx][0]))
            workshell.write(idx + 1, 3, str(errors[idx][1]))
            workshell.write(idx + 1, 4, str(errors[idx][2]))
            workshell.write(idx + 1, 5, str(errors[idx][3]))
            workshell.write(idx + 1, 6, str(errors[idx][4]))
            workshell.write(idx + 1, 7, str(errors[idx][5]))
            workshell.write(idx + 1, 8, str(errors[idx][6]))
            workshell.write(idx + 1, 9, str(ratio))


            print(f"   Processed {idx + 1:d} of {len(paths):d} images")





        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

        for i in range(7):
            workshell.write(idx  + 3, i+2, str( mean_errors[i] ))


        #excel save
        excel_save_dir = f'excel5/{args.model_name}_{image_class_name}_{mean_errors[0]:.4f}_{mean_errors[1]:.4f}_{mean_errors[2]:.4f}_{mean_errors[3]:.4f}_{mean_errors[4]:.4f}_{mean_errors[5]:.4f}_{mean_errors[6]:.4f}_{med:.3f}.xls'
        if not os.path.exists(os.path.dirname(excel_save_dir)): os.makedirs(os.path.dirname(excel_save_dir))
        # new_excel.save(excel_save_dir)
        print('-> Done!')


















if __name__ == '__main__':
    args = parse_args()
    id_list = os.listdir(args.image_path)
    for id in tqdm(id_list):
        temp = args.image_path
        args.image_path = os.path.join(args.image_path, id, 'image_02', 'data', 'images')
        evaluate(args)
        args.image_path = temp
