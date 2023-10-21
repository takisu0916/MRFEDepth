from __future__ import absolute_import, division, print_function
from datetime import datetime

import cv2
import numpy as np
import math, os
import time
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import torchvision

# from Computer_Loss import *
from utils import *
# from kitti_utils import *
from layers import *
import datasets
import networks
import sys
import matplotlib.pyplot as plt




project_dir = os.path.dirname(__file__)






class Trainer:
    def __init__(self, options):


        current_time_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f'$current_time_date$:{current_time_date}')

        self.opt = options
        self.log_path = os.path.join(self.opt.log_path, self.opt.model_name, current_time_date)
        print(f'$self.log_path$:{self.log_path}')

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo: self.opt.frame_ids.append("s")




        # 2.
        # 2.1Depth-Encoder(hrnet18)
        self.models["DepthEncoder"] = networks.DepthEncoder.hrnet18(True)
        self.models["DepthEncoder"].num_ch_enc = [64, 18, 36, 72, 144]
        self.models["DepthEncoder"].to(self.device)
        self.parameters_to_train += list(self.models["DepthEncoder"].parameters())
        para_sum = sum(p.numel() for p in self.models['DepthEncoder'].parameters())
        print('DepthEncoder params are ', para_sum)

        # 2.2Depth-Decoder
        self.models["DepthDecoder"] = networks.DepthDecoder(self.models["DepthEncoder"].num_ch_enc, self.opt.scales)
        self.models["DepthDecoder"].to(self.device)
        self.parameters_to_train += list(self.models["DepthDecoder"].parameters())
        para_sum = sum(p.numel() for p in self.models['DepthDecoder'].parameters())
        print('DepthDecoder params are ', para_sum)





        # 2.3Pose-Encoder(resnet18)
        self.models["PoseEncoder"] = networks.PoseEncoder(self.opt.num_layers,self.opt.weights_init == "pretrained",num_input_images=self.num_pose_frames)  # num_input_images=2
        self.models["PoseEncoder"].to(self.device)
        self.parameters_to_train += list(self.models["PoseEncoder"].parameters())
        para_sum = sum(p.numel() for p in self.models['PoseEncoder'].parameters())
        print('PoseEncoder params are ', para_sum)

        # 2.4Pose-Decoder
        self.models["PoseDecoder"] = networks.PoseDecoder(self.models["PoseEncoder"].num_ch_enc,num_input_features=1,num_frames_to_predict_for=2)
        self.models["PoseDecoder"].to(self.device)
        self.parameters_to_train += list(self.models["PoseDecoder"].parameters())
        para_sum = sum(p.numel() for p in self.models['PoseDecoder'].parameters())
        print('PoseDecoder params are ', para_sum)


        # 2.5Feature-Encoder(resnet18)
        self.models["FeatureEncoder"] = networks.FeatureEncoder(self.opt.num_layers, True)  #resnet18
        self.models["FeatureEncoder"].to(self.device)
        self.parameters_to_train += list(self.models["FeatureEncoder"].parameters())
        para_sum = sum(p.numel() for p in self.models['FeatureEncoder'].parameters())
        print('FeatureEncoder params are ', para_sum)



        # 2.6Feature-Decoder
        self.models["FeatureDecoder"] = networks.FeatureDecoder(self.models["FeatureEncoder"].num_ch_enc, 3)
        self.models["FeatureDecoder"].to(self.device)
        self.parameters_to_train += list(self.models["FeatureDecoder"].parameters())
        para_sum = sum(p.numel() for p in self.models['FeatureDecoder'].parameters())
        print('FeatureDecoder params are ', para_sum)



        # 3.
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)  # learning_rate=1e-4
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size,0.1)  #0-14（1e-4） 15-20（1e-5）

        #
        if self.opt.load_weights_folder is not None:  self.load_model()

        print("Training model named: ", self.opt.model_name)
        print("Models and tensorboard events files are saved to: ", self.log_path)
        print("Training is using: ", self.device)




        #
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset }

        self.dataset_k = datasets_dict[self.opt.dataset]
        fpath = os.path.join(project_dir, "splits", self.opt.split,"{}_files.txt")  # F:/Git/mrfedepth\splits\eigen_zhou\{}_files.txt

        # 4.1
        train_filenames_k = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.jpg'
        num_train_samples = len(train_filenames_k)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs  # 总的imgs / batchsize * epoch

        # 4.2dataloader
        #
        train_dataset_k = self.dataset_k(
            self.opt.data_path, train_filenames_k, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext='.jpg')
        self.train_loader_k = DataLoader(
            train_dataset_k, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        #
        val_dataset = self.dataset_k(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext='.jpg')
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        #ssim
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        self.num_batch_k = train_dataset_k.__len__() // self.opt.batch_size

        #
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)  # defualt=[0,1,2,3]'scales used in the loss'
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)  # in layers.py
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)


        print("Using split: ", self.opt.split)
        print(f"There are {len(train_dataset_k):d} training items and {len(val_dataset):d} validation items.")

        # self.save_opts()



    # 5.train
    # 5.0
    def set_train(self):
        for k, m in self.models.items():  m.train()

    # 5.0
    def set_eval(self):
        for m in self.models.values(): m.eval()

    # 5.1
    def train(self):
        self.init_time = time.time()
        self.epoch_start = 0 if not isinstance(self.opt.load_weights_folder, str) else int(self.opt.load_weights_folder[-1]) + 1
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs - self.epoch_start):
            self.epoch = self.epoch_start + self.epoch
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:   #100
                self.save_model()
        self.total_training_time = time.time() - self.init_time
        print(rf'====>total training time:{sec_to_hm_str(self.total_training_time)}')


    # 5.2single epoch train
    def run_epoch(self):
        print("Threads: " + str(torch.get_num_threads()))
        print("Training a epoch...")
        self.set_train()
        self.every_epoch_start_time = time.time()

        for batch_idx , inputs in enumerate(self.train_loader_k):
            before_op_time = time.time()
            outputs, losses = self.run_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time
            # print(f'this batch:{batch_idx} cost time:{duration}')

            #
            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx,duration, losses["loss"].cpu().data)  #
                # self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()
        print(f'train time of this epoch:{sec_to_hm_str(time.time()-self.every_epoch_start_time)}')





    # 5.3single batch train
    def run_batch(self,inputs):
        for key,ipt in inputs.items(): #inputs.values() has :12x3x196x640.
            inputs[key] = ipt.to(self.device)

        #depth net
        Depthfeatures = self.models["DepthEncoder"](inputs["color_aug", 0, 0])  #DepthEncoder
        outputs = self.models["DepthDecoder"](Depthfeatures)  #DepthDecoder



        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](Depthfeatures)  #output: 2:*:* mask maps

        # posenet
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, Depthfeatures))



        #featurenet
        features = self.models['FeatureEncoder'](inputs['color',0,0])  #FeatureEncoder
        outputs.update(self.models['FeatureDecoder'](features,0))








        #loss!!!!
        self.generate_images_pred(inputs, outputs)
        self.generate_features_pred(inputs, outputs)
        losses = self.compute_losses(inputs,outputs,features)
        return outputs,losses






    def generate_images_pred(self,inputs,outputs):
        for scale in self.opt.scales: #0123
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear",align_corners=False)

            _, depth = disp_to_depth(disp, self.opt.min_depth,self.opt.max_depth)  # disp_to_depth function is in layers.py

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):  #-1 1
                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
                pix_coords = self.project_3d[0](cam_points, inputs[("K", 0)], T)
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, 0)],outputs[("sample", frame_id, scale)],padding_mode="border")

                if not self.opt.disable_automasking:  #True
                    outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, 0)]


    def generate_features_pred(self, inputs, outputs):
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [int(self.opt.height/2), int(self.opt.width/2)], mode="bilinear",align_corners=False)
        _, depth = disp_to_depth(disp, self.opt.min_depth,self.opt.max_depth)  # disp_to_depth function is in layers.py
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):  # -1 1
            T = outputs[("cam_T_cam", 0, frame_id)]   #T = (-1，0)  (1，0)

            cam_points = self.backproject_depth[1](depth, inputs[("inv_K", 1)])
            pix_coords = self.project_3d[1](cam_points, inputs[("K", 1)], T)  # [batch,height,width,2] ==> torch.Size([2, 96, 128 , 2])


            src_f = self.models['FeatureEncoder'] (inputs[("color", frame_id, 0)] )[0]   #torch.Size([2, 64, 96, 128])

            outputs[("feature", frame_id, 0)] = F.grid_sample(src_f, pix_coords, padding_mode="border")

        # print('OK_generate_features_pred !! ')







    #L_AM
    def robust_l1(self,pred,target):
        # abs_diff = torch.abs(target - pred)
        # return abs_diff
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) +eps ** 2)


    # depth_loss
    def computer_depth_loss(self, tgt_f, src_f):
        alafa = 0.05
        ru = 0.85
        gi = tgt_f - src_f
        loss = alafa * torch.sqrt(torch.pow(gi, 2).mean(1, True) - ru * (torch.pow(gi.mean(1, True), 2)))

        return loss




    def compute_perceptional_loss(self, tgt_f, src_f):
        # loss = self.robust_l1(tgt_f, src_f).mean(1, True)
        loss = 0.9 * self.robust_l1(tgt_f, src_f).mean(1, True) +  self.computer_depth_loss(tgt_f,src_f)
        return loss

    #(SSIM+L1)
    def compute_reprojection_loss(self, pred, target):
        '''由-1，1重建0后，和真实的0进行计算损失

        :param pred: 将-1，1 重建成0的姿态的图像
        :param target: 0时刻的真实图像
        '''
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * photometric_loss

        return reprojection_loss





    def compute_losses(self,inputs,outputs,features):



        losses = {}
        total_loss = 0
        target = inputs[("color", 0, 0)]  # [0,0]  [2,3,192,256]



        for i in range(5):
            f = features[i]
            regularization_loss = self.get_feature_regularization_loss(f, target)

            total_loss+=(regularization_loss/(2 ** i)) / 5



        for scale in self.opt.scales:   #scales=[0,1,2,3]

            loss = 0
            reprojection_losses = []
            perceptional_losses = []
            disp = outputs[("disp", scale)]   #[0,1,2,3]
            color = inputs[("color", 0, scale)]  # [0,0] [0,1] [0,2] [0,3]





            res_img = outputs[("res_img", 0, scale)]  # featureNet decoder
            _, _, h, w = res_img.size()
            target_resize = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)  # 把target缩小到res_img大小
            img_reconstruct_loss = self.compute_reprojection_loss(res_img, target_resize)


            loss += img_reconstruct_loss.mean() / len(self.opt.scales)





            for frame_id in self.opt.frame_ids[1:]:  #-1，1
                pred = outputs[("color", frame_id, scale)]   #[-1,0][1,0]  [-1,1][1,1]  [-1,2][1,2]  [-1,3][1,3]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1) # torch.Size([2, 2, 192, 256])




            if not self.opt.disable_automasking: #do this
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]: #-1,1
                    pred = inputs[("color", frame_id, 0)]  #[-1,0]  [1,0]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))  #-1 的和 0 的img 做loss

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)  #torch.Size([2, 2, 192, 256])
                if self.opt.avg_reprojection:#no
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    #this
                    identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses   #torch.Size([2, 2, 192, 256])

            if not self.opt.disable_automasking:  #do this
                # add random numbers to break ties
                    #identity_reprojection_loss.shape).cuda() * 0.00001
                if torch.cuda.is_available():
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)   #torch.Size([2, 4, 192, 256])
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1: to_optimise = combined
            else:
                #doing this
                to_optimise, idxs = torch.min(combined, dim=1) #torch.Size([2, 192, 256])


            if not self.opt.disable_automasking:  #do this
                outputs["identity_selection/0"] = (idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()/ len(self.opt.scales)




            for frame_id in self.opt.frame_ids[1:]:  #-1,1
                src_f = outputs[("feature", frame_id, 0)]   #torch.Size([2, 64, 96, 128])
                tgt_f = self.models['FeatureEncoder'](inputs[("color", 0, 0)])[0]  #torch.Size([2, 64, 96, 128])
                perceptional_losses.append(self.compute_perceptional_loss(tgt_f, src_f))
            perceptional_loss = torch.cat(perceptional_losses, 1)  #torch.Size([2, 128, 96, 128])
            min_perceptional_loss, outputs[("min_index", scale)] = torch.min(perceptional_loss, dim=1)
            # losses[('min_perceptional_loss', scale)] = 1e-3 * min_perceptional_loss.mean() / len(self.opt.scales)
            fm_perceptional_loss = 1e-3 * min_perceptional_loss.mean() / len(self.opt.scales)

            loss += fm_perceptional_loss




            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = self.get_smooth_loss(norm_disp, color)

            loss += 1e-3 * smooth_loss / (2 ** scale) / len(self.opt.scales)


            total_loss += loss

        # total_loss /= self.num_scales
        losses["loss"] = total_loss
        # print(f'total_loss ！！{total_loss}')
        return losses







    def predict_poses(self,inputs,features):
        outputs = {}
        if self.num_pose_frames ==2:
            pose_feats =  {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]: #frame_ids = [-1,1]
                if f_i != 's':
                    if  f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]] # -1 --> 0
                        # print(len(pose_inputs))   #2
                        # print(pose_inputs[0].shape)  #torch.Size([2, 3, 192, 256])
                        # print(pose_inputs[1].shape)  #torch.Size([2, 3, 192, 256])

                    else:
                        pose_inputs = [ pose_feats[0], pose_feats[f_i]] # 0 --> 1

                    if self.opt.pose_model_type == 'separate_resnet': #default this
                        pose_inputs = [self.models["PoseEncoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["PoseDecoder"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0],invert=(f_i < 0))
        return outputs



    def get_feature_regularization_loss(self, feature, img):
        b, _, h, w = feature.size()  ##[batch, channels, height, width]
        # print(rf'==> feature shape is {b} {_} {h} {w}')
        img = F.interpolate(img, (h, w), mode='area')

        feature_dx, feature_dy = self.gradient(feature)
        img_dx, img_dy = self.gradient(img)

        feature_dxx, feature_dxy = self.gradient(feature_dx)  # feature dxx, dyx
        feature_dyx, feature_dyy = self.gradient(feature_dy)  # feature dyx, dyy

        img_dxx, img_dxy = self.gradient(img_dx)  # img dxx, dyx
        img_dyx, img_dyy = self.gradient(img_dy)  # img dyx, dyy

        # dx+dy
        smooth1 = torch.mean(feature_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
                  torch.mean(feature_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

        # dxx + 2dxy +dyy
        smooth2 = torch.mean(feature_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
                  torch.mean(feature_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
                  torch.mean(feature_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
                  torch.mean(feature_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

        # print(f"smooth1:{smooth1}  smooth2:{smooth2}")
        total_feature_smooth_loss = -1e-3 * smooth1 + 1e-3 * smooth2
        # print(f"total_smooth:{total_feature_smooth_loss}")


        return total_feature_smooth_loss


    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()  # [batch, channels, height, width]
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)  # disp
        img_dx, img_dy = self.gradient(img)  # img

        disp_dxx, disp_dxy = self.gradient(disp_dx)  # disp dxx, dyx
        disp_dyx, disp_dyy = self.gradient(disp_dy)  # disp dyx, dyy

        img_dxx, img_dxy = self.gradient(img_dx)  # img  dxx, dyx
        img_dyx, img_dyy = self.gradient(img_dy)  # img  dyx, dyy

        # dx+dy
        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-0.5 * img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-0.5 * img_dy.abs().mean(1, True)))

        # dxx + 2dxy +dyy
        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-0.5 * img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-0.5 * img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-0.5 * img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-0.5 * img_dyy.abs().mean(1, True)))

        return smooth1 + smooth2

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy


    def val(self):
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs,losses = self.run_batch(inputs)

            print(f"val loss:{losses} " ) #val loss
            # self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


    def log_time(self, batch_idx, duration, loss):
        '''

        :param batch_idx: (int)第n个batch
        :param duration: ()训练 单个batch+反向传播 的时间
        :param loss: ()特定batch的loss
        '''
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = f"Epoch {self.epoch:>3} | Batch_ID {batch_idx:>6} | Examples/s: {samples_per_sec:5.1f} |" \
                       f" Loss: {loss:.5f} | Time cost: {sec_to_hm_str(time_sofar)} | Time remain: {sec_to_hm_str(training_time_left)}"


        print(print_string)



    def save_opts(self):
        models_dir = self.log_path
        if not os.path.exists(models_dir):  os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)


    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):  os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'DepthEncoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "adam.pth")
        torch.save(self.model_optimizer.state_dict(), save_path)


    def load_model(self):
        assert  os.path.isdir(self.opt.load_weights_folder), rf"Cannot find folder {self.opt.load_weights_folder}"
        print(rf"loading model from folder: {self.opt.load_weights_folder}")
        for n in self.opt.models_to_load:
            print(rf'Loading {n} weights...')
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")







