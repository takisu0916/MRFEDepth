from __future__ import absolute_import, division, print_function
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1, bias=bias)
    def forward(self, x):
        out = self.conv(x)
        return out


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")



class FeatureDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=3):
        super(FeatureDecoder, self).__init__()
        num_ch_dec = [16, 32, 64, 128, 256]

        # upconv
        self.upconv5 = ConvBlock(num_ch_enc[4], num_ch_dec[4])  # 256,256,3x3,p1
        self.upconv4 = ConvBlock(num_ch_dec[4], num_ch_dec[3])  # 256,128,3x3,p1
        self.upconv3 = ConvBlock(num_ch_dec[3], num_ch_dec[2])  # 128,64,3x3,p1
        self.upconv2 = ConvBlock(num_ch_dec[2], num_ch_dec[1])  # 64,32,3x3,p1
        self.upconv1 = ConvBlock(num_ch_dec[1], num_ch_dec[0])  # 32,16,3x3,p1

        # iconv
        self.iconv5 = ConvBlock(num_ch_dec[4], num_ch_dec[4])  # 256,256,3x3,p1
        self.iconv4 = ConvBlock(num_ch_dec[3], num_ch_dec[3])  # 128,128,3x3,p1
        self.iconv3 = ConvBlock(num_ch_dec[2], num_ch_dec[2])  # 64,64,3x3,p1
        self.iconv2 = ConvBlock(num_ch_dec[1], num_ch_dec[1])  # 32,32,3x3,p1
        self.iconv1 = ConvBlock(num_ch_dec[0], num_ch_dec[0])  # 16,16,3x3,p1

        # disp
        self.disp4 = Conv3x3(num_ch_dec[3], num_output_channels)  # 128,3,3x3,p1
        self.disp3 = Conv3x3(num_ch_dec[2], num_output_channels)  # 64,3,3x3,p1
        self.disp2 = Conv3x3(num_ch_dec[1], num_output_channels)  # 32,3,3x3,p1
        self.disp1 = Conv3x3(num_ch_dec[0], num_output_channels)  # 16,3,3x3,p1

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, frame_id=0):
        self.outputs = {}
        _, _, _, _, econv5 = input_features
        # (64,64,128,256,512)*4

        upconv5 = upsample(self.upconv5(econv5))  # 256-->256
        iconv5 = self.iconv5(upconv5)

        upconv4 = upsample(self.upconv4(iconv5))  # 256-->128
        iconv4 = self.iconv4(upconv4)

        upconv3 = upsample(self.upconv3(iconv4))  # 128-->64
        iconv3 = self.iconv3(upconv3)

        upconv2 = upsample(self.upconv2(iconv3))  # 64-->32
        iconv2 = self.iconv2(upconv2)

        upconv1 = upsample(self.upconv1(iconv2))  # 32-->16
        iconv1 = self.iconv1(upconv1)  #

        self.outputs[("res_img", frame_id, 3)] = self.sigmoid(self.disp4(iconv4))  # res_img3
        self.outputs[("res_img", frame_id, 2)] = self.sigmoid(self.disp3(iconv3))  # res_img2
        self.outputs[("res_img", frame_id, 1)] = self.sigmoid(self.disp2(iconv2))  # res_img1
        self.outputs[("res_img", frame_id, 0)] = self.sigmoid(self.disp1(iconv1))  # res_img0
        return self.outputs






















