from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
# from hr_layers import *
from layers import upsample
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self,  num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super().__init__()
        self.num_output_channels = num_output_channels  # 1
        self.num_ch_enc = num_ch_enc  # [ 64, 18, 36, 72, 144 ]
        self.scales = scales  # [ 0, 1, 2, 3 ]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()



        #
        self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])    # [32,16]
        self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])    # [16,16]

        self.convs["72"] = Attention_Module(self.num_ch_enc[4], self.num_ch_enc[3] * 2, 256)  # [144,144,256]  ！！
        self.convs["36"] = Attention_Module(256, self.num_ch_enc[2] * 3, 128)  # [256,108,128]  ！！
        self.convs["18"] = Attention_Module(128, self.num_ch_enc[1] * 3 + 64, 64)  # [128,118,64]  ！！
        self.convs["9"] = Attention_Module(64, 64, 32)  # [64,64,32]


        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i],self.num_output_channels)  # dispConvScale0=[16 ,1]
                                                                                                            # dispConvScale1=[32 ,1]
                                                                                                            # dispConvScale2=[64 ,1]
                                                                                                            # dispConvScale3=[128 ,1]
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    #输入为depthfeature
    def forward(self, input_features):
        outputs = {}
        feature144 = input_features[4]  #[8x6x144]
        feature72 = input_features[3]  ##[16x12x72]
        feature36 = input_features[2]  ##[32x24x36]
        feature18 = input_features[1]  ##[64x48x18]
        feature64 = input_features[0]  ##[64x48x64]


        x72 = self.convs["72"](feature144, feature72)  # attention
        x36 = self.convs["36"](x72, feature36)  # attention
        x18 = self.convs["18"](x36, feature18)  # attention
        x9 = self.convs["9"](x18, [feature64])  # attention
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)))  # 32-->16-->16

        outputs[("disp", 0)] = self.sigmoid(self.convs["dispConvScale0"](x6))  # disp0
        outputs[("disp", 1)] = self.sigmoid(self.convs["dispConvScale1"](x9))  # disp1
        outputs[("disp", 2)] = self.sigmoid(self.convs["dispConvScale2"](x18))  # disp2
        outputs[("disp", 3)] = self.sigmoid(self.convs["dispConvScale3"](x36))  # disp3
        return outputs
