import os,sys
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
import matplotlib.pyplot as plt
logger = logging.getLogger('DepthEncoder_hrnet_backbone')

__all__ = ['hrnet18', 'hrnet32', 'hrnet48','hrnet64']

model_urls = {
    'hrnet18_imagenet': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    'hrnet32_imagenet': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
    'hrnet48_imagenet': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
    'hrnet48_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:  norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)   #18  18  1
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)   #18  18  1
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,   #256  64
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:  norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups   #64
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)  #256  64
        self.bn1 = norm_layer(width)   #64
        self.conv2 = conv3x3(width, width, stride, groups, dilation)   #64  64  3  1  1  1
        self.bn2 = norm_layer(width)   #64
        self.conv3 = conv1x1(width, planes * self.expansion)   #64  256
        self.bn3 = norm_layer(planes * self.expansion)  #256
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class HighResolutionModule(nn.Module): #(4,BASIC,[4,4,4,4],[18,36,72,144],[18,36,72,144],SUM,True,BN)
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()

        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:  norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)  #(3,BASIC,[4,4,4],[18,36,72],   [18,36,72],    SUM,      True   ,BN)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)


    def _check_branches(self, num_branches, blocks, num_blocks,num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)


    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,stride=1):  #(3,BASIC,[4,4,4],[18,36,72]
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:  #18 = 18
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],num_channels[branch_index] * block.expansion,kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),)

        layers = []
        layers.append(block(self.num_inchannels[branch_index],num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],num_channels[branch_index], norm_layer=self.norm_layer))

        return nn.Sequential(*layers)


    def _make_branches(self, num_branches, block, num_blocks, num_channels):  #(3,BASIC,[4,4,4],[18,36,72],   [18,36,72],    SUM,      True   ,BN)
        branches = []

        for i in range(num_branches):  #0 ,1 ,2
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels)) #(3,BASIC,[4,4,4],[18,36,72]

        return nn.ModuleList(branches)


    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches   #3
        num_inchannels = self.num_inchannels  #【18，36，72】
        fuse_layers = []

        for i in range(num_branches if self.multi_scale_output else 1):   #0,1,2
            fuse_layer = []
            for j in range(num_branches):  #循环      10  11 12  20 21 22
                if j > i:  #12
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],num_inchannels[i],1,1,0,bias=False),self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:  #20
                    conv3x3s = []
                    for k in range(i-j):  #1
                        if k == i - j - 1:   #0=1-1
                            num_outchannels_conv3x3 = num_inchannels[i]  #36
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],num_outchannels_conv3x3,3, 2, 1, bias=False),  #18-->36
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],num_outchannels_conv3x3,3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)




    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):   #0,1,2
            #print(self.branches)

            x[i] = self.branches[i](x[i])     #x0 = self.branches[0](x[0])
            #print(x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):   #0,1,2
            if i == 0 :
                y = x[0]
            else:
                y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches): #1,2
                if i == j:
                    y = y + x[j]
                elif j > i:  #j2>i1
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]),size=[height_output, width_output],mode='bilinear',align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, norm_layer=None):
        super(HighResolutionNet, self).__init__()

        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        # stem network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 256x192x3 ==> 128x96x64
        self.bn1 = self.norm_layer(64)  # 128x96x64 ==> 128x96x64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 128x96x64 ==> 64x48x64
        self.bn2 = self.norm_layer(64)  # 64x48x64 ==> 64x48x64
        self.relu = nn.ReLU(inplace=True)  # relu

        # stage 1  bottleneck（1x1,3x3,1x1）：4
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]  # 64
        block = blocks_dict[self.stage1_cfg['BLOCK']]  # Bottleneck
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]  # 4
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)  # （Bottleneck,64,64,4）
        stage1_out_channel = block.expansion * num_channels  #  64x4 = 256

        # stage 2  basic(3x3,3x3):4,4
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']  # [18, 36]
        block = blocks_dict[self.stage2_cfg['BLOCK']]  # BASIC
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]  # [18, 36]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)  # ([256],[18,36])
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)  # stage2_cfg ,[18,36]

        # stage 3  basic(3x3,3x3):4,4,4
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']  # [18, 36, 72]
        block = blocks_dict[self.stage3_cfg['BLOCK']]  # BASIC
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]  # [18, 36, 72]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)  # ([18,36],[18, 36, 72])
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # stage 4  basic(3x3,3x3):4,4,4,4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']  # [18, 36, 72, 144]
        block = blocks_dict[self.stage4_cfg['BLOCK']]  # BASIC
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]  # [18, 36, 72]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)  # [18, 36, 72],[18, 36, 72,144]
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)


    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):  # ([18,36，72],[18,36,72，144])

        num_branches_cur = len(num_channels_cur_layer)  # 4
        num_branches_pre = len(num_channels_pre_layer)  # 3

        transition_layers = []
        for i in range(num_branches_cur):  # 0 ,1 ,2    #([18 ,36],[18 ,36 ,72])
            if i < num_branches_pre:  # i = 0 ,1
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:  # 18 = 18
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        # 256 --> 18
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:  # i=2
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):  # range(1)
                    inchannels = num_channels_pre_layer[-1]  # 36
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels  # 72
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),  # 36 --> 72
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):  # （Bottleneck,64,64,4）
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:  # stage1(64-->256)
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.norm_layer(planes * block.expansion), )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))  # （Bottleneck,256,64,bn）

        return nn.Sequential(*layers)

    # stage2
    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):  # (stage3_cfg,[18,36,72])
        num_modules = layer_config['NUM_MODULES']  # 1          4
        num_branches = layer_config['NUM_BRANCHES']  # 2          3
        num_blocks = layer_config['NUM_BLOCKS']  # [4,4]      [4,4]
        num_channels = layer_config['NUM_CHANNELS']  # [18,36]    [18,36,72]
        block = blocks_dict[layer_config['BLOCK']]  # BASIC      BASIC
        fuse_method = layer_config['FUSE_METHOD']  # SUM        SUM

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method,
                                     reset_multi_scale_output, norm_layer=self.norm_layer)
            )  # (2,BASIC,[4,4],[18,36],[18,36],SUM,True,BN)
            # (3,BASIC,[4,4,4],[18,36,72],[18,36,72],SUM,True,BN)
            # (4,BASIC,[4,4,4,4],[18,36,72,144],[18,36,72,144],SUM,True,BN)
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    #
    def forward(self, x):
        features = []
        mixed_featurs = []
        list18 = []
        list36 = []
        list72 = []

        # stem network
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        list18.append(x)
        x = self.layer1(x)  # layer1

        # stage2
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):  # 2
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        list18.append(y_list[0])
        list36.append(y_list[1])

        # stage3
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):  # 3
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        list18.append(y_list[0])
        list36.append(y_list[1])
        list72.append(y_list[2])

        # stage4
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):  # 4
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
                    # here generate new scale features (downsample)
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        list18.append(x[0])  # 18
        list36.append(x[1])  # 36
        list72.append(x[2])  # 72



        for text in list18:
            atis_2 = text


        for text in list36:
            atis_3 = text

        for text in list72:
            atis_4 = text

        for text in x[3]:
            atis_5 = text



        mixed_features = [list18] + [list36] + [list72] + [x[3]]  # 4个+3个+2个+1个
        return features + mixed_features


def _hrnet(arch, pretrained, progress, **kwargs):  # hrnet18 ,true ,true
    from .hrnet_config import MODEL_CONFIGS
    model = HighResolutionNet(MODEL_CONFIGS[arch], **kwargs)
    # print('DepthEncoder struct loaded')
    #
    if pretrained:
        if arch == 'hrnet64':
            arch = 'hrnet32_imagenet'
            model_url = model_urls[arch]
            loaded_state_dict = load_state_dict_from_url(model_url, progress=progress)
            # add weights demention to adopt input change
            exp_layers = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                          'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']
            lista = ['transition1.0.0.weight', 'transition1.1.0.0.weight', 'transition2.2.0.0.weight',
                     'transition3.3.0.0.weight']
            for k, v in loaded_state_dict.items():
                if k not in exp_layers:
                    if ('layer' not in k) and 'conv' in k or k in lista and len(v.size()) > 1:
                        if k in ['transition1.0.0.weight', 'transition1.1.0.0.weight']:
                            loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2, 0)
                        else:
                            loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
                            loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2, 0)

                    if 'fuse_layer' in k and 'weight' in k and len(v.size()) > 1:
                        loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
                        loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2, 0)

                    if 'layer' not in k and len(v.size()) == 1:
                        v = v.unsqueeze(1)
                        v = torch.cat([v] * 2, 0)
                        loaded_state_dict[k] = v.squeeze(1)
                    if 'fuse_layer' in k and len(v.size()) == 1:
                        v = v.unsqueeze(1)
                        v = torch.cat([v] * 2, 0)
                        loaded_state_dict[k] = v.squeeze(1)
                    if len(loaded_state_dict[k].size()) == 2:
                        loaded_state_dict[k] = loaded_state_dict[k].squeeze(1)
                    # for multi-input
                    # if k == 'conv1.weight':
                    #  loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
        else:
            arch = arch + '_imagenet'
            model_url = model_urls[arch]
            loaded_state_dict = load_state_dict_from_url(model_url, progress=progress,
                                                         file_name='hrnetv2_w18_imagenet_pretrained.pth')
        # if k == 'conv1.weight':
        #    loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2

        model.load_state_dict({k: v for k, v in loaded_state_dict.items() if k in model.state_dict()})
    # print('DepthEncoder pretrain weights loaded')
    return model






def hrnet18(pretrained=True, progress=True, **kwargs):
    r"""HRNet-18 model
    """
    return _hrnet('hrnet18', pretrained, progress,**kwargs)

def hrnet32(pretrained=True, progress=True, **kwargs):
    r"""HRNet-32 model
    """
    return _hrnet('hrnet32', pretrained, progress,**kwargs)

def hrnet48(pretrained=True, progress=True, **kwargs):
    r"""HRNet-48 model
    """
    return _hrnet('hrnet48', pretrained, progress,**kwargs)

def hrnet64(pretrained=True, progress=True, **kwargs):
    r"""HRNet-64 model
    """
    return _hrnet('hrnet64', pretrained, progress,**kwargs)
















