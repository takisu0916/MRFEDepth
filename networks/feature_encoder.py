from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo



class FeatureEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self,num_layers,pretrained=True):
        super().__init__()
        self.num_ch_enc = np.array([64,64,128,256,512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101
                   }


        if num_layers not in resnets:  raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if pretrained:
            loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
            loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] , 1)
            self.encoder.load_state_dict(loaded)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self,input_image):


        self.features = []

        self.features.append(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(input_image))))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))


        return self.features














