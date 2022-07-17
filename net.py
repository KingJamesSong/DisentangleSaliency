#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg      = cfg
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/home/resnet50-19c8e357.pth'), strict=False)

class Attention_unit(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_unit, self).__init__()
        # Three attention units, basic architecture of MDAB
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.spatial_attention3 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.channel_attention1 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.channel_attention2 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.channel_attention3 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

    def forward(self, g, x):
        # spa and cha attention at original scale
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        spa1 = F.relu(g1 + x1, inplace=True)
        cha1 = F.sigmoid(F.avg_pool2d(self.channel_attention1(spa1), spa1.size(2)))
        spa1 = F.sigmoid(self.spatial_attention1(spa1))

        # spa and cha attention at scale pooled by 3x3
        x2 = F.avg_pool2d(x, (3, 3))
        g2 = F.avg_pool2d(g, (3, 3))
        g2 = self.W_g(g2)
        x2 = self.W_x(x2)
        spa2 = F.relu( g2 + x2, inplace=True)
        cha2 = F.sigmoid(F.avg_pool2d(self.channel_attention2(spa2), spa2.size(2)))
        spa2 = F.sigmoid(self.spatial_attention2(spa2))
        spa2 = F.upsample(spa2, size=spa1.size()[2:], mode='bilinear', align_corners=True)
        cha2 = F.upsample(cha2, size=cha1.size()[2:], mode='bilinear', align_corners=True)

        # spa and cha attention at scale pooled by 6x6
        x3 = F.avg_pool2d(x, (6, 6))
        g3 = F.avg_pool2d(g, (6, 6))
        g3 = self.W_g(g3)
        x3 = self.W_x(x3)
        spa3 = F.relu( g3 + x3, inplace=True)
        cha3 = F.sigmoid(F.avg_pool2d(self.channel_attention3(spa3), spa3.size(2)))
        spa3 = F.sigmoid(self.spatial_attention3(spa3))
        spa3 = F.upsample(spa3, size=spa1.size()[2:], mode='bilinear', align_corners=True)
        cha3 = F.upsample(cha3, size=cha1.size()[2:], mode='bilinear', align_corners=True)
        #multi-scale attentive feature
        out = (x * spa1 + x * spa2 + x * spa3 + x * cha1 + x * cha2 + x * cha3)
        return out

    def initialize(self):
        weight_init(self)

class Multi_Attention_unit(nn.Module):
    def __init__(self, F_k, F_g, F_l, F_int):
        super(Multi_Attention_unit, self).__init__()
        # Attention Units for three inputs, basic architecture for MBAB
        self.W_k = nn.Sequential(
            nn.Conv2d(F_k, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.spatial_attention3 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.channel_attention1 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.channel_attention2 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.channel_attention3 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

    def forward(self, k, g, x):
        # spa and cha attention original scale
        k1 = self.W_k(k)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        spa1 = F.relu(k1 + g1 + x1,inplace=True)
        cha1 = F.sigmoid(F.avg_pool2d(self.channel_attention1(spa1), spa1.size(2)))
        spa1 = F.sigmoid(self.spatial_attention1(spa1))
        # spa and cha attention at scale pooled by 3x3
        x2 = F.avg_pool2d(x, (3, 3))
        g2 = F.avg_pool2d(g, (3, 3))
        k2 = F.avg_pool2d(k, (3, 3))
        k2 = self.W_k(k2)
        g2 = self.W_g(g2)
        x2 = self.W_x(x2)
        spa2 = F.relu(k2 + g2 + x2, inplace=True)
        cha2 = F.sigmoid(F.avg_pool2d(self.channel_attention2(spa2), spa2.size(2)))
        spa2 = F.sigmoid(self.spatial_attention2(spa2))
        spa2 = F.upsample(spa2,size=spa1.size()[2:],mode='bilinear',align_corners=True)
        cha2 = F.upsample(cha2, size=cha1.size()[2:], mode='bilinear', align_corners=True)
        # spa and cha attention at scale pooled by 6x6
        x3 = F.avg_pool2d(x, (6, 6))
        g3 = F.avg_pool2d(g, (6, 6))
        k3 = F.avg_pool2d(k, (6, 6))
        k3 = self.W_k(k3)
        g3 = self.W_g(g3)
        x3 = self.W_x(x3)
        spa3 = F.relu(k3 + g3 + x3, inplace=True)
        cha3 = F.sigmoid(F.avg_pool2d(self.channel_attention3(spa3), spa3.size(2)))
        spa3 = F.sigmoid(self.spatial_attention3(spa3))
        spa3 = F.upsample(spa3, size=spa1.size()[2:], mode='bilinear', align_corners=True)
        cha3 = F.upsample(cha3, size=cha1.size()[2:], mode='bilinear', align_corners=True)
        #Multi-scale attentive feature or detail flow
        out = (x * spa1 + x * spa2 + x * spa3 + x * cha1 + x * cha2 + x * cha3)
        return out

    def initialize(self):
        weight_init(self)

class Decoder_detail(nn.Module):
    def __init__(self):
        super(Decoder_detail, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        self.att1 = Attention_unit(64, 64, 64)
        self.att2 = Attention_unit(64, 64, 64)
        self.att3 = Attention_unit(64, 64, 64)

    def forward(self, input1):

        # MDAB 1
        out0 = F.relu(self.bn0(self.conv0(input1[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        input11 = self.att1(out0,input1[1])
        out1 = F.relu(self.bn1(self.conv1(input11+out0)), inplace=True)
        # MDAB 2
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        input12 = self.att2(out1,input1[2])
        out2 = F.relu(self.bn2(self.conv2(input12+out1)), inplace=True)
        # MDAB 3
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        input13 = self.att3(out2,input1[3])
        out3 = F.relu(self.bn3(self.conv3(input13+out2)), inplace=True)
        return out3
    
    def initialize(self):
        weight_init(self)

class Decoder_body(nn.Module):
    def __init__(self):
        super(Decoder_body, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.att20 = Attention_unit(64, 64, 64)

        self.att21 = Multi_Attention_unit(64, 64, 64, 64)
        self.att11 = Multi_Attention_unit(64, 64, 64, 64)

        self.att22 = Multi_Attention_unit(64, 64, 64, 64)
        self.att12 = Multi_Attention_unit(64, 64, 64, 64)

        self.att23 = Multi_Attention_unit(64, 64, 64, 64)
        self.att13 = Multi_Attention_unit(64, 64, 64, 64)

    def forward(self, input1, input2):
        # MBAB 1
        input20=self.att20(input1[0], input2[0])
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input20)), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        input21 = self.att21(out0, input1[1], input2[1])
        input11 = self.att11(out0, input2[1], input1[1])
        out1 = F.relu(self.bn1(self.conv1(input11 + input21 + out0)), inplace=True)
        #MBAB 2
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        input22 = self.att22(out1, input1[2], input2[2])
        input12 = self.att12(out1, input2[2], input1[2])
        out2 = F.relu(self.bn2(self.conv2(input12 + input22 + out1)), inplace=True)
        #MBAB 3
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        input23 = self.att23(out2, input1[3], input2[3])
        input13 = self.att13(out2, input2[3], input1[3])
        out3 = F.relu(self.bn3(self.conv3(input13 + input23 + out2)), inplace=True)
        return out3

    def initialize(self):
        weight_init(self)


class Encoder_detail(nn.Module):
    def __init__(self):
        super(Encoder_detail, self).__init__()

        self.conv_img = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn_img = nn.BatchNorm2d(64)

        self.conv_detail = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn_detail = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b   = nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b   = nn.BatchNorm2d(64)

        self.att_detail = Multi_Attention_unit(64, 64, 64, 64)
        self.att_img = Multi_Attention_unit(64, 64, 64, 64)

    def forward(self, out1, image, out_detail):
        image = F.relu(self.bn_img(self.conv_img(image)), inplace=True)
        out_detail = F.relu(self.bn_detail(self.conv_detail(out_detail)),inplace=True)

        image = F.interpolate(image,size=out1.size()[2:], mode='bilinear')
        out_detail = F.interpolate(out_detail, size=out1.size()[2:], mode='bilinear')

        image = self.att_img(out1,out_detail,image)
        out_detail = self.att_detail(out1,image,out_detail)
        # detail flow, detail image, and image feature fusion
        out1 = F.relu(self.bn1(self.conv1(out1+image+out_detail)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)

        out1b = F.relu(self.bn1b(self.conv1b(out1)), inplace=True)
        out2b = F.relu(self.bn2b(self.conv2b(out2)), inplace=True)
        out3b = F.relu(self.bn3b(self.conv3b(out3)), inplace=True)
        out4b = F.relu(self.bn4b(self.conv4b(out4)), inplace=True)

        return (out4b, out3b, out2b, out1b)

    def initialize(self):
        weight_init(self)

class SaliencyDisentangle(nn.Module):
    def __init__(self, cfg):
        super(SaliencyDisentangle, self).__init__()
        # Our Saliency Disentangle Framework
        self.cfg      = cfg
        self.bkbone   = ResNet(cfg)
        self.conv5b   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.conv5d   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.encoder  = Encoder_detail()
        self.decoderb = Decoder_body()
        self.decoderd = Decoder_detail()
        self.linearb  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.initialize()

    def forward(self, x, shape=None):
        out1, out2, out3, out4, out5 = self.bkbone(x)
        #Two sets of features passed from backbone network
        out2b, out3b, out4b, out5b   = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d   = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        if shape is None:
            shape = x.size()[2:]
        # Detail decoder
        out_detail = self.decoderd([out5d, out4d, out3d, out2d])
        out_detail_img = F.interpolate(self.lineard(out_detail), size=shape, mode='bilinear')
        # Detail Encoder
        out_label = self.encoder(out_detail,x,out_detail_img)
        # Body Decoder
        out_label = self.decoderb(out_label, [out5b, out4b, out3b, out2b])
        # Saliency Map fusion
        out_label_img = F.interpolate(self.linearb(out_label), size=shape, mode='bilinear')+out_detail_img
        return out_detail_img, out_label_img

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)