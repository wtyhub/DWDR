import argparse
import math
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import timm

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

class LinearBlock(nn.Module):
    def __init__(self, input_dim, num_bottleneck=512):
        super(LinearBlock, self).__init__()
        self.Linear = nn.Linear(input_dim, num_bottleneck)
        init.kaiming_normal_(self.Linear.weight.data, a=0, mode='fan_out')
        init.constant_(self.Linear.bias.data, 0.0) 

    def forward(self, x):
        x = self.Linear(x)
        return x

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, mid_dim=256):
        super(ClassBlock, self).__init__()
        add_block = []
        self.Linear = nn.Linear(input_dim, num_bottleneck)
        self.bnorm = nn.BatchNorm1d(num_bottleneck)

        init.kaiming_normal_(self.Linear.weight.data, a=0, mode='fan_out')
        init.constant_(self.Linear.bias.data, 0.0) 
        init.normal_(self.bnorm.weight.data, 1.0, 0.02)
        init.constant_(self.bnorm.bias.data, 0.0)


        classifier = []
        if droprate>0:
            classifier += [nn.Dropout(p=droprate)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.Linear(x)
        x = self.bnorm(x)
        x = self.classifier(x)
        return x

class ft_net_VGG16(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', decouple=False):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(512, affine=False)
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
            if self.decouple:
                xf = self.bn(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        if self.decouple:
            return [x, xf]
        else:
            return x

# Define the VGG16-based part Model
class ft_net_VGG16_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=8, row = True, decouple=False):
        super(ft_net_VGG16_LPN, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(512, affine=False)
        self.avg = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,block))
        self.maxpool = nn.AdaptiveMaxPool2d((1,block))
        if row:  # row partition the ground view image
            self.avgpool = nn.AdaptiveAvgPool2d((block,1))
            self.maxpool = nn.AdaptiveMaxPool2d((block,1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.decouple:
            xf = self.avg(x)
            xf = xf.view(xf.size(0), xf.size(1))
            xf = self.bn(xf)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.maxpool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        if self.decouple:
            return [x, xf]
        else:
            return x

# Define vgg16 based square ring partition for satellite images of cvusa/cvact
class ft_net_VGG16_LPN_R(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=4, decouple=False):
        super(ft_net_VGG16_LPN_R, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)
        self.pool = pool
        self.model = model_ft
        self.block = block
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(512, affine=False)
        self.avg = nn.AdaptiveMaxPool2d((1, 1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.decouple:
            xf = self.avg(x)
            xf = xf.view(xf.size(0), xf.size(1))
            xf = self.bn(xf)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        if self.decouple:
            return [x, xf]
        else:
            return x
    # VGGNet's output: 8*8 part:4*4, 6*6, 8*8
    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)

# resnet50 backbone
class ft_net_cvusa_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6, row=True, decouple=False):
        super(ft_net_cvusa_LPN, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(2048, affine=False)
        self.avg = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,block))
        self.maxpool = nn.AdaptiveMaxPool2d((1,block))
        if row:
            self.avgpool = nn.AdaptiveAvgPool2d((block,1))
            self.maxpool = nn.AdaptiveMaxPool2d((block,1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.decouple:
            xf = self.avg(x)
            xf = xf.view(xf.size(0), xf.size(1))
            xf = self.bn(xf)
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.maxpool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        if self.decouple:
            return [x, xf]
        else:
            return x

class ft_net_cvusa_LPN_R(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6, decouple=False):
        super(ft_net_cvusa_LPN_R, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(2048, affine=False)
        self.avg = nn.AdaptiveMaxPool2d((1, 1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.decouple:
            xf = self.avg(x)
            xf = xf.view(xf.size(0), xf.size(1))
            xf = self.bn(xf)
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        if self.decouple:
            return [x, xf]
        else:
            return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.75, stride=2, init_model=None, pool='avg', decouple=False, block=8):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(2048, affine=False)
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate=droprate, num_bottleneck=512)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
        elif pool=='lpn':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.avgpool3 = nn.AdaptiveAvgPool2d((1,block))
            self.model = model_ft
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
            if self.decouple:
                xf = self.bn(x)
                # xf = x
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'lpn':
            if self.decouple:
                xf = self.model.avgpool2(x)
                xf = xf.view(xf.size(0), xf.size(1))
                xf = self.bn(xf)
            x = self.model.avgpool3(x)
            x = x.view(x.size(0), x.size(1), -1)
            if self.decouple:
                return [x, xf]
            else:
                return x
        x = self.classifier(x)
        if self.decouple:
            return [x, xf]
        else:
            return x

# Define the ResNet50-based part Model
class ft_net_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.75, stride=2, init_model=None, pool='avg', block=8, decouple=False):
        super(ft_net_LPN, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        self.pool = pool
        self.model = model_ft   
        self.block = block
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(2048, affine=False)
        self.avg = nn.AdaptiveMaxPool2d((1,1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
        for i in range(self.block):
            clas = 'classifier'+str(i)
            setattr(self, clas, ClassBlock(2048, class_num, droprate=droprate))
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.decouple:
            xf = self.avg(x)
            xf = xf.view(xf.size(0),xf.size(1))
            xf = self.bn(xf)
        # print(x.shape)
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.part_classifier(x)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.part_classifier(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
            x = self.part_classifier(x)

        if self.decouple:
            return [x, xf]
        else:
            return x

    def get_part_pool(self, x, no_overlap=True):
        result = []
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                x_pre = None
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = self.avg_pool(x_curr, x_pre)
                result.append(avgpool)

            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = self.avg_pool(x, x_pre)
                result.append(avgpool)
        return torch.stack(result, dim=2)

    def avg_pool(self, x_curr, x_pre=None):
        h, w = x_curr.size(2), x_curr.size(3)
        if x_pre == None:
            h_pre = w_pre = 0.0
        else:
            h_pre, w_pre = x_pre.size(2), x_pre.size(3)
        pix_num = h*w - h_pre*w_pre
        avg = x_curr.flatten(start_dim=2).sum(dim=2).div_(pix_num)
        return avg

    def part_classifier(self, x):
        
        out_p = []
        for i in range(self.block):
            o_tmp = x[:,:,i].view(x.size(0),-1)
            name = 'classifier'+str(i)
            c = getattr(self, name)
            out_p.append(c(o_tmp))
        
        if not self.training:
            return torch.stack(out_p, dim=2)
        else:
            return out_p

class ft_net_swin(nn.Module):
    def __init__(self, class_num, droprate=0.75, decouple=False):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
        model_ft.head = nn.Sequential()
        self.model = model_ft
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(1024, affine=False)
        self.classifier = ClassBlock(1024, class_num, droprate)
    def forward(self, x):
        x = self.model.forward_features(x)
        if len(x.shape) != 2:
            x = x.mean(dim=1)
        # print(x.shape)
        if self.decouple:
            xf = self.bn(x)
        x = self.classifier(x)
        if self.decouple:
            return [x, xf]
        else:
            return x

class ft_net_swin_base(nn.Module):
    def __init__(self, decouple=False):
        super(ft_net_swin_base, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
        model_ft.head = nn.Sequential()
        self.model = model_ft
        self.decouple = decouple
        self.bn = nn.BatchNorm1d(1024, affine=False)
        # self.classifier = ClassBlock(1024, class_num, droprate)
    def forward(self, x):
        x = self.model.forward_features(x)
        if len(x.shape) != 2:
            x = x.mean(dim=1)
        if self.decouple:
            xf = self.bn(x)
        # x = self.classifier(x)
        if self.decouple:
            return [x, xf]
        else:
            return x

class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 1, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=2, decouple=False, swin=False):
        super(two_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        self.decouple = decouple
        self.pool=pool
        self.sqr = True # if the satellite image is square ring partition and the ground image is row partition, self.sqr is True. Otherwise it is False.
        if VGG16:
            if LPN:
                # satelite
                self.model_1 = ft_net_VGG16_LPN_R(class_num, stride=stride, pool=pool, block=block, decouple=decouple)
            else:
                self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool=pool, decouple=decouple)
        elif swin:
            self.model_1 = ft_net_swin_base(decouple=decouple)
        else:
            #resnet50 LPN cvusa/cvact
            if LPN:
                self.model_1 = ft_net_cvusa_LPN_R(class_num, stride=stride, pool=pool, block=block, decouple=decouple)
                self.block = self.model_1.block
            else:
                self.model_1 = ft_net(class_num, stride=stride, pool=pool, decouple=decouple, block=block)
                self.model_1.classifier = nn.Sequential()
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                if LPN:
                    #street
                    self.model_2 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block = block, row = self.sqr, decouple=decouple)
                else:
                    self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool, decouple=decouple)
            elif swin:
                self.model_2 = ft_net_swin_base(decouple=decouple)
            else:
                if LPN:
                    self.model_2 =  ft_net_cvusa_LPN(class_num, stride = stride, pool = pool, block=block, row = self.sqr, decouple=decouple)
                else:
                    self.model_2 = ft_net(class_num, stride=stride, pool=pool, decouple=decouple, block=block)
                    self.model_2.classifier = nn.Sequential()
        if LPN or self.pool=='lpn':
            if VGG16:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(1024, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(512, class_num, droprate))
            else:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(4096, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(2048, class_num, droprate))
        elif swin:
            self.classifier = ClassBlock(1024, class_num, droprate)
        else:
            self.classifier = ClassBlock(2048, class_num, droprate)
            if pool =='avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)
            if VGG16:
                self.classifier = ClassBlock(512, class_num, droprate)
                # self.classifier = ClassBlock(4096, class_num, droprate, num_bottleneck=512) #safa 情况下
                if pool =='avg+max':
                    self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x1, x2):
        if self.LPN or self.pool=='lpn':
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
                if self.decouple:
                    y1 = self.part_classifier(x1[0])
                else:
                    y1 = self.part_classifier(x1)

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                if self.decouple:
                    y2 = self.part_classifier(x2[0])
                else:
                    y2 = self.part_classifier(x2)
        else:
            if x1 is None:
                y1 = None
            else:
                # x1 = self.vgg1.features(x1)
                x1 = self.model_1(x1)
                if self.decouple:
                    y1 = self.classifier(x1[0])
                else:
                    y1 = self.classifier(x1)

            if x2 is None:
                y2 = None
            else:
                # x2 = self.vgg2.features(x2)
                x2 = self.model_2(x2)
                if self.decouple:
                    y2 = self.classifier(x2[0])
                else:
                    y2 = self.classifier(x2)
        if self.decouple:
            return [y1, x1[1]], [y2, x2[1]]
        return y1, y2

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            # part[i] = torch.squeeze(x[:,:,i])
            part[i] = x[:,:,i].view(x.size(0),-1)
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y



'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    # net = two_view_net(701, droprate=0.5, pool='avg', stride=1, VGG16=True, LPN=True, block=8, decouple=True)
    # net = ft_net_swin_base(701, droprate=0.5, decouple=False)
    # net = three_view_net(701, droprate=0.5, stride=1, share_weight=True, VGG16=False, LPN=True, block=4, decouple=False)
    # net = ft_net_LPN(701,0.75,1,block=4)
    # net.eval()

    # net = ft_net_VGG16_LPN_R(701)
    # net = ft_net_cvusa_LPN(701, stride=1)
    net = ft_net_swin(701, droprate=0.75, decouple=True)
    print(net)

    input = Variable(torch.FloatTensor(2, 3, 224, 224))
    output1 = net(input)[0]
    # output1,output2 = net(input,input)
    # output1,output2,output3 = net(input,input,input)
    # output1 = net(input,decouple=False)
    # print('net output size:')
    print(output1.shape)
    # print(output.shape)
    # for i in range(len(output1)):
    #     print(output1[i].shape)
    # x = torch.randn(2,512,8,8)
    # x_shape = x.shape
    # pool = AzimuthPool2d(x_shape, 8)
    # out = pool(x)
    # print(out.shape)
