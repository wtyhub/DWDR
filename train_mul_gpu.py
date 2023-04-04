# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
from torch.utils.tensorboard import SummaryWriter   
# import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import copy
import time
import os
from model import ft_net_LPN, ft_net, ft_net_swin
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy
import yaml
import math
from shutil import copyfile
from utils import update_average, get_model_list, load_network, save_network, make_weights_for_balanced_classes
import numpy as np
from image_folder import SatData, DroneData, ImageFolder_selectID, ImageFolder_expandID

version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='debug', type=str, help='output model name')
parser.add_argument('--pool',default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir',default='/home/wangtyu/datasets/University-Release/train',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--stride', default=1, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.75, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
parser.add_argument('--resume', action='store_true', help='use resume trainning' )
parser.add_argument('--share', action='store_true', help='share weight between different view' )
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google' )
parser.add_argument('--LPN', action='store_true', help='use LPN' )
parser.add_argument('--decouple', action='store_true', help='use decouple' )
parser.add_argument('--block', default=4, type=int, help='the num of block' )
parser.add_argument('--scale', default=1/32, type=float, metavar='S', help='scale the loss')
parser.add_argument('--lambd', default=3.9e-3, type=float, metavar='L', help='weight on off-diagonal terms')
parser.add_argument('--g', default=0.9, type=float, metavar='L', help='weight on loss and deloss')
parser.add_argument('--t', default=4.0, type=float, metavar='L', help='temperature of conv matrix')
parser.add_argument('--experiment_name',default='debug',type=str, help='log dir name')
parser.add_argument('--adam', action='store_true', help='using adam optimization' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--balance', action='store_true', help='using balance sampler' )
parser.add_argument('--select_id', action='store_true', help='select id' )
parser.add_argument('--multi_image', action='store_true', help='only inputs3 + inputs3_s training' )
parser.add_argument('--expand_id', action='store_true', help='expand id' )
parser.add_argument('--dro_lead', action='store_true', help='drone leading sampling' )
parser.add_argument('--sat_lead', action='store_true', help='satellite leading sampling' )
parser.add_argument('--normal', action='store_true', help='normal training' )
parser.add_argument('--only_decouple', action='store_true', help='do not use balance losss' )
parser.add_argument('--e1', default=1, type=int, help='the exponent of on diag' )
parser.add_argument('--e2', default=1, type=int, help='the exponent of off diag' )
parser.add_argument('--swin', action='store_true', help='using swin as backbone' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
opt = parser.parse_args()

def seed_torch(seed=opt.seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	# torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
if opt.seed > 0:
    print('random seed---------------------:', opt.seed)
    seed_torch(opt.seed)

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

# debug
# opt.LPN=True
# opt.decouple = True




fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>1:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids 
    cudnn.enabled = True
    cudnn.benchmark = True
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpu_ids))
    cudnn.benchmark = True
print('---------------Pool Strategy------------:', opt.pool)
######################################################################
# Load Data
# ---------
#

transform_train_list = [
        #transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad( opt.pad, padding_mode='edge'),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_satellite_list = [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad( opt.pad, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
    'satellite': transforms.Compose(transform_satellite_list)
    }


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
if opt.expand_id:
    print('--------------------expand id-----------------------')
    image_datasets['satellite'] = ImageFolder_expandID(os.path.join(data_dir, 'satellite'), transform=data_transforms['satellite'])
else:
    image_datasets['satellite'] = SatData(data_dir, data_transforms['satellite'], data_transforms['train'])

if opt.select_id:
    print('--------------------select id-----------------------')
    image_datasets['drone'] = ImageFolder_selectID(os.path.join(data_dir, 'drone'), transform=data_transforms['train'])
else:
    image_datasets['drone'] = DroneData(data_dir, data_transforms['train'], data_transforms['satellite'])

def _init_fn(worker_id):
    np.random.seed(int(opt.seed)+worker_id)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                            shuffle=True, num_workers=8, pin_memory=False, worker_init_fn=_init_fn) # 8 workers may work faster
            for x in ['satellite', 'drone']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}
class_names = image_datasets['satellite'].classes
print(dataset_sizes)
use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

# work channel loss
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def decouple_loss(y1, y2, scale_loss, lambd):
    batch_size = y1.size(0)
    c = y1.T @ y2
    c.div_(batch_size)
    on_diag = torch.diagonal(c)
    p_on = (1 - on_diag) / 2
    on_diag = torch.pow(p_on, opt.e1) * torch.pow(torch.diagonal(c).add_(-1), 2)
    on_diag = on_diag.sum().mul(scale_loss)

    off_diag = off_diagonal(c)
    p_off = torch.abs(off_diag)
    off_diag = torch.pow(p_off, opt.e2) * torch.pow(off_diagonal(c), 2)
    off_diag = off_diag.sum().mul(scale_loss)
    loss = on_diag + off_diag * lambd
    return loss, on_diag, off_diag * lambd


def one_LPN_output(outputs, labels, criterion, block):
    # part = {}
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)

    _, preds = torch.max(score.data, 1)

    return preds, loss 

def train_model(model, model_test, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite']/opt.batchsize)*opt.warm_epoch # first 5 epoch

    for epoch in range(num_epochs-start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_corrects3 = 0.0
            ins_loss = 0.0
            dec_loss = 0.0
            on_loss = 0.0
            off_loss = 0.0
            # Iterate over data.
            for data,data3 in zip(dataloaders['satellite'], dataloaders['drone']) :
                # get the inputs
                inputs, inputs_d, labels = data
                inputs3, inputs3_s, labels3 = data3
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                if use_gpu:
                    if opt.normal:
                        inputs = Variable(inputs.cuda().detach())
                        inputs3 = Variable(inputs3.cuda().detach())
                        labels = Variable(labels.cuda().detach())
                        labels3 = Variable(labels3.cuda().detach())
                    else:
                        inputs = Variable(inputs.cuda().detach())
                        inputs_d = Variable(inputs_d.cuda().detach())
                        inputs3 = Variable(inputs3.cuda().detach())
                        inputs3_s = Variable(inputs3_s.cuda().detach())
                        labels = Variable(labels.cuda().detach())
                        labels3 = Variable(labels3.cuda().detach())
                    
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
 
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if opt.decouple:
                    outs_c, outs_f = model(inputs)
                else:
                    outs_c = model(inputs)
                if opt.balance:
                    if opt.decouple:
                        outs_d_c, outs_d_f = model(inputs_d)
                    else:
                        outs_d_c = model(inputs_d)

                if opt.decouple:
                    outd_c, outs3_f = model(inputs3)
                else:
                    outd_c = model(inputs3)
                if opt.balance:
                    if opt.decouple:
                        outs3_s_c, outs3_s_f = model(inputs3_s)
                    else:
                        outs3_s_c = model(inputs3_s)
                # calculate loss
                if opt.LPN:
                    if opt.balance:
                        # print('--------------------- using data balance---------------------------')
                        if opt.only_decouple:
                            # print('--------------------- only decouple---------------------------')
                            preds, loss = one_LPN_output(outs_c, labels, criterion, opt.block)
                            preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, opt.block)
                            loss = loss + loss3
                        else:
                            preds, loss = one_LPN_output(outs_c, labels, criterion, opt.block)
                            _, loss_d = one_LPN_output(outs_d_c, labels, criterion, opt.block)
                            loss = loss + loss_d
                            preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, opt.block)
                            _, loss3_s = one_LPN_output(outs3_s_c, labels3, criterion, opt.block)
                            loss3 = loss3 + loss3_s
                            loss = (loss + loss3) / 2
                    else:
                        preds, loss = one_LPN_output(outs_c, labels, criterion, opt.block)
                        preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, opt.block)
                        loss = loss + loss3
                    if opt.decouple:
                        if opt.balance:
                            deloss1, on, off = decouple_loss(outs_f, outs_d_f, opt.scale, opt.lambd)
                            deloss2, on1, off1 = decouple_loss(outs3_s_f, outs3_f, opt.scale, opt.lambd)
                            deloss = (deloss1 + deloss2) / 2
                            # deloss = deloss2
                            on = (on + on1) / 2
                            off = (off + off1) / 2
                            insloss = loss
                            loss = opt.g*insloss + (1-opt.g)*deloss
                else:
                    _, preds = torch.max(outs_c.data, 1)
                    _, preds3 = torch.max(outd_c.data, 1)
                    if opt.balance:
                        # print('--------------------- using data balance---------------------------')
                        if opt.only_decouple:
                            loss = criterion(outs_c, labels)
                            loss3 = criterion(outd_c, labels3)
                            loss = loss + loss3
                        elif opt.multi_image:
                            # print('-------multi image----------')
                            loss3 = criterion(outd_c, labels3)
                            loss3_s = criterion(outs3_s_c, labels3)
                            loss = loss3 + loss3_s
                        elif opt.dro_lead: #batch is 16
                            # print('drone-view leading sampling')
                            loss3 = criterion(outd_c, labels3)
                            loss3_s = criterion(outs3_s_c, labels3)
                            loss = (loss3 + loss3_s)
                        elif opt.sat_lead: #batch is 16
                            loss = criterion(outs_c, labels)
                            loss_d = criterion(outs_d_c, labels)
                            loss = (loss + loss_d) 
                        else: # batch is 8  
                            loss = criterion(outs_c, labels)
                            loss_d = criterion(outs_d_c, labels)
                            loss = loss + loss_d
                            loss3 = criterion(outd_c, labels3)
                            loss3_s = criterion(outs3_s_c, labels3)
                            loss3 = loss3 + loss3_s
                            loss = (loss + loss3) / 2
                    else:
                        loss = criterion(outs_c, labels)
                        loss3 = criterion(outd_c, labels3)
                        if opt.normal:
                            loss = (loss + loss3) 
                        else:
                            loss = loss + loss3
                    if opt.decouple:
                        if opt.balance:
                            if opt.dro_lead:
                                deloss, on, off = decouple_loss(outs3_s_f, outs3_f, opt.scale, opt.lambd)
                                insloss = loss
                                loss = opt.g*insloss + (1-opt.g)*deloss
                                # loss = deloss
                            elif opt.sat_lead:
                                deloss, on, off = decouple_loss(outs_f, outs_d_f, opt.scale, opt.lambd)
                                insloss = loss
                                loss = opt.g*insloss + (1-opt.g)*deloss
                                # loss = deloss
                            else:
                                # outs_f = torch.cat([outs_f, outs3_s_f], dim=0)
                                # outs3_f = torch.cat([outs_d_f, outs3_f], dim=0)
                                # deloss, on, off = decouple_loss(outs_f, outs3_f, opt.scale, opt.lambd)
                                deloss1, on, off = decouple_loss(outs_f, outs_d_f, opt.scale, opt.lambd)
                                deloss2, on1, off1 = decouple_loss(outs3_s_f, outs3_f, opt.scale, opt.lambd)
                                deloss = (deloss1 + deloss2) / 2
                                # deloss = deloss2
                                on = (on + on1) / 2
                                off = (off + off1) / 2
                                insloss = loss
                                loss = opt.g*insloss + (1-opt.g)*deloss
                                # loss = deloss
                            # loss = insloss + opt.g*deloss
                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if fp16: # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    ##########
                    if opt.moving_avg<1.0:
                        update_average(model_test, model, opt.moving_avg)

                # statistics
                running_loss += loss.item() * now_batch_size
                if opt.decouple:
                    ins_loss += insloss.item() * now_batch_size
                    dec_loss += deloss.item() * now_batch_size
                    on_loss += on.item() * now_batch_size
                    off_loss += off.item() *now_batch_size

                running_corrects += float(torch.sum(preds == labels.data))                
                running_corrects3 += float(torch.sum(preds3 == labels3.data))

            epoch_loss = running_loss / dataset_sizes['satellite']
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc3 = running_corrects3 / dataset_sizes['satellite']

            if opt.decouple:
                epoch_ins_loss = ins_loss / dataset_sizes['satellite']
                epoch_dec_loss = dec_loss / dataset_sizes['satellite']
                epoch_on_loss = on_loss / dataset_sizes['satellite']
                epoch_off_loss = off_loss / dataset_sizes['satellite']
            
            if opt.decouple:
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f}, On_Loss: {:.4f}, Off_Loss: {:.4f},'.format(phase, epoch_loss, epoch_acc, epoch_acc3, epoch_on_loss, epoch_off_loss))
            else:   
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_acc3))
 
        
            writer.add_scalar('Train loss', epoch_loss, epoch+1)
            writer.add_scalar('Learning rate', optimizer.param_groups[1]['lr'], epoch+1)
            writer.add_scalar('Satellite Acc', epoch_acc, epoch+1)           
            writer.add_scalar('Drone Acc', epoch_acc3, epoch+1)
            if opt.decouple:
                writer.add_scalar('instance loss', epoch_ins_loss, epoch+1)
                writer.add_scalar('decouple loss', epoch_dec_loss, epoch+1)
                writer.add_scalar('on loss', epoch_on_loss, epoch+1)
                writer.add_scalar('off loss', epoch_off_loss, epoch+1)



            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            
            # saving last model:
            if phase == 'train':
                scheduler.step()
            if epoch+1 == num_epochs and len(gpu_ids)>1:
                save_network(model.module, opt.name, epoch)
            elif epoch+1 > 100 and (epoch+1) % 10 == 0:
                save_network(model, opt.name, epoch)
            #draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     best_epoch = epoch
        #     last_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(last_model_wts)
    # if len(gpu_ids)>1:
    #     save_network(model.module, opt.name, 'last')
    #     print('best_epoch:', best_epoch)
    # else:
    #     save_network(model, opt.name, 'last')
    #     print('best_epoch:', best_epoch)

    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
if opt.LPN:
    model = ft_net_LPN(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, block=opt.block, decouple=opt.decouple)
elif opt.swin:
    model = ft_net_swin(len(class_names), droprate=opt.droprate, decouple=opt.decouple)
else:
    model = ft_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, decouple=opt.decouple)

opt.nclasses = len(class_names)
print('nclass--------------------:', opt.nclasses)
print(model)
# For resume:
if start_epoch>=40:
    opt.lr = opt.lr*0.1
if not opt.LPN:
    model = model.cuda()
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': model.classifier.parameters(), 'lr': opt.lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    # ignored_params = list(map(id, model.model.fc.parameters() ))
    if len(gpu_ids)>1:
        model = torch.nn.DataParallel(model).cuda()
        ignored_params = list()
        for i in range(opt.block):
            cls_name = 'classifier'+str(i)
            c = getattr(model.module, cls_name)
            ignored_params += list(map(id, c.parameters() ))
        
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1*opt.lr}]
        for i in range(opt.block):
            cls_name = 'classifier'+str(i)
            c = getattr(model.module, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': opt.lr})

    else:
        model = model.cuda()
        print('---------------------use one gpu-----------------------')
        ignored_params =list()
        # ignored_params += list(map(id, model.rdim.parameters() ))
        for i in range(opt.block):
            cls_name = 'classifier'+str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters() ))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1*opt.lr}]
        # optim_params.append({'params': model.rdim.parameters(), 'lr': opt.lr})
        for i in range(opt.block):
            cls_name = 'classifier'+str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': opt.lr})

    optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)
    if opt.adam:
        optimizer_ft = optim.Adam(optim_params, opt.lr, weight_decay=5e-4)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
# exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[60,120,160], gamma=0.1)
# exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=120, eta_min=0.001)
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
# neptune.init('wtyu/decouple')
# neptune.create_experiment('LPN+norm(batch*512*4)')

log_dir = './log/'+ opt.experiment_name
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
writer = SummaryWriter(log_dir)
dir_name = os.path.join('./model',name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
#record every run
    copyfile('./run_mul_gpu_view.sh', dir_name+'/run_mul_gpu_view.sh')
    copyfile('./train_mul_gpu.py', dir_name+'/train_mul_gpu.py')
    copyfile('./model.py', dir_name+'/model.py')
# save opts
    with open('%s/opts.yaml'%dir_name,'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")

# if len(gpu_ids)>1:
#     model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
# else:
#     model = model.cuda()

criterion = nn.CrossEntropyLoss()
if opt.moving_avg<1.0:
    model_test = copy.deepcopy(model)
    num_epochs = 140
else:
    model_test = None
    num_epochs = 120

model = train_model(model, model_test, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
# neptune.stop()
writer.close()