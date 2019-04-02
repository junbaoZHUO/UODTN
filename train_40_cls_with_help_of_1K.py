import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as util_data
from torch.autograd import Variable

import time
import json
import random

from data_list import ImageList
import network
import loss
import pre_process as prep
import lr_schedule

optim_dict = {"SGD": optim.SGD}




def train_classification(config):
    ## set pre-process
    prep_train  = prep.image_train(resize_size=256, crop_size=224)
               
    ## set loss
    class_criterion = nn.CrossEntropyLoss()

    ## prepare data
    TRAIN_LIST = 'data/WEB_3D3_2.txt'
    BSZ = args.batch_size

    dsets_train1 = ImageList(open(TRAIN_LIST).readlines(), shape = (args.img_size,args.img_size), transform=prep_train, train=False)
    loaders_train1 = util_data.DataLoader(dsets_train1, batch_size=BSZ, shuffle=True, num_workers=6, pin_memory=True)

    begin_num = 127
    class_num = 40
    all_num = 50
    ## set base network
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]]()
    classifier_layer1 = nn.Linear(base_network.output_num(), 24)
    classifier_layer2 = nn.Linear(base_network.output_num(), 16)
    weight_bias=torch.load('GCN/materials/AWA2/151_cls_from_1K')['fc151']
    classifier_layer1.weight.data = weight_bias[127:127+24,:2048]
    classifier_layer1.bias.data = weight_bias[127:127+24,-1]

    ## initialization
    for param in base_network.parameters():
        param.requires_grad = False
    for param in base_network.layer4.parameters():
        param.requires_grad = True
    for param in base_network.layer3.parameters():
        param.requires_grad = True
    for param in classifier_layer1.parameters():
        param.requires_grad = False
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        classifier_layer1 = classifier_layer1.cuda()
        classifier_layer2 = classifier_layer2.cuda()
        base_network = base_network.cuda()

    ## collect parameters
    parameter_list = [{"params":classifier_layer2.parameters(), "lr":10},
                     {"params": base_network.layer3.parameters(), "lr":1},
                     {"params": base_network.layer4.parameters(), "lr":5}]

 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    
    
    len_train_source = len(loaders_train1) - 1
    optimizer.zero_grad()
    for i in range(config["num_iterations"]):
        if (i + 1) % config["test_interval"] == 0:
            if not osp.exists(osp.join('save',args.save_name)):
                os.mkdir(osp.join('save',args.save_name))
            weight_bias[127:127+24,:2048] = classifier_layer1.weight.data 
            weight_bias[127:127+24,-1]    = classifier_layer1.bias.data  
            weight_bias2    = torch.cat((classifier_layer2.weight.data,classifier_layer2.bias.data.unsqueeze(1)),dim=1)  
            torch.save(base_network.state_dict(),osp.join('save',args.save_name,'base_net%d.pkl'%(i+1)))
            torch.save({'fc151+16_ft':torch.cat((weight_bias, weight_bias2.cpu()),dim=0)},'151+16_cls_from_1K_ft')

        classifier_layer1.train(True)
        classifier_layer2.train(True)
        base_network.train(True)
        
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)

        if i % (len_train_source-1) == 0:
            iter_source1 = iter(loaders_train1)

        inputs_source, labels_source = iter_source1.next()

        if use_gpu:
            inputs_source, labels_source = Variable(inputs_source).cuda(), Variable(labels_source).cuda()
        else:
            inputs_source, labels_source = Variable(inputs_source), Variable(labels_source)
           
        features_source = base_network(inputs_source)
        
        outputs_source1 = classifier_layer1(features_source)
        outputs_source2 = classifier_layer2(features_source)
        cls_loss = class_criterion(torch.cat((outputs_source1, outputs_source2), dim=1), labels_source)
        
        total_loss = cls_loss 
        print("Step "+str(i)+": cls_loss: "+str(cls_loss.cpu().data.numpy()))

        total_loss.backward(retain_graph=True)
        if (i+1)% config["opt_num"] ==0:
              optimizer.step()
              optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, nargs='?', default=64, help="batch size")
    parser.add_argument('--img_size', type=int, nargs='?', default=256, help="image size")
    parser.add_argument('--save_name', type=str, nargs='?', default='SOURCE_ONLY', help="loss name")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    config = {}
    config["num_iterations"] = 2000
    config["test_interval"] = 500
    config["save_num"] = 500
    config["opt_num"] = 1
    config["network"] = {"name":"ResNet50"}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, "weight_decay":0.0001, "nesterov":True}, "lr_type":"inv", "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }
    print(config)
    print(args)
    train_classification(config)
