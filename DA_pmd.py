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
from gcn.gcn import GCN

optim_dict = {"SGD": optim.SGD}


def image_classification_test(iter_test,len_now, base, class1,bottelneck, gpu=True):
    start_test = True
    COR = 0.
    Total = 0.
    print('Testing ...')
    for i in range(len_now):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        output = base(inputs)
        outputs = class1(output)
        if start_test:
            all_output = outputs.data.float()
            all_label = labels.data.float()
            _, predict = torch.max(all_output, 1)
            COR = COR + torch.sum(torch.squeeze(predict).float() == all_label)
            Total = Total + all_label.size()[0]
    accuracy = float(COR)/float(Total)
    return accuracy

def train_classification(config):
    ## set pre-process
    prep_train  = prep.image_train(resize_size=256, crop_size=224)
    prep_test = prep.image_test(resize_size=256, crop_size=224)
               
    ## set loss
    class_criterion = nn.CrossEntropyLoss()
    transfer_criterion = loss.loss_dict["LP"]

    ## prepare data
    TEST_LIST = 'data/new_AwA2_common.txt'#AWA_T.txt'#'data/WEB_72.txt'
    TRAIN_LIST = 'data/I2AWA2_40.txt'#'AWA_SS.txt#'data/new_AwA2_common.txt'
    BSZ = args.batch_size

    dsets_train1 = ImageList(open(TRAIN_LIST).readlines(), shape = (args.img_size,args.img_size), transform=prep_train, train=True)
    loaders_train1 = util_data.DataLoader(dsets_train1, batch_size=BSZ, shuffle=True, num_workers=8, pin_memory=True)

    dsets_test = ImageList(open(TEST_LIST).readlines(), shape = (args.img_size,args.img_size),transform=prep_test, train=False)
    loaders_test = util_data.DataLoader(dsets_test, batch_size=BSZ, shuffle=True, num_workers=4, pin_memory=True)
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]]()
    classifier_layer1 = nn.Linear(base_network.output_num(), class_num)
    ## initialization
    for param in base_network.parameters():
        param.requires_grad = False
    for param in base_network.layer4.parameters():
        param.requires_grad = True
    for param in base_network.layer3.parameters():
        param.requires_grad = True
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        classifier_layer1 = classifier_layer1.cuda()
        base_network = base_network.cuda()

    ## collect parameters
    parameter_list = [{"params":classifier_layer1.parameters(), "lr":10},
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
    len_test_source = len(loaders_test) - 1
    optimizer.zero_grad()
    for i in range(config["num_iterations"]):
        if (i + 1) % config["test_interval"] == 0:
            base_network.train(False)
            classifier_layer1.train(False)
            print(str(i)+' ACC:')
            iter_target = iter(loaders_test)
            print(image_classification_test(iter_target,len_test_source, base_network, classifier_layer1, bottelneck, gpu=use_gpu))
            iter_target = iter(loaders_test)
            if not osp.exists(osp.join('model',args.save_name)):
                os.mkdir(osp.join('model',args.save_name))
            torch.save(base_network.state_dict(),osp.join('model',args.save_name,'base_net%d.pkl'%(i+1)))
            torch.save(classifier_layer1.state_dict(),osp.join('model',args.save_name,'class%d.pkl'%(i+1)))

        classifier_layer1.train(True)
        base_network.train(True)
        
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)

        if i % (len_train_source-1) == 0:
            iter_source = iter(loaders_train1)
        if i % (len_test_source ) == 0:
            iter_target = iter(loaders_test)

        inputs_source, labels_source, labels_source_father, inputs_target = iter_source.next()

        if use_gpu:
            inputs_source, labels_source, inputs_target = Variable(inputs_source).cuda(), Variable(labels_source).cuda(), Variable(inputs_target).cuda()
        else:
            inputs_source, labels_source, inputs_target = Variable(inputs_source), Variable(labels_source),Variable(inputs_target)
           
        features_source = base_network(inputs_source)
        features_target = base_network(inputs_target)
        
        outputs_source1 = classifier_layer1(features_source)
        outputs_target1 = classifier_layer1(features_target)
        

        cls_loss = class_criterion(outputs_source1, labels_source)
        
        transfer_loss = transfer_criterion(features_source, features_target)
        


        total_loss = cls_loss + transfer_loss * args.w_align
        print("Step "+str(i)+": cls_loss: "+str(cls_loss.cpu().data.numpy())+
                             " transfer_loss: "+str(transfer_loss.cpu().data.numpy()))

        total_loss.backward(retain_graph=True)
        if (i+1)% config["opt_num"] ==0:
              optimizer.step()
              optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, nargs='?', default=32, help="batch size")
    parser.add_argument('--img_size', type=int, nargs='?', default=256, help="image size")
    parser.add_argument('--save_name', type=str, nargs='?', default='base', help="loss name")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    config = {}
    config["num_iterations"] = 3000
    config["test_interval"] = 200
    config["save_num"] = 200
    config["opt_num"] = 1
    config["network"] = {"name":"ResNet50"}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, "weight_decay":0.0001, "nesterov":True}, "lr_type":"inv", "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }
    print(config)
    print(args)
    train_classification(config)
