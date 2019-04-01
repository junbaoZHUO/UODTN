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


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def gfl_hook(coeff):
    def fun1(grad):
        return coeff*grad.clone()
    return fun1

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def calc_coeff_(iter_num):
    return np.float(np.power(1 + 0.1 * iter_num, - 0.5))

class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -1)

def grad_reverse(x):
    return GradReverse()(x)

# def grad_reverse(grad):
#     return grad.clone() * -1







def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)















def my_l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)

def image_classification_test(iter_test,len_now, base, class1,bottelneck, gpu=True):
    start_test = True
    #iter_test = iter(loader)
    Bd = 29410
    Total_1k = 0.
    Total_4k = 0.
    COR_1k = 0.
    COR_4k = 0.
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
        # output = bottelneck(output)
        outputs = class1(output)
        if start_test:
            all_output = outputs.data.float()
            #all_output[:,:40] = all_output[:,:40]#-100000
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
    transfer_criterion = loss.loss_dict["MMD"]
    # transfer_criterion = loss.loss_dict["L1N"]
    # transfer_criterion = loss.loss_dict["LP"]

    ## prepare data
    TEST_LIST = 'data/new_AwA2_common.txt'#AWA_T.txt'#'data/WEB_72.txt'
    # TRAIN_LIST = 'data/new_IMG_WEB_AwA2_common.txt'
    TRAIN_LIST = 'data/WEB_3D3_2.txt'#'AWA_SS.txt#'data/new_AwA2_common.txt'
    BSZ = args.batch_size

    # dsets_train = ImageList(open(TRAIN_LIST).readlines(), shape = (args.img_size,args.img_size), transform=prep_train)
    # loaders_train = util_data.DataLoader(dsets_train, batch_size=BSZ, shuffle=True, num_workers=8, pin_memory=True)
    dsets_train1 = ImageList(open(TRAIN_LIST).readlines(), shape = (args.img_size,args.img_size), transform=prep_train, train=False)
    loaders_train1 = util_data.DataLoader(dsets_train1, batch_size=BSZ, shuffle=True, num_workers=6, pin_memory=True)
    dsets_train2 = ImageList(open(TEST_LIST).readlines(), shape = (args.img_size,args.img_size), transform=prep_train, train=False)
    loaders_train2 = util_data.DataLoader(dsets_train2, batch_size=BSZ, shuffle=True, num_workers=6, pin_memory=True)

    dsets_test = ImageList(open(TEST_LIST).readlines(), shape = (args.img_size,args.img_size),transform=prep_test, train=False)
    loaders_test = util_data.DataLoader(dsets_test, batch_size=BSZ, shuffle=True, num_workers=4, pin_memory=True)
    begin_num = 127
    class_num = 40
    all_num = 50
    ## set base network
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]]()
    bottelneck = nn.Linear(base_network.output_num(), 256)
    # classifier_layer1 = nn.Linear(256, class_num)
    classifier_layer1 = nn.Linear(base_network.output_num(), class_num)
    ad_net = AdversarialNetwork(2048, 1024)
    ## initialization
    for param in base_network.parameters():
        param.requires_grad = False
    for param in base_network.layer4.parameters():
        param.requires_grad = True
    for param in base_network.layer3.parameters():
        param.requires_grad = True
    # for param in base_network.layer2.parameters():
    #     param.requires_grad = True
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        classifier_layer1 = classifier_layer1.cuda()
        base_network = base_network.cuda()
        bottelneck = bottelneck.cuda()
        ad_net = ad_net.cuda()
    if (False):
        base_network.layer1 = torch.nn.DataParallel(base_network.layer1)
        base_network.layer2 = torch.nn.DataParallel(base_network.layer2)
        base_network.layer3 = torch.nn.DataParallel(base_network.layer3)
        base_network.layer4 = torch.nn.DataParallel(base_network.layer4)
        classifier_layer1 = torch.nn.DataParallel(classifier_layer1)

    ## collect parameters
    parameter_list = [{"params":classifier_layer1.parameters(), "lr":10},
                     {"params":ad_net.parameters(), "lr":10},
                     # {"params": bottelneck.parameters(), "lr":10},
                     # {"params": filter(lambda p: p.requires_grad, base_network)}]
                     # {"params": base_network.layer2.parameters(), "lr":1},
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
            # base_network.layer3.train(False)
            # base_network.layer4.train(False)
            classifier_layer1.train(False)
            bottelneck.train(False)
            # ad_net.train(False)
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
        bottelneck.train(True)
        # base_network.layer4.train(True)
        # base_network.layer3.train(True)
        # ad_net.train(True)
        
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        #optimizer.zero_grad()

        if i % (len_train_source-1) == 0:
        # if i % len_train_source == 0:
            #print(i)
            # iter_source = iter(loaders_train)
            iter_source1 = iter(loaders_train1)
            iter_source2 = iter(loaders_train2)
        if i % (len_test_source ) == 0:
            iter_target = iter(loaders_test)

        # inputs_source, labels_source, labels_source_father, inputs_target = iter_source.next()
        inputs_source, labels_source = iter_source1.next()
        inputs_target, _ = iter_source2.next()
        # print((labels_source-labels_source_father).eq(0).float())
        # print(labels_source_father)

        if use_gpu:
            inputs_source, labels_source, inputs_target = Variable(inputs_source).cuda(), Variable(labels_source).cuda(), Variable(inputs_target).cuda()
        else:
            inputs_source, labels_source, inputs_target = Variable(inputs_source), Variable(labels_source),Variable(inputs_target)
           
        features_source = base_network(inputs_source)
        features_target = base_network(inputs_target)
        
        # features_source = F.relu(bottelneck(features_source1))
        # features_target = F.relu(bottelneck(features_target1))
        outputs_source1 = classifier_layer1(features_source)
        outputs_target1 = classifier_layer1(features_target)
        

        cls_loss = class_criterion(outputs_source1, labels_source)
        
        outputs_softmax = F.softmax(outputs_target1, dim=1)
        WEIGHT = torch.sum(torch.softmax(outputs_source1, dim=1) * outputs_softmax, 1)# - 0.2

        # print(WEIGHT.gt(.6).float())
        # transfer_loss = transfer_criterion(features_source, features_target, WEIGHT, True)
        # transfer_loss = DANN(torch.cat((features_source, features_target), dim=0), ad_net)#, max_s.gt(0.5).float()*WEIGHT.gt(0.6).float(), True)
        # transfer_loss = transfer_criterion(features_source, features_target, max_s.gt(0.5).float()*WEIGHT.gt(0.6).float(), True)
        transfer_loss = transfer_criterion(features_source, features_target)#, WEIGHT.gt(0.6).float(), True)
        # print(torch.sum(outputs_softmax[:,class_num:],1)* (1.0 - max_s.gt(0.5).float()*WEIGHT.gt(0.6).float()))
        


        # total_loss = cls_loss + args.w_gcn * (gcn_loss_1k +  gcn_loss_4k ) + args.w_entropy * entropy_loss #+ args.w_data * args.w_data / entropy_loss) + transfer_loss * args.w_align
        total_loss = cls_loss + transfer_loss * args.w_align #+ torch.mean(entropy) * 0.0+ min_loss* args.w_min
        # total_loss = cls_loss + args.w_gcn * (gcn_loss_1k +  gcn_loss_4k ) + args.w_entropy * (entropy_loss + args.w_data * args.w_data / entropy_loss) + transfer_loss * args.w_align + torch.mean(entropy) * 0.3+ min_loss* args.w_min
        print("Step "+str(i)+": cls_loss: "+str(cls_loss.cpu().data.numpy())+
                             " transfer_loss: "+str(transfer_loss.cpu().data.numpy()))

        total_loss.backward(retain_graph=True)
        if (i+1)% config["opt_num"] ==0:
              optimizer.step()
              optimizer.zero_grad()
        # if (i>600):
        #     args.align = 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, nargs='?', default=128, help="batch size")
    parser.add_argument('--img_size', type=int, nargs='?', default=256, help="image size")
    parser.add_argument('--source', type=str, nargs='?', default='amazon', help="source data")
    parser.add_argument('--target', type=str, nargs='?', default='webcam', help="target data")
    parser.add_argument('--loss_name', type=str, nargs='?', default='JAN', help="loss name")
    parser.add_argument('--save_name', type=str, nargs='?', default='SOURCE_ONLY', help="loss name")
    parser.add_argument('--w_entropy', type=float, nargs='?', default=8, help="weight of entropy for target domain")
    parser.add_argument('--w_gcn', type=float, nargs='?', default=2, help="weight of gcn")
    parser.add_argument('--w_data', type=float, nargs='?', default=0.25, help="percent of unseen data")
    parser.add_argument('--w_align', type=float, nargs='?', default=0.1, help="percent of unseen data")
    parser.add_argument('--w_min', type=float, nargs='?', default=1.5, help="percent of unseen data")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    config = {}
    config["num_iterations"] = 3000
    config["test_interval"] = 200
    config["save_num"] = 200
    config["opt_num"] = 1
    config["loss"] = {"name":args.loss_name, "trade_off":0.0}#args.tradeoff }
    config["network"] = {"name":"ResNet50"}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, "weight_decay":0.0001, "nesterov":True}, "lr_type":"inv", "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }
    print(config)
    print(args)
    train_classification(config)
