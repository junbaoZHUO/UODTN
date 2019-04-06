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
from gcn_lib.gcn import GCN

optim_dict = {"SGD": optim.SGD}


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-7
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

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=1200.0): 
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -1)

def grad_reverse(x):
    return GradReverse()(x)

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
        self.max_iter = 1200.0

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
def DANN(features, ad_net,size1,size2, entropy=None):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * size1 + [[0]] * size2)).float().cuda()
    if entropy is not None:
        return torch.mean(entropy.view(-1 ,1)*nn.BCELoss(reduction='none')(ad_out, dc_target))
    else:
        return nn.BCELoss()(ad_out, dc_target)



def my_l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)

def image_classification_test(iter_test,len_now, base, class1, class2, gpu=True):
    start_test = True
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
        out1 = class1(output)
        out2 = class2(output)
        outputs = torch.cat((out1,out2),dim=1)
        if start_test:
            all_output = outputs.data.float()
            all_label = labels.data.float()
            _, predict = torch.max(all_output, 1)
            ind_1K = all_label.gt(39)
            ind_4K = 1-all_label.gt(39)
            COR = COR + torch.sum(torch.squeeze(predict).float() == all_label)
            Total = Total + all_label.size()[0]
            COR_1k = COR_1k + torch.sum(torch.squeeze(predict).float()[ind_1K] == all_label[ind_1K])
            Total_1k = Total_1k + torch.sum(ind_1K)
            COR_4k = COR_4k + torch.sum(torch.squeeze(predict).float()[ind_4K] == all_label[ind_4K])
            Total_4k = Total_4k + torch.sum(ind_4K)
    print('Unkown_acc: '+ str(float(COR_1k)/float(Total_1k)))                                                                                                    
    print('Known_acc: '+ str(float(COR_4k)/float(Total_4k)))
    accuracy = float(COR)/float(Total)
    return accuracy

def train_classification(config):
    ## set pre-process
    prep_train  = prep.image_train(resize_size=256, crop_size=224)
    prep_test = prep.image_test(resize_size=256, crop_size=224)
               
    ## set loss
    class_criterion = nn.CrossEntropyLoss()

    ## prepare data
    TRAIN_LIST = 'data/I2AWA2_40.txt'#'AWA_SS.txt#'data/new_AwA2_common.txt'
    TEST_LIST = 'data/new_AwA2.txt'
    BSZ = args.batch_size

    dsets_train = ImageList(open(TRAIN_LIST).readlines(), shape = (args.img_size,args.img_size), transform=prep_train)
    loaders_train = util_data.DataLoader(dsets_train, batch_size=BSZ, shuffle=True, num_workers=8, pin_memory=True)

    dsets_test = ImageList(open(TEST_LIST).readlines(), shape = (args.img_size,args.img_size),transform=prep_test, train=False)
    loaders_test = util_data.DataLoader(dsets_test, batch_size=BSZ, shuffle=True, num_workers=4, pin_memory=True)
    begin_num = 127
    class_num = 40
    all_num = 50
    ## set base network
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]]()
    base_network.load_state_dict(torch.load('GCN/materials/AWA2/base_net_pretrained_on_I2AwA2_source_only.pkl'))
    classifier_layer1 = nn.Linear(base_network.output_num(), class_num)
    classifier_layer2 = nn.Linear(base_network.output_num(), all_num-class_num)
    for param in base_network.parameters():
        param.requires_grad = False
    for param in base_network.layer4.parameters():
        param.requires_grad = True
    for param in base_network.layer3.parameters():
        param.requires_grad = True
    ## initialization
    
    weight_bias=torch.load('GCN/awa_50_cls_basic')['fc50']
    weight_bias_127=torch.load('GCN/materials/AWA2/151_cls_from_1K')['fc151']
    classifier_layer1.weight.data = weight_bias[:class_num,:2048]
    classifier_layer2.weight.data = weight_bias[class_num:,:2048]
    classifier_layer1.bias.data = weight_bias[:class_num,-1]
    classifier_layer2.bias.data = weight_bias[class_num:,-1]
    ad_net = AdversarialNetwork(2048, 1024)

    graph = json.load(open('GCN/materials/AWA2/animals_graph_all.json','r'))
    word_vectors = torch.tensor(graph['vectors'])
    wnids = graph['wnids']
    n = len(wnids)
    use_att =False
    if use_att:
        edges_set = graph['edges_set']
        print('edges_set', [len(l) for l in edges_set])
        lim = 4
        for i in range(lim + 1, len(edges_set)):
            edges_set[lim].extend(edges_set[i])
            edges_set = edges_set[:lim + 1]
            print('edges_set', [len(l) for l in edges_set])
            edges = edges_set
    else:
        edges = graph['edges']
        edges = edges + [(v, u) for (u, v) in edges]
        edges = edges + [(u, u) for u in range(n)]
    word_vectors = F.normalize(word_vectors).cuda()
    hidden_layers = 'd2048,d'
    gcn = GCN(n, edges, word_vectors.shape[1], 2049, hidden_layers)
    gcn.load_state_dict(torch.load('GCN/RESULTS_MODELS/awa-basic/epoch-3000.pth'))
    gcn.train()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        classifier_layer1 = classifier_layer1.cuda()
        classifier_layer2 = classifier_layer2.cuda()
        base_network = base_network.cuda()
        gcn =gcn.cuda()
        ad_net = ad_net.cuda()


    ## collect parameters
    parameter_list = [{"params": classifier_layer2.parameters(), "lr":2},
                      {"params": classifier_layer1.parameters(), "lr":5},
                      {"params": ad_net.parameters(), "lr":5},
                      {"params": base_network.layer3.parameters(), "lr":1},
                      {"params": base_network.layer4.parameters(), "lr":2},
                      {"params": gcn.parameters(), "lr":5}]

 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    
    
    len_train_source = len(loaders_train) - 1
    len_test_source = len(loaders_test) - 1
    optimizer.zero_grad()
    for i in range(config["num_iterations"]):
        if ((i + 0) % config["test_interval"] == 0 and i > 100) or i== config["num_iterations"]-1 :
            base_network.layer3.train(False)
            base_network.layer4.train(False)
            classifier_layer1.train(False)
            classifier_layer2.train(False)
            print(str(i)+' ACC:')
            iter_target = iter(loaders_test)
            print(image_classification_test(iter_target,len_test_source, base_network, classifier_layer1,classifier_layer2, gpu=use_gpu))
            iter_target = iter(loaders_test)

        classifier_layer1.train(True)
        classifier_layer2.train(True)
        base_network.layer3.train(True)
        base_network.layer4.train(True)
        ad_net.train(True)
        
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)

        if i % len_train_source == 0:
            iter_source = iter(loaders_train)
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
        outputs_source2 = classifier_layer2(features_source)
        outputs_target1 = classifier_layer1(features_target)
        outputs_target2 = classifier_layer2(features_target)
        
        outputs_source = torch.cat((outputs_source1,outputs_source2),dim=1)
        outputs_target = torch.cat((outputs_target1,outputs_target2),dim=1)
        output_vectors = gcn(word_vectors)

        cls_loss = class_criterion(outputs_source, labels_source)
        
        outputs_softmax = F.softmax(outputs_target, dim=1)

        WEIGHT = torch.sum(torch.softmax(outputs_source, dim=1)[:,:40] * outputs_softmax[:,:40], 1)# - 0.2
        max_s = torch.max(outputs_softmax[:,:40], 1)[0]# - 0.2
        coeff = calc_coeff(i)
        entropy = Entropy(outputs_softmax)
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        entropy_s = Entropy(F.softmax(outputs_source, dim=1))
        entropy_s.register_hook(grl_hook(coeff))
        entropy_s = 1.0+torch.exp(-entropy_s)
        Mask_ = (max_s.gt(0.4).float()*WEIGHT.gt(0.5).float()).byte()
        Mask =  (max_s.gt(0.4).float()*WEIGHT.gt(0.5).float()).view(-1, 1).repeat(1, 2048).byte()
        source_matched = torch.masked_select(features_source, Mask).view(-1, 2048)
        target_matched = torch.masked_select(features_target, Mask).view(-1, 2048)
        transfer_loss = DANN(torch.cat((features_source, target_matched), dim=0), ad_net, BSZ, target_matched.size()[0], torch.cat((entropy_s, torch.masked_select(entropy, Mask_)), dim=0))
        entropy_loss = torch.sum(torch.sum(outputs_softmax[:,class_num:],1) * (1.0 - max_s.gt(0.5).float()*WEIGHT.gt(0.6).float()))/torch.sum(1.0 - max_s.gt(0.5).float()*WEIGHT.gt(0.6).float())
        
        class1_weightbias = torch.cat((classifier_layer1.weight,classifier_layer1.bias.view(-1, 1)),dim=1)
        class2_weightbias = torch.cat((classifier_layer2.weight,classifier_layer2.bias.view(-1, 1)),dim=1)
        classifier_weight_bias = torch.cat((class1_weightbias,class2_weightbias), dim=0)

        gcn_loss_1k = my_l2_loss(output_vectors[begin_num:(begin_num+class_num)], class1_weightbias)
        gcn_loss_4k = my_l2_loss(output_vectors[(begin_num+class_num):(begin_num+all_num)], class2_weightbias)
        gcn_loss_127 = my_l2_loss(output_vectors[:begin_num], weight_bias_127[:127].cuda())


        total_loss = cls_loss + args.w_gcn * (gcn_loss_1k +  gcn_loss_4k +  gcn_loss_127 ) + args.w_entropy * (entropy_loss + args.w_data * args.w_data / entropy_loss) + transfer_loss * args.w_align 
        print("Step "+str(i)+": cls_loss: "+str(cls_loss.cpu().data.numpy())+
                             " entropy_loss: "+str(entropy_loss.cpu().data.numpy())+
                             " transfer_loss: "+str(transfer_loss.cpu().data.numpy())+
                             " gcn_1k_loss: "+str(gcn_loss_1k.cpu().data.numpy())+
                             " gcn_4k_loss: "+str(gcn_loss_4k.cpu().data.numpy())+
                             " gcn_127_loss: "+str(gcn_loss_4k.cpu().data.numpy()))

        if ( i + 0 ) % config["save_num"] == 0:
            if not osp.exists(osp.join('save',args.save_name)):
                os.mkdir(osp.join('save',args.save_name))
            torch.save(base_network.state_dict(),osp.join('save',args.save_name,'base_net%d.pkl'%(i+1)))
            torch.save(classifier_weight_bias,osp.join('save',args.save_name,'class%d.pkl'%(i+1)))
            torch.save(gcn.state_dict(),osp.join('save',args.save_name,'gcn_net%d.pkl'%(i+1)))

        total_loss.backward(retain_graph=True)
        if (i+1)% config["opt_num"] ==0:
              optimizer.step()
              optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, nargs='?', default=90, help="batch size")
    parser.add_argument('--img_size', type=int, nargs='?', default=256, help="image size")
    parser.add_argument('--save_name', type=str, nargs='?', default='UODTN', help="loss name")
    parser.add_argument('--w_entropy', type=float, nargs='?', default=3, help="weight of entropy for target domain")
    parser.add_argument('--w_gcn', type=float, nargs='?', default=1.5, help="weight of gcn")
    parser.add_argument('--w_data', type=float, nargs='?', default=0.09, help="percent of unseen data")
    parser.add_argument('--w_align', type=float, nargs='?', default=0.7, help="percent of unseen data")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    config = {}
    config["num_iterations"] = 1200
    config["test_interval"] = 200
    config["save_num"] = 200
    config["opt_num"] = 1
    config["network"] = {"name":"ResNet50"}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, "weight_decay":0.0001, "nesterov":True}, "lr_type":"inv", "lr_param":{"init_lr":0.0001, "gamma":0.001, "power":0.75} }
    print(config)
    print(args)
    train_classification(config)
