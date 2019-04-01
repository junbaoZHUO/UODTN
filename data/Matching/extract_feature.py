import json
import os.path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as util_data


from data_list import ImageList
import pre_process as prep
from resnet import make_resnet50_base


model_path ='resnet50-base.pth'
cnn = make_resnet50_base()
cnn.load_state_dict(torch.load(model_path))
for param in cnn.parameters():
    param.requires_grad = False
cnn.cuda()
cnn.eval()
infile = open('S_features.csv','a')
TEST_LIST = 'WEB_3D3_2.txt'
prep_test = prep.image_test(resize_size=256, crop_size=224)
dsets_test = ImageList(open(TEST_LIST).readlines(), shape = (256,256),transform=prep_test, train=False)
loaders_test = util_data.DataLoader(dsets_test, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
for batch_id ,batch in enumerate(loaders_test,1):
    print(str(batch_id))
    data,label = batch
    data = data.cuda()
    features = cnn(data)
    now = features.cpu().data.numpy()
    np.savetxt(infile, now, delimiter = ',')
    torch.cuda.empty_cache()

infile = open('T_features.csv','a')
TEST_LIST = 'new_AwA2.txt'
prep_test = prep.image_test(resize_size=256, crop_size=224)
dsets_test = ImageList(open(TEST_LIST).readlines(), shape = (256,256),transform=prep_test, train=False)
loaders_test = util_data.DataLoader(dsets_test, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
for batch_id ,batch in enumerate(loaders_test,1):
    print(str(batch_id))
    data,label = batch
    data = data.cuda()
    features = cnn(data)
    now = features.cpu().data.numpy()
    np.savetxt(infile, now, delimiter = ',')
    torch.cuda.empty_cache()
