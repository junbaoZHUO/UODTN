import torch
import json
import numpy as np
import torch
import torch.nn.functional as F
p = torch.load('resnet50-raw.pth')
w = p['fc.weight'].data
b = p['fc.bias'].data
v = torch.cat([w, b.unsqueeze(1)], dim=1).tolist()

wnids = json.load(open('imagenet-split.json', 'r'))['train']
wnids = sorted(wnids)



test_sets = json.load(open('animals_graph_all.json'))
good_id = test_sets['wnids'][:127+24]

mytensor = torch.zeros(len(good_id),2049)
# from IPython import embed;embed();exit();
for j in range(len(good_id)):
    find=0
    for i in range(len(wnids)):
        if wnids[i] ==good_id[j]:
            mytensor[j]=torch.FloatTensor(v[i])
            find=1
            break
    if find ==0:
        print(good_id[0])
torch.save({'fc151':mytensor},'151_cls_from_1K')
    
