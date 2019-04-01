import torch
import json
import numpy as np
# from IPython import embed;embed();exit();
all_good = torch.load('RESULTS_MODELS/awa-att/epoch-3000.pred')
# all_good = torch.load('RESULTS_MODELS/awa-dense/epoch-3000.pred')
# all_good = torch.load('RESULTS_MODELS/awa-basic3/epoch-3000.pred')
all_id = all_good['wnids']
all_matric = all_good['pred']
test_sets = json.load(open('materials/AWA2/animals_graph_grouped.json'))
# test_sets = json.load(open('materials/AWA2/animals_graph_dense.json'))
# test_sets = json.load(open('materials/AWA2/animals_graph_all.json'))
#test_sets = json.load(open('../ADM/AWA2/awa2-ezhuo-graph.json', 'r'))
good_id = test_sets['wnids'][127:177]

mytensor = torch.zeros(len(good_id),all_matric.shape[1])
# mytensor = torch.zeros(len(good_id),all_matric[0].shape[1])
for j in range(len(good_id)):
    find=0
    for i in range(len(all_id)):
        if all_id[i] ==good_id[j]:
            # mytensor[j]=all_matric[0][i]
            mytensor[j]=all_matric[i]
            find=1
            break
    if find ==0:
        print(good_id[0])
torch.save({'fc50':mytensor},'awa_50_cls_att')
    
