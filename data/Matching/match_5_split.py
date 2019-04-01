import numpy as np
import os, sys
import json
import random
import hungarian

import os

os.system('cat ../WEB_3D3_2.txt ../WEB_3D3_2.txt ../WEB_3D3_2.txt ../WEB_3D3_2.txt ../WEB_3D3_2.txt ../WEB_3D3_2.txt ../WEB_3D3_2.txt ../WEB_3D3_2.txt > WEB_3D3_.txt')
print('Loading S')
imgs = np.loadtxt(open('S_features.csv','rb'),delimiter=',',skiprows=0)
imgs = np.concatenate((imgs, imgs), axis=0)
imgs = np.concatenate((imgs, imgs), axis=0)
imgs = np.concatenate((imgs, imgs), axis=0)
paths = open('./WEB_3D3_.txt','r').readlines()
print('Loading T')
imgs_web = np.loadtxt(open('T_features.csv','rb'),delimiter=',',skiprows=0)
paths_web = open('./new_AwA2.txt','r').readlines()
print('Loaded !')

L = imgs.shape[0]
print(L)
print(imgs.shape[0])
print(imgs.shape[1])
shuffle = [i for i in range(L)]
random.shuffle(shuffle)
Tail = imgs_web.shape[0] - imgs.shape[0]
imgs = np.concatenate((imgs, imgs[shuffle[:Tail]]), axis=0)
print(len(paths))
paths_tail = [paths[i] for i in shuffle[:Tail]]
paths = paths + paths_tail

L = imgs_web.shape[0]
shuffle = [i for i in range(L)]
random.shuffle(shuffle)

imgs_web = imgs_web[shuffle]
paths_web = [paths_web[i] for i in shuffle]

random.shuffle(shuffle)
imgs = imgs[shuffle]
paths = [paths[i] for i in shuffle]



d_h1 = imgs[:8500].astype(np.float32)
d_h2 = imgs_web[:8500].astype(np.float32)
L1_ = np.zeros((d_h1.shape[0], d_h2.shape[0]))
ws = []
BSZ = 50
for i in range(int(d_h1.shape[0]/BSZ)+1):
    print(i)
    if i!=int(d_h1.shape[0]/BSZ):
        d_h1_ = np.repeat(d_h1[int(i*BSZ):int((i+1)*BSZ)], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [BSZ,1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):int((i+1)*BSZ),:] = np.reshape(L1, (BSZ, d_h2.shape[0]))
    else:
        d_h1_ = np.repeat(d_h1[int(i*BSZ):], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [int(d_h1.shape[0]%BSZ),1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):,:] = np.reshape(L1, (int(d_h1.shape[0]%BSZ), d_h2.shape[0]))


obj_matrix = L1_
f=open('COR1.txt','w')
print('Begin')
rs, cs = hungarian.lap(obj_matrix)
print('Write result')
for i in range(8500):
    f.write(paths[i][:-1]+' '+ paths_web[rs[i]])
f.close()
f=open('COR2.txt','w')
d_h1 = imgs[8500:17000].astype(np.float32)
d_h2 = imgs_web[8500:17000].astype(np.float32)
L1_ = np.zeros((d_h1.shape[0], d_h2.shape[0]))
ws = []
BSZ = 50
for i in range(int(d_h1.shape[0]/BSZ)+1):
    print(i)
    if i!=int(d_h1.shape[0]/BSZ):
        d_h1_ = np.repeat(d_h1[int(i*BSZ):int((i+1)*BSZ)], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [BSZ,1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):int((i+1)*BSZ),:] = np.reshape(L1, (BSZ, d_h2.shape[0]))
    else:
        d_h1_ = np.repeat(d_h1[int(i*BSZ):], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [int(d_h1.shape[0]%BSZ),1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):,:] = np.reshape(L1, (int(d_h1.shape[0]%BSZ), d_h2.shape[0]))


obj_matrix = L1_
print('Begin')
rs, cs = hungarian.lap(obj_matrix)
print('Write result')
for i in range(8500):
    f.write(paths[i+8500][:-1]+' '+ paths_web[rs[i]+8500])
f.close()
f=open('COR3.txt','w')

d_h1 = imgs[17000:25500].astype(np.float32)
d_h2 = imgs_web[17000:25500].astype(np.float32)
L1_ = np.zeros((d_h1.shape[0], d_h2.shape[0]))
ws = []
BSZ = 50
for i in range(int(d_h1.shape[0]/BSZ)+1):
    print(i)
    if i!=int(d_h1.shape[0]/BSZ):
        d_h1_ = np.repeat(d_h1[int(i*BSZ):int((i+1)*BSZ)], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [BSZ,1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):int((i+1)*BSZ),:] = np.reshape(L1, (BSZ, d_h2.shape[0]))
    else:
        d_h1_ = np.repeat(d_h1[int(i*BSZ):], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [int(d_h1.shape[0]%BSZ),1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):,:] = np.reshape(L1, (int(d_h1.shape[0]%BSZ), d_h2.shape[0]))


obj_matrix = L1_
print('Begin')
rs, cs = hungarian.lap(obj_matrix)
print('Write result')
for i in range(8500):
    f.write(paths[i+17000][:-1]+' '+ paths_web[rs[i]+17000])

f.close()
f=open('COR4.txt','w')
d_h1 = imgs[25500:34000].astype(np.float32)
d_h2 = imgs_web[25500:34000].astype(np.float32)
L1_ = np.zeros((d_h1.shape[0], d_h2.shape[0]))
ws = []
BSZ = 50
for i in range(int(d_h1.shape[0]/BSZ)+1):
    print(i)
    if i!=int(d_h1.shape[0]/BSZ):
        d_h1_ = np.repeat(d_h1[int(i*BSZ):int((i+1)*BSZ)], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [BSZ,1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):int((i+1)*BSZ),:] = np.reshape(L1, (BSZ, d_h2.shape[0]))
    else:
        d_h1_ = np.repeat(d_h1[int(i*BSZ):], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [int(d_h1.shape[0]%BSZ),1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):,:] = np.reshape(L1, (int(d_h1.shape[0]%BSZ), d_h2.shape[0]))


obj_matrix = L1_
print('Begin')
rs, cs = hungarian.lap(obj_matrix)
print('Write result')
for i in range(min(8500,8268)):
    f.write(paths[i+25500][:-1]+' '+ paths_web[rs[i]+25500])
f.close()
f=open('COR5.txt','w')

d_h1 = imgs[34000:].astype(np.float32)
d_h2 = imgs_web[34000:].astype(np.float32)
L1_ = np.zeros((d_h1.shape[0], d_h2.shape[0]))
ws = []
BSZ = 50
for i in range(int(d_h1.shape[0]/BSZ)+1):
    print(i)
    if i!=int(d_h1.shape[0]/BSZ):
        d_h1_ = np.repeat(d_h1[int(i*BSZ):int((i+1)*BSZ)], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [BSZ,1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):int((i+1)*BSZ),:] = np.reshape(L1, (BSZ, d_h2.shape[0]))
    else:
        d_h1_ = np.repeat(d_h1[int(i*BSZ):], d_h2.shape[0], axis=0)
        d_h2_ = np.tile(d_h2, [int(d_h1.shape[0]%BSZ),1])
        L1 = np.sum(np.abs(d_h1_ - d_h2_), 1)
        L1_[int(i*BSZ):,:] = np.reshape(L1, (int(d_h1.shape[0]%BSZ), d_h2.shape[0]))


obj_matrix = L1_
print('Begin')
rs, cs = hungarian.lap(obj_matrix)
print('Write result')
for i in range(len(imgs)-34000):
    f.write(paths[i+34000][:-1]+' '+ paths_web[rs[i]+34000])

f.close()

os.system('cat COR1.txt COR2.txt COR3.txt COR4.txt COR5.txt > Matched_pairs.txt')
