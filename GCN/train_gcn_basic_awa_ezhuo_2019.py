import argparse
import json
import random
import os.path as osp

import torch
import torch.nn.functional as F

from utils import ensure_path, set_gpu, l2_loss
from models.gcn import GCN


def save_checkpoint(name):
    torch.save(gcn.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj, osp.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    #good = [127]*len(mask)
    # print([i for i in range(127,127+40)])
    # from IPython import embed;embed();exit();
    return l2_loss(a[0][:127+40][mask], b[mask])
    # return l2_loss(a[mask], b[mask])
    #return l2_loss(a[[(x+127) for x in mask]], b[mask])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=3000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save-epoch', type=int, default=300)
    parser.add_argument('--save-path', default='RESULTS_MODELS/awa-basic3')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()

    set_gpu(args.gpu)

    save_path = args.save_path
    ensure_path(save_path)
    # graph =json.load(open('materials/AWA2/animals_graph_dense.json'))
    graph =json.load(open('materials/AWA2/animals_graph_all.json'))
    #graph = json.load(open('../ADM/AWA2/awa2-ezhuo-graph.json', 'r'))
    wnids = graph['wnids']
    n = len(wnids)
    edges = graph['edges']
    
    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    word_vectors = torch.tensor(graph['vectors']).cuda()
    word_vectors = F.normalize(word_vectors)
    #fc_vectors = torch.load('../ADM/AWA2/class54000.pkl')
    #fcfile = json.load(open('materials/fc-weights.json', 'r'))
    #train_wnids = [x[0] for x in fcfile]
    #fc_vectors = [x[1] for x in fcfile]
    #assert train_wnids == wnids[:len(train_wnids)]
    # fc_vectors_=torch.load('materials/AWA2/fc_weights_pretrained_on_I2AwA2_source_only.pkl')#['fc151']
    fc_vectors =torch.load('materials/AWA2/151+16_cls_from_1K_ft')['fc151+16_ft']
    #fc_vectors = torch.cat((fc_vectors_['weight'].cpu(), fc_vectors_['bias'].cpu().view(-1,1)),dim=1)
    #fc_vectors=F.normalize(torch.load('materials/fc151')['fc151'])
    fc_vectors = torch.tensor(fc_vectors).cuda()
    #fc_vectors =torch.cat((fc_vectors, torch.load('../ADM/AWA2/gcn/class16_fix5000.pkl')),dim=0)
    #laji = torch.load('../ADM/AWA2/gcn/class24_fix5000.pkl')
    #assert laji ==fc_vectors[127:151]
    #fc_vectors = torch.cat((fc_vectors,torch.load('class248000.pkl')[8:]),dim=0)
    #fc_vectors = torch.cat((fc_vectors,torch.load('class168000.pkl')),dim=0)
    # fc_vectors = F.normalize(fc_vectors)

    hidden_layers = 'd2048,d'
    gcn = GCN(n, edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers).cuda()

    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    for epoch in range(1, args.max_epoch + 1):
        gcn.train()
        output_vectors = gcn(word_vectors)
        # simi = torch.matmul(output_vectors[:127], output_vectors[:127].transpose(0,1))
        # simi_loss = torch.mean(torch.abs(simi - torch.diag(torch.diag(simi))))
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:151+16]) #+ 0.15 * simi_loss
        optimizer.zero_grad()
        #loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()

        gcn.eval()
        output_vectors = gcn(word_vectors)
        train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:151+16]).item()
        if v_val > 0:
            val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss
        print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'
              .format(epoch, train_loss, val_loss))

        trlog['train_loss'].append(train_loss)
        trlog['val_loss'].append(val_loss)
        trlog['min_loss'] = min_loss
        torch.save(trlog, osp.join(save_path, 'trlog'))

        if (epoch % args.save_epoch == 0):
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors
                }

        if epoch % args.save_epoch == 0:
            save_checkpoint('epoch-{}'.format(epoch))
        
        pred_obj = None

