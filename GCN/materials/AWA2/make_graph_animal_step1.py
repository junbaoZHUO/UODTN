import argparse
import json

from nltk.corpus import wordnet as wn
import torch

from glove import GloVe


def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s


def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges


def induce_parents(s, stop_set):
    q = s
    vis = set(s)
    l = 0
    while l < len(q):
        u = q[l]
        l += 1
        #if len(u.hypernyms())>1:
        #    from IPython import embed;embed();exit();
        for p in u.hypernyms():
            if p not in vis:
                vis.add(p)
                q.append(p)
    add = stop_set
    #from IPython import embed;embed();exit();
    l = 0
    while l<len(add):
        u = add[l]
        l+=1
        m =u.hypernyms()
        if len(m)==0:
            continue
        for ii in range(len(m)):
            if m[ii] in q[50:] and u not in q:
                vis.add(u)
                q.append(u)
            a = m[ii].hypernyms()
            if len(a)==0:
                continue
            for jj in range(len(a)):
                if a[0] in q[50:] and u not in q:
                     vis.add(u)
                     q.append(u)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='awa2-split.json')
    parser.add_argument('--output', default='animals_graph_step1.json')
    args = parser.parse_args()

    print('making graph ...')
    
    xml_wnids = json.load(open('web_wnids.json', 'r'))['train']
    xml_set = list(map(getnode, xml_wnids))
    #xml_set = set(xml_nodes)
    #'''
    js = json.load(open(args.input, 'r'))
    train_wnids = js['train']
    test_wnids = js['test']
    key_wnids = train_wnids + test_wnids
    '''
    all_id = json.load(open('animals.json'))['wnids']
    key_wnids = all_id[106:]
    for thing in all_id[50:106]:
        if thing in xml_wnids:
            key_wnids.append(thing)
    print(len(key_wnids))
    for thing in all_id[:50]:
        if thing in xml_wnids:
            key_wnids.append(thing)
    print(len(key_wnids))
    for thing in all_id[:50]:
        if thing not in xml_wnids:
            key_wnids.append(thing)
    print(len(key_wnids))'''

    s = list(map(getnode, key_wnids))
    induce_parents(s, xml_set)

    s_set = set(s)

    wnids = list(map(getwnid, s))
    edges = getedges(s)

    print('making glove embedding ...')

    glove = GloVe('glove.6B.300d.txt')
    vectors = []
    for wnid in wnids:
        vectors.append(glove[getnode(wnid).lemma_names()])
    vectors = torch.stack(vectors)

    print('dumping ...')

    obj = {}
    obj['wnids'] = wnids
    obj['vectors'] = vectors.tolist()
    obj['edges'] = edges
    json.dump(obj, open(args.output, 'w'))

