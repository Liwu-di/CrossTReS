# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 15:42
# @Author  : 银尘
# @FileName: dastnet_base.py
# @Software: PyCharm
# @Email   : liwudi@liwudi.fun
# @Info    : why create this file
from virtual_city import *

def load_all_adj(device):
    dirs = "./data/{}/{}_roads.npy"
    ny, chi, dc = None, None, None
    for i in ["NY", "CHI", "DC"]:
        t = dirs.format(i, i)
        t = np.load(t)
        t = t.reshape((t.shape[0] * t.shape[1], t.shape[0] * t.shape[1]))
        t = np.where(t >= 1, 1, t)
        t = add_self_loop(t)
        for m in range(t.shape[0]):
            for n in range(t.shape[1]):
                a, b = idx_1d22d(m, t.shape)
                c, d = idx_1d22d(n, t.shape)
                dis = abs(a - c) + abs(b - d)
                if t[m][n] - 0 > 1e-6 and dis != 0:
                    t[m][n] = t[m][n] / dis
        if t.shape[0] == 460:
            ny = t
        elif t.shape[0] == 476:
            chi = t
        elif t.shape[0] == 420:
            dc = t

    return torch.tensor(ny).to(device), torch.tensor(chi).to(device), torch.tensor(dc).to(device)


adj_pems04, adj_pems07, adj_pems08 = load_all_adj(device)
vec_pems04 = vec_pems07 = vec_pems08 = None, None, None
tttttttttttt = virtual_road.copy()
tttttttttttt = np.where(tttttttttttt >= 1, 1, tttttttttttt)
tttttttttttt = add_self_loop(tttttttttttt)
for m in range(tttttttttttt.shape[0]):
    for n in range(tttttttttttt.shape[1]):
        a, b = idx_1d22d(m, tttttttttttt.shape)
        c, d = idx_1d22d(n, tttttttttttt.shape)
        dis = abs(a - c) + abs(b - d)
        if tttttttttttt[m][n] - 0 > 1e-6 and dis != 0:
            tttttttttttt[m][n] = tttttttttttt[m][n] / dis
adj_virtual = torch.tensor(tttttttttttt).to(device)
cur_dir = os.getcwd()
dc = np.load("./data/DC/{}DC_{}.npy".format(args.dataname, args.datatype))
dcmask = dc.sum(0) > 0
th_maskdc = torch.from_numpy(dcmask.reshape(1, 420)).to(device)
chi = np.load("./data/CHI/{}CHI_{}.npy".format(args.dataname, args.datatype))
chimask = chi.sum(0) > 0
th_maskchi = torch.from_numpy(chimask.reshape(1, 476)).to(device)
ny = np.load("./data/NY/{}NY_{}.npy".format(args.dataname, args.datatype))
nymask = ny.sum(0) > 0
th_maskny = torch.from_numpy(nymask.reshape(1, 460)).to(device)
cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]

pems04_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems04',
                               '{}_vecdim.pkl'.format(args.vec_dim))
pems07_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems07',
                               '{}_vecdim.pkl'.format(args.vec_dim))
pems08_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems08',
                               '{}_vecdim.pkl'.format(args.vec_dim))
v_p = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems08',
                   '{}{}{}{}{}{}_vecdim.pkl'.format(args.vec_dim, args.dataname, args.datatype,
                                                    str(args.s1_rate).replace(".", ""),
                                                    str(args.s2_rate).replace(".", ""),
                                                    str(args.s3_rate).replace(".", "")))

for i in [pems04_emb_path, pems07_emb_path, pems08_emb_path, v_p]:
    a = i.split(os.path.sep)
    b = []
    for i in a:
        if "pkl" in i:
            continue
        else:
            b.append(i)
    local_path_generate(folder_name=os.path.sep.join(b), create_folder_only=True)


def generate_vector(adj, args):
    graph = nx.DiGraph(adj)
    spl = dict(nx.all_pairs_shortest_path_length(graph))
    from node2vec import Node2Vec as node2vec
    n2v = node2vec(G=graph, distance=spl, emb_size=args.vec_dim, length_walk=args.walk_length,
                   num_walks=args.num_walks, window_size=5, batch=4, p=args.p, q=args.q,
                   workers=(int(cpu_count() / 2)))
    n2v = n2v.train()

    gfeat = []
    for i in range(len(adj)):
        nodeivec = n2v.wv.get_vector(str(i))
        gfeat.append(nodeivec)
    g = torch.tensor(np.array(gfeat))

    return g, gfeat
if os.path.exists(pems04_emb_path):
    log(f'Loading pems04 embedding...')
    vec_pems04 = torch.load(pems04_emb_path, map_location='cpu')
    vec_pems04 = vec_pems04.to(device)
else:
    log(f'Generating pems04 embedding...')
    args.dataset = '4'
    vec_pems04, _ = generate_vector(adj_pems04.cpu().numpy(), args)
    vec_pems04 = vec_pems04.to(device)
    log(f'Saving pems04 embedding...')
    torch.save(vec_pems04.cpu(), pems04_emb_path)

if os.path.exists(pems07_emb_path):
    log(f'Loading pems07 embedding...')
    vec_pems07 = torch.load(pems07_emb_path, map_location='cpu')
    vec_pems07 = vec_pems07.to(device)
else:
    log(f'Generating pems07 embedding...')
    args.dataset = '7'
    vec_pems07, _ = generate_vector(adj_pems07.cpu().numpy(), args)
    vec_pems07 = vec_pems07.to(device)
    log(f'Saving pems07 embedding...')
    torch.save(vec_pems07.cpu(), pems07_emb_path)

if os.path.exists(pems08_emb_path):
    log(f'Loading pems08 embedding...')
    vec_pems08 = torch.load(pems08_emb_path, map_location='cpu')
    vec_pems08 = vec_pems08.to(device)
else:
    log(f'Generating pems08 embedding...')
    args.dataset = '8'
    vec_pems08, _ = generate_vector(adj_pems08.cpu().numpy(), args)
    vec_pems08 = vec_pems08.to(device)
    log(f'Saving pems08 embedding...')
    torch.save(vec_pems08.cpu(), pems08_emb_path)

if os.path.exists(v_p):
    log(f'Loading v embedding...')
    vec_virtual = torch.load(v_p, map_location='cpu')
    vec_virtual = vec_virtual.to(device)
else:
    log(f'Generating virtual embedding...')
    args.dataset = '8'
    vec_virtual, _ = generate_vector(virtual_road, args)
    vec_virtual = vec_virtual.to(device)
    log(f'Saving virtual embedding...')
    torch.save(vec_virtual.cpu(), v_p)
log(
    f'Successfully load embeddings, 4: {vec_pems04.shape}, 7: {vec_pems07.shape}, 8: {vec_pems08.shape}, vec_virtual:{vec_virtual.shape}')

domain_criterion = torch.nn.NLLLoss()
domain_classifier = Domain_classifier_DG(num_class=2, encode_dim=args.enc_dim)

domain_classifier = domain_classifier.to(device)
state = g = None, None

batch_seen = 0
cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]
assert args.models in ["DASTNet"]

bak_epoch = args.epoch
bak_val = args.val
bak_test = args.test
type = 'pretrain'
pretrain_model_path = os.path.join('{}'.format(cur_dir), 'pretrained', 'transfer_models',
                                   '{}'.format(args.dataset), '{}_prelen'.format(args.pre_len),
                                   'flow_model4_{}_epoch_{}_{}_{}_{}_{}_{}.pkl'.format(
                                       args.models, args.epoch, args.dataname, args.datatype,
                                       str(args.s1_rate).replace(".", ""),
                                       str(args.s2_rate).replace(".", ""),
                                       str(args.s3_rate).replace(".", ""))
                                   )

a = pretrain_model_path.split(os.path.sep)
b = []
for i in a:
    if "pkl" not in i:
        b.append(i)
local_path_generate(os.path.sep.join(b), create_folder_only=True)
vec_pems04 = vec_virtual
adj_pems04 = adj_virtual
args.dataset = "8"