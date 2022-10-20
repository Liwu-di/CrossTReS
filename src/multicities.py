# coding: utf-8

# In[ ]:


import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PaperCrawlerUtil.common_util import *
from PaperCrawlerUtil.constant import *
from PaperCrawlerUtil.crawler_util import *
from dgl.nn import GATConv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from model import *
from utils import *


# In[2]:


basic_config(logs_style=LOG_STYLE_ALL)
p_bar = process_bar(final_prompt="初始化准备完成", unit="part")
p_bar.process(0, 1, 5)
# This file implements the full version of using region embeddings to select good source data.
parser = argparse.ArgumentParser()
# 源城市
parser.add_argument('--scity', type=str, default='NY')
# 目标城市
parser.add_argument('--tcity', type=str, default='DC')
# 数据集名称
parser.add_argument('--dataname', type=str, default='Taxi', help='Within [Bike, Taxi]')
# 数据类型
parser.add_argument('--datatype', type=str, default='pickup', help='Within [pickup, dropoff]')
# 尝试减小，看显存能不能撑住 32 -> 16
parser.add_argument('--batch_size', type=int, default=16)
# 模型
parser.add_argument("--model", type=str, default='STNet_nobn', help='Within [STResNet, STNet, STNet_nobn]')
# 学习率
parser.add_argument('--learning_rate', type=float, default=1e-3)
# 权重
parser.add_argument('--weight_decay', type=float, default=5e-5)
# 100回合跑下来数据有问题，改成40epoch看看，论文也是这个
parser.add_argument('--num_epochs', type=int, default=80, help='Number of source training epochs')
parser.add_argument('--num_tuine_epochs', type=int, default=80, help='Number of fine tuine epochs')
# gpu设备序号
parser.add_argument('--gpu', type=int, default=0)
# 随机种子 不知道是干嘛的
parser.add_argument('--seed', type=int, default=-1, help='Random seed. -1 means do not manually set. ')
# 数据量
parser.add_argument('--data_amount', type=int, default=0, help='0: full data, 30/7/3 correspond to days of data')
# 内循环 源训练数量
parser.add_argument('--sinneriter', type=int, default=3, help='Number of inner iterations (source) for meta learning')
# 内循环 微调数量
parser.add_argument('--tinneriter', type=int, default=1, help='Number of inner iterations (target) for meta learning')
# 内循环元学习学习率
parser.add_argument('--innerlr', type=float, default=5e-5, help='Learning rate for inner loop of meta-learning')
# 外循环数量
parser.add_argument('--outeriter', type=int, default=20, help='Number of outer iterations for meta-learning')
# 外循环学习率
parser.add_argument('--outerlr', type=float, default=1e-4, help='Learning rate for the outer loop of meta-learning')
# 前k个参数
parser.add_argument('--topk', type=int, default=15)
# 最大平均误差参数 ，也就是beta1
parser.add_argument('--mmd_w', type=float, default=2, help='mmd weight')
# 边缘分类器参数， beta2
parser.add_argument('--et_w', type=float, default=2, help='edge classifier weight')
# 源域权重的移动平均参数
parser.add_argument("--ma_coef", type=float, default=0.6, help='Moving average parameter for source domain weights')
# 源域权重的正则化器。
parser.add_argument("--weight_reg", type=float, default=1e-3, help="Regularizer for the source domain weights.")
# 预训练回合数
parser.add_argument("--pretrain_iter", type=int, default=-1, help='Pre-training iterations per pre-training epoch. ')
args = parser.parse_args(args=[])

if args.seed != -1:
    # seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，
    # 如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    # random.seed(something)只能是一次有效
    # seed( ) 用于指定随机数生成时所用算法开始的整数值。
    # 1.如果使用相同的seed( )值，则每次生成的随即数都相同；
    # 2.如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    # 3.设置的seed()值仅一次有效
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
# 设置训练设备
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
p_bar.process(1, 1, 5)
dataname = args.dataname
scity = args.scity
scity2 = "CHI"
tcity = args.tcity
datatype = args.datatype
num_epochs = args.num_epochs
num_tuine_epochs = args.num_tuine_epochs
start_time = time.time()
log("Running CrossTReS, from %s and %s to %s, %s %s experiments, with %d days of data, on %s model" % \
    (scity, scity2, tcity, dataname, datatype, args.data_amount, args.model))


# In[3]:


# Load spatio temporal data
# (8784, 21, 20)
# 8784 = 366 * 24
target_data = np.load("../data/%s/%s%s_%s.npy" % (tcity, dataname, tcity, datatype))
# (21, 20) 经纬度分割
lng_target, lat_target = target_data.shape[1], target_data.shape[2]
# numpy.sum()，求和某一维度或者维度为none时，求和所有，减掉一个维度
# 此处，target_data (8784, 21, 20) -> (21, 20)
# 然后，通过对于每个元素判断是否大于0， 转成Bool向量
mask_target = target_data.sum(0) > 0
# reshape （21， 20） -》 （1， 21， 20）
th_mask_target = torch.Tensor(mask_target.reshape(1, lng_target, lat_target)).to(device)
log("%d valid regions in target" % np.sum(mask_target))
# (（21， 20）-> 420, （21， 20）-> 420)
target_emb_label = masked_percentile_label(target_data.sum(0).reshape(-1), mask_target.reshape(-1))
p_bar.process(2, 1, 5)


# In[4]:


# (8784, 20, 23)
source_data = np.load("../data/%s/%s%s_%s.npy" % (scity, dataname, scity, datatype))
log(source_data.shape)
# (20, 23)
lng_source, lat_source = source_data.shape[1], source_data.shape[2]
mask_source = source_data.sum(0) > 0
# mask -> th_mask = (20, 23) -> (1, 20, 23)
th_mask_source = torch.Tensor(mask_source.reshape(1, lng_source, lat_source)).to(device)
log("%d valid regions in source" % np.sum(mask_source))

source_data2 = np.load("../data/%s/%s%s_%s.npy" % (scity2, dataname, scity2, datatype))
log(source_data2.shape)
lng_source2, lat_source2 = source_data2.shape[1], source_data2.shape[2]
mask_source2 = source_data2.sum(0) > 0
th_mask_source2 = torch.Tensor(mask_source2.reshape(1, lng_source2, lat_source2)).to(device)
log("%d valid regions in source" % np.sum(mask_source2))


# In[5]:


# 按照百分比分配标签
source_emb_label = masked_percentile_label(source_data.sum(0).reshape(-1), mask_source.reshape(-1))
p_bar.process(3, 1, 5)
lag = [-6, -5, -4, -3, -2, -1]
source_data, smax, smin = min_max_normalize(source_data)
target_data, max_val, min_val = min_max_normalize(target_data)

source_emb_label2 = masked_percentile_label(source_data2.sum(0).reshape(-1), mask_source2.reshape(-1))
source_data2, smax2, smin2 = min_max_normalize(source_data2)


# In[6]:


# [(5898, 6, 20, 23), (5898, 1, 20, 23), (1440, 6, 20, 23), (1440, 1, 20, 23), (1440, 6, 20, 23), (1440, 1, 20, 23)]
# 第一维是数量，第二维是每条数据中的数量
source_train_x, source_train_y, source_val_x, source_val_y, source_test_x, source_test_y = split_x_y(source_data, lag)
source_train_x2, source_train_y2, source_val_x2, source_val_y2, source_test_x2, source_test_y2 = split_x_y(source_data2, lag)
# we concatenate all source data
# (8778, 6, 20, 23)
source_x = np.concatenate([source_train_x, source_val_x, source_test_x], axis=0)
# (8778, 1, 20, 23)
source_y = np.concatenate([source_train_y, source_val_y, source_test_y], axis=0)
source_x2 = np.concatenate([source_train_x2, source_val_x2, source_test_x2], axis=0)
source_y2 = np.concatenate([source_train_y2, source_val_y2, source_test_y2], axis=0)
target_train_x, target_train_y, target_val_x, target_val_y, target_test_x, target_test_y = split_x_y(target_data, lag)


# In[7]:


if args.data_amount != 0:
    # 负号表示从倒数方向数，
    # i.e.
    # a = [12, 3, 4, 5, 6, 7, 8]
    # c, d = a[-2:], a[:-2]
    # print(c)
    # print(d)
    # [7, 8]
    # [12, 3, 4, 5, 6]
    target_train_x = target_train_x[-args.data_amount * 24:, :, :, :]
    target_train_y = target_train_y[-args.data_amount * 24:, :, :, :]
log("Source split to: x %s, y %s" % (str(source_x.shape), str(source_y.shape)))
# log("val_x %s, val_y %s" % (str(source_val_x.shape), str(source_val_y.shape)))
# log("test_x %s, test_y %s" % (str(source_test_x.shape), str(source_test_y.shape)))
log("Source2 split to: x %s, y %s" % (str(source_x2.shape), str(source_y2.shape)))
log("Target split to: train_x %s, train_y %s" % (str(target_train_x.shape), str(target_train_y.shape)))
log("val_x %s, val_y %s" % (str(target_val_x.shape), str(target_val_y.shape)))
log("test_x %s, test_y %s" % (str(target_test_x.shape), str(target_test_y.shape)))


# In[8]:


# 这些代码 numpy -> Tensor -> TensorDataset -> DataLoader
target_train_dataset = TensorDataset(torch.Tensor(target_train_x), torch.Tensor(target_train_y))
target_val_dataset = TensorDataset(torch.Tensor(target_val_x), torch.Tensor(target_val_y))
target_test_dataset = TensorDataset(torch.Tensor(target_test_x), torch.Tensor(target_test_y))
target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True)
target_val_loader = DataLoader(target_val_dataset, batch_size=args.batch_size)
target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size)
source_test_dataset = TensorDataset(torch.Tensor(source_test_x), torch.Tensor(source_test_y))
source_test_loader = DataLoader(source_test_dataset, batch_size=args.batch_size)
source_dataset = TensorDataset(torch.Tensor(source_x), torch.Tensor(source_y))
source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
source_test_dataset2 = TensorDataset(torch.Tensor(source_test_x2), torch.Tensor(source_test_y2))
source_test_loader2 = DataLoader(source_test_dataset2, batch_size=args.batch_size)
source_dataset2 = TensorDataset(torch.Tensor(source_x2), torch.Tensor(source_y2))
source_loader2 = DataLoader(source_dataset2, batch_size=args.batch_size, shuffle=True)
p_bar.process(4, 1, 5)


# In[9]:


# Load auxiliary data: poi data
# (20, 23, 14)
source_poi = np.load("../data/%s/%s_poi.npy" % (scity, scity))
source_poi2 = np.load("../data/%s/%s_poi.npy" % (scity2, scity2))
target_poi = np.load("../data/%s/%s_poi.npy" % (tcity, tcity))
# (460, 14)
source_poi = source_poi.reshape(lng_source * lat_source, -1)  # regions * classes
source_poi2 = source_poi2.reshape(lng_source2 * lat_source2, -1)  # regions * classes
target_poi = target_poi.reshape(lng_target * lat_target, -1)  # regions * classes
transform = TfidfTransformer()
# 规范正则化到（0，1）
source_norm_poi = np.array(transform.fit_transform(source_poi).todense())
transform = TfidfTransformer()
# 规范正则化到（0，1）
source_norm_poi2 = np.array(transform.fit_transform(source_poi2).todense())
transform = TfidfTransformer()
target_norm_poi = np.array(transform.fit_transform(target_poi).todense())


# In[10]:


# Build graphs
# add_self_loop 增加一个自循环，对角线的值=1
source_prox_adj = add_self_loop(build_prox_graph(lng_source, lat_source))
source_prox_adj2 = add_self_loop(build_prox_graph(lng_source2, lat_source2))
target_prox_adj = add_self_loop(build_prox_graph(lng_target, lat_target))
source_road_adj = add_self_loop(build_road_graph(scity, lng_source, lat_source))
source_road_adj2 = add_self_loop(build_road_graph(scity2, lng_source2, lat_source2))
target_road_adj = add_self_loop(build_road_graph(tcity, lng_target, lat_target))
source_poi_adj, source_poi_cos = build_poi_graph(source_norm_poi, args.topk)
source_poi_adj2, source_poi_cos2 = build_poi_graph(source_norm_poi2, args.topk)
target_poi_adj, target_poi_cos = build_poi_graph(target_norm_poi, args.topk)
source_poi_adj = add_self_loop(source_poi_adj)
source_poi_adj2 = add_self_loop(source_poi_adj2)
target_poi_adj = add_self_loop(target_poi_adj)
source_s_adj, source_d_adj, source_od_adj = build_source_dest_graph(scity, dataname, lng_source, lat_source, args.topk)
source_s_adj2, source_d_adj2, source_od_adj2 = build_source_dest_graph(scity2, dataname, lng_source2, lat_source2, args.topk)
target_s_adj, target_d_adj, target_od_adj = build_source_dest_graph(tcity, dataname, lng_target, lat_target, args.topk)
source_s_adj = add_self_loop(source_s_adj)
source_s_adj2 = add_self_loop(source_s_adj2)
source_t_adj = add_self_loop(source_d_adj)
source_t_adj2 = add_self_loop(source_d_adj2)
source_od_adj = add_self_loop(source_od_adj)
source_od_adj2 = add_self_loop(source_od_adj2)
target_s_adj = add_self_loop(target_s_adj)
target_t_adj = add_self_loop(target_d_adj)
target_od_adj = add_self_loop(target_od_adj)
log("Source graphs: ")
log("prox_adj: %d nodes, %d edges" % (source_prox_adj.shape[0], np.sum(source_prox_adj)))
log("road adj: %d nodes, %d edges" % (source_road_adj.shape[0], np.sum(source_road_adj > 0)))
log("poi_adj, %d nodes, %d edges" % (source_poi_adj.shape[0], np.sum(source_poi_adj > 0)))
log("s_adj, %d nodes, %d edges" % (source_s_adj.shape[0], np.sum(source_s_adj > 0)))
log("d_adj, %d nodes, %d edges" % (source_d_adj.shape[0], np.sum(source_d_adj > 0)))
log()
log("Source2 graphs: ")
log("prox_adj: %d nodes, %d edges" % (source_prox_adj2.shape[0], np.sum(source_prox_adj2)))
log("road adj: %d nodes, %d edges" % (source_road_adj2.shape[0], np.sum(source_road_adj2 > 0)))
log("poi_adj, %d nodes, %d edges" % (source_poi_adj2.shape[0], np.sum(source_poi_adj2 > 0)))
log("s_adj, %d nodes, %d edges" % (source_s_adj2.shape[0], np.sum(source_s_adj2 > 0)))
log("d_adj, %d nodes, %d edges" % (source_d_adj2.shape[0], np.sum(source_d_adj2 > 0)))
log()
log("Target graphs:")
log("prox_adj: %d nodes, %d edges" % (target_prox_adj.shape[0], np.sum(target_prox_adj)))
log("road adj: %d nodes, %d edges" % (target_road_adj.shape[0], np.sum(target_road_adj > 0)))
log("poi_adj, %d nodes, %d edges" % (target_poi_adj.shape[0], np.sum(target_poi_adj > 0)))
log("s_adj, %d nodes, %d edges" % (target_s_adj.shape[0], np.sum(target_s_adj > 0)))
log("d_adj, %d nodes, %d edges" % (target_d_adj.shape[0], np.sum(target_d_adj > 0)))
log()
source_graphs = adjs_to_graphs([source_prox_adj, source_road_adj, source_poi_adj, source_s_adj, source_d_adj])
source_graphs2 = adjs_to_graphs([source_prox_adj2, source_road_adj2, source_poi_adj2, source_s_adj2, source_d_adj2])
target_graphs = adjs_to_graphs([target_prox_adj, target_road_adj, target_poi_adj, target_s_adj, target_d_adj])
for i in range(len(source_graphs)):
    source_graphs[i] = source_graphs[i].to(device)
    source_graphs2[i] = source_graphs2[i].to(device)
    target_graphs[i] = target_graphs[i].to(device)


# In[11]:


# This function is a preparation for the edge type discriminator
def graphs_to_edge_labels(graphs):
    """
    准备边缘分类器的训练材料， 边的表示（起点，终点），边的标识【0,0,0,0,0】
    :param graphs:
    :return:
    """
    edge_label_dict = {}
    for i, graph in enumerate(graphs):
        src, dst = graph.edges()
        for s, d in zip(src, dst):
            s = s.item()
            d = d.item()
            if (s, d) not in edge_label_dict:
                edge_label_dict[(s, d)] = np.zeros(len(graphs))
            edge_label_dict[(s, d)][i] = 1
    edges = []
    edge_labels = []
    for k in edge_label_dict.keys():
        edges.append(k)
        edge_labels.append(edge_label_dict[k])
    edges = np.array(edges)
    edge_labels = np.array(edge_labels)
    return edges, edge_labels


source_edges, source_edge_labels = graphs_to_edge_labels(source_graphs)
source_edges2, source_edge_labels2 = graphs_to_edge_labels(source_graphs2)
target_edges, target_edge_labels = graphs_to_edge_labels(target_graphs)
p_bar.process(5, 1, 5)



# In[12]:


# build models
# we need one embedding model, one scoring model, one prediction model
# 图注意力
class MVGAT(nn.Module):
    def __init__(self, num_graphs=3, num_gat_layer=2, in_dim=14, hidden_dim=64, emb_dim=32, num_heads=2, residual=True):
        super().__init__()
        self.num_graphs = num_graphs
        self.num_gat_layer = num_gat_layer
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.residual = residual

        self.multi_gats = nn.ModuleList()
        for j in range(self.num_gat_layer):
            gats = nn.ModuleList()
            for i in range(self.num_graphs):
                if j == 0:
                    gats.append(GATConv(self.in_dim,
                                        self.hidden_dim,
                                        self.num_heads,
                                        residual=self.residual,
                                        allow_zero_in_degree=True))
                elif j == self.num_gat_layer - 1:
                    gats.append(GATConv(self.hidden_dim * self.num_heads,
                                        self.emb_dim // self.num_heads,
                                        self.num_heads,
                                        residual=self.residual,
                                        allow_zero_in_degree=True))
                else:
                    gats.append(GATConv(self.hidden_dim * self.num_heads,
                                        self.hidden_dim,
                                        self.num_heads,
                                        residual=self.residual,
                                        allow_zero_in_degree=True))
            self.multi_gats.append(gats)

    def forward(self, graphs, feat):
        views = []
        for i in range(self.num_graphs):
            for j in range(self.num_gat_layer):
                if j == 0:
                    z = self.multi_gats[j][i](graphs[i], feat)
                else:
                    z = self.multi_gats[j][i](graphs[i], z)
                if j != self.num_gat_layer - 1:
                    z = F.relu(z)
                z = z.flatten(1)
            views.append(z)
        return views


# 融合模型
class FusionModule(nn.Module):
    """
    融合多图模型的特征，使用了注意力机制，用全连接实现
    """
    def __init__(self, num_graphs, emb_dim, alpha):
        super().__init__()
        self.num_graphs = num_graphs
        self.emb_dim = emb_dim
        self.alpha = alpha

        self.fusion_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.self_q = nn.ModuleList()
        self.self_k = nn.ModuleList()
        for i in range(self.num_graphs):
            self.self_q.append(nn.Linear(self.emb_dim, self.emb_dim))
            self.self_k.append(nn.Linear(self.emb_dim, self.emb_dim))

    def forward(self, views):
        """
        views -> cat_views cat_views = torch.stack(views, dim=0)
        for 1 - 5
            cat_views = 5*460*64
            attn = torch.matmul(Q, K.transpose(1, 2))
            output = torch.matmul(attn, cat_views)
        average
        views = self.alpha * self_attentions[i] + (1 - self.alpha) * views[i]
        for 1 - 5
            mv_outputs.append(torch.sigmoid(self.fusion_linear(views[i])) * views[i])
        fused_outputs = sum(mv_outputs)
        mv_outputs.append(torch.sigmoid(self.fusion_linear(views[i])) * views[i])
        fused_outputs = sum(mv_outputs)
        return fused_outputs, [(views[i] + fused_outputs) / 2 for i in range(self.num_graphs)]
        :param views:
        :return:
        """
        # run fusion by self attention
        # 5个460*64 -> 5*460*64
        cat_views = torch.stack(views, dim=0)
        self_attentions = []
        # 注意力分数计算
        for i in range(self.num_graphs):
            Q = self.self_q[i](cat_views)
            K = self.self_k[i](cat_views)
            # (3, num_nodes, 64)
            attn = F.softmax(torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.emb_dim), dim=-1)
            # (3, num_nodes, num_nodes)
            output = torch.matmul(attn, cat_views)
            self_attentions.append(output)
        self_attentions = sum(self_attentions) / self.num_graphs
        # (3, num_nodes, 64 * 2)
        for i in range(self.num_graphs):
            views[i] = self.alpha * self_attentions[i] + (1 - self.alpha) * views[i]

        # further run multi-view fusion
        mv_outputs = []
        for i in range(self.num_graphs):
            mv_outputs.append(torch.sigmoid(self.fusion_linear(views[i])) * views[i])

        fused_outputs = sum(mv_outputs)
        # next_in = [(view + fused_outputs) / 2 for view in views]
        return fused_outputs, [(views[i] + fused_outputs) / 2 for i in range(self.num_graphs)]


# 评分模型
class Scoring(nn.Module):
    def __init__(self, emb_dim, source_mask, target_mask):
        super().__init__()
        self.emb_dim = emb_dim
        self.score = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim // 2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.emb_dim // 2, self.emb_dim // 2))
        self.source_mask = source_mask
        self.target_mask = target_mask

    def forward(self, source_emb, target_emb):
        """
        求源城市评分
        注意这里求评分，是source的每一个区域对于目标城市整体
        换句话说，是形参2的每一个区域，对于形参3整体
        :param target_mask:
        :param source_mask:
        :param source_emb:
        :param target_emb:
        :return:
        """
        # target_context = tanh(self.score(target_emb[bool mask]).mean(0))
        # 对于横向的进行求平均 460*64 -> 460*32 -> 207*32 -> 纵向求平均 1*32 代表所有目标城市
        target_context = torch.tanh(
            torch.quantile(
                self.score(target_emb[self.target_mask.view(-1).bool()]),
                torch.Tensor([0.1, 0.25, 0.5, 0.75, 0.9]).to(device), dim=0).mean(0)
        )
        source_trans_emb = self.score(source_emb)
        # 460*32 * 1*32 = 462*32, 这里乘法表示1*32列表去乘460*32的每一行，逐元素
        # i.e.
        # tensor([[2, 2, 2],
        #         [1, 2, 2],
        #         [2, 2, 1]])
        # tensor([[2, 2, 2]])
        # tensor([[4, 4, 4],
        #         [2, 4, 4],
        #         [4, 4, 2]])
        source_score = (source_trans_emb * target_context).sum(1)
        # the following lines modify inner product similarity to cosine similarity
        # target_norm = target_context.pow(2).sum().pow(1/2)
        # source_norm = source_trans_emb.pow(2).sum(1).pow(1/2)
        # source_score /= source_norm
        # source_score /= target_norm
        # log(source_score)
        return F.relu(torch.tanh(source_score))[self.source_mask.view(-1).bool()]


# 最大平均误差
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


mmd = MMD_loss()


# 边类型分类器
class EdgeTypeDiscriminator(nn.Module):
    def __init__(self, num_graphs, emb_dim):
        super().__init__()
        self.num_graphs = num_graphs
        self.emb_dim = emb_dim
        self.edge_network = nn.Sequential(nn.Linear(2 * self.emb_dim, self.emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.emb_dim, self.num_graphs))

    def forward(self, src_embs, dst_embs):
        edge_vec = torch.cat([src_embs, dst_embs], dim=1)
        return self.edge_network(edge_vec)


num_gat_layers = 2
in_dim = 14
hidden_dim = 64
emb_dim = 64
num_heads = 2
mmd_w = args.mmd_w
et_w = args.et_w
ma_param = args.ma_coef

mvgat = MVGAT(len(source_graphs), num_gat_layers, in_dim, hidden_dim, emb_dim, num_heads, True).to(device)
fusion = FusionModule(len(source_graphs), emb_dim, 0.8).to(device)
scoring = Scoring(emb_dim, th_mask_source, th_mask_target).to(device)
edge_disc = EdgeTypeDiscriminator(len(source_graphs), emb_dim).to(device)
mmd = MMD_loss()
# we still need a scoring model.
# [NS, 64], [NT, 64] -> [NS]

# build model
if args.model == 'STResNet':
    net = STResNet(len(lag), 1, 3).to(device)
elif args.model == 'STNet_nobn':
    net = STNet_nobn(1, 3, th_mask_target, sigmoid_out=True).to(device)
    log(net)
elif args.model == 'STNet':
    net = STNet(1, 3, th_mask_target).to(device)
    log(net)

# net估计是预测网络
pred_optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# 图卷积，融合，边类型分类器参数单独训练
emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
emb_optimizer = optim.Adam(emb_param_list, lr=args.learning_rate, weight_decay=args.weight_decay)
# 元学习部分
meta_optimizer = optim.Adam(scoring.parameters(), lr=args.outerlr, weight_decay=args.weight_decay)
best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999


# In[13]:


def evaluate(net_, loader, spatial_mask):
    """
    评估函数，spatial_mask去掉了一些无效数据
    :param net_:
    :param loader:
    :param spatial_mask:
    :return:
    """
    net_.eval()
    with torch.no_grad():
        se = 0
        ae = 0
        valid_points = 0
        losses = []
        for it_ in loader:
            if len(it_) == 2:
                (x, y) = it_
            elif len(it_) == 4:
                _, _, x, y = it_
            x = x.to(device)
            y = y.to(device)
            lng = x.shape[2]
            lat = x.shape[3]
            out = net_(x, spatial_mask=spatial_mask.bool())
            valid_points += x.shape[0] * spatial_mask.sum().item()
            if len(out.shape) == 4:  # STResNet
                se += (((out - y) ** 2) * (spatial_mask.view(1, 1, lng, lat))).sum().item()
                ae += ((out - y).abs() * (spatial_mask.view(1, 1, lng, lat))).sum().item()
                eff_batch_size = y.shape[0]
                loss = ((out - y) ** 2).view(eff_batch_size, 1, -1)[:, :, spatial_mask.view(-1).bool()]
                losses.append(loss)
            elif len(out.shape) == 3:  # STNet
                batch_size = y.shape[0]
                lag = y.shape[1]
                y = y.view(batch_size, lag, -1)[:, :, spatial_mask.view(-1).bool()]
                # log("out", out.shape)
                # log("y", y.shape)
                se += ((out - y) ** 2).sum().item()
                ae += (out - y).abs().sum().item()
                loss = ((out - y) ** 2)
                losses.append(loss)
    return np.sqrt(se / valid_points), ae / valid_points, losses


def batch_sampler(tensor_list, batch_size):
    """
    返回抽样数据
    :param tensor_list: 元组或者list，随机抽取batchsize的数量
    :param batch_size:
    :return:
    """
    num_samples = tensor_list[0].size(0)
    idx = np.random.permutation(num_samples)[:batch_size]
    return (x[idx] for x in tensor_list)


def get_weights_bn_vars(module):
    """
    获取未命名的参数名称,以及命名参数
    :param module:
    :return:
    """
    fast_weights = OrderedDict(module.named_parameters())
    bn_vars = OrderedDict()
    for k in module.state_dict():
        if k not in fast_weights.keys():
            bn_vars[k] = module.state_dict()[k]
    return fast_weights, bn_vars


def train_epoch(net_, loader_, optimizer_, weights=None, mask=None, num_iters=None):
    """
    训练预测网络pred net网络，依据权重weights 修改loss，如
    loss = ((out - y) ** 2)
    loss = (loss * weights.view(1, 1, -1)).mean(0).sum()
    再反向传播optimizer_
    :param net_:
    :param loader_:
    :param optimizer_:
    :param weights:
    :param mask:
    :param num_iters:
    :return:
    """
    net_.train()
    epoch_loss = []
    for i, (x, y) in enumerate(loader_):
        x = x.to(device)
        y = y.to(device)
        out = net_(x, spatial_mask=mask.bool())
        if len(out.shape) == 4:  # STResNet
            eff_batch_size = y.shape[0]
            loss = ((out - y) ** 2).view(eff_batch_size, 1, -1)[:, :, mask.view(-1).bool()]
            # log("loss", loss.shape)
            if weights is not None:
                loss = (loss * weights)
                # log("weights", weights.shape)
                # log("loss * weights", loss.shape)
                loss = loss.mean(0).sum()
            else:
                loss = loss.mean(0).sum()
        elif len(out.shape) == 3:  # STNet
            eff_batch_size = y.shape[0]
            y = y.view(eff_batch_size, 1, -1)[:, :, mask.view(-1).bool()]
            loss = ((out - y) ** 2)
            if weights is not None:
                # log(loss.shape)
                # log(weights.shape)
                loss = (loss * weights.view(1, 1, -1)).mean(0).sum()
            else:
                loss = loss.mean(0).sum()
        optimizer_.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_.parameters(), max_norm=2)
        optimizer_.step()
        epoch_loss.append(loss.item())
        if num_iters is not None and num_iters == i:
            break
    return epoch_loss


# In[14]:


# 这个代码还用不到，有报错，单独拿出来，不执行
def train_rt_epoch(net_, loader_, optimizer_):
    net_.train()
    epoch_predloss = []
    epoch_rtloss = []
    epoch_loss = []
    for i, (source_x, source_y, target_x, target_y) in enumerate(loader_):
        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        source_feat, _ = net_(source_x, spatial_mask=th_mask_source.bool(), return_feat=True)
        target_feat, target_out = net_(target_x, return_feat=True)
        batch_size = target_y.shape[0]
        lag = target_y.shape[1]
        target_y = target_y.view(batch_size, lag, -1)[:, :, th_mask_target.view(-1).bool()]
        loss_pred = ((target_out - target_y) ** 2).mean(0).sum()
        matching_source_feat = source_feat[:, matching_indices, :]
        loss_rt = (((target_feat - matching_source_feat) ** 2).sum(2) * matching_weight).sum(1).mean()
        loss = loss_pred + args.rt_weight * loss_rt
        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()
        epoch_predloss.append(loss_pred.item())
        epoch_rtloss.append(loss_rt.item())
        epoch_loss.append(loss.item())
    return np.mean(epoch_predloss), np.mean(epoch_rtloss), np.mean(epoch_loss)


# In[15]:


def forward_emb(graphs_, in_feat_, od_adj_, poi_cos_):
    """
    1. 图卷积提取图特征 mvgat
    2. 融合多图特征 fusion
    3. 对于多图中的s，d，poi进行预测，并计算损失函数
    :param graphs_:
    :param in_feat_:
    :param od_adj_:
    :param poi_cos_:
    :return:
    """
    # 图注意，注意这里用了小写，指的是forward方法
    views = mvgat(graphs_, torch.Tensor(in_feat_).to(device))
    fused_emb, embs = fusion(views)
    # embs嵌入是5个图，以下找出start，destination， poi图
    s_emb = embs[-2]
    d_emb = embs[-1]
    poi_emb = embs[-3]
    # start和destination相乘求出记录预测s和d
    recons_sd = torch.matmul(s_emb, d_emb.transpose(0, 1))
    # 注意dim维度0和1分别求s和d
    pred_d = torch.log(torch.softmax(recons_sd, dim=1) + 1e-5)
    loss_d = (torch.Tensor(od_adj_).to(device) * pred_d).mean()
    pred_s = torch.log(torch.softmax(recons_sd, dim=0) + 1e-5)
    loss_s = (torch.Tensor(od_adj_).to(device) * pred_s).mean()
    # poi预测求差，loss
    poi_sim = torch.matmul(poi_emb, poi_emb.transpose(0, 1))
    loss_poi = ((poi_sim - torch.Tensor(poi_cos_).to(device)) ** 2).mean()
    loss = -loss_s - loss_d + loss_poi

    return loss, fused_emb, embs


def meta_train_epoch(s_embs, t_embs):
    """
    0. 计算source_weights，通过scoring网络
    1. 获取net 也就是预测网络的参数，分为两部分，一部分是命名参数，另一部分是非命名 fast_weights, bn_vars
    2. 从源城市抽样，计算预测值，使用net网络，从输出格式判断使用的具体网络，计算损失
    3. 通过torch.autograd.grad 计算loss对于fast_weights的梯度，并且更新fast_weights，注意2,3都是在更新net网络
    4. 同理抽样目的城市，计算梯度，更新fast_weights
    5. meta_loss = q_loss + weights_mean * args.weight_reg，计算meta loss
    6. 根据meta loss 更新scoring meta_loss.backward(inputs=list(scoring.parameters()))
    7. 循环以上，则总体更新了scoring网络，net预测网络
    :param s_embs:
    :param t_embs:
    :return:
    """
    meta_query_losses = []
    for meta_ep in range(args.outeriter):
        fast_losses = []
        fast_weights, bn_vars = get_weights_bn_vars(net)
        source_weights = scoring(s_embs, t_embs)
        # inner loop on source, pre-train with weights
        for meta_it in range(args.sinneriter):
            s_x, s_y = batch_sampler((torch.Tensor(source_x), torch.Tensor(source_y)), args.batch_size)
            s_x = s_x.to(device)
            s_y = s_y.to(device)
            pred_source = net.functional_forward(s_x, th_mask_source.bool(), fast_weights, bn_vars, bn_training=True)
            if len(pred_source.shape) == 4:  # STResNet
                loss_source = ((pred_source - s_y) ** 2).view(args.batch_size, 1, -1)[:, :,
                              th_mask_source.view(-1).bool()]
                # log(loss_source.shape)
                loss_source = (loss_source * source_weights).mean(0).sum()
            elif len(pred_source.shape) == 3:  # STNet
                s_y = s_y.view(args.batch_size, 1, -1)[:, :, th_mask_source.view(-1).bool()]
                loss_source = (((pred_source - s_y) ** 2) * source_weights.view(1, 1, -1))
                # log(loss_source.shape)
                # log(source_weights.shape)
                loss_source = loss_source.mean(0).sum()
            # size = 1 基本可以认为是标量
            fast_loss = loss_source
            fast_losses.append(fast_loss.item())  #
            # 计算输出对于输入独立的梯度，
            # fast_weights.values()。size = 22
            # grad。size=22
            # 此处对于fast_weights 进行梯度下降学习
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True)
            for name, grad in zip(fast_weights.keys(), grads):
                fast_weights[name] = fast_weights[name] - args.innerlr * grad
                # fast_weights[name].add_(grad, alpha = -args.innerlr)

        # inner loop on target, simulate fine-tune
        # 模拟微调和源训练都是在训练net预测网络，并没有提及权重和特征
        for meta_it in range(args.tinneriter):
            t_x, t_y = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size)
            t_x = t_x.to(device)
            t_y = t_y.to(device)
            pred_t = net.functional_forward(t_x, th_mask_target.bool(), fast_weights, bn_vars, bn_training=True)
            if len(pred_t.shape) == 4:  # STResNet
                loss_t = ((pred_t - t_y) ** 2).view(args.batch_size, 1, -1)[:, :, th_mask_target.view(-1).bool()]
                # log(loss_source.shape)
                loss_t = loss_t.mean(0).sum()
            elif len(pred_t.shape) == 3:  # STNet
                t_y = t_y.view(args.batch_size, 1, -1)[:, :, th_mask_target.view(-1).bool()]
                # log(t_y.shape)
                loss_t = ((pred_t - t_y) ** 2)  # .view(1, 1, -1))
                # log(loss_t.shape)
                # log(loss_source.shape)
                # log(source_weights.shape)
                loss_t = loss_t.mean(0).sum()
            fast_loss = loss_t
            fast_losses.append(fast_loss.item())  #
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True)
            for name, grad in zip(fast_weights.keys(), grads):
                fast_weights[name] = fast_weights[name] - args.innerlr * grad
                # fast_weights[name].add_(grad, alpha = -args.innerlr)

        q_losses = []
        target_iter = max(args.sinneriter, args.tinneriter)
        for k in range(3):
            # query loss
            x_q, y_q = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size)
            x_q = x_q.to(device)
            y_q = y_q.to(device)
            pred_q = net.functional_forward(x_q, th_mask_target.bool(), fast_weights, bn_vars, bn_training=True)
            if len(pred_q.shape) == 4:  # STResNet
                loss = (((pred_q - y_q) ** 2) * (th_mask_target.view(1, 1, lng_target, lat_target)))
                loss = loss.mean(0).sum()
            elif len(pred_q.shape) == 3:  # STNet
                y_q = y_q.view(args.batch_size, 1, -1)[:, :, th_mask_target.view(-1).bool()]
                loss = ((pred_q - y_q) ** 2).mean(0).sum()
            q_losses.append(loss)
        q_loss = torch.stack(q_losses).mean()
        # ** 乘方
        weights_mean = (source_weights ** 2).mean()
        # meta_loss = q_loss + weights_mean * args.weight_reg
        # 这里对于权重开始了联系
        # meta loss 只训练了scoring网络
        meta_loss = q_loss + weights_mean * args.weight_reg
        meta_optimizer.zero_grad()
        meta_loss.backward(inputs=list(scoring.parameters()))
        torch.nn.utils.clip_grad_norm_(scoring.parameters(), max_norm=2)
        meta_optimizer.step()
        meta_query_losses.append(q_loss.item())
    return np.mean(meta_query_losses)


def train_emb_epoch():
    """
    训练图网络-特征网络，融合网络，边类型分类器
    1. 通过forward_emb融合特征，计算损失，
    2. 抽样边，标签，训练边缘分类器，抽样计算MMD误差
    3. 反向传播计算
    emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
    emb_optimizer = optim.Adam(emb_param_list, lr=args.learning_rate, weight_decay=args.weight_decay)
    训练特征网络 mvgat，fusion，边缘分类器，节点MMD，在训练的同时，对于mvgat和fusion的特征进行指导，特征重新对齐分布
    :return:
    """
    # loss， 460*64， 5*460*64
    loss_source, fused_emb_s, embs_s = forward_emb(source_graphs, source_norm_poi, source_od_adj, source_poi_cos)
    loss_source2, fused_emb_s2, embs_s2 = forward_emb(source_graphs2, source_norm_poi2, source_od_adj2, source_poi_cos2)
    loss_target, fused_emb_t, embs_t = forward_emb(target_graphs, target_norm_poi, target_od_adj, target_poi_cos)

    loss_emb = loss_source + loss_target + loss_source2
    # compute domain adaptation loss
    # 随机抽样128个，计算最大平均误差
    source_ids = np.random.randint(0, np.sum(mask_source), size=(128,))
    source_ids2 = np.random.randint(0, np.sum(mask_source2), size=(128,))
    target_ids = np.random.randint(0, np.sum(mask_target), size=(128,))
    # source1 & target
    mmd_loss = mmd(fused_emb_s[th_mask_source.view(-1).bool()][source_ids, :],
                   fused_emb_t[th_mask_target.view(-1).bool()][target_ids, :])
    mmd_loss_source2_target = mmd(fused_emb_s2[th_mask_source2.view(-1).bool()][source_ids2, :],
                                  fused_emb_t[th_mask_target.view(-1).bool()][target_ids, :])
    mmd_loss_source2_source1 = mmd(fused_emb_s2[th_mask_source2.view(-1).bool()][source_ids2, :],
                                   fused_emb_s[th_mask_source.view(-1).bool()][source_ids, :])
    mmd_losses = mmd_loss + mmd_loss_source2_target + mmd_loss_source2_source1
    # 随机抽样边256
    source_batch_edges = np.random.randint(0, len(source_edges), size=(256,))
    source_batch_edges2 = np.random.randint(0, len(source_edges2), size=(256,))
    target_batch_edges = np.random.randint(0, len(target_edges), size=(256,))
    source_batch_src = torch.Tensor(source_edges[source_batch_edges, 0]).long()
    source_batch_dst = torch.Tensor(source_edges[source_batch_edges, 1]).long()
    source_emb_src = fused_emb_s[source_batch_src, :]
    source_emb_dst = fused_emb_s[source_batch_dst, :]
    source_batch_src2 = torch.Tensor(source_edges2[source_batch_edges2, 0]).long()
    source_batch_dst2 = torch.Tensor(source_edges2[source_batch_edges2, 1]).long()
    source_emb_src2 = fused_emb_s2[source_batch_src2, :]
    source_emb_dst2 = fused_emb_s2[source_batch_dst2, :]
    target_batch_src = torch.Tensor(target_edges[target_batch_edges, 0]).long()
    target_batch_dst = torch.Tensor(target_edges[target_batch_edges, 1]).long()
    target_emb_src = fused_emb_t[target_batch_src, :]
    target_emb_dst = fused_emb_t[target_batch_dst, :]
    # 源城市目的城市使用同样的边分类器
    pred_source = edge_disc.forward(source_emb_src, source_emb_dst)
    pred_source2 = edge_disc.forward(source_emb_src2, source_emb_dst2)
    pred_target = edge_disc.forward(target_emb_src, target_emb_dst)
    source_batch_labels = torch.Tensor(source_edge_labels[source_batch_edges]).to(device)
    source_batch_labels2 = torch.Tensor(source_edge_labels2[source_batch_edges2]).to(device)
    target_batch_labels = torch.Tensor(target_edge_labels[target_batch_edges]).to(device)
    # -（label*log(sigmod(pred)+0.000001)) + (1-label)*log(1-sigmod+0.000001) sum mean
    loss_et_source = -((source_batch_labels * torch.log(torch.sigmoid(pred_source) + 1e-6)) + (
            1 - source_batch_labels) * torch.log(1 - torch.sigmoid(pred_source) + 1e-6)).sum(1).mean()
    loss_et_source2 = -((source_batch_labels2 * torch.log(torch.sigmoid(pred_source2) + 1e-6)) + (
            1 - source_batch_labels2) * torch.log(1 - torch.sigmoid(pred_source2) + 1e-6)).sum(1).mean()
    loss_et_target = -((target_batch_labels * torch.log(torch.sigmoid(pred_target) + 1e-6)) + (
            1 - target_batch_labels) * torch.log(1 - torch.sigmoid(pred_target) + 1e-6)).sum(1).mean()
    loss_et = loss_et_source + loss_et_target + loss_et_source2

    emb_optimizer.zero_grad()
    # 公式11
    loss = loss_emb + mmd_w * mmd_losses + et_w * loss_et
    loss.backward()
    emb_optimizer.step()
    return loss_emb.item(), mmd_losses.item(), loss_et.item()


# In[16]:


emb_losses = []
mmd_losses = []
edge_losses = []
pretrain_emb_epoch = 80
# 预训练图数据嵌入，边类型分类，节点对齐 ——> 获得区域特征
for emb_ep in range(pretrain_emb_epoch):
    loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch()
    emb_losses.append(loss_emb_)
    mmd_losses.append(loss_mmd_)
    edge_losses.append(loss_et_)
log("[%.2fs]Pretrain embeddings for %d epochs, average emb loss %.4f, mmd loss %.4f, edge loss %.4f" % (
    time.time() - start_time, pretrain_emb_epoch, np.mean(emb_losses), np.mean(mmd_losses), np.mean(edge_losses)))
with torch.no_grad():
    views = mvgat(source_graphs, torch.Tensor(source_norm_poi).to(device))
    # 融合模块指的是把多图的特征融合
    fused_emb_s, _ = fusion(views)
    views = mvgat(source_graphs2, torch.Tensor(source_norm_poi2).to(device))
    fused_emb_s2, _ = fusion(views)
    views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
    fused_emb_t, _ = fusion(views)
# mask_source.reshape(-1) 返回的是一系列bool值，整行的含义是去除false对应的值
# reshape(-1)的含义是，不指定变换之后有多少行，将原来的tensor变成一列（default）
emb_s = fused_emb_s.cpu().numpy()[mask_source.reshape(-1)]
emb_s2 = fused_emb_s2.cpu().numpy()[mask_source2.reshape(-1)]
emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)]
logreg = LogisticRegression(max_iter=500)
"""
交叉验证，有时亦称循环估计[1] [2] [3]， 是一种统计学上将数据样本切割成较小子集的实用方法。于是可以先在一个子集上做分析，而其它子集则用来做后续对此分析的确认及验证。一开始的子集被称为训练集。而其它的子集则被称为验证集或测试集。
交叉验证的目的，是用未用来给模型作训练的新数据，测试模型的性能，以便减少诸如过拟合和选择偏差等问题，并给出模型如何在一个独立的数据集上通用化（即，一个未知的数据集，如实际问题中的数据）。
交叉验证的理论是由Seymour Geisser所开始的。它对于防范根据数据建议的测试假设是非常重要的，特别是当后续的样本是危险、成本过高或科学上不适合时去搜集。
"""
cvscore_s = cross_validate(logreg, emb_s, source_emb_label)['test_score'].mean()
cvscore_s2 = cross_validate(logreg, emb_s2, source_emb_label2)['test_score'].mean()
cvscore_t = cross_validate(logreg, emb_t, target_emb_label)['test_score'].mean()
log("[%.2fs]Pretraining embedding, source cvscore %.4f, source2 cvscore %.4f, target cvscore %.4f" % \
    (time.time() - start_time, cvscore_s, cvscore_s2, cvscore_t))
log()


# In[17]:


def score_of_two_city(s, t, smask, tmask):
    """
    计算两个城市的分数，region of s vs whole t
    :param tmask: 目标城市区域特征的的有效值向量
    :param smask: 源城市区域特征的的有效值向量
    :param s: 源城市区域特征
    :param t: 目标城市区域特征
    :return:
    """
    tt = t[tmask.view(-1).bool()].sum(0).reshape(1, 64)
    res = torch.cosine_similarity(s, tt)
    return res


with torch.no_grad():
        source_weights_s1_t = score_of_two_city(fused_emb_s, fused_emb_t, th_mask_source, th_mask_target)
        source_weights_s2_t = score_of_two_city(fused_emb_s2, fused_emb_t, th_mask_source2, th_mask_target)
        source_weights_s2_s1 = score_of_two_city(fused_emb_s2, fused_emb_s, th_mask_source2, th_mask_source)
log(source_weights_s1_t.shape)
log(source_weights_s2_t.shape)
log(source_weights_s2_s1.shape)


# In[18]:


"""
1.考虑去掉数据中除了八邻域的数据，其他地方全部设置为0，得到s2‘，组成一个图A'
2.考虑不要邻域了，直接reshape，合并
3.考虑直接在S1图的周围附加数据，（这样可能会影响S1）
"""
# res_score = torch.div(source_weights_s2_t, source_weights_s2_s1)
res_score = source_weights_s2_t
idx = source_weights_s2_t.argsort()
idx = idx[mask_source2.reshape(-1)]
log(res_score[idx[-args.topk:]])
log(idx[-args.topk:])
area_tuple = []
include_8_nearist = []
for i in idx[-args.topk:]:
    area_tuple.append((idx_1d22d(i, (lng_source2, lat_source2))))
log(area_tuple)
def yield_8_near(i, ranges):
    """
    产生i的8邻域，i，ranges都是元组或者可下标访问的元素，
    :param i:
    :param ranges:
    :return:
    """
    for k in [-1, 0, 1]:
        for p in [-1, 0, 1]:
            if k == 0 and p == 0:
                continue
            elif 0<= i[0] + k < ranges[0] and 0 <= i[1] + p < ranges[1]:
                yield i[0] + k, i[1] + p
for i in area_tuple:
    include_8_nearist.extend(list(yield_8_near(i, (lng_source2, lat_source2))))
include_8_nearist = list(set(include_8_nearist))
include_8_nearist_1d = []
for i in include_8_nearist:
    include_8_nearist_1d.append(idx_2d_2_1d(i, (lng_source2, lat_source2)))
log(include_8_nearist_1d)


# In[19]:


"""
构建A‘, 通过对角添加方式[[S1][0][0][S2]]
"""
def merge_multi_source_into_diagonal_matrix(s1, s2, s1_shape, s2_shape, date_num, check_num=1):
    """
    合并2个源城市数据到一个对角矩阵
    :param s2_shape: 排除时间维度的形状，例如（20，23）
    :param s1_shape:
    :param date_num: 两个城市时间维上的数量，比如source_data和source_data2合并时，这个值就是8764
    :param check_num:
    :param s1:
    :param s2:
    :return:
    """
    log(s1.shape)
    log(s2.shape)
    check_num = check_num if check_num < date_num else date_num
    # s1 右边扩充s2的宽度，s2左边扩充s1宽度，然后对角线填充
    zero_padding_s1_right = torch.nn.ZeroPad2d((0, abs(s2_shape[1]), 0, 0))
    zero_padding_s2_left = torch.nn.ZeroPad2d((abs(s1_shape[1]), 0, 0, 0))
    s1_zero_pad_list = []
    s2_zero_pad_list = []
    for i in range(date_num):
        s1_zero_pad_list.append(zero_padding_s1_right(torch.from_numpy(s1[i])))
        s2_zero_pad_list.append(zero_padding_s2_left(torch.from_numpy(s2[i])))
    s1_zero_pad = torch.stack(s1_zero_pad_list)
    s2_zero_pad = torch.stack(s2_zero_pad_list)
    log(s1_zero_pad.shape)
    log(s2_zero_pad.shape)
    s1_zero_pad = s1_zero_pad.numpy()
    s2_zero_pad = s2_zero_pad.numpy()
    temp = np.concatenate((s1_zero_pad, s2_zero_pad), axis=1)
    log(temp.shape)
    log("check:")
    for i in range(check_num):
        # log("left and up sum:{}".format(str(temp[0, 0: s1.shape[1]-1, 0: s1.shape[2]-1].sum())))
        # log("right and up sum:{}".format(str(temp[0, 0: s1.shape[1], s1.shape[2]:].sum())))
        # log("left and down sum:{}".format(str(temp[0, s1.shape[1]:, 0: s1.shape[2]-1].sum())))
        # log("left and up sum:{}".format(str(temp[0, s1.shape[1]:, s1.shape[2]:].sum())))
        if temp[0, 0: s1_shape[0], s1_shape[1]:].sum() == 0 and temp[0, s1_shape[0]:, 0: s1_shape[1]-1].sum() == 0:
            pass
        else:
            log("failure:{}".format(str(i)))
            return None
    log("success check {}".format(str(check_num)))
    return temp
log(source_data.shape)
log(source_data2.shape)
A_star = merge_multi_source_into_diagonal_matrix(source_data, source_data2, s1_shape=(source_data.shape[1], source_data.shape[2]), s2_shape=(source_data2.shape[1], source_data2.shape[2]),date_num=source_data.shape[0], check_num=100)
log(A_star.shape)
log()


# In[20]:


"""
构造A'的mask和th——mask
"""
A_star_mask = A_star.sum(0) > 0
"""
一下代码判断如下逻辑，
已知区域列表L中的区域（属于S2-CHI）均满足与S1不相似但是与T（NY）相似，
然后又已知其8邻域列表L'
判断L'中与CHI城市时空数据求和大于0的区域列表(L-sum>0)的交集
"""
count = 0
for i in include_8_nearist_1d:
    if A_star_mask[20:, 23:].reshape(-1)[i]:
        count += 1
log(count)
"""
找出S2 mask，对于非8邻域数据置为False
"""
ks = []
for i in range(source_data.shape[1], A_star_mask.shape[0]):
    for j in range(source_data.shape[2], A_star_mask.shape[1]):
        k = idx_2d_2_1d(((i - source_data.shape[1]), (j - source_data.shape[2])), (17, 28))
        if k not in include_8_nearist_1d:
            A_star_mask[i][j] = False
log(A_star_mask[20:, 23:].shape)
log(A_star_mask[20:, 23:].sum())

log()
log()
np.save(local_path_generate("..\\data\\mutlti", "Astar", suffix="npz"), A_star)
np.save(local_path_generate("..\\data\\mutlti", "Astar_mask", suffix="npz"), A_star_mask)

# In[21]:


A_star_lng, A_star_lat = A_star.shape[1], A_star.shape[2]
A_th_mask =  torch.Tensor(A_star_mask.reshape(1, A_star_lng, A_star_lat)).to(device)
log("%d valid regions in multi" % np.sum(A_star_mask))
# 按照百分比分配标签
A_star_emb_label = masked_percentile_label(A_star.sum(0).reshape(-1), A_star_mask.reshape(-1))
A_star, amax, amin = min_max_normalize(A_star)
log((A_star.shape, amax, amin))
A_star_train_x, A_star_train_y, A_star_val_x, A_star_val_y, A_star_test_x, A_star_test_y = split_x_y(A_star, lag)
A_star_x = np.concatenate([A_star_train_x, A_star_val_x, A_star_test_x], axis=0)
A_star_y = np.concatenate([A_star_train_y, A_star_val_y, A_star_test_y], axis=0)
log("multi split to: x %s, y %s" % (str(A_star_x.shape), str(A_star_y.shape)))
A_star_test_dataset = TensorDataset(torch.Tensor(A_star_test_x), torch.Tensor(A_star_test_y))
A_star_test_loader = DataLoader(A_star_test_dataset, batch_size=args.batch_size)
A_star_dataset = TensorDataset(torch.Tensor(A_star_x), torch.Tensor(A_star_y))
A_star_loader = DataLoader(A_star_dataset, batch_size=args.batch_size, shuffle=True)


# In[22]:


a = source_poi.reshape((lng_source, lat_source, 14))
b = source_poi2.reshape((lng_source2, lat_source2, 14))
log(a.shape)
log(b.shape)
zero_padding_s1_right = torch.nn.ZeroPad2d((0, abs(b.shape[1]), 0, 0))
zero_padding_s2_left = torch.nn.ZeroPad2d((abs(a.shape[1]), 0, 0, 0))
s1_zero_pad_list = []
s2_zero_pad_list = []
for i in range(a.shape[2]):
    s1_zero_pad_list.append(zero_padding_s1_right(torch.from_numpy(a[:, :, i])))
    s2_zero_pad_list.append(zero_padding_s2_left(torch.from_numpy(b[:, :, i])))
s1_zero_pad = torch.stack(s1_zero_pad_list, dim=2)
s2_zero_pad = torch.stack(s2_zero_pad_list, dim=2)
log(s1_zero_pad.shape)
log(s2_zero_pad.shape)
log((s1_zero_pad[:, 0: a.shape[1]].numpy() == a).sum())
log((s2_zero_pad[:, a.shape[1]:].numpy() == b).sum())

s1_zero_pad = s1_zero_pad.numpy()
s2_zero_pad = s2_zero_pad.numpy()
A_star_poi = np.concatenate((s1_zero_pad, s2_zero_pad), axis=0)
log(A_star_poi.shape)
log("check:")
for i in range(14):
    # log("left and up sum:{}".format(str(temp[0, 0: s1.shape[1]-1, 0: s1.shape[2]-1].sum())))
    # log("right and up sum:{}".format(str(temp[0, 0: s1.shape[1], s1.shape[2]:].sum())))
    # log("left and down sum:{}".format(str(temp[0, s1.shape[1]:, 0: s1.shape[2]-1].sum())))
    # log("left and up sum:{}".format(str(temp[0, s1.shape[1]:, s1.shape[2]:].sum())))
    if A_star_poi[a.shape[0]:, 0:a.shape[1], i].sum() == 0 and A_star_poi[0: a.shape[0],a.shape[1]:, i].sum() == 0:
        pass
    else:
        log("failure:{}".format(str(i)))
log("success check {}".format(str(14)))
transform = TfidfTransformer()
A_star_poi = A_star_poi.reshape(((a.shape[0] + b.shape[0]) * (a.shape[1] + b.shape[1]), 14))
# 规范正则化到（0，1）
A_star_norm_poi = np.array(transform.fit_transform(A_star_poi).todense())


# In[23]:


def merge_static_feature(s1, s2):
    zero_padding_s1_right = torch.nn.ZeroPad2d((0, ((A_star_lng * A_star_lat) - (lat_source * lng_source)), 0, 0))
    zero_padding_s1_right_down = torch.nn.ZeroPad2d((0, 0, 0, ((A_star_lng * A_star_lat) - (lat_source * lng_source) - (lat_source2 * lng_source2))))
    zero_padding_s2_left = torch.nn.ZeroPad2d(((A_star_lng * A_star_lat) - (lat_source2 * lng_source2), 0, 0, 0))
    a = zero_padding_s1_right(torch.from_numpy(s1))
    a = zero_padding_s1_right_down(a)
    b = zero_padding_s2_left(torch.from_numpy(s2))
    temp = np.concatenate((a.numpy(), b.numpy()))
    log(temp.shape)
    if temp[0:(lat_source * lng_source), (lat_source * lng_source):].sum() == 0 and temp[(A_star_lng * A_star_lat) - (lat_source2 * lng_source2): , 0: (A_star_lng * A_star_lat) - (lat_source2 * lng_source2)].sum() == 0 and temp[(lat_source * lng_source) :  (A_star_lng * A_star_lat) - (lat_source2 * lng_source2), :].sum() == 0:
        log("success")
    else:
        log("failure")
    return temp
A_star_prox_adj = add_self_loop(build_prox_graph(A_star_lng, A_star_lat))
A_star_road_adj = merge_static_feature(source_road_adj, source_s_adj2)
A_star_road_adj = add_self_loop(A_star_road_adj)
A_star_poi_adj, A_star_poi_cos = build_poi_graph(A_star_norm_poi, args.topk)
A_star_poi_adj = add_self_loop(A_star_poi_adj)
# @todo 这里暂时也是合并成对角矩阵，之后可以试试先合成矩阵，再经过函数计算
A_star_s_adj = merge_static_feature(source_s_adj, source_s_adj2)
A_star_d_adj = merge_static_feature(source_d_adj, source_d_adj2)
A_star_od_adj = merge_static_feature(source_od_adj, source_od_adj2)
A_star_s_adj = add_self_loop(A_star_s_adj)
A_star_d_adj = add_self_loop(A_star_d_adj)
A_star_od_adj = add_self_loop(A_star_od_adj)

log("A_star graphs: ")
log("prox_adj: %d nodes, %d edges" % (A_star_prox_adj.shape[0], np.sum(A_star_prox_adj)))
log("road adj: %d nodes, %d edges" % (A_star_road_adj.shape[0], np.sum(A_star_road_adj > 0)))
log("poi_adj, %d nodes, %d edges" % (A_star_poi_adj.shape[0], np.sum(A_star_poi_adj > 0)))
log("s_adj, %d nodes, %d edges" % (A_star_s_adj.shape[0], np.sum(A_star_s_adj > 0)))
log("d_adj, %d nodes, %d edges" % (A_star_d_adj.shape[0], np.sum(A_star_d_adj > 0)))
log()

A_star_graphs = adjs_to_graphs([A_star_prox_adj, A_star_road_adj, A_star_poi_adj, A_star_s_adj, A_star_d_adj])
target_graphs = adjs_to_graphs([target_prox_adj, target_road_adj, target_poi_adj, target_s_adj, target_d_adj])
for i in range(len(source_graphs)):
    A_star_graphs[i] = A_star_graphs[i].to(device)
    target_graphs[i] = target_graphs[i].to(device)


# In[24]:


A_star_edges, A_star_edge_labels = graphs_to_edge_labels(A_star_graphs)
num_gat_layers = 2
in_dim = 14
hidden_dim = 64
emb_dim = 64
num_heads = 2
mmd_w = args.mmd_w
et_w = args.et_w
ma_param = args.ma_coef

mvgat = MVGAT(len(A_star_graphs), num_gat_layers, in_dim, hidden_dim, emb_dim, num_heads, True).to(device)
fusion = FusionModule(len(A_star_graphs), emb_dim, 0.8).to(device)
scoring = Scoring(emb_dim, A_th_mask, th_mask_target).to(device)
edge_disc = EdgeTypeDiscriminator(len(A_star_graphs), emb_dim).to(device)
mmd = MMD_loss()
# we still need a scoring model.
# [NS, 64], [NT, 64] -> [NS]

# build model
if args.model == 'STResNet':
    net = STResNet(len(lag), 1, 3).to(device)
elif args.model == 'STNet_nobn':
    net = STNet_nobn(1, 3, th_mask_target, sigmoid_out=True).to(device)
    log(net)
elif args.model == 'STNet':
    net = STNet(1, 3, th_mask_target).to(device)
    log(net)
pred_optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# 图卷积，融合，边类型分类器参数单独训练
emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
emb_optimizer = optim.Adam(emb_param_list, lr=args.learning_rate, weight_decay=args.weight_decay)
# 元学习部分
meta_optimizer = optim.Adam(scoring.parameters(), lr=args.outerlr, weight_decay=args.weight_decay)
best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999
p_bar.process(5, 1, 5)


# In[30]:


"""
需要对meta_train_epoch修改一下，符合多城市要求
"""
def meta_train_epoch(s_embs, t_embs, th_mask_source, th_mask_target):
    """
    0. 计算source_weights，通过scoring网络
    1. 获取net 也就是预测网络的参数，分为两部分，一部分是命名参数，另一部分是非命名 fast_weights, bn_vars
    2. 从源城市抽样，计算预测值，使用net网络，从输出格式判断使用的具体网络，计算损失
    3. 通过torch.autograd.grad 计算loss对于fast_weights的梯度，并且更新fast_weights，注意2,3都是在更新net网络
    4. 同理抽样目的城市，计算梯度，更新fast_weights
    5. meta_loss = q_loss + weights_mean * args.weight_reg，计算meta loss
    6. 根据meta loss 更新scoring meta_loss.backward(inputs=list(scoring.parameters()))
    7. 循环以上，则总体更新了scoring网络，net预测网络
    :param s_embs:
    :param t_embs:
    :return:
    """
    meta_query_losses = []
    for meta_ep in range(args.outeriter):
        fast_losses = []
        fast_weights, bn_vars = get_weights_bn_vars(net)
        source_weights = scoring(s_embs, t_embs)
        # inner loop on source, pre-train with weights
        for meta_it in range(args.sinneriter):
            s_x, s_y = batch_sampler((torch.Tensor(A_star_x), torch.Tensor(A_star_y)), args.batch_size)
            s_x = s_x.to(device)
            s_y = s_y.to(device)
            pred_source = net.functional_forward(s_x, th_mask_source.bool(), fast_weights, bn_vars, bn_training=True)
            if len(pred_source.shape) == 4:  # STResNet
                loss_source = ((pred_source - s_y) ** 2).view(args.batch_size, 1, -1)[:, :,
                              th_mask_source.view(-1).bool()]
                # log(loss_source.shape)
                loss_source = (loss_source * source_weights).mean(0).sum()
            elif len(pred_source.shape) == 3:  # STNet
                s_y = s_y.view(args.batch_size, 1, -1)[:, :, th_mask_source.view(-1).bool()]
                loss_source = (((pred_source - s_y) ** 2) * source_weights.view(1, 1, -1))
                # log(loss_source.shape)
                # log(source_weights.shape)
                loss_source = loss_source.mean(0).sum()
            # size = 1 基本可以认为是标量
            fast_loss = loss_source
            fast_losses.append(fast_loss.item())  #
            # 计算输出对于输入独立的梯度，
            # fast_weights.values()。size = 22
            # grad。size=22
            # 此处对于fast_weights 进行梯度下降学习
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True)
            for name, grad in zip(fast_weights.keys(), grads):
                fast_weights[name] = fast_weights[name] - args.innerlr * grad
                # fast_weights[name].add_(grad, alpha = -args.innerlr)

        # inner loop on target, simulate fine-tune
        # 模拟微调和源训练都是在训练net预测网络，并没有提及权重和特征
        for meta_it in range(args.tinneriter):
            t_x, t_y = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size)
            t_x = t_x.to(device)
            t_y = t_y.to(device)
            pred_t = net.functional_forward(t_x, th_mask_target.bool(), fast_weights, bn_vars, bn_training=True)
            if len(pred_t.shape) == 4:  # STResNet
                loss_t = ((pred_t - t_y) ** 2).view(args.batch_size, 1, -1)[:, :, th_mask_target.view(-1).bool()]
                # log(loss_source.shape)
                loss_t = loss_t.mean(0).sum()
            elif len(pred_t.shape) == 3:  # STNet
                t_y = t_y.view(args.batch_size, 1, -1)[:, :, th_mask_target.view(-1).bool()]
                # log(t_y.shape)
                loss_t = ((pred_t - t_y) ** 2)  # .view(1, 1, -1))
                # log(loss_t.shape)
                # log(loss_source.shape)
                # log(source_weights.shape)
                loss_t = loss_t.mean(0).sum()
            fast_loss = loss_t
            fast_losses.append(fast_loss.item())  #
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True)
            for name, grad in zip(fast_weights.keys(), grads):
                fast_weights[name] = fast_weights[name] - args.innerlr * grad
                # fast_weights[name].add_(grad, alpha = -args.innerlr)

        q_losses = []
        target_iter = max(args.sinneriter, args.tinneriter)
        for k in range(3):
            # query loss
            x_q, y_q = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size)
            x_q = x_q.to(device)
            y_q = y_q.to(device)
            pred_q = net.functional_forward(x_q, th_mask_target.bool(), fast_weights, bn_vars, bn_training=True)
            if len(pred_q.shape) == 4:  # STResNet
                loss = (((pred_q - y_q) ** 2) * (th_mask_target.view(1, 1, lng_target, lat_target)))
                loss = loss.mean(0).sum()
            elif len(pred_q.shape) == 3:  # STNet
                y_q = y_q.view(args.batch_size, 1, -1)[:, :, th_mask_target.view(-1).bool()]
                loss = ((pred_q - y_q) ** 2).mean(0).sum()
            q_losses.append(loss)
        q_loss = torch.stack(q_losses).mean()
        # ** 乘方
        weights_mean = (source_weights ** 2).mean()
        # meta_loss = q_loss + weights_mean * args.weight_reg
        # 这里对于权重开始了联系
        # meta loss 只训练了scoring网络
        meta_loss = q_loss + weights_mean * args.weight_reg
        meta_optimizer.zero_grad()
        meta_loss.backward(inputs=list(scoring.parameters()))
        torch.nn.utils.clip_grad_norm_(scoring.parameters(), max_norm=2)
        meta_optimizer.step()
        meta_query_losses.append(q_loss.item())
    return np.mean(meta_query_losses)

def train_emb_epoch(source_graphs, source_norm_poi, source_od_adj, source_poi_cos, target_graphs, target_norm_poi, target_od_adj, target_poi_cos):
    """
    训练图网络-特征网络，融合网络，边类型分类器
    1. 通过forward_emb融合特征，计算损失，
    2. 抽样边，标签，训练边缘分类器，抽样计算MMD误差
    3. 反向传播计算
    emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
    emb_optimizer = optim.Adam(emb_param_list, lr=args.learning_rate, weight_decay=args.weight_decay)
    训练特征网络 mvgat，fusion，边缘分类器，节点MMD，在训练的同时，对于mvgat和fusion的特征进行指导，特征重新对齐分布
    :return:
    """
    # loss， 460*64， 5*460*64
    loss_source, fused_emb_s, embs_s = forward_emb(source_graphs, source_norm_poi, source_od_adj, source_poi_cos)
    loss_target, fused_emb_t, embs_t = forward_emb(target_graphs, target_norm_poi, target_od_adj, target_poi_cos)
    loss_emb = loss_source + loss_target
    # compute domain adaptation loss
    # 随机抽样128个，计算最大平均误差
    source_ids = np.random.randint(0, np.sum(A_star_mask), size=(128,))
    target_ids = np.random.randint(0, np.sum(mask_target), size=(128,))
    mmd_loss = mmd(fused_emb_s[A_th_mask.view(-1).bool()][source_ids, :],
                   fused_emb_t[th_mask_target.view(-1).bool()][target_ids, :])
    # 随机抽样边256
    source_batch_edges = np.random.randint(0, len(source_edges), size=(256,))
    target_batch_edges = np.random.randint(0, len(target_edges), size=(256,))
    source_batch_src = torch.Tensor(source_edges[source_batch_edges, 0]).long()
    source_batch_dst = torch.Tensor(source_edges[source_batch_edges, 1]).long()
    source_emb_src = fused_emb_s[source_batch_src, :]
    source_emb_dst = fused_emb_s[source_batch_dst, :]
    target_batch_src = torch.Tensor(target_edges[target_batch_edges, 0]).long()
    target_batch_dst = torch.Tensor(target_edges[target_batch_edges, 1]).long()
    target_emb_src = fused_emb_t[target_batch_src, :]
    target_emb_dst = fused_emb_t[target_batch_dst, :]
    # 源城市目的城市使用同样的边分类器
    pred_source = edge_disc.forward(source_emb_src, source_emb_dst)
    pred_target = edge_disc.forward(target_emb_src, target_emb_dst)
    source_batch_labels = torch.Tensor(source_edge_labels[source_batch_edges]).to(device)
    target_batch_labels = torch.Tensor(target_edge_labels[target_batch_edges]).to(device)
    # -（label*log(sigmod(pred)+0.000001)) + (1-label)*log(1-sigmod+0.000001) sum mean
    loss_et_source = -((source_batch_labels * torch.log(torch.sigmoid(pred_source) + 1e-6)) + (
            1 - source_batch_labels) * torch.log(1 - torch.sigmoid(pred_source) + 1e-6)).sum(1).mean()
    loss_et_target = -((target_batch_labels * torch.log(torch.sigmoid(pred_target) + 1e-6)) + (
            1 - target_batch_labels) * torch.log(1 - torch.sigmoid(pred_target) + 1e-6)).sum(1).mean()
    loss_et = loss_et_source + loss_et_target

    emb_optimizer.zero_grad()
    # 公式11
    loss = loss_emb + mmd_w * mmd_loss + et_w * loss_et
    loss.backward()
    emb_optimizer.step()
    return loss_emb.item(), mmd_loss.item(), loss_et.item()
emb_losses = []
mmd_losses = []
edge_losses = []
pretrain_emb_epoch = 80
# 预训练图数据嵌入，边类型分类，节点对齐 ——> 获得区域特征
for emb_ep in range(pretrain_emb_epoch):
    loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch(
        source_graphs=A_star_graphs, source_norm_poi=A_star_norm_poi, source_od_adj=A_star_od_adj, source_poi_cos=A_star_poi_cos,
        target_graphs=target_graphs, target_norm_poi=target_norm_poi, target_od_adj=target_od_adj, target_poi_cos=target_poi_cos
    )
    emb_losses.append(loss_emb_)
    mmd_losses.append(loss_mmd_)
    edge_losses.append(loss_et_)
log("[%.2fs]Pretrain embeddings for %d epochs, average emb loss %.4f, mmd loss %.4f, edge loss %.4f" % (
    time.time() - start_time, pretrain_emb_epoch, np.mean(emb_losses), np.mean(mmd_losses), np.mean(edge_losses)))
with torch.no_grad():
    views = mvgat(A_star_graphs, torch.Tensor(A_star_norm_poi).to(device))
    # 融合模块指的是把多图的特征融合
    fused_emb_A_star, _ = fusion(views)
    views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
    fused_emb_t, _ = fusion(views)


# In[31]:


emb_s = fused_emb_A_star.cpu().numpy()[A_star_mask.reshape(-1)]
emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)]
logreg = LogisticRegression(max_iter=500)
"""
交叉验证，有时亦称循环估计[1] [2] [3]， 是一种统计学上将数据样本切割成较小子集的实用方法。于是可以先在一个子集上做分析，而其它子集则用来做后续对此分析的确认及验证。一开始的子集被称为训练集。而其它的子集则被称为验证集或测试集。
交叉验证的目的，是用未用来给模型作训练的新数据，测试模型的性能，以便减少诸如过拟合和选择偏差等问题，并给出模型如何在一个独立的数据集上通用化（即，一个未知的数据集，如实际问题中的数据）。
交叉验证的理论是由Seymour Geisser所开始的。它对于防范根据数据建议的测试假设是非常重要的，特别是当后续的样本是危险、成本过高或科学上不适合时去搜集。
"""
cvscore_s = cross_validate(logreg, emb_s, A_star_emb_label)['test_score'].mean()
cvscore_t = cross_validate(logreg, emb_t, target_emb_label)['test_score'].mean()
log("[%.2fs]Pretraining embedding, source cvscore %.4f, target cvscore %.4f" % \
    (time.time() - start_time, cvscore_s, cvscore_t))
log()


# In[32]:


# 后期要用这个参数
source_weights_ma_list = []
source_weight_list = []
p_bar = process_bar(final_prompt="训练完成", unit="epoch")
p_bar.process(0, 1, num_epochs + num_tuine_epochs)
writer = SummaryWriter("log_{}".format(get_timestamp(split="-")))
for ep in range(num_epochs):
    net.train()
    mvgat.train()
    fusion.train()
    scoring.train()

    # train embeddings
    emb_losses = []
    mmd_losses = []
    edge_losses = []
    for emb_ep in range(5):
        loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch(
        source_graphs=A_star_graphs, source_norm_poi=A_star_norm_poi, source_od_adj=A_star_od_adj, source_poi_cos=A_star_poi_cos,
        target_graphs=target_graphs, target_norm_poi=target_norm_poi, target_od_adj=target_od_adj, target_poi_cos=target_poi_cos
    )
        emb_losses.append(loss_emb_)
        mmd_losses.append(loss_mmd_)
        edge_losses.append(loss_et_)
    # evaluate embeddings
    with torch.no_grad():
        # mvgat 是把邻接矩阵转换成tensor，大小是城市的长宽之积 * 64（demb）也就是定义的区域特征向量的维度
        views = mvgat(A_star_graphs, torch.Tensor(A_star_norm_poi).to(device))
        fused_emb_s, _ = fusion(views)
        views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
        fused_emb_t, _ = fusion(views)
    if ep % 2 == 0:
        """
        每两个epoch显示一些数据
        """
        emb_s = fused_emb_s.cpu().numpy()[A_star_mask.reshape(-1)]
        emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)]
        mix_embs = np.concatenate([emb_s, emb_t], axis=0)
        mix_labels = np.concatenate([A_star_emb_label, target_emb_label])
        logreg = LogisticRegression(max_iter=500)
        cvscore_s = cross_validate(logreg, emb_s, A_star_emb_label)['test_score'].mean()
        cvscore_t = cross_validate(logreg, emb_t, target_emb_label)['test_score'].mean()
        cvscore_mix = cross_validate(logreg, mix_embs, mix_labels)['test_score'].mean()
        log(
            "[%.2fs]Epoch %d, embedding loss %.4f, mmd loss %.4f, edge loss %.4f, source cvscore %.4f, target cvscore %.4f, mixcvscore %.4f" % \
            (time.time() - start_time, ep, np.mean(emb_losses), np.mean(mmd_losses), np.mean(edge_losses), cvscore_s,
             cvscore_t, cvscore_mix))
    if ep == num_epochs - 1:
        """
        最后一个epoch，
        """
        emb_s = fused_emb_s.cpu().numpy()[mask_source.reshape(-1)]
        emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)]
        # np.save("%s.npy" % args.scity, arr = emb_s)
        # np.save("%s.npy" % args.tcity, arr = emb_t)
        with torch.no_grad():
            trans_emb_s = scoring.score(fused_emb_s)
            trans_emb_t = scoring.score(fused_emb_t)
        # np.save("%s_trans.npy" % args.scity, arr = trans_emb_s.cpu().numpy()[mask_source.reshape(-1)])
        # np.save("%s_trans.npy" % args.tcity, arr = trans_emb_t.cpu().numpy()[mask_target.reshape(-1)])

    # meta train scorings
    avg_q_loss = meta_train_epoch(fused_emb_s, fused_emb_t, th_mask_source=A_th_mask, th_mask_target=th_mask_target)
    with torch.no_grad():
        source_weights = scoring(fused_emb_s, fused_emb_t)
        source_weight_list.append(list(source_weights.cpu().numpy()))

    # For debug: use fixed weightings.
    # with torch.no_grad():
    #     source_weights_ = scoring(fused_emb_s, fused_emb_t)
    # avg_q_loss = 0
    # source_weights = torch.ones_like(source_weights_)

    # implement a moving average
    if ep == 0:
        source_weights_ma = torch.ones_like(source_weights, device=device, requires_grad=False)
    source_weights_ma = ma_param * source_weights_ma + (1 - ma_param) * source_weights
    source_weights_ma_list.append(list(source_weights_ma.cpu().numpy()))
    # train network on source
    # 有了参数lambda rs，训练net网络
    source_loss = train_epoch(net, A_star_loader, pred_optimizer, weights=source_weights_ma, mask=A_th_mask,
                              num_iters=args.pretrain_iter)
    avg_source_loss = np.mean(source_loss)
    avg_target_loss = evaluate(net, target_train_loader, spatial_mask=th_mask_target)[0]
    log(
        "[%.2fs]Epoch %d, average meta query loss %.4f, source weight mean %.4f, var %.6f, source loss %.4f, target_loss %.4f" % \
        (time.time() - start_time, ep, avg_q_loss, source_weights_ma.mean().item(), torch.var(source_weights_ma).item(),
         avg_source_loss, avg_target_loss))
    writer.add_scalar("average meta query loss", avg_q_loss, ep)
    writer.add_scalar("source weight mean", source_weights_ma.mean().item(), ep)
    writer.add_scalar("var", torch.var(source_weights_ma).item(), ep)
    writer.add_scalar("avg_source_loss", avg_source_loss, ep)
    writer.add_scalar("avg_target_loss", avg_target_loss, ep)
    log(torch.var(source_weights).item())
    log(source_weights.mean().item())
    if source_weights_ma.mean() < 0.005:
        # stop pre-training
        break
    net.eval()
    rmse_val, mae_val, val_losses = evaluate(net, target_val_loader, spatial_mask=th_mask_target)
    rmse_s_val, mae_s_val, test_losses = evaluate(net, A_star_loader, spatial_mask=A_th_mask)
    log(
        "Epoch %d, source validation rmse %.4f, mae %.4f" % (ep, rmse_s_val * (smax - smin), mae_s_val * (smax - smin)))
    log("Epoch %d, target validation rmse %.4f, mae %.4f" % (
        ep, rmse_val * (max_val - min_val), mae_val * (max_val - min_val)))
    log()
    writer.add_scalar("source validation rmse", rmse_s_val * (smax - smin), ep)
    writer.add_scalar("source validation mse", mae_s_val * (smax - smin), ep)
    writer.add_scalar("target validation rmse_val", rmse_val * (max_val - min_val), ep)
    writer.add_scalar("target validation mae_val", mae_val * (max_val - min_val), ep)
    sums = 0
    for i in range(len(val_losses)):
        sums = sums + val_losses[i].mean(0).sum().item()
    writer.add_scalar("source train val loss", sums, ep)
    sums = 0
    for i in range(len(test_losses)):
        sums = sums + test_losses[i].mean(0).sum().item()
    writer.add_scalar("source train test loss", sums, ep)
    p_bar.process(0, 1, num_epochs + num_tuine_epochs)
save_obj(source_weights_ma_list, path="source_weights_ma_list_{}.list".format(scity))
save_obj(source_weight_list, path="source_weight_list_{}.list".format(scity))
for ep in range(num_epochs, num_tuine_epochs + num_epochs):
    # fine-tuning
    net.train()
    avg_loss = train_epoch(net, target_train_loader, pred_optimizer, mask=th_mask_target)
    log('[%.2fs]Epoch %d, target pred loss %.4f' % (time.time() - start_time, ep, np.mean(avg_loss)))
    writer.add_scalar("target pred loss", np.mean(avg_loss), ep - num_epochs)
    net.eval()
    rmse_val, mae_val, val_losses = evaluate(net, target_val_loader, spatial_mask=th_mask_target)
    rmse_test, mae_test, test_losses = evaluate(net, target_test_loader, spatial_mask=th_mask_target)
    sums = 0
    for i in range(len(val_losses)):
        sums = sums + val_losses[i].mean(0).sum().item()
    writer.add_scalar("target train val loss", sums, ep)
    sums = 0
    for i in range(len(test_losses)):
        sums = sums + test_losses[i].mean(0).sum().item()
    writer.add_scalar("target train test loss", sums, ep)
    if rmse_val < best_val_rmse:
        best_val_rmse = rmse_val
        best_test_rmse = rmse_test
        best_test_mae = mae_test
        log("Update best test...")
    log("validation rmse %.4f, mae %.4f" % (rmse_val * (max_val - min_val), mae_val * (max_val - min_val)))
    log("test rmse %.4f, mae %.4f" % (rmse_test * (max_val - min_val), mae_test * (max_val - min_val)))
    writer.add_scalar("validation rmse", rmse_val * (max_val - min_val), ep - num_epochs)
    writer.add_scalar("validation mae", mae_val * (max_val - min_val), ep - num_epochs)
    writer.add_scalar("test rmse", rmse_test * (max_val - min_val), ep - num_epochs)
    writer.add_scalar("test mae", mae_test * (max_val - min_val), ep - num_epochs)
    log()
    p_bar.process(0, 1, num_epochs + num_tuine_epochs)

log("Best test rmse %.4f, mae %.4f" % (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val)))

