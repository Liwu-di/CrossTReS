# -*- coding: utf-8 -*-
# @Time    : 2023/11/7 13:13
# @Author  : 银尘
# @FileName: deweight.py
# @Software: PyCharm
# @Email   : liwudi@liwudi.fun
# @Info    : why create this file
import argparse
import ast
from collections import OrderedDict

import PaperCrawlerUtil.common_util
import tensorboard.summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from PaperCrawlerUtil.common_util import *
from PaperCrawlerUtil.constant import *
from PaperCrawlerUtil.crawler_util import *
from PaperCrawlerUtil.research_util import *
from dgl.nn import GATConv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from params import *
from model import *
from utils import *

basic_config(logs_style=LOG_STYLE_PRINT)
p_bar = process_bar(final_prompt="初始化准备完成", unit="part")
p_bar.process(0, 1, 5)
# This file implements the full version of using region embeddings to select good source data.
args = params()
long_term_save = {}
args = params()
long_term_save["args"] = args.__str__()
if args.c != "default":
    c = ast.literal_eval(args.c)
    record = ResearchRecord(**c)
    record_id = record.insert(__file__, get_timestamp(), args.__str__())
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
tcity = args.tcity
datatype = args.datatype
num_epochs = args.num_epochs
num_tuine_epochs = args.num_tuine_epochs
start_time = time.time()
log("Running CrossTReS, from %s to %s, %s %s experiments, with %d days of data, on %s model" % \
    (scity, tcity, dataname, datatype, args.data_amount, args.model))

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
# (8784, 20, 23)
source_data = np.load("../data/%s/%s%s_%s.npy" % (scity, dataname, scity, datatype))
if args.cut_data != 0:
    source_data = source_data[0: args.cut_data, :, :]
# (20, 23)
lng_source, lat_source = source_data.shape[1], source_data.shape[2]
mask_source = source_data.sum(0) > 0
# mask -> th_mask = (20, 23) -> (1, 20, 23)
th_mask_source = torch.Tensor(mask_source.reshape(1, lng_source, lat_source)).to(device)
log("%d valid regions in source" % np.sum(mask_source))
# 按照百分比分配标签
source_emb_label = masked_percentile_label(source_data.sum(0).reshape(-1), mask_source.reshape(-1))
p_bar.process(3, 1, 5)
lag = [-6, -5, -4, -3, -2, -1]
threshold = args.cut_data
source_data, smax, smin = min_max_normalize(source_data)
target_data, max_val, min_val = min_max_normalize(target_data)
source_data = source_data[0: threshold, :, :]

# [(5898, 6, 20, 23), (5898, 1, 20, 23), (1440, 6, 20, 23), (1440, 1, 20, 23), (1440, 6, 20, 23), (1440, 1, 20, 23)]
# 第一维是数量，第二维是每条数据中的数量
source_train_x, source_train_y, source_val_x, source_val_y, source_test_x, source_test_y = split_x_y(source_data, lag, val_num=int(source_data.shape[0] / 6), test_num=int(source_data.shape[0] / 6))
# we concatenate all source data
# (8778, 6, 20, 23)
source_x = np.concatenate([source_train_x, source_val_x, source_test_x], axis=0)
# (8778, 1, 20, 23)
source_y = np.concatenate([source_train_y, source_val_y, source_test_y], axis=0)
target_train_x, target_train_y, target_val_x, target_val_y, target_test_x, target_test_y = split_x_y(target_data, lag)
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

log("Target split to: train_x %s, train_y %s" % (str(target_train_x.shape), str(target_train_y.shape)))
log("val_x %s, val_y %s" % (str(target_val_x.shape), str(target_val_y.shape)))
log("test_x %s, test_y %s" % (str(target_test_x.shape), str(target_test_y.shape)))

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
p_bar.process(4, 1, 5)
# Load auxiliary data: poi data
# (20, 23, 14)
source_poi = np.load("../data/%s/%s_poi.npy" % (scity, scity))
target_poi = np.load("../data/%s/%s_poi.npy" % (tcity, tcity))
# (460, 14)
source_poi = source_poi.reshape(lng_source * lat_source, -1)  # regions * classes
target_poi = target_poi.reshape(lng_target * lat_target, -1)  # regions * classes
transform = TfidfTransformer()
# 规范正则化到（0，1）
source_norm_poi = np.array(transform.fit_transform(source_poi).todense())
transform = TfidfTransformer()
target_norm_poi = np.array(transform.fit_transform(target_poi).todense())

# Build graphs
# add_self_loop 增加一个自循环，对角线的值=1
source_prox_adj = add_self_loop(build_prox_graph(lng_source, lat_source))
target_prox_adj = add_self_loop(build_prox_graph(lng_target, lat_target))
source_road_adj = add_self_loop(build_road_graph(scity, lng_source, lat_source))
target_road_adj = add_self_loop(build_road_graph(tcity, lng_target, lat_target))
source_poi_adj, source_poi_cos = build_poi_graph(source_norm_poi, args.topk)
target_poi_adj, target_poi_cos = build_poi_graph(target_norm_poi, args.topk)
source_poi_adj = add_self_loop(source_poi_adj)
target_poi_adj = add_self_loop(target_poi_adj)
source_s_adj, source_d_adj, source_od_adj = build_source_dest_graph(scity, dataname, lng_source, lat_source, args.topk)
target_s_adj, target_d_adj, target_od_adj = build_source_dest_graph(tcity, dataname, lng_target, lat_target, args.topk)
source_s_adj = add_self_loop(source_s_adj)
source_t_adj = add_self_loop(source_d_adj)
source_od_adj = add_self_loop(source_od_adj)
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
log("Target graphs:")
log("prox_adj: %d nodes, %d edges" % (target_prox_adj.shape[0], np.sum(target_prox_adj)))
log("road adj: %d nodes, %d edges" % (target_road_adj.shape[0], np.sum(target_road_adj > 0)))
log("poi_adj, %d nodes, %d edges" % (target_poi_adj.shape[0], np.sum(target_poi_adj > 0)))
log("s_adj, %d nodes, %d edges" % (target_s_adj.shape[0], np.sum(target_s_adj > 0)))
log("d_adj, %d nodes, %d edges" % (target_d_adj.shape[0], np.sum(target_d_adj > 0)))
log()
source_graphs = adjs_to_graphs([source_prox_adj, source_road_adj, source_poi_adj, source_s_adj, source_d_adj])
target_graphs = adjs_to_graphs([target_prox_adj, target_road_adj, target_poi_adj, target_s_adj, target_d_adj])
for i in range(len(source_graphs)):
    source_graphs[i] = source_graphs[i].to(device)
    target_graphs[i] = target_graphs[i].to(device)


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
target_edges, target_edge_labels = graphs_to_edge_labels(target_graphs)
p_bar.process(5, 1, 5)


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
        :param source_emb:
        :param target_emb:
        :return:
        """
        # target_context = tanh(self.score(target_emb[bool mask]).mean(0))
        # 对于横向的进行求平均 460*64 -> 460*32 -> 207*32 -> 纵向求平均 1*32 代表所有目标城市
        target_context = torch.tanh(self.score(target_emb[self.target_mask.view(-1).bool()]).mean(0))
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
pred_optimizer = optim.Adam(net.parameters(), lr=args.pred_lr, weight_decay=args.weight_decay)
# 图卷积，融合，边类型分类器参数单独训练
emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
emb_optimizer = optim.Adam(emb_param_list, lr=args.learning_rate, weight_decay=args.weight_decay)
# 元学习部分
meta_optimizer = optim.Adam(scoring.parameters(), lr=args.outerlr, weight_decay=args.weight_decay)
best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999
best_test_mape = 999


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
        mape = 0
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
                with np.errstate(divide='ignore', invalid='ignore'):
                    ape = (out - y).abs() / y
                    ape = ape.cpu().numpy().flatten()
                    ape[~ np.isfinite(ape)] = 0  # 对 -inf, inf, NaN进行修正，置为0
                    mape += ape.sum().item()
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
                with np.errstate(divide='ignore', invalid='ignore'):
                    ape = (out - y).abs() / y
                    ape = ape.cpu().numpy().flatten()
                    ape[~ np.isfinite(ape)] = 0  # 对 -inf, inf, NaN进行修正，置为0
                    mape += ape.sum().item()
    return np.sqrt(se / valid_points), ae / valid_points, losses, mape / valid_points

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
            s_x, s_y = batch_sampler((torch.Tensor(source_x), torch.Tensor(source_y)), args.meta_batch_size)
            s_x = s_x.to(device)
            s_y = s_y.to(device)
            pred_source = net.functional_forward(s_x, th_mask_source.bool(), fast_weights, bn_vars, bn_training=True)
            if len(pred_source.shape) == 4:  # STResNet
                loss_source = ((pred_source - s_y) ** 2).view(args.batch_size, 1, -1)[:, :,
                              th_mask_source.view(-1).bool()]
                # log(loss_source.shape)
                loss_source = (loss_source * source_weights).mean(0).sum()
            elif len(pred_source.shape) == 3:  # STNet
                s_y = s_y.view(args.meta_batch_size, 1, -1)[:, :, th_mask_source.view(-1).bool()]
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
    loss_target, fused_emb_t, embs_t = forward_emb(target_graphs, target_norm_poi, target_od_adj, target_poi_cos)
    loss_emb = loss_source + loss_target
    # compute domain adaptation loss
    # 随机抽样128个，计算最大平均误差
    source_ids = np.random.randint(0, np.sum(mask_source), size=(128,))
    target_ids = np.random.randint(0, np.sum(mask_target), size=(128,))
    mmd_loss = mmd(fused_emb_s[th_mask_source.view(-1).bool()][source_ids, :],
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


# 后期要用这个参数
source_weights_ma_list = []
source_weight_list = []
train_emb_losses = []
average_meta_query_loss = []
source_validation_rmse = []
target_validation_rmse = []
source_validation_mae = []
target_validation_mae = []
train_target_val_loss = []
train_source_val_loss = []
target_pred_loss = []
target_train_val_loss = []
target_train_test_loss = []
validation_rmse = []
validation_mae = []
test_rmse = []
test_mae = []
p_bar = process_bar(final_prompt="训练完成", unit="epoch")
p_bar.process(0, 1, num_epochs + num_tuine_epochs)
writer = SummaryWriter("log-{}-batch-{}-name-{}-type-{}-model-{}-amount-{}-topk-{}-time-{}".
                       format("单城市{}-{}".format(args.scity, args.tcity), args.batch_size, args.dataname,
                              args.datatype, args.model, args.data_amount, args.topk, get_timestamp(split="-")))
num_tuine_epochs = 5
for ep in range(num_epochs):
    net.train()
    mvgat.train()
    fusion.train()
    scoring.train()

    # train embeddings
    emb_losses = []
    mmd_losses = []
    edge_losses = []
    source_loss = train_epoch(net, source_loader, pred_optimizer, weights=None, mask=th_mask_source,
                              num_iters=args.pretrain_iter)
    avg_source_loss = np.mean(source_loss)
    avg_target_loss = evaluate(net, target_train_loader, spatial_mask=th_mask_target)[0]
    net.eval()
    rmse_val, mae_val, val_losses, _ = evaluate(net, target_val_loader, spatial_mask=th_mask_target)
    rmse_s_val, mae_s_val, test_losses, _ = evaluate(net, source_loader, spatial_mask=th_mask_source)
    log(rmse_s_val, mae_s_val)
    p_bar.process(0, 1, num_epochs + num_tuine_epochs)

for ep in range(num_epochs, num_tuine_epochs + num_epochs):
    net.train()
    avg_loss = train_epoch(net, target_train_loader, pred_optimizer, mask=th_mask_target)
    log('[%.2fs]Epoch %d, target pred loss %.4f' % (time.time() - start_time, ep, np.mean(avg_loss)))
    net.eval()
    rmse_val, mae_val, val_losses, target_val_ape = evaluate(net, target_val_loader, spatial_mask=th_mask_target)
    rmse_test, mae_test, test_losses, target_test_ape = evaluate(net, target_test_loader, spatial_mask=th_mask_target)
    sums = 0
    for i in range(len(val_losses)):
        sums = sums + val_losses[i].mean(0).sum().item()
    writer.add_scalar("target train val loss", sums, ep - num_epochs)
    target_train_val_loss.append(sums)
    sums = 0
    for i in range(len(test_losses)):
        sums = sums + test_losses[i].mean(0).sum().item()
    writer.add_scalar("target train test loss", sums, ep - num_epochs)
    target_train_test_loss.append(sums)
    if rmse_val < best_val_rmse:
        best_val_rmse = rmse_val
        best_test_rmse = rmse_test
        best_test_mae = mae_test
        best_test_mape = target_test_ape
        log("Update best test...")
    log("validation rmse %.4f, mae %.4f, mape %.4f" % (rmse_val * (max_val - min_val), mae_val * (max_val - min_val), target_val_ape * 100))
    log("test rmse %.4f, mae %.4f, mape %.4f" % (rmse_test * (max_val - min_val), mae_test * (max_val - min_val), target_test_ape * 100))
    p_bar.process(0, 1, num_epochs + num_tuine_epochs)

log("Best test rmse %.4f, mae %.4f, mape %.4f" % (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val), best_test_mape * 100))
root_dir = local_path_generate(
    "./model/{}".format(
        "{}-batch-{}-{}-{}-{}-amount-{}-topk-{}-time-{}".format(
            "单城市{}-{}".format(args.scity, args.tcity),
            args.batch_size, args.dataname, args.datatype, args.model, args.data_amount,
            args.topk, get_timestamp(split="-")
        )
    ), create_folder_only=True)
torch.save(net, root_dir + "/net.pth")
torch.save(mvgat, root_dir + "/mvgat.pth")
torch.save(fusion, root_dir + "/fusion.pth")
torch.save(scoring, root_dir + "/scoring.pth")
torch.save(edge_disc, root_dir + "/edge_disc.pth")
if args.c != "default":
    record.update(record_id, get_timestamp(),
                  "Best test rmse %.4f, mae %.4f" %
                  (best_test_rmse * (max_val - min_val), best_test_mae * 100))
save_obj(long_term_save,
         local_path_generate("experiment_data",
                             "data_{}.collection".format(
                                 "{}-batch-{}-{}-{}-{}-amount-{}-time-{}".format(
                                     "单城市{}-{}".format(args.scity, args.tcity),
                                     args.batch_size, args.dataname, args.datatype, args.model, args.data_amount,
                                     get_timestamp(split="-")
                                 )
                             )
                             )
         )
if args.c != "default":
    record.update(record_id, get_timestamp(),
                  "%.4f,%.4f, %.4f" %
                  (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val), best_test_mape * 100))
