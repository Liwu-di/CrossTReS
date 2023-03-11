# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 15:17
# @Author  : 银尘
# @FileName: funcs.py
# @Software: PyCharm
# @Email   ：liwudi@liwudi.fun
from typing import Dict

import numpy as np
import argparse
from collections import OrderedDict

import numpy as np
import scipy.stats
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
from dtaidistance import dtw
from scipy.stats import wasserstein_distance
from scipy.spatial import distance


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


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class Grad(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.constant
        return grad_output, None

    def grad(x, constant):
        return Grad.apply(x, constant)


class city_adversarial_classify(nn.Module):
    def __init__(self, num_class, encode_dim):
        super(city_adversarial_classify, self).__init__()

        self.num_class = num_class
        self.encode_dim = encode_dim

        self.fc1 = nn.Linear(self.encode_dim, 16)
        self.fc2 = nn.Linear(16, num_class)

    def forward(self, input, constant, Reverse):
        if Reverse:
            input = GradReverse.grad_reverse(input, constant)
        else:
            input = Grad.grad(input, constant)
        logits = torch.tanh(self.fc1(input))
        logits = self.fc2(logits)
        logits = F.log_softmax(logits, 1)

        return logits


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


def load_process_data(args, p_bar):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    dataname = args.dataname
    scity = args.scity
    scity2 = args.scity2
    tcity = args.tcity
    datatype = args.datatype
    num_epochs = args.num_epochs
    num_tuine_epochs = args.num_tuine_epochs
    start_time = time.time()
    log("Running CrossTReS, from %s and %s to %s, %s %s experiments, with %d days of data, on %s model" % \
        (scity, scity2, tcity, dataname, datatype, args.data_amount, args.model))
    p_bar.process(1, 1, 5)
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

    p_bar.process(2, 1, 5)
    # 按照百分比分配标签
    source_emb_label = masked_percentile_label(source_data.sum(0).reshape(-1), mask_source.reshape(-1))

    lag = [-6, -5, -4, -3, -2, -1]
    source_data, smax, smin = min_max_normalize(source_data)
    target_data, max_val, min_val = min_max_normalize(target_data)

    source_emb_label2 = masked_percentile_label(source_data2.sum(0).reshape(-1), mask_source2.reshape(-1))
    source_data2, smax2, smin2 = min_max_normalize(source_data2)

    # [(5898, 6, 20, 23), (5898, 1, 20, 23), (1440, 6, 20, 23), (1440, 1, 20, 23), (1440, 6, 20, 23), (1440, 1, 20, 23)]
    # 第一维是数量，第二维是每条数据中的数量
    source_train_x, source_train_y, source_val_x, source_val_y, source_test_x, source_test_y = split_x_y(source_data,
                                                                                                         lag)
    source_train_x2, source_train_y2, source_val_x2, source_val_y2, source_test_x2, source_test_y2 = split_x_y(
        source_data2,
        lag)
    # we concatenate all source data
    # (8778, 6, 20, 23)
    source_x = np.concatenate([source_train_x, source_val_x, source_test_x], axis=0)
    # (8778, 1, 20, 23)
    source_y = np.concatenate([source_train_y, source_val_y, source_test_y], axis=0)
    source_x2 = np.concatenate([source_train_x2, source_val_x2, source_test_x2], axis=0)
    source_y2 = np.concatenate([source_train_y2, source_val_y2, source_test_y2], axis=0)
    target_train_x, target_train_y, target_val_x, target_val_y, target_test_x, target_test_y = split_x_y(target_data,
                                                                                                         lag)
    p_bar.process(3, 1, 5)
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
    if args.alin_month == 1:
        source_x = source_x[-30 * 6 * 24:, :, :, :]
        source_y = source_y[-30 * 6 * 24:, :, :, :]
        source_x2 = source_x2[-30 * 6 * 24:, :, :, :]
        source_y2 = source_y2[-30 * 6 * 24:, :, :, :]
    log("Source split to: x %s, y %s" % (str(source_x.shape), str(source_y.shape)))
    # log("val_x %s, val_y %s" % (str(source_val_x.shape), str(source_val_y.shape)))
    # log("test_x %s, test_y %s" % (str(source_test_x.shape), str(source_test_y.shape)))
    log("Source2 split to: x %s, y %s" % (str(source_x2.shape), str(source_y2.shape)))
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
    source_test_dataset2 = TensorDataset(torch.Tensor(source_test_x2), torch.Tensor(source_test_y2))
    source_test_loader2 = DataLoader(source_test_dataset2, batch_size=args.batch_size)
    source_dataset2 = TensorDataset(torch.Tensor(source_x2), torch.Tensor(source_y2))
    source_loader2 = DataLoader(source_dataset2, batch_size=args.batch_size, shuffle=True)

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
    source_s_adj, source_d_adj, source_od_adj = build_source_dest_graph(scity, dataname, lng_source, lat_source,
                                                                        args.topk)
    source_s_adj2, source_d_adj2, source_od_adj2 = build_source_dest_graph(scity2, dataname, lng_source2, lat_source2,
                                                                           args.topk)
    target_s_adj, target_d_adj, target_od_adj = build_source_dest_graph(tcity, dataname, lng_target, lat_target,
                                                                        args.topk)
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

    source_edges, source_edge_labels = graphs_to_edge_labels(source_graphs)
    source_edges2, source_edge_labels2 = graphs_to_edge_labels(source_graphs2)
    target_edges, target_edge_labels = graphs_to_edge_labels(target_graphs)
    p_bar.process(4, 1, 5)

    return source_emb_label2, source_t_adj, source_edge_labels2, lag, source_poi, source_data2, \
           source_train_y, source_test_x, source_val_x, source_poi_adj, source_poi_adj2, dataname, target_train_x, \
           th_mask_source2, th_mask_source, target_test_loader, target_poi, target_od_adj, \
           source_dataset, mask_source, target_graphs, target_val_dataset, max_val, scity2, smin2, \
           target_emb_label, tcity, source_road_adj2, gpu_available, source_edges2, \
           mask_source2, source_poi_cos, source_data, source_graphs, lng_source, source_road_adj, target_d_adj, \
           target_val_x, source_poi2, scity, target_t_adj, lat_source, lat_target, target_test_x, \
           source_x, target_val_y, lng_source2, num_tuine_epochs, source_d_adj, source_edge_labels, source_prox_adj, \
           source_loader, source_graphs2, transform, source_t_adj2, smax2, target_train_loader, \
           source_test_dataset2, source_poi_cos2, source_od_adj2, target_s_adj, target_test_dataset, \
           source_test_y2, source_y, source_dataset2, target_road_adj, source_test_loader, target_poi_adj, \
           smax, start_time, target_test_y, lng_target, source_test_loader2, \
           source_prox_adj2, target_data, source_x2, target_train_dataset, source_test_dataset, source_test_x2, source_od_adj, target_val_loader, smin, target_poi_cos, target_edge_labels, \
           source_edges, source_train_x2, source_s_adj, source_y2, source_val_x2, source_emb_label, \
           target_norm_poi, source_norm_poi, source_train_x, datatype, source_val_y, mask_target, \
           source_train_y2, source_norm_poi2, source_s_adj2, num_epochs, lat_source2, min_val, target_edges, \
           source_val_y2, target_prox_adj, source_loader2, source_test_y, source_d_adj, \
           target_train_y, th_mask_target, device, p_bar


class Road(nn.Module):

    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.poi = nn.Sequential(nn.Linear(14, self.emb_dim // 2),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(self.emb_dim // 2, self.emb_dim))
        self.distance = nn.Sequential(nn.Linear(1, self.emb_dim // 2),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.emb_dim // 2, self.emb_dim))
        self.road = nn.Sequential(nn.Linear(self.emb_dim * 3, self.emb_dim),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(self.emb_dim, 1),
                                  nn.ReLU(inplace=True))

    def forward(self, poi1, poi2, distance):
        poi1 = self.poi(poi1)
        poi2 = self.poi(poi2)
        dis = self.distance(distance)
        fus = torch.concat((poi1, poi2, dis), dim=1)
        road = self.road(fus)
        return road


def generate_road_loader(city_adjs: List[tuple], args):
    sums = 0
    for c in city_adjs:
        sums = sums + (c[1].shape[0] - 1) * (c[1].shape[1] / 2)
    sums = int(sums)
    x = np.zeros((sums, 29))
    y = np.zeros((sums, 1))
    count = 0
    train_num = int(sums * 0.7)
    val_num = int(sums * 0.15)
    test_num = sums - train_num - val_num

    for c in city_adjs:
        for i in range(c[1].shape[0]):
            for j in range(c[1].shape[0]):
                if i >= j:
                    continue
                p, q = idx_1d22d(i, c[0].shape)
                m, n = idx_1d22d(j, c[0].shape)
                poi1 = c[0][i, :]
                poi2 = c[0][j, :]
                dis = abs(p - m) + abs(q - n)
                x[count, :] = np.concatenate((poi1, poi2, np.array([dis])), axis=0)
                road = c[1][i][j]
                y[count, :] = road
                count = count + 1
    random_ids = np.random.randint(0, x.shape[0], size=x.shape[0])
    x = x[random_ids]
    y = y[random_ids]
    train_x = x[0: train_num, :]
    train_y = y[0: train_num, :]
    val_x = x[train_num: train_num + val_num, :]
    val_y = y[train_num: train_num + val_num, :]
    test_x = x[train_num + val_num:, :]
    test_y = y[train_num + val_num:, :]
    train_x, train_y, val_x, val_y, test_x, test_y = (torch.from_numpy(i) for i in
                                                      [train_x, train_y, val_x, val_y, test_x, test_y])
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    return train_loader, val_loader, test_loader


def yield_8_near(i, ranges):
    """
    产生i的8邻域，i，ranges都是元组或者可下标访问的元素，
    :param i:
    :param ranges:
    :return:
    """
    if i[0] - 1 >= 0 and i[1] - 1 >= 0 and i[0] + 1 < ranges[0] and i[1] + 1 < ranges[1]:
        for k in [-1, 0, 1]:
            for p in [-1, 0, 1]:
                yield i[0] + k, i[1] + p
    elif i == (0, 0):
        for k in [0, 1, 2]:
            for p in [0, 1, 2]:
                yield i[0] + k, i[1] + p
    elif i == (ranges[0] - 1, 0):
        for k in [-2, -1, 0]:
            for p in [0, 1, 2]:
                yield i[0] + k, i[1] + p
    elif i == (0, ranges[1] - 1):
        for k in [0, 1, 2]:
            for p in [-2, -1, 0]:
                yield i[0] + k, i[1] + p
    elif i == (ranges[0] - 1, ranges[1] - 1):
        for k in [-2, -1, 0]:
            for p in [-2, -1, 0]:
                yield i[0] + k, i[1] + p
    elif i[0] == 0 and 0 < i[1] < ranges[1] - 1:
        for k in [0, 1, 2]:
            for p in [-1, 0, 1]:
                yield i[0] + k, i[1] + p
    elif 0 < i[0] < ranges[0] - 1 and i[1] == 0:
        for k in [-1, 0, 1]:
            for p in [0, 1, 2]:
                yield i[0] + k, i[1] + p
    elif i[0] == ranges[0] - 1 and 0 < i[1] < ranges[1] - 1:
        for k in [-2, -1, 0]:
            for p in [-1, 0, 1]:
                yield i[0] + k, i[1] + p
    elif 0 < i[0] < ranges[0] - 1 and i[1] == ranges[1] - 1:
        for k in [-1, 0, 1]:
            for p in [-2, -1, 0]:
                yield i[0] + k, i[1] + p


def save_model(args, net, mvgat, fusion, scoring, edge_disc, root_dir):
    log(" ============== save model ================ ")
    torch.save(net, root_dir + "/net.pth")
    torch.save(mvgat, root_dir + "/mvgat.pth")
    torch.save(fusion, root_dir + "/fusion.pth")
    torch.save(scoring, root_dir + "/scoring.pth")
    torch.save(edge_disc, root_dir + "/edge_disc.pth")


def processGeo(spoi, sroad, s_s, s_t):
    scity_width = spoi.shape[0]
    scity_height = spoi.shape[1]
    s_geo_features = np.zeros((scity_width, scity_height, 41))
    for i in range(scity_width):
        for j in range(scity_height):
            temps, tempt, tempr = np.zeros(9), np.zeros(9), np.zeros(9)
            coords = list(yield_8_near((i, j), (scity_width, scity_height)))
            count = 0
            for p in coords:
                temps[count] = s_s[idx_2d_2_1d((i, j), (scity_width, scity_height)), idx_2d_2_1d((p[0], p[1]), (
                    scity_width, scity_height))]
                tempt[count] = s_t[idx_2d_2_1d((i, j), (scity_width, scity_height)), idx_2d_2_1d((p[0], p[1]), (
                    scity_width, scity_height))]
                tempr[count] = sroad[idx_2d_2_1d((i, j), (scity_width, scity_height)), idx_2d_2_1d((p[0], p[1]), (
                    scity_width, scity_height))]
                count = count + 1
            s_geo_features[i, j, :] = np.concatenate((spoi[i, j, :], temps, tempt, tempr))
    return s_geo_features


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def calculateGeoSimilarity(spoi, sroad, s_s, s_t, mask_s, tpoi, troad, t_s, t_t, mask_t, dis_method="MMD"):
    scity_width = spoi.shape[0]
    scity_height = spoi.shape[1]
    tcity_width = tpoi.shape[0]
    tcity_height = tpoi.shape[1]
    mmd = None
    if dis_method == "MMD":
        mmd = MMD_loss()
    s_geo_features = processGeo(spoi, sroad, s_s, s_t)
    t_geo_features = processGeo(tpoi, troad, t_s, t_t)

    sim = np.zeros((scity_width, scity_height))
    for i in range(scity_width):
        for j in range(scity_height):
            for p in range(tcity_width):
                for q in range(tcity_height):
                    if dis_method == "DTW":
                        sim[i][j] = sim[i][j] + dtw.distance_fast(s_geo_features[i, j, :],
                                                                  t_geo_features[p, q, :])
                    elif dis_method == "MMD":
                        sim[i][j] = sim[i][j] + mmd.forward(
                            torch.unsqueeze(torch.from_numpy(s_geo_features[i, j, :]), dim=0),
                            torch.unsqueeze(torch.from_numpy(t_geo_features[p, q, :]), dim=0))
                    elif dis_method == "KL":
                        sim[i][j] = sim[i][j] + scipy.stats.entropy(
                            scipy.special.softmax(s_geo_features[i, j, :]),
                            scipy.special.softmax(t_geo_features[p, q, :]))
                    elif dis_method == "wasserstein":
                        sim[i][j] = sim[i][j] + wasserstein_distance(s_geo_features[i, j, :],
                                                                     t_geo_features[p, q, :])
                    elif dis_method == "JS":
                        sim[i][j] = sim[i][j] + JS_divergence(
                            scipy.special.softmax(s_geo_features[i, j, :]),
                            scipy.special.softmax(t_geo_features[p, q, :]))
    sim = min_max_normalize(sim)[0]
    if dis_method in ["JS", "KL"]:
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                sim[i][j] = 1.0 / sim[i][j] if sim[i][j] - 0 > 0.00001 else 0

    return min_max_normalize(sim)[0]
