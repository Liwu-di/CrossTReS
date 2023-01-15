# -*- coding: utf-8 -*-
# @Time    : 2023/1/12 9:39
# @Author  : 银尘
# @FileName: multi_graph_merge_7.py
# @Software: PyCharm
# @Email   : liwudi@liwudi.fun
# @Info    : 3城市生成虚拟城市


import argparse
import ast
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PaperCrawlerUtil.common_util import *
from PaperCrawlerUtil.constant import *
from PaperCrawlerUtil.crawler_util import *
from dgl.nn import GATConv
from dtaidistance import dtw
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from model import *
from funcs import *
from params import *
from utils import *
from PaperCrawlerUtil.research_util import *


basic_config(logs_style=LOG_STYLE_ALL)
p_bar = process_bar(final_prompt="初始化准备完成", unit="part")
long_term_save = {}
args = params()
long_term_save["args"] = args.__str__()
if args.c != "default":
    c = ast.literal_eval(args.c)
    record = ResearchRecord(**c)
    record_id = record.insert(__file__, get_timestamp(), args.__str__())
p_bar.process(0, 1, 5)
source_emb_label2, source_t_adj, source_edge_labels2, lag, source_poi, source_data2, \
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
target_train_y, th_mask_target, device, p_bar = load_process_data(args, p_bar)


if args.need_third == 1:
    scity3 = args.scity3
    source_data3 = np.load("../data/%s/%s%s_%s.npy" % (scity3, dataname, scity3, datatype))
    lng_source3, lat_source3 = source_data3.shape[1], source_data3.shape[2]
    mask_source3 = source_data3.sum(0) > 0
    th_mask_source3 = torch.Tensor(mask_source3.reshape(1, lng_source3, lat_source3)).to(device)
    log("%d valid regions in source3" % np.sum(mask_source3))
    # 按照百分比分配标签
    source_emb_label3 = masked_percentile_label(source_data3.sum(0).reshape(-1), mask_source3.reshape(-1))
    lag = [-6, -5, -4, -3, -2, -1]
    source_data3, smax3, smin3 = min_max_normalize(source_data3)
    source_train_x3, source_train_y3, source_val_x3, source_val_y3, source_test_x3, source_test_y3 = split_x_y(source_data3,
                                                                                                               lag)
    # we concatenate all source data
    source_x3 = np.concatenate([source_train_x3, source_val_x3, source_test_x3], axis=0)
    source_y3 = np.concatenate([source_train_y3, source_val_y3, source_test_y3], axis=0)
    source_test_dataset3 = TensorDataset(torch.Tensor(source_test_x3), torch.Tensor(source_test_y3))
    source_test_loader3 = DataLoader(source_test_dataset3, batch_size=args.batch_size)
    source_dataset3 = TensorDataset(torch.Tensor(source_x3), torch.Tensor(source_y3))
    source_loader3 = DataLoader(source_dataset3, batch_size=args.batch_size, shuffle=True)
    source_poi3 = np.load("../data/%s/%s_poi.npy" % (scity3, scity3))
    source_poi3 = source_poi3.reshape(lng_source3 * lat_source3, -1)
    transform3 = TfidfTransformer()
    source_norm_poi3 = np.array(transform3.fit_transform(source_poi3).todense())
    source_prox_adj3 = add_self_loop(build_prox_graph(lng_source3, lat_source3))
    source_road_adj3 = add_self_loop(build_road_graph(scity3, lng_source3, lat_source3))
    source_poi_adj3, source_poi_cos3 = build_poi_graph(source_norm_poi3, args.topk)
    source_poi_adj3 = add_self_loop(source_poi_adj3)
    source_s_adj3, source_d_adj3, source_od_adj3 = build_source_dest_graph(scity3, dataname, lng_source3, lat_source3,
                                                                           args.topk)
    source_s_adj3 = add_self_loop(source_s_adj3)
    source_t_adj3 = add_self_loop(source_d_adj3)
    source_od_adj3 = add_self_loop(source_od_adj3)
    log("Source graphs3: ")
    log("prox_adj3: %d nodes, %d edges" % (source_prox_adj3.shape[0], np.sum(source_prox_adj3)))
    log("road adj3: %d nodes, %d edges" % (source_road_adj3.shape[0], np.sum(source_road_adj3 > 0)))
    log("poi_adj3, %d nodes, %d edges" % (source_poi_adj3.shape[0], np.sum(source_poi_adj3 > 0)))
    log("s_adj3, %d nodes, %d edges" % (source_s_adj3.shape[0], np.sum(source_s_adj3 > 0)))
    log("d_adj3, %d nodes, %d edges" % (source_d_adj3.shape[0], np.sum(source_d_adj3 > 0)))
    log()
    source_graphs3 = adjs_to_graphs([source_prox_adj3, source_road_adj3, source_poi_adj3, source_s_adj3, source_d_adj3])
    for i in range(len(source_graphs3)):
        source_graphs3[i] = source_graphs3[i].to(device)
    source_edges3, source_edge_labels3 = graphs_to_edge_labels(source_graphs3)


# =========================================
# 这里使用已经生成好的DTW进行筛选重要节点
# =========================================

path = "./time_weight/time_weight{}_{}_{}_{}_{}.npy"
s1_time_weight = np.load(path.format(scity, tcity, datatype, dataname, args.data_amount)).sum(2)
s1_time_weight, _, _ = min_max_normalize(s1_time_weight)
s2_time_weight = np.load(path.format(scity2, tcity, datatype, dataname, args.data_amount)).sum(2)
s2_time_weight, _, _ = min_max_normalize(s2_time_weight)
if args.need_third == 1:
    s3_time_weight = np.load(path.format(scity3, tcity, datatype, dataname, args.data_amount)).sum(2)
    s3_time_weight, _, _ = min_max_normalize(s3_time_weight)
virtual_regions = source_data.shape[1] * source_data.shape[2]
threshold = 0.0
time_threshold = 3312
for i in range(10, 95, 5):
    if args.need_third == 1:
        l = (s1_time_weight > (i / 100)).sum() + (s2_time_weight > (i / 100)).sum() + (s3_time_weight > (i / 100)).sum()
        r = (s1_time_weight > ((i / 100) +0.05)).sum() + (s2_time_weight > ((i / 100) +0.05)).sum() + (s3_time_weight > ((i / 100) +0.05)).sum()
    else:
        l = (s1_time_weight > (i / 100)).sum() + (s2_time_weight > (i / 100)).sum()
        r = (s1_time_weight > ((i / 100) + 0.05)).sum() + (s2_time_weight > ((i / 100) + 0.05)).sum()

    if l > virtual_regions > r:
        threshold = (i / 100) + 0.05
        break
    threshold = 0.25
count = 0
virtual_city = np.zeros((time_threshold, source_data.shape[1], source_data.shape[2]))
virtual_poi = np.zeros((source_data.shape[1] * source_data.shape[2], 14))
for i in range(source_data.shape[1]):
    for j in range(source_data.shape[2]):
        if s1_time_weight[i][j] > threshold:
            x, y = idx_1d22d(count, (virtual_city.shape[1], virtual_city.shape[2]))
            virtual_city[:, x, y] = source_data[0: time_threshold, i, j]
            virtual_poi[count, :] = source_poi[idx_2d_2_1d((i, j), (source_data.shape[1], source_data.shape[2])), :]
            count = count + 1
for i in range(source_data2.shape[1]):
    for j in range(source_data2.shape[2]):
        if s2_time_weight[i][j] > threshold:
            x, y = idx_1d22d(count, (virtual_city.shape[1], virtual_city.shape[2]))
            virtual_city[:, x, y] = source_data2[0: time_threshold, i, j]
            virtual_poi[count, :] = source_poi2[idx_2d_2_1d((i, j), (source_data2.shape[1], source_data2.shape[2])), :]
            count = count + 1
if args.need_third == 1:
    for i in range(source_data3.shape[1]):
        for j in range(source_data3.shape[2]):
            if s3_time_weight[i][j] > threshold:
                x, y = idx_1d22d(count, (virtual_city.shape[1], virtual_city.shape[2]))
                virtual_city[:, x, y] = source_data3[0: time_threshold, i, j]
                virtual_poi[count, :] = source_poi3[idx_2d_2_1d((i, j), (source_data3.shape[1], source_data3.shape[2])), :]
                count = count + 1


lng_virtual, lat_virtual = virtual_city.shape[1], virtual_city.shape[2]
mask_virtual = virtual_city.sum(0) > 0
th_mask_virtual = torch.Tensor(mask_virtual.reshape(1, lng_virtual, lat_virtual)).to(device)
log("%d valid regions in virtual" % np.sum(mask_virtual))
virtual_emb_label = masked_percentile_label(virtual_city.sum(0).reshape(-1), mask_virtual.reshape(-1))
lag = [-6, -5, -4, -3, -2, -1]
virtual_city, virtual_max, virtual_min = min_max_normalize(virtual_city)
virtual_train_x, virtual_train_y, virtual_val_x, virtual_val_y, virtual_test_x, virtual_test_y \
    = split_x_y(virtual_city, lag)
# we concatenate all source data
virtual_x = np.concatenate([virtual_train_x, virtual_val_x, virtual_test_x], axis=0)
virtual_y = np.concatenate([virtual_train_y, virtual_val_y, virtual_test_y], axis=0)
virtual_test_dataset = TensorDataset(torch.Tensor(virtual_test_x), torch.Tensor(virtual_test_y))
virtual_test_loader = DataLoader(virtual_test_dataset, batch_size=args.batch_size)
virtual_dataset = TensorDataset(torch.Tensor(virtual_x), torch.Tensor(virtual_y))
virtual_loader = DataLoader(virtual_dataset, batch_size=args.batch_size, shuffle=True)
virtual_transform = TfidfTransformer()
virtual_norm_poi = np.array(virtual_transform.fit_transform(virtual_poi).todense())
virtual_poi_adj, virtual_poi_cos = build_poi_graph(virtual_norm_poi, args.topk)
virtual_poi_adj = add_self_loop(virtual_poi_adj)
virtual_prox_adj = add_self_loop(build_prox_graph(lng_virtual, lat_virtual))
virtual_road_adj = source_road_adj
virtual_s_adj, virtual_d_adj, virtual_od_adj = source_s_adj, source_d_adj, source_od_adj
virtual_s_adj = add_self_loop(virtual_s_adj)
virtual_d_adj = add_self_loop(virtual_d_adj)
virtual_od_adj = add_self_loop(virtual_od_adj)
log()

log("virtual graphs: ")
log("virtual_poi_adj, %d nodes, %d edges" % (virtual_poi_adj.shape[0], np.sum(virtual_poi_adj > 0)))
log("prox_adj3: %d nodes, %d edges" % (virtual_prox_adj.shape[0], np.sum(virtual_prox_adj)))
log("road adj3: %d nodes, %d edges" % (virtual_road_adj.shape[0], np.sum(virtual_road_adj > 0)))
log("s_adj3, %d nodes, %d edges" % (virtual_s_adj.shape[0], np.sum(virtual_s_adj > 0)))
log("d_adj3, %d nodes, %d edges" % (virtual_d_adj.shape[0], np.sum(virtual_d_adj > 0)))
log()

virtual_graphs = adjs_to_graphs([virtual_prox_adj, virtual_road_adj, virtual_poi_adj, virtual_s_adj, virtual_d_adj])
for i in range(len(virtual_graphs)):
    virtual_graphs[i] = virtual_graphs[i].to(device)
virtual_edges, virtual_edge_labels = graphs_to_edge_labels(virtual_graphs)


class Scoring(nn.Module):
    def __init__(self, emb_dim, source_mask, target_mask):
        super().__init__()
        self.emb_dim = emb_dim
        self.score = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim // 2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.emb_dim // 2, self.emb_dim // 2))
        self.source_mask = source_mask
        self.target_mask = target_mask

    def forward(self, source_emb, target_emb, source_mask, target_mask):
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
        target_context = torch.tanh(self.score(target_emb[target_mask.view(-1).bool()]).mean(0))
        source_trans_emb = self.score(source_emb)
        source_score = (source_trans_emb * target_context).sum(1)
        return F.relu(torch.tanh(source_score))[source_mask.view(-1).bool()]


mmd = MMD_loss()
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
scoring = Scoring(emb_dim, th_mask_virtual, th_mask_target).to(device)
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
mvgat_optimizer = optim.Adam(list(mvgat.parameters()) + list(fusion.parameters()), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
# 元学习部分
meta_optimizer = optim.Adam(scoring.parameters(), lr=args.outerlr, weight_decay=args.weight_decay)
best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999
p_bar.process(5, 1, 5)


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




with torch.no_grad():
    views = mvgat(virtual_graphs, torch.Tensor(virtual_norm_poi).to(device))
    # 融合模块指的是把多图的特征融合
    fused_emb_s, _ = fusion(views)
    views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
    fused_emb_t, _ = fusion(views)


class DomainClassify(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.dc = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim // 2),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.emb_dim // 2, self.emb_dim // 2),
                                nn.Linear(self.emb_dim // 2, 2))

    def forward(self, feature):
        res = torch.sigmoid(self.dc(feature))
        return res


if args.node_adapt == "DT":
    # ============================================================================================
    # 预训练特征提取网络mvgat， 方便训练域识别网络
    # ============================================================================================
    loss_mvgats = []
    # 实验确定
    pre = 25
    for i in range(pre):
        loss_source, fused_emb_s, embs_s = forward_emb(virtual_graphs, virtual_norm_poi, virtual_od_adj, virtual_poi_cos)
        loss_target, fused_emb_t, embs_t = forward_emb(target_graphs, target_norm_poi, target_od_adj, target_poi_cos)

        loss_mvgat = loss_source + loss_target
        meta_optimizer.zero_grad()
        loss_mvgat.backward()
        emb_optimizer.step()
        loss_mvgats.append(loss_mvgat.item())
    #     log("loss_mvgat:{}".format(str(loss_mvgat)))
    # loss_mvgats = np.array(loss_mvgats)
    # x = np.array([i + 1 for i in range(pre)])
    # plt.plot(x, loss_mvgats)
    # plt.grid()
    # plt.legend()
    # plt.show()

    with torch.no_grad():
        views = mvgat(virtual_graphs, torch.Tensor(virtual_norm_poi).to(device))
        # 融合模块指的是把多图的特征融合
        fused_emb_s, _ = fusion(views)
        views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
        fused_emb_t, _ = fusion(views)

    s1 = np.array([1, 0])
    st = np.array([0, 1])
    x = torch.concat((fused_emb_s[th_mask_virtual.view(-1).bool()],
                      fused_emb_t[th_mask_target.view(-1).bool()]), dim=0)
    y = []
    y.extend([s1 for i in range(fused_emb_s[th_mask_virtual.view(-1).bool()].shape[0])])
    y.extend([st for i in range(fused_emb_t[th_mask_target.view(-1).bool()].shape[0])])
    y = torch.from_numpy(np.array(y))
    x = x.cpu().numpy()
    y = y.numpy()
    random_ids = np.random.randint(0, x.shape[0], size=x.shape[0])
    x = x[random_ids]
    y = y[random_ids]
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    dt_train = (x[0: 400], y[0: 400])
    dt_val = (x[400: 600], y[400: 600])
    dt_test = (x[600:], y[600:])
    dt_train_dataset = TensorDataset(dt_train[0], dt_train[1])
    dt_val_dataset = TensorDataset(dt_val[0], dt_val[1])
    dt_test_dataset = TensorDataset(dt_test[0], dt_test[1])
    dt_train_loader = DataLoader(dt_train_dataset, batch_size=args.batch_size, shuffle=True)
    dt_val_loader = DataLoader(dt_val_dataset, batch_size=args.batch_size)
    dt_test_loader = DataLoader(dt_test_dataset, batch_size=args.batch_size)
    dt = DomainClassify(emb_dim=emb_dim)
    dt.to(device)
    dt_optimizer = optim.Adam(dt.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    dc_epoch = 10
    epoch_loss = []
    val_loss = []
    test_loss = []
    test_accuracy = []
    for i in range(dc_epoch):
        temp = []
        dt.train()
        for i, (x, y) in enumerate(dt_train_loader):
            x = x.to(device)
            y = y.to(device)
            out = dt(x)
            loss = ((out - y) ** 2)
            loss = loss.sum()
            dt_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dt.parameters(), max_norm=2)
            dt_optimizer.step()
            temp.append(loss.item())
        epoch_loss.append(np.array(temp).mean())
        dt.eval()
        temp = []
        for i, (x, y) in enumerate(dt_val_loader):
            x = x.to(device)
            y = y.to(device)
            out = dt(x)
            loss = ((out - y) ** 2)
            loss = loss.sum()
            temp.append(loss.item())
        val_loss.append(np.array(temp).mean())
        temp = []
        for i, (x, y) in enumerate(dt_test_loader):
            x = x.to(device)
            y = y.to(device)
            out = dt(x)
            loss = ((out - y) ** 2)
            loss = loss.sum()
            temp.append(loss.item())
        test_loss.append(np.array(temp).mean())
        count_sum = 0
        count_true = 0
        for i, (x, y) in enumerate(dt_test_loader):
            x = x.to(device)
            y = y.to(device)
            out = dt(x)
            for i in range(out.shape[0]):
                xx = out[i]
                yy = y[i]
                count_sum = count_sum + 1
                xxx = xx.argmax()
                yyy = yy.argmax()
                if xxx.item() == yyy.item():
                    count_true = count_true + 1
        test_accuracy.append(count_true / count_sum)

    #     log((epoch_loss[-1], val_loss[-1], test_loss[-1], test_accuracy[-1]))
    # plt.plot(np.array([i + 1 for i in range(dc_epoch)]), np.array(epoch_loss), label="train")
    # plt.plot(np.array([i + 1 for i in range(dc_epoch)]), np.array(val_loss), label="val")
    # plt.plot(np.array([i + 1 for i in range(dc_epoch)]), np.array(test_loss), label="test")
    # plt.plot(np.array([i + 1 for i in range(dc_epoch)]), np.array(test_accuracy), label="acc")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.grid()
    # plt.legend()
    # plt.show()

    log("============================")
    log("训练DT网络结束")
    log("============================")


def train_emb_epoch2():

    # loss， 460*64， 5*460*64
    loss_source, fused_emb_s, embs_s = forward_emb(virtual_graphs, virtual_norm_poi, virtual_od_adj, virtual_poi_cos)
    loss_target, fused_emb_t, embs_t = forward_emb(target_graphs, target_norm_poi, target_od_adj, target_poi_cos)

    loss_emb = loss_source + loss_target
    mmd_losses = None
    if args.node_adapt == "MMD":
        # compute domain adaptation loss
        # 随机抽样128个，计算最大平均误差
        source_ids = np.random.randint(0, np.sum(mask_virtual), size=(128,))
        target_ids = np.random.randint(0, np.sum(mask_target), size=(128,))
        # source1 & target
        mmd_loss = mmd(fused_emb_s[th_mask_virtual.view(-1).bool()][source_ids, :],
                       fused_emb_t[th_mask_target.view(-1).bool()][target_ids, :])

        mmd_losses = mmd_loss
    elif args.node_adapt == "DT":
        mmd_losses = dt(fused_emb_s[th_mask_virtual.view(-1).bool()]).sum() + \
                     dt(fused_emb_t[th_mask_target.view(-1).bool()]).sum()

    # 随机抽样边256
    source_batch_edges = np.random.randint(0, len(virtual_edges), size=(256,))
    target_batch_edges = np.random.randint(0, len(target_edges), size=(256,))
    source_batch_src = torch.Tensor(virtual_edges[source_batch_edges, 0]).long()
    source_batch_dst = torch.Tensor(virtual_edges[source_batch_edges, 1]).long()
    source_emb_src = fused_emb_s[source_batch_src, :]
    source_emb_dst = fused_emb_s[source_batch_dst, :]
    target_batch_src = torch.Tensor(target_edges[target_batch_edges, 0]).long()
    target_batch_dst = torch.Tensor(target_edges[target_batch_edges, 1]).long()
    target_emb_src = fused_emb_t[target_batch_src, :]
    target_emb_dst = fused_emb_t[target_batch_dst, :]
    # 源城市目的城市使用同样的边分类器
    pred_source = edge_disc.forward(source_emb_src, source_emb_dst)
    pred_target = edge_disc.forward(target_emb_src, target_emb_dst)
    source_batch_labels = torch.Tensor(virtual_edge_labels[source_batch_edges]).to(device)
    target_batch_labels = torch.Tensor(target_edge_labels[target_batch_edges]).to(device)
    # -（label*log(sigmod(pred)+0.000001)) + (1-label)*log(1-sigmod+0.000001) sum mean
    loss_et_source = -((source_batch_labels * torch.log(torch.sigmoid(pred_source) + 1e-6)) + (
            1 - source_batch_labels) * torch.log(1 - torch.sigmoid(pred_source) + 1e-6)).sum(1).mean()
    loss_et_target = -((target_batch_labels * torch.log(torch.sigmoid(pred_target) + 1e-6)) + (
            1 - target_batch_labels) * torch.log(1 - torch.sigmoid(pred_target) + 1e-6)).sum(1).mean()
    loss_et = loss_et_source + loss_et_target

    emb_optimizer.zero_grad()
    # 公式11
    loss = None
    if args.node_adapt == "MMD":
        loss = loss_emb + mmd_w * mmd_losses + et_w * loss_et
    elif args.node_adapt == "DT":
        loss = loss_emb - mmd_w * mmd_losses + et_w * loss_et
    loss.backward()
    emb_optimizer.step()
    return loss_emb.item(), mmd_losses.item(), loss_et.item()


emb_losses = []
mmd_losses = []
edge_losses = []
pretrain_emb_epoch = 80
# 预训练图数据嵌入，边类型分类，节点对齐 ——> 获得区域特征
for emb_ep in range(pretrain_emb_epoch):
    loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch2()
    emb_losses.append(loss_emb_)
    mmd_losses.append(loss_mmd_)
    edge_losses.append(loss_et_)
log("[%.2fs]Pretrain embeddings for %d epochs, average emb loss %.4f, node loss %.4f, edge loss %.4f" % (
    time.time() - start_time, pretrain_emb_epoch, np.mean(emb_losses), np.mean(mmd_losses), np.mean(edge_losses)))
with torch.no_grad():
    views = mvgat(virtual_graphs, torch.Tensor(virtual_norm_poi).to(device))
    # 融合模块指的是把多图的特征融合
    fused_emb_s, _ = fusion(views)
    views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
    fused_emb_t, _ = fusion(views)


long_term_save["emb_losses"] = emb_losses
long_term_save["mmd_losses"] = mmd_losses
long_term_save["edge_losses"] = edge_losses

emb_s = fused_emb_s.cpu().numpy()[mask_virtual.reshape(-1)]
emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)]
logreg = LogisticRegression(max_iter=500)
cvscore_s = cross_validate(logreg, emb_s, virtual_emb_label)['test_score'].mean()
cvscore_t = cross_validate(logreg, emb_t, target_emb_label)['test_score'].mean()
log("[%.2fs]Pretraining embedding, source cvscore %.4f, target cvscore %.4f" % \
    (time.time() - start_time, cvscore_s, cvscore_t))
log()


def net_fix(source, y, weight, mask, fast_weights, bn_vars):
    pred_source = net.functional_forward(source, mask.bool(), fast_weights, bn_vars, bn_training=True)
    if len(pred_source.shape) == 4:  # STResNet
        loss_source = ((pred_source - y) ** 2).view(args.meta_batch_size, 1, -1)[:, :,
                      mask.view(-1).bool()]
        loss_source = (loss_source * weight).mean(0).sum()
    elif len(pred_source.shape) == 3:  # STNet
        y = y.view(args.meta_batch_size, 1, -1)[:, :, mask.view(-1).bool()]
        loss_source = (((pred_source - y) ** 2) * weight.view(1, 1, -1))
        loss_source = loss_source.mean(0).sum()
    fast_loss = loss_source
    grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True)
    for name, grad in zip(fast_weights.keys(), grads):
        fast_weights[name] = fast_weights[name] - args.innerlr * grad
    return fast_loss, fast_weights, bn_vars


def meta_train_epoch(s_embs, t_embs):
    meta_query_losses = []
    for meta_ep in range(args.outeriter):
        fast_losses = []
        fast_weights, bn_vars = get_weights_bn_vars(net)
        source_weights = scoring(s_embs, t_embs, th_mask_virtual, th_mask_target)
        # inner loop on source, pre-train with weights
        for meta_it in range(args.sinneriter):
            s_x1, s_y1 = batch_sampler((torch.Tensor(virtual_train_x), torch.Tensor(virtual_train_y)),
                                       args.meta_batch_size)
            s_x1 = s_x1.to(device)
            s_y1 = s_y1.to(device)
            fast_loss, fast_weights, bn_vars = net_fix(s_x1, s_y1, source_weights, th_mask_virtual, fast_weights,
                                                       bn_vars)
            fast_losses.append(fast_loss.item())


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
            x_q = None
            y_q = None
            temp_mask = None

            x_q, y_q = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size)
            temp_mask = th_mask_target
            x_q = x_q.to(device)
            y_q = y_q.to(device)
            pred_q = net.functional_forward(x_q, temp_mask.bool(), fast_weights, bn_vars, bn_training=True)
            if len(pred_q.shape) == 4:  # STResNet
                loss = (((pred_q - y_q) ** 2) * (temp_mask.view(1, 1, lng_target, lat_target)))
                loss = loss.mean(0).sum()
            elif len(pred_q.shape) == 3:  # STNet
                y_q = y_q.view(args.batch_size, 1, -1)[:, :, temp_mask.view(-1).bool()]
                loss = ((pred_q - y_q) ** 2).mean(0).sum()
            q_losses.append(loss)
        q_loss = torch.stack(q_losses).mean()
        weights_mean = (source_weights ** 2).mean()
        meta_loss = q_loss + weights_mean * args.weight_reg
        meta_optimizer.zero_grad()
        meta_loss.backward(inputs=list(scoring.parameters()), retain_graph=True)
        torch.nn.utils.clip_grad_norm_(scoring.parameters(), max_norm=2)
        meta_optimizer.step()
        meta_query_losses.append(q_loss.item())
    return np.mean(meta_query_losses)


avg_q_loss = meta_train_epoch(fused_emb_s, fused_emb_t)
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

writer = SummaryWriter("log-{}-batch-{}-name-{}-type-{}-model-{}-amount-{}-topk-{}-time-{}".
                       format("多城市{},{} and {}-{}".format(args.scity, args.scity2, args.scity3, args.tcity), args.batch_size,
                              args.dataname,
                              args.datatype, args.model, args.data_amount, args.topk, get_timestamp(split="-")))


if args.is_st_weight_static == 1:
    time_weight = np.zeros((virtual_city.shape[1], virtual_city.shape[2], target_data.shape[1] * target_data.shape[2]))
    sum = virtual_city.shape[1] * virtual_city.shape[2]

    p_bar = process_bar(final_prompt="时间权重生成完成", unit="part")
    for i in range(virtual_city.shape[1]):
        for j in range(virtual_city.shape[2]):
            if mask_virtual[i][j]:
                for p in range(target_data.shape[1]):
                    for q in range(target_data.shape[2]):
                        time_weight[i][j][
                            idx_2d_2_1d((p, q), (target_data.shape[1], target_data.shape[2]))] = dtw.distance_fast(
                            virtual_city[:, i, j], target_data[(8 * 30 - args.data_amount) * 24 : 8 * 30 * 24, p, q])
            p_bar.process(0, 1, sum)

    time_weight, time_weight_max1, time_weight_min1 = min_max_normalize(time_weight)
    time_weight, _, _ = min_max_normalize(time_weight.sum(axis=2))

p_bar = process_bar(final_prompt="训练完成", unit="epoch")
p_bar.process(0, 1, num_epochs + num_tuine_epochs)
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
        loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch2()
        emb_losses.append(loss_emb_)
        mmd_losses.append(loss_mmd_)
        edge_losses.append(loss_et_)
        train_emb_losses.append((loss_emb_, loss_mmd_, loss_et_))
    # evaluate embeddings
    with torch.no_grad():
        views = mvgat(virtual_graphs, torch.Tensor(virtual_norm_poi).to(device))
        # 融合模块指的是把多图的特征融合
        fused_emb_s, _ = fusion(views)
        views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
        fused_emb_t, _ = fusion(views)
    if ep % 2 == 0:
        """
        每两个epoch显示一些数据
        """
        emb_s = fused_emb_s.cpu().numpy()[mask_virtual.reshape(-1)]
        emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)]
        mix_embs = np.concatenate([emb_s, emb_t], axis=0)
        mix_labels = np.concatenate([virtual_emb_label, target_emb_label])
        logreg = LogisticRegression(max_iter=500)
        cvscore_s = cross_validate(logreg, emb_s, virtual_emb_label)['test_score'].mean()
        cvscore_t = cross_validate(logreg, emb_t, target_emb_label)['test_score'].mean()
        cvscore_mix = cross_validate(logreg, mix_embs, mix_labels)['test_score'].mean()
        log(
            "[%.2fs]Epoch %d, embedding loss %.4f, mmd loss %.4f, edge loss %.4f, source cvscore %.4f, target cvscore %.4f, mixcvscore %.4f" % \
            (time.time() - start_time, ep, np.mean(emb_losses), np.mean(mmd_losses), np.mean(edge_losses), cvscore_s,
             cvscore_t, cvscore_mix))


    avg_q_loss = meta_train_epoch(fused_emb_s, fused_emb_t)
    with torch.no_grad():
        source_weights = scoring(fused_emb_s, fused_emb_t, th_mask_virtual, th_mask_target)
        source_weight_list.append(list(source_weights.cpu().numpy()))


    if ep == 0:
        source_weights_ma = torch.ones_like(source_weights, device=device, requires_grad=False)
    source_weights_ma = ma_param * source_weights_ma + (1 - ma_param) * source_weights
    if args.is_st_weight_static == 1:
        time_score = args.time_score_weight
        space_score = args.space_score_weight
        source_weights_ma = space_score * source_weights_ma + time_score * torch.from_numpy(time_weight[mask_virtual]).to(device)
    source_weights_ma_list.append(list(source_weights_ma.cpu().numpy()))
    weight = None if args.need_weight == 0 else source_weights_ma
    virtual_source_loss = train_epoch(net, virtual_loader, pred_optimizer, weights=weight, num_iters=args.pretrain_iter, mask=th_mask_virtual)
    avg_target_loss = evaluate(net, target_train_loader, spatial_mask=th_mask_target)[0]
    log("[%.2fs]Epoch %d, virtual_source_loss %.4f" % (time.time() - start_time, ep, np.mean(virtual_source_loss)))
    log("[%.2fs]Epoch %d, target_loss %.4f" % (time.time() - start_time, ep, avg_target_loss))
    if source_weights_ma.mean() < 0.005:
        # stop pre-training
        break
    net.eval()
    rmse_val, mae_val, target_val_losses = evaluate(net, target_val_loader, spatial_mask=th_mask_target)
    log("Epoch %d, target validation rmse %.4f, mae %.4f" % (
        ep, rmse_val * (max_val - min_val), mae_val * (max_val - min_val)))
    log()
    target_validation_mae.append(mae_val * (max_val - min_val))
    target_validation_rmse.append(rmse_val * (max_val - min_val))
    sums = 0
    for i in range(len(target_val_losses)):
        sums = sums + target_val_losses[i].mean(0).sum().item()
    writer.add_scalar("train target val loss", sums, ep)
    train_target_val_loss.append(sums)
    sums = 0
    train_source_val_loss.append(sums)
    p_bar.process(0, 1, num_epochs + num_tuine_epochs)


for ep in range(num_epochs, num_tuine_epochs + num_epochs):
    # fine-tuning
    net.train()
    avg_loss = train_epoch(net, target_train_loader, pred_optimizer, mask=th_mask_target)
    log('[%.2fs]Epoch %d, target pred loss %.4f' % (time.time() - start_time, ep, np.mean(avg_loss)))
    writer.add_scalar("target pred loss", np.mean(avg_loss), ep - num_epochs)
    target_pred_loss.append(np.mean(avg_loss))
    net.eval()
    rmse_val, mae_val, val_losses = evaluate(net, target_val_loader, spatial_mask=th_mask_target)
    rmse_test, mae_test, test_losses = evaluate(net, target_test_loader, spatial_mask=th_mask_target)
    sums = 0
    for i in range(len(val_losses)):
        sums = sums + val_losses[i].mean(0).sum().item()
    writer.add_scalar("target train val loss", sums, ep)
    target_train_val_loss.append(sums)
    sums = 0
    for i in range(len(test_losses)):
        sums = sums + test_losses[i].mean(0).sum().item()
    target_train_test_loss.append(sums)
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
    validation_rmse.append(rmse_val * (max_val - min_val))
    validation_mae.append(mae_val * (max_val - min_val))
    test_rmse.append(rmse_test * (max_val - min_val))
    test_mae.append(mae_test * (max_val - min_val))
    log()
    p_bar.process(0, 1, num_epochs + num_tuine_epochs)

long_term_save["source_weights_ma_list"] = source_weights_ma_list
long_term_save["source_weight_list"] = source_weight_list
long_term_save["train_emb_losses"] = train_emb_losses
long_term_save["average_meta_query_loss"] = average_meta_query_loss
long_term_save["source_validation_rmse"] = source_validation_rmse
long_term_save["target_validation_rmse"] = target_validation_rmse
long_term_save["source_validation_mae"] = source_validation_mae
long_term_save["target_validation_mae"] = target_validation_mae
long_term_save["train_target_val_loss"] = train_target_val_loss
long_term_save["train_source_val_loss"] = train_source_val_loss
long_term_save["target_pred_loss"] = target_pred_loss
long_term_save["target_train_val_loss"] = target_train_val_loss
long_term_save["target_train_test_loss"] = target_train_test_loss
long_term_save["validation_rmse"] = validation_rmse
long_term_save["validation_mae"] = validation_mae
long_term_save["test_rmse"] = test_rmse
long_term_save["test_mae"] = test_mae

log("Best test rmse %.4f, mae %.4f" % (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val)))
root_dir = local_path_generate(
    "./model/{}".format(
        "{}-batch-{}-{}-{}-{}-amount-{}-topk-{}-time-{}".format(
            "多城市{},{}and{}-{}".format(args.scity, args.scity2, args.scity3, args.tcity),
            args.batch_size, args.dataname, args.datatype, args.model, args.data_amount,
            args.topk, get_timestamp(split="-")
        )
    ), create_folder_only=True)
torch.save(net, root_dir + "/net.pth")
torch.save(mvgat, root_dir + "/mvgat.pth")
torch.save(fusion, root_dir + "/fusion.pth")
torch.save(scoring, root_dir + "/scoring.pth")
torch.save(edge_disc, root_dir + "/edge_disc.pth")
save_obj(long_term_save,
         local_path_generate("experiment_data",
                             "data_{}.collection".format(
                                 "{}-batch-{}-{}-{}-{}-amount-{}-time-{}".format(
                                     "多城市{},{}and{}-{}".format(args.scity, args.scity2, args.scity3, args.tcity),
                                     args.batch_size, args.dataname, args.datatype, args.model, args.data_amount,
                                     get_timestamp(split="-")
                                 )
                             )
                             )
         )
if args.c != "default":
    record.update(record_id, get_timestamp(),
                  "%.4f,%.4f" %
                  (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val)))


