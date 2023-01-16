# -*- coding: utf-8 -*-
# @Time    : 2023/1/16 21:20
# @Author  : 银尘
# @FileName: test_trained_model.py
# @Software: PyCharm
# @Email   : liwudi@liwudi.fun
# @Info    : 测试训练好的模型

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
p_bar.process(0, 1, 5)
args = params()
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


if args.model == 'STResNet':
    net = STResNet(len(lag), 1, 3).to(device)
elif args.model == 'STNet_nobn':
    net = STNet_nobn(1, 3, th_mask_target, sigmoid_out=True).to(device)
    log(net)
elif args.model == 'STNet':
    net = STNet(1, 3, th_mask_target).to(device)
    log(net)


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


p_bar = process_bar(final_prompt="训练完成", unit="epoch")
p_bar.process(0, 1, num_tuine_epochs)
net = torch.load(args.test_mode_path)
for ep in range(num_epochs, num_tuine_epochs + num_epochs):

    net.eval()
    rmse_val, mae_val, val_losses = evaluate(net, target_val_loader, spatial_mask=th_mask_target)
    rmse_test, mae_test, test_losses = evaluate(net, target_test_loader, spatial_mask=th_mask_target)
    sums = 0
    for i in range(len(val_losses)):
        sums = sums + val_losses[i].mean(0).sum().item()

    sums = 0
    for i in range(len(test_losses)):
        sums = sums + test_losses[i].mean(0).sum().item()

    if rmse_val < best_val_rmse:
        best_val_rmse = rmse_val
        best_test_rmse = rmse_test
        best_test_mae = mae_test
        log("Update best test...")
    log("validation rmse %.4f, mae %.4f" % (rmse_val * (max_val - min_val), mae_val * (max_val - min_val)))
    log("test rmse %.4f, mae %.4f" % (rmse_test * (max_val - min_val), mae_test * (max_val - min_val)))

    log()
    p_bar.process(0, 1, num_epochs + num_tuine_epochs)


log("Best test rmse %.4f, mae %.4f" % (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val)))

