# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 15:36
# @Author  : 银尘
# @FileName: time_weight.py
# @Software: PyCharm
# @Email   ：liwudi@liwudi.fun
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
from dtaidistance import dtw


basic_config(logs_style=LOG_STYLE_ALL)
p_bar = process_bar(final_prompt="初始化准备完成", unit="part")
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

if args.dataname == "Taxi":
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
time_weight1 = np.zeros((source_data.shape[1], source_data.shape[2], target_data.shape[1] * target_data.shape[2]))
time_weight2 = np.zeros((source_data2.shape[1], source_data2.shape[2], target_data.shape[1] * target_data.shape[2]))
if args.dataname == "Taxi":
    time_weight3 = np.zeros((source_data3.shape[1], source_data3.shape[2], target_data.shape[1] * target_data.shape[2]))
    sum = source_data.shape[1] * source_data.shape[2] + \
          source_data2.shape[1] * source_data2.shape[2] + \
          source_data3.shape[1] * source_data3.shape[2]
else:
    sum = source_data.shape[1] * source_data.shape[2] + \
          source_data2.shape[1] * source_data2.shape[2]

p_bar = process_bar(final_prompt="时间权重生成完成", unit="part")
for i in range(source_data.shape[1]):
    for j in range(source_data.shape[2]):
        if mask_source[i][j]:
            for p in range(target_data.shape[1]):
                for q in range(target_data.shape[2]):
                    time_weight1[i][j][idx_2d_2_1d((p, q), (target_data.shape[1], target_data.shape[2]))] = dtw.distance_fast(source_data[:, i, j], target_data[:, p, q])
        p_bar.process(0, 1, sum)
for i in range(source_data2.shape[1]):
    for j in range(source_data2.shape[2]):
        if mask_source2[i][j]:
            for p in range(target_data.shape[1]):
                for q in range(target_data.shape[2]):
                    time_weight2[i][j][idx_2d_2_1d((p, q), (target_data.shape[1], target_data.shape[2]))] = dtw.distance_fast(source_data2[:, i, j], target_data[:, p, q])
        p_bar.process(0, 1, sum)
if args.dataname == "Taxi":
    for i in range(source_data3.shape[1]):
        for j in range(source_data3.shape[2]):
            if mask_source3[i][j]:
                for p in range(target_data.shape[1]):
                    for q in range(target_data.shape[2]):
                        time_weight3[i][j][idx_2d_2_1d((p, q), (target_data.shape[1], target_data.shape[2]))] = dtw.distance_fast(source_data3[:, i, j], target_data[:, p, q])
            p_bar.process(0, 1, sum)
time_weight1, time_weight_max1, time_weight_min1 = min_max_normalize(time_weight1)
time_weight2, time_weight_max2, time_weight_min2 = min_max_normalize(time_weight2)

np.save(local_path_generate("time_weight2", "time_weight{}_{}_{}_{}_{}".
                            format(scity, tcity, datatype, dataname, args.data_amount)), time_weight1)
np.save(local_path_generate("time_weight2", "time_weight{}_{}_{}_{}_{}".
                            format(scity2, tcity, datatype, dataname, args.data_amount)), time_weight2)
if args.dataname == "Taxi":
    time_weight3, time_weight_max3, time_weight_min3 = min_max_normalize(time_weight3)
    np.save(local_path_generate("time_weight2", "time_weight{}_{}_{}_{}_{}".
                                format(scity3, tcity, datatype, dataname, args.data_amount)), time_weight3)
