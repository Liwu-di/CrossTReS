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

basic_config(logs_style=LOG_STYLE_ALL)
p_bar = process_bar(final_prompt="时间权重生成完成", unit="part")
args = params()


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

# 按照百分比分配标签
source_emb_label = masked_percentile_label(source_data.sum(0).reshape(-1), mask_source.reshape(-1))

lag = [-6, -5, -4, -3, -2, -1]
source_data, smax, smin = min_max_normalize(source_data)
target_data, max_val, min_val = min_max_normalize(target_data)

source_emb_label2 = masked_percentile_label(source_data2.sum(0).reshape(-1), mask_source2.reshape(-1))
source_data2, smax2, smin2 = min_max_normalize(source_data2)
if args.data_amount != 0:
    target_data = target_data[-args.data_amount * 24:, :, :]
from dtaidistance import dtw
bj2016 = np.load("../data/TaxiBJ16.npy")
bj2016 = bj2016[:,:,:,0]
bj2016_one_hour = []
i = 0
while i < bj2016.shape[0] - 1:
    bj2016_one_hour.append(bj2016[i] + bj2016[i+1])
    i = i + 2
bj2016_one_hour = np.array(bj2016_one_hour)
log(bj2016_one_hour.shape)
# (20, 23)
lng_sourcebj, lat_sourcebj = bj2016_one_hour.shape[1], bj2016_one_hour.shape[2]
mask_sourcebj = bj2016_one_hour.sum(0) > 0
# mask -> th_mask = (20, 23) -> (1, 20, 23)
th_mask_sourcebj = torch.Tensor(mask_sourcebj.reshape(1, lng_sourcebj, lat_sourcebj)).to(device)
log("%d valid regions in source" % np.sum(mask_sourcebj))
bj2016_one_hour, bjmax, bjmin = min_max_normalize(bj2016_one_hour)
bj_train_x, bj_train_y, bj_val_x, bj_val_y, bj_test_x, bj_test_y = split_x_y(bj2016_one_hour, lag)
bj_x = np.concatenate([bj_train_x, bj_val_x, bj_test_x], axis=0)
bj_y = np.concatenate([bj_train_y, bj_val_y, bj_test_y], axis=0)
bj_dataset = TensorDataset(torch.Tensor(bj_x), torch.Tensor(bj_y))
bj_loader = DataLoader(bj_dataset, batch_size = args.batch_size, shuffle=True)
bj_test_dataset = TensorDataset(torch.Tensor(bj_test_x), torch.Tensor(bj_test_y))
bj_test_loader = DataLoader(bj_test_dataset, batch_size = args.batch_size)
time_weight_bj = np.zeros((bj2016_one_hour.shape[1], bj2016_one_hour.shape[2], target_data.shape[1] * target_data.shape[2]))
for i in range(bj2016_one_hour.shape[1]):
    for j in range(bj2016_one_hour.shape[2]):
        if mask_sourcebj[i][j]:
            for p in range(target_data.shape[1]):
                for q in range(target_data.shape[2]):
                    time_weight_bj[i][j][idx_2d_2_1d((p, q), (target_data.shape[1], target_data.shape[2]))] = dtw.distance_fast(bj2016_one_hour[:, i, j], target_data[:, p, q])
    log("=")
time_weight_bj, time_weight_bjmax, time_weight_bjmin = min_max_normalize(time_weight_bj)
np.save(local_path_generate("time_weight", "time_weight{}_{}_{}_{}_{}".format("bj2016", "DC", datatype, dataname, args.data_amount)), time_weight_bj)
# time_weight1 = np.zeros((source_data.shape[1], source_data.shape[2], target_data.shape[1] * target_data.shape[2]))
# time_weight2 = np.zeros((source_data2.shape[1], source_data2.shape[2], target_data.shape[1] * target_data.shape[2]))
# sum = source_data.shape[1] * source_data.shape[2] + \
#       source_data2.shape[1] * source_data2.shape[2]
# p_bar.process(0, 1, sum)
# for i in range(source_data.shape[1]):
#     for j in range(source_data.shape[2]):
#         if mask_source[i][j]:
#             for p in range(target_data.shape[1]):
#                 for q in range(target_data.shape[2]):
#                     time_weight1[i][j][idx_2d_2_1d((p, q), (target_data.shape[1], target_data.shape[2]))] = dtw.distance_fast(source_data[:, i, j], target_data[:, p, q])
#         p_bar.process(0, 1, sum)
# for i in range(source_data2.shape[1]):
#     for j in range(source_data2.shape[2]):
#         if mask_source2[i][j]:
#             for p in range(target_data.shape[1]):
#                 for q in range(target_data.shape[2]):
#                     time_weight2[i][j][idx_2d_2_1d((p, q), (target_data.shape[1], target_data.shape[2]))] = dtw.distance_fast(source_data2[:, i, j], target_data[:, p, q])
#         p_bar.process(0, 1, sum)
# time_weight1, time_weight_max1, time_weight_min1 = min_max_normalize(time_weight1)
# time_weight2, time_weight_max2, time_weight_min2 = min_max_normalize(time_weight2)
# np.save(local_path_generate("time_weight", "time_weight{}_{}_{}_{}_{}".
#                             format(scity, tcity, datatype, dataname, args.data_amount)), time_weight1)
# np.save(local_path_generate("time_weight", "time_weight{}_{}_{}_{}_{}".
#                             format(scity2, tcity, datatype, dataname, args.data_amount)), time_weight2)
