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
import seaborn as sns

basic_config(logs_style=LOG_STYLE_ALL)
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

scity3 = args.scity
tcity = args.tcity
log("time_weight {} to {}".format(scity3, tcity))
source_data3 = np.load("../data/%s/%s%s_%s.npy" % (scity3, dataname, scity3, datatype))
lng_source3, lat_source3 = source_data3.shape[1], source_data3.shape[2]
mask_source3 = source_data3.sum(0) > 0
th_mask_source3 = torch.Tensor(mask_source3.reshape(1, lng_source3, lat_source3)).to(device)
log("%d valid regions in source3" % np.sum(mask_source3))


target_data = np.load("../data/%s/%s%s_%s.npy" % (tcity, dataname, tcity, datatype))
lng_target, lat_target = target_data.shape[1], target_data.shape[2]
mask_target = target_data.sum(0) > 0
th_mask_target = torch.Tensor(mask_target.reshape(1, lng_target, lat_target)).to(device)
source_data3, smax, smin = min_max_normalize(source_data3)
target_data, max_val, min_val = min_max_normalize(target_data)
if args.data_amount != 0:
    target_data = target_data[-args.data_amount * 24:, :, :]
    log("data_amount={}".format(str(args.data_amount)))
log("%d valid regions in target" % np.sum(mask_target))

time_weight1 = np.zeros((source_data3.shape[1], source_data3.shape[2], target_data.shape[1] * target_data.shape[2]))
sum = source_data3.shape[1] * source_data3.shape[2]

p_bar = process_bar(final_prompt="时间权重生成完成", unit="part")
for i in range(source_data3.shape[1]):
    for j in range(source_data3.shape[2]):
        if mask_source3[i][j]:
            for p in range(target_data.shape[1]):
                for q in range(target_data.shape[2]):
                    time_weight1[i][j][
                        idx_2d_2_1d((p, q), (target_data.shape[1], target_data.shape[2]))] = dtw.distance_fast(
                        source_data3[:, i, j], target_data[:, p, q])
        p_bar.process(0, 1, sum)

time_weight1, time_weight_max1, time_weight_min1 = min_max_normalize(time_weight1)

np.save(local_path_generate("time_weight2", "time_weight{}_{}_{}_{}_{}".
                            format(scity3, tcity, datatype, dataname, args.data_amount)), time_weight1)

time_weight1, _, _ = min_max_normalize(time_weight1.sum(axis=2))
fig = sns.heatmap(time_weight1)
heatmap = fig.get_figure()
heatmap.savefig(local_path_generate("time_weight2", "time_weight{}_{}_{}_{}_{}.png".
                                    format(scity3, tcity, datatype, dataname, args.data_amount)), dpi=400)
