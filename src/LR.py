# -*- coding: utf-8 -*-
# @Time    : 2023/5/9 10:43
# @Author  : 银尘
# @FileName: LR.py
# @Software: PyCharm
# @Email   : liwudi@liwudi.fun
# @Info    : why create this file
import argparse
import ast
import math
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
source_emb_label2, source_t_adj, source_edge_labels2, lag, source_poi, source_data2, source_train_y, source_test_x, source_val_x, source_poi_adj, source_poi_adj2, dataname, target_train_x, th_mask_source2, th_mask_source, target_test_loader, target_poi, target_od_adj, source_dataset, mask_source, target_graphs, target_val_dataset, max_val, scity2, smin2, target_emb_label, tcity, source_road_adj2, gpu_available, source_edges2, mask_source2, source_poi_cos, source_data, source_graphs, lng_source, source_road_adj, target_d_adj, target_val_x, source_poi2, scity, target_t_adj, lat_source, lat_target, target_test_x, source_x, target_val_y, lng_source2, num_tuine_epochs, source_d_adj, source_edge_labels, source_prox_adj, source_loader, source_graphs2, transform, source_t_adj2, smax2, target_train_loader, source_test_dataset2, source_poi_cos2, source_od_adj2, target_s_adj, target_test_dataset, source_test_y2, source_y, source_dataset2, target_road_adj, source_test_loader, target_poi_adj, smax, start_time, target_test_y, lng_target, source_test_loader2, source_prox_adj2, target_data, source_x2, target_train_dataset, source_test_dataset, source_test_x2, source_od_adj, target_val_loader, smin, target_poi_cos, target_edge_labels, source_edges, source_train_x2, source_s_adj, source_y2, source_val_x2, source_emb_label, target_norm_poi, source_norm_poi, source_train_x, datatype, source_val_y, mask_target, source_train_y2, source_norm_poi2, source_s_adj2, num_epochs, lat_source2, min_val, target_edges, source_val_y2, target_prox_adj, source_loader2, source_test_y, source_d_adj, target_train_y, th_mask_target, device, p_bar = load_process_data(
    args, p_bar)

this_use_data = target_data[-((args.data_amount + 60) * 24):, :, :]
this_use_data2 = this_use_data[0: args.data_amount * 24]

mae = []
rmse = []
mape = []
xtrain, ytrain,_, __, xtest, ytest = split_x_y(this_use_data, [-6, -5, -4, -3, -2, -1], 1, 60 * 24)
models = []
y_prediction = np.zeros(ytest.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
for i in range(ytest.shape[2]):
    models_temp = []
    for j in range(ytest.shape[3]):
        if mask_target[i][j]:
            log(i, j)
            reg = LinearRegression().fit(xtrain[:,:,i,j], ytrain[:,:,i,j])
            y_pred = reg.predict(xtest[:,:,i,j])
            models_temp.append(reg)
            y_prediction[:, 0, i, j] = y_pred.reshape(-1)

mae = abs(y_prediction - ytest).sum() / (mask_target.sum() * ytest.shape[0]) * (max_val - min_val)
rmse = np.sqrt(((y_prediction-ytest) ** 2).sum() / (mask_target.sum() * ytest.shape[0])) * (max_val - min_val)
with np.errstate(divide='ignore', invalid='ignore'):
    mape = (abs(y_prediction - ytest) / ytest)
    mape[~ np.isfinite(mape)] = 0
    mape = mape.sum() / (ytest > 1e-6).sum()
    mape = mape * 100
log(rmse, mae, mape)

if args.c != "default":
    record.update(record_id, get_timestamp(),
                  "%.4f,%.4f, %.4f" % (rmse, mae, mape),
                  remark="{}".format(args.machine_code))
