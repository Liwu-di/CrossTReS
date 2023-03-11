# -*- coding: utf-8 -*-
# @Time    : 2023/3/11 20:33
# @Author  : 银尘
# @FileName: geo_weight.py
# @Software: PyCharm
# @Email   : liwudi@liwudi.fun
# @Info    : why create this file
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
source_emb_label3 = masked_percentile_label(source_data3.sum(0).reshape(-1), mask_source3.reshape(-1))
lag = [-6, -5, -4, -3, -2, -1]
source_data3, smax3, smin3 = min_max_normalize(source_data3)

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

target_data = np.load("../data/%s/%s%s_%s.npy" % (tcity, dataname, tcity, datatype))
lng_target, lat_target = target_data.shape[1], target_data.shape[2]
mask_target = target_data.sum(0) > 0
th_mask_target = torch.Tensor(mask_target.reshape(1, lng_target, lat_target)).to(device)
source_data3, smax, smin = min_max_normalize(source_data3)
target_data, max_val, min_val = min_max_normalize(target_data)
target_data = target_data[0: 8 * 30 * 24]
if args.data_amount != 0:
    target_data = target_data[-args.data_amount * 24:, :, :]
    log("data_amount={}".format(str(args.data_amount)))
log("%d valid regions in target" % np.sum(mask_target))
target_poi = np.load("../data/%s/%s_poi.npy" % (tcity, tcity))
target_poi = target_poi.reshape(lng_target * lat_target, -1)  # regions * classes
transform = TfidfTransformer()
target_norm_poi = np.array(transform.fit_transform(target_poi).todense())
target_prox_adj = add_self_loop(build_prox_graph(lng_target, lat_target))
target_road_adj = add_self_loop(build_road_graph(tcity, lng_target, lat_target))
target_poi_adj, target_poi_cos = build_poi_graph(target_norm_poi, args.topk)
target_poi_adj = add_self_loop(target_poi_adj)
target_s_adj, target_d_adj, target_od_adj = build_source_dest_graph(tcity, dataname, lng_target, lat_target,
                                                                    args.topk)
target_s_adj = add_self_loop(target_s_adj)
target_t_adj = add_self_loop(target_d_adj)
target_od_adj = add_self_loop(target_od_adj)
c3shape = source_data3.shape[1], source_data3.shape[2], 14
ctshape = target_data.shape[1], target_data.shape[2], 14
source_poi3 = source_norm_poi3.reshape(c3shape)
target_poi = target_norm_poi.reshape(ctshape)
geo_weight = calculateGeoSimilarity(spoi=source_poi3, sroad=source_road_adj3, s_s=source_s_adj3, s_t=source_t_adj3,
                                    mask_s=mask_source3, tpoi=target_poi, troad=target_road_adj, t_s=target_s_adj,
                                    t_t=target_t_adj, mask_t=mask_target, dis_method="KL")



np.save(local_path_generate("geo_weight", "geo_weight{}_{}_{}_{}_{}".
                            format(scity3, tcity, datatype, dataname, args.data_amount)), geo_weight)

fig = sns.heatmap(geo_weight)
heatmap = fig.get_figure()
heatmap.savefig(local_path_generate("geo_weight2", "geo_weight{}_{}_{}_{}_{}.png".
                                    format(scity3, tcity, datatype, dataname, args.data_amount)), dpi=600)
