# -*- coding: utf-8 -*-
# @Time    : 2023/3/31 16:48
# @Author  : 银尘
# @FileName: multi_graph_merge_8.py
# @Software: PyCharm
# @Email   : liwudi@liwudi.fun
# @Info    : why create this file


import argparse
import ast
import os
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
from multiprocessing import cpu_count
import networkx as nx
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
    source_train_x3, source_train_y3, source_val_x3, source_val_y3, source_test_x3, source_test_y3 = split_x_y(
        source_data3,
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

if args.need_geo_weight == 1:
    log("============================")
    log("=======use geo score========")
    log("============================")
    path2 = "./geo_weight/geo_weight{}_{}_{}_{}_{}.npy"
    geo_weight1 = np.load(path2.format(scity, tcity, datatype, dataname, args.data_amount))
    geo_weight2 = np.load(path2.format(scity2, tcity, datatype, dataname, args.data_amount))
    if args.need_third == 1:
        geo_weight3 = np.load(path2.format(scity3, tcity, datatype, dataname, args.data_amount))
    # c1shape = source_data.shape[1], source_data.shape[2], 14
    # c2shape = source_data2.shape[1], source_data2.shape[2], 14
    # ctshape = target_data.shape[1], target_data.shape[2], 14
    # spoi1 = source_norm_poi.reshape(c1shape)
    # spoi2 = source_norm_poi2.reshape(c2shape)
    # tpoi = target_norm_poi.reshape(ctshape)
    # dis_method = args.geo_dis
    # log("geo dis meth :{}".format(dis_method))
    # geo_weight1 = calculateGeoSimilarity(spoi1, source_road_adj, source_s_adj, source_t_adj, mask_source, tpoi,
    #                                      target_road_adj, target_s_adj, target_t_adj, mask_target, dis_method=dis_method)
    # geo_weight2 = calculateGeoSimilarity(spoi2, source_road_adj2, source_s_adj2, source_t_adj2, mask_source2, tpoi,
    #                                      target_road_adj, target_s_adj, target_t_adj, mask_target, dis_method=dis_method)
    # if args.need_third == 1:
    #     c3shape = source_data3.shape[1], source_data3.shape[2], 14
    #     spoi3 = source_norm_poi3.reshape(c3shape)
    #     geo_weight3 = calculateGeoSimilarity(spoi3, source_road_adj3, source_s_adj3, source_t_adj3, mask_source3, tpoi,
    #                                          target_road_adj, target_s_adj, target_t_adj, mask_target, dis_method=dis_method)

virtual_city = None
virtual_poi = None
virtual_road = None
virtual_od = None
virtual_source_coord = None
if args.use_linked_region == 0:
    # =========================================
    # 这里使用已经生成好的DTW进行筛选重要节点
    # =========================================

    # %%
    path = "./time_weight/time_weight{}_{}_{}_{}_{}.npy"
    s1_time_weight = np.load(path.format(scity, tcity, datatype, dataname, args.data_amount)).sum(2)
    s1_time_weight, _, _ = min_max_normalize(s1_time_weight)
    s1_time_weight = s1_time_weight.reshape(-1)

    s2_time_weight = np.load(path.format(scity2, tcity, datatype, dataname, args.data_amount)).sum(2)
    s2_time_weight, _, _ = min_max_normalize(s2_time_weight)
    s2_time_weight = s2_time_weight.reshape(-1)

    s1_regions = []
    s2_regions = []
    s3_regions = []

    if args.need_third == 1:
        s3_time_weight = np.load(path.format(scity3, tcity, datatype, dataname, args.data_amount)).sum(2)
        s3_time_weight, _, _ = min_max_normalize(s3_time_weight)
        s3_time_weight = s3_time_weight.reshape(-1)

    if args.need_geo_weight == 1:
        s1_time_weight = args.time_rate * s1_time_weight + args.geo_rate * geo_weight1
        s2_time_weight = args.time_rate * s2_time_weight + args.geo_rate * geo_weight2
        if args.need_third == 1:
            s3_time_weight = args.time_rate * s3_time_weight + args.geo_rate * geo_weight3
    threshold = args.threshold
    s1_amont = args.s1_amont
    s2_amont = args.s2_amont
    s3_amont = args.s3_amont
    time_threshold = args.cut_data

    s1_regions.extend([idx_1d22d(s1_time_weight.argsort()[-i], (source_data.shape[1], source_data.shape[2])) for i in
                       range(s1_amont)])
    s2_regions.extend([idx_1d22d(s2_time_weight.argsort()[-i], (source_data2.shape[1], source_data2.shape[2])) for i in
                       range(s2_amont)])
    if args.need_third == 1:
        s3_regions.extend(
            [idx_1d22d(s3_time_weight.argsort()[-i], (source_data3.shape[1], source_data3.shape[2])) for i in
             range(s3_amont)])

    log("s1 r = {} s2 r = {} s3 r = {}".format(str(len(s1_regions)), str(len(s2_regions)), str(len(s3_regions))))

    s1_regions_, s2_regions_, s3_regions_ = [], [], []
    np_3_3 = []
    np_3_3_poi = []
    np_3_3_coord = []
    for i in s1_regions:
        if i in s1_regions_:
            continue
        b = list(yield_8_near((i[0], i[1]), (source_data.shape[1], source_data.shape[2])))
        for m in b:
            if m not in s1_regions_ and m in s1_regions:
                s1_regions_.append(m)
        count = 0
        temp1 = np.zeros((time_threshold, 3, 3))
        temp2 = np.zeros((3, 3, 14))
        temp3 = np.zeros((3, 3, 3))
        for p in range(3):
            for q in range(3):
                temp1[:, p, q] = source_data[0:time_threshold, b[count][0], b[count][1]]
                temp2[p, q, :] = source_poi[
                                 idx_2d_2_1d((b[count][0], b[count][1]), (source_data.shape[1], source_data.shape[2])),
                                 :]
                temp3[p, q, :] = np.array([b[count][0], b[count][1], 1])
                count = count + 1
        np_3_3.append(temp1)
        np_3_3_poi.append(temp2)
        np_3_3_coord.append(temp3)
        if len(s1_regions_) == len(s1_regions):
            break

    for i in s2_regions:
        if i in s2_regions_:
            continue
        b = list(yield_8_near((i[0], i[1]), (source_data2.shape[1], source_data2.shape[2])))
        for m in b:
            if m not in s2_regions_ and m in s2_regions:
                s2_regions_.append(m)
        count = 0
        temp1 = np.zeros((time_threshold, 3, 3))
        temp2 = np.zeros((3, 3, 14))
        temp3 = np.zeros((3, 3, 3))
        for p in range(3):
            for q in range(3):
                temp1[:, p, q] = source_data2[0:time_threshold, b[count][0], b[count][1]]
                temp2[p, q, :] = source_poi2[idx_2d_2_1d((b[count][0], b[count][1]),
                                                         (source_data2.shape[1], source_data2.shape[2])), :]
                temp3[p, q, :] = np.array([b[count][0], b[count][1], 2])
                count = count + 1
        np_3_3.append(temp1)
        np_3_3_poi.append(temp2)
        np_3_3_coord.append(temp3)
        if len(s2_regions_) == len(s2_regions):
            break

    for i in s3_regions:
        if i in s3_regions_:
            continue
        b = list(yield_8_near((i[0], i[1]), (source_data3.shape[1], source_data3.shape[2])))
        for m in b:
            if m not in s3_regions_ and m in s3_regions:
                s3_regions_.append(m)
        count = 0
        temp1 = np.zeros((time_threshold, 3, 3))
        temp2 = np.zeros((3, 3, 14))
        temp3 = np.zeros((3, 3, 3))
        for p in range(3):
            for q in range(3):
                temp1[:, p, q] = source_data3[0:time_threshold, b[count][0], b[count][1]]
                temp2[p, q, :] = source_poi3[idx_2d_2_1d((b[count][0], b[count][1]),
                                                         (source_data3.shape[1], source_data3.shape[2])), :]
                temp3[p, q, :] = np.array([b[count][0], b[count][1], 3])
                count = count + 1
        np_3_3.append(temp1)
        np_3_3_poi.append(temp2)
        np_3_3_coord.append(temp3)
        if len(s3_regions_) == len(s3_regions):
            break


    def is_integer(number):
        if int(number) == number:
            return True
        else:
            return False


    def is_3_product(number):
        if int(number) % 3 == 0:
            return True
        else:
            return False


    def crack(integer):
        start = int(np.sqrt(integer))
        factor = integer / start
        while not (is_integer(start) and is_integer(factor) and is_3_product(factor) and is_3_product(start)):
            start += 1
            factor = integer / start
        return int(factor), start


    shape = crack(len(np_3_3) * 9)
    log("virtual shape {}".format(str(shape)))
    virtual_city = np.zeros((time_threshold, shape[0], shape[1]))
    virtual_poi = np.zeros((shape[0], shape[1], 14))
    virtual_source_coord = np.zeros((shape[0], shape[1], 3))

    virtual_road = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    virtual_od = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    count = 0
    for i in range(0, shape[0], 3):
        for j in range(0, shape[1], 3):
            virtual_city[:, i: i + 3, j: j + 3] = np_3_3[count]
            virtual_poi[i: i + 3, j: j + 3, :] = np_3_3_poi[count]
            virtual_source_coord[i: i + 3, j: j + 3, :] = np_3_3_coord[count]
            count = count + 1
    for i in range(virtual_source_coord.shape[0] * virtual_source_coord.shape[1]):
        for j in range(virtual_source_coord.shape[0] * virtual_source_coord.shape[1]):
            m, n = idx_1d22d(i, (virtual_source_coord.shape[0], virtual_source_coord.shape[1]))
            p, q = idx_1d22d(j, (virtual_source_coord.shape[0], virtual_source_coord.shape[1]))
            if virtual_source_coord[m][n][2] == virtual_source_coord[p][q][2]:
                od = None
                road = None
                shape = None
                if virtual_source_coord[m][n][2] == 1:
                    od = source_od_adj
                    road = source_road_adj
                    shape = (source_data.shape[1], source_data.shape[2])
                elif virtual_source_coord[m][n][2] == 2:
                    od = source_od_adj2
                    road = source_road_adj2
                    shape = (source_data2.shape[1], source_data2.shape[2])
                elif virtual_source_coord[m][n][2] == 3:
                    od = source_od_adj3
                    road = source_road_adj3
                    shape = (source_data3.shape[1], source_data3.shape[2])
                c = idx_2d_2_1d(
                    (virtual_source_coord[m][n][0], virtual_source_coord[m][n][1]
                     ), shape)
                d = idx_2d_2_1d(
                    (virtual_source_coord[p][q][0], virtual_source_coord[p][q][1]
                     ), shape)
                c = int(c)
                d = int(d)
                virtual_od[i][j] = od[c][d]
                virtual_road[i][j] = road[c][d]
    for i in range(virtual_road.shape[0]):
        virtual_road[i][i] = 1
elif args.use_linked_region == 1:
    # =========================================
    # 这里使用已经生成好的DTW进行筛选重要节点
    # =========================================

    path = "./time_weight/time_weight{}_{}_{}_{}_{}.npy"
    s1_time_weight = np.load(path.format(scity, tcity, datatype, dataname, args.data_amount)).sum(2)
    s1_time_weight, _, _ = min_max_normalize(s1_time_weight)

    s2_time_weight = np.load(path.format(scity2, tcity, datatype, dataname, args.data_amount)).sum(2)
    s2_time_weight, _, _ = min_max_normalize(s2_time_weight)

    s1_regions = []
    s2_regions = []
    s3_regions = []

    if args.need_third == 1:
        s3_time_weight = np.load(path.format(scity3, tcity, datatype, dataname, args.data_amount)).sum(2)
        s3_time_weight, _, _ = min_max_normalize(s3_time_weight)

    if args.need_geo_weight == 1:
        s1_time_weight = args.time_rate * s1_time_weight + args.geo_rate * geo_weight1
        s2_time_weight = args.time_rate * s2_time_weight + args.geo_rate * geo_weight2
        if args.need_third == 1:
            s3_time_weight = args.time_rate * s3_time_weight + args.geo_rate * geo_weight3
    threshold = args.threshold
    s1_amont = args.s1_amont
    s2_amont = args.s2_amont
    s3_amont = args.s3_amont
    time_threshold = args.cut_data


    def dfs(maps, i, j):
        """

        @param maps: two dimension array like
        @param i: coord
        @param j: coord
        """
        if i < 0 or i >= maps.shape[0] or j < 0 or j >= maps.shape[1] or maps[i][j] == False:
            return []
        maps[i][j] = False
        coord_list = []
        coord_list.append((i, j))
        for p in [-1, 0, 1]:
            for q in [-1, 0, 1]:
                if p == q and p == 0:
                    continue
                coord_list.extend(dfs(maps, i + p, j + q))
        return coord_list


    def calculate_linked_regions(t1, need_graph=False, threshold=0.2):
        mask_t1 = t1 > threshold
        if need_graph:
            import seaborn as sns
            fig = sns.heatmap(mask_t1)
            heatmap = fig.get_figure()
            heatmap.show()
        # =======================
        # 求连通域
        # =======================
        city_regions = []
        count = 0
        for i in range(mask_t1.shape[0]):
            for j in range(mask_t1.shape[1]):

                if mask_t1[i][j]:
                    coord_list = []
                    count += 1
                    coord_list.extend(dfs(mask_t1, i, j))
                    city_regions.append(coord_list)
        log("连通域的数量：{}".format(str(count)))
        linked_regions = np.zeros(mask_t1.shape)
        for i, x in enumerate(city_regions):
            for j in x:
                linked_regions[j[0]][j[1]] = i + 1
        if need_graph:
            fig = sns.heatmap(linked_regions, annot=True)
            heatmap = fig.get_figure()
            heatmap.show()
        # ==================
        # 排除重复的内部区域
        # ==================
        linked_regions_range = []
        area_max = (0, 0, 0, 0, 0)
        for i in city_regions:
            x, y = [], []
            for j in i:
                x.append(j[0])
                y.append(j[1])
            x_max = np.max(x)
            x_min = np.min(x)
            y_max = np.max(y)
            y_min = np.min(y)
            a = abs(x_max - x_min) * abs(y_max - y_min)
            if a > area_max[4]:
                area_max = [x_min, x_max, y_min, y_max, a, True]
            linked_regions_range.append([x_min, x_max, y_min, y_max, a, True])

        for i in linked_regions_range:
            if i[0] >= area_max[0] and i[1] <= area_max[1] \
                    and i[2] >= area_max[2] and i[3] <= area_max[3] and i[4] <= area_max[4]:
                if i == area_max:
                    continue
                i[5] = False

        # ================
        # 求组件范围
        # ================
        linked_regions = np.zeros(mask_t1.shape)
        ccc = 1
        for i in linked_regions_range:
            if not i[5]:
                continue
            for p in range(mask_t1.shape[0]):
                for q in range(mask_t1.shape[1]):
                    if i[0] - 1 <= p <= i[1] + 1 and i[2] - 1 <= q <= i[3] + 1 and i[5] == True:
                        linked_regions[p][q] = ccc
            ccc += 1
        log("排除包含关系之后连通域数量：{}".format(str(ccc - 1)))
        if need_graph:
            fig = sns.heatmap(linked_regions, annot=True)
            heatmap = fig.get_figure()
            heatmap.show()
        # =======================
        # 组合起来
        # =======================
        boxes = []
        coord_range = []
        for i in linked_regions_range:
            if i[5]:
                a, b, c, d = i[0] - 1 if i[0] - 1 > 0 else 0, i[1] + 1 if i[1] + 1 < t1.shape[0] else t1.shape[0] - 1, \
                             i[2] - 1 if i[2] - 1 > 0 else 0, i[3] + 1 if i[3] + 1 < t1.shape[1] else t1.shape[1] - 1
                coord_range.append([a, b, c, d,
                                    (b - a + 1) * (d - c + 1),
                                    True])
                boxes.append([abs(coord_range[-1][1] - coord_range[-1][0]) + 1,
                              abs(coord_range[-1][3] - coord_range[-1][2]) + 1])
        return boxes, coord_range


    boxes1, linked_regions_range1 = calculate_linked_regions(s1_time_weight, False, args.s1_rate)
    boxes2, linked_regions_range2 = calculate_linked_regions(s2_time_weight, False, args.s2_rate)
    boxes3, linked_regions_range3 = [], []
    if args.need_third == 1:
        boxes3, linked_regions_range3 = calculate_linked_regions(s3_time_weight, False, args.s3_rate)
    log(boxes1, boxes2, boxes3)
    log(linked_regions_range1, linked_regions_range2, linked_regions_range3)
    log([sum(j[4] for j in i) for i in [linked_regions_range1, linked_regions_range2, linked_regions_range3]])
    from ph import phspprg, phsppog
    from visualize import visualize
    from collections import namedtuple

    Rectangle = namedtuple('Rectangle', ['x', 'y', 'w', 'h'])

    boxes = []
    boxes.extend(boxes1)
    boxes.extend(boxes2)
    if args.need_third == 1:
        boxes.extend(boxes3)
    sum_area = 0
    for i in [linked_regions_range1, linked_regions_range2, linked_regions_range3]:
        for j in i:
            sum_area += j[4]
    sum_min = 999999999
    width_min = 0


    def verify(width, height, rectangles):
        for i in rectangles:
            if i.x + i.w > width or i.y + i.h > height:
                return False
        return True


    for i in range(10, (int(math.sqrt(sum_area)) + 1) * 2, 1):
        width = i
        height, rectangles = phspprg(width, boxes)
        if height + width < sum_min and verify(width, height, rectangles):
            width_min = i
            sum_min = height + width
    height, rectangles = phspprg(width_min, boxes)
    # visualize(width_min, height, rectangles)
    log("The width for min height is {}".format(str(width_min)))
    log("The height is: {}".format(height))
    width = int(width_min)
    height = int(height)

    virtual_city = np.zeros((time_threshold, width, height))
    virtual_source_coord = np.zeros((3, width, height))
    virtual_poi = np.zeros((width, height, 14))
    virtual_road = np.zeros((width * height, width * height))
    virtual_od = np.zeros((width * height, width * height))
    city_regions_expand = []
    for i in linked_regions_range1:
        city_regions_expand.append([i[0], i[1], i[2], i[3], i[4], abs(i[1] - i[0]) + 1, abs(i[3] - i[2]) + 1, 1, False])
    for i in linked_regions_range2:
        city_regions_expand.append([i[0], i[1], i[2], i[3], i[4], abs(i[1] - i[0]) + 1, abs(i[3] - i[2]) + 1, 2, False])
    for i in linked_regions_range3:
        city_regions_expand.append([i[0], i[1], i[2], i[3], i[4], abs(i[1] - i[0]) + 1, abs(i[3] - i[2]) + 1, 3, False])


    def find_city_regions(w, h):
        for i in city_regions_expand:
            if not i[-1] and i[5] == w and i[6] == h:
                i[-1] = True
                return i[-2], i[0], i[1], i[2], i[3]
        return None


    test_mask = np.zeros((width, height))
    for i in rectangles:
        res = None
        res = find_city_regions(int(i.w), int(i.h))
        across_flag = False
        if res is None:
            """
            拼接的时候可能会旋转
            """
            res = find_city_regions(int(i.h), int(i.w))
            across_flag = True
        data = None
        data_poi = None
        log(i)
        log(res)
        log(across_flag)
        if res[0] == 1:
            data = source_data
            data_poi = source_poi
        elif res[0] == 2:
            data = source_data2
            data_poi = source_poi2
        elif res[0] == 3:
            data = source_data3
            data_poi = source_poi3

        for p in range(int(i.w)):
            for q in range(int(i.h)):
                if across_flag:
                    virtual_city[:, i.x + p, i.y + q] = data[0: time_threshold, res[1] + q, res[3] + p]
                    virtual_source_coord[:, i.x + p, i.y + q] = np.array([res[1] + q, res[3] + p, res[0]])
                    data_poi = data_poi.reshape((data.shape[1], data.shape[2], 14))
                    virtual_poi[i.x + p, i.y + q, :] = data_poi[res[1] + q, res[3] + p, :]
                    test_mask[i.x + p, i.y + q] = 1
                else:
                    virtual_city[:, i.x + p, i.y + q] = data[0: time_threshold, res[1] + p, res[3] + q]
                    virtual_source_coord[:, i.x + p, i.y + q] = np.array([res[1] + p, res[3] + q, res[0]])
                    data_poi = data_poi.reshape((data.shape[1], data.shape[2], 14))
                    virtual_poi[i.x + p, i.y + q, :] = data_poi[res[1] + p, res[3] + q, :]
                    test_mask[i.x + p, i.y + q] = 1

    log()
    # import seaborn as sns
    #
    # fig = sns.heatmap(test_mask)
    # heatmap = fig.get_figure()
    # heatmap.show()

    for i in range(virtual_source_coord.shape[1] * virtual_source_coord.shape[2]):
        for j in range(virtual_source_coord.shape[1] * virtual_source_coord.shape[2]):
            m, n = idx_1d22d(i, (virtual_source_coord.shape[1], virtual_source_coord.shape[2]))
            p, q = idx_1d22d(j, (virtual_source_coord.shape[1], virtual_source_coord.shape[2]))
            if virtual_source_coord[2][m][n] == virtual_source_coord[2][p][q]:
                od = None
                road = None
                shape = None
                if virtual_source_coord[2][m][n] == 1:
                    od = source_od_adj
                    road = source_road_adj
                    shape = (source_data.shape[1], source_data.shape[2])
                elif virtual_source_coord[2][m][n] == 2:
                    od = source_od_adj2
                    road = source_road_adj2
                    shape = (source_data2.shape[1], source_data2.shape[2])
                elif virtual_source_coord[2][m][n] == 3:
                    od = source_od_adj3
                    road = source_road_adj3
                    shape = (source_data3.shape[1], source_data3.shape[2])
                else:
                    continue
                c = idx_2d_2_1d(
                    (virtual_source_coord[0][m][n], virtual_source_coord[1][m][n]
                     ), shape)
                d = idx_2d_2_1d(
                    (virtual_source_coord[0][p][q], virtual_source_coord[1][p][q]
                     ), shape)
                c = int(c)
                d = int(d)
                # log("m, n, p, q, c, d", " ", m, n, p, q, c, d)
                virtual_od[i][j] = od[c][d]
                virtual_road[i][j] = road[c][d]
    for i in range(virtual_road.shape[0]):
        virtual_road[i][i] = 1
    log()
elif args.use_linked_region == 2:
    # =========================================
    # 这里使用已经生成好的DTW进行筛选重要节点
    # =========================================

    path = "./time_weight/time_weight{}_{}_{}_{}_{}.npy"
    s1_time_weight = np.load(path.format(scity, tcity, datatype, dataname, args.data_amount)).sum(2)
    s1_time_weight, _, _ = min_max_normalize(s1_time_weight)

    s2_time_weight = np.load(path.format(scity2, tcity, datatype, dataname, args.data_amount)).sum(2)
    s2_time_weight, _, _ = min_max_normalize(s2_time_weight)

    s1_regions = []
    s2_regions = []
    s3_regions = []

    if args.need_third == 1:
        s3_time_weight = np.load(path.format(scity3, tcity, datatype, dataname, args.data_amount)).sum(2)
        s3_time_weight, _, _ = min_max_normalize(s3_time_weight)

    if args.need_geo_weight == 1:
        s1_time_weight = args.time_rate * s1_time_weight + args.geo_rate * geo_weight1
        s2_time_weight = args.time_rate * s2_time_weight + args.geo_rate * geo_weight2
        if args.need_third == 1:
            s3_time_weight = args.time_rate * s3_time_weight + args.geo_rate * geo_weight3
    threshold = args.threshold
    s1_amont = args.s1_amont
    s2_amont = args.s2_amont
    s3_amont = args.s3_amont
    time_threshold = args.cut_data


    def dfs(maps, i, j):
        """

        @param maps: two dimension array like
        @param i: coord
        @param j: coord
        """
        if i < 0 or i >= maps.shape[0] or j < 0 or j >= maps.shape[1] or maps[i][j] == False:
            return []
        maps[i][j] = False
        coord_list = []
        coord_list.append((i, j))
        for p in [-1, 0, 1]:
            for q in [-1, 0, 1]:
                if p == q and p == 0:
                    continue
                coord_list.extend(dfs(maps, i + p, j + q))
        return coord_list


    def calculate_linked_regions(t1, need_graph=False, threshold=0.2):
        mask_t1 = t1 > threshold
        if need_graph:
            import seaborn as sns
            fig = sns.heatmap(mask_t1)
            heatmap = fig.get_figure()
            heatmap.show()
        # =======================
        # 求连通域
        # =======================
        city_regions = []
        count = 0
        for i in range(mask_t1.shape[0]):
            for j in range(mask_t1.shape[1]):

                if mask_t1[i][j]:
                    coord_list = []
                    count += 1
                    coord_list.extend(dfs(mask_t1, i, j))
                    city_regions.append(coord_list)
        log("连通域的数量：{}".format(str(count)))
        linked_regions = np.zeros(mask_t1.shape)
        for i, x in enumerate(city_regions):
            for j in x:
                linked_regions[j[0]][j[1]] = i + 1
        if need_graph:
            fig = sns.heatmap(linked_regions, annot=True)
            heatmap = fig.get_figure()
            heatmap.show()
        # ==================
        # 排除重复的内部区域
        # ==================
        linked_regions_range = []
        area_max = (0, 0, 0, 0, 0)
        for i in city_regions:
            x, y = [], []
            for j in i:
                x.append(j[0])
                y.append(j[1])
            x_max = np.max(x)
            x_min = np.min(x)
            y_max = np.max(y)
            y_min = np.min(y)
            a = abs(x_max - x_min) * abs(y_max - y_min)
            if a > area_max[4]:
                area_max = [x_min, x_max, y_min, y_max, a, True]
            linked_regions_range.append([x_min, x_max, y_min, y_max, a, True])

        for i in linked_regions_range:
            if i[0] >= area_max[0] and i[1] <= area_max[1] \
                    and i[2] >= area_max[2] and i[3] <= area_max[3] and i[4] <= area_max[4]:
                if i == area_max:
                    continue
                i[5] = False

        # ================
        # 求组件范围
        # ================
        linked_regions = np.zeros(mask_t1.shape)
        ccc = 1
        for i in linked_regions_range:
            if not i[5]:
                continue
            for p in range(mask_t1.shape[0]):
                for q in range(mask_t1.shape[1]):
                    if i[0] - 1 <= p <= i[1] + 1 and i[2] - 1 <= q <= i[3] + 1 and i[5] == True:
                        linked_regions[p][q] = ccc
            ccc += 1
        log("排除包含关系之后连通域数量：{}".format(str(ccc - 1)))
        if need_graph:
            fig = sns.heatmap(linked_regions, annot=True)
            heatmap = fig.get_figure()
            heatmap.show()
        # =======================
        # 组合起来
        # =======================
        boxes = []
        coord_range = []
        for i in linked_regions_range:
            if i[5]:
                coord_range.append([i[0] - 1 if i[0] - 1 > 0 else 0,
                                    i[1] + 1 if i[1] + 1 < t1.shape[0] else t1.shape[0] - 1,
                                    i[2] - 1 if i[2] - 1 > 0 else 0,
                                    i[3] + 1 if i[3] + 1 < t1.shape[1] else t1.shape[1] - 1,
                                    (i[1] - i[0] + 3) * (i[3] - i[2] + 3),
                                    True])
                boxes.append([abs(coord_range[-1][1] - coord_range[-1][0]) + 1,
                              abs(coord_range[-1][3] - coord_range[-1][2]) + 1])
        return boxes, coord_range


    boxes1, linked_regions_range1 = calculate_linked_regions(s1_time_weight, False, args.s1_rate)
    boxes2, linked_regions_range2 = calculate_linked_regions(s2_time_weight, False, args.s2_rate)
    boxes3, linked_regions_range3 = [], []
    if args.need_third == 1:
        boxes3, linked_regions_range3 = calculate_linked_regions(s3_time_weight, False, args.s3_rate)
    boxes1_expand, boxes2_expand, boxes3_expand = [[ii[0], ii[1]] for ii in boxes1], [[ii[0], ii[1]] for ii in
                                                                                      boxes2], [[ii[0], ii[1]] for ii in
                                                                                                boxes3]
    for box_exp in [boxes1_expand, boxes2_expand, boxes3_expand]:
        for box in box_exp:
            for pp in range(len(box)):
                while box[pp] % 3 != 0:
                    box[pp] = box[pp] + 1
    np_3_3 = []
    np_3_3_poi = []
    np_3_3_coord = []
    boxes_exp = [[boxes1, boxes1_expand, linked_regions_range1, source_data, source_poi, 1],
                 [boxes2, boxes2_expand, linked_regions_range2, source_data2, source_poi2, 2]]
    if args.need_third == 1:
        boxes_exp.append([boxes3, boxes3_expand, linked_regions_range3, source_data3, source_poi3, 3])
    for _p in range(len(boxes_exp)):
        for _q in range(len(boxes_exp[_p][1])):
            for _w in range(0, boxes_exp[_p][1][_q][0], 3):
                for _h in range(0, boxes_exp[_p][1][_q][1], 3):
                    # if _w + 2 < boxes_exp[_p][0][_q][0] and _h + 2 < boxes_exp[_p][0][_q][1]:
                    temp1 = np.zeros((time_threshold, 3, 3))
                    temp2 = np.zeros((3, 3, 14))
                    temp3 = np.zeros((3, 3, 3))
                    for _m in range(3):
                        for _n in range(3):
                            _x = boxes_exp[_p][2][_q][0] + _w + _m
                            _y = boxes_exp[_p][2][_q][2] + _h + _n
                            if _x <= boxes_exp[_p][2][_q][1] and _y <= boxes_exp[_p][2][_q][3]:
                                temp3[_m, _n, :] = np.array([_x, _y, boxes_exp[_p][-1]])
                                poi = boxes_exp[_p][4]
                                data = boxes_exp[_p][3]
                                poi = poi.reshape((data.shape[1], data.shape[2], 14))
                                temp2[_m, _n, :] = poi[_x, _y, :]
                                temp1[:, _m, _n] = data[0: time_threshold, _x, _y]

                            else:
                                temp3[_m, _n, :] = np.array([-1, -1, -1])

                                temp2[_m, _n, :] = np.zeros(14)
                                temp1[:, _m, _n] = np.zeros(time_threshold)
                    np_3_3.append(temp1)
                    np_3_3_poi.append(temp2)
                    np_3_3_coord.append(temp3)

    """
    boxes_expand 就是扩张之后的3*3，其中符合boxes的直接复制，其他的赋值0
    坐标按照linked_regions_range指示
    然后按照3*3的方式去做
    """


    def is_integer(number):
        if int(number) == number:
            return True
        else:
            return False


    def is_3_product(number):
        if int(number) % 3 == 0:
            return True
        else:
            return False


    def crack(integer):
        start = int(np.sqrt(integer))
        factor = integer / start
        while not (is_integer(start) and is_integer(factor) and is_3_product(factor) and is_3_product(start)):
            start += 1
            factor = integer / start
        return int(factor), start


    shape = crack(len(np_3_3) * 9)
    log("virtual shape {}".format(str(shape)))
    virtual_city = np.zeros((time_threshold, shape[0], shape[1]))
    virtual_poi = np.zeros((shape[0], shape[1], 14))
    virtual_source_coord = np.zeros((shape[0], shape[1], 3))

    virtual_road = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    virtual_od = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    count = 0
    for i in range(0, shape[0], 3):
        for j in range(0, shape[1], 3):
            virtual_city[:, i: i + 3, j: j + 3] = np_3_3[count]
            virtual_poi[i: i + 3, j: j + 3, :] = np_3_3_poi[count]
            virtual_source_coord[i: i + 3, j: j + 3, :] = np_3_3_coord[count]
            count = count + 1
    for i in range(virtual_source_coord.shape[0] * virtual_source_coord.shape[1]):
        for j in range(virtual_source_coord.shape[0] * virtual_source_coord.shape[1]):
            m, n = idx_1d22d(i, (virtual_source_coord.shape[0], virtual_source_coord.shape[1]))
            p, q = idx_1d22d(j, (virtual_source_coord.shape[0], virtual_source_coord.shape[1]))
            if virtual_source_coord[m][n][2] == virtual_source_coord[p][q][2] and virtual_source_coord[m][n][
                2] != -1 and virtual_source_coord[m][n][2] != 0:
                od = None
                road = None
                shape = None
                if virtual_source_coord[m][n][2] == 1:
                    od = source_od_adj
                    road = source_road_adj
                    shape = (source_data.shape[1], source_data.shape[2])
                elif virtual_source_coord[m][n][2] == 2:
                    od = source_od_adj2
                    road = source_road_adj2
                    shape = (source_data2.shape[1], source_data2.shape[2])
                elif virtual_source_coord[m][n][2] == 3:
                    od = source_od_adj3
                    road = source_road_adj3
                    shape = (source_data3.shape[1], source_data3.shape[2])
                c = idx_2d_2_1d(
                    (virtual_source_coord[m][n][0], virtual_source_coord[m][n][1]
                     ), shape)
                d = idx_2d_2_1d(
                    (virtual_source_coord[p][q][0], virtual_source_coord[p][q][1]
                     ), shape)
                c = int(c)
                d = int(d)
                virtual_od[i][j] = od[c][d]
                virtual_road[i][j] = road[c][d]
    for i in range(virtual_road.shape[0]):
        virtual_road[i][i] = 1

long_term_save["virtual_source_coord"] = virtual_source_coord
long_term_save["virtual_city"] = virtual_city
long_term_save["virtual_poi"] = virtual_poi
long_term_save["virtual_road"] = virtual_road
long_term_save["virtual_od"] = virtual_od
virtual_poi = virtual_poi.reshape((virtual_city.shape[1] * virtual_city.shape[2], 14))
lng_virtual, lat_virtual = virtual_city.shape[1], virtual_city.shape[2]
mask_virtual = virtual_city.sum(0) > 0
th_mask_virtual = torch.Tensor(mask_virtual.reshape(1, lng_virtual, lat_virtual)).to(device)
log("%d valid regions in virtual" % np.sum(mask_virtual))
virtual_emb_label = masked_percentile_label(virtual_city.sum(0).reshape(-1), mask_virtual.reshape(-1))
lag = [-6, -5, -4, -3, -2, -1]
if args.normal == 1:
    virtual_city, virtual_max, virtual_min = min_max_normalize(virtual_city)
virtual_train_x, virtual_train_y, virtual_val_x, virtual_val_y, virtual_test_x, virtual_test_y \
    = split_x_y(virtual_city, lag, val_num=int(virtual_city.shape[0] / 6), test_num=int(virtual_city.shape[0] / 6))
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
virtual_road_adj = virtual_road
d_sim = np.dot(virtual_od, virtual_od.transpose())
s_sim = np.dot(virtual_od.transpose(), virtual_od)
d_norm = np.sqrt((virtual_od ** 2).sum(1))
s_norm = np.sqrt((virtual_od ** 2).sum(0))
d_sim /= (np.outer(d_norm, d_norm) + 1e-5)
s_sim /= (np.outer(s_norm, s_norm) + 1e-5)
s_adj = np.copy(s_sim)
d_adj = np.copy(d_sim)
n_nodes = s_adj.shape[0]
for i in range(n_nodes):
    s_adj[i, np.argsort(s_sim[i, :])[:-args.topk]] = 0
    s_adj[np.argsort(s_sim[:, i])[:-args.topk], i] = 0
    d_adj[i, np.argsort(d_sim[i, :])[:-args.topk]] = 0
    d_adj[np.argsort(d_sim[:, i])[:-args.topk], i] = 0
virtual_s_adj, virtual_d_adj, virtual_od_adj = s_adj, d_adj, virtual_od
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
emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
emb_optimizer = optim.Adam(emb_param_list, lr=args.learning_rate, weight_decay=args.weight_decay)
mvgat_optimizer = optim.Adam(list(mvgat.parameters()) + list(fusion.parameters()), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
# 元学习部分
meta_optimizer = optim.Adam(scoring.parameters(), lr=args.outerlr, weight_decay=args.weight_decay)
best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999
best_test_mape = 999
p_bar.process(5, 1, 5)


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


def train_emb_epoch2():
    # loss， 460*64， 5*460*64
    loss_source, fused_emb_s, embs_s = forward_emb(virtual_graphs, virtual_norm_poi, virtual_od_adj,
                                                   virtual_poi_cos)
    loss_target, fused_emb_t, embs_t = forward_emb(target_graphs, target_norm_poi, target_od_adj,
                                                   target_poi_cos)

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
    time.time() - start_time, pretrain_emb_epoch, np.mean(emb_losses), np.mean(mmd_losses),
    np.mean(edge_losses)))
with torch.no_grad():
    views = mvgat(virtual_graphs, torch.Tensor(virtual_norm_poi).to(device))
    # 融合模块指的是把多图的特征融合
    fused_emb_s, _ = fusion(views)
    views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
    fused_emb_t, _ = fusion(views)

emb_s = fused_emb_s.cpu().numpy()[mask_virtual.reshape(-1)]
emb_t = fused_emb_t.cpu().numpy()[mask_target.reshape(-1)]
logreg = LogisticRegression(max_iter=500)
cvscore_s = cross_validate(logreg, emb_s, virtual_emb_label)['test_score'].mean()
cvscore_t = cross_validate(logreg, emb_t, target_emb_label)['test_score'].mean()
log("[%.2fs]Pretraining embedding, source cvscore %.4f, target cvscore %.4f" % \
    (time.time() - start_time, cvscore_s, cvscore_t))
log()

def load_all_adj(device):
    dirs = "./data/{}/{}_roads.npy"
    ny, chi, dc = None, None, None
    for i in ["NY", "CHI", "DC"]:
        t = dirs.format(i, i)
        t = np.load(t)
        t = t.reshape((t.shape[0] * t.shape[1], t.shape[0] * t.shape[1]))
        t = np.where(t >= 1, 1, t)
        t = add_self_loop(t)
        for m in range(t.shape[0]):
            for n in range(t.shape[1]):
                a, b = idx_1d22d(m, t.shape)
                c, d = idx_1d22d(n, t.shape)
                dis = abs(a - c) + abs(b - d)
                if t[m][n] - 0 > 1e-6 and dis != 0:
                    t[m][n] = t[m][n] / dis
        if t.shape[0] == 460:
            ny = t
        elif t.shape[0] == 476:
            chi = t
        elif t.shape[0] == 420:
            dc = t

    return torch.tensor(ny).to(device), torch.tensor(chi).to(device), torch.tensor(dc).to(device)


adj_pems04, adj_pems07, adj_pems08 = load_all_adj(device)
vec_pems04 = vec_pems07 = vec_pems08 = None, None, None
tttttttttttt = virtual_road.copy()
tttttttttttt = np.where(tttttttttttt >= 1, 1, tttttttttttt)
tttttttttttt = add_self_loop(tttttttttttt)
for m in range(tttttttttttt.shape[0]):
    for n in range(tttttttttttt.shape[1]):
        a, b = idx_1d22d(m, tttttttttttt.shape)
        c, d = idx_1d22d(n, tttttttttttt.shape)
        dis = abs(a - c) + abs(b - d)
        if tttttttttttt[m][n] - 0 > 1e-6 and dis != 0:
            tttttttttttt[m][n] = tttttttttttt[m][n] / dis
adj_virtual = torch.tensor(tttttttttttt).to(device)
cur_dir = os.getcwd()
dc = np.load("./data/DC/{}DC_{}.npy".format(args.dataname, args.datatype))
dcmask = dc.sum(0) > 0
th_maskdc = dcmask.reshape(1, 420)
chi = np.load("./data/CHI/{}CHI_{}.npy".format(args.dataname, args.datatype))
chimask = chi.sum(0) > 0
th_maskchi = chimask.reshape(1, 476)
ny = np.load("./data/NY/{}NY_{}.npy".format(args.dataname, args.datatype))
nymask = ny.sum(0) > 0
th_maskny = nymask.reshape(1, 460)
cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]

pems04_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems04',
                               '{}_vecdim.pkl'.format(args.vec_dim))
pems07_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems07',
                               '{}_vecdim.pkl'.format(args.vec_dim))
pems08_emb_path = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems08',
                               '{}_vecdim.pkl'.format(args.vec_dim))
v_p = os.path.join('{}'.format(cur_dir), 'embeddings', 'node2vec', 'pems08',
                   '{}{}{}{}{}{}_vecdim.pkl'.format(args.vec_dim, args.dataname, args.datatype,
                                                    str(args.s1_rate).replace(".", ""),
                                                    str(args.s2_rate).replace(".", ""),
                                                    str(args.s3_rate).replace(".", "")))

for i in [pems04_emb_path, pems07_emb_path, pems08_emb_path, v_p]:
    a = i.split(os.path.sep)
    b = []
    for i in a:
        if "pkl" in i:
            continue
        else:
            b.append(i)
    local_path_generate(folder_name=os.path.sep.join(b), create_folder_only=True)


def generate_vector(adj, args):
    graph = nx.DiGraph(adj)
    spl = dict(nx.all_pairs_shortest_path_length(graph))
    from node2vec import Node2Vec as node2vec
    n2v = node2vec(G=graph, distance=spl, emb_size=args.vec_dim, length_walk=args.walk_length,
                   num_walks=args.num_walks, window_size=5, batch=4, p=args.p, q=args.q,
                   workers=(int(cpu_count() / 2)))
    n2v = n2v.train()

    gfeat = []
    for i in range(len(adj)):
        nodeivec = n2v.wv.get_vector(str(i))
        gfeat.append(nodeivec)
    g = torch.tensor(np.array(gfeat))

    return g, gfeat
if os.path.exists(pems04_emb_path):
    log(f'Loading pems04 embedding...')
    vec_pems04 = torch.load(pems04_emb_path, map_location='cpu')
    vec_pems04 = vec_pems04.to(device)
else:
    log(f'Generating pems04 embedding...')
    args.dataset = '4'
    vec_pems04, _ = generate_vector(adj_pems04.cpu().numpy(), args)
    vec_pems04 = vec_pems04.to(device)
    log(f'Saving pems04 embedding...')
    torch.save(vec_pems04.cpu(), pems04_emb_path)

if os.path.exists(pems07_emb_path):
    log(f'Loading pems07 embedding...')
    vec_pems07 = torch.load(pems07_emb_path, map_location='cpu')
    vec_pems07 = vec_pems07.to(device)
else:
    log(f'Generating pems07 embedding...')
    args.dataset = '7'
    vec_pems07, _ = generate_vector(adj_pems07.cpu().numpy(), args)
    vec_pems07 = vec_pems07.to(device)
    log(f'Saving pems07 embedding...')
    torch.save(vec_pems07.cpu(), pems07_emb_path)

if os.path.exists(pems08_emb_path):
    log(f'Loading pems08 embedding...')
    vec_pems08 = torch.load(pems08_emb_path, map_location='cpu')
    vec_pems08 = vec_pems08.to(device)
else:
    log(f'Generating pems08 embedding...')
    args.dataset = '8'
    vec_pems08, _ = generate_vector(adj_pems08.cpu().numpy(), args)
    vec_pems08 = vec_pems08.to(device)
    log(f'Saving pems08 embedding...')
    torch.save(vec_pems08.cpu(), pems08_emb_path)

if os.path.exists(v_p):
    log(f'Loading v embedding...')
    vec_virtual = torch.load(v_p, map_location='cpu')
    vec_virtual = vec_virtual.to(device)
else:
    log(f'Generating virtual embedding...')
    args.dataset = '8'
    vec_virtual, _ = generate_vector(virtual_road, args)
    vec_virtual = vec_virtual.to(device)
    log(f'Saving virtual embedding...')
    torch.save(vec_virtual.cpu(), v_p)
log(
    f'Successfully load embeddings, 4: {vec_pems04.shape}, 7: {vec_pems07.shape}, 8: {vec_pems08.shape}, vec_virtual:{vec_virtual.shape}')

domain_criterion = torch.nn.NLLLoss()
domain_classifier = Domain_classifier_DG(num_class=2, encode_dim=args.enc_dim)

domain_classifier = domain_classifier.to(device)
state = g = None, None

batch_seen = 0
cur_dir = os.getcwd()
if cur_dir[-2:] == 'sh':
    cur_dir = cur_dir[:-2]
assert args.models in ["DASTNet"]

bak_epoch = args.epoch
bak_val = args.val
bak_test = args.test
type = 'pretrain'
pretrain_model_path = os.path.join('{}'.format(cur_dir), 'pretrained', 'transfer_models',
                                   '{}'.format(args.dataset), '{}_prelen'.format(args.pre_len),
                                   'flow_model4_{}_epoch_{}_{}_{}_{}_{}_{}.pkl'.format(
                                       args.models, args.epoch, args.dataname, args.datatype,
                                       str(args.s1_rate).replace(".", ""),
                                       str(args.s2_rate).replace(".", ""),
                                       str(args.s3_rate).replace(".", ""))
                                   )

a = pretrain_model_path.split(os.path.sep)
b = []
for i in a:
    if "pkl" not in i:
        b.append(i)
local_path_generate(os.path.sep.join(b), create_folder_only=True)
vec_pems04 = vec_virtual
adj_pems04 = adj_virtual
args.dataset = "8"

def net_fix(source, y, weight, mask, fast_weights, bn_vars, net):
    pred_source = net.functional_forward(vec_pems04, vec_pems07, vec_pems08, source, True, fast_weights, bn_vars, bn_training=True, data_set="4")
    label = y.reshape((pred_source.shape[0], -1, pred_source.shape[2]))
    mask = mask.reshape((1, mask.shape[1] * mask.shape[2], 1))
    fast_loss = torch.abs(pred_source - label)[:, mask.view(-1).bool(),:]
    fast_loss = (fast_loss * weight.view((1, -1, 1))).mean(0).sum()
    a = [(i, torch.autograd.grad(fast_loss, fast_weights[i], create_graph=True, allow_unused=True)) for i in fast_weights.keys()]
    grads = {}
    used_fast_weight = OrderedDict()
    for i in a:
        if i[1][0] is not None:
            grads[i[0]] = i[1][0]
            used_fast_weight[i[0]] = fast_weights[i[0]]

    for name, grad in zip(grads.keys(), grads.values()):
        fast_weights[name] = fast_weights[name] - args.innerlr * grad
    return fast_loss, fast_weights, bn_vars
def meta_train_epoch(s_embs, t_embs, net):
    meta_query_losses = []
    for meta_ep in range(args.outeriter):
        fast_losses = []
        fast_weights, bn_vars = get_weights_bn_vars(net)
        source_weights = scoring(s_embs, t_embs, th_mask_virtual, th_mask_target)
        # inner loop on source, pre-train with weights
        for meta_it in range(args.sinneriter):
            s_x1, s_y1 = batch_sampler((torch.Tensor(virtual_train_x), torch.Tensor(virtual_train_y)),
                                       args.batch_size)
            s_x1 = s_x1.reshape((s_x1.shape[0], s_x1.shape[1], s_x1.shape[2] * s_x1.shape[3]))
            s_y1 = s_y1.reshape((s_y1.shape[0], s_y1.shape[1], s_y1.shape[2] * s_y1.shape[3]))
            s_x1 = s_x1.to(device)
            s_y1 = s_y1.to(device)
            fast_loss, fast_weights, bn_vars = net_fix(s_x1, s_y1, source_weights, th_mask_virtual, fast_weights, bn_vars, net)
            fast_losses.append(fast_loss.item())

        # inner loop on target, simulate fine-tune
        # 模拟微调和源训练都是在训练net预测网络，并没有提及权重和特征
        for meta_it in range(args.tinneriter):
            t_x, t_y = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size)
            t_x = t_x.reshape((t_x.shape[0], t_x.shape[1], t_x.shape[2] * t_x.shape[3]))
            t_y = t_y.reshape((t_y.shape[0], t_y.shape[1], t_y.shape[2] * t_y.shape[3]))

            t_x = t_x.to(device)
            t_y = t_y.to(device)
            pred_source = net.functional_forward(vec_pems04, vec_pems07, vec_pems08, t_x, True, fast_weights, bn_vars, bn_training=True, data_set="8")
            label = t_y.reshape((pred_source.shape[0], -1, pred_source.shape[2]))
            mask = th_mask_target
            mask = mask.reshape((1, mask.shape[1] * mask.shape[2], 1))
            fast_loss = torch.abs(pred_source - label)[:, mask.view(-1).bool(),:]
            fast_loss = fast_loss.mean(0).sum()
            a = [(i, torch.autograd.grad(fast_loss, fast_weights[i], create_graph=True, allow_unused=True)) for i in fast_weights.keys()]
            grads = {}
            used_fast_weight = OrderedDict()
            for i in a:
                if i[1][0] is not None:
                    grads[i[0]] = i[1][0]
                    used_fast_weight[i[0]] = fast_weights[i[0]]

            for name, grad in zip(grads.keys(), grads.values()):
                fast_weights[name] = fast_weights[name] - args.innerlr * grad

        q_losses = []
        target_iter = max(args.sinneriter, args.tinneriter)
        for k in range(3):
            # query loss
            x_q = None
            y_q = None
            temp_mask = None

            x_q, y_q = batch_sampler((torch.Tensor(target_train_x), torch.Tensor(target_train_y)), args.batch_size)
            temp_mask = th_mask_target
            x_q = x_q.reshape((x_q.shape[0], x_q.shape[1], x_q.shape[2] * x_q.shape[3]))
            y_q = y_q.reshape((y_q.shape[0], y_q.shape[1], y_q.shape[2] * y_q.shape[3]))

            x_q = x_q.to(device)
            y_q = y_q.to(device)
            pred_source = net.functional_forward(vec_pems04, vec_pems07, vec_pems08, x_q, True, fast_weights, bn_vars, bn_training=True, data_set="8")
            label = y_q.reshape((pred_source.shape[0], -1, pred_source.shape[2]))
            mask = temp_mask.reshape((1, temp_mask.shape[1] * temp_mask.shape[2], 1))
            fast_loss = torch.abs(pred_source - label)[:, mask.view(-1).bool(),:]
            fast_loss = fast_loss.mean(0).sum()

            q_losses.append(fast_loss)
        q_loss = torch.stack(q_losses).mean()
        weights_mean = source_weights.mean()
        meta_loss = q_loss + weights_mean * args.weight_reg
        meta_optimizer.zero_grad()
        meta_loss.backward(inputs=list(scoring.parameters()), retain_graph=True)
        torch.nn.utils.clip_grad_norm_(scoring.parameters(), max_norm=2)
        meta_optimizer.step()
        meta_query_losses.append(q_loss.item())
    return np.mean(meta_query_losses)
def select_mask(a):
    if a == 420:
        return th_maskdc
    elif a == 476:
        return th_maskchi
    elif a == 460:
        return th_maskny
# def test(testloader):
#     if type == 'pretrain':
#         domain_classifier.eval()
#     model.eval()
#
#     test_mape, test_rmse, test_mae = list(), list(), list()
#
#     for i, (feat, label) in enumerate(testloader.get_iterator()):
#         feat = torch.FloatTensor(feat).to(device)
#         label = torch.FloatTensor(label).to(device)
#         mask = select_mask(feat.shape[2])
#         if mask is None:
#             mask = mask_virtual
#         if torch.sum(scaler.inverse_transform(label)) <= 0.001:
#             continue
#
#         pred = model(vec_pems04, vec_pems07, vec_pems08, feat, True)
#         pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
#         label = label.reshape((-1, label.size(2)))
#
#         mae_test, rmse_test, mape_test = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),maskp=mask, weight=None)
#
#         test_mae.append(mae_test.item())
#         test_rmse.append(rmse_test.item())
#         test_mape.append(mape_test.item())
#
#     test_rmse = np.mean(test_rmse)
#     test_mae = np.mean(test_mae)
#     test_mape = np.mean(test_mape)
#
#     return test_mae, test_rmse, test_mape
#
# def train(dur, model, optimizer, total_step, start_step, need_road, weight, mask, tdl, vdl, types, tstl):
#     t0 = time.time()
#     train_mae, val_mae, train_rmse, val_rmse, train_acc = list(), list(), list(), list(), list()
#     train_correct = 0
#
#     model.train()
#     if types == 'pretrain':
#         domain_classifier.train()
#
#     for i, (feat, label) in enumerate(tdl.get_iterator()):
#         maskt = select_mask(feat.shape[2])
#         if maskt is None:
#             maskt = mask
#         Reverse = False
#         if i > 0:
#             if train_acc[-1] > 0.333333:
#                 Reverse = True
#         p = float(i + start_step) / total_step
#         constant = 2. / (1. + np.exp(-10 * p)) - 1
#
#         feat = torch.FloatTensor(feat).to(device)
#         label = torch.FloatTensor(label).to(device)
#         if torch.sum(scaler.inverse_transform(label)) <= 0.001:
#             continue
#
#         optimizer.zero_grad()
#         if args.models not in ['DCRNN', 'STGCN', 'HA']:
#             if types == 'pretrain':
#                 pred, shared_pems04_feat, shared_pems07_feat, shared_pems08_feat = model(vec_pems04, vec_pems07,vec_pems08, feat, False)
#             elif types == 'fine-tune':
#                 pred = model(vec_pems04, vec_pems07, vec_pems08, feat, False)
#
#             pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
#             label = label.reshape((-1, label.size(2)))
#
#             if types == 'pretrain':
#                 pems04_pred = domain_classifier(shared_pems04_feat, constant, Reverse)
#
#                 pems08_pred = domain_classifier(shared_pems08_feat, constant, Reverse)
#
#                 pems04_label = 0 * torch.ones(pems04_pred.shape[0]).long().to(device)
#
#                 pems08_label = 1 * torch.ones(pems08_pred.shape[0]).long().to(device)
#
#                 pems04_pred_label = pems04_pred.max(1, keepdim=True)[1]
#                 pems04_correct = pems04_pred_label.eq(pems04_label.view_as(pems04_pred_label)).sum()
#
#
#                 pems08_pred_label = pems08_pred.max(1, keepdim=True)[1]
#                 pems08_correct = pems08_pred_label.eq(pems08_label.view_as(pems08_pred_label)).sum()
#
#                 pems04_loss = domain_criterion(pems04_pred, pems04_label)
#
#                 pems08_loss = domain_criterion(pems08_pred, pems08_label)
#
#                 domain_loss = pems04_loss + pems08_loss
#
#         if types == 'pretrain':
#             train_correct = pems04_correct + pems08_correct
#
#         mae_train, rmse_train, mape_train = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label), maskp=maskt, weight=weight)
#
#         if types == 'pretrain':
#             loss = (mae_train + args.beta * (args.theta * domain_loss))
#         elif types == 'fine-tune':
#             loss = mae_train
#
#         loss.backward()
#         optimizer.step()
#
#         train_mae.append(mae_train.item())
#         train_rmse.append(rmse_train.item())
#
#         if types == 'pretrain':
#             train_acc.append(train_correct.item() / 855)
#         elif types == 'fine-tune':
#             train_acc.append(0)
#
#     if types == 'pretrain':
#         domain_classifier.eval()
#     model.eval()
#
#     for i, (feat, label) in enumerate(vdl.get_iterator()):
#         feat = torch.FloatTensor(feat).to(device)
#         label = torch.FloatTensor(label).to(device)
#         if torch.sum(scaler.inverse_transform(label)) <= 0.001:
#             continue
#
#         pred = model(vec_pems04, vec_pems07, vec_pems08, feat, True)
#         pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
#         label = label.reshape((-1, label.size(2)))
#         mae_val, rmse_val, mape_val = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),maskp=maskt, weight=weight)
#         val_mae.append(mae_val.item())
#         val_rmse.append(rmse_val.item())
#
#     test_mae, test_rmse, test_mape = test(tstl)
#     dur.append(time.time() - t0)
#     return np.mean(train_mae), np.mean(train_rmse), np.mean(val_mae), np.mean(
#         val_rmse), test_mae, test_rmse, test_mape, np.mean(train_acc)
#
# def model_train(args, model, optimizer, trainloader, valloader ,testloader, types):
#     eps = 0
#     cnt = 0
#     if types == 'pretrain':
#         p_bar = process_bar(final_prompt="预训练完成", unit="epoch")
#         p_bar.process(0, 1, args.epoch)
#         eps = args.epoch
#     else:
#         p_bar = process_bar(final_prompt="微调完成", unit="epoch")
#         p_bar.process(0, 1, args.fine_epoch)
#         eps = args.fine_epoch
#     for ep in range(eps):
#         model.train()
#         # mvgat.train()
#         # fusion.train()
#         # scoring.train()
#         # # train embeddings
#         # if types == 'pretrain':
#         #     emb_losses = []
#         #     mmd_losses = []
#         #     edge_losses = []
#         #     for emb_ep in range(5):
#         #         loss_emb_, loss_mmd_, loss_et_ = train_emb_epoch2()
#         #         emb_losses.append(loss_emb_)
#         #         mmd_losses.append(loss_mmd_)
#         #         edge_losses.append(loss_et_)
#         #     with torch.no_grad():
#         #         views = mvgat(virtual_graphs, torch.Tensor(virtual_norm_poi).to(device))
#         #         fused_emb_s, _ = fusion(views)
#         #         views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
#         #         fused_emb_t, _ = fusion(views)
#         #     avg_q_loss = meta_train_epoch(fused_emb_s, fused_emb_t, model)
#         #     with torch.no_grad():
#         #         source_weights = scoring(fused_emb_s, fused_emb_t, th_mask_virtual, th_mask_target)
#         #     if ep == 0:
#         #         source_weights_ma = torch.ones_like(source_weights, device=device, requires_grad=False)
#         #     source_weights_ma = ma_param * source_weights_ma + (1 - ma_param) * source_weights
#         dur = []
#         epoch = 1
#         best = 999999999999999
#         acc = list()
#
#         step_per_epoch = trainloader.get_num_batch()
#         total_step = 200 * step_per_epoch
#
#         source_weights_ma = None
#         start_step = ep * step_per_epoch
#         if types == 'fine-tune' and ep > 1000:
#             args.val = True
#         if types == 'fine-tune':
#             source_weights_ma = None
#         else:
#             log(source_weights_ma.mean() if source_weights_ma is not None else 0)
#         mae_train, rmse_train, mae_val, rmse_val, mae_test, rmse_test, mape_test, train_acc = train(dur, model, optimizer, total_step, start_step, args.need_road, source_weights_ma, mask_virtual, trainloader, valloader, types, testloader)
#         log(f'Epoch {ep} | acc_train: {train_acc: .4f} | mae_train: {mae_train: .4f} | rmse_train: {rmse_train: .4f} | mae_val: {mae_val: .4f} | rmse_val: {rmse_val: .4f} | mae_test: {mae_test: .4f} | rmse_test: {rmse_test: .4f} | mape_test: {mape_test: .4f} | Time(s) {dur[-1]: .4f}')
#         epoch += 1
#         acc.append(train_acc)
#         if mae_val <= best:
#             if types == 'fine-tune' and mae_val > 0.001:
#                 best = mae_val
#                 state = dict([('model', copy.deepcopy(model.state_dict())),
#                               ('optim', copy.deepcopy(optimizer.state_dict())),
#                               ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
#                 cnt = 0
#             elif types == 'pretrain':
#                 best = mae_val
#                 state = dict([('model', copy.deepcopy(model.state_dict())),
#                               ('optim', copy.deepcopy(optimizer.state_dict())),
#                               ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
#                 cnt = 0
#         else:
#             cnt += 1
#         if types == 'pretrain':
#             p_bar.process(ep + 1, 1, args.epoch)
#         else:
#             p_bar.process(ep + 1, 1, args.fine_epoch)
#         if cnt == args.patience or epoch > args.epoch:
#             log(f'Stop!!')
#             log(f'Avg acc: {np.mean(acc)}')
#             break
#     log("Optimization Finished!")
#     return state

def test(testdl):
    if type == 'pretrain':
        domain_classifier.eval()
    model.eval()

    test_mape, test_rmse, test_mae = list(), list(), list()

    for i, (feat, label) in enumerate(testdl.get_iterator()):
        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        mask = select_mask(feat.shape[2])
        if torch.sum(scaler.inverse_transform(label)) <= 0.001:
            continue

        pred = model(vec_pems04, vec_pems07, vec_pems08, feat, True, args.need_road)
        pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
        label = label.reshape((-1, label.size(2)))

        mae_test, rmse_test, mape_test = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),
                                                     maskp=mask)

        test_mae.append(mae_test.item())
        test_rmse.append(rmse_test.item())
        test_mape.append(mape_test.item())

    test_rmse = np.mean(test_rmse)
    test_mae = np.mean(test_mae)
    test_mape = np.mean(test_mape)

    return test_mae, test_rmse, test_mape


def model_train(args, model, optimizer, tdl, vdl, testdl, type):
    dur = []
    epoch = 1
    best = 999999999999999
    acc = list()

    step_per_epoch = train_dataloader.get_num_batch()
    total_step = 200 * step_per_epoch

    while epoch <= args.epoch:
        start_step = epoch * step_per_epoch
        if type == 'fine-tune' and epoch > 1000:
            args.val = True
        mae_train, rmse_train, mae_val, rmse_val, mae_test, rmse_test, mape_test, train_acc = train(dur, model,
                                                                                                    optimizer,
                                                                                                    total_step,
                                                                                                    start_step,
                                                                                                    args.need_road,
                                                                                                    tdl, vdl, testdl, type)
        log(f'Epoch {epoch} | acc_train: {train_acc: .4f} | mae_train: {mae_train: .4f} | rmse_train: {rmse_train: .4f} | mae_val: {mae_val: .4f} | rmse_val: {rmse_val: .4f} | mae_test: {mae_test: .4f} | rmse_test: {rmse_test: .4f} | mape_test: {mape_test: .4f} | Time(s) {dur[-1]: .4f}')
        epoch += 1
        acc.append(train_acc)
        if mae_val <= best:
            if type == 'fine-tune' and mae_val > 0.001:
                best = mae_val
                state = dict([('model', copy.deepcopy(model.state_dict())),
                              ('optim', copy.deepcopy(optimizer.state_dict())),
                              ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
                cnt = 0
            elif type == 'pretrain':
                best = mae_val
                state = dict([('model', copy.deepcopy(model.state_dict())),
                              ('optim', copy.deepcopy(optimizer.state_dict())),
                              ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
                cnt = 0
        else:
            cnt += 1
        if cnt == args.patience or epoch > args.epoch:
            print(f'Stop!!')
            print(f'Avg acc: {np.mean(acc)}')
            break
    print("Optimization Finished!")
    return state

def train(dur, model, optimizer, total_step, start_step, need_road, tdl, vdl, testdl, type):
    t0 = time.time()
    train_mae, val_mae, train_rmse, val_rmse, train_acc = list(), list(), list(), list(), list()
    train_correct = 0

    model.train()
    if type == 'pretrain':
        domain_classifier.train()

    for i, (feat, label) in enumerate(tdl.get_iterator()):
        mask = select_mask(feat.shape[2])
        Reverse = False
        if i > 0:
            if train_acc[-1] > 0.333333:
                Reverse = True
        p = float(i + start_step) / total_step
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        if torch.sum(scaler.inverse_transform(label)) <= 0.001:
            continue

        optimizer.zero_grad()
        if args.models not in ['DCRNN', 'STGCN', 'HA']:
            if type == 'pretrain':
                pred, shared_pems04_feat, shared_pems07_feat, shared_pems08_feat = model(vec_pems04, vec_pems07,
                                                                                         vec_pems08, feat, False,
                                                                                         need_road)
            elif type == 'fine-tune':
                pred = model(vec_pems04, vec_pems07, vec_pems08, feat, False, need_road)

            pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
            label = label.reshape((-1, label.size(2)))

            if type == 'pretrain':
                pems04_pred = domain_classifier(shared_pems04_feat, constant, Reverse)
                pems08_pred = domain_classifier(shared_pems08_feat, constant, Reverse)

                pems04_label = 0 * torch.ones(pems04_pred.shape[0]).long().to(device)
                pems08_label = 1 * torch.ones(pems08_pred.shape[0]).long().to(device)

                pems04_pred_label = pems04_pred.max(1, keepdim=True)[1]
                pems04_correct = pems04_pred_label.eq(pems04_label.view_as(pems04_pred_label)).sum()

                pems08_pred_label = pems08_pred.max(1, keepdim=True)[1]
                pems08_correct = pems08_pred_label.eq(pems08_label.view_as(pems08_pred_label)).sum()

                pems04_loss = domain_criterion(pems04_pred, pems04_label)

                pems08_loss = domain_criterion(pems08_pred, pems08_label)

                domain_loss = pems04_loss + pems08_loss

        if type == 'pretrain':
            train_correct = pems04_correct + pems08_correct

        mae_train, rmse_train, mape_train = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),
                                                        maskp=mask)

        if type == 'pretrain':
            loss = mae_train + args.beta * (args.theta * domain_loss)
        elif type == 'fine-tune':
            loss = mae_train

        loss.backward()
        optimizer.step()

        train_mae.append(mae_train.item())
        train_rmse.append(rmse_train.item())

        if type == 'pretrain':
            train_acc.append(train_correct.item() / 855)
        elif type == 'fine-tune':
            train_acc.append(0)

    if type == 'pretrain':
        domain_classifier.eval()
    model.eval()

    for i, (feat, label) in enumerate(vdl.get_iterator()):
        mask = select_mask(feat.shape[2])
        feat = torch.FloatTensor(feat).to(device)
        label = torch.FloatTensor(label).to(device)
        if torch.sum(scaler.inverse_transform(label)) <= 0.001:
            continue
        pred = model(vec_pems04, vec_pems07, vec_pems08, feat, True, need_road)
        pred = pred.transpose(1, 2).reshape((-1, feat.size(2)))
        label = label.reshape((-1, label.size(2)))
        mae_val, rmse_val, mape_val = masked_loss(scaler.inverse_transform(pred), scaler.inverse_transform(label),
                                                  maskp=mask)
        val_mae.append(mae_val.item())
        val_rmse.append(rmse_val.item())

    test_mae, test_rmse, test_mape = test(testdl)
    dur.append(time.time() - t0)
    return np.mean(train_mae), np.mean(train_rmse), np.mean(val_mae), np.mean(
        val_rmse), test_mae, test_rmse, test_mape, np.mean(train_acc)


if os.path.exists(pretrain_model_path):
    log(f'Loading pretrained model at {pretrain_model_path}')
    state = torch.load(pretrain_model_path, map_location='cpu')
else:
    log(f'No existing pretrained model at {pretrain_model_path}')
    args.val = args.test = False
    datasets = ["4"]
    dataset_bak = args.dataset
    labelrate_bak = args.labelrate
    args.labelrate = 100
    dataset_count = 0

    for dataset in [item for item in datasets if item not in [dataset_bak]]:
        dataset_count = dataset_count + 1

        log(
            f'\n\n****************************************************************************************************************')
        log(f'dataset: {dataset}, model: {args.models}, pre_len: {args.pre_len}, labelrate: {args.labelrate}')
        log(
            f'****************************************************************************************************************\n\n')

        if dataset == '4':
            g = vec_pems04
        elif dataset == '7':
            g = vec_pems07
        elif dataset == '8':
            g = vec_pems08

        args.dataset = dataset


        def load_graphdata_channel3(args, feat_dir, time, scaler=None, visualize=False, cut=False):
            data = virtual_city
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
            log(data.shape)
            if time:
                num_data, num_sensor = data.shape
                data = np.expand_dims(data, axis=-1)
                data = data.tolist()

                for i in range(num_data):
                    time = (i % 288) / 288
                    for j in range(num_sensor):
                        data[i][j].append(time)

                data = np.array(data)

            max_val = np.max(data)
            time_len = data.shape[0]
            seq_len = args.seq_len
            pre_len = args.pre_len
            split_ratio = args.split_ratio
            train_size = int(time_len * split_ratio)
            val_size = int(time_len * (1 - split_ratio) / 3)
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:time_len]
            log(data.shape)
            if args.labelrate != 100:
                import random
                new_train_size = int(train_size * args.labelrate / 100)
                start = random.randint(0, train_size - new_train_size - 1)
                train_data = train_data[start:start + new_train_size]

            train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()

            for i in range(len(train_data) - seq_len - pre_len):
                train_X.append(np.array(train_data[i: i + seq_len]))
                train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
            for i in range(len(val_data) - seq_len - pre_len):
                val_X.append(np.array(val_data[i: i + seq_len]))
                val_Y.append(np.array(val_data[i + seq_len: i + seq_len + pre_len]))
            for i in range(len(test_data) - seq_len - pre_len):
                test_X.append(np.array(test_data[i: i + seq_len]))
                test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))

            if visualize:
                test_X = test_X[-288:]
                test_Y = test_Y[-288:]

            if args.labelrate != 0:
                train_X = np.array(train_X)
                train_Y = np.array(train_Y)
            val_X = np.array(val_X)
            val_Y = np.array(val_Y)
            test_X = np.array(test_X)
            test_Y = np.array(test_Y)

            if args.labelrate != 0:
                max_xtrain = np.max(train_X)
                max_ytrain = np.max(train_Y)
            max_xval = np.max(val_X)
            max_yval = np.max(val_Y)
            max_xtest = np.max(test_X)
            max_ytest = np.max(test_Y)

            if args.labelrate != 0:
                min_xtrain = np.min(train_X)
                min_ytrain = np.min(train_Y)
            min_xval = np.min(val_X)
            min_yval = np.min(val_Y)
            min_xtest = np.min(test_X)
            min_ytest = np.min(test_Y)

            if args.labelrate != 0:
                max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
                min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

                # scaler = StandardScaler(mean=train_X[..., 0].mean(), std=train_X[..., 0].std())
                scaler = StandardScaler(mean=train_X.mean(), std=train_X.std())

                train_X = scaler.transform(train_X)
                train_Y = scaler.transform(train_Y)
            else:
                max_speed = max(max_xval, max_yval, max_xtest, max_ytest)
                min_speed = min(min_xval, min_yval, min_xtest, min_ytest)

            val_X = scaler.transform(val_X)
            val_Y = scaler.transform(val_Y)
            test_X = scaler.transform(test_X)
            test_Y = scaler.transform(test_Y)

            if args.labelrate != 0:
                max_xtrain = np.max(train_X)
                max_ytrain = np.max(train_Y)
            max_xval = np.max(val_X)
            max_yval = np.max(val_Y)
            max_xtest = np.max(test_X)
            max_ytest = np.max(test_Y)

            if args.labelrate != 0:
                min_xtrain = np.min(train_X)
                min_ytrain = np.min(train_Y)
            min_xval = np.min(val_X)
            min_yval = np.min(val_Y)
            min_xtest = np.min(test_X)
            min_ytest = np.min(test_Y)

            if args.labelrate != 0:
                max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
                min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

            else:
                max_speed = max(max_xval, max_yval, max_xtest, max_ytest)
                min_speed = min(min_xval, min_yval, min_xtest, min_ytest)
            if cut:
                train_X = train_X[-args.data_amount * 24:, :, :]
                train_Y = train_Y[-args.data_amount * 24:, :, :]
            return train_X, train_Y, val_X, val_Y, test_X, test_Y, max_val, scaler


        train_dataloader, val_dataloader, test_dataloader, adj, max_speed, scaler = load_data(args)
        train_X, train_Y, val_X, val_Y, test_X, test_Y, max_speed, scaler = load_graphdata_channel3(args, "", False,scaler,visualize=False)
        log([i.shape for i in [train_X, train_Y, val_X, val_Y, test_X, test_Y]])
        train_dataloader = MyDataLoader(torch.FloatTensor(train_X), torch.FloatTensor(train_Y),
                                        batch_size=args.batch_size)
        val_dataloader = MyDataLoader(torch.FloatTensor(val_X), torch.FloatTensor(val_Y),
                                      batch_size=args.batch_size)
        test_dataloader = MyDataLoader(torch.FloatTensor(test_X), torch.FloatTensor(test_Y),
                                       batch_size=args.batch_size)
        adj = 0
        model = DASTNet(input_dim=args.vec_dim, hidden_dim=args.hidden_dim, encode_dim=args.enc_dim,
                        device=device, batch_size=args.batch_size, etype=args.etype, pre_len=args.pre_len,
                        dataset=args.dataset, ft_dataset=dataset_bak,
                        adj_pems04=adj_pems04, adj_pems07=adj_pems07, adj_pems08=adj_pems08).to(device)
        optimizer = optim.SGD([{'params': model.parameters()},
                               {'params': domain_classifier.parameters()}], lr=args.learning_rate, momentum=0.8)

        if dataset_count != 1:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optim'])

        state = model_train(args, model, optimizer, train_dataloader, val_dataloader, test_dataloader, "pretrain")

    log(f'Saving model to {pretrain_model_path} ...')
    torch.save(state, pretrain_model_path)
    args.dataset = dataset_bak
    args.labelrate = labelrate_bak
    args.val = bak_val
    args.test = bak_test

type = 'fine-tune'
args.epoch = args.fine_epoch

log(f'\n\n*******************************************************************************************')
log(
    f'dataset: {args.dataset}, model: {args.models}, pre_len: {args.pre_len}, labelrate: {args.labelrate}, seed: {args.division_seed}')
log(f'*******************************************************************************************\n\n')

if args.dataset == '4':
    g = vec_pems04
elif args.dataset == '7':
    g = vec_pems07
elif args.dataset == '8':
    g = vec_pems08

train_dataloader, val_dataloader, test_dataloader, adj, max_speed, scaler = load_data(args, cut=True)
model = DASTNet(input_dim=args.vec_dim, hidden_dim=args.hidden_dim, encode_dim=args.enc_dim,
                device=device, batch_size=args.batch_size, etype=args.etype, pre_len=args.pre_len,
                dataset=args.dataset, ft_dataset=args.dataset,
                adj_pems04=adj_pems04, adj_pems07=adj_pems07, adj_pems08=adj_pems08).to(device)
optimizer = optim.SGD([{'params': model.parameters()},
                       {'params': domain_classifier.parameters()}], lr=args.learning_rate, momentum=0.8)
model.load_state_dict(state['model'])
optimizer.load_state_dict(state['optim'])

if args.labelrate != 0:
    test_state = model_train(args, model, optimizer, train_dataloader, val_dataloader, test_dataloader, "fine-tune")
    model.load_state_dict(test_state['model'])
    optimizer.load_state_dict(test_state['optim'])



test_mae, test_rmse, test_mape = test(test_dataloader)
log(f'mae: {test_mae: .4f}, rmse: {test_rmse: .4f}, mape: {test_mape * 100: .4f}\n\n')
if args.c != "default":
    if args.need_remark == 1:
        record.update(record_id, get_timestamp(),
                      "%.4f,%.4f,%.4f" %
                      (test_rmse, test_mae, test_mape * 100),
                      remark="{}C {} {} {} {}".format("2" if args.need_third == 0 else "3", str(args.data_amount),
                                                      args.dataname, args.datatype, args.machine_code))
    else:
        record.update(record_id, get_timestamp(),
                      "%.4f,%.4f, %.4f" %
                      (test_rmse, test_mae, test_mape * 100),
                      remark="{}".format(args.machine_code))



