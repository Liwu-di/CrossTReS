# -*- coding: utf-8 -*-
# @Time    : 2022/12/22 19:44
# @Author  : 银尘
# @FileName: test_three_city.py
# @Software: PyCharm
# @Email   ：liwudi@liwudi.fun
import argparse
import ast
import time
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
p_bar = process_bar(final_prompt="初始化准备完成", unit="part")
long_term_save = {}
args = params()
long_term_save["args"] = args.__str__()
bj2016 = np.load("../data/TaxiBJ16.npy")
bj2016 = bj2016[:,:,:,0]
bj2016_one_hour = []
i = 0
while i < bj2016.shape[0] - 1:
    bj2016_one_hour.append(bj2016[i] + bj2016[i+1])
    i = i + 2
bj2016_one_hour = np.array(bj2016_one_hour)
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

mmd = MMD_loss()
num_gat_layers = 2
in_dim = 14
hidden_dim = 64
emb_dim = 64
num_heads = 2
mmd_w = args.mmd_w
et_w = args.et_w
ma_param = args.ma_coef


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

weight1 = None
weight2 = None
weightbj = None
if args.is_st_weight_static == 1:
    s1_time_t = np.load(
        "./time_weight/time_weight{}_{}_{}_{}_{}.npy".format(scity, tcity, datatype, dataname, args.data_amount))
    s2_time_t = np.load(
        "./time_weight/time_weight{}_{}_{}_{}_{}.npy".format(scity2, tcity, datatype, dataname, args.data_amount))
    s1_time_t, _, __ = min_max_normalize(s1_time_t.sum(axis=2))
    log(s1_time_t.shape, _, __)
    s2_time_t, _, __ = min_max_normalize(s2_time_t.sum(axis=2))
    log(s2_time_t.shape, _, __)
    weightbj = np.load(
        "./time_weight/time_weight{}_{}_{}_{}_{}.npy".format("bj2016", tcity, datatype, dataname, args.data_amount))
    weightbj, _, __ = min_max_normalize(weightbj.sum(axis=2))
    log(weightbj.shape, _, __)
    weight1 = s1_time_t
    weight2 = s2_time_t
else:
    weight1 = None
    weight2 = None
    weightbj = None


writer = SummaryWriter("log-{}-batch-{}-name-{}-type-{}-model-{}-amount-{}-topk-{}-time-{}".
                       format("多城市{} and {}-{}".format(args.scity, args.scity2, args.tcity), args.batch_size,
                              args.dataname,
                              args.datatype, args.model, args.data_amount, args.topk, get_timestamp(split="-")))
target_train_val_loss = []
target_train_test_loss = []
for ep in range(num_epochs):
    net.train()
    source_loss = train_epoch(net, source_loader, pred_optimizer, mask=th_mask_source,
                              num_iters=args.pretrain_iter, weights=weight1)
    if args.need_third != 2:
        source_loss2 = train_epoch(net, source_loader2, pred_optimizer, mask=th_mask_source2,
                                   num_iters=args.pretrain_iter, weights=weight2)
    if args.need_third == 1:
        source_loss3 = train_epoch(net, bj_loader, pred_optimizer, mask=th_mask_sourcebj,
                                   num_iters=args.pretrain_iter, weights=weightbj)
        avg_source_loss3 = np.mean(source_loss3)
    avg_source_loss = np.mean(source_loss)
    # avg_source_loss2 = np.mean(source_loss2)

    avg_target_loss = evaluate(net, target_train_loader, spatial_mask=th_mask_target)[0]
    net.eval()
    rmse_val, mae_val, target_val_losses = evaluate(net, target_val_loader, spatial_mask=th_mask_target)
    rmse_s_val, mae_s_val, source_val_losses = evaluate(net, source_loader, spatial_mask=th_mask_source)
    # rmse_s_val2, mae_s_val2, source_val_losses2 = evaluate(net, source_loader2, spatial_mask=th_mask_source2)
    log(
        "Epoch %d, source validation rmse %.4f, mae %.4f" % (ep, rmse_s_val * (smax - smin), mae_s_val * (smax - smin)))
    # log(
    #     "Epoch %d, source validation rmse %.4f, mae %.4f" % (
    #         ep, rmse_s_val2 * (smax2 - smin2), mae_s_val * (smax2 - smin2)))
    log("Epoch %d, target validation rmse %.4f, mae %.4f" % (
        ep, rmse_val * (max_val - min_val), mae_val * (max_val - min_val)))
    log()
    sums = 0
    for i in range(len(target_val_losses)):
        sums = sums + target_val_losses[i].mean(0).sum().item()
    writer.add_scalar("train target val loss", sums, ep)
    sums = 0
    for i in range(len(source_val_losses)):
        sums = sums + source_val_losses[i].mean(0).sum().item()
    writer.add_scalar("train source val loss", sums, ep)
    p_bar.process(0, 1, num_epochs + num_tuine_epochs)

for ep in range(num_epochs, num_tuine_epochs + num_epochs):
    # fine-tuning
    net.train()
    avg_loss = train_epoch(net, target_train_loader, pred_optimizer, mask=th_mask_target)
    log('[%.2fs]Epoch %d, target pred loss %.4f' % (time.time() - start_time, ep, np.mean(avg_loss)))
    writer.add_scalar("target pred loss", np.mean(avg_loss), ep - num_epochs)
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
    log()
    p_bar.process(0, 1, num_epochs + num_tuine_epochs)
long_term_save["target_train_val_loss"] = target_train_val_loss
long_term_save["target_train_test_loss"] = target_train_test_loss

log("Best test rmse %.4f, mae %.4f" % (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val)))
root_dir = local_path_generate(
    "./model/{}".format(
        "{}-batch-{}-{}-{}-{}-amount-{}-topk-{}-time-{}".format(
            "多城市{}and{}-{}".format(args.scity, args.scity2, args.tcity),
            args.batch_size, args.dataname, args.datatype, args.model, args.data_amount,
            args.topk, get_timestamp(split="-")
        )
    ), create_folder_only=True)

save_obj(long_term_save,
         local_path_generate("experiment_data",
                             "data_{}.collection".format(
                                 "{}-batch-{}-{}-{}-{}-amount-{}-time-{}".format(
                                     "多城市{}and{}-{}".format(args.scity, args.scity2, args.tcity),
                                     args.batch_size, args.dataname, args.datatype, args.model, args.data_amount,
                                     get_timestamp(split="-")
                                 )
                             )
                             )
         )
record.update(record_id, get_timestamp(),
              "%.4f,%.4f" %
              (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val)))
