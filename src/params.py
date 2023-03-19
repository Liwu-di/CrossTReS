# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 15:11
# @Author  : 银尘
# @FileName: params.py
# @Software: PyCharm
# @Email   ：liwudi@liwudi.fun
import argparse
import os
import random

import numpy as np
import torch

"""
文件的参数
"""


def params():
    parser = argparse.ArgumentParser()
    # 源城市
    parser.add_argument('--scity', type=str, default='NY')
    parser.add_argument('--scity2', type=str, default='CHI')
    parser.add_argument('--scity3', type=str, default='BJ')
    # 目标城市
    parser.add_argument('--tcity', type=str, default='DC')
    # 数据集名称
    parser.add_argument('--dataname', type=str, default='Taxi', help='Within [Bike, Taxi]')
    # 数据类型
    parser.add_argument('--datatype', type=str, default='pickup', help='Within [pickup, dropoff]')
    # 尝试减小，看显存能不能撑住 32 -> 16
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_time_sample', type=int, default=8)
    # 模型
    parser.add_argument("--model", type=str, default='STNet_nobn', help='Within [STResNet, STNet, STNet_nobn]')
    # 学习率
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    # 权重
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    # 100回合跑下来数据有问题，改成40epoch看看，论文也是这个
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of source training epochs')
    parser.add_argument('--num_tuine_epochs', type=int, default=80, help='Number of fine tuine epochs')
    # gpu设备序号
    parser.add_argument('--gpu', type=int, default=0)
    # 随机种子 不知道是干嘛的
    parser.add_argument('--seed', type=int, default=-1, help='Random seed. -1 means do not manually set. ')
    # 数据量
    parser.add_argument('--data_amount', type=int, default=0, help='0: full data, 30/7/3 correspond to days of data')
    # 内循环 源训练数量
    parser.add_argument('--sinneriter', type=int, default=3,
                        help='Number of inner iterations (source) for meta learning')
    # 内循环 微调数量
    parser.add_argument('--tinneriter', type=int, default=1,
                        help='Number of inner iterations (target) for meta learning')
    # 内循环元学习学习率
    parser.add_argument('--innerlr', type=float, default=5e-5, help='Learning rate for inner loop of meta-learning')
    # 外循环数量
    parser.add_argument('--outeriter', type=int, default=20, help='Number of outer iterations for meta-learning')
    # 外循环学习率
    parser.add_argument('--outerlr', type=float, default=1e-4, help='Learning rate for the outer loop of meta-learning')
    # 前k个参数
    parser.add_argument('--topk', type=int, default=15)
    # 多城市中第二个城市需要被融合的区域数量
    parser.add_argument('--topk_m', type=int, default=15)
    # 最大平均误差参数 ，也就是beta1
    parser.add_argument('--mmd_w', type=float, default=2, help='mmd weight')
    # 边缘分类器参数， beta2
    parser.add_argument('--et_w', type=float, default=2, help='edge classifier weight')
    # 源域权重的移动平均参数
    parser.add_argument("--ma_coef", type=float, default=0.6, help='Moving average parameter for source domain weights')
    # 源域权重的正则化器。
    parser.add_argument("--weight_reg", type=float, default=1e-3, help="Regularizer for the source domain weights.")
    # 预训练回合数
    parser.add_argument("--pretrain_iter", type=int, default=-1,
                        help='Pre-training iterations per pre-training epoch. ')
    # 是否启用邻域
    parser.add_argument("--near", type=int, default=1,
                        help='0 启用 1 不启用 ')
    # 是否启用全局平均还是分位数平均
    parser.add_argument("--mean", type=int, default=0,
                        help='0 全局 1 分位数 ')

    # 是否启用修正余弦相似度
    parser.add_argument("--fix_cos", type=int, default=0,
                        help='0 是 1 否 ')
    # 预测网络学习率
    parser.add_argument("--pred_lr", type=float, default=8e-4, help="prediction learning rate")
    parser.add_argument("--c", type=str, default="default", help="research record")
    parser.add_argument("--scoring", type=int, default=1, help="score method")
    parser.add_argument("--time_meta", type=int, default=1, help="time_meta")
    parser.add_argument("--meta_batch_size", type=int, default=16, help="time_meta")
    parser.add_argument("--is_st_weight_static", type=int, default=1, help="0不使用，1使用时序评分")
    parser.add_argument("--time_score_weight", type=float, default=1.0, help="0不使用，1使用时序评分")
    parser.add_argument("--space_score_weight", type=float, default=1.0, help="0不使用，1使用时序评分")
    parser.add_argument("--node_domain_adapt", type=str, default="MMD", help="在MMD和adversarial选择")
    parser.add_argument("--fine_tuning_lr", type=float, default=8e-4, help="微调时的学习率")
    parser.add_argument("--need_third", type=int, default=1, help="为1使用三个城市")
    parser.add_argument("--alin_month", type=int, default=0, help="为1将三个城市月份统一")
    parser.add_argument("--node_adapt", type=str, default="MMD", help="[MMD, DT]")
    parser.add_argument("--pretrain", type=str, default="supervise", help="[supervise, meta]")
    parser.add_argument("--tasks_val_rate", type=float, default=0.3)
    parser.add_argument("--tasks_test_rate", type=float, default=0.5)
    parser.add_argument("--need_weight", type=int, default=0)
    parser.add_argument("--cut_data", type=int, default=3312)
    parser.add_argument("--mae_rate", type=float, default=1)
    parser.add_argument("--rmse_rate", type=float, default=1)
    parser.add_argument("--zero_rate", type=float, default=0.01)
    parser.add_argument("--flat_rate", type=float, default=20)
    parser.add_argument("--road_epoch", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--accuracy", type=float, default=0.2)
    parser.add_argument("--s1_amont", type=int, default=200)
    parser.add_argument("--s2_amont", type=int, default=200)
    parser.add_argument("--s3_amont", type=int, default=200)
    parser.add_argument("--s1_rate", type=float, default=0.1)
    parser.add_argument("--s2_rate", type=float, default=0.1)
    parser.add_argument("--s3_rate", type=float, default=0.1)
    parser.add_argument("--test_mode_path", type=str)
    parser.add_argument("--use_linked_region", type=int, default=1, help="0使用8邻域构建，1使用8-连通域构建")
    parser.add_argument("--need_geo_weight", type=int, default=0, help="1使用geo weight")
    parser.add_argument("--geo_dis", type=str, default="MMD", help="within [MMD, KL, DTW, wasserstein, JS]")
    parser.add_argument("--need_remark", type=int, default=0)
    parser.add_argument("--geo_rate", type=float, default=1.0, help="rate of geo")
    parser.add_argument("--time_rate", type=float, default=1.0, help="rate of time")
    parser.add_argument("--machine_code", type=str, default="", help="code of machine")

    args = parser.parse_args()

    if args.seed != -1:
        # seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，
        # 如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
        # random.seed(something)只能是一次有效
        # seed( ) 用于指定随机数生成时所用算法开始的整数值。
        # 1.如果使用相同的seed( )值，则每次生成的随即数都相同；
        # 2.如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
        # 3.设置的seed()值仅一次有效
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    return args
