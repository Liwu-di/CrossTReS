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
    # 目标城市
    parser.add_argument('--tcity', type=str, default='DC')
    # 数据集名称
    parser.add_argument('--dataname', type=str, default='Taxi', help='Within [Bike, Taxi]')
    # 数据类型
    parser.add_argument('--datatype', type=str, default='pickup', help='Within [pickup, dropoff]')
    # 尝试减小，看显存能不能撑住 32 -> 16
    parser.add_argument('--batch_size', type=int, default=16)
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
