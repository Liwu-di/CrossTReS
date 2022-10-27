# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 21:12
# @Author  : 银尘
# @FileName: tttt.py
# @Software: PyCharm
# @Email   ：liwudi@liwudi.fun
import argparse
import ast
import os
import time

import sshtunnel
from PaperCrawlerUtil.common_util import *
from PaperCrawlerUtil.research_util import ResearchRecord

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
parser.add_argument("--near", type=int, default=0,
                    help='0 启用 1 不启用 ')
# 是否启用全局平均还是分位数平均
parser.add_argument("--mean", type=int, default=0,
                    help='0 全局 1 分位数 ')

# 是否启用修正余弦相似度
parser.add_argument("--fix_cos", type=int, default=0,
                    help='0 是 1 否 ')
# 预测网络学习率
parser.add_argument("--pred_lr", type=float, default=8e-4, help="prediction learning rate")
parser.add_argument("--c", type=str, default="", help="research record")
args = parser.parse_args()


if len(args.c) > 0:
    c = ast.literal_eval(args.c)
    check_ssl = True if c.get("ssl_ip") is not None and c.get("ssl_admin") is not None and \
                        c.get("ssl_pwd") is not None and c.get("ssl_db_port") is not None \
                        and c.get("ssl_port") is not None else False
    log(check_ssl)
    record = ResearchRecord(**c)
    log(c)
    p = record.insert(os.path.abspath(""), get_timestamp())
    log(p)
    time.sleep(1)
    record.update(p, get_timestamp(), "fsdsds")
    record.__del__()
