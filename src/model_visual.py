# -*- coding: utf-8 -*-
# @Time    : 2022/9/15 10:38
# @Author  : 银尘
# @FileName: model_visual.py
# @Software: PyCharm
# @Email   ：liwudi@liwudi.fun

""""
STNet_nobn(
  (conv1): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (layers): ModuleList(
    (0): ResUnit_nobn(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1): ResUnit_nobn(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (2): ResUnit_nobn(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (lstm): LSTM(64, 128)
  (linear1): Linear(in_features=256, out_features=64, bias=True)
  (linear2): Linear(in_features=64, out_features=1, bias=True)
)
"""
import numpy

"""
MVGAT(
  (multi_gats): ModuleList(
    (0): ModuleList(
      (0): GATConv(
        (fc): Linear(in_features=14, out_features=128, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=14, out_features=128, bias=False)
      )
      (1): GATConv(
        (fc): Linear(in_features=14, out_features=128, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=14, out_features=128, bias=False)
      )
      (2): GATConv(
        (fc): Linear(in_features=14, out_features=128, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=14, out_features=128, bias=False)
      )
      (3): GATConv(
        (fc): Linear(in_features=14, out_features=128, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=14, out_features=128, bias=False)
      )
      (4): GATConv(
        (fc): Linear(in_features=14, out_features=128, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=14, out_features=128, bias=False)
      )
    )
    (1): ModuleList(
      (0): GATConv(
        (fc): Linear(in_features=128, out_features=64, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=128, out_features=64, bias=False)
      )
      (1): GATConv(
        (fc): Linear(in_features=128, out_features=64, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=128, out_features=64, bias=False)
      )
      (2): GATConv(
        (fc): Linear(in_features=128, out_features=64, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=128, out_features=64, bias=False)
      )
      (3): GATConv(
        (fc): Linear(in_features=128, out_features=64, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=128, out_features=64, bias=False)
      )
      (4): GATConv(
        (fc): Linear(in_features=128, out_features=64, bias=False)
        (feat_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=128, out_features=64, bias=False)
      )
    )
  )
)
"""

"""
EdgeTypeDiscriminator(
  (edge_network): Sequential(
    (0): Linear(in_features=128, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=5, bias=True)
  )
)
"""

from PaperCrawlerUtil.document_util import *


def calculate_std_deviation(data: List[float]) -> float:
    ndarr = numpy.array(data)
    std = ndarr.std()
    return std


def std(a):
    max = 0
    index = 0
    value = 0
    for i in range(len(a)):
        std = calculate_std_deviation(a[i])
        log("第{}个列表的标准差为{}".format(str(i), str(std)))
        if std > max:
            index = i
            value = std
            max = value
    log("max={}:{}".format(str(index), str(value)))


if __name__ == "__main__":
    cooperatePdf(path="D:\\1")

