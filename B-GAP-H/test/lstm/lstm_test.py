import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import Callback
# import keras.backend.tensorflow_backend as KTF
from keras.backend import set_session
import tensorflow as tf
import pandas as pd
import os
import keras.callbacks
import matplotlib.pyplot as plt
import copy

# 设定为自增长
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
set_session(session)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()


def reshape_y_hat(y_hat, dim):
    re_y = []
    i = 0
    while i < len(y_hat):
        tmp = []
        for j in range(dim):
            tmp.append(y_hat[i + j])
        i = i + dim
        re_y.append(tmp)
    re_y = np.array(re_y, dtype='float64')
    return re_y


# 多维反归一化
def FNormalizeMult(data, normalize):
    data = np.array(data, dtype='float64')
    # 列
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        print("listlow, listhigh, delta", listlow, listhigh, delta)
        # 行
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i] * delta + listlow

    return data


# 使用训练数据的归一化
def NormalizeMultUseData(data, normalize):
    for i in range(0, data.shape[1]):

        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow

        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta

    return data


from math import sin, asin, cos, radians, fabs, sqrt

EARTH_RADIUS = 6371  # 地球平均半径，6371km


# 计算两个经纬度之间的直线距离
def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(lat0, lng0, lat1, lng1):
    # "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance

def create_dataset(data, traintime_num, pertime_num, train_proportion, vehicle_count):
    '''
    对数据进行处理
    '''
    dim = data.shape[0]
    train_X, train_Y = [], []
    for i in range(int(vehicle_count * train_proportion)):
        temp_X1, temp_Y1 = [], []
        j = 0
        while i+(traintime_num+j)*vehicle_count < dim:
            vehicle_X1 = data[i+j*vehicle_count:(traintime_num+j)*vehicle_count:vehicle_count, :]
            temp_X1.append(vehicle_X1)
            vehicle_Y1 = data[i+(traintime_num+j)*vehicle_count:i+(traintime_num+j+pertime_num)*vehicle_count:vehicle_count, 1:3]
            temp_Y1.append(vehicle_Y1)
            j += 1
        train_X.append(temp_X1)
        train_Y.append(temp_Y1)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    test_X, test_Y = [], []
    for i in range(int(vehicle_count * train_proportion), vehicle_count):
        temp_X2, temp_Y2 = [], []
        j = 0
        while i+(traintime_num+j)*vehicle_count < dim:
            vehicle_X2 = data[i + j * vehicle_count:(traintime_num + j) * vehicle_count:vehicle_count, :]
            temp_X2.append(vehicle_X2)
            vehicle_Y2 = data[i+(traintime_num + j) * vehicle_count:i+(traintime_num + j + pertime_num) * vehicle_count:vehicle_count, 1:3]
            temp_Y2.append(vehicle_Y2)
            j += 1
        test_X.append(temp_X2)
        test_Y.append(temp_Y2)
    test_X = np.array(test_X, dtype='float64')
    test_Y = np.array(test_Y, dtype='float64')

    return train_X, train_Y, test_X, test_Y

if __name__ == '__main__':
    # test_num = 6
    # per_num = 1
    # data_all = pd.read_csv('20080403010747.txt', sep=',').iloc[-2 * (test_num + per_num):-1 * (test_num + per_num),
    #            0:2].values
    # data_all.dtype = 'float64'
    #
    # data = copy.deepcopy(data_all[:-per_num, :])
    # y = data_all[-per_num:, :]

    # # #归一化
    # normalize = np.load("./traj_model_trueNorm.npy")
    # data = NormalizeMultUseData(data, normalize)

    traintime_num = 3
    pertime_num = 1
    set_range = True
    train_proportion = 0.8  # the proportion of the vehicle to train

    # 读入时间序列的文件数据
    npy = np.load('aggressive.npy')
    # 转DataFrame
    data = np.array(npy)
    time_count = data.shape[0]
    vehicle_count = data.shape[1]
    data = np.resize(data, (time_count * vehicle_count, 7))

    model = load_model("./traj_model_120.h5")
    train_X, train_Y, test_X, test_Y = create_dataset(data, traintime_num, pertime_num, train_proportion, vehicle_count)
    # train_X = train_X.reshape(train_X.shape[0], train_X.shape[2], train_X.shape[3])
    # train_Y = train_Y.reshape(train_Y.shape[0], train_Y.shape[2], train_Y.shape[3])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[2], test_X.shape[3])
    test_Y = test_Y.reshape(test_Y.shape[0], test_Y.shape[3])
    print(test_X)


    # print(test_X)
    # print(test_Y)
    # test_X = data.reshape(1, data.shape[0], data.shape[1])
    # y_hat = model.predict(test_X)
    # print(y_hat)
    # y_hat = y_hat.reshape(y_hat.shape[1])
    # y_hat = reshape_y_hat(y_hat, 2)
    #
    # # 反归一化
    # y_hat = FNormalizeMult(y_hat, normalize)
    # print("predict: {0}\ntrue：{1}".format(y_hat, y))
    # print('预测均方误差：', mse(y_hat, test_Y))
    # print('预测直线距离：{:.4f} KM'.format(get_distance_hav(y_hat[0, 0], y_hat[0, 1], y[0, 0], y[0, 1])))
    #
    # # 画测试样本数据库
    # p1 = plt.scatter(data_all[:-per_num, 1], data_all[:-per_num, 0], c='b', marker='o', label='traj_A')
    # p2 = plt.scatter(y_hat[:, 1], y_hat[:, 0], c='r', marker='o', label='pre')
    # p3 = plt.scatter(y[:, 1], y[:, 0], c='g', marker='o', label='pre_true')
    # plt.legend(loc='upper left')
    # plt.grid()
    # plt.show()