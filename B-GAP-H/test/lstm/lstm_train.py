import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import Callback
# import keras.backend.tensorflow_backend as KTF
from keras.backend import set_session
import tensorflow as tf
import os
import keras.callbacks
import matplotlib.pyplot as plt

# 设定为自增长
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
set_session(session)


def create_dataset(data, traintime_num, pertime_num, train_proportion, vehicle_count, times):
    '''
    对数据进行处理
    首先是对于多个预测维度，全部划分为3预测1的形式
    其次多次获取到的数据分开
    '''
    dim = data.shape[0] # time_count * times * vehicle_count
    train_X, train_Y = [], []
    for i in range(int(vehicle_count * train_proportion)):
        j = 0
        for time in range(times):
            if time > 0:
                j = j + traintime_num
            temp_X1, temp_Y1 = [], []
            while dim/times*time <= i+(traintime_num+j)*vehicle_count < dim/times*(time + 1):
                vehicle_X1 = data[i+j*vehicle_count:i+(traintime_num+j)*vehicle_count:vehicle_count, :]
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


def NormalizeMult(data, set_range):
    '''
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2 * data.shape[1], dtype='float64')
    normalize = normalize.reshape(data.shape[1], 2)

    for i in range(0, data.shape[1]):
        if set_range == True:
            list = data[:, i]
            listlow, listhigh = np.percentile(list, [0, 100])
        else:
            if i == 0:
                listlow = -90
                listhigh = 90
            else:
                listlow = -180
                listhigh = 180

        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta

    return data, normalize


def trainModel(train_X, train_Y):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()
    model.add(LSTM(
        120,
        input_shape=(train_X.shape[1], train_X.shape[2]),
        return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(
        120,
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        train_Y.shape[1]))
    model.add(Activation("relu"))

    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    model.fit(train_X, train_Y, epochs=100, batch_size=64, verbose=1)
    model.summary()

    return model


if __name__ == "__main__":
    traintime_num = 3
    pertime_num = 1
    # set_range = False
    set_range = True
    train_proportion = 1 # the proportion of the vehicle to train
    times = 400
    during = 40
    agg_rate = 20

    # 读入时间序列的文件数据
    npy = np.load('./npy/agg_'+ str(agg_rate) +'%_'+ str(during) +'s_'+ str(times) +'times.npy')
    # print(agg)
    # print(npy)

    # 转DataFrame
    data = np.array(npy)
    time_count = int(data.shape[0]/times)
    vehicle_count = data.shape[1]
    data = np.resize(data, (time_count * times * vehicle_count, 7))
    # data = pd.DataFrame(data=data[0:, 0:],
    #                   columns=["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
    #                   )
    # data = pd.read_csv('20080403010747.txt', sep=',').iloc[:, 0:2].values
    print("时间维度：{0}，车辆数量：{1}, 运行次数：{2}".format(time_count, vehicle_count, times))
    # print(data)

    # 画样本数据库
    # plt.scatter(data[:, 1], data[:, 0], c='b', marker='o')#, label='traj_A')
    # plt.legend(loc='upper left')
    # plt.grid()
    # plt.show()

    # # 归一化
    # data, normalize = NormalizeMult(data, set_range)
    # # print(normalize)

    # 生成训练数据
    train_X, train_Y, test_X, test_Y = create_dataset(data, traintime_num, pertime_num, train_proportion, vehicle_count, times)
    train_X = train_X.reshape(train_X.shape[0]*train_X.shape[1], train_X.shape[2], train_X.shape[3])
    train_Y = train_Y.reshape(train_Y.shape[0]*train_Y.shape[1], train_Y.shape[3])
    # train_Y = train_Y.reshape(train_Y.shape[0], train_Y.shape[3])
    # test_X = test_X.reshape(test_X.shape[0], test_X.shape[2], test_X.shape[3])
    # test_Y = test_Y.reshape(test_Y.shape[0], test_Y.shape[2], test_Y.shape[3])
    # print(type(train_X))
    print("x:", train_X.shape)
    print("y:", train_Y.shape)
    # print("x:", test_X)
    # print("y:", test_Y)

    # # 训练模型
    model = trainModel(train_X, train_Y)
    # y_hat = model.predict(train_X)
    # print(y_hat)
    # print(train_Y)
    # model.summary()
    # loss, acc = model.evaluate(train_X, train_Y, verbose=2)
    # print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    #
    # # 保存模型
    # #np.save("./traj_model_trueNorm.npy", normalize)
    model.save('./model/intpre_num'+ str(vehicle_count) +'_agg'+ str(agg_rate) +'%.h5')