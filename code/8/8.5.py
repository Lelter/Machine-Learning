# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:50:11 2020

@author: MS
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置出图显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def decision_sdumps_MaxInfoGain(X, Y):
    # 基学习器---决策树桩
    # 以信息增益最大来选择划分属性和划分点
    m, n = X.shape  # 样本数和特征数
    results = []  # 存储各个特征下的最佳划分点,左分支取值，右分支取值，信息增益
    for i in range(n):  # 遍历各个候选特征
        x = X[:, i]  # 样本在该特征下的取值
        x_values = np.unique(x)  # 当前特征的所有取值
        ts = (x_values[1:] + x_values[:-1]) / 2  # 候选划分点
        Gains = []  # 存储各个划分点下的信息增益
        for t in ts:
            Gain = 0
            Y_left = Y[x <= t]  # 左分支样本的标记
            Dl = len(Y_left)  # 左分支样本数
            p1 = sum(Y_left == 1) / Dl  # 左分支正样本比例
            p0 = sum(Y_left == -1) / Dl  # 左分支负样本比例
            Gain += Dl / m * (np.log2(p1 ** p1) + np.log2(p0 ** p0))

            Y_right = Y[x > t]  # 右分支样本的标记
            Dr = len(Y_right)  # 右分支总样本数
            p1 = sum(Y_right == 1) / Dr  # 右分支正样本比例
            p0 = sum(Y_right == -1) / Dr  # 右分支负样本比例
            Gain += Dr / m * (np.log2(p1 ** p1) + np.log2(p0 ** p0))
            Gains.append(Gain)
        best_t = ts[np.argmax(Gains)]  # 当前特征下的最佳划分点
        best_gain = max(Gains)  # 当前特征下的最佳信息增益
        left_value = (sum(Y[x <= best_t]) >= 0) * 2 - 1  # 左分支取值(多数类的类别)
        right_value = (sum(Y[x > best_t]) >= 0) * 2 - 1  # 右分支取值(多数类的类别)
        results.append([best_t, left_value, right_value, best_gain])
    results = np.array(results)
    df = np.argmax(results[:, -1])  # df表示divide_feature，划分特征
    h = [df] + list(results[df, :3])  # 划分特征,划分点,左枝取值，右枝取值
    return h


def predict(H, X1, X2):
    # 预测结果
    # 仅X1和X2两个特征,X1和X2同维度
    pre = np.zeros(X1.shape)
    for h in H:
        df, t, lv, rv = h  # 划分特征,划分点,左枝取值，右枝取值
        X = X1 if df == 0 else X2
        pre += (X <= t) * lv + (X > t) * rv
    return np.sign(pre)


# >>>>>西瓜数据集3.0α
X = np.array([[0.697, 0.46], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
              [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
              [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.36, 0.37],
              [0.593, 0.042], [0.719, 0.103]])
Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
m = len(Y)
# >>>>>Bagging
T = 20
H = []  # 存储各个决策树桩，
# 每行为四元素列表，分别表示划分特征,划分点,左枝取值，右枝取值
H_pre = np.zeros(m)  # 存储每次迭代后H对于训练集的预测结果
error = []  # 存储每次迭代后H的训练误差
for t in range(T):
    boot_strap_sampling = np.random.randint(0, m, m)  # 产生m个随机数
    Xbs = X[boot_strap_sampling]  # 自助采样
    Ybs = Y[boot_strap_sampling]  # 自助采样
    h = decision_sdumps_MaxInfoGain(Xbs, Ybs)  # 训练基学习器
    H.append(h)  # 存入基学习器
    # 计算并存储训练误差
    df, t, lv, rv = h  # 基学习器参数
    Y_pre_h = (X[:, df] <= t) * lv + (X[:, df] > t) * rv  # 基学习器预测结果
    H_pre += Y_pre_h  # 更新集成预测结果
    error.append(sum(((H_pre >= 0) * 2 - 1) != Y) / m)  # 当前集成预测的训练误差
H = np.array(H)

# >>>>>绘制训练误差变化曲线
plt.title('训练误差的变化')
plt.plot(range(1, T + 1), error, 'o-', markersize=2)
plt.xlabel('基学习器个数')
plt.ylabel('错误率')
plt.show()
# >>>>>观察结果
x1min, x1max = X[:, 0].min(), X[:, 0].max()
x2min, x2max = X[:, 1].min(), X[:, 1].max()
x1 = np.linspace(x1min - (x1max - x1min) * 0.2, x1max + (x1max - x1min) * 0.2, 100)
x2 = np.linspace(x2min - (x2max - x2min) * 0.2, x2max + (x2max - x2min) * 0.2, 100)
X1, X2 = np.meshgrid(x1, x2)

for t in [3, 5, 11, 15, 20]:
    plt.title('前%d个基学习器' % t)
    plt.xlabel('密度')
    plt.ylabel('含糖量')
    # plt.contourf(X1,X2,Ypre)
    # 画样本数据点
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='+', c='r', s=100, label='好瓜')
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='_', c='k', s=100, label='坏瓜')
    plt.legend()
    # 画基学习器划分边界
    for i in range(t):
        feature, point = H[i, :2]
        if feature == 0:
            plt.plot([point, point], [x2min, x2max], 'k', linewidth=1)
        else:
            plt.plot([x1min, x1max], [point, point], 'k', linewidth=1)
    # 画基集成效果的划分边界
    Ypre = predict(H[:t], X1, X2)
    plt.contour(X1, X2, Ypre, colors='r', linewidths=5, levels=[0])
    plt.show()
