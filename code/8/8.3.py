# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
#
#
# class Node(object):
#     def __init__(self):
#         self.feature_index = None
#         self.split_point = None
#         self.deep = None
#         self.left_tree = None
#         self.right_tree = None
#         self.leaf_class = None
#
#
# def gini(y, D):
#     '''
#     计算样本集y下的加权基尼指数
#     :param y: 数据样本标签
#     :param D: 样本权重
#     :return:  加权后的基尼指数
#     '''
#     unique_class = np.unique(y)
#     total_weight = np.sum(D)
#
#     gini = 1
#     for c in unique_class:
#         gini -= (np.sum(D[y == c]) / total_weight) ** 2
#
#     return gini
#
#
# def calcMinGiniIndex(a, y, D):
#     '''
#     计算特征a下样本集y的的基尼指数
#     :param a: 单一特征值
#     :param y: 数据样本标签
#     :param D: 样本权重
#     :return:
#     '''
#
#     feature = np.sort(a)
#     total_weight = np.sum(D)
#
#     split_points = [(feature[i] + feature[i + 1]) /
#                     2 for i in range(feature.shape[0] - 1)]
#
#     min_gini = float('inf')
#     min_gini_point = None
#
#     for i in split_points:
#         yv1 = y[a <= i]
#         yv2 = y[a > i]
#
#         Dv1 = D[a <= i]
#         Dv2 = D[a > i]
#         gini_tmp = (np.sum(Dv1) * gini(yv1, Dv1) + np.sum(Dv2)
#                     * gini(yv2, Dv2)) / total_weight
#
#         if gini_tmp < min_gini:
#             min_gini = gini_tmp
#             min_gini_point = i
#
#     return min_gini, min_gini_point
#
#
# def chooseFeatureToSplit(X, y, D):
#     '''
#     :param X:
#     :param y:
#     :param D:
#     :return: 特征索引, 分割点
#     '''
#     gini0, split_point0 = calcMinGiniIndex(X[:, 0], y, D)
#     gini1, split_point1 = calcMinGiniIndex(X[:, 1], y, D)
#
#     if gini0 > gini1:
#         return 1, split_point1
#     else:
#         return 0, split_point0
#
#
# def createSingleTree(X, y, D, deep=0):
#     '''
#     这里以C4.5 作为基学习器，限定深度为2，使用基尼指数作为划分点，基尼指数的计算会基于样本权重，
#     不确定这样的做法是否正确，但在西瓜书p87, 4.4节中, 处理缺失值时, 其计算信息增益的方式是将样本权重考虑在内的，
#     这里就参考处理缺失值时的方法。
#     :param X: 训练集特征
#     :param y: 训练集标签
#     :param D: 训练样本权重
#     :param deep: 树的深度
#     :return:
#     '''
#
#     node = Node()
#     node.deep = deep
#
#     if (deep == 2) | (X.shape[0] <= 2):  # 当前分支下，样本数量小于等于2 或者 深度达到2时，直接设置为也节点
#         pos_weight = np.sum(D[y == 1])
#         neg_weight = np.sum(D[y == -1])
#         if pos_weight > neg_weight:
#             node.leaf_class = 1
#         else:
#             node.leaf_class = -1
#
#         return node
#
#     feature_index, split_point = chooseFeatureToSplit(X, y, D)
#
#     node.feature_index = feature_index
#     node.split_point = split_point
#
#     left = X[:, feature_index] <= split_point
#     right = X[:, feature_index] > split_point
#
#     node.left_tree = createSingleTree(X[left, :], y[left], D[left], deep + 1)
#     node.right_tree = createSingleTree(
#         X[right, :], y[right], D[right], deep + 1)
#
#     return node
#
#
# def predictSingle(tree, x):
#     '''
#     基于基学习器，预测单个样本
#     :param tree:
#     :param x:
#     :return:
#     '''
#     if tree.leaf_class is not None:
#         return tree.leaf_class
#
#     if x[tree.feature_index] > tree.split_point:
#         return predictSingle(tree.right_tree, x)
#     else:
#         return predictSingle(tree.left_tree, x)
#
#
# def predictBase(tree, X):
#     '''
#     基于基学习器预测所有样本
#     :param tree:
#     :param X:
#     :return:
#     '''
#     result = []
#
#     for i in range(X.shape[0]):
#         result.append(predictSingle(tree, X[i, :]))
#
#     return np.array(result)
#
#
# def adaBoostTrain(X, y, tree_num=20):
#     '''
#     以深度为2的决策树作为基学习器，训练adaBoost
#     :param X:
#     :param y:
#     :param tree_num:
#     :return:
#     '''
#     D = np.ones(y.shape) / y.shape  # 初始化权重
#
#     trees = []  # 所有基学习器
#     a = []  # 基学习器对应权重
#
#     agg_est = np.zeros(y.shape)
#
#     for _ in range(tree_num):
#         tree = createSingleTree(X, y, D)
#
#         hx = predictBase(tree, X)
#         err_rate = np.sum(D[hx != y])
#
#         at = np.log((1 - err_rate) / max(err_rate, 1e-16)) / 2
#
#         agg_est += at * hx
#         trees.append(tree)
#         a.append(at)
#
#         if (err_rate > 0.5) | (err_rate == 0):  # 错误率大于0.5 或者 错误率为0时，则直接停止
#             break
#
#         # 更新每个样本权重
#         err_index = np.ones(y.shape)
#         err_index[hx == y] = -1
#
#         D = D * np.exp(err_index * at)
#         D = D / np.sum(D)
#
#     return trees, a, agg_est
#
#
# def adaBoostPredict(X, trees, a):
#     agg_est = np.zeros((X.shape[0],))
#
#     for tree, am in zip(trees, a):
#         agg_est += am * predictBase(tree, X)
#
#     result = np.ones((X.shape[0],))
#
#     result[agg_est < 0] = -1
#
#     return result.astype(int)
#
#
# def pltAdaBoostDecisionBound(X_, y_, trees, a):
#     pos = y_ == 1
#     neg = y_ == -1
#     x_tmp = np.linspace(0, 1, 600)
#     y_tmp = np.linspace(-0.2, 0.7, 600)
#
#     X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
#
#     Z_ = adaBoostPredict(
#         np.c_[X_tmp.ravel(), Y_tmp.ravel()], trees, a).reshape(X_tmp.shape)
#     plt.contour(X_tmp, Y_tmp, Z_, [0], colors='orange', linewidths=1)
#
#     plt.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
#     plt.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')
#     plt.legend()
#     plt.show()
#
#
# if __name__ == "__main__":
#     data_path = r'E:\Machine-Learning\code\dataset\3.0a.csv'
#
#     data = pd.read_csv(data_path)
#
#     X = data.iloc[:, 1:3].values
#     y = data.iloc[:, 3].values
#
#     y[y == 0] = -1
#
#     trees, a, agg_est = adaBoostTrain(X, y)
#
#     pltAdaBoostDecisionBound(X, y, trees, a)
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


def Adaboost(X, Y, T, rule='MaxInfoGain', show=False):
    # 以决策树桩为基学习器的Adaboost算法
    # 输入:
    #     X:特征，m×n维向量
    #     Y：标记,m×1维向量
    #     T：训练次数
    #     rule:决策树桩属性划分规则，可以是：
    #          'MaxInfoGain'，'MinError','Random'
    #     show:是否计算并显示迭代过程中的损失函数和错误率
    # 输出:
    #     H:学习结果，T×3维向量，每行对应一个基学习器，
    #       第一列表示αt,第二列表示决策树桩的分类特征，第三列表示划分点

    m, n = X.shape  # 样本数和特征数
    D = np.ones(m) / m  # 初始样本分布
    H = np.zeros([T, 3])  # 初始化学习结果为全零矩阵
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 这部分用于计算迭代过程中的损失函数和错误率的变化
    # 可有可无
    H_pre = np.zeros(m)  # H对各个样本的预测结果
    L = []  # 存储每次迭代后的损失函数
    erro = []  # 存储每次迭代后的错误率
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for t in range(T):
        if rule == 'MaxInfoGain':
            ht = decision_sdumps_MaxInfoGain(X, Y, D)
        elif rule == 'MinError':
            ht = decision_sdumps_MinError(X, Y, D)
        else:  # rule=='Random'或者其他未知取值时，随机产生
            ht = decision_sdumps_Random(X, Y, D)
        ht_pre = (X[:, ht[0]] <= ht[1]) * 2 - 1  # 左分支为1，右分支为-1
        et = sum((ht_pre != Y) * D)
        while abs(et - 0.5) < 1E-3:
            # 若et=1/2，重新随机生成
            ht = decision_sdumps_Random(X, Y, D)
            ht_pre = (X[:, ht[0]] <= ht[1]) * 2 - 1
            et = sum((ht_pre != Y) * D)
        alphat = 0.5 * np.log((1 - et) / et)
        H[t, :] = [alphat] + ht
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 这部分用于计算迭代过程中的损失函数和错误率的变化
        # 可有可无
        if show:
            H_pre += alphat * ht_pre
            L.append(np.mean(np.exp(-Y * H_pre)))
            erro.append(np.mean(np.sign(H_pre) != Y))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        D *= np.exp(-alphat * Y * ht_pre)
        D = D / D.sum()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 这部分用于显示迭代过程中的损失函数和错误率的变化
    # 可有可无
    if show:
        try:
            plt.title('t=%d时错误率归0' % (np.where(np.array(erro) == 0)[0][0] + 1))
        except:
            plt.title('错误率尚未达到0')
        plt.plot(range(1, len(L) + 1), L, 'o-', markersize=2, label='损失函数的变化')
        plt.plot(range(1, len(L) + 1), erro, 'o-', markersize=2, label='错误率的变化')
        plt.plot([1, len(L) + 1], [1 / m, 1 / m], 'k', linewidth=1, label='1/m 线')
        plt.xlabel('基学习器个数')
        plt.ylabel('指数损失函数/错误率')
        plt.legend()
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return H


def decision_sdumps_MinError(X, Y, D):
    # 基学习器---决策树桩
    # 以最小化错误率来选择划分属性和划分点
    m, n = X.shape  # 样本数和特征数
    results = []  # 存储各个特征下的最佳划分点和错误率
    for i in range(n):  # 遍历各个候选特征
        x = X[:, i]  # 样本在该特征下的取值
        x_sorted = np.unique(x)  # 该特征下的可能取值并排序
        ts = (x_sorted[1:] + x_sorted[:-1]) / 2  # 候选划分点
        Errors = []  # 存储各个划分点下的|错误率-0.5|的值
        for t in ts:
            Ypre = (x <= t) * 2 - 1
            Errors.append(abs(sum(D[Ypre != Y]) - 0.5))
        Bestindex = np.argmax(Errors)  # 距离0.5最远的错误率的索引号
        results.append([ts[Bestindex], Errors[Bestindex]])

    results = np.array(results)
    divide_feature = np.argmax(results[:, 1])  # 划分特征
    h = [divide_feature, results[divide_feature, 0]]  # 划分特征和划分点
    return h


def decision_sdumps_MaxInfoGain(X, Y, D):
    # 基学习器---决策树桩
    # 以信息增益最大来选择划分属性和划分点
    m, n = X.shape  # 样本数和特征数
    results = []  # 存储各个特征下的最佳划分点和信息增益
    for i in range(n):  # 遍历各个候选特征
        x = X[:, i]  # 样本在该特征下的取值
        x_sorted = np.unique(x)  # 该特征下的可能取值并排序
        ts = (x_sorted[1:] + x_sorted[:-1]) / 2  # 候选划分点
        Gains = []  # 存储各个划分点下的信息增益
        for t in ts:
            Gain = 0
            Y_left, D_left = Y[x <= t], D[x <= t]  # 左分支样本的标记和分布
            Dl = sum(D_left)  # 左分支总分布数
            p1 = sum(D_left[Y_left == 1]) / Dl  # 左分支正样本分布比例
            p0 = sum(D_left[Y_left == -1]) / Dl  # 左分支负样本分布比例
            Gain += Dl * (np.log2(p1 ** p1) + np.log2(p0 ** p0))

            Y_right, D_right = Y[x > t], D[x > t]  # 右分支样本的标记和分布
            Dr = sum(D_right)  # 右分支总分布数
            p1 = sum(D_right[Y_right == 1]) / Dr  # 右分支正样本分布比例
            p0 = sum(D_right[Y_right == -1]) / Dr  # 右分支负样本分布比例
            Gain += Dr * (np.log2(p1 ** p1) + np.log2(p0 ** p0))

            Gains.append(Gain)

        results.append([ts[np.argmax(Gains)], max(Gains)])

    results = np.array(results)
    divide_feature = np.argmax(results[:, 1])  # 划分特征
    h = [divide_feature, results[divide_feature, 0]]  # 划分特征和划分点
    return h


def decision_sdumps_Random(X, Y, D):
    # 基学习器---决策树桩
    # 随机选择划分属性和划分点
    m, n = X.shape  # 样本数和特征数
    bestfeature = np.random.randint(2)
    x = X[:, bestfeature]  # 样本在该特征下的取值
    x_sorted = np.sort(x)  # 特征取值排序
    ts = (x_sorted[1:] + x_sorted[:-1]) / 2  # 候选划分点
    bestt = ts[np.random.randint(len(ts))]
    h = [bestfeature, bestt]
    return h


def predict(H, X1, X2):
    # 预测结果
    # 仅X1和X2两个特征,X1和X2同维度
    pre = np.zeros(X1.shape)
    for h in H:
        alpha, feature, point = h
        pre += alpha * (((X1 * (feature == 0) + X2 * (feature == 1)) <= point) * 2 - 1)
    return np.sign(pre)


##############################
#      主程序
##############################
# >>>>>西瓜数据集3.0α
X = np.array([[0.697, 0.46], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
              [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
              [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.36, 0.37],
              [0.593, 0.042], [0.719, 0.103]])
Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

# >>>>>运行Adaboost
T = 50
H = Adaboost(X, Y, T, rule='MaxInfoGain', show=True)
# rule:决策树桩属性划分规则，可以取值：'MaxInfoGain'，'MinError','Random'

# >>>>>观察结果
x1min, x1max = X[:, 0].min(), X[:, 0].max()
x2min, x2max = X[:, 1].min(), X[:, 1].max()
x1 = np.linspace(x1min - (x1max - x1min) * 0.2, x1max + (x1max - x1min) * 0.2, 100)
x2 = np.linspace(x2min - (x2max - x2min) * 0.2, x2max + (x2max - x2min) * 0.2, 100)
X1, X2 = np.meshgrid(x1, x2)

for t in [3, 5, 11, 30, 40, 50]:
    plt.title('前%d个基学习器' % t)
    plt.xlabel('密度')
    plt.ylabel('含糖量')
    # 画样本数据点
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='+', c='r', s=100, label='好瓜')
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='_', c='k', s=100, label='坏瓜')
    plt.legend()
    # 画基学习器划分边界
    for i in range(t):
        feature, point = H[i, 1:]
        if feature == 0:
            plt.plot([point, point], [x2min, x2max], 'k', linewidth=1)
        else:
            plt.plot([x1min, x1max], [point, point], 'k', linewidth=1)
    # 画集成学习器划分边界
    Ypre = predict(H[:t], X1, X2)
    plt.contour(X1, X2, Ypre, colors='r', linewidths=5, levels=[0])
    plt.show()
