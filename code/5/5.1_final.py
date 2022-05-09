import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Data:
    def __init__(self, file_name, type, iteration):
        self.file_name = file_name
        dataset = pd.read_csv(file_name)
        attributes = {}
        attributes['浅白'] = 0
        attributes['青绿'] = 0.5
        attributes['乌黑'] = 1
        attributes['蜷缩'] = 0
        attributes['稍蜷'] = 0.5
        attributes['硬挺'] = 1
        attributes['沉闷'] = 0
        attributes['浊响'] = 0.5
        attributes['清脆'] = 1
        attributes['模糊'] = 0
        attributes['稍糊'] = 0.5
        attributes['清晰'] = 1
        attributes['凹陷'] = 0
        attributes['稍凹'] = 0.5
        attributes['平坦'] = 1
        attributes['硬滑'] = 0
        attributes['软粘'] = 1
        attributes['否'] = 0
        attributes['是'] = 1
        del dataset["编号"]
        dataset = np.array(dataset)
        m, n = np.shape(dataset)
        # print(dataset)
        # print(m, n)
        for i in range(m):
            for j in range(n):
                if dataset[i][j] in attributes:
                    dataset[i][j] = attributes[dataset[i][j]]
                dataset[i][j] = round(dataset[i][j], 3)  # 保留三位小数
        self.dataset = dataset
        self.iteration = iteration
        train = [1, 2, 3, 6, 7, 10, 14, 15, 16, 17]
        train = [i - 1 for i in train]
        test = [4, 5, 8, 9, 11, 12, 13]
        test = [i - 1 for i in test]
        self.train_X = dataset[train, :-1]
        self.train_Y = dataset[train, -1]
        self.test_X = dataset[test, :-1]
        self.test_Y = dataset[test, -1]
        self.trueY = dataset[:, -1]
        self.X = dataset[:, :-1]
        self.m, self.n = np.shape(self.train_X)
        self.theta = None
        self.gamma = None
        self.v = None
        self.w = None
        # print(self.X)
        # print(self.m, self.n)
        # print(self.trueY)
        if type == "BP_ST":
            self.BP_ST()
        if type == "BP_AST":
            self.BP_AST()

    def BP_ST(self):
        X = self.train_X
        trueY = self.train_Y

        d = self.n  # 输入层节点数
        l = 1  # 输出层节点数
        q = d + 1  # 隐藏层节点数
        theta = [random.random() for i in range(l)]  # 初始化输出层阈值
        gamma = [random.random() for i in range(q)]  # 初始化隐藏层阈值
        v = [[random.random() for i in range(q)] for j in range(d)]  # 初始化输入层到隐藏层的权重
        w = [[random.random() for i in range(l)] for j in range(q)]  # 初始化隐藏层到输出层的权重
        eta = 0.1  # 学习率
        max_iteration = self.iteration  # 迭代次数
        iteration = 0
        sumE_list = []
        while iteration < max_iteration:
            iteration += 1
            sumE = 0  # 每次迭代时的均方误差之和
            for i in range(self.m):
                alpha = np.dot(X[i], v)  # 隐藏层第i个节点的输入=输入层第i个节点的输入*输入层到隐藏层的权重
                b = self.sigmoid(alpha - gamma, 1)  # 隐藏层第i个节点的输出=sigmoid(输入层第i个节点的输入*输入层到隐藏层的权重-隐藏层阈值)
                beta = np.dot(b, w)  # 输出层第i个节点的输入=隐藏层第i个节点的输出*隐藏层到输出层的权重
                predictY = self.sigmoid(beta - theta, 1)  # 输出层第i个节点的输出=sigmoid(隐藏层第i个节点的输出*隐藏层到输出层的权重-输出层阈值)
                E = (sum(predictY - trueY[i]) ** 2) / 2  # 计算均方误差
                sumE += E  # 计算累计误差，目标是让累计误差最小
                g = predictY * (1 - predictY) * (trueY[i] - predictY)  # p103 5.10
                e = b * (1 - b) * (np.dot(w, g.T)).T  # p104 5.15
                w += eta * np.dot(b.reshape((q, 1)), g.reshape((1, l)))  # p103 5.11
                theta -= eta * g  # p103 5.12
                v += eta * np.dot(X[i].reshape((d, 1)), e.reshape((1, q)))  # p103 5.13
                gamma -= eta * e  # p103 5.14
            print(sumE)
            sumE_list.append(sumE)
        self.theta = theta
        self.gamma = gamma
        self.v = v
        self.w = w
        sumE_list = np.array(sumE_list)
        x_plot = np.arange(0, max_iteration)
        sns.set()
        sns.scatterplot(x=x_plot, y=sumE_list, s=3)
        plt.show()

    def BP_AST(self):
        X = self.train_X
        trueY = self.train_Y

        d = self.n  # 输入层节点数
        l = 1  # 输出层节点数
        q = d + 1  # 隐藏层节点数
        theta = [random.random() for i in range(l)]  # 初始化输出层阈值
        gamma = [random.random() for i in range(q)]  # 初始化隐藏层阈值
        v = [[random.random() for i in range(q)] for j in range(d)]  # 初始化输入层到隐藏层的权重
        w = [[random.random() for i in range(l)] for j in range(q)]  # 初始化隐藏层到输出层的权重
        eta = 0.1  # 学习率
        max_iteration = self.iteration  # 迭代次数
        iteration = 0
        sumE_list = []
        trueY = trueY.reshape((self.m, l))
        while iteration < max_iteration:
            iteration += 1
            alpha = np.dot(X, v)  # 隐藏层第i个节点的输入=输入层第i个节点的输入*输入层到隐藏层的权重
            b = self.sigmoid(alpha - gamma, 2)  # 隐藏层第i个节点的输出=sigmoid(输入层第i个节点的输入*输入层到隐藏层的权重-隐藏层阈值)
            beta = np.dot(b, w)  # 输出层第i个节点的输入=隐藏层第i个节点的输出*隐藏层到输出层的权重
            predictY = self.sigmoid(beta - theta, 2)  # 输出层第i个节点的输出=sigmoid(隐藏层第i个节点的输出*隐藏层到输出层的权重-输出层阈值)
            E = sum(sum(predictY - trueY) ** 2) / 2
            g = predictY * (1 - predictY) * (trueY - predictY)  # p103 5.10
            e = b * (1 - b) * (np.dot(w, g.T)).T  # p104 5.15
            w += eta * np.dot(b.T, g)  # p103 5.11
            theta -= eta * g  # p103 5.12
            v += eta * np.dot(X.T, e)  # p103 5.13
            gamma -= eta * e  # p103 5.14
            print(E)
            sumE_list.append(E)
        self.theta = np.mean(theta, axis=0)
        self.gamma = np.mean(gamma, axis=0)
        self.v = v
        self.w = w
        sumE_list = np.array(sumE_list)
        x_plot = np.arange(0, max_iteration)
        sns.set()
        sns.scatterplot(x=x_plot, y=sumE_list, s=3)
        plt.show()

    def Predict(self):
        alpha = np.dot(self.test_X, self.v)  # ast和st不一样
        b = self.sigmoid(alpha - self.gamma, 2)
        beta = np.dot(b, self.w)
        predictY = self.sigmoid(beta - self.theta, 2)
        result = []
        for i in predictY.flat:
            if i >= 0.5:
                result.append(1)
            else:
                result.append(0)
        predict = 0
        count = 0
        for i in self.test_Y.flat:
            if result[count] == i:
                predict += 1
            count += 1
        predict = predict / len(result)
        return predict

    def sigmoid(self, ix, dimension):
        if dimension == 1:
            for i in range(len(ix)):
                ix[i] = 1 / (1 + np.exp(-ix[i]))
        else:
            for i in range(len(ix)):
                ix[i] = self.sigmoid(ix[i], dimension - 1)
        return ix
        pass


if __name__ == '__main__':
    '''
    编程实现标准BP算法，在西瓜数据集3.0上训练一个单隐层网络，并做数据分析和结果评价。
    '''
    D = Data(r"E:\Machine-Learning\code\dataset\3.0.csv", "BP_ST", 10000)
    print("准确率为:{}%".format(round(D.Predict() * 100, 3)))
