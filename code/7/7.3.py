import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from collections import namedtuple


def values_counts(x):
    unique_value = x.unique()  # 取值情况
    x_value_count = pd.value_counts(x)  # 计算每个取值的数量，并且加1，即拉普拉斯平滑
    x_value_count = x_value_count[unique_value].fillna(0) + 1  # 拉普拉斯平滑
    return x_value_count


def train_nb(X, y):
    m, n = X.shape
    p1 = (len(y[y == '是']) + 1) / (m + 2)  # 拉普拉斯平滑

    p1_list = []  # 用于保存正例下各属性的条件概率
    p0_list = []

    X1 = X[y == '是']
    X0 = X[y == '否']

    m1, _ = X1.shape
    m0, _ = X0.shape

    for i in range(n):
        xi = X.iloc[:, i]
        p_xi = namedtuple(
            X.columns[i], ['is_continuous', 'conditional_pro'])  # 用于储存每个变量的情况

        is_continuous = type_of_target(xi) == 'continuous'
        xi1 = X1.iloc[:, i]  # 正例下的每个特征列表
        xi0 = X0.iloc[:, i]  # 负例下的每个特征列表
        if is_continuous:  # 连续值时，conditional_pro 储存的就是 [mean, var] 即均值和方差
            xi1_mean = np.mean(xi1)  # 求均值
            xi1_var = np.var(xi1)  # 求方差
            xi0_mean = np.mean(xi0)
            xi0_var = np.var(xi0)

            p1_list.append(p_xi(is_continuous, [xi1_mean, xi1_var]))  # 训练集中正例的每个属性的条件概率,
            p0_list.append(p_xi(is_continuous, [xi0_mean, xi0_var]))  # 训练集中负例的每个属性的条件概率
        else:  # 离散值时直接计算各类别的条件概率
            unique_value = xi.unique()  # 取值情况
            nvalue = len(unique_value)  # 取值个数

            xi1_value_count = values_counts(xi1)  # 计算正样本中，该属性每个取值的数量，并且加1，即拉普拉斯平滑
            xi0_value_count = values_counts(xi0)  # 计算负样本中，该属性每个取值的数量，并且加1，即拉普拉斯平滑

            p1_list.append(p_xi(is_continuous, np.log(  # 计算正样本中，该属性每个取值的条件概率
                xi1_value_count / (m1 + nvalue))))
            p0_list.append(p_xi(is_continuous, np.log(  # 计算负样本中，该属性每个取值的条件概率
                xi0_value_count / (m0 + nvalue))))

    return p1, p1_list, p0_list


def predict_nb(x, p1, p1_list, p0_list):
    n = len(x)  # 特征个数

    x_p1 = np.log(p1)
    x_p0 = np.log(1 - p1)
    for i in range(n):
        p1_xi = p1_list[i]
        p0_xi = p0_list[i]

        if p1_xi.is_continuous:
            mean1, var1 = p1_xi.conditional_pro
            mean0, var0 = p0_xi.conditional_pro
            x_p1 += np.log(1 / (np.sqrt(2 * np.pi) * var1) *
                           np.exp(- (x[i] - mean1) ** 2 / (2 * var1 ** 2)))
            x_p0 += np.log(1 / (np.sqrt(2 * np.pi) * var0) *
                           np.exp(- (x[i] - mean0) ** 2 / (2 * var0 ** 2)))
        else:
            x_p1 += p1_xi.conditional_pro[x[i]]
            x_p0 += p0_xi.conditional_pro[x[i]]

    if x_p1 > x_p0:
        return '是'
    else:
        return '否'


if __name__ == '__main__':
    data_path = r'E:\Machine-Learning\code\dataset\3.0.csv'
    data = pd.read_csv(data_path)
    del data['编号']
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    p1, p1_list, p0_list = train_nb(X, y)

    x_test = X.iloc[0, :]  # 书中测1 其实就是第一个数据

    print(predict_nb(x_test, p1, p1_list, p0_list))
