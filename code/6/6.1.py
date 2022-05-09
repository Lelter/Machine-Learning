from sklearn import svm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")#设置背景颜色
    ax.patch.set_alpha(0.1)#设置透明度
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')#设置隐藏坐标轴
    ax.spines['bottom'].set_color('none')#设置隐藏坐标轴
    ax.spines['left'].set_color('none')#设置隐藏坐标轴
    ax.grid(axis='y', linestyle='-.')#设置网格线


def plt_support_(clf, X_, y_, kernel, c):
    pos = y_ == 1#获取正例
    neg = y_ == -1#获取负例
    ax = plt.subplot()#创建子图

    x_tmp = np.linspace(0, 1, 600)#设置x轴范围
    y_tmp = np.linspace(0, 0.8, 600)#设置y轴范围

    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)#设置网格点

    Z_rbf = clf.predict(#计算预测值
        np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)

    cs = ax.contour(X_tmp, Y_tmp, Z_rbf, [0], colors='orange', linewidths=1)#绘制超平面
    ax.clabel(cs, fmt={cs.levels[0]: 'decision boundary'})#设置超平面标签

    set_ax_gray(ax)#设置子图属性

    ax.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')#绘制正例
    ax.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')#绘制负例

    ax.scatter(X_[clf.support_, 0], X_[clf.support_, 1], marker='o',c='w', edgecolors='g', s=150,
               label='support_vectors',alpha=0.5)#绘制支持向量

    ax.legend()#绘制图例
    ax.set_title('{} kernel, C={}'.format(kernel, c))#设置标题
    plt.show()


path = r'E:\Machine-Learning\code\dataset\3.0a.csv'
data = pd.read_csv(path)

X = data.iloc[:, [1, 2]].values
y = data.iloc[:, 3].values
print(X)
print(y)
y[y == 0] = -1

C = 1000

clf_rbf = svm.SVC(C=C)#创建SVC分类器
clf_rbf.fit(X, y.astype(int))#训练模型
print('高斯核：')#输出模型信息
print('预测值：', clf_rbf.predict(X))#输出预测值
print('真实值：', y.astype(int))#输出真实值
print('支持向量：', clf_rbf.support_)#输出支持向量

print('-' * 40)#输出分隔符
clf_linear = svm.SVC(C=C, kernel='linear')#创建SVC分类器
clf_linear.fit(X, y.astype(int))#训练模型
print('线性核：')#输出模型信息
print('预测值：', clf_linear.predict(X))#输出预测值
print('真实值：', y.astype(int))#输出真实值
print('支持向量：', clf_linear.support_)#输出支持向量

plt_support_(clf_rbf, X, y, 'rbf', C)

plt_support_(clf_linear, X, y, 'linear', C)
