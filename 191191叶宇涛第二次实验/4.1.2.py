import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

import treePlottter


class Node(object):
    def __init__(self):
        self.feature_name = None  # 特征名称
        self.feature_index = None  # 索引
        self.feature_value = None  # 最佳属性对应的特征值
        self.subtree = {}  # 子树
        self.impurity = None  # gini or entropy
        self.is_continuous = False  # 是否连续
        self.split_value = None  # 分割值
        self.is_leaf = False  # 是否是叶子节点
        self.leaf_class = None  # 叶子节点的类别
        self.leaf_num = None  # 叶子节点的类别数量
        self.high = -1  # 该节点高度
        self.v = ""
        self.train_data = None  # 该节点的训练集
        self.train_label = None  # 该节点的训练集标签


class DecisionTree(object):
    def __init__(self, criterion='infogain', ):
        self.root = Node()
        assert criterion in ['gini', 'infogain']
        self.criterion = criterion

    def fit(self, X_train, y_train, max_height=3):
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        self.columns = list(X_train.columns)
        self.tree_ = self.geerate_tree(X_train, y_train, max_height)
        return self

    def geerate_tree(self, X_train, y_train, max_height):
        my_tree = Node()  # 创建节点
        my_tree.leaf_num = 0  # 初始化叶子节点数量
        my_tree.train_data = X_train  # 初始化训练集
        my_tree.train_label = y_train  # 初始化训练集标签
        if y_train.nunique() == 1:  # 如果只有一个类别，则为叶子节点
            my_tree.is_leaf = True  # 叶子节点
            my_tree.leaf_class = y_train.values[0]  # 叶子节点的类别
            my_tree.high = 0  # 叶子节点的高度
            my_tree.leaf_num += 1  # 叶子节点数量加1
            return my_tree  # 返回叶子节点
        if X_train.empty:  # 如果特征用完了，数据为空，则为叶子节点
            my_tree.is_leaf = True
            my_tree.leaf_class = y_train.value_counts().idxmax()  # 返回样本最多的类别
            my_tree.high = 0
            my_tree.leaf_num += 1
            return my_tree
        best_feature_name, best_impurity = self.choose_best_feature_to_split(X_train, y_train)  # 选择最佳特征
        my_tree.feature_name = best_feature_name  # 最佳特征名称
        my_tree.impurity = best_impurity[0]  # 最佳分割值的名称
        my_tree.feature_index = self.columns.index(best_feature_name)  # 最佳特征索引
        my_tree.feature_value = X_train.loc[:, best_feature_name]
        my_tree.is_continuous = False  # 离散值
        node_queue = [my_tree]  # 栈数据结构
        my_tree.v = my_tree.feature_name + "=?"
        print("加入", my_tree.v)
        while node_queue:  # 栈不为空
            curnode = node_queue.pop()  # 出栈，取当前节点为栈顶元素
            print("取出", curnode.v)
            if curnode.high >= max_height:  # 如果高度大于最大高度
                curnode.is_leaf = True  # 设置为叶子节点
                curnode.leaf_class = curnode.train_label.value_counts().idxmax()  # 类别设置为样本最多的标签
                continue
            unique_values = curnode.feature_value.unique()  # 最佳特征的唯一值
            sub_X_train = curnode.train_data.drop(best_feature_name, axis=1)  # 删除最佳特征
            # max_high = -1  # 初始化最大高度
            for value in unique_values:  # 对每一个特征值
                nextnode = Node()  # 创建节点
                nextnode.high = curnode.high + 1  # 高度为现有高度+1
                nextnode.is_continuous = False

                D_X = sub_X_train.loc[curnode.feature_value == value]  # 最佳特征X
                D_y = curnode.train_label.loc[curnode.feature_value == value]  # 最佳特征Y
                nextnode.train_data = D_X  # 赋值给nextnode.train_data，方便后续使用
                nextnode.train_label = D_y
                if D_X.empty:  # 如果数据集为空
                    nextnode.is_leaf = True  # 设置为叶子节点
                    nextnode.leaf_class = D_y.value_counts().idxmax()  # 返回样本最多的类别
                    nextnode.high = 0
                    nextnode.leaf_num = 1
                    curnode.subtree[value] = nextnode  # 创建子树
                    print(nextnode.leaf_class)
                    continue
                elif D_y.nunique() == 1:  # 如果标签集为空
                    nextnode.is_leaf = True
                    nextnode.leaf_class = D_y.values[0]
                    nextnode.high = 0
                    nextnode.leaf_num = 1
                    curnode.subtree[value] = nextnode  # 创建子树
                    print(nextnode.leaf_class)
                    continue
                else:
                    best_feature_name, best_impurity = self.choose_best_feature_to_split(D_X, D_y)  # 选择最佳特征
                    nextnode.feature_value = D_X.loc[:, best_feature_name]  # 最佳特征对应的特征值
                    nextnode.feature_name = best_feature_name  # 最佳特征名称
                    nextnode.impurity = best_impurity[0]  # 最佳分割值的名称
                    nextnode.feature_index = self.columns.index(best_feature_name)  # 最佳特征索引
                    nextnode.v = nextnode.feature_name + "=?"
                    curnode.subtree[value] = nextnode  # 创建子树
                    node_queue.append(nextnode)  # 加入nextnode
                    print("加入", nextnode.v)
        self.calculate_leafnum(my_tree)  # 计算叶子数量
        self.calculate_high(my_tree)  # 计算高度
        return my_tree

    def choose_best_feature_to_split_infogain(self, X_train, y_train):
        feature_names = X_train.columns  # 特征名称
        best_feature_name = None  # 最佳特征名称
        best_info_gain = [float('-inf')]  # 最佳信息增益
        entD = self.entropy(y_train)  # 计算数据集的熵
        info_gain_list = []
        for feature_name in feature_names:  # 对每一个特征
            is_continuous = type_of_target(X_train[feature_name]) == 'continuous'
            info_gain = self.info_gain(X_train[feature_name], y_train, entD, is_continuous)  # 对每个特征计算信息增益
            info_gain_list.append([feature_name, info_gain])
            if info_gain[0] >= best_info_gain[0]:  # 如果信息增益大于最佳信息增益
                best_info_gain = info_gain  # 更新最佳信息增益
                best_feature_name = feature_name  # 更新最佳特征名称
        return best_feature_name, best_info_gain  # 返回最佳特征名称和最佳信息增益

    def choose_best_feature_to_split(self, X_train, y_train):
        return self.choose_best_feature_to_split_infogain(X_train, y_train)  # 返回最佳特征和最佳特征的信息增益

    pass

    def entropy(self, y_train):
        p = pd.value_counts(y_train) / y_train.shape[0]
        ent = np.sum(-p * np.log2(p))
        return ent
        pass

    def info_gain(self, param, y_train, entD, is_continuous):
        '''
        :param param: 特征值
        :param y_train: y值
        :param entD: 总体的熵
        :param is_continuous:是否是连续值
        :return:
        '''
        m = y_train.shape[0]
        unique_values = param.unique()
        feature_ent_ = []
        for value in unique_values:
            Dv = y_train[param == value]
            feature_ent_.append(Dv.shape[0] / m * self.entropy(Dv))
        gain = entD - np.sum(feature_ent_)  # 书中4.2
        return [gain]

    def predict(self, X_test):
        '''
        :param X_test: 测试集X
        :return:
        '''
        if X_test.ndim == 1:
            return self.predict_single(X_test)
        else:
            return X_test.apply(self.predict_single, axis=1)

    def predict_single(self, X_test, subtree=None):
        '''
        :param X_test: 测试集X
        :return:
        '''
        if subtree is None:
            subtree = self.tree_
        if subtree.is_leaf:
            return subtree.leaf_class
        if subtree.is_continuous:
            if X_test[subtree.feature_name] >= subtree.split_value:
                return self.predict_single(X_test, subtree.subtree['>={:.4f}'.format(subtree.split_value)])
            else:
                return self.predict_single(X_test, subtree.subtree['<{:.4f}'.format(subtree.split_value)])
        else:
            return self.predict_single(X_test, subtree.subtree[X_test[subtree.feature_index]])
        pass

    def calculate_leafnum(self, my_tree):
        node = my_tree
        if node.is_leaf:
            node.leaf_num = 1
            return
        else:
            for value in node.subtree.values():
                self.calculate_leafnum(value)
            node.leaf_num = sum([value.leaf_num for value in node.subtree.values()])
        pass

        pass

    def calculate_high(self, my_tree):
        node = my_tree
        if node.is_leaf:
            node.high = 0
            return
        else:
            for value in node.subtree.values():
                self.calculate_high(value)
            node.high = max([value.high for value in node.subtree.values()]) + 1

        pass


if __name__ == '__main__':
    data_path2 = r'..\dataset\table_4.2.csv'
    data = pd.read_csv(data_path2, encoding='utf8', index_col=0)
    train = [1, 2, 3, 6, 7, 10, 14, 15, 16, 17]
    train = [i - 1 for i in train]
    X = data.iloc[train, :6]
    y = data.iloc[train, 6]
    test = [4, 5, 8, 9, 11, 12, 13]
    test = [i - 1 for i in test]

    X_val = data.iloc[test, :6]
    y_val = data.iloc[test, 6]
    tree = DecisionTree('infogain')
    tree.fit(X, y, 2)

    print("精度为：", np.mean(tree.predict(X_val) == y_val))
    treePlottter.create_plot(tree.tree_)
