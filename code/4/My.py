import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
import treePlottter

class Node(object):
    def __init__(self):
        self.feature_name = None  # 特征名称
        self.feature_index = None  # 索引
        self.subtree = {}  # 子树
        self.impurity = None  # gini or entropy
        self.is_continuous = False  # 是否连续
        self.split_value = None  # 分割值
        self.is_leaf = False  # 是否是叶子节点
        self.leaf_class = None  # 叶子节点的类别
        self.leaf_num = None  # 叶子节点的类别数量
        self.high = -1  # 该节点高度


class DecisionTree(object):
    def __init__(self, criterion='gini', ):
        self.root = Node()
        assert criterion in ['gini', 'infogain']
        self.criterion = criterion

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        self.columns = list(X_train.columns)
        self.tree_ = self.geerate_tree(X_train, y_train)
        return self

    def geerate_tree(self, X_train, y_train):
        my_tree = Node()#创建根节点
        my_tree.leaf_num = 0#初始化叶子节点数量
        if y_train.nunique() == 1:  # 如果只有一个类别，则为叶子节点
            my_tree.is_leaf = True#叶子节点
            my_tree.leaf_class = y_train.values[0]#叶子节点的类别
            my_tree.high = 0#叶子节点的高度
            my_tree.leaf_num += 1#叶子节点数量加1
            return my_tree#返回叶子节点
        if X_train.empty:  # 如果特征用完了，数据为空，则为叶子节点
            my_tree.is_leaf = True
            my_tree.leaf_class = y_train.value_counts().idxmax()  # 返回样本最多的类别
            my_tree.high = 0
            my_tree.leaf_num += 1
            return my_tree
        best_feature_name, best_impurity = self.choose_best_feature_to_split(X_train, y_train)  # 选择最佳特征
        my_tree.feature_name = best_feature_name#最佳特征名称
        my_tree.impurity = best_impurity[0]  # 最佳分割值的名称
        my_tree.feature_index = self.columns.index(best_feature_name)#最佳特征索引
        feature_values = X_train.loc[:, best_feature_name]#最佳特征值
        if len(best_impurity) == 1:  # 如果是离散值
            my_tree.is_continuous = False#离散值
            unique_values = feature_values.unique()#最佳特征值的唯一值
            sub_X_train = X_train.drop(best_feature_name, axis=1)  # 删除最佳特征
            max_high = -1#初始化最大高度
            for value in unique_values:  # 对每一个特征值
                my_tree.subtree[value] = self.geerate_tree(sub_X_train[feature_values == value],
                                                           y_train[feature_values == value])#递归生成子树
                if my_tree.subtree[value].high > max_high:#如果子树的高度大于最大高度
                    max_high = my_tree.subtree[value].high  # 取最大的高度为子树高度
                my_tree.leaf_num += my_tree.subtree[value].leaf_num  # 添加子树的叶子数量
            my_tree.high = max_high + 1
        else:  # 如果是连续值
            my_tree.is_continuous = True
            my_tree.split_value = best_impurity[1]  # 最佳分割点
            up_part = '>={:.4f}'.format(my_tree.split_value)#大于等于最佳分割点的特征值
            down_part = '<{:.4f}'.format(my_tree.split_value)#小于最佳分割点的特征值
            my_tree.subtree[up_part] = self.geerate_tree(X_train[feature_values >= my_tree.split_value],
                                                         y_train[feature_values >= my_tree.split_value])#递归生成大于等于最佳分割点的子树
            my_tree.subtree[down_part] = self.geerate_tree(X_train[feature_values < my_tree.split_value],
                                                           y_train[feature_values < my_tree.split_value])

            my_tree.leaf_num += my_tree.subtree[up_part].leaf_num + my_tree.subtree[down_part].leaf_num#添加子树的叶子数量
            my_tree.high = max(my_tree.subtree[up_part].high, my_tree.subtree[down_part].high) + 1#计算子树高度
        return my_tree

    def choose_best_feature_to_split_gini(self, X_train, y_train):
        pass

    def choose_best_feature_to_split_infogain(self, X_train, y_train):
        feature_names = X_train.columns#特征名称
        best_feature_name = None#最佳特征名称
        best_info_gain = [float('-inf')]#最佳信息增益
        entD = self.entropy(y_train)  # 计算数据集的熵
        for feature_name in feature_names:#对每一个特征
            # is_continuous = type_of_target(X_train[feature_name]) == 'continuous'
            is_continuous = True#设定为连续值
            info_gain = self.info_gain(X_train[feature_name], y_train, entD, is_continuous)  # 对每个特征计算信息增益
            if info_gain[0] > best_info_gain[0]:#如果信息增益大于最佳信息增益
                best_info_gain = info_gain#更新最佳信息增益
                best_feature_name = feature_name#更新最佳特征名称
        return best_feature_name, best_info_gain#返回最佳特征名称和最佳信息增益

    def choose_best_feature_to_split(self, X_train, y_train):
        assert self.criterion in ['gini', 'infogain']
        if self.criterion == 'gini':
            return self.choose_best_feature_to_split_gini(X_train, y_train)  # 返回最佳特征和最佳特征的信息增益
        else:
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
        if is_continuous:
            unique_values.sort()
            split_point_set = [(unique_values[i] + unique_values[i + 1]) / 2 for i in
                               range(len(unique_values) - 1)]  # 计算分割点
            min_ent = float('inf')
            min_ent_point = None
            for split_point in split_point_set:
                Dv1 = y_train[param <= split_point]
                Dv2 = y_train[param > split_point]
                feature_ent_ = Dv1.shape[0] / m * self.entropy(Dv1) + Dv2.shape[0] / m * self.entropy(Dv2)
                if feature_ent_ < min_ent:
                    min_ent = feature_ent_
                    min_ent_point = split_point
            gain = entD - min_ent
            return [gain, min_ent_point]
        else:
            feature_ent_ = []
            for value in unique_values:
                Dv = y_train[param == value]
                feature_ent_.append(Dv.shape[0] / m * self.entropy(Dv))
            gain = entD - np.sum(feature_ent_)  # 书中4.2
            return [gain]
        pass

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


if __name__ == '__main__':
    # diabetes = pd.read_csv('..\dataset\diabetes.csv', header=None)
    # diabetes.columns = ['pregnant', 'plasma_glucose', 'blood_pressure', 'triceps', 'serum_insulin', 'body_mass_index',
    #                     'diabetes_pedigree', 'age', 'class']
    # data = diabetes.iloc[:, :-1].values
    # target = diabetes.iloc[:, -1].values
    # feature_names = list(diabetes.columns[:-1])
    # target_names = ['1', '-1']
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    y = pd.Series(iris['target_names'][iris['target']])
    print(X.head())
    print(y.head())
    # 取30个样本为测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=15)

    # 剩下120个样本中，取30个作为剪枝时的验证集F
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=15)
    tree_no_pruning = DecisionTree('infogain')
    tree_no_pruning.fit(X_train, y_train, X_val, y_val)
    print('不剪枝：', np.mean(tree_no_pruning.predict(X_test) == y_test))
    treePlottter.create_plot(tree_no_pruning.tree_)
