import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import treeCreater
import treePlottter
pd.set_option('display.max_columns', None)

iris = datasets.load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.Series(iris['target_names'][iris['target']])
print(X.head())
print(y.head())
# 取三个样本为测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15)

# 剩下120个样本中，取30个作为剪枝时的验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=15)


# 不剪枝
tree_no_pruning = treeCreater.DecisionTree('gini')
tree_no_pruning.fit(X_train, y_train, X_val, y_val)
# print('不剪枝：', np.mean(tree_no_pruning.predict(X_test) == y_test))
treePlottter.create_plot(tree_no_pruning.tree_)

# # 预剪枝
# tree_pre_pruning = treeCreater.DecisionTree('gini', 'pre_pruning')
# tree_pre_pruning.fit(X_train, y_train, X_val, y_val)
# # print('预剪枝：', np.mean(tree_pre_pruning.predict(X_test) == y_test))
# treePlottter.create_plot(tree_pre_pruning.tree_)
#
# # 后剪枝
# tree_post_pruning = treeCreater.DecisionTree('gini', 'post_pruning')
# tree_post_pruning.fit(X_train, y_train, X_val, y_val)
# # print('后剪枝：', np.mean(tree_post_pruning.predict(X_test) == y_test))
# # treePlottter.create_plot(tree_post_pruning.tree_)
