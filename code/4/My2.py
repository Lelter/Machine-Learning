# 选择四个数据集，未剪枝算法，计算每个数据集的准确率，输出所有决策树图像
# 分别用基尼指数、信息增益

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import pandas as pd
import pydotplus  # for writing dot files


class decisionTree:
    def __init__(self, data, target, feature_names, target_names):  # 初始化
        self.data = data  # 训练数据
        self.target = target  # 训练数据标签
        self.feature_names = feature_names  # 特征名称
        self.target_names = target_names  # 目标名称
        # self.clf = self.predict(self)
        # self.print(self)

    def predict(self):
        x = self.data
        y = self.target
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
        clf = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=0)
        clf.fit(X_train, y_train)
        print("Accuracy:", clf.score(X_test, y_test))
        return clf

    def print(self, clf):
        dot_data = export_graphviz(clf, out_file=None,
                                   feature_names=self.feature_names,
                                   class_names=self.target_names,
                                   filled=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        return Image(graph.create_png())


iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
diabetes = pd.read_csv('..\dataset\diabetes.csv', header=None)
diabetes.columns = ['pregnant', 'plasma_glucose', 'blood_pressure', 'triceps', 'serum_insulin', 'body_mass_index',
                    'diabetes_pedigree', 'age', 'class']


# %%

def choose_data(choose_data):
    if (choose_data == 'iris'):
        data = iris.data
        target = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
    elif (choose_data == 'wine'):
        data = wine.data
        target = wine.target
        feature_names = wine.feature_names
        target_names = wine.target_names
    elif (choose_data == 'breast_cancer'):
        data = breast_cancer.data
        target = breast_cancer.target
        feature_names = breast_cancer.feature_names
        target_names = breast_cancer.target_names
    elif (choose_data == 'diabetes'):
        data = diabetes.iloc[:, :-1].values
        target = diabetes.iloc[:, -1].values
        feature_names = list(diabetes.columns[:-1])
        target_names = ['1', '-1']
    else:
        print('Please choose data')
        return
    # print(data, target, feature_names, target_names)
    DT = decisionTree(data, target, feature_names, target_names)
    clf = DT.predict()  # 训练数据
    return DT.print(clf)  # 输出决策树


image=choose_data('iris')
display(image)
# choose_data('iris')
# choose_data('wine')
# choose_data('diabetes')
