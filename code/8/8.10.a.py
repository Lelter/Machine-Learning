
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def get_data(filename):
    '''
    读取文件
    :param filename: 文件名
    :return: 特征 & 标签，DataFrame
    '''
    data = pd.read_csv(filename,delimiter=',')
    x = data[['density', 'sugar_rate']].values
    y = data['label']
    return x, y


def deal_data(x, y, ratio=0.2):
    '''
    将数据分为训练集和测试集
    :param x: 样本, 特征, DataFrame
    :param y: 标签, DataFrame
    :param ratio: 训练集与测试集比例
    :return: 训练集、测试集、训练集标签、测试集标签
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio, shuffle=True)
    return x_train, x_test, y_train, y_test


def train_test(x_train, x_test, y_train):
    '''
    调用sklearn中的bagging算法,基学习器选用决策树
    :param x_train: 训练集
    :param y_train: 训练集标签
    :param x_test: 测试集
    :return: 对测试集预测结果,得到测试集预测标签
    '''
    # # 基学习器为决策树
    # tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    # clf = BaggingClassifier(base_estimator=tree, n_estimators=100, max_samples=1.0, bootstrap=True)

    # # 基学习器为逻辑回归
    # lr = LogisticRegression(penalty='l2', multi_class="multinomial", solver="newton-cg")
    # clf = BaggingClassifier(base_estimator=lr, n_estimators=100, max_samples=1.0, bootstrap=True)

    # 基学习器为KNN
    knn = KNeighborsClassifier(7)
    clf = BaggingClassifier(base_estimator=knn, n_estimators=100, max_samples=1.0, bootstrap=True)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)

    return y_pred, y_pred_train


def grid_train_test(x_train, x_test, y_train):
    """
    加入网格参数调参法，调参得到最优随机森林参数
    :param x_train: 训练集
    :param y_train: 训练集标签
    :param x_test: 测试集
    :return: 测试集预测结果、训练集预测结果
    """
    # 基学习器为逻辑回归
    lr = LogisticRegression(penalty='l2', multi_class="multinomial", solver="newton-cg")

    bagging_param_grid = {'n_estimators': [50, 100], 'bootstrap': [True, False]}
    clf = GridSearchCV(BaggingClassifier(base_estimator=lr), param_grid=bagging_param_grid, cv=5)

    clf.fit(x_train, y_train)
    print(clf.best_params_, clf.best_score_)

    y_pred = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)

    return y_pred, y_pred_train


def  evaluate(y_test, y_pred, y_train, y_pred_train):
    '''
    评估预测结果
    :param y_test: 测试集真实标签
    :param y_pred: 测试集预测标签
    :return: 打印出评估的结果
    '''
    test_accuracy = accuracy_score(y_pred, y_test)
    train_accuracy = accuracy_score(y_pred_train, y_train)

    result = classification_report(y_test, y_pred)

    print("测试集的准确率为: " + str(test_accuracy))
    print("训练集的准确率为: " + str(train_accuracy))
    print(result)


if __name__ == "__main__":
    # 1、读取文件, 并将数据分为训练集和测试集
    filename = '../dataset/watermelon3_0a.csv'
    x, y = get_data(filename)

    # 2、数据分析
    # 3、数据处理：划分数据集（注意数据的乱序）
    x_train, x_test, y_train, y_test = deal_data(x, y, 0.2)

    # 4、模型训练与测试：使用Bagging模型预测玻璃类别；并使用使用“GridSearchCV”对Bagging调参，得到最优模型；
    # y_pred, y_pred_train = train_test(x_train, x_test, y_train)
    # 使用网格参数调参训练
    y_pred, y_pred_train = grid_train_test(x_train, x_test, y_train)

    # 5、模型评估
    print("Bagging算法的预测结果如下: ")
    evaluate(y_test, y_pred, y_train, y_pred_train)



