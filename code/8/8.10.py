# 可以使用Bagging来提升k近邻分类器的性能，每次随机抽样出一个子样本，
# 并训练一个k近邻分类器，对测试样本进行分类。最终取最多的一种分类。
import numpy as np
import matplotlib.pyplot as plt



def knn(train, test, num):
    output = []
    m, n = len(train), len(test)
    for i in range(n):
        dist_ij = []
        for j in range(m):
            d = np.linalg.norm(test[i, :] - train[j, :-1])
            dist_ij.append((j, d))
        id_min = sorted(dist_ij, key=lambda x: x[1])[:num]
        rate = [train[i[0], -1] for i in id_min]
        if sum(rate) / num >= 0.5:
            output.append(1)
        else:
            output.append(0)
    return output

data = np.array([
    [0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1], [0.608, 0.318, 1], [0.556, 0.215, 1],
    [0.403, 0.237, 1], [0.481, 0.149, 1], [0.437, 0.211, 1], [0.666, 0.091, 0], [0.243, 0.267, 0],
    [0.245, 0.057, 0], [0.343, 0.099, 0], [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0],
    [0.593, 0.042, 0], [0.719, 0.103, 0]])
X = data[:, 0:2]
Y = data[:, -1]
a = np.arange(0, 1.01, 0.01)
b = np.arange(0, 0.61, 0.01)
x, y = np.meshgrid(a, b)
k = 5#k近邻的数目
z = knn(data, np.c_[x.ravel(), y.ravel()], k)#训练
z = np.array(z).reshape(x.shape)#转换成数组
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.contourf(x, y, z, cmap=plt.cm.summer, alpha=.6)
label_map = {1: 'good', 0: 'bad'}
ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=plt.cm.Dark2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.6)
ax.set_ylabel('sugar')
ax.set_xlabel('density')
ax.set_title('decision boundary: %s-NN' % k)
plt.show()
