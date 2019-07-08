import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

# 路径
path = 'LogiReg_data.txt'
# 原文件没有表头，加一个表头：'Exam1','Exam2','Admitted'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
# 输出读取数据
# print(pdData)
pdData.head()
pdData.shape

# 录取
positive = pdData[pdData['Admitted'] == 1]
# 未录取
negative = pdData[pdData['Admitted'] == 0]

# fig表示窗口，ax表示坐标轴
fig, ax = plt.subplots(figsize=(10, 5))
# 散布点，c='b'表示蓝色。 marker='x'表示标记  label左为左上角的标签
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


# 绘图
# plt.()


# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# creates a vector containing 20 equally spaced values from -10 to 10
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nums, sigmoid(nums), 'r')


# 绘图
# plt.()


# 定义模型函数
# X 是样本数据，它的每一行都是一个样本，每一列为样本的某一个特征。
# theta 表示参数，它是我们通过学习获得的，其中，对于每一个特征，都对应一个 theta
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


# 在第 0 列，插入一列，名称为"Onces"，数值全为 1
pdData.insert(0, 'Onces', 1)
# X：训练数据   Y：目标值
# 将数据的panda表示形式转换为对进一步计算有用的数组
# 这个方法过时会有警告
orig_data = pdData.as_matrix()
cols = orig_data.shape[1]
X = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:cols]
# 初始化theta
theta = np.zeros([1, 3])

X.shape, y.shape, theta.shape


# 定义损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply((1 - y), np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


# 将数值带入计算损失值
cost(X, y, theta)


# 计算梯度
# 计算每个参数的梯度方向
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):  # for each parmeter
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)
    return grad


# 设置三种策略
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stopCriterion(type, value, threshold):
    # 设定三种停止策略
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD:
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        return np.linalg.norm(value) < threshold


import numpy.random


# 洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y


import time


def descent(data, theta, batchSize, stopType, thresh, alpha):
    #  梯度下降
    init_time = time.time()
    # 迭代次数
    i = 0
    # batch
    k = 0
    X, y = shuffleData(data)
    # 计算的梯度
    grad = np.zeros(theta.shape)
    # 损失值
    costs = [cost(X, y, theta)]

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        # 取batch数量个数据
        k += batchSize
        # 这个 n 是在运行的时候指定的，为样本的个数
        if k >= n:
            k = 0
            # 重新洗牌
            X, y = shuffleData(data)
        # 参数更新
        theta = theta - alpha * grad
        # 计算新的损失
        costs.append(cost(X, y, theta))
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh):
            break
    return theta, i - 1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    # import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta


# 选择的梯度下降方法是基于所有样本的
# 当n值指定为100的时候,相当于整体对于梯度下降,为什么呢?因为我的数据样本就100个.
# 传进来的数据是按照迭代次数进行停止的,
# 指定迭代次数的参数是thresh=5000.学习率是alpha=0.000001.
n = 10
# runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)
# plt.show()

# 设定阈值 1E-6, 差不多需要110 000次迭代
runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)
# plt.()

from sklearn import preprocessing as pp

scaled_data = orig_data.copy()
scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])


# 设定阈值
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]


scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
