# -*- coding: utf-8 -*-
__author__ = 'yangwenfeng'
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt
import matplotlib
import operator
import time


def createData(dim = 200, cnoise = 0.2):
    '''
    生成数据集


    '''
    x, y = sklearn.datasets.make_moons(dim, noise = cnoise)


    plt.scatter(x[:,0], x[:,1], s = 40, c=y, cmap=plt.cm.Spectral)


    return x,y
def initSuperParameter(x):
    '''
    初始化超参数
    '''


    global num_examples
    num_examples = len(x) # 训练集的大小
    global nn_input_dim
    nn_input_dim = 2 # 输入层维数
    global nn_hiden_dim
    nn_hiden_dim = 3 # 隐藏层维数
    global nn_output_dim
    nn_output_dim = 2 # 输出层维数


    #global num_iter # 迭代代数
    #num_iter = 20000


    # 梯度下降参数
    global epsilon
    epsilon = 0.01 # 梯度下降学习步长
    global reg_lambda
    reg_lambda = 0.01 # 修正的指数
def buildModel(x, y, nn_hiden_dim, num_iter = 20000, print_loss = False):
    """
     输入：数据集, 类别标签, 隐藏层层数, 迭代次数, 是否输出误判率
     输出：神经网络模型
     描述：生成一个指定层数的神经网络模型
     """
    # 根据initSuperParameter 中的超参数随机生成网络初始参数
    np.random.seed(0)
    #np.random.randn函数生成随机矩阵
    #np.sqrt 返回均方根
    W1 = np.random.randn(nn_input_dim, nn_hiden_dim)/np.sqrt(nn_input_dim)#分母的意义是什么？
    b1 = np.zeros((1, nn_hiden_dim))
    W2 = np.random.randn(nn_hiden_dim, nn_output_dim)/np.sqrt(nn_hiden_dim)
    b2 = np.zeros((1, nn_output_dim))


    model = {}


    #梯度下降执行流程
    for i in range(0, num_iter):
        #前向计算部分
        z1 = x.dot(W1) + b1 # x点积W1 :[样本数， nn_input_dim]点积[nn_input_dim, nn_hiden_dim] = [样本数， nn_hiden_dim]
                            # 这里加上b1 不像matlab里需要扩展b1 可以直接加到所有的行中。
        a1 = np.tanh(z1)   # 激活函数使用tanh = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)  # 原始归一化#对输出层求exp函数，使得所有输出值大于0
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # 因为输出维度是2，所以求出这两个类别
        # 后向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # 加入修正项
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
        # 更新梯度下降参数
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        # 更新模型
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


        #在一定迭代次数之后输出一次误判率
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, x, y)))


        #画出决策边界，因为现在在对有标签对数据分类，搞个分类面出来数很有必要的
    plot_decision_boundary(lambda n: predict(model, n), x, y)
    plt.title("Decision Boundary for hidden layer size %d" % nn_hiden_dim)
        # plt.show()
    return model
def calculate_loss(model, x, y):
    """
    输入：训练模型, 数据集, 类别标签
    输出：误判的概率
    描述：计算整个模型的性能
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


    #计算误判率
    temp = probs[range(num_examples), y]# 这部分数我用来测试学习python基本语法对
    corect_logprobs = -np.log(probs[range(num_examples), y])# 为什么求-log（准确率）可以表征误判率，因为越接近1,-log(准确率)的值越小
    data_loss = np.sum(corect_logprobs)
    # 加入正则项修正错误(可选)
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss



def plot_decision_boundary(pred_func, x, y):
    """
      输入：边界函数, 数据集, 类别标签
      描述：绘制决策边界,用来直观对把两类数据分开
    """
    # 设置最小最大值, 加上一点外边界
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    # 根据最小最大值和一个网格距离生成整个网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 对整个网格预测边界值
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 绘制边界和数据集的点
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
def predict(model, n):
    """
        输入：训练模型, 预测向量
        输出：判决类别
        描述：预测类别属于(0 or 1)
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传播计算
    z1 = n.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


















def main():


    #第一步产生数据
    dataSet, labels = createData(200, 0.20)
    #第二步设置超参数
    initSuperParameter(dataSet)
    #基于参数构建网络
    my_nn_model = buildModel(dataSet, labels, 3,  print_loss=True)


    print("Loss is %f" % calculate_loss(my_nn_model, dataSet, labels))




if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s'% str(end - start))
    plt.show()


####最后输出结果
"""
/usr/bin/python2.7 /home/yuanlei/PycharmProjects/NN/nn_test.py
Loss after iteration 0: 0.441225
Loss after iteration 1000: 0.035236
Loss after iteration 2000: 0.033899
Loss after iteration 3000: 0.033623
Loss after iteration 4000: 0.033537
Loss after iteration 5000: 0.033503
Loss after iteration 6000: 0.033489
Loss after iteration 7000: 0.033481
Loss after iteration 8000: 0.033477
Loss after iteration 9000: 0.033474
Loss after iteration 10000: 0.033472
Loss after iteration 11000: 0.033471
Loss after iteration 12000: 0.033470
Loss after iteration 13000: 0.033470
Loss after iteration 14000: 0.033469
Loss after iteration 15000: 0.033469
Loss after iteration 16000: 0.033469
Loss after iteration 17000: 0.033469
Loss after iteration 18000: 0.033469
Loss after iteration 19000: 0.033468
Loss is 0.033468
finish all in 11.251442


Process finished with exit code 0
"""
