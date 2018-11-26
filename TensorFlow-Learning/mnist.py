# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 它能解决单层感知机所不能解决的非线性问题
# 获取mnist数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 注册默认session 后面操作无需指定session 不同sesson直接的数据是独立的
sess = tf.InteractiveSession()
# ----------------------------------参数设置---------------------------------------------
# in_units隐含层输入节点数 h1_units隐含层输出节点数 truncated_normal截断正太分布stddev为标准差
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# ----------------------------------网络输入设置---------------------------------------------
# 训练和预测的Dropout 的比率keep_prob(保留节点的概率)是不一样的通常训练小于1防止过拟合 预测等于1即全部特征来预测 所以这个不是个常量
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

# ----------------------------------模型设置----------------------------------------------
# 随机将一些节点设置为0 keep_prob便是保留不变的比例
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# ----------------------------------loss设置 优化算法设置----------------------------------------------
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 优化算法Adagrad
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# ----------------------------------通过优化算法训练模型--------------------------------------------------------------
# 全局初始化器run  模型训练流程需要 x , y_,keep_prob输入
tf.global_variables_initializer().run()
# 更多的层数需要更多的数据3000 keep_prob 0.75 就是有25%的节点置为0
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
# ----------------------------------评测结果--------------------------------------------------------------
# tf.argmax(y,1)为预测结果 tf.argmax(y_,1)为真实结果 返回是否正确
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# bool转float32 计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 因为是测试阶段 keep_prob为1         accuracy 98%左右  模型测试流程需要x,y_,keep_prob输入
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))