import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
print(trX.shape, trY.shape, teX.shape, teY.shape)
# 按照高斯分布初始化权重矩阵
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 定义神经网络模型
def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h))    # 激活函数采用sigmoid函数
    return tf.matmul(h, w_o)
X = tf.placeholder("float", [None, 784])#创建占位符,在训练时传入图片的向量
Y = tf.placeholder("float", [None, 10])#图像的label用一个10维向量表示

w_h = init_weights([784, 625]) # 输入层到隐藏层的权重矩阵,隐藏层包含625个隐藏单元
w_o = init_weights([625, 10])#隐藏层到输出层的权重矩阵

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # 计算py_x与Y的交叉熵
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)  #通过步长为0.05的梯度下降算法求参数
predict_op = tf.argmax(py_x, 1)# 预测阶段,返回py_x中值最大的index作为预测结果

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(101):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        if i % 20 == 0:
            print("迭代次数：" + str(i) + "\t预测准确度：", np.mean(np.argmax(teY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: teX, Y: teY})))
