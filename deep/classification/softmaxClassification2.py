# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# softmax classification(fancy ver.)
xy = np.loadtxt("./data-04-zoo.csv", delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# 0 ~ 6이 결
nb_classes = 7

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

# one-hot-encoding
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

#1 이부분
'''
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
'''
#1 이렇게 대체
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# arg_max는 만약 결과 값이 [0.85, 0.12, 0.03] 나왔다면 이를 [1, 0, 0]으로 one-hot encoding시켜주는 함수
predict = tf.arg_max(hypothesis, 1)
real = tf.arg_max(Y_one_hot, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, real), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(4001):
        cost_val, acc_val, _ = sess.run([cost, accuracy, train], feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, "\n", cost_val, "\n", acc_val)

    pred = sess.run(predict, feed_dict={X:x_data, Y:y_data})
    # zip은 p, y를 묶겠다는 의미
    # flatten은 현재 y_data가 [[1], [0], ... ] 이런 형태인데 이를 [1, 0, ...]으로 바꾸겠다
    for p, y in zip(pred, y_data.flatten()):
        print("[{}]\nreal:{}\nprediction:{}".format(int(y) == p, y, p))