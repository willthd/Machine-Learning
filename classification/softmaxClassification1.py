# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# softmax classification
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])
W = tf.Variable(tf.random_normal([4, 3]), name="weight")
b = tf.Variable(tf.random_normal([3]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# arg_max는 만약 결과 값이 [0.85, 0.12, 0.03] 나왔다면 이를 0(가장 큰 index)으로 반환하는 함수
# 만약 hypothesis의 rank가 1이었다면 tf.arg_max(hypothesis, 0)으로 해야한다
predict = tf.arg_max(hypothesis, 1)
real = tf.arg_max(Y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, real), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, predict, train], feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, cost_val, hy_val)

    print(sess.run(accuracy, feed_dict={X:x_data, Y:y_data}))