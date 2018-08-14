# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Linear Regression4
# 파일에서 데이터 불러오기

# data type이 전부 float32로 동일해야한다
xy = np.loadtxt("./data-01-test-score.csv", delimiter=",", dtype=np.float32)
X_data = xy[:, 0:-1]
Y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# tesnorflow가 자체적으로 변경시키는 변수. variable
# random_normal은 shape만 지정해주고 아무 값이나 선정
W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


# 세션에 그래프를 launch시킨다
sess = tf.Session()
# W, b와 같은 variable을 미리 초기화한다 !!
sess.run(tf.global_variables_initializer())

for step in range(2000):
    W_val, b_val, hy_val, _ = sess.run([W, b, hypothesis, train], feed_dict={X:X_data, Y:Y_data})
    if step % 20 == 0:
        print(step, W_val, b_val, hy_val)

print("i got 90, 94, 97. and my score will be", sess.run(hypothesis, feed_dict={X:[[90, 94, 97]]}))