# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Linear Regression5
# 여러개의 파일에서 데이터 불러오기. 파일의 개수가 많으면 메모리에 못담으니까 queue이용한다

filename_queue = tf.train.string_input_producer(["./data-01-test-score.csv"], shuffle=False, name="filename_queue")
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

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

# start populating the filename queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2000):
    X_batch, Y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:X_batch, Y:Y_batch})
    if step % 10 == 0:
        print(step, "cost : ", cost_val, "\nhypothesis : \n", hy_val)

coord.request_stop()
coord.join(threads)