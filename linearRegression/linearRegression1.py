# -*- coding: utf-8 -*-
import tensorflow as tf

# Linear Regression1

# train data를 sess.run할 때 값을 지정해준다
# shape=[None]은 어떤 shape의 data도 상관 없다는 뜻
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# tesnorflow가 자체적으로 변경시키는 변수. variable
# random_normal은 shape만 지정해주고 아무 값이나 선정
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(cost)

# 세션에 그래프를 launch시킨다
sess = tf.Session()
# W, b와 같은 variable을 미리 초기화한다 !!
sess.run(tf.global_variables_initializer())

for step in range(2000):
    # 한번에 묶어서 계산한다. train은 값이 따로 없기에 비워둔다. train 찍으면 none 나옴
    cost_val, W_val, b_val, _  = sess.run([cost, W, b, train], feed_dict={X:[1, 2, 3], Y:[2.1, 3.1, 4.1]})
    if (step % 20 == 0):
        print("step : ", step, "cost : ", cost_val, "W : ", W_val, "b : ", b_val)

# test my model
print(sess.run(hypothesis, feed_dict={X:[5]}))