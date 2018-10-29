# -*- coding: utf-8 -*-
import tensorflow as tf

# Linear Regression3
# multi variables

# train data를 sess.run할 때 값을 지정해준다
X_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
Y_data = [[152.], [185.], [180.], [196.], [142.]]
# shape은 3이고, data의 개수는 n개다 (=none)
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
