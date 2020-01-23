# -*- coding: utf-8 -*-
import tensorflow as tf

# Linear Regression2
# GradintDescentOptimizer 라이브러리 사용하지 않고 구현하기

# train data를 sess.run할 때 값을 지정해준다
# shape=[None]은 어떤 shape의 data도 상관 없다는 뜻
X = [1,2,3]
Y = [1,2,3]

# tesnorflow가 자체적으로 변경시키는 변수. variable
# random_normal은 shape만 지정해주고 아무 값이나 선정
W = tf.Variable(5.0)

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

# linaerRegression1에서 optimizer 부분을 풀어 쓴 것. 그냥 tf.train.GradientDescentOptimizer 쓰는게 편하다
learning_rate = 0.1
gradient = tf.reduce_mean((hypothesis - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# # gradient를 변경하고 싶다면
# # 아래 부분 동일하게 적어주고
# cost = tf.reduce_sum(tf.square(hypothesis - Y))
# optimizer = tf.train.GradintDescentOptimizer(learning_rate=0.1)
# # gradient를 구해서 변경한 다음(여기선 변경하지 않았음)
# gvs = optimizer.compute_gradients(cost, [W])
# # 변경된 gvs를 gradient로 적용한다
# apply_gradients = optimizer.apply_gradients(gvs)


# 세션에 그래프를 launch시킨다
sess = tf.Session()
# W, b와 같은 variable을 미리 초기화한다 !!
sess.run(tf.global_variables_initializer())

for step in range(20):
    sess.run(update)
    print(step, sess.run([cost, W]))
