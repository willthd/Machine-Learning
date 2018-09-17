# neural network

> perceptron에서는 가중치를 인간이 직접 설정해줬었다면, neural network에서는 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력을 갖춘다
>
> 단순 perceptron은 단층 네트워크에서 계단함수(임계값을 경계로 출력이 바뀌는 함수)를 활성함수로 사용한 모델을 가리키고, 다층 perceptron은 neural network(여러 층으로 구성되고 시그모이드 함수 등의 매끈한 활성화 함수를 사용하는 네트워크)을 가리킨다
>
> 활성화 함수를 계단 함수에서 다른 함수(예를 들어 시그모이드 함수)로 변경하는 것이 neural network의 시작 !



## neural network의 구조

input layer - hidden layer - ouput layer



## neural network 잘 작동하려면?



### vanishing gradient 없애자 ! 



#### 1. ReLU



hidden layer가 많은 상황에서 sigmoid를  사용하게 되면, 뒤의 미분값과 0~1(sigmoid의 결과, actiovation function)의 값이 계속해서 곱해지기 때문에 가장 앞부분 layer로 도달할 수록 input은 최종 output에 미미한 영향을 주게 된다

하지만  가장 마지막단은 0~1 사이로 출력을 해야하기 때문에 sigmoid를 사용한다



- layer는 input layer(input가 연결되어 있는), hidden layer(중간), output layer(output가 연결되어 있는)로 구분한다



![00](./000.jpg)



### solution

0보다 큰 값에 한해서는 그 값의 영향을 남겨두자는 취지로 ReLU function 도입

![01](./001.jpg)





### 기타 activation function



![af](./af.jpg)







#### 2. Weight 초기화 잘하기

가중치의 초기값을 작게 만들고 싶다고 0으로 설정한다면?

안된다. 정확히는 0보다 가중치를 균일한 값으로 설정하게 되면 back propagation에서 모든 가중치의 값이 똑같이 갱신되기 때문

**RBM**

Restricted Boatman Machine, weight의 초기값을 잘 주는 방법으로 Deep Belief Network에 쓰임

기존의 방법은 W(weight)에 랜덤 값을 주어서, 모델을 학습시켰다. 하지만 W값을 초기에 얼마나 잘 주느냐에 따라 학습 속도가 달라지기 때문에 좋은 초기값을 정해주는 것은 중요하다



![02](./002.jpg)



두개의 인접한 layer에만 초점을 맞춘다. x값을 통해 어떤 결과 값이 나온다면 backword 방향으로 그 결과 값을 이용해 기존의 x와 가장 유사한 값이 나올 수 있도록 weight를 설정한다.

이와 같은 방법으로 모든 인접한 레이어 사이의 weight를 결정한다

하지만 RBM을 구해 적용하는 것 자체에 어려움이 따른다…따라서 보다 간단한 Xaivier initialization이나 He's initialization을 사용한다. 초기화 방법에는 다양한 방법들이 있지만 data에 따라 최적의 방법이 다르다. 아직 연구중인 단계



### Xavier/He initialization

- Xavier

  앞층의 node가 n개라면 표준편차가 1/루트n 인 분포를 사용한다

  activation function이 좌우 대칭일 경우(sigmoid, tanh)에 xavier 사용(mnist 분류할 때, ReLU랑 xavier사용 했던데?)



- He

  앞층의 node가 n개라면 표준편차가 2/루트n인 분포를 사용한다

  activation function이 ReLU일 때 He 사용



![03](./003.jpg)





### 3. Overfitting 줄이자 ! Regularization



#### * regularization strength

weight 벡터의 제곱의 합을 regurlarization strength만큼 곱해 더하는 방법이다. regurlaization을 적용하지 않는다면 regurlarization strength 값을 작게 하고, 중요하게 생각한다면 크게 한다

ex - l2reg

![im08](./08.jpg)



tensorflow에서 사용법

```python
l2reg = 0.001 * tf.reduce_sum(tf.square(W))
```



#### * Drop out

> 이는 훈련층에서 기존의 몇 개의 노드를 임의로 없애버리고 training하겠다는 것. training할 때마다 임의로 선정한 몇 개의 노드를 무시한채로 training하고, 마지막 평가할 때는 모든 노드를 포함한다(훈련 때 삭제한 비율을 )



모델이 깊어지는 경우 overfitting일어나기도 한다. 이 때, overfitting을 막기 위해선 regularization을 하는데, 신경망 모델이 복잡해지면 가중치 감소만으로는 대응하기 어려운 경우가 있다. 이럴 때 흔히 drop out이라는 방법을 사용한다



![04](./004.jpg)



![05](./005.jpg)



- 주의

  학습시에만 drop out을 적용하고, 실전 평가에서는 drop out을 하지 않는다

  train : dropout_rate = 0.7(대개 0.5)

  evaluate : dropout_rate = 1 !!!

  tensorflow 1.0 이후 부터 dropout_reate 대신 keep_prob으로 설정해준다. 얼만큼 노드를 살릴것인지



  ```python
  L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
  ```





![06](./006.jpg)







#### * Ensemble

> 초기값이 서로 다른 몇 개의 모델을 통해 학습 시킨 후 나온 예측값을 조합해 예측하는 방법. 평가율이 2% ~ 5%까지 올라간다. 앙상블 학습은 드롭아웃과 밀접하다. 드롭아웃은 앙상블 학습과 같은 효과를 (대략) 하나의 네트워크로 구현했다고 생각할 수 있다. data set이 많을 때 주로 사용



![07](./007.jpg)





## optimizer

> optimizer마다 장단점이 있기 때문에 모델 마다 다른 optimizer를 사용한다. 대개 SGD와 adam을 많이 사용한다



### SGD (stochastic gradient descent)

SGD는 기울기가 하강하는 방향으로 이동하여 매개변수를 갱신하는 방법. 단순하고 구현하는 방법도 쉽지만, 비등방성(anisotropy) 함수(방향에 따라 기울기가 달라지는 함수)에서는 탐색 경로가 비효율적. 또한 어떤 함수에 대해서는 지그재그로 탐색하는 경우도 있음



### momentum

어떤 함수에 대해 SGD보다 덜 지그재그로 탐색함



### adaGrad

매개변수를 갱신할 때마다 학습률을 조정(더 작게)한다. 매개변수의 원소 중에서 많이 움직인 원소는 학습률이 낮아진다는 뜻인데, 다시 말해 학습률 감소가 매개변수의 원소마다 다르게 적용된다는 것. 지그재그 효과 줄일 수 있음



### adam

momentum과 adaGrad의 융합 방법. 편향 보정이 진행된다. 



```python
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
```





## additional



### 비선형 함수

neural network에서는 activation function으로 비선형 함수를 사용한다. 이유는 선형 함수(y=ax + b)를 사용할 경우 그것은 hidden layer가 없는 구조로도 충분히 만들 수 있기 때문이다. 그럼 층을 쌓는 neural network를 사용할 이유가 없음



### activation function

* Sigmoid

  ```python
  def sigmoid(x):
  	return 1 / (x + np.exp(-x))
  ```

  2클래스 분류에 쓰임



* Relu(Recitified Linear Unit)

  h(x) = x (x > 0)

  	= 0 (x <= 0)

  ```python
  def Relu(x):
      return np.maximum(0, x)
  ```




	sigmoid의 문제점

* 항등 함수(출력층, 주로 regression에서)

  입력을 그대로 출력

  회귀에 쓰임



* softMax(출력층, 주로 classification)

  

  ![01](./01.png)

  

  다중 클래스 분류에 쓰임

  출력층의 각 뉴런이 모든 입력 신호에서 영향을 받는다

  overflow(컴퓨터가 표현할 수 있는 값을 넘는 현상)를 막기 위해 exp 지수값에서 -C(입력값의 최대치) 해준다

  출력의 총 합은 1이다 (중요) -> 함수의 출력을 '확률'로 계산 가능

  기계학습의 문제 풀이는 **학습**과 **추론(예측하는 것, 회귀 or 분류)**의 두 단계를 거친다. neural network도 마찬가지인데, 추론 단계에서 분류 할 때, softmax function은 없어도 된다. softmax function 단조 함수라 적용해도 입력 원소의 대소 관계는 변하지 않기 때문이다. 따라서 현업에서도 계산에 드는 자원 낭비를 줄이고자 추론 단계에서 출력층의 softmax function은 생략하는 것이 일반적이다(학습 단계에서는 softmax 사용)



### 행렬의 곱

주의 !

```python
A = np.array([[1,2],[3,4],[5,6]])
A.shape		# (3,2)
B = np.array([7,8])
B.shape		# (2,)....이것은 (2,1)도 아니고, (1,2)가 아니다...따라서 둘의 곱 가능

np.dot(A, B).shape	# (3,)....

C = np.array([[1,2],[3,4]])
C.shape 	# (2,2)

np.dot(B, C).shape	# (2,)...가능...왜냐 B의 shape이 2이기 때문. 이것은 (2,1)도 아니고 (1,2)도 아니다. 또는 둘 다이기도 하다
```

