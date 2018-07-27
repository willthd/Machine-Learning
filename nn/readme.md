# neural network

> perceptron에서는 가중치를 인간이 직접 설정해줬었다면, neural network에서는 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력을 갖춘다
>
> 단순 perceptron은 단층 네트워크에서 계단함수(임계값을 경계로 출력이 바뀌는 함수)를 활성함수로 사용한 모델을 가리키고, 다층 perceptron은 neural network(여러 층으로 구성되고 시그모이드 함수 등의 매끈한 활성화 함수를 사용하는 네트워크)을 가리킨다
>
> 활성화 함수를 계단 함수에서 다른 함수(예를 들어 시그모이드 함수)로 변경하는 것이 neural network의 시작 !



### neural network의 구조

input layer - hidden layer - ouput layer



### 비선형 함수

neural network에서는 actication function으로 비선형 함수를 사용한다. 이유는 선형 함수(y=ax + b)를 사용할 경우 그것은 hidden layer가 없는 구조로도 충분히 만들 수 있기 때문이다. 그럼 층을 쌓는 neural network를 사용할 이유가 없음



### actication function

* Sigmoid

  ```python
  def sigmoid(x):
  	return 1 / (x + np.exp(-x))
  ```

  2클래스 분류에 쓰임



* Relu(Recitified Linear Unit)

  h(x) = x (x > 0)

  ​	= 0 (x <= 0)

  ```python
  def Relu(x):
      return np.maximum(0, x)
  ```



* 항등 함수(출력층, 주로 regression에서)

  입력을 그대로 출력

  회귀에 쓰임



* softMax(출력층, 주로 classification)

  

  ![01](./01.png)

  

  다중 클래스 분류에 쓰임

  출력층의 각 뉴런이 모든 입력 신호에서 영향을 받는다

  overflow(컴퓨터가 표현할 수 있는 값을 넘는 현상)를 막기 위해 exp 지수값에서 -C(입력값의 최대치) 해준다

  출력의 총 합은 1이다 (중요) -> 함수의 출력을 '확률'로 계산 가능

  기계학습의 문제 풀이는 **학습**과 **추론**(순전파, forward propagation)의 두 단계를 거친다. neural network도 마찬가지인데, 추론 단계에서 분류 할 때, softmax function은 없어도 된다. softmax function 단조 함수라 적용해도 입력 원소의 대소 관계는 변하지 않기 때문이다. 따라서 현업에서도 계산에 드는 자원 낭비를 줄이고자 추론 단계에서 출력층의 softmax function은 생략하는 것이 일반적이다



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

