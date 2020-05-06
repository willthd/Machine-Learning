# Interview

> 머신러닝 Interview 준비



### 손실 함수 종류

* MSE(Mean Squared Error)

* RMSE

* MAE

* CEE(Cross Entropy Error)



정확도를 놔두고 '손실 함수'라는 우회적인 방법을 택하는 이유는 딥러닝은 **오차역전파법**(역전파, back propagation) 방식을 활용해 오차를 줄여 나가는데 이 때 미분을 사용한다. 하지만 정확도 or AUC 등은 미분 불가능하기 때문에 매개변수의 미소 변화가 있어도 반응이 없거나 불연속적으로 변화한다. 따라서 '손실 함수'라는 우회적인 방법을 이용하는 것이며, 궁극적으로 손실 함수가 최소가 되는 최적의 매개 변수 값을 찾는다.(계단 함수를 활성화 함수로 사용하지 않는 이유와도 동일하다.)

기울기 산출 시, 오차역전파법 방식은 각 노드의 미분을 구해 저장하면서 중복 계산을 피할 수 있다. 따라서 수치 미분법보다 빠르고 효율적으로 계산 가능하다.(수치 미분법은 구현하기 더 쉬우며, 오차역전파법으로 구한 기울기를 확인하는 데 사용한다.)

</br>

### 확률적 경사 하강법 (SGD)

무작위로 골라낸 데이터(미니 배치)를 이용해 손실 함수를 기울기가 감소 하는 방향으로 최소화 시켜 최적의 매개변수를 찾는 방법

![sgd](/Users/bagjongsu/Desktop/github/Machine-Learning/interview/sgd.png)

</br>

### batch_size, epoch, iterations

train data의 개수가 20,000개. batch_size가 500이라고 가정하면 1epoch을 돌기 위해 40회 학습 필요하다. 그리고 30 epoch을 돌기 원한다면 30 * 40회 학습을 해야한다.

</br>

### Gradient Vainishing

nn의 학습 방식(weight의 갱신 방식)은 backpropagation으로 이루어지는데, 이 때 gradient(변화량, 기울기)값이 점차 줄어들어 학습을 방해하는 현상을 의미한다. backpropagation식에서 cost function의 gradient 항이 존재하는데, 이 값이 0에 가까워지는 것이다. 이는 활성화 함수를 sigmoid나 tanh를 사용할 경우 도함수의 결과 값이 각각 0~0.25, 0~1이기 때문에 신경망이 깊어질 수록 weight의 영향력이 소실되어 제대로 갱신되지 않고, 학습이 어려워진다. 이를 막기위해 Relu, Leaky Relu, Maxout, Elu 등의 활성화 함수를 사용한다. 더 좋은 해결책은 LSTM, GRU를 사용하는 것이다. 이유는 **이해안되니 나중에 다시 공부하자.**

</br>

### Gradient Exploding

그래디언트 소실과는 반대로 역전파에서 그래디언트가 점점 커져 입력층으로 갈수록 가중치 매개변수가 기하급수적으로 커지게 되는 경우가 있는데 이를 **그래디언트 폭주**(exploding gradient)라고 하며, 이 경우에는 발산(diverse)하게되어 학습이 제대로 이루어지지 않는다. RNN에서 발생하기 쉽다. 초기화로 막는다?

### Batch Noramalization

https://wikidocs.net/61375

