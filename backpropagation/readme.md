# XOR & back propagation

## perceptron

다수의 신호를 입력으로 받아 하나의 신호를 출력

단층 perceptron은 그래프에서 선형 함수라고 생각하면 이해하기 쉽다

ex) AND, NAND, OR, XOR gate...



## XOR

> XOR 도 hypothesis로 만들 수 있을까?
>
> 단층 perceptron으론 XOR gate만들 수 없다!



* truth table



![00](./00.jpg)



하나만으론 안된다. 그래서 세개로 표현했더니 가능 !

실제로 4개의 데이터로 layer1만으로 10000번 돌리면 accuracy : 50% 였던게 layer2를 합쳤을 때 accyracy : 100% 나온다



![01](./01.jpg)





그것이 바로 neural network. layer1의 두 개를 합쳐서 하나로 표현해도 무방



![02](./02.jpg)



![03](./03.jpg)



## back propagation

> 예측값과 실제값의 차이(cost, error)를 통해 뒤에서 부터 앞으로 추적하여 조정해야 되는 값(weight, bias)의 기울기를 계산하겠다







![04](./04.jpg)



* chain rule

  합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다

  chain rule 적용해서 df/dw(w가 f에 얼마나 큰 영향을 미치는지), df/dx, df/db 등등 전부 구할 수 있다

  따라서 실제값과 예측값의 error를 기반으로 구하고자 하는 값들을 변경한다






![05](./05.jpg)



![06](./06.jpg)





### 덧셈 node

- 덧셈 노드의 수식

  ![add1](./add1.jpg)

  

- 덧셈 노드의 로컬 그래디언트

  ![add2](./add2.jpg)

  

- 덧셈 노드의 계산그래프. 현재 입력값에 대한 Loss의 변화량은 로컬 그래디언트에 흘러들어온 그래디언트를 각각 곱한다. 덧셈 노드의 역전파는 흘러들어온 그래디언트를 그대로 흘려보내는 걸 확인할 수 있다



![add](/Users/PJS/Desktop/github/ml/XOR&backpropagation/addNode.png)





### 곱셈 node

- 곱셈 노드의 수식

  ![mul1](./mul1.jpg)

  

- 곱셈 노드의 로컬 그래디언트

  ![mul2](./mul2.jpg)

  

- 곱셈 노드의 계산그래프. 현재 입력값에 대한 Loss의 변화량은 로컬 그래디언트에 흘러들어온 그래디언트를 각각 곱한다. 곱셈 노드의 역전파는 순전파 때 입력 신호들을 서로 바꾼 값을 곱해서 하류로 흘려보내는 걸 확인할 수 있다





![mul](/Users/PJS/Desktop/github/ml/XOR&backpropagation/mulNode.png)





### Sigmoid node

- **시그모이드(sigmoid)** 함수

  ![sigmoid1](./sigmoid1.jpg)

  

- 시그모이드 노드의 로컬 그래디언트

  ![sigmoid2](./sigmoid2.jpg)

- 계산그래프

![simoid](/Users/PJS/Desktop/github/ml/XOR&backpropagation/sigmoid.png)





### ReLU Node

* **활성화함수(activation function)**로 사용되는 **ReLU**는 수식

  ![relu1](./relu1.jpg)

* ReLU 노드의 로컬 그래디언트

  ![relu2](./relu2.jpg)

* 계산그래프



![relu](./relu.png)







### softmax - with - loss node

뉴럴네트워크 말단에 보통 **Softmax-with-Loss** 노드를 둡니다. Softmax-with-Loss란 소프트맥스 함수와 **교차 엔트로피(Cross-Entropy)** 오차를 조합한 노드를 뜻한다. 소프트맥스 함수와 교차 엔트로피의 수식은 아래와 같다

*ak = 노드의 입력값, L=노드의 출력값(Loss), tk=정답 레이블(0 혹은 1), n=정답 범주 개수*

![softMax1](./softMax.jpg)



* 계산 그래프

![softMax1](./softMax.png)









![a](./a.jpg)



code작성 할 때 이런식으로 dimension 맞춰 준다. 처음 W1의 2는 X의 개수이고, 오른쪽은 임의의 값이다. W4의 오른쪽 값은 Y의 dimension이 될 것이고 b4도 이와 같이 맞춰준다





tensorflow에서는 미분을 통해 back propoagation을 적용하기 위해 그래프 형태로 만들어 놓았다



![08](./08.jpg)





![09](./09.jpg)