# perceptron

> 다수의 신호를 입력으로 받아 하나의 신호를 출력



단층 perceptron은 그래프에서 선형 함수라고 생각하면 이해하기 쉽다

perceptron을 활용해 AND, NAND, OR gate까지는 만들었는데, XOR이 안만들어진다 ? 

-> Marvin Minsky 왈(1969년, 첫번째 침체기) : 다층 perceptron(neural net의 시작)으로 가능하다. 그런데 문제는 Weight와 bias학습을 못시킨다(구하지 못한다).

-> Hinton왈 (1986년) : Backpropagation이면 가능하다 !

-> neural network가 deep해지면 Backpropagaton방법으론 앞단 쪽으로 갈 수록 에러가 전달이 잘 되지 않아 학습이 어려워진다. **vanishing gradient**(1995년, 두번째 침체기)

-> Hinton, Bengio (2006, 2007) : Weight를 초기화 잘한다면 가능해 !

잘 안된 또 다른 이유들 by Hinton :

labeled datasets이 너무 작다

컴퓨터 성능이 너무 느리다

잘못된 non-linearity방식을 사용했다(Sigmoid : X, ReLU : O)



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







![a](./a.jpg)



W1, W2, W3,에서 출력부분에 아무 값(10)이나 집어 넣는다. 그런데 정말 아무 값일까...?
