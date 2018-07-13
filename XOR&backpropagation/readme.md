# XOR & back propagation

## XOR

> XOR 도 hypothesis로 만들 수 있을까?



* truth table



![00](./00.jpg)



하나만으론 안된다. 그래서 세개로 표현했더니 가능 !

실제로 4개의 데이터로 layer1만으로 10000번 돌리면 accuracy : 50% 였던게 layer2를 합쳤을 때 accyracy : 100% 나온다



![01](./01.jpg)





그것이 바로 neural network. layer1의 두 개를 합쳐서 하나로 표현해도 무방



![02](./02.jpg)



![03](./03.jpg)



## backpropagation

> 예측값과 실제값의 차이(cost, error)를 통해 뒤에서 부터 앞으로 추적하여  조정해야 되는 값(weight, bias)을 계산하겠다



![04](./04.jpg)

![04-1](./04-1.jpg)



* chain rule

  chain rule 적용해서 df/dw(w가 f에 얼마나 큰 영향을 미치는지), df/dx, df/db 등등 전부 구할 수 있다 !

  따라서 실제값과 예측값의 error를 기반으로 그 값을 변경 가능하다 ...!

  

![05](./05.jpg)



![06](./06.jpg)



* exmaple - sigmoid



![07](./07.jpg)



![a](./a.jpg)



code작성 할 때 이런식으로 dimension 맞춰 준다. 처음 W1의 2는 X의 개수이고, 오른쪽은 임의의 값이다. W4의 오른쪽 값은 Y의 dimension이 될 것이고 b4도 이와 같이 맞춰준다





tensorflow에서는 미분을 통해 back propoagation을 적용하기 위해 그래프 형태로 만들어 놓았다



![08](./08.jpg)





![09](./09.jpg)