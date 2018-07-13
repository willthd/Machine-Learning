# XOR & back propagation

## XOR

> XOR 도 hypothesis로 만들 수 있을까?



![00](./00.jpg)



![01](./01.jpg)





* neural network



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





tensorflow에서는 미분을 통해 back propoagation을 적용하기 위해 그래프 형태로 만들어 놓았다



![08](./08.jpg)





![09](./09.jpg)