# linear regression

### Supervised / Unsupervised learning

supervised - : learning with labeled examples - training set

* regression
* classification(binary or multi-label)

unsupervised - : learning with un-labeled examples



### Linear Regression의 cost 최소화 알고리즘

![im01](./01.jpg)





현재 구하고자 하는 것은 W와 b다 ! 여기서 y는 실제 값. 



![im02](./02.jpg)



알파(learning rate)는 cost(W), 밥그릇 모양의 그래프에서 기울기 step의 size. 데이터에 따라 적당한 값이 다르다



![im03](./03.jpg)



cost(W)를 미분한 값은 곧 cost(W)의 기울기이고 이것이 음수일 때 W는 커져야 한다. 반대로 이것이 양수일 때 W는 작아져야 한다



하지만 cost function의 convex fuction이 다음과 같을 경우에는 gradient decent alrorithm을 사용할 수 없기 때문에 항상 적용 전 확인해보아야 한다



* 불가능한 경우

![im04](./04.jpg)

* 가능한 경우(어느 점에서 시작해도 가장 작은 값 동일)

![im05](./05.jpg)





**linear regression 구성 요소**

![06im](./06.jpg)



**multi variable일 때**

matrix를 사용한다. 대개 수학적으로 표기할 때엔 H(x) = WX라고 하지만, matrix를 사용할 때는 H(x) = XW라고 하고 이 순서에 맞게 matrix를 배치시킨다

![im07](./07.jpg)

![im08](./08.jpg)

![im09](./09.jpg)

