# logistic classification

### example(binary classifiacation)

spam detection : spam or ham

facebook feed : show or hide

credit card fraudulent transaction detection : legitimate or fraud



### description

linear regression만으론 정의역의 값이 너무 크거나 작은 경우 그 결과 값이 크게 벗어날 수 있다

따라서 아래의 g(z)함수를 이용하며 이는 어떤 정의역에 대해서도 0 ~1사이의 결과 값을 갖는다



![im01](./01.jpg)



따라서 logistic 가설을 세운다면 아래와 같다



![im02](./02.jpg)



하지만 위의 H(X)로 cost function을 만들면 각 위치마다 최소가 되는 지점을 찾기가 어렵다. 왜냐면 기울기가 0이 되는 것을 통해 최소 지점을 찾는데, 기울기가 0이 되는 곳이 여러 곳에 위치하기 때문. 아래 참조



![im03](./03.jpg)



따라서 이를 해결하기 위해 cosf function을 기존의 linear regression에서와는 다르게 만들어야 한다



![04im](04.jpg)

이와 같이 적용할 수 있는 이유는 y값이 1일 때와 0일 때(binary기 때문에 y값은 두 가지 밖에 없다) 아래의 그래프처럼 나오기 때문이다



![05im](./05.jpg)



y(실제값)이 1일 때, H(X)는 1에 가까워 질 수록 cost function(둘의 차이)는 0으로 수렴한다. H(X)는 0에 가까워 질 수록 cost function(둘의 차이)는 무한히 커진다

반대로 y(실제값)이 0일 때, H(X)는 0에 가까워 질 수록 cost function은 무한히 커지고, 1에 가까워 질 수록 0으로 가까워진다

실제로 두 그래프를 붙여보면 기존의 linear regression에서 cost function 그래프(밥그릇 모양)과 비슷하다는 것을 확인할 수 있다

![im06](./06.jpg)

따라서 cost function을 정리하면 위와 같다. y의 값에 따라 나누는 것이 불편하기 때문에 가장 아래처럼 한번에 적용