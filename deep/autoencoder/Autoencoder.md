## Autoencoder



### Backpropagation을 통해 DNN을 학습시키기 위한 loss function 가정

1. 전체 train 샘플의 loss는 각 train 샘플의 loss 합과 같다.
2. loss는 DNN의 출력(예측) 및 정답을 이용한 함수다.

</br>

### Maximum likelihood

회귀 또는 분류가 정답과의 예측값의 차이를 비교했다면, Masimum likelihood는 **정한** 확률 분포 **모형**에서 출력이 나올 확률을 최대로 하는 확률 분포를 찾는 것이다. 따라서 loss function은  −log𝑝(𝑦|𝑓 (𝑥))와 같이 정의하고, 조건부 확률에서 log를 씌운 이유는 loss function의 가정을 따르기 위함이다. 정한 확률 분포 모형이 가우시안 분포를 따른다고 가정할 경우 loss function은 MSE와 동일하고, 베르누이 분포를 따른다고 가정할 경우 loss function이 cross entropy와 동일하다. 따라서 회귀 모델에서 MSE를 사용하는 이유는 네트워크의 출력 값이 가우시안 분포를 따를 것이라고 가정하기 때문이라고 할 수 있다. 마찬가지로 분류 모델에서 cross entropy를 사용하는 것도 네트워크의 출력 값이 베르누이 분포를 따른다고 가정하기 때문이라고 할 수 있다.

</br>

### Variational Autoencoder(VAE)

>  생성기에 대한 확률 모델 분포를 가우시안으로 할 경우, MSE 값이 더 작은 것이 p(x) 기여하는 바가 더 크다. 하지만 MSE가 더 작은 데이터 샘플이 의미적으로 더 가깝지 않은 경우가 많기 때문에 올바른 확률 값을 구하기 어렵다. 이런 문제를 해결하기 위해 VAE를 활용한다.

</br>

Naver D2 : https://www.youtube.com/watch?v=o_peo6U7IRM

