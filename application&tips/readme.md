# application & tips

> learning rate, data preprocessing, overfitting



## learning rate

learning rate 굉장히 중요하다. 대개 0.01 정도로 시작

* learning rate가 클 경우 -> overshooting. gradinent decent 알고리즘 적용할 경우 기울기가 적은 지점을 계속해서 찾는데 그 과정 중 아예 범주 밖으로 나가 이상한 결과 값이 나올 수 있다

![im01](./01.jpg)



* learning rate이 작을 경우 -> 결과의 변화가 너무 미세하게 나타난다

![im02](./02.jpg)



따라서 learnig rate의 초기 값을 0.01(대개)로 두고, 그 결과를 관찰하면서 learning rate을 조절한다



## preprocessing

![im04](./04.jpg)



실제 데이터는 한쪽으로 치우쳐 있거나, 산발적으로 흩어져 있는 경우가 많다. 이럴 때 zero-centered 또는 normalized하게 data를 preprocessing해야 한다



normalization의 한 방법으로 standardization이 있다

![im05](./05.jpg)



## overfitting

수집된 데이터에 정확한 결과를 위해 modeling을 과하게 한것. 아래 graph에서 model2는 일반화되어 있지 않고, 특정 data에 한해 적용할 수 있기 때문에 한계가 있다. machine learning에서 가장 큰 문제가 되고 있다

![im06](./06.jpg)



해결 방법은 traing data를 많이 갖고 있거나, regurlarization



![im07](./07.jpg)





regularization은 weight 벡터의 제곱의 합을 regurlarization strength만큼 곱해 더하는 방법이다. regurlaization을 쓰지 않는다면 regurlarization strength 값을 작게 하고, 중요하게 생각한다면 크게 한다



![im08](./08.jpg)



tensorflow에서 사용법

```python
l2reg = 0.001 * tf.reduce_sum(tf.square(W))
```



![im09](./09.jpg)