# 추천 시스템

**종류**

- Content based filtering : 콘텐트 기반 필터링
- Collaborative filtering : 협업 필터링(사용자가 평가한 다른 아이템을 기반으로 사용자가 평가하지 않은 아이템의 예측 평가를 도출하는 방식)
  - Nearest Neighbor : 최근접 이웃 협업 필터링(index 간의 유사도 확인)
    - User-User : User가 Index, Item이 Feature
    - Item-Item : Item이 Index, User가 Feature. 일반적으로 User-User보다 더 자주 사용됨. 
  - Latent Factor : 잠재 요인 협업 필터링



이이템 기반 최근접 이웃 방식은 '아이템 간의 속성'이 얼마나 비슷한지를 기반으로 추천한다고 착각할 수 있다. 하지만 컨텐츠 기반 필터링은 컨텐츠 간의 유사도만을 가지고 측정한 것이고, 아이템 기반 협업 필터링은 개인적인 취향을 반영한 것이다. 주의

일반적으로 최근접 이웃 협업 필터링은 사용자 기반 보다는 아이템 기반 필터링이 정확도가 더 높다. 이유는 비슷한 영화를 좋아한다고 해서 사람들의 취향이 비슷하다고 판단하기는 어려운 경우가 많기 때문이다. 매우 유명한 영화는 취향과 관계없이 대부분의 사람이 관람하는 경우가 많고, 사용자들이 평점을 매긴 영화의 개수가 많지 않은 경우가 일반적인데 이를 기반으로 다른 사람과의 유사도를 비교하기 어려운 부분도 있기 때문이다.

코사인 유사도는 추천 시스템의 유사도 측정에 각장 많이 사용된다. 

</br>

## 잠재 요인 협업 필터링(Latent Factor C.F)

원본 행렬을 SVD(Singular Vector Decomposition), NMF(Non-Negative Matrix Factorization)와 같은 차원 감소 기법으로 분해하여 잠재 요인을 추출한다(행렬 분해, 고차원 희소 행렬을 저차원 밀집 행렬 P와 Q로 분해). 분해된 데이터 세트를 다시 내적 곱으로 결합하면서 사용자가 예측하지 않은 아이템에 대한 평점을 도출하는 방식이다.

행렬 분해에 의해 추출되는 ''잠재 요인''이 정확히 어떤 것인지는 알 수 없다.

SVD는 Null이 없는 행렬에만 적용할 수 있기 때문에 확률적 경사 하강법(SGD)이나 ALS(Alternating Least Squares) 방식을 이용해 수행한다.

</br>

## 논문 및 코드

**Training Deep AutoEncoders for Collaborative Filtering(2017.08)**

https://github.com/NVIDIA/DeepRecommender

**Graph Convolutional Matrix Completion(2017.10)**

https://github.com/tkipf/gae

**Session-based Recommendations with Recurrent Neural Networks(2015.11)**

https://github.com/hidasib/GRU4Rec

**Recurrent Neural Networks with Top-k Gains for Session-based Recommendations(2017.06)**

https://github.com/hidasib/GRU4Rec

**Session-based Recommendation with Graph Neural Networks(2018.11)**

https://github.com/CRIPAC-DIG/SR-GNN

**Wide & Deep Learning for Recommender Systems(2016.06)**

https://github.com/shenweichen/DeepCTR

**DeepFM: A Factorization-Machine based Neural Network for CTR Prediction(2017.03)**

https://github.com/shenweichen/DeepCTR