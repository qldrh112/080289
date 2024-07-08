# 라이브러리 호출 및 데이터셋 내려받기
import numpy as np
from sklearn.datasets import load_digits
from minisom import MiniSom
from pylab import plot, axis, show, pcolor, colorbar, bone

# 숫자 필기 이미지 내려 받기
digits = load_digits()
# 훈련 데이터셋
data = digits.data
# 정답(레이블)
labels = digits.target

# 훈련 데이터셋을 MiniSom 알고리즘에 적용
som = MiniSom(16, 16, 64, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
print('SOM 초기화.')
som.train_random(data, 10000)
print('\n. SOM 진행 종료')

bone()
pcolor(som.distance_map().T)
colorbar()

# 클래스에 대해 레이블 설정 및 색상 할당
for i in range(0, 10):
    labels[labels == str(i)] = i

markers = ['o', 'v', '1', '3', '8', 's', 'p', 'x', 'D', '*']
colors = ['r', 'g', 'b', 'y', 'c', (0, 0.1, 0.8), (1, 0.5, 0), (1, 1, 0.3), 'm', (0.4, 0.6, 0)]

# 시각화 처리
for cnt, xx in enumerate(data):
    # 승자(우승 노드) 식별
    w = som.winner(xx)
    plot(w[0]+.5, w[1]+.5, markers[labels[cnt]], markersize=12, markeredgewidth=2)
show()

