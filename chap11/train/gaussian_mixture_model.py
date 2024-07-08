# 라이브러리 호출 및 데이터 로딩
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
X_train = np.load('../data/data.npy')

# GMM 생성
# n_components는 가우시안 개수를 의미하므로, 가우시안 2개가 겹쳐 보이게 구성
gmm = GaussianMixture(n_components=2)
gmm.fit(X_train)

# 두 개의 가우시안 분포의 평균 벡터
print(gmm.means_)
print('\n')
# 두 개의 가우시안 분포의 공분산행렬
print(gmm.covariances_)

# X, Y, Z, XX의 의미
X, Y = np.meshgrid(np.linspace(-1, 6), np.linspace(-1, 6))
XX = np.array([X.ravel(), Y.ravel()]).T

# 각 격자점에서 로그 확률 밀도를 계산
Z = gmm.score_samples(XX)
Z = Z.reshape((50, 50))

# X, Y 좌표에 대한 Z값(로그 확률 밀도)을 기반으로 등고선을 그림
plt.contour(X, Y, Z)
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.show()