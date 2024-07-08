# 라이브러리 호출
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
# K-평균 군집화
from kmeans_pytorch import kmeans, kmeans_predict

# 데이터셋 불러오기
df = pd.read_csv('../data/iris.csv')
# 데이터셋에 대한 전반적인 정보를 출력
df.info()
print('-' * 30)
# 아이리스 데이터셋의 데이터 출력
print(df)

# 워드 임베딩
data = pd.get_dummies(df, columns=['Species'])
data

# 데이터셋 분리
from sklearn.model_selection import train_test_split

x, y = train_test_split(data, test_size=0.2, random_state=123)

# GPU 사용하도록 설정
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# 특성 스케일링
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# data의 평균과 표준 편차를 계산하여 x에 적용한다.
X_scaled = scaler.fit(data).transform(x)
y_scaled = scaler.fit(data).transform(y)

# 데이터를 텐서로 변경
x = torch.from_numpy(X_scaled)
y = torch.from_numpy(y_scaled)

# 훈련과 테스트 데이터셋 크기
print(x.size())
print(y.size())
print(x)

# K-평균 군집화 적용
# 아이리스(붓꽃) 유형이 3개이므로
num_clusters = 3
cluster_ids_x, cluster_centers = kmeans(
    X=x , num_clusters=num_clusters, distance='euclidean', device=device
)
# 클러스터 ID와 클러스터 중심 값 확인
# [0, 1, 2] 3개의 레이어 중 하나가 나올 것이다.
print(cluster_ids_x)
# K-평균 군집화 알고리즘에서 계산된 클러스터의 중심, 각 클러스터 중심은 해당 클러스터에 속하는 데이터 점의 평균 위치
print(cluster_centers)

# K-평균 군집화 예측
# 예측을 위해 kmeans_predict()를 사용
cluster_ids_y = kmeans_predict(
    y, cluster_centers, 'euclidean', device=device
)

# 테스트 데이터셋에 대한 클러스터 ID
print(cluster_ids_y)

# 예측 결과 그래프로 확인
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3), dpi=160)
# 테스트 데이터셋에 적용된 클러스터 결과 출력
plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='viridis', marker='x')

plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1],
    c='white',
    alpha=0.6,
    edgecolors='black',
    linewidths=2
)

plt.tight_layout()
plt.show()