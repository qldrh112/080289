# 필요한 라이브러리 호출
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터셋 가져오기
data = pd.read_csv('../data/NVDA.csv')
print(data.dtypes)

# 날짜 칼럼을 인덱스로 사용
data['Date'] = pd.to_datetime(data['Date'])
# Date 칼럼을 인덱스로 사용
data.set_index('Date', inplace=True)

# 데이터 형식 변경
# 데이터 타입을 변경할 때는 'astype()' 사용하기
data['Volume'] = data['Volume'].astype(float)

# 훈련과 레이블 처리
X = data.iloc[:, [0, 1, 2, 4]]   # close를 제외한 모든 X를 사용
y = data.iloc[:, 3:4]   # 마지막 'Volume'을 레이블로 사용
print(X)
print(y)

# 데이터 분포 조정
# 데이터의 모든 값이 0~1 사이에 존재하도록 분산 조정
ms = MinMaxScaler()
# 데이터가 평균 0, 분산 1이 되도록 분산 조정
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_ms = ms.fit_transform(y)

X_train = X_ss[:200, :]
X_test = X_ss[200:, :]

y_train = y_ms[:200, :]
y_test = y_ms[200:, :]

print('Training Shape', X_train.shape, y_train.shape)
print('Testing Shape', X_test.shape, y_test.shape)

# 데이터셋의 형태 및 크기 조정

# Variable로 감싼 텐서는 .backwward()를 호출할 때, 자동으로 기울기가 계산
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print('Training Shape', X_train_tensors_f.shape, y_train_tensors.shape)
print('Testing Shape', X_test_tensors_f.shape, y_test_tensors.shape)

# LSTM 네트워크
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        # 클래스 개수
        self.num_classes = num_classes
        # LSTM 계층의 개수
        self.num_layers = num_layers
        # 입력 크기로 훈련 데이터셋의 칼럼 개수를 의미
        self.input_size = input_size
        # 은닉층의 뉴런 개수
        self.hidden_size = hidden_size
        # 시퀀스 길이
        self.seq_length = seq_length

        # LSTM 계층
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # 완전 연결층
        self.fc_1 = nn.Linear(hidden_size, 128)
        # 출력층
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 은닉 상태를 0으로 초기화
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # 셀 상태를 0으로 초기화
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # LSTM 계층에 은닉 상태와 셀 상태 적용
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # 완전 연결층 적용을 위해 데이터의 형태 조정
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

# 변수 값 설정

num_epochs, learning_rate = 5000, 0.0001
# 입력 데이터셋의 칼럼 개수, 은닉층의 뉴런/유닛 개수, LSTM 계층의 개수, 클래스 개수
input_size, hidden_size, num_layers, num_classes = 4, 2, 1, 1

model = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1])
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    # 전방향 학습
    outputs = model.forward(X_train_tensors_f)
    optimizer.zero_grad()
    # 손실 함수를 이용한 오차 계산(학습 결과와 레이블의 차이 계산)
    loss = criterion(outputs, y_train_tensors)
    loss.backward()

    # 오차 업데이트
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, loss: {loss.item():1.5f}')

# 모델 예측 결과를 출력하기 위한 데이터 크기 재구성
df_x_ss = ss.transform(data.iloc[:, [0, 1, 2, 4]])   # 데이터 정규화(분포 조정)
df_y_ms = ms.transform(data.iloc[:, 3:4])   # 데이터 정규화

df_x_ss = Variable(torch.Tensor(df_x_ss))
df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))
df_y_ms = Variable(torch.Tensor(df_y_ms))

# 모델 예측 결과 출력
train_predict = model(df_x_ss)
# 모델 학습 결과를 넘파이로 변경
predicted = train_predict.data.numpy()
label_y = df_y_ms.data.numpy()

# 모델 학습을 위해 전처리(정규화)했던 것을 해제, 그래프는 본래의 데이터를 사용할 것이므로
predicted = ms.inverse_transform(predicted)
label_y = ms.inverse_transform(label_y)
# 그래프로 표시
plt.figure(figsize=(10, 6))
plt.axvline(x=200, c='r', linestyle='--')

plt.plot(label_y, label='Actual Data')
plt.plot(predicted, label='Predicted Data')
plt.title('Time-Series Prediction - NVIDIA')
plt.legend()
plt.show()