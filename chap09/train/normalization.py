# 라이브러리 호출
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 경로 지정 및 훈련과 테스트 용도로 분리
df = pd.read_csv('../data/diabetes.csv')
X = df[df.columns[:-1]]
y = df['Outcome']

X = X.values
y = torch.tensor(y.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# 훈련과 테스트용 데이터를 정규화
ms = MinMaxScaler()
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
# (?, 1) 형태를 갖도록 변경, 열의 수만 1로 설정
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ms.fit_transform(y_train)
y_test = ms.fit_transform(y_test)

# 커스텀 데이터셋 생성
class customdataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
    
# 데이터로더에 데이터 담기
train_data = customdataset(torch.FloatTensor(X_train),
                           torch.FloatTensor(y_train))
test_data = customdataset(torch.FloatTensor(X_test),
                          torch.FloatTensor(y_test))

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# 네트워크 생성
class binaryClassification(nn.Module):

    def __init__(self):
        super(binaryClassification, self).__init__()
        # 칼럼이 8개이므로 입력 크기는 8을 사용
        self.layer_1 = nn.Linear(8, 64, bias=True)
        self.layer_2 = nn.Linear(64, 64, bias=True)
        # 출력으로 당뇨인지 아닌지 나타내는 0과 1의 값만 가지므로
        self.layer_out = nn.Linear(64, 1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 =nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

# 손실 함수와 옵티마이저 지정
epochs = 1000 + 1
print_epoch = 100
LEARNING_RATE = 1e-2

model = binaryClassification()
model.to(device)
print(model)
BCE = nn.BCEWithLogitsLoss()
# 훈련 데이터셋에서 무작위로 샘플을 추출하고, 그 샘플만 이용해서 기울기를 계산
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# 모델 성능 측정 함수 정의
def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

for epoch in range(epochs):
    iteration_loss = 0.
    iteration_accuracy = 0.

    model.train()
    # 데이터로더에서 훈련 데이터셋을 배치 크기만큼 불러옵니다.
    for i, data in enumerate(train_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float()).to(device)
        # 모델에 적용하여 훈련 시킨 결과와 정답을 손실 함수에 적용
        loss = BCE(y_pred, y.reshape(-1, 1).float())
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch % print_epoch == 0):
        print(f'Train: epoch: {epoch} - loss: {iteration_loss/(i+1):.5f}; acc: {iteration_accuracy/(i+1):.3f}')
    
    iteration_loss = 0.
    iteration_accuracy = 0.
    model.eval()
    for i, data in enumerate(test_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float()).to(device)
        loss = BCE(y_pred, y.reshape(-1, 1).float())
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
    if (epoch % print_epoch == 0):
        print(f'Test: epoch: {epoch} - loss: {iteration_loss/(i+1):.5f}; acc: {iteration_accuracy/(i+1):.3f}')
