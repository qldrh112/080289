# 라이브러리 호출
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from torch.autograd import Variable
# 파라미터 목록을 가지고 있는 라이브러리
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

# GPU 사용에 필요
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)

# 데이터 전처리
import torchvision.transforms as transforms

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    # 평균을 0.5, 표준 편차를 1.0으로 데이터를 정규화(데이터 분포 조정)
    transforms.Normalize((0.5,), (1.0,))
])

# 데이터셋 내려 받기
from torchvision.datasets import MNIST

download_root = './MNIST_DATASET'

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)

# 데이터셋을 메모리로 가져오기
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 변수 값 지정
batch_size = 100
n_iters = 6000
# 전체 반복 횟수를 한 에포크 동안 배치의 개수로 나눈다.
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

# LSTM 셀 네트워크 구축
class LSTMcell(nn.Module):
    
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    # 모델의 파라미터 초기화
    def reset_parameters(self):
        # Xavier 초기화 기법
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        # 텐서를 1차원을 기준으로 4개로 쪼개라
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        # 입력 게이트에 시그모이드 활성화 함수 적용
        ingate = F.sigmoid(ingate)
        # 망각 게이트에 시그모이드 활성화 함수 적용
        forgetgate = F.sigmoid(forgetgate)
        # 셀 게이트에 탄젠트 활성화 함수 적용
        cellgate = F.tanh(cellgate)
        # 출력 게이트에 시그모이드 활성화 함수 적용
        outgate = F.sigmoid(outgate)

        # 이전 게이트 + 현재 게이트 상태
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))
        return (hy, cy)

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        # 은닉층의 뉴런/유닛 개수
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = LSTMcell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # (은닉층의 계층 개수, 배치 크기, 은닉층의 뉴런 개수) 형태를 갖는 "은닉" 상태를 0으로 초기화
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # (은닉층의 계층 개수, 배치 크기, 은닉층의 뉴런 개수) 형태를 갖는 "셀" 상태를 0으로 초기화
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), hidden_dim))


        outs = []
        # (은닉층의 계층 개수, 배치 크기, 은닉층의 뉴런 개수)크기를 갖는 셀 상태 텐서
        cn = c0[0, :, :]
        # (은닉층의 계층 개수, 배치 크기, 은닉층의 뉴런 개수)크기를 갖는 은닉 상태 텐서
        hn = h0[0, :, :]

        # x = torch.Size([32, 28, 28])와 같은 형태
        
        # LSTM 셀 계층을 반복하여 쌓아 올림
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)
        
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out
    
# 옵티마이저와 손실 함수 지정

input_dim = 28
hidden_dim = 128
layer_dim = 1
output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

seq_dim = 28
loss_list = []
iter = 0

for epoch in range(num_epochs):
    # 훈련 데이터 학습
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels)
    
        optimizer.zero_grad()
        outputs = model(images)
        # 손실 함수를 통한 오차 계산
        loss = criterion(outputs, labels)

        if torch.cuda.is_available():
            loss.cuda()
        
        loss.backward()
        # 파라미터 업데이트
        optimizer.step()
        loss_list.append(loss.item())
        iter += 1
        
        # 정확도 계산
        if iter % 500 == 0:
            correct, total = 0, 0
            # 검증 데이터셋을 이용한 모델 성능 검증
            for images, labels in valid_loader:
                if torch.cuda.is_available():
                    images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                else:
                    iamges = Variable(images.view(-1, seq_dim, input_dim))
            
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                # 총 레이블 수
                total += labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total
            print(f'Iteration: {iter}. Loss: {loss.item()}. Accuracy: {accuracy}')

# 테스트 데이터셋을 활용한 모델 예측 성능 확인
def evaluate(model, val_iter):
    corrects, total, total_loss = 0, 0, 0
    model.eval()

    for images, labels in val_iter:
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim)).to(device)

        logit = model(images).to(device)
        # reduction='sum'를 지정했기 때문에 모든 오차를 더합니다.
        loss = F.cross_entropy(logit, labels.to(device), reduction='sum')
        _, predicted = torch.max(logit.data, 1)
        total += labels.size(0)
        total_loss += loss.item()
        if torch.cuda.is_available():
            corrects += (predicted.cpu() == labels.cpu()).sum()
        else:
            corrects += (predicted == labels).sum()
    
    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss, avg_accuracy

# 모델 예측 성능 확인
test_loss, test_acc = evaluate(model, test_loader)
print(f'Test Loss: {test_loss:5.2f} Test Accuracy: {test_acc:5.2f}')