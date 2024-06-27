# 라이브러리 호출
import torch
import torch.nn as nn
import torch.optim as optim
# 사전 학습된 모델을 이용하고자 할 때 사용하는 라이브러리
import torchvision.models as models
from torchvision import transforms, datasets

import matplotlib
import matplotlib.pyplot as plt
import time
# 함수에 넘겨주는 인수 값에 따라서 학습률 감소나 조기 종료를 도와주는 라이브러리
import argparse
from tqdm import tqdm
import numpy as np
import sys

# 출력 그래프에서 격자로 숫자 범위가 눈에 잘 띄도록 하는 스타일
matplotlib.style.use('ggplot')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터셋 전처리
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.486, 0.496],
                         std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.486, 0.496],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(
    root=r'../data/archive/train',
    transform=train_transform,
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True,
)
val_dataset = datasets.ImageFolder(
    root=r'../data/archive/test',
    transform=val_transform,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=True,
)

# 모델 생성
def resnet50(pretrained=True, requires_grad=False):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # 파라미터를 고정하여 backward() 중에 기울기가 계산되지 않도록 함
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # 파라미터 값이 backward() 중에 기울기 계산에 반영됩니다.
    else:
        for param in model.parameters():
            param.requires_grad == True
    
    # 마지막 분류를 위한 계층은 학습을 진행
    model.fc = nn.Linear(2048, 2)
    return model

# 학습률 감소
class LRScheduler():
    
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        # 모델 성능에 대한 개선이 없을 때 모델의 개선을 유도하는 콜백 함수
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            # 모델의 오차가 내려가지 않을 때
            mode = 'min',
            patience = self.patience,
            # 학습률 감소 정도
            factor = self.factor,
            min_lr = self.min_lr,
            verbose = True,
        )
    
    def __call__(self, val_loss):
        # 이전 오차와 비교했을 때 차이가 없다면 학습률을 업데이트
        self.lr_scheduler.step(val_loss)

# 조기 종료
class EarlyStopping():

    def __init__(self, patience=5, verbose=False, delta=0, path='../data/checkpoint.pt'):
        # 오차 개선이 없는 에포크를 몇 번 기다려줄지
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        # 검증 데이터셋에 대한 오차 최적화 값(오차가 가장 낮은 값)
        self.best_score = None
        # 조기 종료를 의미하며 초깃값은 False로 설정
        self.early_stop = False
        # np.Inf(infinity)는 넘파이에서 무한대를 표현
        self.val_loss_min = np.Inf
        # 개선이 없다고 생각하는 최소 변화량
        self.dalta = delta
        # 모델이 저장된 경로
        self.path = path

    # 에포크만큼 학습이 반복되면서 best_loss가 갱신되고, best_loss에 진전이 없으면 조기 종료가 하고 모델을 저장
    def __call__(self, val_loss, model):
        score = -val_loss
        # best_score 값이 존재하지 않으면 실행
        if self.best_score is None:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    # 검증 데이터셋에 대한 오차가 감소하면 모델을 저장
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        # 저장된 경로에 모델 저장
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 인수 값 지정
# pip install ipywidgets

# 인수 값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
# 조기 종료에 대한 인수
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
# 입력 받은 인수 값이 실제로 args 변수에 저장
args = vars(parser.parse_args())

# 사전 훈련된 모델의 파라미터 확인
print(f'Computation device: {device}\n')
model = models.resnet50().to(device)
# 총 파라미터 수
total_params = sum(p.numel() for p in model.parameters()) 
print(f'{total_params:,} total parameters.')
# 학습 가능한 파라미터 수
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters')

# 옵티마이저와 손실 함수 지정
Ir = 0.001
epochs = 100
optimizer = optim.Adam(model.parameters(), lr=Ir)
criterion = nn.CrossEntropyLoss()

# 오차, 정확도 및 모델 이름에 대한 문자열

# 오차 출력에 대한 문자열
loss_plot_name = 'loss'
# 정확도 출력에 대한 문자열
acc_plot_name = 'accuracy'
# 모델을 저장하기 위한 문자열
model_name = 'model'

# 오차, 정확도 및 모델의 이름에 대한 문자열
if args['lr_scheduler']:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)
    # 학습률 감소를 적용했을 때 오차에 대한 문자열
    loss_plot_name = 'lrs_loss'
    # 학습률 감소를 적용했을 때의 정확도에 대한 문자열
    acc_plot_name = 'lrs_accuracy'
    # 학습률 감소를 적용했을 때 모델에 대한 문자열
    model_name = 'lrs_model'
if args['early_stopping']:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping()
    loss_plot_name = 'es_loss'
    acc_plot_name = 'es_accuracy'
    model_name = 'es_model'

# 모델 학습 함수
def training(model, train_dataloader, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter, total = 0, 0
    # 훈련 과정을 시각적으로 표현
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size))

    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        train_running_correct += (pred == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    train_accuracy = 100 * train_running_correct / total
    return train_loss, train_accuracy

# 모델 성능 검증 함수
def validate(model, test_dataloader, val_dataset, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0.0
    counter, total = 0, 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset) / test_dataloader.batch_size))

    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            total += target.size(0)
            outputs = model(data)
            loss = criterion(outputs, target)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss / counter
        val_accuracy = 100 * val_running_correct / total
        return val_loss,  val_accuracy

# 모델 학습

# 훈련 데이터셋을 이용한 모델 학습 결과(오차, 정확도)를 저장하기 위한 변수(리스트 형태를 갖습니다.)
train_loss, train_accuracy = [], []
# 검증데이터셋을 이용한 모델 성능 결과(오차, 정확도)를 저장하기 위한 변수(리스트 형태를 갖습니다.)
val_loss, val_accuracy = [], []

start = time.time()
for epoch in range(epochs):
    print(f'Epoch {epoch+1} of {epochs}')
    train_epoch_loss, train_epoch_accuracy = training(model, train_dataloader, train_dataset, optimizer, criterion)
    val_epoch_loss, val_epoch_accuracy = validate(model, val_dataloader, val_dataset, criterion)

    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    if args['lr_scheduler']:
        lr_scheduler(val_epoch_loss)
    if args['early_stopping']:
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            break

    print(f'Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}')
    print(f'Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_accuracy:.2f}')
end = time.time()
print(f'Training time: {(end - start) / 60:.3f} minutes')

# 모델 학습 결과 출력
print('Saving loss and accuracy plots ..')
plt.figure(figsize=(10, 7))
# 데이터셋에 대한 정확도를 그래프로 출력
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'../data/img/{acc_plot_name}.png')
plt.show()
plt.figure(figsize=(10, 7))
# 데이터셋에 대한 오차를 그래프를 출력
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'../data/img/{loss_plot_name}.png')
plt.show()

print('Saving model...')
# 모델을 저장
torch.save(model.state_dict(), f'../data/img/{model_name}.pth')
print('Training COMPLETE')