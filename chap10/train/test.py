import torch
import torch.nn as nn

# 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 및 데이터 이동
model = SimpleModel().to(device)
x = torch.randn(10, 10).to(device)

# 모델과 데이터의 디바이스 확인
print(f"Model is on: {next(model.parameters()).device}")
print(f"Tensor is on: {x.device}")

# GPU 메모리 사용량 확인 (GPU 사용 시)
if torch.cuda.is_available():
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated()} bytes")
    print(f"Cached GPU memory: {torch.cuda.memory_reserved()} bytes")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
