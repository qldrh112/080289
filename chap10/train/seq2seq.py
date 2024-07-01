# 라이브러리 링크
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import re
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 준비
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

# 딕셔너리를 만들기 위한 클래스
class Lang:
    # 단어의 인덱스를 저장하기 위한 컨테이너를 초기화
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        # SOS(Start Of Sequence): 문장의 시작
        # EOS(End Of Sequence): 문장의 끝
        self.index2word = {0: 'SOS', 1: 'EOS'}
        # SOS와 EOS에 대한 카운트
        self.n_words = 2
    
    # 문장을 단어 단위로 분리한 후, 컨테이너(word)에 추가
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    # 컨테이너에 단어가 없다면 추가되고, 있다면 카운트를 업데이트
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# 데이터 정규화
def normalizeString(df, lang):
    # 소문자로 전환
    sentence = df[lang].str.lower()
    # \s = [^ \t\n\r\f\v], 공백 문자가 아닌 것을 모두 공백으로 바꾸시오
    sentence = sentence.str.replace('[^A-Za-z\s]+', ' ')
    # 유니코드 정규화 방식
    sentence = sentence.str.normalize('NFD')
    # 유니코드를 아스키로 전환, 아스키 범위 밖의 문자는 무시됨
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    return sentence

def read_sentence(df, lang1, lang2):
    # 데이터셋의 첫 번째 열(영어)
    sentence1 = normalizeString(df, lang1)
    # 데이터셋의 두 번째 열(프랑스어)
    sentence2 = normalizeString(df, lang2)
    return sentence1, sentence2

def read_file(loc, lang1, lang2):
    df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang1, lang2])
    return df

def process_data(lang1, lang2):
    # 데이터셋 불러오기
    df = read_file(f'../data/{lang1}-{lang2}.txt', lang1, lang2)
    sentence1, sentence2 = read_sentence(df, lang1, lang2)

    input_lang, output_lang = Lang(), Lang()
    pairs = []
    for i in range(len(df)):
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            # 첫 번째와 두 번째 열을 합쳐서 저장
            full = [sentence1[i], sentence2[i]]
            # 입력을 영어로 사용
            input_lang.addSentence(sentence1[i])
            output_lang.addSentence(sentence2[i])
            pairs.append(full)
    return input_lang, output_lang, pairs

# 텐서로 변환
def indexesFromSentence(lang, sentence):
    """문장을 단어로 분리하고 인덱스를 반환"""
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    """딕셔너리에서 단어에 대한 인덱스를 가져오고 문장 끝에 토큰을 추가"""
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair):
    """입력과 출력 문장을 텐서로 변환하여 반환"""
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# 인코더 네트워크
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()
        # 인코더에서 사용할 입력층
        self.input_dim = input_dim
        # 인코더에서 사용할 임베딩 계층
        self.embedded_dim = embbed_dim
        # 인코더에서 사용할 은닉층(이전 은닉층)
        self.hidden_dim = hidden_dim
        # 인코더에서 사용할 GRU의 계층 개수
        self.num_layers = num_layers
        # 임베딩 계층 초기화
        self.embedding = nn.Embedding(input_dim, self.embedded_dim)
        # 임베딩 차원, 은닉층 차원, GRU의 계층 개수를 이용하여 GRU 계층을 초기화
        self.gru = nn.GRU(self.embedded_dim, self.hidden_dim, num_layers=self.num_layers)

    def forward(self, src):
        # 임베딩 처리, src는 self.embedding의 입력으로 사용
        embedded = self.embedding(src).view(1, 1, -1)
        # 임베딩 결과를 GRU 모델에 적용
        outputs, hidden = self.gru(embedded)
        return outputs, hidden
    
# 디코더 네트워크
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # 임베딩 계층 초기화
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        # GRU 계층 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        # 선형 계층 초기화
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 입력을 (1, 배치 크기)로 변경
        input = input.view(1, -1)
        embbeded = F.relu(self.embedding(input))
        output, hidden = self.gru(embbeded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden
    
# seq2seq 네트워크
class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()

        # 인코더 초기화
        self.encoder = encoder
        # 디코더 초기화
        self.decoder = decoder
        self.device = device
    
    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):
        # 입력 문자 길이(문장의 단어 수)
        input_length = input_lang.size(0)
        batch_size = output_lang.shape[1]
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        for i in range(input_length):
            # 문장의 모든 단어를 인코딩
            encoder_output, encoder_hidden = self.encoder(input_lang[i])
        # 인코더의 은닉층을 디코더의 은닉층으로 사용
        decoder_hidden = encoder_hidden.to(device)
        # 첫 번째 예측 단어 앞에 토큰(SOS) 추가
        decoder_input = torch.tensor([SOS_token], device=device)

        # 현재 단어에서 출력 단어를 예측
        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            # topv: 가장 높은 확률 값, topi: 그 값의 인덱스
            topv, topi = decoder_output.topk(1)
            # teacher_force를 활성화하면 목표 단어를 다음 입력으로 사용
            input = (output_lang[t] if teacher_force else topi)
            # teacher_force를 활성화하지 않고, 디코더의 예측이 EOS 토큰인 경우 자체 예측 값을 다음 입력으로 사용
            if (teacher_force == False and input.item() == EOS_token):
                break
        return outputs

def trainModel(model, input_lang, output_lang, pairs, num_iteration=20000):
    model.train()
    # 옵티마이저로 SGD 사용
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(num_iteration)]

    for iter in range(1, num_iteration + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # 옵티마이저 초기화
        optimizer.zero_grad()

        # 모델의 예측 수행
        output = model(input_tensor, target_tensor)
        
        # 손실 계산
        loss = 0
        for di in range(target_tensor.size(0)):
            loss += criterion(output[di], target_tensor[di])
        
        # 역전파
        loss.backward()

        # 옵티마이저 업데이트
        optimizer.step()

        total_loss_iterations += loss.item()

        # 5,000번째마다 오차 값을 출력
        if iter % 5000 == 0:
            average_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print(f'{iter} {average_loss:.4f}%')
    
    torch.save(model.state_dict(), '../data/mytraining.pt')
    return model

# 모델 평가
def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        # 입력 문자열을 텐서로 변환
        input_tensor = tensorFromSentence(input_lang, sentences[0])
        # 출력 문자열을 텐서로 변환
        output_tensor = tensorFromSentence(output_lang, sentences[1])
        decoded_words = []
        output = model(input_tensor, output_tensor)
        
        for ot in range(output.size(0)):
            # 각 출력에서 가장 높은 값을 찾아 인덱스를 반환
            topv, topi = output[ot].topk(1)
            if topi[0].item() == EOS_token:
                # EOS 토큰을 만나면 평가를 멈춥니다.
                decoded_words.append('<EOS>')
                break
            else:
                # 예측 결과를 출력 문자열에 추가
                decoded_words.append(output_lang.index2word[topi[0].item()])
    return decoded_words

#  훈련 데이터셋으로부터 임의의 문장을 가져와서 모델 평가
def evaluateRandomly(model, input_lang, output_lang, pairs, n=10):
    for i in range(n):
        # 임의로 문장을 가져옵니다.
        pair = random.choice(pairs)
        print(f'input {pair[0]}')
        print(f'output {pair[1]}')
        # 모델 평가 결과는 output_words에 저장
        output_words = evaluate(model, input_lang, output_lang, pair)
        output_sentence = ' '.join(output_words)
        print(f'predicted {output_sentence}')

# 모델 훈련
# 입력으로 사용할 언어
lang1 = 'eng'
# 출력으로 사용할 언어
lang2 = 'fra'
input_lang, output_lang, pairs = process_data(lang1, lang2)

randomize = random.choice(pairs)
print(f'random sentence {randomize}')

input_size = input_lang.n_words
output_size = output_lang.n_words
# 입력과 출력 단어 수 출력
print(f'Input: {input_size} Output: {output_size}')

embed_size = 256
hidden_size = 512
num_layers = 1
# 75,000번 반복하여 모델 훈련
num_iteration = 75000

# 인코더에 훈련 데이터셋을 입력하고 모든 출력과 은닉 상태를 저장
encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
decoder = Decoder(output_size, hidden_size, embed_size, num_layers)

# 인코더-디코더 모델(seq2seq) 객체 생성
model = Seq2Seq(encoder, decoder, device).to(device)

print(encoder)
print(decoder)

# 모델 학습
model = trainModel(model, input_lang, output_lang, pairs, num_iteration)

# 임의의 문장에 대한 평가 결과
evaluateRandomly(model, input_lang, output_lang, pairs)