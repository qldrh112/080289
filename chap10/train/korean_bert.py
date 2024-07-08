# 라이브러리 불러오기
import torch
from transformers import BertTokenizer, BertModel
# 한국어를 위한 버트 토크나이저를 내려받습니다.
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 문장의 토크나이징
text = '나는 파이토치를 이용한 딥러닝을 학습중이다.'
# 문장의 시작은 [CLS], 문장의 끝은 [SEP]로 형태를 맞춤
marked_text = '[CLS] ' + text + " [SEP]"
# 사전 훈련된 버트 토크나이저를 이용해서 문장을 단어로 쪼갭니다.
tokenized_text = tokenizer.tokenize(marked_text)
print(tokenized_text)

# 모델을 훈련시킬 텍스트 정의
text = '과수원에 사과가 많았다.'\
        '친구가 나에게 사과했다.'\
        '백설공주는 독이 든 사과를 먹었다.'
# 생성된 문장의 앞에는 [CLS]를 뒤에는 [SEP]를 추가
marked_text = '[CLS] ' + text + ' [SEP]'
# 문장을 토큰으로 분리
tokenized_text = tokenizer.tokenize(marked_text)
# 토크 문자열에 인덱스를 매핑
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
for tup in zip(tokenized_text, indexed_tokens):
    print(f'{tup[0]:<12} {tup[1]:>6,}')

segments_ids = [1] * len(tokenized_text)
print(segments_ids)

# 데이터를 텐서로 변환
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# 모델 생성
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
model.eval()

# 모델 훈련
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    # 네트워크 은닉 상태를 가져옵니다.
    hidden_states = outputs[2]

    # 모델의 은닉 상태 정보 확인
    print('계층 개수:', len(hidden_states), ' (initial embeddings + 12 BERT layers)')
    layer_i = 0
    print('배치 개수:', len(hidden_states[layer_i]))
    batch_i = 0
    print('토큰 개수:', len(hidden_states[layer_i][batch_i]))
    token_i = 0
    print('은닉층의 유닛 개수:', len(hidden_states[layer_i][batch_i][token_i]))

    # 모델의 은닉 상태 정보 확인
    print('은닉 상태의 유형:', type(hidden_states))
    print('각 계층에서의 텐서 형태:', hidden_states[0].size())

    # 텐서의 형태 변경
    # 각 계층의 텐서 결합은 stack을 사용
    token_embeddings = torch.stack(hidden_states, dim=0)
    # 최종 텐서의 형태를 출력
    token_embeddings.size()

    # 텐서의 형태 변경
    # 배치 차원(1) 제거
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # 배치 차원 제거 후 최종 텐서의 형태를 출력
    token_embeddings.size()

    # 텐서 차원 변경
    token_embeddings = token_embeddings.permute(1, 0, 2)
    token_embeddings.size()

    # 각 단어에 대한 벡터 형태 확인
    # 형태가 [33 x (33 x 768)]인 벡터를 [33 x 25,344]로 변경하여 저장
    token_vecs_cat = []
    # token_embeddings는 [33 x 12 x 768] 형태의 텐서를 갖습니다.
    for token in token_embeddings:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_vecs_cat.append(cat_vec)
    print(f'형태는: {len(token_vecs_cat)} x {len(token_vecs_cat[0])}')


    # 계층을 결합하여 최종 단어 벡터 생성
    # [33 x 768] 형태의 토큰을 벡터로 저장
    token_vecs_sum = []
    # 'token_embeddings'는 [33 x 12 x 768] 형태의 토큰을 갖습니다.
    for token in token_embeddings:
        # 마지막 4개 계층의 벡터를 합산
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
        # 'sum_vec'를 사용하여 토큰을 표현
    print(f'형태는: {len(token_vecs_sum)} x {len(token_vecs_sum[0])}')

    # 문장 벡터
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    print('최종 임베딩 벡터의 형태:', sentence_embedding.size())
    
    # 토큰과 인덱스 출력
    for i, token_str in enumerate(tokenized_text):
        print(i, token_str)

    # 단어 벡터 확인
    print('사과가 많았다', str(token_vecs_sum[6][:5]))
    print('나에게 사과했다', str(token_vecs_sum[10][:5]))
    print('사과를 먹었다', str(token_vecs_sum[19][:5]))

    # 코사인 유사도 계산
    from scipy.spatial.distance import cosine

    # '사과가 많았다'와 '나에게 사과했다'에서 단어 '사과' 사이의 코사인 유사도를 계산
    diff_apple = 1 - cosine(token_vecs_sum[5], token_vecs_sum[27])
    # '사과가 많았다'와 '사과를 먹었다'에 있는 '사과' 사이의 코사인 유사도를 계산
    same_apple = 1 - cosine(token_vecs_sum[5], token_vecs_sum[16])
    print(f'*유사한* 의미에 있는 벡터 유사성: {same_apple:.2f}')
    print(f'*다른* 의미에 있는 벡터 유사성: {diff_apple:.2f}')