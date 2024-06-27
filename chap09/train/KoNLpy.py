# 라이브러리 호출 및 데이터셋 준비

import csv
from konlpy.tag import Okt
# scipy 1.13.1에서는  triu를 더 이상 사용할 수 없어서 scipy를 1.10.1으로 다운그레이드해서 사용해야 한다.
from gensim.models import word2vec

f = open(r'..\data\ratings_train.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
rdw = list(rdr)
f.close()

# 오픈 소스 한글 형태소 분석기 호출 
twitter = Okt()
result = []
for line in rdw:
    # 형태소 분석
    malist = twitter.pos(line[1], norm=True, stem=True)
    r = []
    for word in malist:
        # 조사 어미, 문장 부호는 제외하고 처리
        if not word[1] in ['Josa', 'Eomi', 'Punctuation']:
            r.append(word[0])
    # 형태소 사이에 ' '(공백)을 넣고, 양쪽 공백은 삭제
    rl = (' '.join(r)).strip()
    result.append(rl)
    print(rl)

# 형태소 저장
with open('NaverMovie.nlp', 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(result))

# Word2Vec 모델 생성
mData = word2vec.LineSentence('NaverMovie.nlp')
mModel = word2vec.Word2Vec(mData, vector_size=200, window=10, hs=1, min_count=2, sg=1)
# 모델 저장
mModel.save('NaverMovie.nlp')