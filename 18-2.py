#정규화 : 정규화는 최소값, 최대값을 이용함
#정규화 : 최소값,최대값이용, 범위는 :0~1
#딥러닝(인공신경망) 머신러닝에서 많이함. 구현도 직접해보는게 좋다


import  numpy as np
import  pandas as pd
from sklearn.preprocessing import MinMaxScaler,minmax_scale, Binarizer, binarize

X=np.array([[10.,-10.,1.],
          [5.,0.,2.],
          [0.,10.,3.]]) # .이붙은건 타입이 float이다
#함수 중에 x가 array니까 최대값 x.max()
# 가령 중간고사 기말고사 순서대로0 등수로 되어있음. 그럼 중간고사 가지고 정규화 등수에 대한 정규화 기말고사에 대한 정규화를 해야함

#1
print((X-X.min(axis=0)) /(X.max(axis=0)-X.min(axis=0))) #데이터값 - 최소값 /열단위의 최대값 - 최소값을 분모에 둠
# 정규화 끝! 이것만 구현하면 됨,

#2
mms=MinMaxScaler()
xmms=mms.fit_transform(X)
print(xmms)

#3
minmax_scale(X,axis=0,copy=True) # 데이터를 복사한다는 것 원본 속상 없음 copy=false는 원본 변함.
print(xmms)

# 이진화 (0,1) : 연속형 변수값 -> 0또는 1로 변환
# 임계값(threshold) :변환 기준값
# 당뇨병 유/무

#0~1 = > 0.75 //0.5,0.5
# 0.8 이상 0또는 1  이진화

X=np.array([[10.,-10.,1.],
          [5.,0.,2.],
          [0.,10.,3.]])

binarizer=Binarizer().fit(X)
print(binarizer)
print(binarizer.transform(X))


binarizer=Binarizer(threshold=2.0)
print(print(binarizer.transform(X)))

print(binarize(X,threshold=2.0,copy=False))
print(X) # 이진화 0과 1로 구분함.


#범주형 변수 -> 이진화
#성별 : 남(0), 여(1) 로 인코딩 하겠다.
#연령 : 20대(0), 30대(1), 40대(2)로 인코딩 하겠다.
#성적: A(0),B(1),.....F(4)로 인코딩 하겠다.

# SN(student number) 성별 연령대 성적
# 1 0 0 1  원핫인코딩
# 2 1 3 0     =>
# ...

#원 핫 인코딩
# 성별(0,1) => 0 :10, 1 :01
# 연령대(0~3)=>0:1000, 1:0100, 2:0010, 3:0001
# 성적(0~5)=> 0:100000, 1:0100000, 2:001000, ...
# 그 경우에대해서 이렇게 바꿔서 생각
# 지역(서울:0, 부산:1, 강원:2, 대전:3....) #숫자 사이에 관계가 없음
# 그러나 원핫 인코딩 하면 관계가 있는것으로 바뀜. 그래서 원핫 인코딩을 먼저 한 다음에 써야함.

# 번호판 판별기
# 1)원핫 인코딩
# 52가 1234
# 5:0000010000
# 2:0010000000
# 가
# 1
# 2
# 3
# 4
# 각각의 이미지를 분해해서 판별
# 그다음 원 핫 인코딩을 한다.
# 이 원핫인코딩 된 수치를 모델에 입력
# 2) 코드를 판별기에 입력(5)
# 3) 판별기는 판별결과를
# 0 0 0 0 0 0.9 0.05 0 0 0
#0000010000 => 5


# 연속성 값을 할때는 원핫 인코딩을 할 필요가 없지만 분류를 해야할 때는 원핫인코딩이 필요하다.

from sklearn.preprocessing import OneHotEncoder
data_train=np.array(
#성별(2) 연령대(3) 성적(5) 총 열가지가 잇음.
[[0,0,0],
 [0,1,1],
 [0,2,2],
 [1,0,3],
 [1,1,4]])
enc=OneHotEncoder() # 객체 #thing 어떤것
print(enc.fit(data_train)) # fit을하면 각각의 데이터가 어떻게 구성되는지 봄.
print(enc.active_features_)
#[0 1 2 3 4 5 6 7 8 9] # 데이터 종류는 10가지가 존재한다.는 수치
# 남 여 20대,30,40대, 성적 해당(ABCDF)
print(enc.n_values_)#성별 번주 2개 연령대 범주 3개 등급 범주 5개
print(enc.feature_indices_) #print(enc.feature_indices_) # 성별 -0이상 2미만, 연령대는 2이상 5미만 성적은 5이상 10미만
# 이범 위를 알려주는 것.

print("="*50)
#여성(1), 40대(2), 등급:D(3)
data_new=np.array([[1,2,3]])
print(enc.transform(data_new).toarray()) # 원핫 인코딩


# 그룹 바이 함수 적용법
# aaaaaaaaabbbbbbbccccc 를 a,b,c로 나눔 스프릿(split) ! 나누기 작업
# 이 나눈 과정을 처리한다음 a처리,b처리,c처리 즉 그룹으로 나눈 값을 하나의 테이블로 다시 합쳐서(combine) 작업함.
