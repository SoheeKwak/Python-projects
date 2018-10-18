# a a a b b c c => a b c (유니크한 값만 뽑아낼때 )
#
# a a a b b c c => a-3 b-2 c-2 (중복값의 개수 )
# unique(), value_counts():

import pandas as pd
import numpy as np
from  pandas import DataFrame,Series
import matplotlib.pyplot as plt

df=DataFrame({'a':['a1','a1','a2','a2','a3','a3'],
        'b':['b1','b1','b1','b1','b2',np.nan],
        'c':[1,1,3,4,4,4]})

print(df['a'].unique())# unique는 데이터 프레임에서 나오는 것으로 유일한 값들만 나옴 하나씩
#유일한 값을 얻어내고자 할때 쓰는 함수
print(df['b'].unique())
print(df['c'].unique())

print(df['a'].value_counts())
print(df['b'].value_counts())
print(df['c'].value_counts())

print(df['c'].value_counts(normalize=True))
#normalize=True # 각각의 수가 몇개있는지에 대한 비율로 표현.
print(df['c'].value_counts(sort=True,ascending=False)) # 디폴트 설정. 기본값
#(sort=True,ascending=False))sort가 오름차순 정렬인데 어센딩이 펄스니까 내림차순으로 됨.
print(df['c'].value_counts(sort=True,ascending=True))
print(df['c'].value_counts(sort=False)) # 정렬을 하지 않겠다. 인덱스 순번대로 지정됨.
print("="*50)
print(df['b'])
print(df['b'].value_counts(dropna=True)) # 디폴트값 그래서 nan은 나오지 않게 됨.
print(df['b'].value_counts(dropna=False)) # nan이 몇개가 나오는지 알려줌 dropna가 false이기 때문

#beens 속성을 통해 그룹을 지어서
print("="*50)
print(df['c'].value_counts())

print(df['c'].value_counts(bins=[0,1,2,3,4,5],sort=False)) # 정렬하지 않는것이니까 인덱스 번호로 지정
#구간으로 나누는것. bins 구간으로 나누어서 그 데이터 값이 몇개가 있는지 세어주는 함수
#0부터 1까지 1부터 2까지 3부터 4까지 4부터 5까지 몇개인가 이런식으로 생각. 1.0<x <=2.0 이렇게 생각
#bins는 막대그래프 구간을 통해 시각화할때 자주쓰여짐.

#표준화 : 변수들간의 scale(척도)이 서로 다른 경우,
#직접적 상호간 비교를 할 수 없음
#기계학습 모델링하는 과정에서 문제가 발생, 모델을 만들었다 하더라도 정확한 예측을 기대하기 어려움
#모델링 하기 앞서 변수들 간 척도가 다른 경우 > 표준화 (표준화를 하고 모델링을 해야함)
# 평균:0, 표준편차:1 인 표준 정규분포로 표준화

# 표준화로 만드는데에는 함수도 많고 내가 직접 구현을 해도 된다.
# 표준화 하는 이유! 데이터들간의 척도가 달라서 비교를 할 수 없기 때문에 표준화를 해서 비교가 가능하도록 한다.
#모델링이 하기에 앞서서 표준화를 해줘야 한다.

# 표준화 함수들
#1
from numpy import * # 넘파이에 있는 모든 함수 다 가지고와라
data=np.random.randint(30,size=(6,5))
print(data)
print(np.mean(data)) # 각 평균
print(np.mean(data,  axis=0)) # 각각의 컬럼 단위로 평균이 구해짐(행)
print(np.mean(data,  axis=1)) # 행단위로 구해짐(열 )

print((data-mean(data,axis=0))/ std(data,axis=0)) # (데이터 - 평균) / 표준편차

#2
import scipy.stats as ss
data_standardized_ss=ss.zscore(data)
print(data_standardized_ss) #print((data-mean(data,axis=0))/ std(data,axis=0)) 와 동일하게 나옴

#3
from sklearn.preprocessing import StandardScaler
ds=StandardScaler().fit_transform(data) #print((data-mean(data,axis=0))/ std(data,axis=0)) 와 동일하게 나옴
print(ds)

#척도가 다른 거를 표준정규분포로 변환해서 정규화를 시킴. 표준정규분포란 표준편차가 1이고 평균이 0인것.

#
#X~N(0,1) 표준화
# 정규분포,이상치(특이값,outlier)가 없다고 가정을 하고 만든것. 이 조건이 만족해야 표준화를 정상적으로 할 수 있음
#Z=(x-mean)/std
#이상치가 영향을 많이주는것은 mean 평균!
#1번째 방법 : 이상치 제거, 특이값제거를 해서 평균을 구함
#2번째 방법 : 특이값에 대해 민감하지 않는 얘를 찾아서함. 중앙값같은것
#IQR(InterQuantilRange) : 3 사분위 수(75%) - 1사분위수(25%)
#이상치가 섞여있는 데이터를 표준화하는 벙법
#1)이상치를 제거한 후 표준화 수행 - 표준화 된 결과를 통해 분석,모델링을 함.
#2) 중앙값, IQR은 이상치에 영향을 받지 않으므로 이것을 사용해서 표준화한다.


from sklearn.preprocessing import StandardScaler,RobustScaler
np.random.seed(777)
mu,sigma=10,2 # 평균, #표준편차
x=mu+sigma*np.random.randn(100) #표준 정규분포를 따르는 난수 100
#
print(x)
# plt.hist(x)
# plt.show()

print(np.mean(x))
print(np.std(x))

x[98:100]=100 # 맨마지막 두개 !
print(x)

print(np.mean(x))
print(np.std(x))
# plt.hist(x,bins=np.arange(0,102,2)) # 구간을 나눠주기 위해 필요
#각 구간에 빈도가 얼마나 되느냐. 구분해 나갈때는 구간에 대해서 진행
#시각화는 이상치가 있는지 없는지 볼때도 사용함
# plt.show()
#
x=x.reshape(-1,1) # 뒤에 지정할때 1 에 지정할때 나눠어 떨어져야함.  행은 4개인데 -1은 니가 알아서 계산해.
x=x.reshape(-1,1) # 내가 마음대로 지정하고 니가해.
print(x.shape)
print(x[0:10])
# reshape을 할때 -1을 줘서도 가능함.


#표준화해서 출력 정규화 아님  표준편차 1 평균이 0인 표준정규분포를 따르는 정규화
xss=StandardScaler().fit_transform(x)
# [ 6.92272070e+00]
#  [ 6.92272070e+00]]
# 임의적으로 넣어둔 100이 굉장히 많이 벗어나 있음 보통은 -1,-3w저정
print(xss)
#
# plt.hist(xss)
# plt.show()

xss_in=xss[xss<5]
print(xss_in)
plt.hist(xss_in,bins=np.arange(-3.0,3.0,0.2))
plt.show()
#이상치를 제외했을 경우 이상치에 벗어난 값들은 범위가 촘촘 하고 서로 가까이 붙어있을것.
#표준화를 한 과정에서 이상치를 제거하지 않은 상태에서 이상치 값들의 영향. 표준화 값이 표준편차가 5보다 큰 얘들은 빼고
#5보다 작은 얘들에 대해서 시각화를 진행.

#
print(median(x)) #  중앙값 출력 9.91 평균은 10
print(mean(x)) # 평균은 11.79
print(x) # x에는 이상치가 있는 상태
Q1=percentile(x,25,axis=0) # 분위 수 구할때 씀 1사분위수 # 25%지점인 1사분위수
Q3=percentile(x,75,axis=0) # 3사분위수
print("1사분위수",Q1) # 8.77
print("3사분위수",Q3) #11.39
IQR=Q3-Q1 # 3사분위수 - 1사분위수
print(IQR)
#옆에 선을 지나가면 이상치로 봄. 이 지점을 벗어나면 이상치 이걸 제거를 해야될 대상. box plot 박스플롯
#Q3-Q1  은 차이가 50% 그러므로 중앙값/


# 이상치가 포함된 데이터의 중앙값과 iqr을 이용해서 표준화
x_rs=RobustScaler().fit_transform(x)
print(x_rs[-10:])
print(np.median(x_rs)) # 중앙값을 출력 # 중위수가 0이 되도록 출력하는게  RobustScaler() 함수이다
print(np.mean(x_rs))

x_rs_in=x_rs[x_rs<5]
print(np.std(x_rs_in, bins=np.arange(-3,3,0.2)))


# 파란색 히스토그램은 평균과 표준편차를 이용한 결과에 대해서 아웃라이어 부분을 빼서 진행함 5보다 작은것에 대해서만한거
# 100두개 영향을 받음
# 오렌지 히스토그램은 표준화를 중앙값과 iar을 사용해서 사용. 이상치에 대해서 영향을 안받음.
#y값 예측하고 싶을때 산포도는 주항색이 더 큼. 산포도가 더 넓게 퍼져있는것. 주황색 x축이 더 유효하다. 데이터가 널리 퍼져있으니까
# standard scaler 보다 Robustscaler가 더 좋은 결과를 나타냄. 더 유용한 함수다.


#위에는 다 표준화 변수들의 척도가 다를때 상호간에 비교를 위해서 쓰는것. (표준정규분포 형태로의 표준화 작업을함)
# 정규화!