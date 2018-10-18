import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import json


abalone=pd.read_csv("abalone.txt",
            sep=",",
            names=['sex','length','diameter','height','whole_weight','shucked_weight', 'viscera_weight','shell_weight','rings'],
            header=None)
# print(abalone)
#
# print(pd.isnull(abalone)) # nan의 갯수 전체 위에서 false=0 True=1(null이다. 결측값이다)
#
# print(np.sum(pd.isnull(abalone)))
#
# print(abalone.describe()) # R언어에서는 써머리 함수(summary)

# # 그룹화
# # 성별로 몇개씩 있는지 궁금하다
# # 그룹화를 할때의 기준 1)무엇을 중심으로 할것인가
grouped=abalone['whole_weight'].groupby(abalone['sex']) # groupby(abalone['sex'] 어발론의 성별을 기준으로 전체 몸무게를 구분
# print(grouped) #thing의 대한 정보
# print(grouped.size()) # 그룹단위의 크기
# print(grouped.sum())# 그룹단위의 전체 무게의 합
# print(grouped.mean())# 그룹단위 '전체 무게'의 평균

# print(abalone.groupby(abalone['sex']).mean())  # 아발론에 대해서 성별로 구분을하고 그것에 대한 평균을 구하겠다.

abalone['length_cat']=np.where(abalone.length>np.median(abalone.length),'length_long' ,'length_short')
# (조건, 참, 거짓)
#'length_cat' 범주형 변수라고 할 수 있음 둘중에 하나 롱 아니면 숏 둘중 하나이기 때문

# print(abalone[['length','length_cat']][:10])

#sex(1차 그룹화 기준), length_cat(2차 그룹화기준)
# F L L :평균값(whole_weight)
#   L S :평균값                   이 출력이 됫으면 좋겠다.

#1
# print(abalone['whole_weight'].groupby([abalone['sex'],abalone['length_cat']]).mean())
#
# #2
# print(abalone.groupby(['sex','length_cat'])['whole_weight'].mean())
# 집계의 기준['sex','length_cat']을 써주고 맨뒤에는 집계하려는 대상['whole_weight']에 대해서 적어줌


#그룹단위로 반복적인거 하는 작업
#성별로 그룹화 한 다음, for문 사용하여 그룹 이름별로 데이터 셋을 출력한다.

# for sex,group_data in abalone[['sex','length_cat','whole_weight','rings']].groupby('sex'):
#     print(sex) # 그룹화를 했을때 속성에 대해서 들어감
#     print(group_data[:5]) # 성별로 그룹화를 시킨 각각의 데이터 들이 들어감

# for (sex,length_cat),group_data in abalone[['sex','length_cat','whole_weight','rings']].groupby(['sex','length_cat']):
#     print(sex,length_cat) # 그룹화를 했을때 속성에 대해서 들어감
#     print(group_data[:5]) # 성별로 그룹화를 시킨 각각의 데이터 들이 들어감

#6개의 그룹으로 출력이 됨 sex :3 , length_cat :2 니까 3*2해서 6그룹이 나온다.

df=DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index=['a','b','c','d'],columns=['one','two'])
# print(df)
# print(df.sum())
# print(df.mean(axis=1,skipna=False)) # ,skipna Nan을 스킵 할거냐? 안할거냐 하니까 nan는 사칙연산 사용 불가능 함으로 그냥 nan으로 나옴
# print(df)
# print(df.idxmax()) # 컬럼 단위로 최대값을 찾는데 출력 되는건 최대값의 인덱스가 출력됨. nan은 재외하고

# 1. 데이터가 제이슨으로 되어있다
#2. 디스크랩션 안에 음식들이 있고 음식에 해당되는 고유 id와 영양소가 있다.

db=json.load(open('database.json'))
# print(len(db)) #0번~6635번


# 제이슨 데이터란? {키:밸루,키:밸류 },{키:{키,밸류}        },{        } 이렇게 중괄호의 쌍이 6636개 있다는것.
# 제이슨의 구조를 봤을때 키와 밸류로 이뤄져있다.  딕셔너리 안에 또 딕셔너리 구조로도 가능함.
# 데이터가 csv로 나오면 읽기 편하지만 json타입은 딕셔너리구조기때문에 내가 한번 보고 해석한 다음 내가 원하는 값을 빼내야함.
#제이슨은 대신 가벼워서 현재 많이 쓰임.
# print(db[0]['nutrients'])
# print(db[0]['nutrients'][0])
# print(db[0]['nutrients'][0]['value'])
nutrients=DataFrame(db[0]['nutrients']) # 데이터 프레임 안에 딕셔너리 넣어서 만들 수 있음
# print(nutrients[:7])
info_keys=['description',
           'group',
           'id',
           'manufacturer']
info=DataFrame(db,columns=info_keys)
# print(info.info())
#                대한민국(object)
#출생시도 : 서울, 경기, 강원.... 제주
#              단군할아버지(object)
#출생시도 : 나,너,저사람.... 이사람
# 계층구조상에서 맨 꼭대기에 있는게 object
# print(info['group'])
# print(pd.value_counts(info.group)) # 값들 마다 몇개잇는지, 항목들에 대한 빈도수를 찾는것.

nutrients=[]
for rec in db:
    fnuts=DataFrame(rec['nutrients'])
    fnuts['id']=rec['id']
    nutrients.append(fnuts)
# print(nutrients)
print(nutrients.duplicated().sum()) # duplicated() 중복된게 몇개가 있는지 갯수를 셈
# nutrients=nutrients.drop_duplicates() # 중복된 것을 제거함.
# print(nutrients)