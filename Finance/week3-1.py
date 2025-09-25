from pandas import Series
import numpy as np

def sep():
    print("="*60)

data = [10, 20, 30]
s = Series(data)
print(s)
sep()

data = np.arange(5)
s = Series(data)
print(s)
sep()

data = ["시가", "고가"]
s = Series(data)
print(s)
"""
    object 타입 브로드캐스팅 불가능
"""
sep()

"""
    타입 통일 불필요
"""
s = Series(['samsung', 81000])
print(s)
sep()

data = [1000, 2000, 3000]
s = Series(data)
print(f"s.index : {s.index}")
print(f"s.index.to_list() : {s.index.to_list()}")
sep()

# 데이터 입력 후 인덱스 지정
data = [1000, 2000, 3000] 
s = Series(data)
s.index = ["메로나", "구구콘", "하겐다즈"]
print(s)
sep()

# 데이터와 인덱스 입력
data = [1000, 2000, 3000]
index = ["메로나", "구구콘", "하겐다즈"]
s = Series(data, index)
print(s)
sep()

# 파라미터 명시적 입력 (keyword argument)
# 비슷한 데이터들을 입력할 때 명시적 입력이 오류 발생율을 낮춘다
data = [1000, 2000, 3000]
index = ["메로나", "구구콘", "하겐다즈"]
s = Series(index = index, data = data)
print(s)
sep()

k = [1000, 2000, 3000]
v = ["메로나", "구구콘", "하겐다즈"]
s = Series(data=v, index=k)
print(s)
sep()

"""
    s = Series(data, index)
    s = Series(data, index=index)
    s = Series(data=data, index=index)
    s = Series(index=index, data=data)
"""

# reindex : 인덱스 변경 새로운 인덱스의 값을 NaN으로 채움
data = [1000, 2000, 3000]
index = ['메로나', '구구콘', '하겐다즈']
s = Series(data=data, index=index)
print(f"s :\n{s}")
s2 = s.reindex(["메로나", "비비빅", "구구콘"])
print(f"s2 :\n{s2}")
sep()

price = [42500, 42550, 41800, 42550, 42650]
date = ["2019-05-31", "2019-05-30", "2019-05-29", "2019-05-28", "2019-05-27"]
s = Series(data = price, index = date)
print(s)
sep()

data = {
    "2019-05-31" : 42500,
    "2019-05-30" : 42550,
    "2019-05-29" : 41800,
    "2019-05-28" : 42550,
    "2019-05-27" : 42650
}
s = Series(data)
print(s)
sep()

print(s.index)
print(s.index.dtype)
sep()

print(s.values)
sep()

"""
    인덱싱 = 검색
    행 기준 : iloc
    인덱스 기준 : loc
"""
data = [1000, 2000, 3000]
s = Series(data=data)

print(f"s.iloc[0] : {s.iloc[0]}")
print(f"s.iloc[1] : {s.iloc[1]}")
print(f"s.iloc[2] : {s.iloc[2]}")
print(f"s.iloc[-1] : {s.iloc[-1]}")
sep()

print(s.loc[0])
print(s.loc[1])
print(s.loc[2])
# print(s.loc[-1])   # 에러
sep()

data = [1000, 2000, 3000]
index = ["메로나", "구구콘", "하겐다즈"]
s = Series(data=data, index=index)
 
print(f"s.iloc[0] : {s.iloc[0]}")
print(f"s.loc['메로나'] : {s.loc['메로나']}")
sep()

# iloc, loc 연산자 사용을 권장함
# print(s['메로나'])
# print(s[0])
# sep()

###############
data = [1000, 2000, 3000]
index = ["메로나", "구구콘", "하겐다즈"]
s = Series(data=data, index=index)

print(f"s.iloc[0:2] :\n{s.iloc[0:2]}")
sep()

print(f"s.loc['메로나':'구구콘'] :\n{s.loc['메로나':'구구콘']}")
sep()

"""
    시험 문제
    연속적이지 않은 인덱스 추출
    ":" 과 "," 의 차이
    [0:2] = [[0, 1, 2]]
"""
indice = [0, 2]
print("s.iloc[ indice ] :\n", s.iloc[ indice ])
print("s.iloc[ [0, 2] ] : \n", s.iloc[ [0, 2] ])
sep()

indice = ["메로나", "하겐다즈"]
print("s.loc[ indice ] :\n", s.loc[ indice ])
print(s.loc[ ["메로나", "하겐다즈"] ])
sep()

###### 수정, 추가, 삭제 ######
data = [1000, 2000, 3000]
index = ["메로나", "구구콘", "하겐다즈"]
s = Series(data=data, index=index)
s.loc['메로나'] = 500          # 값 수정
print(s)
sep()

s.iloc[0] = 500            # iloc 연산 사용
print(s)
sep()

# 없는 인덱스에 값을 할당하면 추가가 된다
s.loc['비비빅'] = 500          # 값 추가 
print(s)
sep()

# 기존 객체의 내용을 수정하지 않고 새로운 객체를 생성후 리턴한다
print("s.drop('메로나') :\n", s.drop('메로나'))
print(s)  # 기존 객체는 그대로
sep()

#### 시리즈 연산 ####
철수 = Series([10, 20, 30], index=['NAVER', 'SKT', 'KT'])
영희 = Series([10, 30, 20], index=['SKT', 'KT', 'NAVER'])
가족 = 철수 + 영희
print("가족\n", 가족)
sep()

# 스칼라 연산
print(철수 * 10)
sep()

# 브로드캐스팅
high = Series([42800, 42700, 42050, 42950, 43000])
low = Series([42150, 42150, 41300, 42150, 42350])

diff = high - low
print(diff)
sep()

print(diff.max())
print(max(diff))
sep()

date = ["6/1", "6/2", "6/3", "6/4", "6/5"]
high = Series([42800, 42700, 42050, 42950, 43000], index=date)
low = Series([42150, 42150, 41300, 42150, 42350] , index=date)
diff = high - low
print(diff)
sep()

# for문 사용 가능
max_idx = 0
max_val = 0

for i in range(len(diff)):
    if diff.iloc[i] > max_val:
        max_val = diff.iloc[i]
        max_idx = i

print(max_idx)
print(diff.index[max_idx])
sep()

print(diff.idxmax())
print(diff.idxmin())
sep()

# 수익률
date = ["6/1", "6/2", "6/3", "6/4", "6/5"]
high = Series([42800, 42700, 42050, 42950, 43000], index=date)
low = Series([42150, 42150, 41300, 42150, 42350] , index=date)
profit = ((high - low) / low) * 100
print(profit)
sep()

# 누적 수익률
print( profit.cumprod( ) )
sep()

print( profit.cumprod( ).iloc[ -1 ] )
sep()

# 고유값
data = {
    "삼성전자": "전기,전자",
    "LG전자": "전기,전자",
    "현대차": "운수장비",
    "NAVER": "서비스업",
    "카카오": "서비스업"
}
s = Series(data)

print(s.unique())
sep()

"""
    메소드의 기능 목적 정리 필요
"""

print(s.value_counts())
sep()

print(s.value_counts().index)
print(s.value_counts().values)
sep()

##### 문자열 숫자 변환 #####
s = Series(["1,234", "5,678", "9,876"])
# print( int(s) ) # 타입 에러

# , 제거
def remove_comma(x) :
    print(x, 'in function')
    return x

s = Series(["1,234", "5,678", "9,876"])
# map : 시리즈 객체에 함수 적용
result = s.map(remove_comma)    # 함수 이름만 입력
print(result)
sep()

def remove_comma(x) :
    # replace : 문자열 바꾸기
    return int(x.replace(",", ""))

s = Series(["1,234", "5,678", "9,876"])
result = s.map(remove_comma)
print(result)
sep()

def is_greater_than_5000(x):
    if x > 5000:
        return "크다"
    else:
        return "작다"

s = Series([1234, 5678, 9876])
s = s.map(is_greater_than_5000)
print(s)
sep()

##### 필터링 #####
data = [42500, 42550, 41800, 42550, 42650]
index = ['2019-05-31', '2019-05-30', '2019-05-29', '2019-05-28', '2019-05-27']
s = Series(data=data, index=index)
cond = s > 42000
print(cond)
sep()

print(s[cond])
sep()

close = [42500, 42550, 41800, 42550, 42650]     # 종가
open = [42600, 42200, 41850, 42550, 42500]      # 시가
index = ['2019-05-31', '2019-05-30', '2019-05-29', '2019-05-28', '2019-05-27']

open = Series(data=open, index=index)
close = Series(data=close, index=index)

cond = close > open
print(cond)
sep()

print(close[cond])
# print(close[close > open])
sep()

print(close.index[close > open])
# print(close[close > open].index)
sep()

close = [42500, 42550, 41800, 42550, 42650]
open = [42600, 42200, 41850, 42550, 42500]
index = ['2019-05-31', '2019-05-30', '2019-05-29', '2019-05-28', '2019-05-27']

open = Series(data=open, index=index)
close = Series(data=close, index=index)
diff = close - open
print(diff[close > open])
sep()

##### 정렬 #####
data = [3.1, 2.0, 10.1, 5.1]
index = ['000010', '000020', '000030', '000040']
s = Series(data=data, index=index)
print(s)

# 정렬 (오름차순)
s1 = s.sort_values()
print("오름차순:\n", s1)
sep()

# 정렬 (내림차순)
s2 = s.sort_values(ascending=False)
print("내림차순:\n", s2)
sep()

# 순위 매기기
data = [3.1, 2.0, 10.1, 3.1]
index = ['000010', '000020', '000030', '000040']
s = Series(data=data, index=index)
print(s.rank())
sep()

print(s.rank(ascending=False))