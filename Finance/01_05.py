# %% [markdown]
# # Numpy

# %% [markdown]
# ## ndarray

# %%
# numpy 라이브러리 임포트
import numpy as np

# %%
data = [1, 2, 3, 4]
arr = np.array(data)
print("data :", data)
print("arr :", arr)
print("type(arr) :", type(arr))

# %%
data1 = [1, 2, 3, 4]
arr1 = np.array(data1)
print("data1 :", data1)
print("arr1 :", arr1)
print("type(arr1) :", type(arr1))

# %% [markdown]
# * python List 보다 Numpy ndarray가 속도가 빠르다
# * c로 구현된 Numpy가 더 빠르다

# %% [markdown]
# ### 2차원 array

# %%
# 2차원 리스트
data2d = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 2차원 리스트는 행단위 인덱싱만 가능
print(data2d[0])

# %%
arr = np.array(data2d)
print("1번째 열 :", arr[:, 0])    # [행, 열]로 열단위 인덱싱 가능

# %% [markdown]
# ### numpy 함수

# %% [markdown]
# #### ndim, shape, dtype

# %%
data2 = [
    [1, 2],
    [3, 4]
]
arr2 = np.array(data2)

print("arr2 :", arr2, sep='\n')
print("type(arr2) :", type(arr2))
print("type(arr2[0]) :", type(arr2[0]))
print("type(arr2[0, 0]) :", type(arr2[0, 0]))

# %%
print("arr2.shape :", arr2.shape)   # (행, 열)
print("arr2.ndim :", arr2.ndim)    # 차원
print(arr2.dtype)   # 원소 타입

# %%
print("arr1.shape :", arr1.shape)   # (4,) = (1, 4)
# 1차원 배열은 (열,) 형태로 출력됨
print("arr1.ndim :", arr1.ndim)    # 차원

# %%
data = [
    [1],
    [2],
    [3],
    [4]
]
c = np.array(data)
print(c.shape)
print(c.ndim)

# %% [markdown]
# #### ones, zeros, arange

# %% [markdown]
# * zeros, ones : 결측치가 있는 데이터를 불러오기 전 초기화할 떄 사용

# %%
print(np.zeros(3))
print(np.ones(3))

# %%
size = (3, 4)
print(np.zeros(size))

# %%
print(np.arange(5))         # [0, 5)
print(np.arange(1, 5))      # [1, 5)
print(np.arange(1, 20, 2))  # [1, 20) d = 2

# %% [markdown]
# #### reshape

# %%
# 1행 6열 배열
ndarr1 = np.arange(6)           # (6,)
print(ndarr1)

# 2행 3열 배열로 변환
ndarr2 = ndarr1.reshape(2, 3)   # (2, 3)
print(ndarr2)

# %% [markdown]
# * boolean = 논리형
# * int = 정수형
# * float = 실수형
# * string = 문자형
# * 메모리에서 데이터가 차지하는 사이즈를 줄이기 위해 존재 = 최적화를 위해

# %% [markdown]
# ### 인덱싱

# %%
arr = np.arange(4)
print("arr :", arr)
print("arr[0] :", arr[0])

# %%
arr = np.arange(4).reshape(2, 2)
print("np.arange(4).reshape(2, 2) :\n", arr)
print(arr[0])
print(arr.reshape(-1, 2))

# %%
print(arr[0][0])
print(arr[0, 0])

# %% [markdown]
# ### 슬라이싱

# %%
arr = np.arange(4)
print("arr :", arr)
print("arr[:2] :", arr[:2])
print("arr[::] :", arr[::])
print("arr[1:3] :", arr[1:3])

# %%
print(arr[1:2], arr[1])
print(type(arr[1:2]), type(arr[1]))

# %% [markdown]
# * 인덱싱 결과 : 원소
# * 슬라이싱 결과 : ndarray

# %%
arr = np.arange(20).reshape(4, 5)
print(arr[:2])

# %% [markdown]
# * 2차원 배열 인덱싱 순서: 행 접근 -> 열 접근

# %%
result = []
for row in arr:
    row_01 = [row[0], row[1]]
    print("row_01 :", row_01)
    result.append(row_01)
    print("result :", result)
print(result)

# %%
display(arr)

# %%
print(arr[:2, :2])

# %%
print(arr[1:4, 2:5])
print(arr[1:, 2:])

# %% [markdown]
# ### 브로드 캐스팅

# %%
# 뒤축의 길이가 같고, 둘 중 어느 하나의 축이 1인 경우 브로드캐스팅 가능
arr_high = np.array([92700, 92400, 92100, 94300, 92300])    # (5,)
arr_low = np.array([90000, 91100, 91700, 92100, 90900])     # (5,)
arr_diff = arr_high - arr_low
print("arr_diff :", arr_diff)

# %%
data = [
    [92700, 92400, 92100, 94300, 92300],
    [90000, 91100, 91700, 92100, 90900]
]
arr = np.array(data)
print(arr[0] * 3 + arr[1]  * 2)
weight = np.array([3, 2]).reshape(2, 1)     # 가중치 shape(2, 1)
# 둘 중 하나의 축이 1이므로 브로드캐스팅 가능
display((weight * arr))
display((weight * arr).sum(axis=0))           # axis = 0 : x축 axis = 1 : y축
display((weight * arr).sum(axis=1))           # axis = 0 : x축 axis = 1 : y축

# %% [markdown]
# 브로드 캐스팅 조건
# * 뒤축(열)의 길이가 같다
# * 둘 중 어느 하나의 축이 1이다.

# %% [markdown]
# ### 조건 슬라이싱

# %%
arr = np.array([10, 20, 30])
cond = arr > 10
print(arr[cond])
# [10 20 30]
# [False True True]
display(arr[[False, True, True]])
display(arr[arr > 10])

# %% [markdown]
# ### 조건부 매핑

# %%
# 방법1
arr1 = arr.copy()
arr1[cond] = 1
arr1[~cond] = 0
print(arr1)

arr1 = arr.copy()
arr1[cond] = arr1[cond] + 10
arr1[~cond] = arr1[~cond] - 10
print(arr1)

# %%
# 방법2
arr1 = np.where(arr > 10, 1, 0)
print(arr)
print(arr1)

arr1 = np.where(arr > 10 , arr+10, arr-10)
print(arr1)
"""
    where()은 원본 데이터를 조작하지 않고 새로운 객체로 반환
"""

# %%
display(cond)
print(cond.all())   # 모두 True일 때 True 반환
print(cond.any())   # 한 개라도 True일 때 True 반환

# %% [markdown]
# ### 함수와 메소드

# %%
arr = np.arange(8).reshape(4, 2)
display(arr)

# %%
print(arr.sum(axis=0))      # (4, 2) => 행 압축 (2,)
print(arr.sum(axis=1))      # (4, 2) => 열 압축 (4,)
print(arr.sum())

# %% [markdown]
# axis 방향 규칙
# * 차원 표현 규칙으로 항상 첫번째는 행, 그 다음이 열
# * axis = 0 : 행 방향
# * axis = 1 : 열 방향

# %%
# 0~45 까지 숫자를 랜덤으로 선택(복원추출), (행, 열)
print(f"np.random.randint(46, size=(2, 5)) :\n{np.random.randint(46, size=(2, 5))}")

# %%
# 시작, 끝값(포함), 분할개수(데이터 개수)
x = np.linspace(0, 10, 2)
print(x)

# %%
a = np.arange(4)
b = np.arange(4, 8)
print(f"np.vstack([a, b]) :\n{np.vstack([a, b])}")
print(f"np.hstack([a, b]) :\n{np.hstack([a, b])}")

# %% [markdown]
# * vstack, hstack : 원본 데이터 수정X
# * vstack : 수직방향으로 합치기
# * hstack : 수평방향으로 합치기

# %% [markdown]
# # Pandas

# %% [markdown]
# ## Series

# %%
from pandas import Series

# %%
data = [10, 20, 30]
s = Series(data)
display(s)

# %%
data = np.arange(5)
s = Series(data)
display(s)

# %%
data = ["시가", "고가"]
s = Series(data)
display(s)

# %%
s = Series(['samsung', 81000])
display(s)

# %% [markdown]
# * List
#     * 타입 혼용가능
#     * 위치 인덱스 사용 (0 부터 시작)
# * array
#     * 타입 통일 필요
#     * 브로드 캐스팅
#     * 위치 인덱스 사용 (0 부터 시작)
# * Series
#     * 타입 혼용가능 (내부적으로 object dtype)
#     * object 타입 브로드캐스팅 불가능
#     * 레이블 인덱스 사용
# 

# %% [markdown]
# ### 인덱스

# %%
data = [1000, 2000, 3000]
s = Series(data)
print(f"s.index : {s.index}")
print(f"s.index.to_list() : {s.index.to_list()}")

# %%
# print(s[-1])    # 오류 발생

# %%
# 인덱스 지정
s = Series(data)
s.index = ["메로나", "구구콘", "하겐다즈"]
print(s)

# %%
print(s[0]) # 정상 작동

# %% [markdown]
# [] 인덱싱 사용시: 레이블 인덱스를 먼저 탐색 => 위치 인덱스 사용  
# 
# 레이블이 문자열인 경우: 위치 인덱스(정수) 사용  
# 레이블이 정수인 경우: 레이블 인덱스 사용 (위치 인덱스 사용시 오류 발생)

# %% [markdown]
# #### 파라미터 명시적 입력 (Keyword Argument)

# %%
# 파라미터 명시적 입력 (keyword argument)
# 비슷한 데이터들을 입력할 때 명시적 입력이 오류 발생율을 낮춘다
data = [1000, 2000, 3000]
index = ["메로나", "구구콘", "하겐다즈"]
s = Series(index = index, data = data)
print(s)

# %%
k = [1000, 2000, 3000]
v = ["메로나", "구구콘", "하겐다즈"]
s = Series(data=v, index=k)
print(s)

# %% [markdown]
# #### 인덱스 수정  
# * reindex

# %%
# reindex : 인덱스 변경 새로운 인덱스의 값을 NaN으로 채움
data = [1000, 2000, 3000]
index = ['메로나', '구구콘', '하겐다즈']
s = Series(data=data, index=index)
s2 = s.reindex(["메로나", "비비빅", "구구콘"])
print(f"s :\n{s}")
print(f"s2 :\n{s2}")

# %% [markdown]
# ### 시리즈 생성 방법  
# 1. 리스트 사용
# 2. 딕셔너리 사용

# %%
price = [42500, 42550, 41800, 42550, 42650]
date = ["2019-05-31", "2019-05-30", "2019-05-29", "2019-05-28", "2019-05-27"]
s = Series(data = price, index = date)
print(s)

# %%
data = {
    "2019-05-31" : 42500,
    "2019-05-30" : 42550,
    "2019-05-29" : 41800,
    "2019-05-28" : 42550,
    "2019-05-27" : 42650
}
s = Series(data)
print(s)

# %%
print(s.index)
print(s.index.dtype)
print(s.values)

# %% [markdown]
# ### 명시적 인덱싱  
# - iloc
#     - 행 기준 (위치 인덱스)
# - loc
#     - 인덱스 기준 (레이블)

# %%
data = [1000, 2000, 3000]
s = Series(data=data)
print(f"s.iloc[0] : {s.iloc[0]}")
print(f"s.iloc[-1] : {s.iloc[-1]}")

# %%
print(s.loc[0])
# print(s.loc[-1])   # 에러

# %%
index = ["메로나", "구구콘", "하겐다즈"]
s = Series(data=data, index=index)
print(f"s.iloc[0] : {s.iloc[0]}")
print(f"s.loc['메로나'] : {s.loc['메로나']}")

# %% [markdown]
# #### 슬라이싱   
# - 연속적인 슬라이싱 (:)
# - 연속적이지 않은 슬라이싱 (,)

# %%
data = [1000, 2000, 3000]
index = ["메로나", "구구콘", "하겐다즈"]
s = Series(data=data, index=index)
print(f"s.iloc[0:2] :\n{s.iloc[0:2]}")

# %%
print(f"s.loc['메로나':'구구콘'] :\n{s.loc['메로나':'구구콘']}")

# %%
print("s.iloc[ [0, 2] ] : \n", s.iloc[ [0, 2] ])

# %%
print(s.loc[ ["메로나", "하겐다즈"] ])

# %% [markdown]
# ### 추가, 수정, 삭제
# - drop
#     - 삭제 메소드
#     - 기존 객체는 그대로, 새로운 객체를 반환
#         - inplace = True, 원본 데이터 수정가능

# %%
data = [1000, 2000, 3000]
index = ["메로나", "구구콘", "하겐다즈"]
s = Series(data=data, index=index)

# 수정
s.loc['메로나'] = 500          # 값 수정
s.iloc[0] = 500            # iloc 연산 사용
print(s)

# %%
# 없는 인덱스에 값을 할당하면 추가가 된다
s.loc['비비빅'] = 500          # 값 추가 
print(s)

# %%
# 기존 객체의 내용을 수정하지 않고 새로운 객체를 생성후 리턴한다
display(s.drop('메로나'))
display(s)  # 기존 객체는 그대로

s.drop("메로나", inplace = True)    # 원본 데이터 수정
display(s)

# %% [markdown]
# ### 연산
# - 스칼라 연산
# - 브로드캐스팅
# - idxmax, idxmin
# - cumprod
# - unique
# - value_counts

# %%
철수 = Series([10, 20, 30], index=['NAVER', 'SKT', 'KT'])
영희 = Series([10, 30, 20], index=['SKT', 'KT', 'NAVER'])
가족 = 철수 + 영희
display(가족)

# %%
print(철수 * 10)

# %%
high = Series([42800, 42700, 42050, 42950, 43000])
low = Series([42150, 42150, 41300, 42150, 42350])

# %%
diff = high - low
print(diff)

# %%
print(diff.max())
print(max(diff))

# %%
date = ["6/1", "6/2", "6/3", "6/4", "6/5"]
high = Series([42800, 42700, 42050, 42950, 43000], index=date)
low = Series([42150, 42150, 41300, 42150, 42350] , index=date)
diff = high - low
print(diff)

# %%
# for문 사용 가능
max_idx = 0
max_val = 0

for i in range(len(diff)):
    if diff.iloc[i] > max_val:
        max_val = diff.iloc[i]
        max_idx = i

print(max_idx)
print(diff.index[max_idx])

# %%
print(diff.idxmax())
print(diff.idxmin())

# %%
# 수익률
date = ["6/1", "6/2", "6/3", "6/4", "6/5"]
high = Series([42800, 42700, 42050, 42950, 43000], index=date)
low = Series([42150, 42150, 41300, 42150, 42350] , index=date)
profit = ((high - low) / low) * 100
print(profit)

# %%
# 누적 수익률
display( profit.cumprod( ) )
print( profit.cumprod( ).iloc[ -1 ] )

# %%
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

# %%
display(s.value_counts())

# %%
print(s.value_counts().index)
print(s.value_counts().values)

# %% [markdown]
# ### 문자열 숫자 변환
# - map
#     - 시리즈 원소 하나씩 함수 적용
#     - 새로운 시리즈 객체로 반환
# - replace
#     - 문자열 수정 함수

# %%
s = Series(["1,234", "5,678", "9,876"])
# print( int(s) ) # "," 때문에 타입 에러

# %%
# , 제거
def remove_comma(x) :
    print(x, 'in function')
    return int(x.replace(",", ""))

s = Series(["1,234", "5,678", "9,876"])
# map : 시리즈 객체에 함수 적용
result = s.map(remove_comma)    # 함수 이름만 입력
print(result)

# %%
def is_greater_than_5000(x):
    if x > 5000:
        return "크다"
    else:
        return "작다"

s = Series([1234, 5678, 9876])
s = s.map(is_greater_than_5000)
print(s)

# %% [markdown]
# ### 필터링

# %%
data = [42500, 42550, 41800, 42550, 42650]
index = ['2019-05-31', '2019-05-30', '2019-05-29', '2019-05-28', '2019-05-27']
s = Series(data=data, index=index)
cond = s > 42000
display(cond)
print(cond.all())
print(cond.any())

# %%
print(s[cond])

# %%
close = [42500, 42550, 41800, 42550, 42650]     # 종가
open = [42600, 42200, 41850, 42550, 42500]      # 시가
index = ['2019-05-31', '2019-05-30', '2019-05-29', '2019-05-28', '2019-05-27']

open = Series(data=open, index=index)
close = Series(data=close, index=index)

# %%
cond = close > open
print(cond)
display(close[cond])

# %%
close = [42500, 42550, 41800, 42550, 42650]
open = [42600, 42200, 41850, 42550, 42500]
index = ['2019-05-31', '2019-05-30', '2019-05-29', '2019-05-28', '2019-05-27']

open = Series(data=open, index=index)
close = Series(data=close, index=index)
diff = close - open
print(diff[close > open])

# %% [markdown]
# ### 정렬
# - sort_values
#     - 오름차순 정렬
#     - ascending = False: 내림차순 정렬
# - rank
#     - 오름차순 순위 매기기
#     - ascending = False: 내림차순 순위 매기기

# %%
data = [3.1, 2.0, 10.1, 5.1]
index = ['000010', '000020', '000030', '000040']
s = Series(data=data, index=index)

# %%
# 정렬 (오름차순)
s1 = s.sort_values()
print("오름차순:\n", s1)

# %%
# 정렬 (내림차순)
s2 = s.sort_values(ascending=False)
print("내림차순:\n", s2)

# %%
# 순위 매기기
data = [3.1, 2.0, 10.1, 3.1]
index = ['000010', '000020', '000030', '000040']
s = Series(data=data, index=index)
print(s.rank())

# %%
print(s.rank(ascending=False))

# %% [markdown]
# ## DataFrame
# - Series: 1차원
# - DataFrame: 2차원

# %% [markdown]
# ### 데이터프레임 생성 방법
# - 리스트로 생성
# - 딕셔너리로 생성
# - 딕셔너리 + 리스트로 생성

# %%
from pandas import DataFrame

# %%
# 리스트로 데이터 프레임 생성
data_1 = [
    ["037730", "3R", 1510, 7.36],
    ["036360", "3SOFT", 1790, 1.65],
    ["005670", "ACTS", 1185, 1.28]
]

columns = ["종목코드", "종목명", "현재가", "등락률"]
df = DataFrame(data=data_1, columns=columns)
print(df)

# %% [markdown]
# ### 인덱스 지정
# - DataFrame(data=data, index=index, columns=columns)
# - set_index()
#     - 새로운 객체로 반환
#     - inplace = True: 원본 데이터 수정

# %%
df1 = df.set_index("종목코드")   # 새로운 데이터 프레임 반환
print(df1)

# %%
df.set_index("종목코드", inplace=True)  # 원본 데이터프레임 반환
print(df)

# %%
data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)
df.index.name = "종목코드"
print(df)

# %% [markdown]
# ### 인덱싱
# - 컬럼(열) 인덱싱
#     - 대괄호 인덱싱
#     - .라벨 인덱싱
# - 로우(행) 인덱싱
#     - iloc: 위치 인덱스로 인덱싱
#     - loc: 라벨 인덱스로 인덱싱

# %% [markdown]
# #### 열 인덱싱

# %%
data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)
df.index.name = "종목코드"
print(df['현재가'])

# %%
display(df.현재가)  # 시리즈 반환

# 여러 컬럼 인덱싱, 데이터프레임 반환
display(df[['현재가', "등락률"]])
display(df[["현재가"]])

# %% [markdown]
# #### 행 인덱싱

# %%
# 시리즈 반환
display(df.loc["037730"]) # 인덱스로 인덱싱
display(df.iloc[0])       # 위치 인덱스로 인덱싱

# %%
# 불연속적인 인덱싱, 데이터프레임 반환
display(df.loc[ ["037730", "036360"] ])
display(df.iloc[[0, 1]])

# %% [markdown]
# #### 특정값 가져오기

# %%
# 행번호로 행 선택 후 시리즈를 인덱싱 
display(df.iloc[0])
print(df.iloc[0].iloc[1])            # 시리즈 행번호
print(df.iloc[0].loc["현재가"])        # 시리즈 인덱스 
print(df.iloc[0]["현재가"])            # 시리즈 인덱스

# %%
# 인덱스로 행 선택 후 시리즈를 인덱싱
display(df.loc["037730"])
print(df.loc["037730"].iloc[1])      # 시리즈 행번호
print(df.loc["037730"].loc["현재가"])  # 시리즈 인덱스 
print(df.loc["037730"]["현재가"])      # 시리즈 인덱스

# %%
# 행, 열 동시에 인덱싱
print(df.loc["037730", "현재가"])
print(df.iloc[0, 1])

# %% [markdown]
# #### 특정 범위 가져오기

# %%
display(df.loc[["037730", "036360"]])

# %%
display(df.iloc[[0, 1]])

# %%
display(df.loc[["037730", "036360"], ["종목명", "현재가"]])

# %%
display(df.iloc[ [0, 1], [0, 1] ])

# %% [markdown]
# ### 필터링

# %%
data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)
print(df)

# %%
cond = df['현재가'] >= 1400
print("cond :\n", cond)

# %%
print(df.loc[cond])

# %%
# print(df.loc[cond]["현재가"])     # 시간복잡도 측면에서 비효율적 (두 번 슬라이싱 수행)
print(df.loc[cond, "현재가"])       # 시간복잡도 측면에서 더 효율적

# %%
cond = (df['현재가'] >= 1400) & (df['현재가'] < 1700)
print(df.loc[cond])

# %% [markdown]
# ### 열 추가하기

# %%
s = Series(data=[1600, 1600, 1600], index=df.index) # 같은 인덱스를 가진 시리즈 생성
df['목표가'] = s
print(s)
display(df)

# %%
df["괴리율"] = (df["목표가"] - df["현재가"]) / df['현재가'] # 브로드 캐스팅
display(df)

# %% [markdown]
# ### 행 추가하기

# %%
data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)

s = Series(data=["LG전자", 60000, 3.84], index=df.columns)
df.loc["066570"] = s    # 행은 loc, iloc으로 접근
display(df)

# %% [markdown]
# ### 행, 열 삭제하기
# - .drop

# %%
data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)

new_df = df.drop("현재가", axis=1)  # 열 삭제
display(df)
display(new_df)
new_df = df.drop("037730", axis=0)  # 헹 삭제
display(new_df)

# %% [markdown]
# ### 열 레이블 변경
# - .columns
# - .index
# - .index.name
# - .rename(columns = {}, inplace = False)

# %%
data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)

print(df.columns)
print(df.index)

# %%
df.columns = ['name', 'close', 'fluctuation']
df.index.name = 'code'
display(df)

# %%
df = DataFrame(data=data, index=index, columns=columns)
df.rename(columns={'종목명': 'code'}, inplace=True)
print(df)

# %% [markdown]
# ### 문자열로 저장된 데이터 타입 변경
# - map
# - applymap
# - astype(int)

# %%
data = [
    ["1,000", "1,100", '1,510'],
    ["1,410", "1,420", '1,790'],
    ["850", "900", '1,185'],
]
columns = ["03/02", "03/03", "03/04"]
df = DataFrame(data=data, columns=columns)
display(df)

# %%
def remove_comma(x):
    return int(x.replace(",", ""))

df['03/02'] = df['03/02'].map(remove_comma)
display(df)

# %%
df['03/03'] = df['03/03'].map(remove_comma)
display(df)

# %%
df = DataFrame(data=data, columns=columns)
df = df.map(remove_comma)
display(df)

# %%
# applymap 함수 이용
df = DataFrame(data=data, columns=columns)
df = df.applymap(remove_comma)
display(df)

# %%
print(df.dtypes)

# %% [markdown]
# ### 문자열 다루기
# - Series.str
#     - 시리즈 객체에 있는 원소들에 접근

# %%
data = [
    {"cd":"A060310", "nm":"3S", "close":"2,920"},
    {"cd":"A095570", "nm":"AJ네트웍스", "close":"6,250"},
    {"cd":"A006840", "nm":"AK홀딩스", "close":"29,700"},
    {"cd":"A054620", "nm":"APS홀딩스", "close":"19,400"}
]
df = DataFrame(data=data)
display(df)

# %%
df['cd'] = df['cd'].str[1:]
display(df)

# %%
df['close'] = df['close'].str.replace(',', '')
display(df)

# %% [markdown]
# ### query 함수

# %%
data = [
    {"cd":"A060310", "nm":"3S", "open":2920, "close":2800},
    {"cd":"A095570", "nm":"AJ네트웍스", "open":1920, "close":1900},
    {"cd":"A006840", "nm":"AK홀딩스", "open":2020, "close":2010},
    {"cd":"A054620", "nm":"APS홀딩스", "open":3120, "close":3200}
]
df = DataFrame(data=data)
df = df.set_index('cd')

cond = df['open'] >= 2000   # 불리언 인덱싱
display(df[cond])

# %%
print(df.query("open >= 2000")) # 컬럼명은 따옴표 없이 작성

# %%
print(df.query("nm == '3S'"))

# %%
print(df.query("open < close"))

# %%
print(df.query("nm in ['3S', 'AK홀딩스']")) # in 연산자

# %%
# 인덱스 기준으로
print(df.query("cd == 'A060310'"))

# %%
# 변수 이용 (변수 앞에 @ 기호)
name = "AJ네트웍스"
print(df.query('nm == @name'))

# %% [markdown]
# ### filter 함수

# %%
data = [
    [1416, 1416, 2994, 1755],
    [6.42, 17.63, 21.09, 13.93],
    [1.10, 1.49, 2.06, 1.88]
]

columns = ["2018/12", "2019/12", "2020/12", "2021/12(E)"]
index = ["DPS", "PER", "PBR"]

df = DataFrame(data=data, index=index, columns=columns)
display(df)

# %%
print(df.filter(items=["2018/12"]))

# %%
print(df.filter(items=["PER"], axis=0))     # 행 기준으로 필터링

# %%
print(df.filter(regex="2020"))

# %%
print(df.filter(regex="^2020", axis=1))  # 2020으로 시작하는 열

# %%
print(df.filter(regex="R$", axis=0))   # R로 끝나는 행

# %%
print(df.filter(regex="\d{4}"))   # 4자리 숫자

# %%
print(df.filter(regex="\d{4}/\d{2}$"))  # 4자리 숫자/2자리 숫자로 끝나는 열

# %% [markdown]
# ### 정렬 및 순위
# - sort_values
# - rank
#     - Series 객체로 반환

# %%
data = [
    ["037730", "3R", 1510],
    ["036360", "3SOFT", 1790],
    ["005670", "ACTS", 1185]
]

columns = ["종목코드", "종목명", "현재가"]
df = DataFrame(data=data, columns=columns)
df.set_index("종목코드", inplace=True)
display(df)

# %%
display(df.sort_values("현재가")) # 오름차순 정렬

# %%
print(df.sort_values(by="현재가", ascending=False)) # 내림차순 정렬

# %%
print(df['현재가'].rank())  # 오름차순 순위

# %%
df['순위'] = df['현재가'].rank()
display(df)

# %%
display(df.sort_values(by="순위"))

# %% [markdown]
# ### 인덱스 연산
# - pd.Index
# - idx.union
# - idx.intersection
# - idx.difference

# %%
import pandas as pd

# %%
idx1 = pd.Index([1, 2, 3])
idx2 = pd.Index([2, 3, 4])
print(type(idx1))

# %%
# 합집합
print(idx1.union(idx2))

# %%
# 교집합
print(idx1.intersection(idx2))

# %%
# 차집합
print(idx1.difference(idx2))

# %% [markdown]
# ### GroupBy
# - groupby()
# - .agg

# %%
data = [
    ["2차전지(생산)", "SK이노베이션", 10.19, 1.29],
    ["해운", "팬오션", 21.23, 0.95],
    ["시스템반도체", "티엘아이", 35.97, 1.12],
    ["해운", "HMM", 21.52, 3.20],
    ["시스템반도체", "아이에이", 37.32, 3.55],
    ["2차전지(생산)", "LG화학", 83.06, 3.75]
]

columns = ["테마", "종목명", "PER", "PBR"]
df = DataFrame(data=data, columns=columns)
display(df)

# %%
# 필터링을 이용한 그룹화
df1 = df[df["테마"] == "2차전지(생산)"]
df2 = df[df["테마"] == "해운"]
df3 = df[df["테마"] == "시스템반도체"]
display(df1)

# %%
# 각 그룹별 PER 평균
mean1 = df1["PER"].mean()
mean2 = df2["PER"].mean()
mean3 = df3["PER"].mean()
print(mean1)

# %%
data = [mean1, mean2, mean3]
index = ["2차전지(생산)", "해운", "시스템반도체"]
s = pd.Series(data=data, index=index)
print(s)

# %%
# GroupBy 메서드 이용 (컬럼 레이블 기준으로 그룹화)
# get_group(그룹명) : 특정 그룹만 선택
print(df.groupby("테마").get_group("2차전지(생산)"))

# %%
temp = df[["테마", "PER", "PBR"]].groupby("테마").get_group("2차전지(생산)")
print(temp)

# %%
temp = df.groupby("테마")[ ["PER", "PBR"] ].get_group("2차전지(생산)")
print(temp)

# %%
print(df.groupby("테마")["PER"].mean())

# %%
print(df.groupby("테마")[["PER", "PBR"]].mean())

# %%
# 그룹별 연산 지정
print(df.groupby("테마").agg({"PER": "max", "PBR": "min"}))

# %%
# 여러 연산 지정
display(df.groupby("테마").agg({"PER": ["min", "max"], "PBR": ["std", "var"]}))

# %% [markdown]
# ### 데이터 프레임 병합

# %% [markdown]
# #### concat

# %%
data = {
    '종가': [113000, 111500],
    '거래량': [555850, 282163]
}

index = ["2019-06-21", "2019-06-20"]
df1 = DataFrame(data=data, index=index)
print(df1)

# %%
data = {
    '시가': [112500, 110000],
    '고가': [115000, 112000],
    '저가': [111500, 109000]
}
df2 = DataFrame(data=data, index=index)
print(df2)

# %%
df = pd.concat([df1, df2], axis=1)  # 컬럼 기준 병합
print(df)

# %% [markdown]
# ##### 열 순서 수정

# %%
정렬순서 = ['시가', '고가', '저가', '종가', '거래량']
df = df[정렬순서]
print(df)

# %% [markdown]
# ##### 인덱스가 다른 데이터프레임 병합

# %%
data = {
    '종가': [113000, 111500],
    '거래량': [555850, 282163]
}

index = ["2019-06-21", "2019-06-20"]
df1 = DataFrame(data=data, index=index)

data = {
    '시가': [112500, 110000],
    '고가': [115000, 112000],
    '저가': [111500, 109000]
}

index = ["2019-06-20", "2019-06-19"]
df2 = DataFrame(data=data, index=index)

df = pd.concat([df1, df2], axis=1)
print(df)   # 결측치 발생

# %%
# join 메서드 이용 (중요)
# inner : 교집합, outer : 합집합(기본값)
df = pd.concat([df1, df2], axis=1, join='inner')
print(df)
display(pd.concat([df1, df2], axis=1, join='outer'))

# %% [markdown]
# ##### 행 기준 병합

# %%
# append 메서드 이용 (행 기준 병합) 함수가 삭제됨 concat 사용 권장
# 첫번째 데이터프레임
data = {
    '종가': [113000, 111500],
    '거래량': [555850, 282163]
}
index = ["2019-06-21", "2019-06-20"]
df1 = DataFrame(data, index=index)

# 두번째 데이터프레임
data = {
    '종가': [110000, 483689],
    '거래량': [109000, 791946]
}
index = ["2019-06-19", "2019-06-18"]
df2 = DataFrame(data, index=index)

# df = df1.append(df2)
# print(df)

# %%
df = pd.concat([df1, df2], axis=0)  # 행 기준 병합 axis=0 (기본값)
print(df)

# %% [markdown]
# #### merge

# %%
# 첫 번째 데이터프레임
data = [
    ["전기전자", "005930", "삼성전자", 74400],
    ["화학", "051910", "LG화학", 896000],
    ["전기전자", "000660", "SK하이닉스", 101500]
]

columns = ["업종", "종목코드", "종목명", "현재가"]
df1 = DataFrame(data=data, columns=columns)
display(df1)

# 두 번째 데이터프레임
data = [
    ["은행", 2.92],
    ["보험", 0.37],
    ["화학", 0.06],
    ["전기전자", -2.43]
]

columns = ["업종", "등락률"]
df2 = DataFrame(data=data, columns=columns)
display(df2)

# %%
display(pd.merge(left=df1, right=df2, on='업종'))

# %%
display(pd.merge(left=df1, right=df2, how='inner', on='업종')) # 교집합

# %%
display(pd.merge(left=df1, right=df2, how='outer', on='업종')) # 합집합

# %% [markdown]
# ##### 컬럼명이 다를 경우

# %%
# 첫 번째 데이터프레임
data = [
    ["전기전자", "005930", "삼성전자", 74400],
    ["화학", "051910", "LG화학", 896000],
    ["서비스업", "035720", "카카오", 121500]
]

columns = ["업종", "종목코드", "종목명", "현재가"]
df1 = DataFrame(data=data, columns=columns)

# 두 번째 데이터프레임
data = [
    ["은행", 2.92],
    ["보험", 0.37],
    ["화학", 0.06],
    ["전기전자", -2.43]
]

columns = ["항목", "등락률"]
df2 = DataFrame(data=data, columns=columns)

# %%
# left_on, right_on 옵션 이용 각 컬럼명 지정
df = pd.merge(left=df1, right=df2, left_on='업종', right_on='항목')
print(df)

# %% [markdown]
# #### join (인덱스기준 병합)

# %%
# 첫 번째 데이터프레임
data = [
    ["전기전자", "005930", "삼성전자", 74400],
    ["화학", "051910", "LG화학", 896000],
    ["서비스업", "035720", "카카오", 121500]
]

columns = ["업종", "종목코드", "종목명", "현재가"]
df1 = DataFrame(data=data, columns=columns)
df1 = df1.set_index("업종")
display(df1)

# 두 번째 데이터프레임
data = [
    ["은행", 2.92],
    ["보험", 0.37],
    ["화학", 0.06],
    ["전기전자", -2.43]
]

columns = ["항목", "등락률"]
df2 = DataFrame(data=data, columns=columns)
df2 = df2.set_index("항목")
display(df2)

# %%
display(df1.join(other=df2))

# %%
data = [
    ["2017", "삼성", 500],
    ["2017", "LG", 300],    
    ["2017", "SK하이닉스", 200],
    ["2018", "삼성", 600],
    ["2018", "LG", 400],
    ["2018", "SK하이닉스", 300],    
]

columns = ["연도", "회사", "시가총액"]
df = DataFrame(data=data, columns=columns)
display(df)

# %%
df_mean = df.groupby("연도")["시가총액"].mean().to_frame()
df_mean.columns = ['시가총액평균']
display(df_mean)

# %%
df = df.join(df_mean, on='연도')
display(df)

# %%
# 시가총액이 시가총액평균 이상이면 '대형주', 미만이면 '중/소형주'로 분류
df['규모'] = np.where(df['시가총액'] >= df['시가총액평균'], "대형주", "중/소형주")
display(df)

# %% [markdown]
# ### 멀티인덱스
# - set_index([idx1, idx2], inplace = False)
#     - 순서대로 상위 레벨, 하위 레벨
#     - .loc[하위레벨] 에러발생, .loc[상위레벨]로 접근가능
#     - 모든 레벨의 인덱스를 사용할 경우 튜플로 인덱스 접근 .loc[(상위, 하위)]
#     - 슬라이싱 가능 .loc[(상위, 하위) : (상위, 하위)]
# - slice(start, stop, step)
#     - slice(None) = : (전체)
# - idx = pd.IndexSlice
#     - idx[:, "하위레벨"]

# %%
data = [
    ["영업이익", "컨센서스", 1000, 1200],
    ["영업이익", "잠정치", 900, 1400],
    ["당기순이익", "컨센서스", 800, 900],
    ["당기순이익", "잠정치", 700, 800]
]

df = DataFrame(data=data)
df = df.set_index([0, 1])
display(df)

# %%
df.index.names = ["재무연월", ""]
df.columns = ["2020/06", "2020/09"]
display(df)

# %%
display(df.loc["영업이익"])

display(df.loc[ ("영업이익", "컨센서스") ])

display(df.iloc[0])

print(df.iloc[0, 0])

print(df.loc[("영업이익", "컨센서스"), "2020/06"])

# %% [markdown]
# #### slice, IndexSlice

# %%
a = [1, 2, 3, 4, 5]

print(a[0:5:2])
print(a[slice(0, 5, 2)])

# %%
a = [1, 2, 3, 4, 5]
b = [3, 4, 5, 6, 7]

s = slice(0, 5, 2)
print(a[ s ])
print(b[ s ])

# %%
a = [1, 2, 3, 4, 5]

display(a[:])
display(a[slice(None)])
display(a[ : : ])
display(a[slice(None, None)])

# %%
# display(df.loc[ ( :, '컨센서스'), : ])    # error
display(df.loc[ (slice(None), '컨센서스'), :])

# %%
idx = pd.IndexSlice
display(df.loc[idx[:, "컨센서스"], :])

# %% [markdown]
# ### 멀티 컬럼

# %%
data = [
    [100, 900, 800, 700],
    [1200, 1400, 900, 800]
]

columns = [
    ['영업이익', '영업이익', '당기순이익', '당기순이익'],
    ['컨센서스', '잠정치', '컨센서스', '잠정치']
]

index = ["2020/06", "2020/09"]

df = DataFrame(data=data, index=index, columns=columns)
display(df)

# %%
level_0 = ["영업이익", "당기순이익"]
level_1 = ["컨센서스", "잠정치"]

idx = pd.MultiIndex.from_product([level_0, level_1])
display(idx)

display(idx.get_level_values(0))

display(idx.get_level_values(1))

display(df["영업이익"])

# %% [markdown]
# #### 멀티 컬럼 재구조화

# %%
# stack, unstack
data = [
    [100, 900, 800, 700],
    [1200, 1400, 900, 800]
]

columns = [
    ['영업이익', '영업이익', '당기순이익', '당기순이익'],
    ['컨센서스', '잠정치', '컨센서스', '잠정치']
]

index = ["2020/06", "2020/09"]

df = DataFrame(data=data, index=index, columns=columns)
display(df)

display(df.stack())

# %%
display(df.stack(level=0))

display(df.stack().stack())

display(df.stack().unstack())

# %%
data = [
    [1000, 1100, 900, 1200, 1300],
    [800, 2000, 1700, 1500, 1800]
]
index = ['자본금', '부채']
columns = ["2020/03", "2020/06", "2020/09", "2021/03", "2021/06"]
df = DataFrame(data, index, columns)
display(df)

# %%
df_stacked = df.stack().reset_index()
display(df_stacked)

# %%
display(df_stacked['level_1'].str.split('/'))

# %%
df_split = DataFrame( list(df_stacked['level_1'].str.split('/')) )
display(df_split)

# %%
df_merged = pd.concat( [df_stacked, df_split], axis=1 )
df_merged.columns = ['계정', "년월", "금액", "연도", "월"]
display(df_merged)

df_group = df_merged.groupby(["계정", "연도"]).sum()
display(df_group)

# %%
df_unstack = df_group.unstack()
df_unstack

# %%
result = df_unstack['금액']
result.columns.name = ''
result.index.name = ''
result

# %%
data = [
    ["2021-08-12", "삼성전자", 77000],
    ["2021-08-13", "삼성전자", 74400],
    ["2021-08-12", "LG전자", 153000],
    ["2021-08-13", "LG전자", 150500],
    ["2021-08-12", "SK하이닉스", 100500],
    ["2021-08-13", "SK하이닉스", 101500]
]
columns = ["날짜", "종목명", "종가"]
df = DataFrame(data=data, columns=columns)
display(df)

display(pd.pivot(data=df, index="날짜", columns="종목명", values="종가"))

# %%
display(df.groupby(["날짜", "종목명"]).mean().unstack())

# %% [markdown]
# ### melt

# %%
data = [
    ["005930", "삼성전자", 75800, 76000, 74100, 74400],
    ["035720", "카카오", 147500, 147500, 144500, 146000],
    ["000660", "SK하이닉스", 99600, 101500, 98900, 101500]
]

columns = ["종목코드", "종목명", "시가", "고가", "저가", "종가"]
df = DataFrame(data=data, columns=columns)
display(df)

# %%
display(df.melt())

# %%
display(df.melt(id_vars=['종목코드', '종목명']))

# %%
display(df.melt(value_vars=['시가', '종가']))


