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

