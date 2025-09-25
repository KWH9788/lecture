from pandas import DataFrame, Series
import numpy as np
import pandas as pd

def sep():
    print("="*60)

# 데이터 프레임
"""
    시리즈 = 1차원
    데이터프레임 = 2차원
"""

# 리스트로 데이터 프레임 생성
data_1 = [
    ["037730", "3R", 1510, 7.36],
    ["036360", "3SOFT", 1790, 1.65],
    ["005670", "ACTS", 1185, 1.28]
]

columns = ["종목코드", "종목명", "현재가", "등락률"]
df = DataFrame(data=data_1, columns=columns)
print(df)
sep()

# 인덱스 지정
data = [
    ["037730", "3R", 1510, 7.36],
    ["036360", "3SOFT", 1790, 1.65],
    ["005670", "ACTS", 1185, 1.28]
]

columns = ["종목코드", "종목명", "현재가", "등락률"]
df = DataFrame(data=data, columns=columns)
df = df.set_index("종목코드")   # 새로운 데이터 프레임 반환
print(df)
sep()

# 인덱스 지정 2
data = [
    ["037730", "3R", 1510, 7.36],
    ["036360", "3SOFT", 1790, 1.65],
    ["005670", "ACTS", 1185, 1.28]
]

columns = ["종목코드", "종목명", "현재가", "등락률"]
df = DataFrame(data=data, columns=columns)
df.set_index("종목코드", inplace=True)  # 원본 데이터프레임 반환
print(df)
sep()

# 인덱스 지정 3
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
sep()

# 컬럼 인덱싱
# 대괄호로 컬럼 인덱싱
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
sep()

# 컬럼 인덱싱 2
print(df.현재가)
sep()

# 여러 컬럼 인덱싱
print(df[['현재가', "등락률"]])
sep()

print(df["현재가"])     # 시리즈 반환
sep()

print(df[["현재가"]])   # 데이터프레임 반환
sep()

# 로우 인덱싱
# iloc : 행 번호로 인덱싱   (0부터 시작)
# loc : 행 이름(인덱스)으로 인덱싱

print(df.loc["037730"]) # 인덱스로 인덱싱
sep()

print(df.iloc[0])       # 행 번호로 인덱싱
sep()

print(df.iloc[-1])
sep()

# 불연속적인 인덱싱, 데이터프레임 반환
print(df.loc[ ["037730", "036360"] ])
sep()

print(df.iloc[[0, 1]])
sep()

# 특정값 가져오기
# 행번호로 행 선택 후 시리즈를 인덱싱 
print(df.iloc[0])
print(df.iloc[0].iloc[1])            # 시리즈 행번호
print(df.iloc[0].loc["현재가"])        # 시리즈 인덱스 
print(df.iloc[0]["현재가"])            # 시리즈 인덱스
sep()

# 인덱스로 행 선택 후 시리즈를 인덱싱
print(df.loc["037730"])
print(df.loc["037730"].iloc[1])      # 시리즈 행번호
print(df.loc["037730"].loc["현재가"])  # 시리즈 인덱스 
print(df.loc["037730"]["현재가"])      # 시리즈 인덱스
sep()

# 행, 열 동시에 인덱싱
print(df.loc["037730", "현재가"])
print(df.iloc[0, 1])
sep()

# 특정 범위 가져오기
print(df.loc[["037730", "036360"]])
sep()
print(df.iloc[[0, 1]])
sep()
print(df.loc[["037730", "036360"], ["종목명", "현재가"]])
sep()
print(df.iloc[ [0, 1], [0, 1] ])

