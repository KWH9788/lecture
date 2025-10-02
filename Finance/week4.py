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
sep()

# 필터링
data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)
print(df)
sep()

cond = df['현재가'] >= 1400
print("cond :\n", cond)
sep()

print(df.loc[cond])
sep()

# print(df.loc[cond]["현재가"])     # 시간복잡도 측면에서 비효율적 (두 번 슬라이싱 수행)
print(df.loc[cond, "현재가"])       # 시간복잡도 측면에서 더 효율적
sep()

cond = (df['현재가'] >= 1400) & (df['현재가'] < 1700)
print(df.loc[cond])
sep()

"""
    컴퓨터는 (and, or, not)전가산기로 이루어져 있다
"""

# 컬럼(열) 추가하기
s = Series(data=[1600, 1600, 1600], index=df.index) # 같은 인덱스를 가진 시리즈 생성
df['목표가'] = s
print(s)
sep()

print(df)
sep()

df["괴리율"] = (df["목표가"] - df["현재가"]) / df['현재가'] # 브로드 캐스팅
print(df)
sep()

data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)

# 로우(행) 추가하기
s = Series(data=["LG전자", 60000, 3.84], index=df.columns)
df.loc["066570"] = s    # 행은 loc, iloc으로 접근
print(df)
sep()

# data = [
#     ["3R", 1510, 7.36],
#     ["3SOFT", 1790, 1.65],
#     ["ACTS", 1185, 1.28]
# ]

# index = ["037730", "036360", "005760"]
# columns = ["종목명", "현재가", "등락률"]
# df = DataFrame(data=data, index=index, columns=columns)

# s = Series(data=["LG전자", 60000, 3.84], index=df.columns, name="066570")
# df.append(s)
# print(df)
# sep()

# 컬럼(열), 로우(행) 삭제하기
data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)

new_df = df.drop("현재가", axis=1)  # 컬럼 삭제
print(df)
print(new_df)
sep()

# 컬럼 레이블 변경
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
sep()

df.columns = ['name', 'close', 'fluctuation']
df.index.name = 'code'
print(df)
sep()

df = DataFrame(data=data, index=index, columns=columns)
df.rename(columns={'종목명': 'code'}, inplace=True)
print(df)
sep()

# 문자열(object)로 저장된 데이터 타입 변경
data = [
    ["1,000", "1,100", '1,510'],
    ["1,410", "1,420", '1,790'],
    ["850", "900", '1,185'],
]
columns = ["03/02", "03/03", "03/04"]
df = DataFrame(data=data, columns=columns)
print(df)
sep()

# map 함수 이용
def remove_comma(x):
    return int(x.replace(",", ""))

df['03/02'] = df['03/02'].map(remove_comma)
print(df)
sep()

df['03/03'] = df['03/03'].map(remove_comma)
print(df)
sep()

# applymap 함수 이용
# df = DataFrame(data=data, columns=columns)
# df = df.applymap(remove_comma)
# print(df)
# sep()

# print(df.dtypes)
# sep()

# astype 함수 이용
# df = DataFrame(data=data, columns=columns)
# def remove_comma(x):
#     return x.replace(",", "")

# df = df.applymap(remove_comma)
# df = df.astype(int)
# print(df.dtypes)
# sep()

data = [
    {"cd":"A060310", "nm":"3S", "close":"2,920"},
    {"cd":"A095570", "nm":"AJ네트웍스", "close":"6,250"},
    {"cd":"A006840", "nm":"AK홀딩스", "close":"29,700"},
    {"cd":"A054620", "nm":"APS홀딩스", "close":"19,400"}
]
df = DataFrame(data=data)
print(df)
sep()

df['cd'] = df['cd'].str[1:]
print(df)
sep()

df['close'] = df['close'].str.replace(',', '')
print(df)
sep()

# query 함수 이용
data = [
    {"cd":"A060310", "nm":"3S", "open":2920, "close":2800},
    {"cd":"A095570", "nm":"AJ네트웍스", "open":1920, "close":1900},
    {"cd":"A006840", "nm":"AK홀딩스", "open":2020, "close":2010},
    {"cd":"A054620", "nm":"APS홀딩스", "open":3120, "close":3200}
]
df = DataFrame(data=data)
df = df.set_index('cd')
print(df)
sep()

cond = df['open'] >= 2000   # 불리언 인덱싱
print(df[cond])
sep()

print(df.query("open >= 2000")) # 컬럼명은 따옴표 없이 작성
sep()

print(df.query("nm == '3S'"))
sep()

print(df.query("open < close"))
sep()

print(df.query("nm in ['3S', 'AK홀딩스']")) # in 연산자
sep()

# 인덱스 기준으로
print(df.query("cd == 'A060310'"))
sep()

# 변수 이용 (변수 앞에 @ 기호)
name = "AJ네트웍스"
print(df.query('nm == @name'))
sep()

"""
    Hard Coding 이란
    - 코드에 값을 직접 입력하는 것
    - static하다 (정적이다)
    - 유지보수에 굉장히 취약하다
    - 보안적으로 굉장히 취약하다
    .env 파일을 만들어서 관리하는 것이 좋다 (dotenv 라이브러리 이용)
    - ex) df.query("nm == '3S'")  -> df.query("nm == @name")
"""

# filter 함수 이용
data = [
    [1416, 1416, 2994, 1755],
    [6.42, 17.63, 21.09, 13.93],
    [1.10, 1.49, 2.06, 1.88]
]

columns = ["2018/12", "2019/12", "2020/12", "2021/12(E)"]
index = ["DPS", "PER", "PBR"]

df = DataFrame(data=data, index=index, columns=columns)
print(df)
sep()

print(df.filter(items=["2018/12"]))
sep()

print(df.filter(items=["PER"], axis=0))     # 행 기준으로 필터링
sep()

print(df.filter(regex="2020"))
sep()

print(df.filter(regex="^2020", axis=1))  # 2020으로 시작하는 열
sep()

print(df.filter(regex="R$", axis=0))   # R로 끝나는 행
sep()

print(df.filter(regex="\d{4}"))   # 4자리 숫자
sep()

print(df.filter(regex="\d{4}/\d{2}$"))  # 4자리 숫자/2자리 숫자로 끝나는 열
sep()

# 정렬 및 순위
data = [
    ["037730", "3R", 1510],
    ["036360", "3SOFT", 1790],
    ["005670", "ACTS", 1185]
]

columns = ["종목코드", "종목명", "현재가"]
df = DataFrame(data=data, columns=columns)
df.set_index("종목코드", inplace=True)
print(df)
sep()

print(df.sort_values("현재가")) # 오름차순 정렬
sep()

print(df.sort_values(by="현재가", ascending=False)) # 내림차순 정렬
sep()

print(df['현재가'].rank())  # 오름차순 순위
sep()

df['순위'] = df['현재가'].rank()
print(df)
sep()

print(df.sort_values(by="순위"))
sep()