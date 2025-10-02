import pandas as pd
from pandas import DataFrame
import numpy as np

def sep():
    print("="*60)

# 인덱스 연산
idx1 = pd.Index([1, 2, 3])
idx2 = pd.Index([2, 3, 4])
print(type(idx1))
sep()

# 합집합
print(idx1.union(idx2))
sep()

# 교집합
print(idx1.intersection(idx2))
sep()

# 차집합
print(idx1.difference(idx2))
sep()

# GroupBy
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
print(df)
sep()

# 필터링을 이용한 그룹화
df1 = df[df["테마"] == "2차전지(생산)"]
print(df1)
sep()
df2 = df[df["테마"] == "해운"]
df3 = df[df["테마"] == "시스템반도체"]

# 각 그룹별 PER 평균
mean1 = df1["PER"].mean()
print(mean1)
sep()
mean2 = df2["PER"].mean()
mean3 = df3["PER"].mean()

data = [mean1, mean2, mean3]
index = ["2차전지(생산)", "해운", "시스템반도체"]
s = pd.Series(data=data, index=index)
print(s)
sep()

# GroupBy 메서드 이용 (컬럼 레이블 기준으로 그룹화)
# get_group(그룹명) : 특정 그룹만 선택
print(df.groupby("테마").get_group("2차전지(생산)"))
sep()

temp = df[["테마", "PER", "PBR"]].groupby("테마").get_group("2차전지(생산)")
print(temp)
sep()

temp = df.groupby("테마")[ ["PER", "PBR"] ].get_group("2차전지(생산)")
print(temp)
sep()

print(df.groupby("테마")["PER"].mean())
sep()

print(df.groupby("테마")[["PER", "PBR"]].mean())
sep()

# 그룹별 연산 지정
print(df.groupby("테마").agg({"PER": "max", "PBR": "min"}))
sep()

# 여러 연산 지정
# print(df.groupby("테마").agg({"PER": ["min", "max"], "PBR": [np.std, np.var]}))

# 데이터 프레임 병합
data = {
    '종가': [113000, 111500],
    '거래량': [555850, 282163]
}

index = ["2019-06-21", "2019-06-20"]
df1 = DataFrame(data=data, index=index)
print(df1)
sep()

data = {
    '시가': [112500, 110000],
    '고가': [115000, 112000],
    '저가': [111500, 109000]
}
df2 = DataFrame(data=data, index=index)
print(df2)
sep()

df = pd.concat([df1, df2], axis=1)  # 컬럼 기준 병합
print(df)
sep()

# 컬럼 정렬 수정
정렬순서 = ['시가', '고가', '저가', '종가', '거래량']
df = df[정렬순서]
print(df)
sep()

# 인덱스가 다른 데이터 프레임 병합
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
sep()

# join 메서드 이용 (중요)
# inner : 교집합, outer : 합집합(기본값)
df = pd.concat([df1, df2], axis=1, join='inner')
print(df)
sep()

print(pd.concat([df1, df2], axis=1, join='outer'))
sep()

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
# sep()

df = pd.concat([df1, df2], axis=0)  # 행 기준 병합 axis=0 (기본값)
print(df)
sep()

# merge 메서드 이용
# 첫 번째 데이터프레임
data = [
    ["전기전자", "005930", "삼성전자", 74400],
    ["화학", "051910", "LG화학", 896000],
    ["전기전자", "000660", "SK하이닉스", 101500]
]

columns = ["업종", "종목코드", "종목명", "현재가"]
df1 = DataFrame(data=data, columns=columns)
print(df1)
sep()

# 두 번째 데이터프레임
data = [
    ["은행", 2.92],
    ["보험", 0.37],
    ["화학", 0.06],
    ["전기전자", -2.43]
]

columns = ["업종", "등락률"]
df2 = DataFrame(data=data, columns=columns)
print(df2)
sep()

print(pd.merge(left=df1, right=df2, on='업종'))
sep()

# how 옵션
print(pd.merge(left=df1, right=df2, how='inner', on='업종')) # 교집합
sep()

print(pd.merge(left=df1, right=df2, how='outer', on='업종')) # 합집합
sep()

# 컬럼명이 다를 경우
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

# left_on, right_on 옵션 이용 각 컬럼명 지정
df = pd.merge(left=df1, right=df2, left_on='업종', right_on='항목')
print(df)
sep()

# join (인덱스기준 병합)
# 첫 번째 데이터프레임
data = [
    ["전기전자", "005930", "삼성전자", 74400],
    ["화학", "051910", "LG화학", 896000],
    ["서비스업", "035720", "카카오", 121500]
]

columns = ["업종", "종목코드", "종목명", "현재가"]
df1 = DataFrame(data=data, columns=columns)
df1 = df1.set_index("업종")
print(df1)

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
print(df2)
sep()

print(df1.join(other=df2))
sep()

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
print(df)
sep()

df_mean = df.groupby("연도")["시가총액"].mean().to_frame()
df_mean.columns = ['시가총액평균']
print(df_mean)
sep()

df = df.join(df_mean, on='연도')
print(df)
sep()

# 시가총액이 시가총액평균 이상이면 '대형주', 미만이면 '중/소형주'로 분류
df['규모'] = np.where(df['시가총액'] >= df['시가총액평균'], "대형주", "중/소형주")
print(df)
sep()