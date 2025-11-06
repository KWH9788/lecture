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

# 멀티인덱스
data = [
    ["영업이익", "컨센서스", 1000, 1200],
    ["영업이익", "잠정치", 900, 1400],
    ["당기순이익", "컨센서스", 800, 900],
    ["당기순이익", "잠정치", 700, 800]
]

df = DataFrame(data=data)
df = df.set_index([0, 1])
print(df)
sep()

df.index.names = ["재무연월", ""]
df.columns = ["2020/06", "2020/09"]
print(df)
sep()

print(df.loc["영업이익"])
sep()

print(df.loc[ ("영업이익", "컨센서스") ])
sep()

print(df.iloc[0])
sep()

print(df.iloc[0, 0])
sep()

print(df.loc[("영업이익", "컨센서스"), "2020/06"])
sep()

a = [1, 2, 3, 4, 5]

print(a[0:5:2])
print(a[slice(0, 5, 2)])
sep()

a = [1, 2, 3, 4, 5]
b = [3, 4, 5, 6, 7]

s = slice(0, 5, 2)
print(a[ s ])
print(b[ s ])
sep()

a = [1, 2, 3, 4, 5]

print(a[:])
print(a[slice(None)])
print(a[ : : ])
print(a[slice(None, None)])
sep()

# print( df.loc[ ( :, '컨센서스'), : ] )
# sep()

print(df.loc[ (slice(None), '컨센서스'), :])
sep()

idx = pd.IndexSlice
print(df.loc[idx[:, "컨센서스"], :])
sep()

# 멀티 컬럼
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
print(df)
sep()

level_0 = ["영업이익", "당기순이익"]
level_1 = ["컨센서스", "잠정치"]

idx = pd.MultiIndex.from_product([level_0, level_1])
print(idx)
sep()

print(idx.get_level_values(0))
sep()

print(idx.get_level_values(1))
sep()

print(df["영업이익"])
sep()

# stack, unstack
# data = [
#     [100, 900, 800, 700],
#     [1200, 1400, 900, 800]
# ]

# columns = [
#     ['영업이익', '영업이익', '당기순이익', '당기순이익'],
#     ['컨센서스', '잠정치', '컨센서스', '잠정치']
# ]

# index = ["2020/06", "2020/09"]

# df = DataFrame(data=data, index=index, columns=columns)
# print(df)
# sep()

# print(df.stack())
# sep()

# print(df.stack(level=0))
# sep()

# print(df.stack().stack())
# sep()

# print(df.stack().unstack())
# sep()

data = [
    [1000, 1100, 900, 1200, 1300],
    [800, 2000, 1700, 1500, 1800]
]
index = ['자본금', '부채']
columns = ["2020/03", "2020/06", "2020/09", "2021/03", "2021/06"]
df = DataFrame(data, index, columns)
print(df)
sep()

df_stacked = df.stack().reset_index()
print(df_stacked)
sep()

print(df_stacked['level_1'].str.split('/'))
sep()

df_split = DataFrame( list(df_stacked['level_1'].str.split('/')) )
print(df_split)
sep()

df_merged = pd.concat( [df_stacked, df_split], axis=1 )
df_merged.columns = ['계정', "년월", "금액", "연도", "월"]
print(df_merged)
sep()

df_group = df_merged.groupby(["계정", "연도"]).sum()
print(df_group)
sep()

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
print(df)
sep()

print(pd.pivot(data=df, index="날짜", columns="종목명", values="종가"))
sep()

print(df.groupby(["날짜", "종목명"]).mean().unstack())
sep()

# melt
data = [
    ["005930", "삼성전자", 75800, 76000, 74100, 74400],
    ["035720", "카카오", 147500, 147500, 144500, 146000],
    ["000660", "SK하이닉스", 99600, 101500, 98900, 101500]
]

columns = ["종목코드", "종목명", "시가", "고가", "저가", "종가"]
df = DataFrame(data=data, columns=columns)
print(df)
sep()

print(df.melt())
sep()

print(df.melt(id_vars=['종목코드', '종목명']))
sep()

print(df.melt(value_vars=['시가', '종가']))
sep()

# 
data = [
    ["3R", 1510, 7.36],
    ["3SOFT", 1790, 1.65],
    ["ACTS", 1185, 1.28]
]

index = ["037730", "036360", "005760"]
columns = ["종목명", "현재가", "등락률"]
df = DataFrame(data=data, index=index, columns=columns)
df.index.name = '종목코드'
print(df)
sep()

df.to_csv("data.csv")