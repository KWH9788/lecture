# Import
import pandas as pd
import numpy as np
from pykrx import stock
import matplotlib.pyplot as plt
import platform
import time

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ==========================
# === 시가총액별 콤보 전략 ===
# ==========================

# === step 1: 데이터 로드 ===
# 월초 종목별 종가, 시가총액 데이터 로드
df1 = stock.get_market_cap_by_ticker("20100104")
df1 = df1[["종가", "시가총액"]]
df1.columns = ["시가", "시가총액"]

# 월초 종목별 PER, PBR 데이터 로드
df2 = stock.get_market_fundamental_by_ticker("20100104")
df2 = df2[["PER", "PBR"]]
# print(df2.head())

# 월말 종목별 종가 데이터 로드
df3 = stock.get_market_ohlcv_by_ticker("20101231", alternative=True)
df3 = df3[['종가']]
# print(df3.head())

# === step 2: 데이터 전처리 ===

# 시가총액 기준으로 오름차순 정렬
df1 = df1.sort_values('시가총액')
# print(df1.head())

# 시가총액 기준으로 3그룹으로 분할
df1['group'] = pd.cut(df1.reset_index().index, bins=3, labels=['소형주', '중형주', '대형주'])
# print(df1.tail())

# 데이터 병합
t0 = pd.merge(left=df1, right=df2, left_index=True, right_index=True)
df = pd.merge(left=t0, right=df3, left_index=True, right_index=True)
# print(df.head())

# PER, PBR이 0인 종목 제거
df = df.query('PBR != 0')

# 수익률 계산
df['수익률'] = df['종가'] / df['시가']

# PER 2.5 이상 10 이하 조건
cond = (df['PER'] >= 2.5) & (df['PER'] <= 10)
# PER 기준 그룹별 하위 30개 종목 추출
top30 = df[cond].sort_values('PBR').groupby('group').head(30)
# print(top30)

# 그룹별 수익률 평균 계산
how = {
    '수익률' : np.mean
}
yoy = top30.groupby('group').agg(how)
yoy.columns = ['2010']
# print(yoy)

# 데이터 전처리 함수화
def low_per_pbr(year):
    df1 = stock.get_market_cap_by_ticker(f"{year}0101", alternative=True)
    df1 = df1[["종가", "시가총액"]]
    df1.columns = ["시가", "시가총액"]
    df1 = df1.sort_values('시가총액')
    df1['group'] = pd.cut(df1.reset_index().index, bins=3, labels=['소형주', '중형주', '대형주'])
    # print(df1.head())

    df2 = stock.get_market_fundamental_by_ticker(f"{year}0101", alternative=True)
    df2 = df2[['PER', 'PBR']]
    # print(df2.head())

    df3 = stock.get_market_ohlcv_by_ticker(f"{year}1231", alternative=True)
    df3 = df3[['종가']]
    # print(df3.head())

    t0 = pd.merge(left=df1, right=df2, left_index=True, right_index=True)
    df = pd.merge(left=t0, right=df3, left_index=True, right_index=True)
    
    df = df.query('PBR != 0').copy()
    df['수익률'] = df['종가'] / df['시가']
    cond = (df['PER'] >= 2.5) & (df['PER'] <= 10)
    top30 = df[cond].sort_values('PBR').groupby('group').head(30)
    
    how = {
        '수익률' : np.mean
    }
    yoy = top30.groupby('group').agg(how)
    yoy.columns = [year]
    return yoy

dfs = [ ]
for date in range(2010, 2021):
    df = low_per_pbr(f"{date}")
    dfs.append(df)
    print(f"Processed year: {date}")
    print(df, "\n"+"="*40)
    time.sleep(1)

# 데이터 병합
df = pd.concat(dfs, axis=1)
print(df)

# 누적 수익률 계산
print(df.cumprod(axis=1))

# === step 3: 시각화 ===
df.cumprod(axis=1).transpose().plot.line()
plt.show()