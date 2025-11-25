import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrx import stock
import os
import warnings
warnings.filterwarnings('ignore')

# ==================
# === 볼린저 밴드 ===
# ==================

df = stock.get_index_ohlcv_by_date("20000101", "20001231", "1001")
df = df[['종가']]
std = df['종가'].rolling(20).std()
df['중심선'] = df['종가'].rolling(20).mean()
df['상단선'] = df['중심선'] + 2 * std
df['하단선'] = df['중심선'] - 2 * std
df.plot(figsize=(12,5))     # 16.1.2

df.loc["20000630":"20000830"].plot(figsize=(12,5))  # 16.1.3

# ==========================
# === 볼린저 밴드 백테스팅 ===
# ==========================

df = stock.get_market_ohlcv_by_date("20000101", "20191231", "005930")
df = df[['종가']]

std = df['종가'].rolling(20).std()
df['중심선'] = df['종가'].rolling(20).mean()
df['상단선'] = df['중심선'] + 2 * std
df['하단선'] = df['중심선'] - 2 * std

df['일간수익률'] = df['종가'].pct_change() + 1
df.loc[df['종가'] > df['상단선'], '매매신호'] = False
df.loc[df['종가'] < df['하단선'], '매매신호'] = True

df.loc[df['매매신호'].shift(1) == True, '보유여부'] = True
df.loc[df['매매신호'].shift(1) == False, '보유여부'] = False
df['보유여부'].ffill(inplace=True)
df['보유여부'].fillna(False, inplace=True)

print(df.iloc[20:].head())  # 16.2.1

df['보유수익률'] = df.loc[ df['보유여부'] == True, '일간수익률']
df['보유수익률'].fillna(1, inplace=True)
df['누적수익률'] = df['보유수익률'].cumprod()
print(df.tail())    # 16.2.2

df['단순보유수익률'] = df['일간수익률'].cumprod()
df[['단순보유수익률', '누적수익률']].plot(figsize=(12, 4))  # 16.2.3

delta = df.index[-1] - df.index[0]
year = delta.days/365
print(df['누적수익률'].iloc[-1] ** (1/year) )

# ==================================
# === 볼린저 밴드 백테스팅(전종목) ===
# ==================================

def 볼린저밴드(df, window=20):
    df = df[['종가']].copy()

    std = df['종가'].rolling(window).std()
    df['중심선'] = df['종가'].rolling(window).mean()
    df['상단선'] = df['중심선'] + 2 * std
    df['하단선'] = df['중심선'] - 2 * std

    df['일간수익률'] = df['종가'].pct_change() + 1
    df.loc[df['종가'] > df['상단선'], '매매신호'] = False
    df.loc[df['종가'] < df['하단선'], '매매신호'] = True

    df.loc[df['매매신호'].shift(1) == True, '보유여부'] = True
    df.loc[df['매매신호'].shift(1) == False, '보유여부'] = False
    df['보유여부'].ffill(inplace=True)
    df['보유여부'].fillna(False, inplace=True)

    df['보유수익률'] = df.loc[ df['보유여부'] == True, '일간수익률']
    df['보유수익률'].fillna(1, inplace=True)
    return df['보유수익률'].cumprod().iloc[-1]

df = stock.get_market_ohlcv_by_date("20000101", "20191231", "005930")
for window in range(5, 10):
    yeild = 볼린저밴드(df, window)
    print(window, yeild)

idx = [ ]
yeild = [ ]
file_list = os.listdir('./Finance_data/RSI_data')
for file in file_list:    
    df = pd.read_excel(f"./Finance_data/RSI_data/{file}")   
    cond = abs(df['종가'].pct_change()) > 0.3
    if len(df[cond]) != 0:
        continue
       
    val = 볼린저밴드(df, 9)
    idx.append(file.split(".")[0])
    yeild.append(val)
    
s = pd.Series(yeild, index=idx)    
print(s.describe())