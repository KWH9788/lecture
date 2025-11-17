# Import
from pykrx import stock
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import time
import datetime
from dateutil.relativedelta import relativedelta

# ================================
# === 유가증권시장 12개월 모멘텀 ===
# ================================

# 2019-11-01, 2020-09-29 데이터 불러오기
df1 = stock.get_market_ohlcv("20191101")
df2 = stock.get_market_ohlcv("20200929")
kospi = df1.join(df2, lsuffix="_l", rsuffix="_r")
# print(kospi)

# 모멘텀 계산
kospi['모멘텀'] = 100 * (kospi['종가_r'] - kospi['종가_l']) / kospi['종가_l']
kospi = kospi[['종가_l', '종가_r', '모멘텀']]
# 상위 20 종목 추출
kospi_momentum20 = kospi.sort_values('모멘텀', ascending=False).iloc[:20]
# print(kospi_momentum20)
# 컬럼 이름 변경
kospi_momentum20.rename(columns={"종가_l": "매수가", "종가_r": "매도가"}, inplace=True)
# print(kospi_momentum20)

# 2020-11-02, 2021-04-30 데이터 불러오기
df3 = stock.get_market_ohlcv("20201102")
df4 = stock.get_market_ohlcv("20210430")
pct_df = df3.join(df4, lsuffix="_l", rsuffix="_r")
# print(pct_df)

# 데이터 병합
pct_df = pct_df[['종가_l', '종가_r']]
kospi_momentum20_result = kospi_momentum20.join(pct_df)
# print(kospi_momentum20_result)

# 20종목의 CAGR 계산
kospi_momentum20_result['수익률'] = (kospi_momentum20_result['종가_r'] / kospi_momentum20_result['종가_l'])
수익률평균 = kospi_momentum20_result['수익률'].fillna(0).mean()
mom20_cagr = 수익률평균 ** (1 / 0.5) - 1    # 6개월
print("="*40)
print("20종목의 CAGR:", mom20_cagr * 100)   # 22.4%

# 코스피 지수 불러오기
df_ref = fdr.DataReader(
    symbol="KS11", start="2020-11-02", end="2021-04-30"
)
# print(df_ref)

CAGR = ((df_ref['Close'].iloc[-1] / df_ref['Close'].iloc[0]) ** (1 / 0.5)) - 1
print("동일 기간 코스피 지수의 CAGR:", CAGR * 100)

# ===========================
# === 대형주 12개월 모멘텀 ===
# ===========================

# 데이터 불러오기
df1 = stock.get_market_ohlcv("20191101", market = "ALL")
df2 = stock.get_market_ohlcv("20200929", market = "ALL")
all = df1.join(df2, lsuffix="_l", rsuffix="_r")
# print(all)

# 우선주 제외
all2 = all.filter(regex="0$", axis=0).copy()

# 모멘텀 계산
all2['모멘텀'] = 100 * (all2['종가_r'] - all2['종가_l']) / all2['종가_l']
all2 = all2[['모멘텀']]
# print(all2.head())

cap = stock.get_market_cap(date = "20200929", market = "ALL")
cap = cap[['시가총액']]
# print(cap.head())

# 데이터 병합
all3 = all2.join(other=cap)
# print(all3)

# 시가총액 상위 200개 대형주 필터링
big = all3.sort_values('시가총액', ascending=False).iloc[:200]
# print(big)

# 대형주 중 모멘텀 상위 20개 종목 추출
big_pct20 = big.sort_values('모멘텀', ascending=False).iloc[:20]
# print(big_pct20)

# 2020-11-02, 2021-10-15 데이터 불러오기
df3 = stock.get_market_ohlcv("20201102", market = "ALL")
df4 = stock.get_market_ohlcv("20211015", market = "ALL")

# 투자 기간 동안의 전종목 수익률 계산
pct_df = df3.join(df4, lsuffix="_l", rsuffix="_r")
pct_df['수익률'] = pct_df['종가_r'] / pct_df['종가_l']
pct_df = pct_df[['종가_l', '종가_r', '수익률']]
# print(pct_df)

# 데이터 병합
big_mom_result = big_pct20.join(pct_df)
# print(big_mom_result)

# 20종목의 약 1년 동안의 CAGR 계산
평균수익률 = big_mom_result['수익률'].fillna(0).mean()
big_mom_cagr = 평균수익률 ** (1 / 1) - 1
print("="*40)
print("약 1년간 20개 종목의 CAGR:", big_mom_cagr * 100)  # 1.17%
print("="*40)

# ====================================
# === 10년 상대 모멘텀 전략 백테스팅 ===
# ====================================

# 모멘텀 계산 시작월, 모멘텀 계산 종료월, 투자 시작월, 투자 종료월 계산 코드
# 투자 시작 시점
year = 2010
month = 11
# 실제 투자 기간: 6개월
period = 6

inv_start = f"{year}-{month}-01"
inv_start = datetime.datetime.strptime(inv_start, "%Y-%m-%d")
inv_end = inv_start + relativedelta(months=period-1)

mom_start = inv_start - relativedelta(months=12)
mom_end = inv_start - relativedelta(months=2)
# print(mom_start.strftime("%Y-%m"), mom_end.strftime("%Y-%m"), "=>", 
#       inv_start.strftime("%Y-%m"), inv_end.strftime("%Y-%m"))

# 첫 거래일을 반환하는 함수
def get_business_day(df, year, month, index=0):
    str_month = f"{year}-{month}"
    return df.loc[str_month].index[index]

df = fdr.DataReader(symbol='KS11')
# print(get_business_day(df, 2010, 1, 0))

def momentum(df, year=2010, month=11, period=12):
    # 투자 시작일, 종료일
    str_day = f"{year}-{month}-01"
    start = datetime.datetime.strptime(str_day, "%Y-%m-%d")
    end = start + relativedelta(months=period-1)

    inv_start = get_business_day(df, start.year, start.month, 0)   # 첫 번째 거래일의 종가
    inv_end = get_business_day(df, end.year, end.month, -1)
    inv_start = inv_start.strftime("%Y%m%d")
    inv_end = inv_end.strftime("%Y%m%d")
    # print(inv_start, inv_end)

    # 모멘텀 계산 시작일, 종료일
    end = start - relativedelta(months=2)     # 역추세 1개월 제외
    start = start - relativedelta(months=period)

    mom_start = get_business_day(df, start.year, start.month, 0)   # 첫 번째 거래일의 종가
    mom_end = get_business_day(df, end.year, end.month, -1)
    mom_start = mom_start.strftime("%Y%m%d")
    mom_end = mom_end.strftime("%Y%m%d")
    print(mom_start, mom_end, " | ", inv_start, inv_end)

    # 모멘텀 계산
    df1 = stock.get_market_ohlcv_by_ticker(mom_start)
    df2 = stock.get_market_ohlcv_by_ticker(mom_end)
    mon_df = df1.join(df2, lsuffix="l", rsuffix="r")
    mon_df['등락률'] = (mon_df['종가r'] - mon_df['종가l'])/mon_df['종가l']*100
    
    # 우선주 제외
    mon_df = mon_df.filter(regex="0$", axis=0)
    mon20 = mon_df.sort_values(by="등락률", ascending=False)[:20]
    mon20 = mon20[['등락률']]
    #print(mon20)

    # 투자 기간 수익률
    df3 = stock.get_market_ohlcv_by_ticker(inv_start)
    df4 = stock.get_market_ohlcv_by_ticker(inv_end)
    inv_df = df3.join(df4, lsuffix="l", rsuffix="r")

    inv_df['수익률'] = inv_df['종가r'] / inv_df['종가l']    # 수익률 = 매도가 / 매수가
    inv_df = inv_df[['수익률']]

    # join
    result_df = mon20.join(inv_df)
    result = result_df['수익률'].fillna(0).mean()
    return year, result

data = []
for year in range(2010, 2021):
    ret = momentum(df, year, month=11, period=6)
    data.append(ret)
    time.sleep(1)

ret_df = pd.DataFrame(data, columns=['year', 'yield'])
ret_df.set_index('year', inplace=True)
print(ret_df)

cum_yield = ret_df['yield'].cumprod()
print(cum_yield)

CAGR = cum_yield.iloc[-1] ** (1 / 11) - 1
print("="*40)
print("11년 동안의 CAGR:",CAGR * 100) # 9.94%

buy_price = df.loc["2010-11"].iloc[0, 0]
sell_price = df.loc["2021-04"].iloc[-1, 0]
kospi_yield = sell_price / buy_price
kospi_cagr = kospi_yield ** (1/11)-1
print("코스피 11년 동안의 CAGR:", kospi_cagr * 100) # 4.83%
print("="*40)