import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import platform
import FinanceDataReader as fdr


# kospi = fdr.DataReader(symbol="KS11", start="2000")
# kospi.to_excel("./Finance_data/kospi.xlsx")

kospi = pd.read_excel("./Finance_data/kospi.xlsx", index_col=0)
kospi.head()

kospi.loc["2000-11"]

누적수익률 = 1

for year in range(2000, 2021):
    buy_mon = str(year) + "-11" # %Y-%m
    sell_mon = str(year+1) + "-04" # %Y-%m
    # print(buy_mon, sell_mon)

    매수가 = kospi.loc[buy_mon].iloc[0]["Open"]

    매도가 = kospi.loc[sell_mon].iloc[-1]["Close"]

    수익률 = 매도가/매수가
    누적수익률 = 누적수익률 * 수익률

print(누적수익률)

CAGR = (누적수익률 ** (1/21)) - 1
print(CAGR * 100)

단순보유누적수익률 = kospi.iloc[-1]["Close"] / kospi.iloc[0]["Open"]
단순보유CAGR = (단순보유누적수익률 ** (1/21)) - 1
print(단순보유CAGR)

start = datetime.datetime(year=2000, month=11, day=1)
end = start + relativedelta(months=5)

# print(start.strftime("%Y-%m"))
# print(end.strftime("%Y-%m"))

def 투자6개월(df, start_year=2000, end_year=2020, month=11):
    누적수익률 = 1
    for year in range(start_year, end_year):
        start = datetime.datetime(year=year, month=month, day=1)
        end = start + relativedelta(months=5)

        buy_mon = start.strftime("%Y-%m")
        sell_mon = end.strftime("%Y-%m")

        # print(buy_mon, sell_mon)

        매수가 = df.loc[buy_mon].iloc[0]["Open"]
        매도가 = df.loc[sell_mon].iloc[-1]["Close"]

        수익률 = 매도가/매수가
        누적수익률 = 누적수익률 * 수익률
    return 누적수익률

투자6개월(kospi, start_year=2000, end_year=2021, month=11)

for month in range(1, 12+1):
    ret = 투자6개월(kospi, start_year=2000, end_year=2021, month=month)
    print(f"{month:02} {ret:.2f}")

data = {}

for month in range(1, 12+1):
    ret = 투자6개월(kospi, start_year=2000, end_year=2021, month=month)
    data[month] = ret

# print(data)

if platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="Malgun Gothic")

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1)

ax.bar(
    x=list(data.keys()),
    height=list(data.values()),
    width=0.5,
    color="skyblue"
)

plt.title("코스피 투작 시작월별 6개월 성과(2020~)")
plt.xlabel("투자 시작월")
plt.ylabel("누적수익률")

plt.show()