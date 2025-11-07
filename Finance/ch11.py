import datetime
from pykrx import stock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

df = pd.read_excel("./Finance_data/229200.xlsx", index_col="날짜")
# print(df.head())

df['변동'] = df['고가'] - df['저가']
# print(df.head())

df["전일변동"] = df["변동"].shift(1)
# print(df.head())

df["목표가"] = df["시가"] + df["전일변동"] * 0.5
# print(df.head())

df["수익률"] = np.where(df["고가"] >= df["목표가"], df["종가"]/df["목표가"], 1)
# print(df.head())

df["누적수익률"] = df["수익률"].cumprod()
# print(df.tail())

delta = df.index[-1] - df.index[0]
year = delta.days / 365
CAGR = (df["누적수익률"].iloc[-1] ** (1/year)) - 1
print(f"CAGR: {CAGR * 100}")

if platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="Malgun Gothic")

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(df["누적수익률"], label="변동성돌파")
ax.plot(df["종가"] / df["종가"].iloc[0], label="단순보유")

plt.xlabel("날짜")
plt.ylabel("누적수익률(배)")
plt.grid(True, axis="y")
plt.legend()
plt.show()

df["전고점"] = df["누적수익률"].cummax()
df["DD"] = (1 - df["누적수익률"]/df["전고점"]) * 100
# print(df.tail())

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(df.index, df["DD"] * -1)
ax.fill_between(df.index, df["DD"] * -1, alpha=0.1)
plt.grid()
plt.show()

# print(df["DD"].max())

df["MA10"] = df["종가"].rolling(window=10).mean()
# print(df.head())

df["매매신호"] = df["시가"] > df["MA10"].shift(1)

df["수익률2"] = np.where((df["매매신호"] == 1) & (df["고가"] >= df["목표가"]),
                      df["종가"]/df["목표가"],
                      1)

df["누적수익률2"] = df["수익률2"].cumprod()

print(df["누적수익률2"].iloc[-1])

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(df["누적수익률"], label="VB")
ax.plot(df["누적수익률2"], label="VBM")

plt.xlabel("날짜")
plt.ylabel("누적수익률(배)")
plt.grid(True, axis="y")
plt.legend()
plt.show()

df["전고점2"] = df["누적수익률2"].cummax()
df["DD2"] = (1-df["누적수익률2"] / df["전고점2"]) * 100

print(df["DD2"].max())

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 1, 1)

ax.plot(df.index, df['DD'] * -1, label='VB')
ax.plot(df.index, df['DD2'] * -1, label='VBM')

ax.grid()
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend(loc='best')
plt.show()