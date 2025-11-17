# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

# ===================
# === 저 PER 전략 ===
# ===================
# === step 1: 데이터 로드 ===

# 종목별 PER, PBR 데이터 로드
df_factor = pd.read_excel("./Finance_data/data_kosdaq_20210401_per.xlsx",
                          index_col=0,
                          usecols=[0, 1, 6, 8]  # 종목코드, 종목명, PER, PBR
                          )
# print(df_factor.head())
# print(df_factor.info())   # PER, PBR 컬럼 object로 인식

# 종목별 거래량 데이터 로드
df = pd.read_excel("./Finance_data/data_kosdaq_20210401_sise.xlsx",
                   index_col=0)
df_volume = df[["거래량"]]
# print(df_volume.shape)
# print(df_volume.head())

# 종목별 등락률 데이터 불러오기
df_change = pd.read_excel("./Finance_data/data_kosdaq_change_2021.xlsx",
                          index_col=0,
                          usecols=[0, 5])
# print(df_change.head())
# print(df_change.info())

# === step 2: 데이터 전처리 ===

# object 타입을 float 타입으로 변환
df_factor.replace("-", np.nan, inplace=True)  # "-" 값을 NaN으로 변경
# print(df_factor.head())
# print(df_factor.info())

# 데이터 병합
df2 = df_factor.join(df_volume)
# print(df2.head())
# print(df2.shape)
df3 = df2.join(df_change)
# print(df3.head())
# print(df3.shape)

# 거래량이 0인 종목 제거
cond = df3["거래량"] != 0
df4 = df3[cond].copy()
# print(df4.shape)

# PER 기준으로 오름차순 정렬
df5 = df4.sort_values(by="PER", ascending=True)
df5.reset_index(inplace=True)
# print(df5.head())

# 하위 30개 종목의 등락률 평균 계산
low_per30 = df5.iloc[:30]
# print(low_per30["등락률"].mean())

# 20개 그룹으로 나누기
df5["group"] = pd.cut(df5.index, bins = 20, labels=False)
# print(df5.head())

# 그룹별 등락률 평균 계산
df6 = df5.groupby("group")[["등락률"]].mean()
# print(df6)

# === step 3: 시각화 ===
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic') 
else:
    plt.rc('font', family='Malgun Gothic')

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

ax.bar(df6.index, df6["등락률"], width=0.5)
plt.title("PER 그룹별 수익률")
plt.xlabel("PER 그룹")
plt.ylabel("수익률")
plt.show()

# =========================
# === PBR + PER 콤보전략 ===
# =========================
# 종목별 PER, PBR, 거래량, 등락률이 포함된 데이터
# print(df4.head())

# PER 2.5 이상 10 이하 조건 필터링
cond = (df4['PER'] >= 2.5) & (df4['PER'] <= 10)
df5 = df4[cond].copy()
# print(df5.head())

# PBR 기준으로 오름차순 정렬 후 30개 종목 추출
df6 = df5.sort_values(by="PBR").iloc[:30]
# print(df6)
print("="*40)
print(df6.describe())
print("="*40)

print(df6[df6["등락률"] == df6["등락률"].min()])