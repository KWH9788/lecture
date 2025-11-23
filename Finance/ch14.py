import os
import pandas as pd

# ===============
# === 마법공식 ===
# ===============

# 데이터 불러오기
xls = os.listdir("./Finance_data/magic_data")
dfs = []
for i in xls:
    df = pd.read_excel(f"./Finance_data/magic_data/{i}", index_col=0, dtype={"code": str})
    dfs.append(df)
df = pd.concat(dfs)

# 12월 결산 종목만 필터링
cond = df["day"] == "12월 결산"
df = df[cond]
cond = (df['roic'] >0) & (df['ev/ebitda'] > 0)
df2 = df[cond].copy()
print("="*20, "마법공식", "="*20)
print(df2.head())

# 통합순위를 정해서 상위 30개 종목 선택
df2["rank1"] = df2["roic"].rank(ascending=False)
df2['rank2'] = df2['ev/ebitda'].rank()
df2['rank'] = df2['rank1'] + df2['rank2']
print("\n", "="*20, "상위30개(일부)", "="*20)
print(df2.sort_values(by = 'rank').head(30).head(5))

# =====================
# === 신마법공식 1.0 ===
# =====================

# 데이터 불러오기
xls = os.listdir("./Finance_data/magic1_data")
dfs = []
for i in xls:
    df = pd.read_excel(f"./Finance_data/magic1_data/{i}", index_col=0, dtype={"code": str})
    dfs.append(df)
df = pd.concat(dfs)

# 12월 결산 종목만 필터링
cond = df['day'] == '12월 결산'
df = df[cond]
cond = (df['gp/a'] > 0) & (df['pbr'] > 0)
df2 = df[cond].copy()
print("="*20, "신마법공식 1.0", "="*20)
print(df2.head())

# 통합순위를 정해서 상위 30개 종목 선택
df2['rank1'] = df2['gp/a'].rank(ascending=False)    # gp/a는 높으면 1등
df2['rank2'] = df2['pbr'].rank()                    # pbr는 낮으면 1등
df2['rank'] = df2['rank1'] + df2['rank2']
df2.set_index('code', inplace=True)
print("\n", "="*20, "상위30개(일부)", "="*20)
print(df2.sort_values(by='rank').head(30).head(5))

# =====================
# === 신마법공식 2.0 ===
# =====================

# 데이터 불러오기
xls = os.listdir("./Finance_data/magic2_data")
dfs = []
for i in xls:
    df = pd.read_excel(f"./Finance_data/magic2_data/{i}", index_col=0, dtype={"code": str})
    dfs.append(df)
df = pd.concat(dfs)

index = int(df.shape[0] * 0.2)
cap_bound = df.sort_values(by='cap').iloc[index-1]['cap']
print(cap_bound)

# 12월 결산 종목만 필터링
cond = df ['day'] == "12월 결산"
df = df[cond]
cond = (df['gp/a'] > 0) & (df['pbr'] > 0)
df2 = df[cond].copy()
print("="*20, "신마법공식 2.0", "="*20)
print(df2.head())

# 통합순위를 정해서 상위 30개 종목 선택
df2['rank1'] = df2['gp/a'].rank(ascending=False)    # gp/a는 높으면 1등
df2['rank2'] = df2['pbr'].rank()                    # pbr는 낮으면 1등
df2['rank'] = df2['rank1'] + df2['rank2']
df2.set_index('code', inplace=True)

# 시가총액이 하위 20%에 해당하는 종목 중 30개의 종목 선택
df3 = df2.sort_values(by='rank')
cond = df3['cap'] <= cap_bound
print("\n", "="*20, "상위30개(일부)", "="*20)
print(df3[cond].head(30).head(5))