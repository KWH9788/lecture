# modules/performance_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic') 
else:
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

def analyze_performance(value_df):
    results = {}
    for col in value_df.columns:
        series = value_df[col]
        cum_return = (series.iloc[-1] / series.iloc[0] - 1)
        days = (series.index[-1] - series.index[0]).days
        cagr = (series.iloc[-1] / series.iloc[0]) ** (365.25 / days) - 1 if days > 0 else 0
        rolling_max = series.cummax()
        drawdown = series / rolling_max - 1
        mdd = drawdown.min()
        daily_ret = series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (daily_ret.mean() / daily_ret.std()) if daily_ret.std() != 0 else 0
        
        results[col] = {
            '누적수익률': f"{cum_return:.2%}",
            'CAGR': f"{cagr:.2%}",
            '최대낙폭(MDD)': f"{mdd:.2%}",
            '샤프지수': f"{sharpe_ratio:.2f}"
        }
    return pd.DataFrame(results).T

def plot_cumulative_returns(value_df, filepath):
    normalized_df = value_df / value_df.iloc[0]
    
    plt.figure(figsize=(14, 8))
    for col in normalized_df.columns:
        plt.plot(normalized_df.index, normalized_df[col], label=col)
    
    plt.title('전략별 누적 수익률 시뮬레이션', fontsize=16)
    plt.xlabel('날짜')
    plt.ylabel('정규화된 가치 (Normalized Value)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

# (통계 분석 함수들은 현재 메인에서 직접 호출되지 않으므로, 향후 확장을 위해 남겨둠)
def analyze_statistical_validity(stat_data, price_df):
    pass