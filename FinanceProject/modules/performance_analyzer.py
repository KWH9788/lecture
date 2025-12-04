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

def analyze_monthly_performance(value_df):
    """
    월별 수익률을 계산하고 히트맵으로 시각화
    """
    monthly_returns = {}
    
    for col in value_df.columns:
        series = value_df[col]
        # 월말 기준 리샘플링
        monthly = series.resample('M').last()
        monthly_ret = monthly.pct_change().dropna()
        monthly_returns[col] = monthly_ret
    
    monthly_df = pd.DataFrame(monthly_returns)
    
    # 월별 통계
    stats = pd.DataFrame({
        '평균 월수익률': monthly_df.mean() * 100,
        '월수익률 표준편차': monthly_df.std() * 100,
        '최고 월수익률': monthly_df.max() * 100,
        '최저 월수익률': monthly_df.min() * 100,
        '양(+)의 월 비율': (monthly_df > 0).sum() / len(monthly_df) * 100
    })
    
    return monthly_df, stats

def plot_monthly_returns_heatmap(monthly_df, filepath):
    """
    월별 수익률 히트맵 생성
    """
    # 연도-월 형식으로 변환
    monthly_df_copy = monthly_df.copy()
    monthly_df_copy.index = monthly_df_copy.index.to_period('M')
    monthly_df_copy['Year'] = monthly_df_copy.index.year
    monthly_df_copy['Month'] = monthly_df_copy.index.month
    
    fig, axes = plt.subplots(len(monthly_df.columns), 1, figsize=(14, 4 * len(monthly_df.columns)))
    if len(monthly_df.columns) == 1:
        axes = [axes]
    
    for idx, col in enumerate(monthly_df.columns):
        pivot = monthly_df_copy.pivot_table(values=col, index='Year', columns='Month')
        pivot = pivot * 100  # 백분율로 변환
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                    cbar_kws={'label': '수익률 (%)'}, ax=axes[idx],
                    vmin=-10, vmax=10)
        axes[idx].set_title(f'{col} - 월별 수익률 히트맵', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('월')
        axes[idx].set_ylabel('연도')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_drawdown_chart(value_df, filepath):
    """
    드로우다운 차트 생성
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 상단: 누적 수익 곡선
    normalized_df = value_df / value_df.iloc[0]
    for col in normalized_df.columns:
        axes[0].plot(normalized_df.index, normalized_df[col], label=col, linewidth=2)
    axes[0].set_title('누적 수익률 곡선', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('정규화된 가치')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 하단: 드로우다운
    for col in value_df.columns:
        series = value_df[col]
        rolling_max = series.cummax()
        drawdown = (series / rolling_max - 1) * 100
        axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, label=col)
        axes[1].plot(drawdown.index, drawdown, linewidth=1.5)
    
    axes[1].set_title('드로우다운 (Drawdown)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('날짜')
    axes[1].set_ylabel('드로우다운 (%)')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_portfolio_composition_pies(monthly_portfolios, filepath):
    """
    월별 포트폴리오 구성 파이 차트 생성
    monthly_portfolios: {'conservative': {month: [tickers]}, 'aggressive': {...}, 'neutral': {...}}
    """
    from pykrx import stock
    
    strategies = ['conservative', 'aggressive', 'neutral']
    strategy_names = {'conservative': '안정추구형', 'aggressive': '공격투자형', 'neutral': '위험중립형'}
    
    # 티커 이름 캐시
    ticker_names = {}
    
    # 각 전략별로 월별 구성 분석
    for strategy in strategies:
        if strategy not in monthly_portfolios:
            continue
            
        monthly_data = monthly_portfolios[strategy]
        if not monthly_data:
            continue
        
        # 월별로 정렬
        sorted_months = sorted(monthly_data.keys())
        
        # 최대 12개월치만 표시 (3행 4열)
        display_months = sorted_months[-12:] if len(sorted_months) > 12 else sorted_months
        
        n_months = len(display_months)
        n_cols = 4
        n_rows = (n_months + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        fig.suptitle(f'{strategy_names[strategy]} 전략 - 월별 포트폴리오 구성', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, month in enumerate(display_months):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            tickers = monthly_data[month]
            if not tickers:
                ax.text(0.5, 0.5, '종목 없음', ha='center', va='center', fontsize=12)
                ax.set_title(f'{month[:4]}-{month[4:]}', fontsize=12)
                ax.axis('off')
                continue
            
            # 티커명 가져오기 (캐시 활용)
            labels = []
            for ticker in tickers:
                if ticker not in ticker_names:
                    try:
                        name = stock.get_etf_ticker_name(ticker)
                        # 이름이 너무 길면 축약
                        if len(name) > 15:
                            name = name[:13] + '..'
                        ticker_names[ticker] = name
                    except:
                        ticker_names[ticker] = ticker
                labels.append(ticker_names[ticker])
            
            # 동일 가중 가정
            weights = [1/len(tickers)] * len(tickers)
            colors = plt.cm.Set3(range(len(tickers)))
            
            wedges, texts, autotexts = ax.pie(weights, labels=labels, autopct='%1.0f%%',
                                                colors=colors, startangle=90,
                                                textprops={'fontsize': 8})
            
            # 텍스트 가독성 개선
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            for text in texts:
                text.set_fontsize(8)
            
            ax.set_title(f'{month[:4]}-{month[4:]}', fontsize=13, fontweight='bold', pad=10)
        
        # 빈 subplot 숨기기
        for idx in range(n_months, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0.01, 1, 0.99])
        strategy_filepath = filepath.replace('.png', f'_{strategy}.png')
        plt.savefig(strategy_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    → {strategy_names[strategy]} 파이 차트 저장 완료")

def plot_rolling_performance(value_df, window=60, filepath='rolling_performance.png'):
    """
    이동 평균 수익률 및 변동성 차트
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 일별 수익률 계산
    returns_df = value_df.pct_change().dropna()
    
    # 상단: 이동 평균 수익률 (연율화)
    rolling_returns = returns_df.rolling(window=window).mean() * 252 * 100
    for col in rolling_returns.columns:
        axes[0].plot(rolling_returns.index, rolling_returns[col], label=col, linewidth=2)
    
    axes[0].set_title(f'{window}일 이동 평균 수익률 (연율화)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('수익률 (%)')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # 하단: 이동 변동성 (연율화)
    rolling_vol = returns_df.rolling(window=window).std() * np.sqrt(252) * 100
    for col in rolling_vol.columns:
        axes[1].plot(rolling_vol.index, rolling_vol[col], label=col, linewidth=2)
    
    axes[1].set_title(f'{window}일 이동 변동성 (연율화)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('날짜')
    axes[1].set_ylabel('변동성 (%)')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# (통계 분석 함수들은 현재 메인에서 직접 호출되지 않으므로, 향후 확장을 위해 남겨둠)
def analyze_statistical_validity(stat_data, price_df):
    pass