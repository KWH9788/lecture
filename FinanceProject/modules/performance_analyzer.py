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

def calculate_turnover(monthly_portfolios, strategy):
    """
    포트폴리오 회전율 계산
    
    회전율 = (매월 매도된 종목 수 + 매수된 종목 수) / (2 × 총 리밸런싱 횟수)
    연평균 회전율 = 월평균 회전율 × 12
    
    Args:
        monthly_portfolios: {strategy: {month: [tickers]}}
        strategy: 전략 이름
    
    Returns:
        연평균 회전율 (%)
    """
    if strategy not in monthly_portfolios or not monthly_portfolios[strategy]:
        return 0.0
    
    portfolio_history = monthly_portfolios[strategy]
    sorted_months = sorted(portfolio_history.keys())
    
    if len(sorted_months) < 2:
        return 0.0
    
    total_turnover = 0
    rebalancing_count = 0
    
    for i in range(1, len(sorted_months)):
        prev_month = sorted_months[i-1]
        curr_month = sorted_months[i]
        
        prev_portfolio = set(portfolio_history[prev_month])
        curr_portfolio = set(portfolio_history[curr_month])
        
        # 매도된 종목 (이전에 있었지만 현재 없음)
        sold = len(prev_portfolio - curr_portfolio)
        
        # 매수된 종목 (현재 있지만 이전에 없음)
        bought = len(curr_portfolio - prev_portfolio)
        
        # 회전율 = (매도 + 매수) / (2 × 포트폴리오 크기)
        avg_portfolio_size = (len(prev_portfolio) + len(curr_portfolio)) / 2
        if avg_portfolio_size > 0:
            turnover = (sold + bought) / (2 * avg_portfolio_size)
            total_turnover += turnover
            rebalancing_count += 1
    
    if rebalancing_count == 0:
        return 0.0
    
    # 월평균 회전율
    monthly_avg_turnover = total_turnover / rebalancing_count
    
    # 연평균 회전율 (월평균 × 12)
    annual_turnover = monthly_avg_turnover * 12 * 100  # 백분율로 변환
    
    return annual_turnover

def analyze_performance(value_df, monthly_portfolios=None):
    """
    성과 지표 분석 (회전율 포함)
    
    Args:
        value_df: 포트폴리오 가치 데이터프레임
        monthly_portfolios: 월별 포트폴리오 구성 (회전율 계산용)
    """
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
        
        # 승률 계산 (양의 수익률 비율)
        win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100 if len(daily_ret) > 0 else 0
        
        result = {
            '누적수익률': f"{cum_return:.2%}",
            'CAGR': f"{cagr:.2%}",
            '최대낙폭(MDD)': f"{mdd:.2%}",
            '샤프지수': f"{sharpe_ratio:.2f}",
            '승률': f"{win_rate:.1f}%"
        }
        
        # 회전율 계산 (전략에만 적용, 벤치마크 제외)
        if monthly_portfolios and col in ['conservative', 'aggressive', 'neutral']:
            turnover = calculate_turnover(monthly_portfolios, col)
            result['연평균회전율'] = f"{turnover:.1f}%"
        
        results[col] = result
    
    return pd.DataFrame(results).T

def plot_cumulative_returns(value_df, filepath, rebalancing_dates=None):
    """
    전략별 개별 차트에 누적 수익률과 매매 시점 표시 (맞춤 벤치마크 포함)
    
    Args:
        value_df: 포트폴리오 가치 데이터프레임
        filepath: 저장 경로
        rebalancing_dates: 리밸런싱 날짜 리스트 (YYYYMMDD 형식)
    """
    # 첫 번째 유효한 값으로 정규화 (NaN 처리)
    normalized_df = pd.DataFrame()
    
    for col in value_df.columns:
        series = value_df[col].dropna()
        if len(series) > 0:
            first_value = series.iloc[0]
            if first_value > 0:  # 0으로 나누기 방지
                normalized_df[col] = value_df[col] / first_value
            else:
                print(f"  경고: {col}의 첫 값이 0 또는 유효하지 않습니다.")
    
    # 전략별 맞춤 벤치마크 매핑 (투자 시장 상황 반영)
    strategy_benchmarks = {
        'conservative': ['BM_MultiAsset', 'BM_KOSPI200'],      # 멀티 에셋(4개 자산군) vs 시장
        'aggressive': ['BM_HighVolatility', 'BM_KOSPI200'],    # 고변동성(코스닥) vs 시장
        'neutral': ['BM_FactorBalance', 'BM_KOSPI200']         # 팩터 균형 vs 시장
    }
    
    strategies = ['conservative', 'aggressive', 'neutral']
    
    # 리밸런싱 날짜를 datetime으로 변환
    rebal_dates = []
    if rebalancing_dates:
        for date_str in rebalancing_dates:
            try:
                rebal_dates.append(pd.to_datetime(date_str, format='%Y%m%d'))
            except:
                pass
    
    # 서브플롯 생성 (전략 3개)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 벤치마크 색상 및 스타일 정의 (투자 시장 상황별)
    bm_styles = {
        'BM_MultiAsset': {'color': '#9370DB', 'linestyle': '--', 'linewidth': 2, 'label': 'Multi-Asset (4자산군)'},
        'BM_HighVolatility': {'color': '#FF4500', 'linestyle': '-.', 'linewidth': 2, 'label': 'High Vol (KOSDAQ)'},
        'BM_FactorBalance': {'color': '#32CD32', 'linestyle': '--', 'linewidth': 2, 'label': 'Factor Balance'},
        'BM_KOSPI200': {'color': '#808080', 'linestyle': ':', 'linewidth': 1.5, 'label': 'KOSPI200', 'alpha': 0.7}
    }
    
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        
        if strategy not in normalized_df.columns:
            continue
        
        series = normalized_df[strategy].dropna()
        if len(series) == 0:
            continue
        
        # 전략 수익률 라인
        ax.plot(series.index, series.values, label=strategy.upper(), 
                linewidth=2.5, color='#2E86AB', zorder=3)
        
        # 전략별 맞춤 벤치마크 추가
        relevant_benchmarks = strategy_benchmarks.get(strategy, [])
        for bm in relevant_benchmarks:
            if bm in normalized_df.columns:
                bm_series = normalized_df[bm].dropna()
                if len(bm_series) > 0:
                    style = bm_styles.get(bm, {})
                    ax.plot(bm_series.index, bm_series.values, 
                           label=style.get('label', bm),
                           color=style.get('color', '#808080'),
                           linestyle=style.get('linestyle', '--'),
                           linewidth=style.get('linewidth', 1.5),
                           alpha=style.get('alpha', 0.8),
                           zorder=2)
        
        # 매매 시점 표시 (세로 점선)
        for rebal_date in rebal_dates:
            if rebal_date in series.index:
                ax.axvline(rebal_date, color='red', alpha=0.3, 
                          linestyle=':', linewidth=1, zorder=1)
        
        # 최종 수익률 및 벤치마크 대비 성과 계산
        final_return = (series.iloc[-1] - 1) * 100
        
        # 주요 벤치마크와 비교
        comparison_text = f"최종: {final_return:+.2f}%"
        primary_bm = relevant_benchmarks[0] if relevant_benchmarks else None
        if primary_bm and primary_bm in normalized_df.columns:
            bm_series = normalized_df[primary_bm].dropna()
            if len(bm_series) > 0:
                bm_return = (bm_series.iloc[-1] - 1) * 100
                outperformance = final_return - bm_return
                bm_label = bm_styles.get(primary_bm, {}).get('label', primary_bm)
                comparison_text += f" | vs {bm_label}: {outperformance:+.2f}%p"
        
        ax.set_title(f'{strategy.upper()} 전략 ({comparison_text})', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('날짜', fontsize=11)
        ax.set_ylabel('정규화된 가치 (초기=1.0)', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xlim(series.index[0], series.index[-1])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_cumulative_returns_combined(value_df, filepath):
    """
    기존 통합 차트 (모든 전략 + 벤치마크)
    """
    # 첫 번째 유효한 값으로 정규화 (NaN 처리)
    normalized_df = pd.DataFrame()
    
    for col in value_df.columns:
        series = value_df[col].dropna()
        if len(series) > 0:
            first_value = series.iloc[0]
            if first_value > 0:  # 0으로 나누기 방지
                normalized_df[col] = value_df[col] / first_value
            else:
                print(f"  경고: {col}의 첫 값이 0 또는 유효하지 않습니다.")
    
    plt.figure(figsize=(14, 8))
    for col in normalized_df.columns:
        series = normalized_df[col].dropna()
        if len(series) > 0:
            plt.plot(series.index, series.values, label=col, linewidth=2)
    
    plt.title('전략별 누적 수익률 시뮬레이션 (통합)', fontsize=16, fontweight='bold')
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('정규화된 가치 (초기 투자 = 1.0)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
    
    # 연도 개수 계산하여 높이 동적 조정
    num_strategies = len(monthly_df.columns)
    num_years = monthly_df_copy['Year'].nunique()
    
    # 각 서브플롯 높이: 연도당 0.5, 최소 2, 최대 4
    height_per_plot = max(2, min(4, num_years * 0.5))
    total_height = height_per_plot * num_strategies
    
    fig, axes = plt.subplots(num_strategies, 1, figsize=(14, total_height))
    if num_strategies == 1:
        axes = [axes]
    
    for idx, col in enumerate(monthly_df.columns):
        pivot = monthly_df_copy.pivot_table(values=col, index='Year', columns='Month')
        pivot = pivot * 100  # 백분율로 변환
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                    cbar_kws={'label': '수익률 (%)'}, ax=axes[idx],
                    vmin=-10, vmax=10, annot_kws={'fontsize': 9})
        axes[idx].set_title(f'{col} - 월별 수익률 히트맵', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('월', fontsize=10)
        axes[idx].set_ylabel('연도', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_drawdown_chart(value_df, filepath):
    """
    전략별 개별 드로우다운 차트 생성
    """
    strategies = ['conservative', 'aggressive', 'neutral']
    strategy_names = {
        'conservative': '안정추구형 (Conservative)',
        'aggressive': '공격투자형 (Aggressive)',
        'neutral': '위험중립형 (Neutral)'
    }
    strategy_colors = {
        'conservative': '#4682B4',
        'aggressive': '#DC143C',
        'neutral': '#32CD32'
    }
    
    # 벤치마크 컬럼 필터링
    available_strategies = [s for s in strategies if s in value_df.columns]
    
    if not available_strategies:
        print("경고: 전략 데이터를 찾을 수 없습니다.")
        return
    
    # 전략별 개별 차트 생성
    fig, axes = plt.subplots(len(available_strategies), 1, figsize=(14, 5 * len(available_strategies)))
    
    if len(available_strategies) == 1:
        axes = [axes]
    
    for idx, strategy in enumerate(available_strategies):
        ax = axes[idx]
        series = value_df[strategy].dropna()
        
        if len(series) > 0:
            rolling_max = series.cummax()
            drawdown = (series / rolling_max - 1) * 100
            
            # 드로우다운 영역 채우기
            ax.fill_between(drawdown.index, drawdown, 0, 
                           alpha=0.4, color=strategy_colors.get(strategy, '#808080'), 
                           label=f'{strategy_names.get(strategy, strategy)} Drawdown')
            
            # 드로우다운 선 그리기
            ax.plot(drawdown.index, drawdown, 
                   linewidth=2, color=strategy_colors.get(strategy, '#808080'))
            
            # MDD 표시
            mdd_value = drawdown.min()
            mdd_date = drawdown.idxmin()
            ax.scatter([mdd_date], [mdd_value], color='red', s=100, zorder=5, marker='v')
            ax.text(mdd_date, mdd_value - 2, f'MDD: {mdd_value:.2f}%', 
                   ha='center', va='top', fontsize=10, fontweight='bold', color='red')
            
            # 차트 설정
            ax.set_title(f'{strategy_names.get(strategy, strategy)} - 드로우다운 차트', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('드로우다운 (%)', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.legend(loc='lower left', fontsize=10)
            
            # y축 범위 조정
            if mdd_value < -5:
                ax.set_ylim([mdd_value * 1.2, 5])
    
    # 마지막 서브플롯에만 x축 레이블 추가
    axes[-1].set_xlabel('날짜', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_portfolio_composition_pies(monthly_portfolios, monthly_weights, filepath):
    """
    월별 포트폴리오 구성 파이 차트 생성 (실제 가중치 반영)
    
    Args:
        monthly_portfolios: {'conservative': {month: [tickers]}, ...}
        monthly_weights: {'conservative': {month: {ticker: weight}}, ...}
        filepath: 저장 경로
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
        weight_data = monthly_weights.get(strategy, {})
        
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
        fig.suptitle(f'{strategy_names[strategy]} 전략 - 월별 포트폴리오 구성 (가중치 반영)', 
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
            
            # 실제 가중치 사용 (없으면 동일 가중)
            if month in weight_data and weight_data[month]:
                weights = [weight_data[month].get(ticker, 1/len(tickers)) for ticker in tickers]
                # 가중치 합계 정규화
                weight_sum = sum(weights)
                if weight_sum > 0:
                    weights = [w / weight_sum for w in weights]
                else:
                    weights = [1/len(tickers)] * len(tickers)
            else:
                # 가중치 정보 없으면 동일 가중
                weights = [1/len(tickers)] * len(tickers)
            
            # Aggressive 전략: 안전자산 vs 공격자산 색상 구분
            if strategy == 'aggressive':
                colors = []
                safe_asset_keywords = ['채권', '국채', '회사채', '단기', '현금', '금리', 'bond', 'Bond']
                
                for ticker in tickers:
                    ticker_name = ticker_names.get(ticker, ticker)
                    # 안전자산 판별
                    is_safe = any(keyword in ticker_name for keyword in safe_asset_keywords)
                    
                    if is_safe:
                        colors.append('#90EE90')  # 연한 녹색 (안전자산)
                    else:
                        colors.append('#FF6B6B')  # 연한 빨강 (공격자산)
                
                # 범례 추가용 레이블
                labels_with_tag = []
                for i, ticker in enumerate(tickers):
                    ticker_name = ticker_names.get(ticker, ticker)
                    is_safe = any(keyword in ticker_name for keyword in safe_asset_keywords)
                    tag = '[안전]' if is_safe else '[공격]'
                    labels_with_tag.append(f'{tag} {labels[i]}')
                
                labels = labels_with_tag
            else:
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

def calculate_performance_attribution(value_df, monthly_portfolios):
    """
    월별/분기별 성과 분해 (Performance Attribution)
    
    각 기간별로 어떤 요인이 수익에 기여했는지 분석
    
    Args:
        value_df: 포트폴리오 가치 데이터프레임
        monthly_portfolios: {strategy: {month: [tickers]}}
    
    Returns:
        monthly_attribution: 월별 성과 분해 DataFrame
        quarterly_attribution: 분기별 성과 분해 DataFrame
    """
    strategies = ['conservative', 'aggressive', 'neutral']
    
    # 전략별 데이터만 추출
    strategy_values = value_df[[col for col in value_df.columns if col in strategies]]
    
    # 월별 수익률 계산
    monthly_returns = strategy_values.resample('M').last().pct_change().dropna()
    monthly_returns.index = monthly_returns.index.to_period('M')
    
    # 월별 성과 분해
    monthly_attr = []
    for strategy in strategies:
        if strategy not in monthly_returns.columns:
            continue
            
        for period in monthly_returns.index:
            month_return = monthly_returns.loc[period, strategy] * 100
            
            # 해당 월의 포트폴리오 구성 가져오기
            month_key = str(period).replace('-', '')[:6]
            portfolio = monthly_portfolios.get(strategy, {}).get(month_key, [])
            portfolio_size = len(portfolio)
            
            # 성과 기여 분류
            if month_return > 2:
                performance = '강세'
            elif month_return > 0:
                performance = '상승'
            elif month_return > -2:
                performance = '하락'
            else:
                performance = '급락'
            
            monthly_attr.append({
                '전략': strategy.upper(),
                '기간': str(period),
                '월수익률(%)': round(month_return, 2),
                '성과': performance,
                '보유종목수': portfolio_size
            })
    
    monthly_attribution = pd.DataFrame(monthly_attr)
    
    # 분기별 수익률 계산
    quarterly_returns = strategy_values.resample('Q').last().pct_change().dropna()
    quarterly_returns.index = quarterly_returns.index.to_period('Q')
    
    # 분기별 성과 분해
    quarterly_attr = []
    for strategy in strategies:
        if strategy not in quarterly_returns.columns:
            continue
            
        for period in quarterly_returns.index:
            quarter_return = quarterly_returns.loc[period, strategy] * 100
            
            # 분기 성과 분류
            if quarter_return > 5:
                performance = '강세'
            elif quarter_return > 0:
                performance = '상승'
            elif quarter_return > -5:
                performance = '하락'
            else:
                performance = '급락'
            
            quarterly_attr.append({
                '전략': strategy.upper(),
                '기간': str(period),
                '분기수익률(%)': round(quarter_return, 2),
                '성과': performance
            })
    
    quarterly_attribution = pd.DataFrame(quarterly_attr)
    
    return monthly_attribution, quarterly_attribution

def plot_performance_attribution(monthly_attr, quarterly_attr, filepath='attribution.png'):
    """
    성과 분해 시각화
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    strategies = ['CONSERVATIVE', 'AGGRESSIVE', 'NEUTRAL']
    strategy_colors = {
        'CONSERVATIVE': '#4682B4',  # 파란색
        'AGGRESSIVE': '#DC143C',    # 빨간색
        'NEUTRAL': '#32CD32'        # 녹색
    }
    
    # 상단: 월별 성과 분해
    ax1 = axes[0]
    x_positions = {}
    bar_width = 0.25
    
    for idx, strategy in enumerate(strategies):
        strategy_data = monthly_attr[monthly_attr['전략'] == strategy].copy()
        if strategy_data.empty:
            continue
        
        strategy_data = strategy_data.sort_values('기간')
        x_pos = np.arange(len(strategy_data)) * 3 + idx * bar_width
        x_positions[strategy] = x_pos
        
        # 전략별 고정 색상 사용
        strategy_color = strategy_colors[strategy]
        
        bars = ax1.bar(x_pos, strategy_data['월수익률(%)'], 
                      width=bar_width, label=strategy, color=strategy_color,
                      edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # 수익률 값 표시 (절대값 2% 이상만)
        for i, (x, y) in enumerate(zip(x_pos, strategy_data['월수익률(%)'])):
            if abs(y) >= 2:
                ax1.text(x, y, f'{y:.1f}%', ha='center', 
                        va='bottom' if y > 0 else 'top', 
                        fontsize=7, fontweight='bold')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('월별 기간', fontsize=12)
    ax1.set_ylabel('월 수익률 (%)', fontsize=12)
    ax1.set_title('월별 성과 분해 (Performance Attribution)', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # x축 라벨 설정
    if x_positions:
        first_strategy_data = monthly_attr[monthly_attr['전략'] == strategies[0]].sort_values('기간')
        ax1.set_xticks(np.arange(len(first_strategy_data)) * 3 + bar_width)
        ax1.set_xticklabels(first_strategy_data['기간'], rotation=45, ha='right', fontsize=9)
    
    # 하단: 분기별 성과 분해
    ax2 = axes[1]
    x_positions_q = {}
    
    for idx, strategy in enumerate(strategies):
        strategy_data = quarterly_attr[quarterly_attr['전략'] == strategy].copy()
        if strategy_data.empty:
            continue
        
        strategy_data = strategy_data.sort_values('기간')
        x_pos = np.arange(len(strategy_data)) * 3 + idx * bar_width
        x_positions_q[strategy] = x_pos
        
        # 전략별 고정 색상 사용
        strategy_color = strategy_colors[strategy]
        
        bars = ax2.bar(x_pos, strategy_data['분기수익률(%)'], 
                      width=bar_width, label=strategy, color=strategy_color,
                      edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # 수익률 값 표시
        for i, (x, y) in enumerate(zip(x_pos, strategy_data['분기수익률(%)'])):
            ax2.text(x, y, f'{y:.1f}%', ha='center', 
                    va='bottom' if y > 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('분기별 기간', fontsize=12)
    ax2.set_ylabel('분기 수익률 (%)', fontsize=12)
    ax2.set_title('분기별 성과 분해 (Quarterly Attribution)', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # x축 라벨 설정
    if x_positions_q:
        first_strategy_data_q = quarterly_attr[quarterly_attr['전략'] == strategies[0]].sort_values('기간')
        ax2.set_xticks(np.arange(len(first_strategy_data_q)) * 3 + bar_width)
        ax2.set_xticklabels(first_strategy_data_q['기간'], rotation=0, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# (통계 분석 함수들은 현재 메인에서 직접 호출되지 않으므로, 향후 확장을 위해 남겨둠)
def analyze_statistical_validity(stat_data, price_df):
    pass