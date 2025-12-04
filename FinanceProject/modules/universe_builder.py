# modules/universe_builder.py
import pandas as pd
import numpy as np
from pykrx import stock

def build_conservative_universe(classified_df):
    universe = []
    priority_classes = ['해외주식(시장)', '국내주식(코스피)', '국내채권', '대체자산']
    for asset_class in priority_classes:
        candidates = classified_df[classified_df['class'] == asset_class]
        if not candidates.empty:
            top_etf = candidates.sort_values(by='거래대금', ascending=False).iloc[0]
            universe.append(top_etf['ticker'])
    return universe

def build_aggressive_universe(classified_df, base_date, lookback_months, top_n, market_ticker, full_price_df, market_returns_df):
    """
    [최적화 버전] 사전 로드된 가격 데이터를 사용하여 공격형 유니버스를 구성합니다.
    """
    universe = []
    end_dt = pd.to_datetime(base_date)
    start_dt = end_dt - pd.DateOffset(months=lookback_months)

    attacker_classes = [c for c in classified_df['class'].unique() if '주식' in c and '기타' not in c]
    attack_candidates = classified_df[classified_df['class'].isin(attacker_classes)]
    
    # 미리 계산된 시장 수익률 데이터에서 필요한 기간만 슬라이싱
    market_returns = market_returns_df.loc[start_dt:end_dt]
    if market_returns.empty:
        return []

    stats = []
    for ticker in attack_candidates['ticker']:
        # full_price_df에 해당 티커가 있는지 먼저 확인
        if ticker not in full_price_df.columns:
            continue
            
        # 전체 가격 데이터에서 필요한 기간만 슬라이싱
        prices = full_price_df.loc[start_dt:end_dt, ticker].dropna()
        if len(prices) < (lookback_months * 20 * 0.8): continue

        returns = prices.pct_change().dropna()
        volatility = returns.std()
        
        common_index = market_returns.index.intersection(returns.index)
        if len(common_index) < 20: continue
        
        # 베타 계산
        beta = np.cov(returns[common_index], market_returns[common_index])[0, 1] / np.var(market_returns[common_index])
        stats.append({'ticker': ticker, 'vol': volatility, 'beta': beta})
            
    if not stats: return []
    
    stats_df = pd.DataFrame(stats)
    stats_df['vol_rank'] = stats_df['vol'].rank(ascending=False)
    stats_df['beta_rank'] = stats_df['beta'].rank(ascending=False)
    stats_df['total_rank'] = stats_df['vol_rank'] + stats_df['beta_rank']
    
    attackers = stats_df.sort_values(by='total_rank').head(top_n)['ticker'].tolist()
    universe.extend(attackers)

    # 수비수 선정 로직 (최적화)
    defender_classes = ['현금성자산', '국내채권']
    defend_candidates = classified_df[classified_df['class'].isin(defender_classes)]
    
    def_stats = []
    for ticker in defend_candidates['ticker']:
        if ticker not in full_price_df.columns:
            continue
            
        prices = full_price_df.loc[start_dt:end_dt, ticker].dropna()
        if len(prices) < (lookback_months * 20 * 0.8): continue
        
        returns = prices.pct_change().dropna()
        volatility = returns.std()
        def_stats.append({'ticker': ticker, 'vol': volatility})
            
    if def_stats:
      def_stats_df = pd.DataFrame(def_stats)
      defender = def_stats_df.sort_values(by='vol').iloc[0]['ticker']
      universe.append(defender)
    
    return universe

def build_neutral_universe(classified_df):
    universe = []
    factor_map = {
        '시장': ['국내주식(코스피)', '해외주식(시장)'],
        '성장': ['국내주식(코스닥)', '해외주식(기술주)'],
        '안정': ['국내주식(스타일)', '국내채권'],
        '인플레이션헷지': ['대체자산']
    }
    for factor, classes in factor_map.items():
        candidates = classified_df[classified_df['class'].isin(classes)]
        if not candidates.empty:
            top_etf = candidates.sort_values(by='거래대금', ascending=False).iloc[0]
            universe.append(top_etf['ticker'])
    return universe