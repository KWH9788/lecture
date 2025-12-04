# modules/trading_rules.py
import numpy as np
import pandas as pd

def calculate_technical_indicators(price_df):
    tech_indicators = {}
    for ticker in price_df.columns:
        prices = price_df[ticker].dropna()
        if prices.empty: continue
            
        highs = prices
        lows = prices
        
        tr1 = (highs - lows).abs()
        tr2 = (highs - prices.shift(1)).abs()
        tr3 = (lows - prices.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(window=14).mean()
        
        tech_indicators[ticker] = {
            'price': prices,
            'ma20': prices.rolling(window=20).mean(),
            'ma50': prices.rolling(window=50).mean(),
            'ma200': prices.rolling(window=200).mean(),
            'bollinger_upper': prices.rolling(window=20).mean() + 2 * prices.rolling(window=20).std(),
            'atr14': atr14,
            'atr100_avg': atr14.rolling(window=100).mean()
        }
    return tech_indicators

def execute_trade_conservative(today, portfolio_state, tech_indicators, bm_ticker):
    ma50 = tech_indicators[bm_ticker]['ma50'].get(today)
    ma200 = tech_indicators[bm_ticker]['ma200'].get(today)
    
    if pd.isna(ma50) or pd.isna(ma200): return 1.0
    
    # 하락 추세이고, 아직 포지션이 없다면 (1차 매수)
    if ma50 < ma200 and not portfolio_state['positions']:
        return 0.5 # 50%만 투자
    
    # 하락 추세이고, 포지션이 이미 있다면 (2차 매수 조건 확인)
    elif ma50 < ma200 and portfolio_state['positions']:
        price = tech_indicators[bm_ticker]['price'].get(today)
        ma20 = tech_indicators[bm_ticker]['ma20'].get(today)
        if pd.notna(price) and pd.notna(ma20) and price > ma20:
             return 1.0 # 20일선 돌파 시 나머지 현금 모두 투자
        else:
             return 0.0 # 조건 미충족 시 추가 매수 없음
    
    # 상승 추세이면 항상 100% 투자
    return 1.0

def execute_trade_aggressive(today, portfolio_state, tech_indicators, target_portfolio):
    trades_to_execute = {} # {'ticker': target_ratio}
    for ticker in target_portfolio:
        # 안전자산은 항상 100% 비중
        if '채권' in ticker or '금리' in ticker or '현금성' in ticker: # (이름 기반으로 안전자산 추정)
            trades_to_execute[ticker] = 1.0
            continue

        # 공격자산 처리
        # 1차 매수 (50%)
        if ticker not in portfolio_state['positions']:
            trades_to_execute[ticker] = 0.5
        # 2차 매수 조건: 볼린저밴드 상단 돌파 시
        else:
            price = tech_indicators[ticker]['price'].get(today)
            bollinger_upper = tech_indicators[ticker]['bollinger_upper'].get(today)
            if pd.notna(price) and pd.notna(bollinger_upper) and price > bollinger_upper:
                trades_to_execute[ticker] = 1.0 # 100% 비중으로 채우라는 신호
    return trades_to_execute

def execute_trade_neutral_atr(today, target_portfolio, tech_indicators):
    # 합성 ATR 계산
    current_atr_ratio, historical_avg_atr_ratio = 0, 0
    asset_count = 0
    for ticker in target_portfolio:
        if ticker not in tech_indicators: continue
        price = tech_indicators[ticker]['price'].get(today)
        atr = tech_indicators[ticker]['atr14'].get(today)
        atr_avg = tech_indicators[ticker]['atr100_avg'].get(today)
        if pd.notna(price) and price > 0 and pd.notna(atr) and pd.notna(atr_avg):
            current_atr_ratio += (atr / price)
            historical_avg_atr_ratio += (atr_avg / price)
            asset_count += 1
    
    if asset_count == 0 or historical_avg_atr_ratio == 0: return 1.0
    
    current_atr_ratio /= asset_count
    historical_avg_atr_ratio /= asset_count
    
    risk_factor = current_atr_ratio / historical_avg_atr_ratio
    if risk_factor > 1.25: return 0.5
    elif risk_factor > 1.0: return 0.7
    else: return 1.0
