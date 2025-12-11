# modules/universe_builder.py
import pandas as pd
import numpy as np
from pykrx import stock

def build_conservative_universe(classified_df, verbose=False):
    """
    안정추구형 전략: 각 자산군에서 유동성 최고 ETF 1개씩 선택
    """
    universe = []
    priority_classes = ['해외주식(시장)', '국내주식(코스피)', '국내채권', '대체자산']
    
    if verbose:
        print("\n  [Conservative 전략 실행]")
        print("  자산 배분 원칙: 각 자산군 대표 ETF 1개씩 (유동성 기준)")
    
    for asset_class in priority_classes:
        candidates = classified_df[classified_df['class'] == asset_class]
        if not candidates.empty:
            top_etf = candidates.sort_values(by='거래대금', ascending=False).iloc[0]
            universe.append(top_etf['ticker'])
            if verbose:
                print(f"    - {asset_class}: {top_etf['name']} ({top_etf['ticker']}) - 거래대금: {top_etf['거래대금']/1e8:.1f}억")
        elif verbose:
            print(f"    - {asset_class}: 후보 없음")
    
    if verbose:
        print(f"  → 최종 구성: {len(universe)}개 종목")
    
    return universe

def build_aggressive_universe(classified_df, base_date, lookback_months, top_n, market_ticker, full_price_df, market_returns_df, config=None, verbose=False):
    """
    [리스크 통제 버전] 이론 기반 표준 지표를 사용한 공격형 유니버스 구성
    
    핵심 로직 (학술 문헌에서 검증된 표준 지표):
    1. MA200 추세 필터 (Meb Faber, 2007: A Quantitative Approach to Tactical Asset Allocation)
       - 200일 이동평균은 가장 널리 검증된 추세 지표
    
    2. 변동성 급증 감지 (Ang, 2014: Asset Management)
       - 업계 표준: 평균 대비 1.2배 초과 시 리스크 신호
    
    3. 모멘텀 반전 (Jegadeesh & Titman, 1993)
       - 3개월 모멘텀이 음수면 추세 약화
       - 단기 모멘텀(1~3개월)이 예측력이 가장 높음
    
    방어 모드 조건 (2개 이상 신호 발생 시):
    - KOSPI < MA200 (추세 반전)
    - 변동성 > 평균 × 1.2 (시장 불안)
    - 3개월 수익률 < 0 (모멘텀 약화)
    """
    universe = []
    end_dt = pd.to_datetime(base_date)
    start_dt = end_dt - pd.DateOffset(months=lookback_months)
    
    if verbose:
        print(f"\n  [Aggressive 전략 실행]")
        print(f"  자산 배분 원칙: 변동성/베타 상위 종목 (다층 리스크 필터)")
    
    # ===== 이론 기반 표준 지표 적용 =====
    defensive_signals = 0
    signal_messages = []
    
    # Signal 1: MA200 추세 필터 (Meb Faber, 2007)
    ma200_bearish = False
    if market_ticker in full_price_df.columns:
        market_prices = full_price_df.loc[:end_dt, market_ticker].dropna()
        
        if len(market_prices) >= 200:
            ma200 = market_prices.rolling(window=200).mean().iloc[-1]
            current_price = market_prices.iloc[-1]
            
            if current_price < ma200:
                ma200_bearish = True
                defensive_signals += 1
                signal_messages.append(f"MA200 추세 반전 (현재: {current_price:.2f} < MA200: {ma200:.2f})")
    
    # Signal 2: 변동성 급증 감지 (Ang, 2014 - 업계 표준)
    volatility_spike = False
    if market_ticker in full_price_df.columns:
        market_prices = full_price_df.loc[:end_dt, market_ticker].dropna()
        
        if len(market_prices) >= 60:  # 최소 3개월 데이터
            recent_returns = market_prices.pct_change().dropna().tail(20)  # 최근 1개월
            historical_returns = market_prices.pct_change().dropna().tail(120)  # 과거 6개월
            
            if len(recent_returns) > 0 and len(historical_returns) > 20:
                recent_vol = recent_returns.std()
                historical_vol = historical_returns.std()
                
                # 업계 표준 임계값 사용
                vol_threshold = config.VOLATILITY_SPIKE_THRESHOLD if config else 1.2
                if recent_vol > historical_vol * vol_threshold:
                    volatility_spike = True
                    defensive_signals += 1
                    signal_messages.append(f"변동성 급증 (현재: {recent_vol:.4f} > 기준: {historical_vol * vol_threshold:.4f})")
    
    # Signal 3: 모멘텀 반전 (Jegadeesh & Titman, 1993)
    momentum_negative = False
    if market_ticker in full_price_df.columns:
        market_prices = full_price_df.loc[:end_dt, market_ticker].dropna()
        
        # 3개월 모멘텀 계산 (학술적으로 가장 검증된 기간)
        momentum_months = config.MOMENTUM_LOOKBACK_MONTHS if config else 3
        momentum_days = momentum_months * 20  # 약 60 영업일
        
        if len(market_prices) >= momentum_days:
            momentum_return = (market_prices.iloc[-1] / market_prices.iloc[-momentum_days] - 1)
            
            if momentum_return < 0:
                momentum_negative = True
                defensive_signals += 1
                signal_messages.append(f"모멘텀 약화 ({momentum_months}개월 수익률: {momentum_return:.2%})")
    
    # 방어 모드 판단 (2개 이상 신호 발생 시 - 보수적 접근)
    market_trend = 'bearish' if defensive_signals >= 2 else 'bullish'
    
    if verbose:
        if defensive_signals >= 2:
            print(f"    ⚠️ 방어 신호 감지: {defensive_signals}개 (임계값: 2개 이상)")
            for msg in signal_messages:
                print(f"      - {msg}")
            print(f"      → 방어 모드: 안전자산 비중 확대")
        else:
            print(f"    ✓ 시장 상태: 정상 (리스크 신호 {defensive_signals}개)")
            if signal_messages:
                print(f"      경고: {', '.join(signal_messages)}")
            print(f"      → 공격 모드: 모멘텀 종목 편입")
    
    attacker_classes = [c for c in classified_df['class'].unique() if '주식' in c and '기타' not in c]
    attack_candidates = classified_df[classified_df['class'].isin(attacker_classes)]
    
    # 미리 계산된 시장 수익률 데이터에서 필요한 기간만 슬라이싱
    if market_returns_df.empty:
        # 시장 데이터가 없으면 변동성만으로 선정
        stats = []
        for ticker in attack_candidates['ticker']:
            if ticker not in full_price_df.columns:
                continue
            prices = full_price_df.loc[start_dt:end_dt, ticker].dropna()
            if len(prices) < (lookback_months * 20 * 0.8):
                continue
            returns = prices.pct_change().dropna()
            volatility = returns.std()
            stats.append({'ticker': ticker, 'vol': volatility, 'beta': 1.0})  # 베타 기본값
        
        if not stats:
            return []
        
        stats_df = pd.DataFrame(stats)
        attackers = stats_df.nlargest(top_n, 'vol')['ticker'].tolist()
        universe.extend(attackers)
        
        # 수비수 선정
        defender_classes = ['현금성자산', '국내채권']
        defend_candidates = classified_df[classified_df['class'].isin(defender_classes)]
        
        def_stats = []
        for ticker in defend_candidates['ticker']:
            if ticker not in full_price_df.columns:
                continue
            prices = full_price_df.loc[start_dt:end_dt, ticker].dropna()
            if len(prices) < (lookback_months * 20 * 0.8):
                continue
            returns = prices.pct_change().dropna()
            volatility = returns.std()
            def_stats.append({'ticker': ticker, 'vol': volatility})
        
        if def_stats:
            def_stats_df = pd.DataFrame(def_stats)
            defender = def_stats_df.sort_values(by='vol').iloc[0]['ticker']
            universe.append(defender)
        
        return universe
    
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
    
    # ===== 시장 추세에 따른 포트폴리오 구성 =====
    defender_classes = ['현금성자산', '국내채권']
    defend_candidates = classified_df[classified_df['class'].isin(defender_classes)]
    
    # 수비수(안전자산) 후보 준비
    def_stats = []
    for ticker in defend_candidates['ticker']:
        if ticker not in full_price_df.columns:
            continue
            
        prices = full_price_df.loc[start_dt:end_dt, ticker].dropna()
        if len(prices) < (lookback_months * 20 * 0.8): continue
        
        returns = prices.pct_change().dropna()
        volatility = returns.std()
        def_stats.append({'ticker': ticker, 'vol': volatility})
    
    if market_trend == 'bullish':
        # 상승장: 공격수 위주 편입 (기존 로직)
        attackers = stats_df.sort_values(by='total_rank').head(top_n)['ticker'].tolist()
        universe.extend(attackers)
        
        # 수비수 1개 추가 (리스크 헷지)
        if def_stats:
            def_stats_df = pd.DataFrame(def_stats)
            defender = def_stats_df.sort_values(by='vol').iloc[0]['ticker']
            universe.append(defender)
        
        if verbose:
            print(f"    → 포트폴리오: 공격자산 {len(attackers)}개 + 안전자산 1개")
    
    else:  # market_trend == 'bearish'
        # 하락장: 안전자산 중심 편입
        
        # 1. 안전자산을 최소 50% 확보 (수비수 2~3개)
        safe_assets = []
        if def_stats:
            def_stats_df = pd.DataFrame(def_stats)
            # 변동성 낮은 순으로 최대 3개 선택
            safe_assets = def_stats_df.sort_values(by='vol').head(min(3, len(def_stats_df)))['ticker'].tolist()
            universe.extend(safe_assets)
        
        # 2. 공격 자산은 최소한만 편입 (top 1개만)
        if not stats_df.empty:
            best_attacker = stats_df.sort_values(by='total_rank').iloc[0]['ticker']
            universe.append(best_attacker)
        
        if verbose:
            print(f"    → 포트폴리오: 안전자산 {len(safe_assets)}개 + 공격자산 1개")
    
    return universe

def build_neutral_universe(classified_df, verbose=False):
    """
    위험중립형 전략: 팩터 기반 분산 투자
    """
    universe = []
    factor_map = {
        '시장': ['국내주식(코스피)', '해외주식(시장)'],
        '성장': ['국내주식(코스닥)', '해외주식(기술주)'],
        '안정': ['국내주식(스타일)', '국내채권'],
        '인플레이션헷지': ['대체자산']
    }
    
    if verbose:
        print("\n  [Neutral 전략 실행]")
        print("  팩터 분산 원칙: 시장/성장/안정/인플레이션헷지 균형")
    
    for factor, classes in factor_map.items():
        candidates = classified_df[classified_df['class'].isin(classes)]
        if not candidates.empty:
            top_etf = candidates.sort_values(by='거래대금', ascending=False).iloc[0]
            universe.append(top_etf['ticker'])
            if verbose:
                print(f"    - {factor}: {top_etf['name']} ({top_etf['ticker']}) - {top_etf['class']}")
        elif verbose:
            print(f"    - {factor}: 후보 없음")
    
    if verbose:
        print(f"  → 최종 구성: {len(universe)}개 종목")
    
    return universe

def calculate_portfolio_weights(strategy, tickers, price_df, config=None, verbose=False):
    """
    전략별 최적화된 가중치 계산 (이론 기반 표준 지표)
    
    Args:
        strategy: 'conservative', 'aggressive', 'neutral'
        tickers: 포트폴리오 티커 리스트
        price_df: 가격 데이터프레임 (전체 티커 포함)
        config: 설정 모듈 (lookback 기간 참조)
        verbose: 상세 로그 출력 여부
    
    Returns:
        dict: {ticker: weight}
    
    학술적 근거:
    - 역변동성 가중: Meucci (2005) - 6개월 ~ 1년 권장
    - 자산배분 비율: 60/40 (Markowitz), 80/20 (공격형)
    """
    if not tickers:
        return {}
    
    if verbose:
        print(f"\n  [가중치 계산: {strategy.upper()}]")
    
    # 공격형: 학술 표준에 기반한 자산배분
    if strategy == 'aggressive':
        weights = {}
        
        # 안전자산(채권/현금) 식별
        safe_asset_keywords = ['채권', '국채', '회사채', '단기', '현금', '금리', 'bond', 'Bond']
        safe_assets = [t for t in tickers if any(keyword in str(t) for keyword in safe_asset_keywords)]
        risk_assets = [t for t in tickers if t not in safe_assets]
        
        # 포트폴리오 구성에 따라 모드 구분
        # 안전자산 비중이 높으면 (≥ 50%) 방어 모드
        total_assets = len(safe_assets) + len(risk_assets)
        is_defensive_mode = (len(safe_assets) / total_assets >= 0.5) if total_assets > 0 else False
        
        if verbose:
            print(f"    - 안전자산: {len(safe_assets)}개 | 공격자산: {len(risk_assets)}개")
            print(f"    - 모드: {'방어 (40/60)' if is_defensive_mode else '공격 (80/20)'}")
        
        if is_defensive_mode:
            # 방어형: 40% 주식 / 60% 채권 (보수적 표준)
            # 출처: 전통적 자산배분 이론
            if risk_assets:
                risk_weight = 0.4 / len(risk_assets)
                for ticker in risk_assets:
                    weights[ticker] = risk_weight
            
            if safe_assets:
                safe_weight = 0.6 / len(safe_assets)
                for ticker in safe_assets:
                    weights[ticker] = safe_weight
        
        else:
            # 공격형: 80% 주식 / 20% 채권 (업계 표준)
            # 출처: 일반적인 공격형 포트폴리오
            if risk_assets:
                risk_weight = 0.8 / len(risk_assets)
                for ticker in risk_assets:
                    weights[ticker] = risk_weight
            
            if safe_assets:
                safe_weight = 0.2 / len(safe_assets)
                for ticker in safe_assets:
                    weights[ticker] = safe_weight
        
        # 예외 처리: 가중치 합이 1이 되도록 조정
        if weights:
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
        else:
            # 가중치가 없으면 균등 배분
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        
        if verbose:
            print(f"    - 가중치 합계: {sum(weights.values()):.2%}")
        
        return weights
    
    # 안정/중립형: 역변동성 가중치 (저위험 자산에 높은 비중)
    elif strategy in ['conservative', 'neutral']:
        # 학술 표준 lookback 기간 사용 (Meucci, 2005)
        lookback_months = config.VOLATILITY_LOOKBACK_MONTHS if config else 6
        
        # 최근 lookback_months 동안의 변동성 계산
        volatilities = {}
        
        if verbose:
            print(f"    - 역변동성 가중 (과거 {lookback_months}개월 기준 - Meucci 2005)")
        
        for ticker in tickers:
            if ticker not in price_df.columns:
                # 가격 데이터가 없으면 동일 가중치 부여
                volatilities[ticker] = 1.0
                continue
            
            # 최근 데이터만 사용
            recent_prices = price_df[ticker].dropna().tail(lookback_months * 20)
            
            if len(recent_prices) < 20:
                volatilities[ticker] = 1.0
                continue
            
            returns = recent_prices.pct_change().dropna()
            volatilities[ticker] = returns.std()
        
        if verbose:
            sorted_vols = sorted(volatilities.items(), key=lambda x: x[1])
            print(f"    - 변동성 범위: {sorted_vols[0][1]:.4f} ~ {sorted_vols[-1][1]:.4f}")
        
        # 역변동성 가중치 계산
        if all(vol == 0 for vol in volatilities.values()):
            # 모든 변동성이 0이면 동일 가중
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        else:
            # 변동성이 낮을수록 높은 가중치
            inv_vols = {ticker: 1.0 / vol if vol > 0 else 1e6 for ticker, vol in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            weights = {ticker: inv_vol / total_inv_vol for ticker, inv_vol in inv_vols.items()}
        
        if verbose:
            print(f"    - 가중치 합계: {sum(weights.values()):.2%}")
        
        return weights
    
    else:
        # 기본: 동일 가중
        return {ticker: 1.0 / len(tickers) for ticker in tickers}