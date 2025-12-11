# modules/backtest_engine.py
import pandas as pd
from tqdm import tqdm
from pykrx import stock
from . import data_loader, universe_builder, trading_rules
import os
import pickle
from datetime import datetime, timedelta

def run_backtest(start_date, end_date, config):
    # --- Step 1: 월별 유동성 풀(liquid_pool) 구성 및 필요 Ticker 취합 ---
    print("Step 1: 월별 유동성 풀 구성 및 필요 Ticker를 취합합니다...")
    all_days = pd.to_datetime(stock.get_previous_business_days(fromdate=start_date, todate=end_date))
    
    # 월별 첫 영업일 추출 (is_month_start 대신 월별로 그룹화)
    all_days_df = pd.DataFrame({'date': all_days})
    all_days_df['year_month'] = all_days_df['date'].dt.to_period('M')
    first_business_days = all_days_df.groupby('year_month')['date'].first()
    rebalancing_dates = first_business_days.dt.strftime('%Y%m%d').tolist()
    
    first_day_str = all_days[0].strftime('%Y%m%d')
    
    # 백테스팅 첫 날이 월초가 아니더라도, 강제로 리밸런싱 날짜에 추가
    if first_day_str not in rebalancing_dates:
        rebalancing_dates.insert(0, first_day_str)
        print(f" -> 백테스팅 시작일({first_day_str})을 첫 리밸런싱 날짜로 강제 지정합니다.")
    
    print(f" -> 리밸런싱 날짜: {len(rebalancing_dates)}개 ({rebalancing_dates})")

    monthly_liquid_pools = {} 
    all_needed_tickers = set()
    previous_liquid_df = None  # 이전 월 데이터 저장

    for base_date in tqdm(rebalancing_dates, desc="월별 유동성 풀 구성"):
        liquid_df = data_loader.get_liquid_etf_pool_robust(base_date, config.MIN_TRADING_VALUE)
        month = base_date[:6]
        
        # 데이터가 없으면 재시도 (전후 5일 범위)
        if liquid_df.empty:
            base_dt = pd.to_datetime(base_date)
            for offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                retry_date = (base_dt + pd.DateOffset(days=offset)).strftime('%Y%m%d')
                liquid_df = data_loader.get_liquid_etf_pool_robust(retry_date, config.MIN_TRADING_VALUE)
                if not liquid_df.empty:
                    break
        
        # 여전히 데이터가 없으면 직전 월 데이터 재사용
        if liquid_df.empty and previous_liquid_df is not None:
            liquid_df = previous_liquid_df.copy()
            print(f"\n  ⚠️  {month}월 데이터 없음 → 직전 월 데이터 재사용")
        elif liquid_df.empty:
            print(f"\n  ⚠️  {month}월 데이터 수집 실패 (건너뜀)")
            continue
        
        monthly_liquid_pools[month] = liquid_df
        all_needed_tickers.update(liquid_df['ticker'].tolist())
        previous_liquid_df = liquid_df  # 성공한 데이터 저장
    
    # 벤치마크 티커 추가
    all_needed_tickers.add(config.BM_KOSPI_TICKER)
    all_needed_tickers.add(config.BM_BOND_TICKER)
    all_needed_tickers.add(config.BM_SP500_TICKER)
    all_needed_tickers.add(config.BM_KOSDAQ_TICKER)
    all_needed_tickers.add(config.BM_LEVERAGE_TICKER)
    all_needed_tickers.add(config.BM_GOLD_TICKER)
    print(f" -> 총 {len(all_needed_tickers)}개의 ETF에 대한 데이터가 필요합니다.")

    # --- Step 2: 취합된 Ticker들의 OHLCV 가격 데이터 일괄 수집 ---
    print("\nStep 2: 필요한 모든 OHLCV 데이터를 한번에 수집합니다...")
    start_load_date = (pd.to_datetime(start_date) - pd.DateOffset(months=config.AGGRESSIVE_LOOKBACK_MONTHS)).strftime('%Y%m%d')
    
    # 캐시 파일 경로 생성 (상위 디렉토리 확인)
    cache_filename = f"ohlcv_cache_{start_load_date}_{end_date}_{len(all_needed_tickers)}.pkl"
    cache_filepath = os.path.join(config.CACHE_DIR, cache_filename)
    parent_cache_filepath = os.path.join("..", "cache", cache_filename)
    
    # 캐시 사용 여부 확인 (현재 디렉토리 및 상위 디렉토리)
    use_cached_data = False
    actual_cache_path = None
    
    # 우선 상위 디렉토리 캐시 확인
    if config.USE_CACHE and os.path.exists(parent_cache_filepath):
        actual_cache_path = parent_cache_filepath
    elif config.USE_CACHE and os.path.exists(cache_filepath):
        actual_cache_path = cache_filepath
    
    if actual_cache_path:
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(actual_cache_path))
        if cache_age.days < config.CACHE_EXPIRY_DAYS:
            try:
                print(f" -> 캐시 파일 발견: {cache_filename} (생성: {cache_age.days}일 전)")
                print(" -> 캐시에서 데이터를 불러오는 중...")
                with open(actual_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    price_df = cache_data['price_df']
                    failed_tickers = cache_data['failed_tickers']
                print(f" -> ✓ 캐시 데이터 로드 완료! ({len(price_df.columns)}개 컬럼)")
                use_cached_data = True
            except Exception as e:
                print(f" -> ⚠️ 캐시 로드 실패: {str(e)}")
                print(" -> 새로 데이터를 다운로드합니다.")
                use_cached_data = False
        else:
            print(f" -> 캐시 파일이 오래되었습니다. ({cache_age.days}일 전)")
            print(" -> 새로 데이터를 다운로드합니다.")
    
    # 캐시를 사용하지 않는 경우 데이터 수집
    if not use_cached_data:
        price_df = pd.DataFrame()
        failed_tickers = []

        for ticker in tqdm(all_needed_tickers, desc="OHLCV 데이터 수집"):
            try:
                ohlcv = stock.get_etf_ohlcv_by_date(start_load_date, end_date, ticker)
                if ohlcv.empty:
                    failed_tickers.append(ticker)
                    continue
                # 컬럼 이름에 티커를 접두사로 붙여 병합 (예: 069500_종가)
                ohlcv.columns = [f"{ticker}_{col}" for col in ohlcv.columns]
                price_df = pd.concat([price_df, ohlcv], axis=1)
            except Exception as e:
                failed_tickers.append(ticker)
                continue
        
        price_df.index = pd.to_datetime(price_df.index)
        print(f" -> OHLCV 데이터 수집 완료. (실패: {len(failed_tickers)}개)")
        
        if failed_tickers:
            print(f" -> 데이터 수집 실패 티커: {failed_tickers[:5]}{'...' if len(failed_tickers) > 5 else ''}")
        
        # 캐시 저장
        if config.USE_CACHE:
            try:
                os.makedirs(config.CACHE_DIR, exist_ok=True)
                cache_data = {
                    'price_df': price_df,
                    'failed_tickers': failed_tickers,
                    'start_load_date': start_load_date,
                    'end_date': end_date,
                    'tickers': list(all_needed_tickers)
                }
                with open(cache_filepath, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f" -> ✓ 캐시 저장 완료: {cache_filename}")
            except Exception as e:
                print(f" -> ⚠️ 캐시 저장 실패: {str(e)}")

    # --- Step 2.5: 벤치마크 데이터 재시도 ---
    benchmark_tickers = [
        config.BM_KOSPI_TICKER, 
        config.BM_BOND_TICKER,
        config.BM_SP500_TICKER,
        config.BM_KOSDAQ_TICKER,
        config.BM_LEVERAGE_TICKER,
        config.BM_GOLD_TICKER
    ]
    close_price_df = price_df[[col for col in price_df.columns if col.endswith('_종가')]].copy()
    close_price_df.columns = [col.replace('_종가', '') for col in close_price_df.columns]
    
    missing_benchmarks = [ticker for ticker in benchmark_tickers if ticker not in close_price_df.columns]
    
    if missing_benchmarks:
        print(f"\n벤치마크 데이터 누락 감지: {missing_benchmarks}")
        print("벤치마크 데이터 재수집 시도 중...")
        
        for ticker in missing_benchmarks:
            retry_count = 0
            max_retries = 3
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    print(f"  - {ticker} 재시도 ({retry_count + 1}/{max_retries})...", end=" ")
                    
                    # 다양한 날짜 범위로 시도
                    if retry_count == 0:
                        # 첫 시도: 원래 날짜 범위
                        ohlcv = stock.get_etf_ohlcv_by_date(start_load_date, end_date, ticker)
                    elif retry_count == 1:
                        # 두번째 시도: 백테스팅 기간만
                        ohlcv = stock.get_etf_ohlcv_by_date(start_date, end_date, ticker)
                    else:
                        # 세번째 시도: 티커 리스트 확인 후 최근 1년
                        recent_date = pd.to_datetime(end_date) - pd.DateOffset(years=1)
                        ohlcv = stock.get_etf_ohlcv_by_date(recent_date.strftime('%Y%m%d'), end_date, ticker)
                    
                    if not ohlcv.empty:
                        ohlcv.columns = [f"{ticker}_{col}" for col in ohlcv.columns]
                        ohlcv.index = pd.to_datetime(ohlcv.index)
                        price_df = pd.concat([price_df, ohlcv], axis=1)
                        
                        # close_price_df 업데이트
                        close_price_df = price_df[[col for col in price_df.columns if col.endswith('_종가')]].copy()
                        close_price_df.columns = [col.replace('_종가', '') for col in close_price_df.columns]
                        
                        print("✓ 성공")
                        success = True
                    else:
                        print(f"데이터 없음")
                        retry_count += 1
                        
                except Exception as e:
                    print(f"실패: {str(e)[:50]}")
                    retry_count += 1
            
            if not success:
                print(f"  ⚠️  {ticker} 데이터 수집 실패 (모든 재시도 소진)")

    # --- Step 3: 백테스팅 시뮬레이션 시작 ---
    print("\nStep 3: 백테스팅 시뮬레이션을 시작합니다...")
    
    # 월별 포트폴리오(유니버스) 사전 구성 (티커와 가중치 포함)
    monthly_portfolios = {'conservative': {}, 'aggressive': {}, 'neutral': {}}
    monthly_weights = {'conservative': {}, 'aggressive': {}, 'neutral': {}}  # 가중치 저장
    
    # 벤치마크 데이터 최종 확인
    if config.BM_KOSPI_TICKER not in close_price_df.columns:
        print(f"\n⚠️  경고: 벤치마크 티커 '{config.BM_KOSPI_TICKER}'의 데이터를 가져올 수 없습니다.")
        print("    공격형 전략의 베타 계산을 건너뜁니다.")
        market_returns_df = pd.Series(dtype=float)
    else:
        market_returns_df = close_price_df[config.BM_KOSPI_TICKER].pct_change().dropna()
        print(f"✓ 벤치마크 데이터 확인: {config.BM_KOSPI_TICKER} ({len(market_returns_df)}개 데이터)")

    for base_date in tqdm(rebalancing_dates, desc="월별 포트폴리오 구성"):
        month = base_date[:6]
        if month not in monthly_liquid_pools:
            print(f"\n⚠️ 경고: {month}월의 유동성 풀 데이터가 없습니다. (base_date: {base_date})")
            continue
            
        liquid_df = monthly_liquid_pools[month]
        liquid_df['class'] = liquid_df['name'].apply(data_loader.classify_etf_decision_tree)

        # 첫 달과 마지막 달에만 상세 로그 출력
        verbose = (base_date == rebalancing_dates[0] or base_date == rebalancing_dates[-1])
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{base_date}] 월별 포트폴리오 구성 상세")
            print(f"{'='*60}")
        
        cons_port = universe_builder.build_conservative_universe(liquid_df, verbose=verbose)
        aggr_port = universe_builder.build_aggressive_universe(
            liquid_df, base_date, config.AGGRESSIVE_LOOKBACK_MONTHS, 
            config.AGGRESSIVE_TOP_N, config.BM_KOSPI_TICKER,
            close_price_df, market_returns_df, config=config, verbose=verbose
        )
        neut_port = universe_builder.build_neutral_universe(liquid_df, verbose=verbose)
        
        monthly_portfolios['conservative'][month] = cons_port
        monthly_portfolios['aggressive'][month] = aggr_port
        monthly_portfolios['neutral'][month] = neut_port
        
        # 가중치도 계산하여 저장
        if cons_port:
            monthly_weights['conservative'][month] = universe_builder.calculate_portfolio_weights(
                'conservative', cons_port, close_price_df, config=config, verbose=False)
        if aggr_port:
            monthly_weights['aggressive'][month] = universe_builder.calculate_portfolio_weights(
                'aggressive', aggr_port, close_price_df, config=config, verbose=False)
        if neut_port:
            monthly_weights['neutral'][month] = universe_builder.calculate_portfolio_weights(
                'neutral', neut_port, close_price_df, config=config, verbose=False)
        
        if verbose:
            print(f"\n{'='*60}")
    
    # 포트폴리오 생성 결과 확인
    print(f"\n생성된 포트폴리오 월: {sorted(monthly_portfolios['conservative'].keys())}")

    # --- Step 4: 일별 가치 계산 ---
    print("\nStep 4: 일별 포트폴리오 가치를 계산합니다...")
    
    # TRADING_MODE에 따른 분기
    if config.TRADING_MODE == 'monthly':
        daily_returns = close_price_df.pct_change().fillna(0)
        portfolio_values = {}
        
        for strategy in ['conservative', 'aggressive', 'neutral']:
            value_series = pd.Series(index=daily_returns.index, dtype=float)
            value_series.iloc[0] = 100_000_000
            current_portfolio = []
            current_weights = {}  # 각 종목별 보유 금액 추적
            total_transaction_cost = 0  # 누적 거래 비용 추적
            total_slippage = 0  # 누적 슬리피지 추적
            
            for i in range(1, len(daily_returns.index)):
                prev_day = daily_returns.index[i-1]
                today = daily_returns.index[i]
                current_value = value_series.loc[prev_day]
                
                is_rebalancing_day = today.strftime('%Y%m%d') in rebalancing_dates

                # 리밸런싱 전: 기존 포트폴리오로 당일 수익률 먼저 반영
                if current_portfolio and all(ticker in daily_returns.columns for ticker in current_portfolio):
                    for ticker in current_portfolio:
                        if ticker in current_weights:
                            ret = daily_returns.loc[today, ticker]
                            current_weights[ticker] *= (1 + ret)
                    current_value = sum(current_weights.values())
                
                # 리밸런싱 날짜 확인
                if is_rebalancing_day:
                    month = today.strftime('%Y%m')
                    new_portfolio = monthly_portfolios[strategy].get(month, [])
                    
                    # 포트폴리오가 변경되거나 가중치 조정이 필요한 경우 거래비용 발생
                    portfolio_changed = set(current_portfolio) != set(new_portfolio)
                    
                    if portfolio_changed or new_portfolio:
                        # 목표 가중치 계산 (새 포트폴리오 기준)
                        if new_portfolio:
                            target_weights = universe_builder.calculate_portfolio_weights(
                                strategy, new_portfolio, close_price_df, 
                                config=config, verbose=verbose
                            )
                        else:
                            target_weights = {}
                        
                        # 변경된 종목 분석
                        removed = set(current_portfolio) - set(new_portfolio)  # 매도할 종목
                        added = set(new_portfolio) - set(current_portfolio)    # 매수할 종목
                        kept = set(current_portfolio) & set(new_portfolio)     # 유지할 종목
                        
                        # 1. 매도 거래 비용 및 슬리피지
                        sell_amount = sum(current_weights.get(ticker, 0) for ticker in removed)
                        sell_transaction = sell_amount * config.TRANSACTION_COST_RATE
                        sell_slippage = sell_amount * config.SLIPPAGE_RATE
                        sell_cost = sell_transaction + sell_slippage
                        
                        # 2. 리밸런싱으로 인한 조정 (유지 종목도 비중 조정)
                        total_value_after_sell = current_value - sell_cost
                        
                        if new_portfolio:
                            # 목표 금액 계산 (최적 가중치 기준)
                            target_amounts = {ticker: total_value_after_sell * weight 
                                            for ticker, weight in target_weights.items()}
                            
                            # 유지 종목의 조정 금액 (현재 금액 vs 목표 금액의 차이)
                            rebalance_amount = sum(abs(current_weights.get(ticker, 0) - target_amounts.get(ticker, 0)) 
                                                 for ticker in kept)
                            
                            # 신규 매수 금액
                            add_amount = sum(target_amounts.get(ticker, 0) for ticker in added)
                            
                            # 총 매수 금액
                            buy_amount = add_amount + rebalance_amount
                            
                            # 3. 매수 거래 비용 및 슬리피지
                            buy_transaction = buy_amount * config.TRANSACTION_COST_RATE
                            buy_slippage = buy_amount * config.SLIPPAGE_RATE
                            buy_cost = buy_transaction + buy_slippage
                            
                            # 4. 총 거래 비용
                            total_cost = sell_cost + buy_cost
                            total_transaction_cost += sell_transaction + buy_transaction
                            total_slippage += sell_slippage + buy_slippage
                        else:
                            total_cost = sell_cost
                            total_transaction_cost += sell_transaction
                            total_slippage += sell_slippage
                        
                        current_value -= total_cost
                    
                    current_portfolio = new_portfolio
                    
                    # 최적화된 가중치로 포트폴리오 재분배 (거래 비용 반영 후)
                    if current_portfolio:
                        # 전략별 최적 가중치 계산
                        optimal_weights = universe_builder.calculate_portfolio_weights(
                            strategy, current_portfolio, close_price_df, 
                            config=config
                        )
                        
                        # 가중치에 따라 자산 배분
                        current_weights = {ticker: current_value * weight 
                                         for ticker, weight in optimal_weights.items()}
                
                # 당일 최종 가치 저장
                value_series.loc[today] = current_value

            portfolio_values[strategy] = value_series.dropna()
            
            # 거래 비용 통계 출력
            final_value = value_series.iloc[-1]
            cost_ratio = (total_transaction_cost + total_slippage) / 100_000_000 * 100
            print(f"\n  [{strategy}] 거래 비용 통계:")
            print(f"    - 거래수수료: {total_transaction_cost:,.0f}원 ({total_transaction_cost/100_000_000*100:.3f}%)")
            print(f"    - 슬리피지: {total_slippage:,.0f}원 ({total_slippage/100_000_000*100:.3f}%)")
            print(f"    - 총 비용: {total_transaction_cost + total_slippage:,.0f}원 ({cost_ratio:.3f}%)")
            print(f"    - 최종 자산: {final_value:,.0f}원")
    
    elif config.TRADING_MODE == 'daily':
        print("\n'daily' 모드: 일별 분할 매매 로직을 실행합니다...")
        print("전략별 분할 매매 규칙:")
        print("  [Conservative] 이동평균선(50일/200일) 기반 2단계 매수")
        print("  [Aggressive]   볼린저밴드 돌파 기반 2단계 매수")
        print("  [Neutral]      ATR 변동성 기반 동적 포지션 조절\n")
        
        # 기술적 지표 사전 계산 (전체 기간)
        print("기술적 지표 계산 중...")
        tech_indicators = trading_rules.calculate_technical_indicators(close_price_df)
        print(f"  -> {len(tech_indicators)}개 종목 지표 계산 완료\n")
        
        daily_returns = close_price_df.pct_change().fillna(0)
        portfolio_values = {}
        
        for strategy in ['conservative', 'aggressive', 'neutral']:
            value_series = pd.Series(index=daily_returns.index, dtype=float)
            value_series.iloc[0] = 100_000_000
            
            portfolio_state = {
                'positions': {},  # {ticker: amount}
                'cash': 100_000_000,
                'target_portfolio': [],
                'target_weights': {}
            }
            
            total_transaction_cost = 0
            total_slippage = 0
            trade_count = 0
            
            for i in range(1, len(daily_returns.index)):
                prev_day = daily_returns.index[i-1]
                today = daily_returns.index[i]
                current_value = value_series.loc[prev_day]
                
                # 기존 포지션 가치 업데이트
                if portfolio_state['positions']:
                    for ticker in list(portfolio_state['positions'].keys()):
                        if ticker in daily_returns.columns:
                            ret = daily_returns.loc[today, ticker]
                            portfolio_state['positions'][ticker] *= (1 + ret)
                    current_value = sum(portfolio_state['positions'].values()) + portfolio_state['cash']
                
                # 월별 리밸런싱: 목표 포트폴리오 업데이트
                if today.strftime('%Y%m%d') in rebalancing_dates:
                    month = today.strftime('%Y%m')
                    new_target = monthly_portfolios[strategy].get(month, [])
                    
                    if new_target:
                        portfolio_state['target_portfolio'] = new_target
                        portfolio_state['target_weights'] = universe_builder.calculate_portfolio_weights(
                            strategy, new_target, close_price_df, config=config
                        )
                
                # 일별 매매 실행 (전략별 규칙 적용)
                if portfolio_state['target_portfolio']:
                    
                    if strategy == 'conservative':
                        # 안정형: 벤치마크(시장) 기반 2단계 진입
                        invest_ratio = trading_rules.execute_trade_conservative(
                            today, portfolio_state, tech_indicators, config.BM_KOSPI_TICKER
                        )
                        
                        for ticker in portfolio_state['target_portfolio']:
                            if ticker not in tech_indicators:
                                continue
                            
                            target_weight = portfolio_state['target_weights'].get(ticker, 0)
                            target_amount = current_value * target_weight * invest_ratio
                            current_amount = portfolio_state['positions'].get(ticker, 0)
                            diff = target_amount - current_amount
                            
                            # 거래 필요 여부 (1% 이상 차이)
                            if abs(diff) > current_value * 0.01:
                                trade_amount = diff * 0.3  # 30%씩 점진 매수
                                
                                if abs(trade_amount) > current_value * 0.005:
                                    # 거래 비용
                                    cost = abs(trade_amount) * (config.TRANSACTION_COST_RATE + config.SLIPPAGE_RATE)
                                    total_transaction_cost += abs(trade_amount) * config.TRANSACTION_COST_RATE
                                    total_slippage += abs(trade_amount) * config.SLIPPAGE_RATE
                                    
                                    # 포지션 업데이트
                                    if ticker not in portfolio_state['positions']:
                                        portfolio_state['positions'][ticker] = 0
                                    portfolio_state['positions'][ticker] += trade_amount
                                    portfolio_state['cash'] -= trade_amount + cost
                                    trade_count += 1
                    
                    elif strategy == 'aggressive':
                        # 공격형: 볼린저밴드 기반 2단계 진입
                        trades = trading_rules.execute_trade_aggressive(
                            today, portfolio_state, tech_indicators, portfolio_state['target_portfolio']
                        )
                        
                        for ticker, target_ratio in trades.items():
                            if ticker not in tech_indicators:
                                continue
                            
                            target_weight = portfolio_state['target_weights'].get(ticker, 0)
                            target_amount = current_value * target_weight * target_ratio
                            current_amount = portfolio_state['positions'].get(ticker, 0)
                            diff = target_amount - current_amount
                            
                            if abs(diff) > current_value * 0.01:
                                trade_amount = diff * 0.4  # 40%씩 점진 매수
                                
                                if abs(trade_amount) > current_value * 0.005:
                                    cost = abs(trade_amount) * (config.TRANSACTION_COST_RATE + config.SLIPPAGE_RATE)
                                    total_transaction_cost += abs(trade_amount) * config.TRANSACTION_COST_RATE
                                    total_slippage += abs(trade_amount) * config.SLIPPAGE_RATE
                                    
                                    if ticker not in portfolio_state['positions']:
                                        portfolio_state['positions'][ticker] = 0
                                    portfolio_state['positions'][ticker] += trade_amount
                                    portfolio_state['cash'] -= trade_amount + cost
                                    trade_count += 1
                    
                    elif strategy == 'neutral':
                        # 중립형: ATR 변동성 기반 동적 포지션
                        risk_adjust_ratio = trading_rules.execute_trade_neutral_atr(
                            today, portfolio_state['target_portfolio'], tech_indicators
                        )
                        
                        for ticker in portfolio_state['target_portfolio']:
                            if ticker not in tech_indicators:
                                continue
                            
                            target_weight = portfolio_state['target_weights'].get(ticker, 0)
                            target_amount = current_value * target_weight * risk_adjust_ratio
                            current_amount = portfolio_state['positions'].get(ticker, 0)
                            diff = target_amount - current_amount
                            
                            if abs(diff) > current_value * 0.01:
                                trade_amount = diff * 0.35  # 35%씩 점진 매수
                                
                                if abs(trade_amount) > current_value * 0.005:
                                    cost = abs(trade_amount) * (config.TRANSACTION_COST_RATE + config.SLIPPAGE_RATE)
                                    total_transaction_cost += abs(trade_amount) * config.TRANSACTION_COST_RATE
                                    total_slippage += abs(trade_amount) * config.SLIPPAGE_RATE
                                    
                                    if ticker not in portfolio_state['positions']:
                                        portfolio_state['positions'][ticker] = 0
                                    portfolio_state['positions'][ticker] += trade_amount
                                    portfolio_state['cash'] -= trade_amount + cost
                                    trade_count += 1
                
                # 당일 최종 가치 저장
                total_positions_value = sum(portfolio_state['positions'].values())
                value_series.loc[today] = total_positions_value + portfolio_state['cash']
            
            portfolio_values[strategy] = value_series.dropna()
            
            # 거래 비용 통계 출력
            final_value = value_series.iloc[-1]
            cost_ratio = (total_transaction_cost + total_slippage) / 100_000_000 * 100
            print(f"\n  [{strategy}] 거래 통계:")
            print(f"    - 총 거래 횟수: {trade_count}회")
            print(f"    - 거래수수료: {total_transaction_cost:,.0f}원 ({total_transaction_cost/100_000_000*100:.3f}%)")
            print(f"    - 슬리피지: {total_slippage:,.0f}원 ({total_slippage/100_000_000*100:.3f}%)")
            print(f"    - 총 비용: {total_transaction_cost + total_slippage:,.0f}원 ({cost_ratio:.3f}%)")
            print(f"    - 최종 자산: {final_value:,.0f}원")
            print(f"    - 보유 종목: {len(portfolio_state['positions'])}개")
            print(f"    - 잔여 현금: {portfolio_state['cash']:,.0f}원")

    # --- Step 5: 벤치마크 계산 및 최종 결과 반환 ---
    if not portfolio_values:
        return pd.DataFrame(), {}
        
    # 날짜 형식 변환 (YYYYMMDD 문자열 → datetime)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # 벤치마크 데이터를 백테스트 기간으로 필터링
    close_price_df_for_bm = close_price_df.loc[start_dt:end_dt]
    
    print(f"\n벤치마크 계산 기간: {close_price_df_for_bm.index[0].strftime('%Y-%m-%d')} ~ {close_price_df_for_bm.index[-1].strftime('%Y-%m-%d')}")
    print(f"벤치마크 데이터 포인트: {len(close_price_df_for_bm)}개")
    
    # === 전략별 맞춤 벤치마크 계산 ===
    
    # 1. BM_KOSPI200 (국내 시장 대표)
    if config.BM_KOSPI_TICKER in close_price_df_for_bm.columns:
        bm_stock_series = close_price_df_for_bm[config.BM_KOSPI_TICKER].dropna()
        
        if len(bm_stock_series) > 0:
            bm_stock_returns = bm_stock_series.pct_change().fillna(0)
            bm_stock_values = 100_000_000 * (1 + bm_stock_returns).cumprod()
            portfolio_values['BM_KOSPI200'] = bm_stock_values
            print(f"  ✓ BM_KOSPI200 계산 완료: {len(bm_stock_values)}개 데이터")
        else:
            print(f"  ⚠️  BM_KOSPI200 데이터가 백테스트 기간에 없습니다.")
    
    # 2. BM_MultiAsset (Conservative 전략용 - 실제 구성 반영: 해외주식+국내주식+채권+대체자산)
    if (config.BM_SP500_TICKER in close_price_df_for_bm.columns and 
        config.BM_KOSPI_TICKER in close_price_df_for_bm.columns and
        config.BM_BOND_TICKER in close_price_df_for_bm.columns and
        config.BM_GOLD_TICKER in close_price_df_for_bm.columns):
        
        sp500_ret = close_price_df_for_bm[config.BM_SP500_TICKER].pct_change().fillna(0)
        kospi_ret = close_price_df_for_bm[config.BM_KOSPI_TICKER].pct_change().fillna(0)
        bond_ret = close_price_df_for_bm[config.BM_BOND_TICKER].pct_change().fillna(0)
        gold_ret = close_price_df_for_bm[config.BM_GOLD_TICKER].pct_change().fillna(0)
        
        # 멀티 에셋 균등: 각 25% (Conservative 전략의 4개 자산군 균등 가중)
        multiasset_ret = (0.25 * sp500_ret + 0.25 * kospi_ret + 0.25 * bond_ret + 0.25 * gold_ret)
        multiasset_values = 100_000_000 * (1 + multiasset_ret).cumprod()
        portfolio_values['BM_MultiAsset'] = multiasset_values
        print(f"  ✓ BM_MultiAsset (Conservative용) 계산 완료: {len(multiasset_values)}개 데이터")
    
    # 3. BM_HighVolatility (Aggressive 전략용 - 고변동성 시장: 코스닥 중심)
    if config.BM_KOSDAQ_TICKER in close_price_df_for_bm.columns:
        kosdaq_series = close_price_df_for_bm[config.BM_KOSDAQ_TICKER].dropna()
        
        if len(kosdaq_series) > 0:
            kosdaq_ret = kosdaq_series.pct_change().fillna(0)
            kosdaq_values = 100_000_000 * (1 + kosdaq_ret).cumprod()
            portfolio_values['BM_HighVolatility'] = kosdaq_values
            print(f"  ✓ BM_HighVolatility (Aggressive용) 계산 완료: {len(kosdaq_values)}개 데이터")
    
    # 3-2. BM_MarketTrend (시장 추세 참조용)
    if config.BM_KOSPI_TICKER in close_price_df_for_bm.columns:
        kospi_ma200 = close_price_df_for_bm[config.BM_KOSPI_TICKER].rolling(window=200).mean()
        portfolio_values['BM_KOSPI_MA200'] = kospi_ma200
        print(f"  ✓ BM_MarketTrend (MA200) 계산 완료: {len(kospi_ma200)}개 데이터")
    
    # 4. BM_FactorBalance (Neutral 전략용 - 팩터 균형: 시장/성장/안정/헷지)
    if (config.BM_KOSPI_TICKER in close_price_df_for_bm.columns and
        config.BM_KOSDAQ_TICKER in close_price_df_for_bm.columns and
        config.BM_BOND_TICKER in close_price_df_for_bm.columns and
        config.BM_GOLD_TICKER in close_price_df_for_bm.columns):
        
        kospi_ret = close_price_df_for_bm[config.BM_KOSPI_TICKER].pct_change().fillna(0)
        kosdaq_ret = close_price_df_for_bm[config.BM_KOSDAQ_TICKER].pct_change().fillna(0)
        bond_ret = close_price_df_for_bm[config.BM_BOND_TICKER].pct_change().fillna(0)
        gold_ret = close_price_df_for_bm[config.BM_GOLD_TICKER].pct_change().fillna(0)
        
        # 팩터 균등 가중: 시장25% + 성장25% + 안정25% + 헷지25%
        factor_balanced_ret = (0.25 * kospi_ret + 0.25 * kosdaq_ret + 0.25 * bond_ret + 0.25 * gold_ret)
        factor_balanced_values = 100_000_000 * (1 + factor_balanced_ret).cumprod()
        portfolio_values['BM_FactorBalance'] = factor_balanced_values
        print(f"  ✓ BM_FactorBalance (Neutral용) 계산 완료: {len(factor_balanced_values)}개 데이터")
    
    return pd.DataFrame(portfolio_values), monthly_portfolios, monthly_weights, rebalancing_dates
