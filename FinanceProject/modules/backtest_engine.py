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
    
    all_needed_tickers.add(config.BM_KOSPI_TICKER)
    all_needed_tickers.add(config.BM_BOND_TICKER)
    print(f" -> 총 {len(all_needed_tickers)}개의 ETF에 대한 데이터가 필요합니다.")

    # --- Step 2: 취합된 Ticker들의 OHLCV 가격 데이터 일괄 수집 ---
    print("\nStep 2: 필요한 모든 OHLCV 데이터를 한번에 수집합니다...")
    start_load_date = (pd.to_datetime(start_date) - pd.DateOffset(months=config.AGGRESSIVE_LOOKBACK_MONTHS)).strftime('%Y%m%d')
    
    # 캐시 파일 경로 생성
    cache_filename = f"ohlcv_cache_{start_load_date}_{end_date}_{len(all_needed_tickers)}.pkl"
    cache_filepath = os.path.join(config.CACHE_DIR, cache_filename)
    
    # 캐시 사용 여부 확인
    use_cached_data = False
    if config.USE_CACHE and os.path.exists(cache_filepath):
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_filepath))
        if cache_age.days < config.CACHE_EXPIRY_DAYS:
            try:
                print(f" -> 캐시 파일 발견: {cache_filename} (생성: {cache_age.days}일 전)")
                print(" -> 캐시에서 데이터를 불러오는 중...")
                with open(cache_filepath, 'rb') as f:
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
    benchmark_tickers = [config.BM_KOSPI_TICKER, config.BM_BOND_TICKER]
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
    
    # 월별 포트폴리오(유니버스) 사전 구성
    monthly_portfolios = {'conservative': {}, 'aggressive': {}, 'neutral': {}}
    
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

        cons_port = universe_builder.build_conservative_universe(liquid_df)
        aggr_port = universe_builder.build_aggressive_universe(
            liquid_df, base_date, config.AGGRESSIVE_LOOKBACK_MONTHS, 
            config.AGGRESSIVE_TOP_N, config.BM_KOSPI_TICKER,
            close_price_df, market_returns_df
        )
        neut_port = universe_builder.build_neutral_universe(liquid_df)
        
        monthly_portfolios['conservative'][month] = cons_port
        monthly_portfolios['aggressive'][month] = aggr_port
        monthly_portfolios['neutral'][month] = neut_port
    
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
                    
                    # 포트폴리오가 변경된 경우에만 거래비용 및 슬리피지 발생
                    if set(current_portfolio) != set(new_portfolio):
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
                        # 전체 거래 규모 계산
                        # - 매도: removed 종목 전체
                        # - 매수: added 종목 전체
                        # - 조정: kept 종목의 목표 비중과 현재 비중 차이
                        total_value_after_sell = current_value - sell_cost
                        
                        if new_portfolio:
                            target_per_asset = total_value_after_sell / len(new_portfolio)
                            
                            # 유지 종목의 조정 금액 (비중 재조정)
                            rebalance_amount = sum(abs(current_weights.get(ticker, 0) - target_per_asset) for ticker in kept)
                            
                            # 매수 금액 (신규 종목 + 유지 종목 조정)
                            buy_amount = sum(target_per_asset for _ in added) + rebalance_amount
                            
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
                    
                    # 동일가중 포트폴리오로 재분배 (거래 비용 반영 후)
                    if current_portfolio:
                        weight_per_asset = current_value / len(current_portfolio)
                        current_weights = {ticker: weight_per_asset for ticker in current_portfolio}
                
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
        # (Phase 3 분할 매매 로직 - 현재는 미완성 상태이므로 Pass)
        # 이 부분은 Phase 3 최종 구현 시 채워넣어야 합니다.
        print(" -> 'daily' 모드는 현재 개발 중입니다. 'monthly' 모드로 실행하십시오.")
        portfolio_values = {} # 빈 결과 반환
        pass

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
    
    # 벤치마크 주식 (KOSPI200)
    if config.BM_KOSPI_TICKER in close_price_df_for_bm.columns:
        bm_stock_series = close_price_df_for_bm[config.BM_KOSPI_TICKER].dropna()
        
        if len(bm_stock_series) > 0:
            bm_stock_returns = bm_stock_series.pct_change().fillna(0)
            bm_stock_values = 100_000_000 * (1 + bm_stock_returns).cumprod()
            portfolio_values['BM_KOSPI200'] = bm_stock_values
            print(f"  ✓ BM_KOSPI200 계산 완료: {len(bm_stock_values)}개 데이터")
        else:
            print(f"  ⚠️  BM_KOSPI200 데이터가 백테스트 기간에 없습니다.")
        
        # 벤치마크 60/40 포트폴리오
        if config.BM_BOND_TICKER in close_price_df_for_bm.columns:
            bm_bond_series = close_price_df_for_bm[config.BM_BOND_TICKER].dropna()
            
            if len(bm_bond_series) > 0:
                bm_bond_returns = bm_bond_series.pct_change().fillna(0)
                # 날짜 인덱스를 정렬하여 align
                bm_60_40_returns = 0.6 * bm_stock_returns + 0.4 * bm_bond_returns
                bm_60_40_values = 100_000_000 * (1 + bm_60_40_returns).cumprod()
                portfolio_values['BM_60_40'] = bm_60_40_values
                print(f"  ✓ BM_60_40 계산 완료: {len(bm_60_40_values)}개 데이터")
            else:
                print(f"  ⚠️  BM_60_40: 채권 데이터가 백테스트 기간에 없습니다.")
        else:
            print(f"  ⚠️  경고: 벤치마크 채권 티커 '{config.BM_BOND_TICKER}'의 데이터가 없습니다.")
    else:
        print(f"  ⚠️  경고: 벤치마크를 계산할 수 없습니다. 티커 '{config.BM_KOSPI_TICKER}' 데이터 없음.")
    
    return pd.DataFrame(portfolio_values), monthly_portfolios
