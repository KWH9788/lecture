# modules/backtest_engine.py
import pandas as pd
from tqdm import tqdm
from pykrx import stock
from . import data_loader, universe_builder, trading_rules
import os

def run_backtest(start_date, end_date, config):
    # --- Step 1: 월별 유동성 풀(liquid_pool) 구성 및 필요 Ticker 취합 ---
    print("Step 1: 월별 유동성 풀 구성 및 필요 Ticker를 취합합니다...")
    all_days = pd.to_datetime(stock.get_previous_business_days(fromdate=start_date, todate=end_date))
    rebalancing_dates = all_days[all_days.is_month_start].strftime('%Y%m%d').tolist()
    first_day_str = all_days[0].strftime('%Y%m%d')
    
    # 백테스팅 첫 날이 월초가 아니더라도, 강제로 리밸런싱 날짜에 추가
    if first_day_str not in rebalancing_dates:
        rebalancing_dates.insert(0, first_day_str)
        print(f" -> 백테스팅 시작일({first_day_str})을 첫 리밸런싱 날짜로 강제 지정합니다.")

    monthly_liquid_pools = {} 
    all_needed_tickers = set()

    for base_date in tqdm(rebalancing_dates, desc="월별 유동성 풀 구성"):
        liquid_df = data_loader.get_liquid_etf_pool_robust(base_date, config.MIN_TRADING_VALUE)
        if liquid_df.empty: continue
        month = base_date[:6]
        monthly_liquid_pools[month] = liquid_df
        all_needed_tickers.update(liquid_df['ticker'].tolist())
    
    all_needed_tickers.add(config.BM_KOSPI_TICKER)
    all_needed_tickers.add(config.BM_BOND_TICKER)
    print(f" -> 총 {len(all_needed_tickers)}개의 ETF에 대한 데이터가 필요합니다.")

    # --- Step 2: 취합된 Ticker들의 OHLCV 가격 데이터 일괄 수집 ---
    print("\nStep 2: 필요한 모든 OHLCV 데이터를 한번에 수집합니다...")
    start_load_date = (pd.to_datetime(start_date) - pd.DateOffset(months=config.AGGRESSIVE_LOOKBACK_MONTHS)).strftime('%Y%m%d')
    price_df = pd.DataFrame()

    for ticker in tqdm(all_needed_tickers, desc="OHLCV 데이터 수집"):
        try:
            ohlcv = stock.get_etf_ohlcv_by_date(start_load_date, end_date, ticker)
            # 컬럼 이름에 티커를 접두사로 붙여 병합 (예: 069500_종가)
            ohlcv.columns = [f"{ticker}_{col}" for col in ohlcv.columns]
            price_df = pd.concat([price_df, ohlcv], axis=1)
        except Exception:
            continue
    price_df.index = pd.to_datetime(price_df.index)
    print(" -> OHLCV 데이터 수집 완료.")

    # --- Step 3: 백테스팅 시뮬레이션 시작 ---
    print("\nStep 3: 백테스팅 시뮬레이션을 시작합니다...")
    
    # 월별 포트폴리오(유니버스) 사전 구성
    monthly_portfolios = {'conservative': {}, 'aggressive': {}, 'neutral': {}}
    
    close_price_df = price_df[[col for col in price_df.columns if col.endswith('_종가')]].copy()
    close_price_df.columns = [col.replace('_종가', '') for col in close_price_df.columns]
    market_returns_df = close_price_df[config.BM_KOSPI_TICKER].pct_change().dropna()

    for base_date in tqdm(rebalancing_dates, desc="월별 포트폴리오 구성"):
        month = base_date[:6]
        if month not in monthly_liquid_pools: continue
            
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

                # 리밸런싱 날짜 확인
                if today.strftime('%Y%m%d') in rebalancing_dates:
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
                        # 유지 종목의 현재 가치
                        kept_value = sum(current_weights.get(ticker, 0) for ticker in kept)
                        
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
                    
                    # 동일가중 포트폴리오로 재분배 (슬리피지 반영된 가격으로)
                    if current_portfolio:
                        weight_per_asset = current_value / len(current_portfolio)
                        current_weights = {ticker: weight_per_asset for ticker in current_portfolio}

                # 포트폴리오가 없거나 데이터가 없는 경우
                if not current_portfolio or not all(ticker in daily_returns.columns for ticker in current_portfolio):
                    value_series.loc[today] = current_value
                    continue

                # 각 종목별로 수익률 반영하여 가치 계산
                new_value = 0
                for ticker in current_portfolio:
                    if ticker in current_weights:
                        ret = daily_returns.loc[today, ticker]
                        new_value += current_weights[ticker] * (1 + ret)
                
                value_series.loc[today] = new_value
                
                # 다음 날을 위해 current_weights 업데이트
                if current_portfolio:
                    for ticker in current_portfolio:
                        if ticker in current_weights:
                            ret = daily_returns.loc[today, ticker]
                            current_weights[ticker] *= (1 + ret)

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
        
    close_price_df_for_bm = close_price_df.loc[start_date:end_date]
        
    bm_stock_returns = close_price_df_for_bm[config.BM_KOSPI_TICKER].pct_change().fillna(0)
    bm_stock_values = 100_000_000 * (1 + bm_stock_returns).cumprod()
    portfolio_values['BM_KOSPI200'] = bm_stock_values
    
    if config.BM_BOND_TICKER in close_price_df.columns:
        bm_bond_returns = close_price_df_for_bm[config.BM_BOND_TICKER].pct_change().fillna(0)
        bm_60_40_returns = 0.6 * bm_stock_returns + 0.4 * bm_bond_returns
        bm_60_40_values = 100_000_000 * (1 + bm_60_40_returns).cumprod()
        portfolio_values['BM_60_40'] = bm_60_40_values
    
    return pd.DataFrame(portfolio_values), monthly_portfolios
