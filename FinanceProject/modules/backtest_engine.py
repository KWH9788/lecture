# modules/backtest_engine.py
import pandas as pd
from tqdm import tqdm
from pykrx import stock
import os
from . import data_loader, universe_builder

def run_backtest(start_date, end_date, config):
    # --- [캐싱 로직] 데이터 파일 경로 설정 ---
    CACHE_DIR = "cache"
    PRICE_CACHE_FILE = os.path.join(CACHE_DIR, f"price_data_{start_date}_{end_date}.parquet")
    
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # --- Step 0: 데이터 로딩 (캐시 확인 후 결정) ---
    if os.path.exists(PRICE_CACHE_FILE):
        # [캐시 히트!] 파일이 존재하면, API 호출 없이 즉시 파일을 읽어온다.
        print(f"'{PRICE_CACHE_FILE}' 에서 캐시된 가격 데이터를 로딩합니다...")
        full_price_df = pd.read_parquet(PRICE_CACHE_FILE)
        print("데이터 로딩 완료.")

    else:
        # [캐시 미스!] 파일이 없으면, 기존의 API 호출 로직을 수행한다.
        print("캐시된 데이터가 없습니다. API를 통해 신규 데이터를 수집합니다 (시간 소요)...")
        
        # [최적화 1단계] 전체 기간에 대한 모든 ETF 리스트 선(先)취합
        print("Step 0.1: 전체 기간의 ETF 목록을 사전 취합합니다...")
        all_historical_tickers = set()
        all_years = range(int(start_date[:4]), int(end_date[:4]) + 1)
        for year in tqdm(all_years, desc="연도별 ETF 목록 취합"):
            first_day_of_year = stock.get_nearest_business_day_in_a_week(f"{year}0101")
            try:
                tickers_of_year = stock.get_etf_ticker_list(first_day_of_year)
                all_historical_tickers.update(tickers_of_year)
            except Exception:
                continue
        all_historical_tickers.add(config.BM_KOSPI_TICKER)
        all_historical_tickers.add(config.BM_BOND_TICKER)

        # [최적화 2단계] 필요한 모든 가격 데이터를 한번에 사전 로딩
        print("\nStep 0.2: 필요한 모든 가격 데이터를 한번에 사전 로딩합니다...")
        full_price_df = pd.DataFrame()
        start_load_date = (pd.to_datetime(start_date) - pd.DateOffset(months=config.AGGRESSIVE_LOOKBACK_MONTHS)).strftime('%Y%m%d')
        
        for ticker in tqdm(all_historical_tickers, desc="전체 가격 데이터 로딩"):
            try:
                prices = stock.get_etf_ohlcv_by_date(start_load_date, end_date, ticker)['종가']
                full_price_df[ticker] = prices
            except Exception:
                continue
        
        # [캐시 저장!] 로딩이 끝나면, 다음 실행을 위해 데이터를 파일로 저장한다.
        print(f"\n데이터 수집 완료. 다음 실행을 위해 '{PRICE_CACHE_FILE}' 파일에 데이터를 저장합니다.")
        full_price_df.to_parquet(PRICE_CACHE_FILE)

    full_price_df.index = pd.to_datetime(full_price_df.index)
    market_returns_df = full_price_df[config.BM_KOSPI_TICKER].pct_change().dropna()

    all_days = pd.to_datetime(stock.get_previous_business_days(fromdate=start_date, todate=end_date))
    rebalancing_dates = all_days[all_days.is_month_start].strftime('%Y%m%d').tolist()

    monthly_portfolios = {'conservative': {}, 'aggressive': {}, 'neutral': {}}
    all_tickers = set()

    print("Step 1: 월별 포트폴리오 구성 시작...")
    for base_date in tqdm(rebalancing_dates, desc="월별 유니버스 생성"):
        liquid_df = data_loader.get_liquid_etf_pool_robust(base_date, config.MIN_TRADING_VALUE)
        if liquid_df.empty: continue
        
        liquid_df['class'] = liquid_df['name'].apply(data_loader.classify_etf_decision_tree)
        
        cons_port = universe_builder.build_conservative_universe(liquid_df)
        aggr_port = universe_builder.build_aggressive_universe(liquid_df, base_date, config.AGGRESSIVE_LOOKBACK_MONTHS, config.AGGRESSIVE_TOP_N, config.BM_KOSPI_TICKER,
            full_price_df, market_returns_df # 사전 로딩된 데이터 전달
        )
        neut_port = universe_builder.build_neutral_universe(liquid_df)
        
        month = base_date[:6]
        monthly_portfolios['conservative'][month] = cons_port
        monthly_portfolios['aggressive'][month] = aggr_port
        monthly_portfolios['neutral'][month] = neut_port
        
        all_tickers.update(cons_port, aggr_port, neut_port)

    # [수정] Step 2: 가격 데이터 수집 단계는 이미 완료되었으므로 생략 또는 단순 참조
    print("\nStep 2: 사전 로딩된 가격 데이터 참조...")
    price_df = full_price_df.loc[start_date:end_date]

    print("\nStep 3: 일별 수익률 및 포트폴리오 가치 계산...")
    daily_returns = price_df.pct_change().fillna(0)
    portfolio_values = {}

    for strategy in ['conservative', 'aggressive', 'neutral']:
        value_series = pd.Series(index=daily_returns.index, dtype=float)
        value_series.iloc[0] = 100_000_000
        current_portfolio = []
        
        for i in range(1, len(daily_returns.index)):
            prev_day = daily_returns.index[i-1]
            today = daily_returns.index[i]
            current_value = value_series.loc[prev_day]

            if today.strftime('%Y%m%d') in rebalancing_dates:
                month = today.strftime('%Y%m')
                new_portfolio = monthly_portfolios[strategy].get(month, [])
                
                if current_portfolio != new_portfolio:
                    turnover_rate = 1.0 # 월별 리밸런싱은 100% 교체를 가정
                    cost = current_value * turnover_rate * config.TRANSACTION_COST
                    current_value -= cost
                
                current_portfolio = new_portfolio

            if not current_portfolio or not all(ticker in daily_returns.columns for ticker in current_portfolio):
                value_series.loc[today] = current_value
                continue

            returns_today = daily_returns.loc[today, current_portfolio]
            avg_return = returns_today.mean()
            value_series.loc[today] = current_value * (1 + avg_return)

        portfolio_values[strategy] = value_series.dropna()
        
    bm_stock_returns = daily_returns[config.BM_KOSPI_TICKER]
    bm_stock_values = 100_000_000 * (1 + bm_stock_returns).cumprod()
    portfolio_values['BM_KOSPI200'] = bm_stock_values
    
    bm_bond_returns = daily_returns[config.BM_BOND_TICKER]
    bm_60_40_returns = 0.6 * bm_stock_returns + 0.4 * bm_bond_returns
    bm_60_40_values = 100_000_000 * (1 + bm_60_40_returns).cumprod()
    portfolio_values['BM_60_40'] = bm_60_40_values
    
    return pd.DataFrame(portfolio_values)