# -*- coding: utf-8 -*-

# 1. 필요 라이브러리 임포트
# yfinance가 없다면 먼저 설치해야 합니다: pip install yfinance
import pandas as pd
import yfinance as yf
from pykrx import stock
import time

print("pykrx, yfinance, pandas 라이브러리를 성공적으로 임포트했습니다.")

# 2. 데이터 수집 기간 및 기준일 설정
# -----------------------------------------------------------------------------
# 시세 데이터 수집 기간 (yfinance용, YYYY-MM-DD 형식)
start_date_ohlcv = '2024-01-01'
end_date_ohlcv = '2024-12-31'

# ETF 목록 조회 기준일 (pykrx용, YYYYMMDD 형식)
base_date_list = '20241231' 
# -----------------------------------------------------------------------------

# =============================================================================
# 1단계: [pykrx] '고배당' 관련 키워드를 포함한 ETF 티커 탐색
# =============================================================================
print("\n" + "="*50)
print(f"1단계: {base_date_list} 기준 '고배당' 관련 ETF 탐색 시작 (pykrx 사용)")
print("="*50)

try:
    # 기준일의 전체 ETF 티커 리스트 가져오기
    all_etf_tickers = stock.get_etf_ticker_list(base_date_list)
    print(f"총 {len(all_etf_tickers)}개의 ETF 목록을 조회했습니다. 이제 이름 필터링을 시작합니다.")

    # 고배당 관련 키워드 리스트
    keywords = ['고배당', '배당', '인컴', '커버드콜', '디비디', '분배']

    # 필터링된 ETF 정보를 저장할 리스트
    high_dividend_etfs = []

    for ticker in all_etf_tickers:
        # ETF 종목명 조회
        etf_name = stock.get_etf_ticker_name(ticker)
        
        # 키워드가 종목명에 포함되어 있는지 확인
        if any(keyword in etf_name for keyword in keywords):
            high_dividend_etfs.append({'티커': ticker, '종목명': etf_name})
            print(f"  [발견] {etf_name} ({ticker})")
            
        time.sleep(0.01) # 아주 짧은 딜레이

    # 리스트를 데이터프레임으로 변환
    if high_dividend_etfs:
        filtered_etfs_df = pd.DataFrame(high_dividend_etfs)
        etf_tickers_to_fetch = filtered_etfs_df['티커'].tolist()
        
        print("\n[탐색 완료] 아래 ETF들의 시세 데이터 수집을 진행합니다:")
        print(filtered_etfs_df.to_string(index=False))
    else:
        print("관련 키워드를 포함한 ETF를 찾지 못했습니다. 기본 예시 티커로 수집을 시도합니다.")
        etf_tickers_to_fetch = ['139320', '373490', '136340'] # 예시 티커
        filtered_etfs_df = pd.DataFrame({'티커': etf_tickers_to_fetch, '종목명': [stock.get_etf_ticker_name(t) for t in etf_tickers_to_fetch]})


except Exception as e:
    print(f"1단계 ETF 탐색 중 오류 발생: {e}")
    print("기본 예시 티커 리스트로 데이터 수집을 시도합니다.")
    etf_tickers_to_fetch = ['139320', '373490', '136340'] # 예시 티커
    filtered_etfs_df = pd.DataFrame({'티커': etf_tickers_to_fetch, '종목명': [stock.get_etf_ticker_name(t) for t in etf_tickers_to_fetch]})


# =============================================================================
# 2단계: [yfinance] 선정된 ETF의 60분봉 시세 데이터 수집
# =============================================================================
print("\n" + "="*50)
print(f"2단계: 선정된 ETF의 60분봉 데이터 수집 시작 (yfinance 사용)")
print("="*50)

# yfinance 형식으로 티커 변환 (예: '139320' -> '139320.KS')
yf_tickers = [f"{ticker}.KS" for ticker in etf_tickers_to_fetch]

try:
    print(f"데이터 수집 중... 대상: {yf_tickers}")
    df_60m = yf.download(
        tickers=yf_tickers,
        start=start_date_ohlcv,
        end=end_date_ohlcv,
        interval="60m",
        group_by='ticker'
    )

    if df_60m.empty:
        print("\n수집된 데이터가 없습니다. 기간이나 티커를 확인해주세요.")
    else:
        print("\n데이터 수집 완료. 이제 CSV 파일로 정리합니다.")
        
        all_data_list = []
        for ticker_with_suffix in yf_tickers:
            original_ticker = ticker_with_suffix.split('.')[0]
            
            temp_df = df_60m[ticker_with_suffix].dropna()
            if temp_df.empty: continue

            temp_df['티커'] = original_ticker
            etf_name = filtered_etfs_df.loc[filtered_etfs_df['티커'] == original_ticker, '종목명'].iloc[0]
            temp_df['종목명'] = etf_name
            all_data_list.append(temp_df)

        final_df = pd.concat(all_data_list).reset_index()
        final_df.rename(columns={'Datetime': 'Date'}, inplace=True)
        final_df = final_df[['Date', '티커', '종목명', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        file_name = 'krx_keyword_filtered_etf_60min_data.csv'
        final_df.to_csv(file_name, index=False, encoding='utf-8-sig')
        print(f"\n✅ 최종 데이터가 '{file_name}' 파일로 성공적으로 저장되었습니다.")

except Exception as e:
    print(f"\n2단계 데이터 수집 중 오류 발생: {e}")