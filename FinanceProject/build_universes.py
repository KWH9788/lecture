from pykrx import stock
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import re

# --- [공통 모듈 - 최종 결정판] ---
def get_liquid_etf_pool(base_date):
    """
    [최종 결정판]
    오직 기준일(base_date)만을 입력받아, 해당 시점에 존재했던 모든 ETF 정보를
    스스로 수집하고 유동성 기준에 따라 필터링합니다.
    
    :param base_date: 데이터 수집 기준일 (YYYYMMDD)
    :return: 유동성 기준을 통과한 ETF 정보가 담긴 DataFrame
    """
    print(f"\n[{base_date} 기준 데이터 수집 및 1차 유동성 필터링 (최종 결정판)]")
    
    try:
        # 1. 해당 날짜의 모든 ETF 티커와 OHLCV 정보를 스스로 찾아냄
        tickers = stock.get_etf_ticker_list(base_date)
        all_etf_ohlcv_df = stock.get_etf_ohlcv_by_ticker(base_date)
    except Exception as e:
        print(f"  -> {base_date}의 기본 데이터를 가져오는 데 실패했습니다: {e}")
        return pd.DataFrame()

    print(f"  -> 총 {len(tickers)}개 ETF의 유효성을 검증하고 정보를 결합합니다...")
    
    etf_info_list = []
    for ticker in tqdm(tickers, desc="ETF 정보 유효성 검증"):
        try:
            name = stock.get_etf_ticker_name(ticker)
            trading_value = all_etf_ohlcv_df.loc[ticker, '거래대금']
            etf_info_list.append({'ticker': ticker, 'name': name, '거래대금': trading_value})
        except (KeyError, IndexError, ValueError):
            continue
            
    raw_df = pd.DataFrame(etf_info_list)
    
    # 2. 유동성 필터링 (거래대금 기준)
    MIN_TRADING_VALUE = 10 * 1_000_000_00
    liquid_df = raw_df[raw_df['거래대금'] >= MIN_TRADING_VALUE].copy()
    
    print(f"\n  -> 총 {len(tickers)}개 ETF 중 {len(liquid_df)}개가 유효성 검증 및 최소 거래대금 기준을 통과했습니다.")
    return liquid_df

# --- [핵심 모듈 - 최종 버전] 계층적 규칙 기반 ETF 분류기 ---
def classify_etf_engine(etf_name):
    name = etf_name.lower()

    # Level 1: 파생/전략형 및 현금성 자산 우선 제외
    if any(k in name for k in ['레버리지', '인버스', '선물', '커버드콜', '채권혼합', '스트립', 'TRF', '타겟']):
        return '파생/전략형'
    elif any(k in name for k in ['cd금리', 'kofr금리', '머니마켓', '단기통안채', '초단기채권', '단기채권']):
        return '현금성자산'
    
    # Level 2: 지역(Geography) 분류
    elif '미국' in name:
        if '국채' in name or '하이일드' in name:
            return '해외채권'
        elif any(k in name for k in ['나스닥', '필라델피아반도체', '테크', '빅테크', '반도체']):
            return '해외주식(기술주)'
        elif 's&p' in name or '배당' in name:
            return '해외주식(시장)'
        else:
            return '해외주식(미국기타)'
            
    elif any(k in name for k in ['차이나', '항셍', 'csi300', '과창판']):
        return '해외주식(중국)'
    elif any(k in name for k in ['일본', 'nikkei']):
        return '해외주식(일본)'
    elif any(k in name for k in ['인도', '베트남', '인도니프티']):
        return '해외주식(신흥국)'
    elif '글로벌' in name:
        if '반도체' in name: return '해외주식(글로벌반도체)'
        if 'ai' in name or '인공지능' in name: return '해외주식(글로벌AI)'
        else: return '해외주식(기타)'

    # Level 3: 국내 자산 분류
    else:
        if any(k in name for k in ['골드', '금현물', '원유', '리츠']):
            return '대체자산'
        elif any(k in name for k in ['국고채', '국채', '종합채권']):
            return '국내채권'
        elif any(k in name for k in ['2차전지', '반도체', '바이오', '헬스케어', 'ai', '로봇', 'k방산', '조선', '자동차', '은행', '미디어']):
            return '국내주식(섹터/테마)'
        elif any(k in name for k in ['고배당', '그룹', '밸류']):
            return '국내주식(스타일)'
        elif '코스닥150' in name:
            return '국내주식(코스닥)'
        elif '200' in name or '코스피' in name:
            return '국내주식(코스피)'
        else:
            return '기타'

# --- [전략 1] 안정추구형 유니버스 구성 ---
def build_conservative_universe(liquid_etf_pool):
    print("\n" + "="*60); print("1. 안정추구형 (저변동성 자산배분) 유니버스 구성"); print("="*60)
    
    classified_pool = liquid_etf_pool.copy()
    classified_pool['asset_class'] = classified_pool['name'].apply(classify_etf_engine)
    
    # [수정] 목표 자산 그룹과 그 안에서의 '선택 우선순위'를 정의
    target_priority_groups = [
        ('국내주식', ['국내주식(코스피)']),
        ('해외주식', ['해외주식(미국시장)', '해외주식(기술주)']), # 1순위: 미국시장, 2순위: 기술주
        ('채권', ['국내채권', '해외채권']),                  # 1순위: 국내채권, 2순위: 해외채권
        ('대체자산', ['대체자산'])
    ]
    
    universe = []
    for group_name, priority_classes in target_priority_groups:
        found_representative = False
        # 우선순위 순서대로 탐색
        for asset_class in priority_classes:
            candidates = classified_pool[classified_pool['asset_class'] == asset_class]
            
            if not candidates.empty:
                # 해당 우선순위에서 후보를 찾으면, 대표를 선정하고 탐색 중단
                representative = candidates.loc[candidates['거래대금'].idxmax()]
                representative['asset_class_group'] = group_name
                universe.append(representative)
                found_representative = True
                break # 다음 우선순위는 더 이상 탐색하지 않음
        
        if not found_representative:
            print(f"  [경고] '{group_name}' 그룹에 해당하는 ETF를 찾지 못했습니다.")
            
    universe_df = pd.DataFrame(universe)
    print("--- 최종 선정 결과 (계층적 선택 로직 기준) ---"); 
    print(universe_df[['asset_class_group', 'name']])
    return universe_df

# --- [전략 2] 공격투자형 유니버스 구성 ---
def build_aggressive_universe(liquid_etf_pool, base_date):
    print("\n" + "="*60); print("2. 공격투자형 (듀얼 모멘텀) 유니버스 구성"); print("="*60)
    
    classified_pool = liquid_etf_pool.copy()
    classified_pool['asset_class'] = classified_pool['name'].apply(classify_etf_engine)
    
    # 1. 통계치 계산
    start_date = (pd.to_datetime(base_date) - pd.DateOffset(years=1)).strftime("%Y%m%d")
    print(f"  -> 과거 1년({start_date}~{base_date}) 데이터로 변동성/베타 계산 중...")
    try:
        kospi_returns = stock.get_index_ohlcv(start_date, base_date, "1001")['종가'].pct_change().dropna()
    except Exception as e:
        print(f"KOSPI 지수 데이터를 가져올 수 없습니다: {e}")
        return pd.DataFrame(), pd.DataFrame()

    etf_stats = []
    for index, row in tqdm(classified_pool.iterrows(), total=classified_pool.shape[0], desc="ETF 변동성/베타 계산"):
        try:
            ticker = row['ticker']; name = row['name']; asset_class = row['asset_class']
            etf_returns = stock.get_etf_ohlcv_by_date(start_date, base_date, ticker)['종가'].pct_change().dropna()
            if len(etf_returns) < 200: continue
            
            beta = etf_returns.cov(kospi_returns) / kospi_returns.var()
            volatility = etf_returns.std() * np.sqrt(252)
            etf_stats.append({'ticker': ticker, 'name': name, 'asset_class': asset_class, 'volatility': volatility, 'beta': beta})
        except Exception:
            continue
            
    if not etf_stats:
        print("통계치를 계산할 수 있는 ETF가 없습니다.")
        return pd.DataFrame(), pd.DataFrame()
    stats_df = pd.DataFrame(etf_stats)

    # 2. 공격 자산군 선정: 순수 주식형 ETF 중에서 변동성/베타 상위 선정
    attack_classes = ['국내주식(코스피)', '국내주식(코스닥)', '국내주식(섹터)', 
                      '해외주식(기술주)', '해외주식(미국시장)', '해외주식(중국)', '해외주식(신흥국)']
    attack_candidates = stats_df[stats_df['asset_class'].isin(attack_classes)].copy()
    attack_candidates['rank_sum'] = attack_candidates['volatility'].rank(ascending=False) + attack_candidates['beta'].rank(ascending=False)
    attack_universe = attack_candidates.sort_values(by='rank_sum').head(3)

    # 3. 안전 자산 선정: 채권/현금성자산 중에서 변동성 최하위 선정
    safe_classes = ['국내채권', '해외채권', '현금성자산']
    safe_candidates = stats_df[stats_df['asset_class'].isin(safe_classes)].copy()
    safe_universe = safe_candidates.sort_values(by='volatility').head(1)
    
    print("--- 최종 선정 결과 ---")
    print("\n[공격 자산군]"); print(attack_universe[['ticker', 'name', 'asset_class', 'volatility', 'beta']])
    print("\n[안전 자산]"); print(safe_universe[['ticker', 'name', 'asset_class', 'volatility']])
    return attack_universe, safe_universe

# --- [전략 3] 위험중립형 유니버스 구성 ---
def build_neutral_universe(liquid_etf_pool):
    print("\n" + "="*60); print("3. 위험중립형 (멀티팩터) 유니버스 구성"); print("="*60)
    
    classified_pool = liquid_etf_pool.copy()
    classified_pool['factor'] = classified_pool['name'].apply(classify_etf_engine)
    
    target_factors = ['국내주식(코스피)', '해외주식(기술주)', '국내채권', '대체자산']
    
    filtered_pool = classified_pool[classified_pool['factor'].isin(target_factors)]

    universe_df = filtered_pool.loc[filtered_pool.groupby('factor')['거래대금'].idxmax()]

    print("--- 최종 선정 결과 (계층적 분류 엔진 기준) ---")
    print(universe_df[['factor', 'name']])
    return universe_df

def get_universe_returns(universe_df, start_date, end_date):
    """주어진 유니버스의 과거 1년 수익률 데이터를 가져오는 헬퍼 함수"""
    data = pd.DataFrame()
    for ticker, name in zip(universe_df['ticker'], universe_df['name']):
        try:
            data[name] = stock.get_etf_ohlcv_by_date(start_date, end_date, ticker)['종가']
        except:
            continue
    return data.pct_change().dropna()

# --- [메인 실행 블록] ---
# if __name__ == "__main__":
#     base_date_for_2024_backtest = "20231228"
#     liquid_etfs_pool_final = get_liquid_etf_pool(base_date_for_2024_backtest)
#     print("\n[최종 유동성 기준 통과 ETF 목록]"); print(liquid_etfs_pool_final)
#     liquid_etfs_pool_temp = liquid_etfs_pool_final.copy()
#     liquid_etfs_pool_temp['factor'] = liquid_etfs_pool_temp['name'].apply(classify_etf_engine)
#     liquid_etfs_pool_temp.to_excel(f"liquid_etfs_pool_{base_date_for_2024_backtest}.xlsx", index=False)

#     if not liquid_etfs_pool_final.empty:
#         conservative_uni = build_conservative_universe(liquid_etfs_pool_final)
#         aggressive_uni_attack, aggressive_uni_safe = build_aggressive_universe(liquid_etfs_pool_final, base_date_for_2024_backtest)
#         neutral_uni = build_neutral_universe(liquid_etfs_pool_final)
#         print("\n[안정추구형 유니버스]")
#         print(conservative_uni)
#         print("\n[공격투자형 유니버스]")
#         print(aggressive_uni_attack)
#         print(aggressive_uni_safe)
#         print("\n[위험중립형 유니버스]")
#         print(neutral_uni)
#     else:
#         print(f"{base_date_for_2024_backtest}에 유동성 기준을 만족하는 ETF가 없습니다.")