# modules/data_loader.py
from pykrx import stock
import pandas as pd

def get_liquid_etf_pool_robust(base_date, min_value):
    """
    [최종 안정화 버전]
    이름 조회가 가능한 현재 상장 ETF 중, 최소 유동성 기준을 통과한 풀을 생성합니다.
    """
    try:
        tickers = stock.get_etf_ticker_list(base_date)
        all_etf_ohlcv_df = stock.get_etf_ohlcv_by_ticker(base_date)
    except Exception:
        return pd.DataFrame()

    etf_info_list = []
    for ticker in tickers:
        try:
            trading_value = all_etf_ohlcv_df.loc[ticker, '거래대금']
            if trading_value >= min_value:
                name = stock.get_etf_ticker_name(ticker)
                etf_info_list.append({'ticker': ticker, 'name': name, '거래대금': trading_value})
        except (KeyError, IndexError, ValueError):
            continue
            
    if not etf_info_list:
        return pd.DataFrame()
        
    return pd.DataFrame(etf_info_list)

def classify_etf_decision_tree(etf_name):
    """
    [핵심 분류 엔진]
    의사결정 트리(중첩 if/elif/else) 구조로 ETF를 단일 자산군으로 분류합니다.
    """
    name = etf_name.lower()
    if any(k in name for k in ['레버리지', '인버스', '선물', '커버드콜', '채권혼합', '스트립', 'TRF', '타겟']):
        return '파생/전략형'
    elif any(k in name for k in ['cd금리', 'kofr금리', '머니마켓', '단기통안채', '초단기채권', '단기채권']):
        return '현금성자산'
    elif '미국' in name:
        if '국채' in name or '하이일드' in name: return '해외채권'
        elif any(k in name for k in ['나스닥', '필라델피아반도체', '테크', '빅테크', '반도체']): return '해외주식(기술주)'
        elif 's&p' in name or '배당' in name: return '해외주식(시장)'
        else: return '해외주식(미국기타)'
    elif any(k in name for k in ['차이나', '항셍', 'csi300', '과창판']): return '해외주식(중국)'
    elif any(k in name for k in ['일본', 'nikkei']): return '해외주식(일본)'
    elif any(k in name for k in ['인도', '베트남', '인도니프티']): return '해외주식(신흥국)'
    elif '글로벌' in name:
        if '반도체' in name: return '해외주식(글로벌반도체)'
        if 'ai' in name or '인공지능' in name: return '해외주식(글로벌AI)'
        else: return '해외주식(기타)'
    else:
        if any(k in name for k in ['골드', '금현물', '원유', '리츠']): return '대체자산'
        elif any(k in name for k in ['국고채', '국채', '종합채권']): return '국내채권'
        elif any(k in name for k in ['2차전지', '반도체', '바이오', '헬스케어', 'ai', '로봇', 'k방산', '조선', '자동차', '은행', '미디어']): return '국내주식(섹터/테마)'
        elif any(k in name for k in ['고배당', '그룹', '밸류']): return '국내주식(스타일)'
        elif '코스닥150' in name: return '국내주식(코스닥)'
        elif '200' in name or '코스피' in name: return '국내주식(코스피)'
        else: return '기타'