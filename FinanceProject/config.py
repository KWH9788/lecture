# config.py

# --- 백테스팅 기간 설정 ---
START_DATE = "20190101"
END_DATE = "20241231"

# --- 핵심 파라미터 설정 ---
MIN_TRADING_VALUE = 1_000_000_000  # 최소 거래대금 (10억)
TRANSACTION_COST = 0.002           # 거래 비용 (0.2%)
AGGRESSIVE_LOOKBACK_MONTHS = 12    # 공격형: 과거 데이터 참조 기간
AGGRESSIVE_TOP_N = 3               # 공격형: 상위 ETF 선정 개수

# --- 벤치마크 Ticker ---
BM_KOSPI_TICKER = "069500"
BM_BOND_TICKER = "114260"