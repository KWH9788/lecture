# config.py

# --- 백테스팅 기간 설정 ---
# 최종 분석 시 "20190101" ~ "20241231"로 변경
START_DATE = "20240101"
END_DATE = "20241231"

# --- 매매 실행 모드 설정 ---
# 'monthly': 월초 일괄 리밸런싱
# 'daily': 일별 분할 매매 로직 (Phase 3)
# 현재는 'monthly'로 설정하여 기본 성능 확인
TRADING_MODE = 'monthly' 

# --- 핵심 파라미터 설정 ---
MIN_TRADING_VALUE = 1_000_000_000  # 최소 거래대금 (10억)

# --- 거래 비용 설정 ---
# 한국 ETF 거래 비용 구조:
# - 증권거래세: 면제 (ETF는 증권거래세 면제)
# - 거래수수료: 0.015% (온라인 증권사 기준, 최대 0.015%)
# - 호가 스프레드: 유동성에 따라 다름 (대형 ETF는 0.01~0.05%)
# 
# 보수적 추정치 적용:
TRANSACTION_COST_RATE = 0.0002      # 거래 수수료 (편도 0.02%, 매수+매도 시 0.04%)
                                     # 실제: 온라인 증권사 평균 0.015% + 기타 0.005%
SLIPPAGE_RATE = 0.0001              # 슬리피지 (편도 0.01%)
                                     # 시장가 주문 시 불리한 가격 체결
                                     # 대형 ETF 기준, 소형 ETF는 더 클 수 있음
# 
# 총 거래 비용 = (TRANSACTION_COST_RATE + SLIPPAGE_RATE) × 2 = 0.06% (왕복)

AGGRESSIVE_LOOKBACK_MONTHS = 12    # 공격형: 과거 데이터 참조 기간
AGGRESSIVE_TOP_N = 3               # 공격형: 상위 ETF 선정 개수

# --- 벤치마크 Ticker ---
BM_KOSPI_TICKER = "069500"
BM_BOND_TICKER = "114260"

# --- 데이터 캐시 설정 ---
USE_CACHE = True                    # True: 캐시 사용, False: 항상 새로 다운로드
CACHE_DIR = "cache"                # 캐시 저장 디렉토리
CACHE_EXPIRY_DAYS = 30               # 캐시 유효 기간 (일)