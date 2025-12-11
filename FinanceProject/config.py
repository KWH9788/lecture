# config.py

# --- 백테스팅 기간 설정 ---
# 2024년 백테스팅 (최신 데이터 분석)
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

# --- 리스크 관리 설정 (이론 기반 표준 지표) ---
# 출처: 학술 문헌 및 업계 표준에 근거한 파라미터

# 변동성 임계값 (VIX 연구 기반)
VOLATILITY_SPIKE_THRESHOLD = 1.2   # 평균 대비 20% 증가 시 리스크 신호
                                   # 출처: Ang, A. (2014). Asset Management
                                   # 업계 표준: 1.2~1.3배 사용

# 모멘텀 기간 (Jegadeesh & Titman, 1993)
MOMENTUM_LOOKBACK_MONTHS = 3       # 3개월 모멘텀 (학술적으로 가장 검증된 기간)
                                   # 1~3개월 단기 모멘텀이 예측력 높음

# 드로다운 한계 (리스크 관리 원칙)
MAX_PORTFOLIO_MDD = 0.10           # 포트폴리오 최대 허용 낙폭 (10%)
                                   # 보수적 자산배분 표준: -10% 손절선
                                   # 출처: Meb Faber's Tactical Asset Allocation

TARGET_VOLATILITY = 0.15           # 목표 연간 변동성 (15%)
                                   # 60/40 포트폴리오 역사적 변동성

# 역변동성 가중치 계산 기간 (Meucci, 2005)
VOLATILITY_LOOKBACK_MONTHS = 6     # 변동성 계산 기간 (6개월)
                                   # 출처: Meucci, A. (2005). Risk and Asset Allocation
                                   # 최소 6개월 ~ 1년 권장

# --- 벤치마크 Ticker ---
# 기본 벤치마크
BM_KOSPI_TICKER = "069500"        # KODEX 200 (국내 주식 시장 대표)
BM_BOND_TICKER = "114260"         # KODEX 단기채권 (안전자산 대표)

# 전략별 맞춤 벤치마크 (시장 상황별 효과성 측정)
BM_SP500_TICKER = "360750"        # TIGER 미국S&P500 (글로벌 주식)
BM_KOSDAQ_TICKER = "229200"       # KODEX 코스닥150 (국내 성장주)
BM_LEVERAGE_TICKER = "122630"     # KODEX 레버리지 (고위험/고수익)
BM_GOLD_TICKER = "411060"         # ACE KRX금현물 (안전자산/인플레이션 헷지)

# --- 데이터 캐시 설정 ---
USE_CACHE = True                    # True: 캐시 사용, False: 항상 새로 다운로드
CACHE_DIR = "cache"                # 캐시 저장 디렉토리
CACHE_EXPIRY_DAYS = 30               # 캐시 유효 기간 (일)

# --- 일별 분할 매매 설정 (TRADING_MODE = 'daily'일 때 적용) ---
DAILY_TRADE_LIMIT_RATIO = 0.2       # 일일 거래 한도 (총 자산의 20%)
ATR_PERIOD = 14                     # ATR 계산 기간
MA_SHORT_PERIOD = 5                 # 단기 이동평균선 기간
MA_LONG_PERIOD = 20                 # 장기 이동평균선 기간
MIN_TRADE_THRESHOLD = 0.005         # 최소 거래 금액 (총 자산의 0.5%)
TARGET_TOLERANCE = 0.01             # 목표 달성 허용 오차 (1%)