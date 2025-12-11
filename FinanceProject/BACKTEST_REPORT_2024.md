# 포트폴리오 백테스팅 보고서 (2024년)

## 📊 프로젝트 개요

### 목적
투자 성향별 최적화된 포트폴리오 전략을 개발하고, 2024년 한국 ETF 시장 데이터를 활용한 백테스팅을 통해 각 전략의 실효성을 검증합니다.

### 백테스팅 기간
- **시작일**: 2024-01-01
- **종료일**: 2024-12-31
- **리밸런싱**: 월초 (매월 첫 거래일)
- **초기 자본**: 100,000,000원 (1억원)

---

## 🎯 투자 전략

### 1. Conservative (안정추구형)
**투자 철학**: 자산 배분을 통한 안정적 수익 추구

**자산군 구성**:
- 해외 주식 (S&P 500): 글로벌 분산
- 국내 주식 (KOSPI): 국내 시장 노출
- 채권 (국채): 안정성 확보
- 대체 자산 (금/원자재): 인플레이션 헤지

**리스크 관리**:
- MA200 필터: Meb Faber (2007) 기반
- 6개월 역변동성 가중: Meucci (2005) 이론
- 자산군 균형 유지

### 2. Aggressive (공격투자형)
**투자 철학**: 리스크를 감수하고 높은 수익 추구

**종목 선정 기준** (3가지 리스크 신호):
1. **MA200 돌파**: 상승 추세 확인
2. **변동성 1.2배 이상**: Ang (2014) 연구 기반
3. **3개월 모멘텀 양수**: Jegadeesh & Titman (1993)

**리스크 관리**:
- 모든 신호 충족 시에만 투자
- 신호 미충족 시 현금 보유
- 6개월 역변동성 가중

### 3. Neutral (위험중립형)
**투자 철학**: 다양한 팩터에 균형있게 노출

**팩터 구성**:
- 시장 팩터: 시장 베타 노출
- 성장 팩터: 성장주 투자
- 안정 팩터: 배당/가치주 투자
- 인플레이션 헤지: 원자재/금

**리스크 관리**:
- 4개 팩터 균등 배분
- 80/20 가중 (주식 80% + 안정자산 20%)
- 팩터 다각화를 통한 리스크 분산

---

## 💰 거래 비용 설정

### 거래 비용 구성
```python
TRANSACTION_COST_RATE = 0.0002  # 0.02% (수수료)
SLIPPAGE_RATE = 0.0001          # 0.01% (슬리피지)
# 총 편도 비용: 0.03%
# 총 왕복 비용: 0.06%
```

### 적용 방식
- **매도 시**: (매도금액) × 0.03%
- **매수 시**: (매수금액) × 0.03%
- **리밸런싱**: 가중치 변화에 따른 추가 비용 계산

---

## 📈 벤치마크 설정

### 전략별 맞춤 벤치마크

#### 1. Conservative → **BM_MultiAsset**
```
구성: S&P500 25% + KOSPI 25% + 채권 25% + 금 25%
근거: 다자산 배분 전략의 표준 벤치마크
ETF: 360750(TIGER S&P500), 069500(KODEX 200), 
     148070(KOSEF 국고채10년), 411060(ACE 금현물)
```

#### 2. Aggressive → **BM_HighVolatility**
```
구성: KOSDAQ 150 (고변동성 지수)
근거: 공격적 투자의 리스크-수익 특성 반영
ETF: 229200(KODEX 코스닥150)
```

#### 3. Neutral → **BM_FactorBalance**
```
구성: KOSPI 25% + KOSDAQ 25% + 채권 25% + 금 25%
근거: 다양한 팩터 노출의 균형잡힌 비교
ETF: 069500(KODEX 200), 229200(KODEX 코스닥150),
     148070(KOSEF 국고채10년), 411060(ACE 금현물)
```

---

## 📊 성과 지표

### 주요 KPI
1. **누적 수익률**: 백테스팅 기간 동안의 총 수익률
2. **CAGR**: 연평균 복리 수익률
3. **MDD**: 최대 낙폭 (Maximum Drawdown)
4. **샤프 비율**: 위험 대비 수익 (무위험 수익률 3% 가정)
5. **승률**: 일별 수익률 기준 양(+)의 비율
6. **회전율**: 연평균 포트폴리오 회전율

### 추가 분석
- **월별 수익률 히트맵**: 월별 성과 패턴 분석
- **드로우다운 차트**: 손실 구간 시각화
- **포트폴리오 구성**: 대표 월별 자산 배분
- **이동 평균 성과**: 60일 이동 평균 수익률 및 변동성
- **성과 분해**: 월별/분기별 수익 기여도 분석

---

## 🔬 과적합 방지

### 이론 기반 파라미터 사용
모든 핵심 파라미터는 학술 문헌에 근거하여 설정:

1. **MA200**: Meb Faber (2007) - "A Quantitative Approach to Tactical Asset Allocation"
2. **변동성 1.2배**: Ang et al. (2014) - "High Idiosyncratic Volatility and Low Returns"
3. **3개월 모멘텀**: Jegadeesh & Titman (1993) - "Returns to Buying Winners and Selling Losers"
4. **6개월 역변동성**: Meucci (2005) - "Risk and Asset Allocation"

### 검증 절차
- 2024년 데이터로 조정된 파라미터 전수 제거
- 표준 학술 지표로 전면 재구축
- 과거 데이터 피팅 금지

---

## 📁 출력 파일

### CSV 파일
1. `kpi_summary_monthly.csv`: 전략별 KPI 요약
2. `monthly_stats_monthly.csv`: 월별 수익률 통계
3. `monthly_attribution_monthly.csv`: 월별 성과 분해
4. `quarterly_attribution_monthly.csv`: 분기별 성과 분해

### 차트 파일
1. `cumulative_returns_monthly.png`: 전략별 누적 수익률 (매매 시점 표시)
2. `cumulative_returns_combined_monthly.png`: 통합 누적 수익률
3. `monthly_heatmap_monthly.png`: 월별 수익률 히트맵
4. `drawdown_monthly.png`: 전략별 드로우다운 차트
5. `portfolio_composition_monthly_*.png`: 전략별 포트폴리오 구성 파이 차트
6. `rolling_performance_monthly.png`: 이동 평균 수익률 및 변동성
7. `performance_attribution_monthly.png`: 월별/분기별 성과 분해

### 로그 파일
- `backtest_log_YYYYMMDD_HHMMSS.txt`: 전체 백테스팅 로그

---

## 🎓 학술적 근거

### Conservative 전략
- **Markowitz (1952)**: Modern Portfolio Theory - 자산 배분의 이론적 기초
- **Faber (2007)**: 추세 추종 필터를 활용한 위험 관리
- **Meucci (2005)**: 역변동성 가중을 통한 리스크 균형

### Aggressive 전략
- **Jegadeesh & Titman (1993)**: 모멘텀 효과의 실증적 증거
- **Ang et al. (2014)**: 변동성과 수익의 관계
- **Faber (2007)**: 이동평균을 통한 시장 타이밍

### Neutral 전략
- **Fama & French (1993)**: 다중 팩터 모델의 이론적 기초
- **Asness et al. (2015)**: 팩터 다각화의 효과
- **Meucci (2005)**: 리스크 균형 접근법

---

## 💻 기술 스택

### 개발 환경
- **Python**: 3.11.14
- **패키지**: pandas, numpy, pykrx, matplotlib, seaborn, scipy

### 코드 구조
```
FinanceProject/
├── main.py                      # 메인 실행 스크립트
├── config/
│   └── config.py               # 전역 설정 (파라미터, 벤치마크)
└── modules/
    ├── backtest_engine.py      # 백테스팅 엔진
    ├── universe_builder.py     # 전략별 유니버스 구성
    └── performance_analyzer.py # 성과 분석 및 시각화
```

---

## 📝 실행 방법

### 기본 실행
```bash
python main.py
```

### 결과 확인
```
results/
├── kpi_summary_monthly.csv
├── monthly_stats_monthly.csv
├── monthly_attribution_monthly.csv
├── quarterly_attribution_monthly.csv
├── cumulative_returns_monthly.png
├── cumulative_returns_combined_monthly.png
├── monthly_heatmap_monthly.png
├── drawdown_monthly.png
├── portfolio_composition_monthly_*.png
├── rolling_performance_monthly.png
├── performance_attribution_monthly.png
└── backtest_log_*.txt
```

---

## 🔍 2024년 시장 환경

### 거시경제 상황
- **금리**: 미국 Fed 금리 동결 → 하반기 인하 시작
- **환율**: 원/달러 변동성 확대
- **인플레이션**: 둔화 추세

### 국내 시장 특징
- **KOSPI**: 박스권 등락 (2,400~2,700)
- **KOSDAQ**: 변동성 확대 (기술주 부진)
- **채권**: 금리 인하 기대로 상승
- **원자재**: 금 강세, 원유 약세

### 전략별 영향
- **Conservative**: 채권/금 강세로 방어 효과
- **Aggressive**: 변동성 확대로 신호 이탈 빈번
- **Neutral**: 팩터 분산으로 안정적 성과

---

## 📞 문의

**프로젝트**: 투자 성향별 포트폴리오 전략 백테스팅  
**기간**: 2024년 1월 ~ 12월  
**작성일**: 2024년 12월

---

## 📚 참고문헌

1. Faber, M. (2007). "A Quantitative Approach to Tactical Asset Allocation"
2. Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers"
3. Ang, A., et al. (2014). "High Idiosyncratic Volatility and Low Returns"
4. Meucci, A. (2005). "Risk and Asset Allocation"
5. Markowitz, H. (1952). "Portfolio Selection"
6. Fama, E., & French, K. (1993). "Common Risk Factors in the Returns on Stocks and Bonds"
7. Asness, C., et al. (2015). "Fact, Fiction, and Momentum Investing"

---

*본 보고서는 학술 연구 목적으로 작성되었으며, 실제 투자 권유가 아닙니다.*
