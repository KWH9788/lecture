# main.py
import config
from modules import backtest_engine
from modules import performance_analyzer
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

class Tee:
    """콘솔과 파일에 동시 출력하는 클래스"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def main():
    # 로그 디렉토리 생성
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일 경로 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"backtest_log_{timestamp}.txt")
    
    # 로그 파일 열기
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # stdout을 콘솔과 파일 동시 출력으로 리다이렉트
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    try:
        print("="*50)
        print("퀀트 투자 전략 백테스팅 및 분석을 시작합니다.")
        print(f"분석 기간: {config.START_DATE} ~ {config.END_DATE}")
        print(f"매매 모드: {config.TRADING_MODE}")
        print(f"로그 파일: {log_file_path}")
        print("="*50)
        
        # 1. 백테스팅 실행
        final_values, monthly_portfolios, monthly_weights, rebalancing_dates = backtest_engine.run_backtest(
            config.START_DATE, 
            config.END_DATE, 
            config
        )
        
        if final_values.empty:
            print("\n백테스팅 결과가 비어있습니다. 프로세스를 종료합니다.")
            return
        
        # 백테스팅 기간으로 필터링 (2023년 데이터 제거)
        import pandas as pd
        final_values = final_values.loc[config.START_DATE:config.END_DATE]
        
        # NaN 값 제거 및 검증
        print(f"\n데이터 검증 중...")
        for col in final_values.columns:
            nan_count = final_values[col].isna().sum()
            if nan_count > 0:
                print(f"  ⚠️  {col}: {nan_count}개 NaN 값 발견 → 제거")
                final_values[col] = final_values[col].fillna(method='ffill').fillna(method='bfill')
        
        # 모든 컬럼이 NaN인 행 제거
        final_values = final_values.dropna(how='all')
        
        print(f"  ✓ 최종 데이터 기간: {final_values.index[0].date()} ~ {final_values.index[-1].date()}")
        print(f"  ✓ 데이터 포인트: {len(final_values)}개")
        
        # 벤치마크 데이터 검증
        if 'BM_KOSPI200' in final_values.columns:
            bm_valid = final_values['BM_KOSPI200'].notna().sum()
            print(f"  ✓ BM_KOSPI200: {bm_valid}/{len(final_values)} 유효 데이터")
        if 'BM_60_40' in final_values.columns:
            bm_valid = final_values['BM_60_40'].notna().sum()
            print(f"  ✓ BM_60_40: {bm_valid}/{len(final_values)} 유효 데이터")
            
        # 1.5 포트폴리오 구성 요약
        print("\n" + "="*50)
        print("전략별 포트폴리오 구성 요약")
        print("="*50)
        
        for strategy in ['conservative', 'aggressive', 'neutral']:
            if strategy in monthly_portfolios and monthly_portfolios[strategy]:
                portfolio_history = monthly_portfolios[strategy]
                sorted_months = sorted(portfolio_history.keys())
                
                print(f"\n[{strategy.upper()}]")
                print(f"  리밸런싱 횟수: {len(sorted_months)}회")
                print(f"  기간: {sorted_months[0]} ~ {sorted_months[-1]}")
                
                # 평균 종목 수
                avg_size = sum(len(portfolio_history[m]) for m in sorted_months) / len(sorted_months)
                print(f"  평균 보유 종목 수: {avg_size:.1f}개")
                
                # 가장 자주 등장한 종목 (Top 3)
                from collections import Counter
                all_tickers = []
                for month in sorted_months:
                    all_tickers.extend(portfolio_history[month])
                
                ticker_counts = Counter(all_tickers)
                print(f"  가장 자주 편입된 종목 (Top 3):")
                for ticker, count in ticker_counts.most_common(3):
                    freq = count / len(sorted_months) * 100
                    print(f"    - {ticker}: {count}회 ({freq:.1f}%)")
        
        # 2. 성과 분석 및 결과 저장 (회전율 포함)
        kpi_summary = performance_analyzer.analyze_performance(final_values, monthly_portfolios)
        
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        kpi_summary.to_csv(os.path.join(results_dir, f'kpi_summary_{config.TRADING_MODE}.csv'), encoding='utf-8-sig')
        
        print("\n--- 핵심 성과 지표 (KPI) ---")
        print(kpi_summary)
        
        # 3. 월별 성과 분석
        print("\n" + "="*50)
        print("월별 성과 분석 중...")
        print("="*50)
        monthly_df, monthly_stats = performance_analyzer.analyze_monthly_performance(final_values)
        
        print("\n--- 월별 수익률 통계 ---")
        print(monthly_stats.round(2))
        monthly_stats.to_csv(os.path.join(results_dir, f'monthly_stats_{config.TRADING_MODE}.csv'), encoding='utf-8-sig')
        
        # 4. 시각화 자료 생성 및 저장
        print("\n" + "="*50)
        print("시각화 자료 생성 중...")
        print("="*50)
        
        # 4-1. 누적 수익률 차트 (전략별 개별)
        print("  [1/6] 전략별 누적 수익률 차트 생성 (매매 시점 포함)...")
        performance_analyzer.plot_cumulative_returns(
            final_values, 
            os.path.join(results_dir, f'cumulative_returns_{config.TRADING_MODE}.png'),
            rebalancing_dates=rebalancing_dates
        )
        
        # 4-2. 월별 수익률 히트맵
        print("  [2/6] 월별 수익률 히트맵 생성...")
        performance_analyzer.plot_monthly_returns_heatmap(
            monthly_df,
            os.path.join(results_dir, f'monthly_heatmap_{config.TRADING_MODE}.png')
        )
        
        # 4-4. 드로우다운 차트
        print("  [4/7] 드로우다운 차트 생성...")
        performance_analyzer.plot_drawdown_chart(
            final_values,
            os.path.join(results_dir, f'drawdown_{config.TRADING_MODE}.png')
        )
        
        # 4-3. 드로우다운 차트
        print("  [3/6] 드로우다운 차트 생성...")
        performance_analyzer.plot_drawdown_chart(
            final_values,
            os.path.join(results_dir, f'drawdown_{config.TRADING_MODE}.png')
        )
        
        # 4-4. 월별 포트폴리오 구성 파이 차트
        print("  [4/5] 월별 포트폴리오 구성 파이 차트 생성...")
        performance_analyzer.plot_portfolio_composition_pies(
            monthly_portfolios,
            monthly_weights,  # 가중치 정보 추가
            os.path.join(results_dir, f'portfolio_composition_{config.TRADING_MODE}.png')
        )
        
        # 4-5. 성과 분해 (Performance Attribution)
        print("  [5/5] 월별/분기별 성과 분해 차트 생성...")
        monthly_attr, quarterly_attr = performance_analyzer.calculate_performance_attribution(
            final_values, monthly_portfolios
        )
        
        # 성과 분해 결과 저장
        monthly_attr.to_csv(os.path.join(results_dir, f'monthly_attribution_{config.TRADING_MODE}.csv'), 
                           index=False, encoding='utf-8-sig')
        quarterly_attr.to_csv(os.path.join(results_dir, f'quarterly_attribution_{config.TRADING_MODE}.csv'), 
                             index=False, encoding='utf-8-sig')
        
        # 성과 분해 차트 생성
        performance_analyzer.plot_performance_attribution(
            monthly_attr, quarterly_attr,
            filepath=os.path.join(results_dir, f'performance_attribution_{config.TRADING_MODE}.png')
        )
        
        # 주요 성과 기여 요약 출력
        print("\n" + "="*50)
        print("성과 분해 요약 (Performance Attribution)")
        print("="*50)
        
        for strategy in ['CONSERVATIVE', 'AGGRESSIVE', 'NEUTRAL']:
            strategy_monthly = monthly_attr[monthly_attr['전략'] == strategy]
            if not strategy_monthly.empty:
                best_month = strategy_monthly.loc[strategy_monthly['월수익률(%)'].idxmax()]
                worst_month = strategy_monthly.loc[strategy_monthly['월수익률(%)'].idxmin()]
                
                print(f"\n[{strategy}]")
                print(f"  최고 성과 월: {best_month['기간']} (+{best_month['월수익률(%)']:.2f}%) - {best_month['성과']}")
                print(f"  최저 성과 월: {worst_month['기간']} ({worst_month['월수익률(%)']:.2f}%) - {worst_month['성과']}")
        
        print(f"\n'{results_dir}' 폴더에 모든 분석 결과가 저장되었습니다.")
        print("\n생성된 파일 목록:")
        print(f"  - kpi_summary_{config.TRADING_MODE}.csv")
        print(f"  - monthly_stats_{config.TRADING_MODE}.csv")
        print(f"  - monthly_attribution_{config.TRADING_MODE}.csv (NEW)")
        print(f"  - quarterly_attribution_{config.TRADING_MODE}.csv (NEW)")
        print(f"  - cumulative_returns_{config.TRADING_MODE}.png (전략별 개별 + 매매 시점)")
        print(f"\n'{results_dir}' 폴더에 모든 분석 결과가 저장되었습니다.")
        print("\n생성된 파일 목록:")
        print(f"  - kpi_summary_{config.TRADING_MODE}.csv")
        print(f"  - monthly_stats_{config.TRADING_MODE}.csv")
        print(f"  - monthly_attribution_{config.TRADING_MODE}.csv")
        print(f"  - quarterly_attribution_{config.TRADING_MODE}.csv")
        print(f"  - cumulative_returns_{config.TRADING_MODE}.png (전략별 개별 + 매매 시점)")
        print(f"  - monthly_heatmap_{config.TRADING_MODE}.png")
        print(f"  - drawdown_{config.TRADING_MODE}.png")
        print(f"  - portfolio_composition_{config.TRADING_MODE}_*.png (전략별)")
        print(f"  - performance_attribution_{config.TRADING_MODE}.png")
        print("\n모든 프로세스가 완료되었습니다.")
        print(f"\n📋 로그 파일이 저장되었습니다: {log_file_path}")
    
    finally:
        # stdout 복원 및 로그 파일 닫기
        sys.stdout = original_stdout
        log_file.close()
        print(f"\n✅ 로그 파일 저장 완료: {log_file_path}")

if __name__ == '__main__':
    main()