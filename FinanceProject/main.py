# main.py
import config
from modules import backtest_engine
from modules import performance_analyzer
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("="*50)
    print("퀀트 투자 전략 백테스팅 및 분석을 시작합니다.")
    print(f"분석 기간: {config.START_DATE} ~ {config.END_DATE}")
    print(f"매매 모드: {config.TRADING_MODE}")
    print("="*50)
    
    # 1. 백테스팅 실행
    final_values, monthly_portfolios = backtest_engine.run_backtest(
        config.START_DATE, 
        config.END_DATE, 
        config
    )
    
    if final_values.empty:
        print("\n백테스팅 결과가 비어있습니다. 프로세스를 종료합니다.")
        return
        
    # 2. 성과 분석 및 결과 저장
    kpi_summary = performance_analyzer.analyze_performance(final_values)
    
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
    
    # 4-1. 누적 수익률 차트
    print("  [1/5] 누적 수익률 차트 생성...")
    performance_analyzer.plot_cumulative_returns(
        final_values, 
        os.path.join(results_dir, f'cumulative_returns_{config.TRADING_MODE}.png')
    )
    
    # 4-2. 월별 수익률 히트맵
    print("  [2/5] 월별 수익률 히트맵 생성...")
    performance_analyzer.plot_monthly_returns_heatmap(
        monthly_df,
        os.path.join(results_dir, f'monthly_heatmap_{config.TRADING_MODE}.png')
    )
    
    # 4-3. 드로우다운 차트
    print("  [3/5] 드로우다운 차트 생성...")
    performance_analyzer.plot_drawdown_chart(
        final_values,
        os.path.join(results_dir, f'drawdown_{config.TRADING_MODE}.png')
    )
    
    # 4-4. 월별 포트폴리오 구성 파이 차트
    print("  [4/5] 월별 포트폴리오 구성 파이 차트 생성...")
    performance_analyzer.plot_portfolio_composition_pies(
        monthly_portfolios,
        os.path.join(results_dir, f'portfolio_composition_{config.TRADING_MODE}.png')
    )
    
    # 4-5. 이동 평균 성과 차트
    print("  [5/5] 이동 평균 성과 차트 생성...")
    performance_analyzer.plot_rolling_performance(
        final_values,
        window=60,
        filepath=os.path.join(results_dir, f'rolling_performance_{config.TRADING_MODE}.png')
    )
    
    print(f"\n'{results_dir}' 폴더에 모든 분석 결과가 저장되었습니다.")
    print("\n생성된 파일 목록:")
    print(f"  - kpi_summary_{config.TRADING_MODE}.csv")
    print(f"  - monthly_stats_{config.TRADING_MODE}.csv")
    print(f"  - cumulative_returns_{config.TRADING_MODE}.png")
    print(f"  - monthly_heatmap_{config.TRADING_MODE}.png")
    print(f"  - drawdown_{config.TRADING_MODE}.png")
    print(f"  - portfolio_composition_{config.TRADING_MODE}_*.png (전략별)")
    print(f"  - rolling_performance_{config.TRADING_MODE}.png")
    print("\n모든 프로세스가 완료되었습니다.")

if __name__ == '__main__':
    main()