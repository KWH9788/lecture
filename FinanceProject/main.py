# main.py
import config
from modules import backtest_engine
from modules import performance_analyzer
from modules import data_loader
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*50)
    print("퀀트 투자 전략 백테스팅 및 분석을 시작합니다.")
    print(f"분석 기간: {config.START_DATE} ~ {config.END_DATE}")
    print("="*50)
    
    # 1. 백테스팅 실행
    final_values = backtest_engine.run_backtest(
        config.START_DATE, 
        config.END_DATE, 
        config
    )
    
    # 2. 성과 분석 및 결과 저장
    kpi_summary = performance_analyzer.analyze_performance(final_values)
    kpi_summary.to_csv('results/kpi_summary.csv', encoding='utf-8-sig')
    
    print("\n--- 핵심 성과 지표 (KPI) ---")
    print(kpi_summary)
    
    # 3. 시각화 자료 저장
    performance_analyzer.plot_cumulative_returns(
        final_values, 
        'results/cumulative_returns.png'
    )
    print("\n'results' 폴더에 분석 결과(kpi_summary.csv, cumulative_returns.png)가 저장되었습니다.")
    print("\n모든 프로세스가 완료되었습니다.")

if __name__ == '__main__':
    # results 폴더가 없으면 생성
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
        
    main()