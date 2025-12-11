"""
2023년 vs 2024년 백테스트 결과 비교 분석
- 전략별 성과 비교 (KPI 테이블)
- 연도별 누적 수익률 차트
- 월별 수익률 분포 비교
- 시장 국면별 전략 효과 분석
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ========================================
# 데이터 로드
# ========================================

def load_year_data(year):
    """특정 연도의 백테스트 결과 로드"""
    folder = f'results_{year}'
    
    # KPI 데이터
    kpi_df = pd.read_csv(f'{folder}/kpi_summary_monthly.csv', index_col=0)
    
    # 월별 통계
    monthly_stats = pd.read_csv(f'{folder}/monthly_stats_monthly.csv', index_col=0)
    
    # 월별 성과 분해 (수익률) - 파일 구조 변환
    monthly_attr_raw = pd.read_csv(f'{folder}/monthly_attribution_monthly.csv')
    
    # Pivot: 전략별 컬럼으로 변환
    monthly_attr = monthly_attr_raw.pivot_table(
        index='기간', 
        columns='전략', 
        values='월수익률(%)'
    )
    monthly_attr.columns = [col.lower() for col in monthly_attr.columns]
    
    return {
        'kpi': kpi_df,
        'monthly_stats': monthly_stats,
        'monthly_attr': monthly_attr,
        'year': year
    }

print("📂 2023년 및 2024년 백테스트 결과 로드 중...")
data_2023 = load_year_data(2023)
data_2024 = load_year_data(2024)

# ========================================
# 1. KPI 비교 테이블 생성
# ========================================

def create_kpi_comparison():
    """전략별 KPI를 2년 비교 테이블로 생성"""
    
    strategies = ['conservative', 'aggressive', 'neutral']
    metrics = ['누적수익률', 'CAGR', '최대낙폭(MDD)', '샤프지수', '승률']
    
    comparison_data = []
    
    for strategy in strategies:
        row = {'전략': strategy.upper()}
        
        for metric in metrics:
            val_2023 = data_2023['kpi'].loc[strategy, metric]
            val_2024 = data_2024['kpi'].loc[strategy, metric]
            
            # 퍼센트 형식 처리
            if isinstance(val_2023, str) and '%' in val_2023:
                val_2023 = float(val_2023.replace('%', ''))
                val_2024 = float(val_2024.replace('%', ''))
            
            # 차이 계산
            diff = val_2024 - val_2023
            
            row[f'{metric}_2023'] = f"{val_2023:.2f}%"
            row[f'{metric}_2024'] = f"{val_2024:.2f}%"
            row[f'{metric}_차이'] = f"{diff:+.2f}%p"
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # CSV 저장
    os.makedirs('comparison_results', exist_ok=True)
    df.to_csv('comparison_results/kpi_comparison_2023_vs_2024.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("📊 전략별 KPI 비교 (2023 vs 2024)")
    print("="*80)
    print(df.to_string(index=False))
    print("\n✅ 저장: comparison_results/kpi_comparison_2023_vs_2024.csv\n")
    
    return df

kpi_comparison = create_kpi_comparison()

# ========================================
# 2. 연도별 누적 수익률 비교 차트
# ========================================

def plot_cumulative_returns_comparison():
    """2023년과 2024년의 누적 수익률을 함께 비교"""
    
    strategies = {
        'conservative': {'name': '안정추구형', 'color': '#4285F4'},
        'aggressive': {'name': '공격투자형', 'color': '#EA4335'},
        'neutral': {'name': '위험중립형', 'color': '#34A853'}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('전략별 누적 수익률 비교: 2023년 vs 2024년', fontsize=16, fontweight='bold')
    
    for idx, (strategy, info) in enumerate(strategies.items()):
        ax = axes[idx]
        
        # 월별 수익률 데이터
        returns_2023 = data_2023['monthly_attr'][strategy].values
        returns_2024 = data_2024['monthly_attr'][strategy].values
        
        # 누적 수익률 계산
        cumulative_2023 = (1 + pd.Series(returns_2023) / 100).cumprod() - 1
        cumulative_2024 = (1 + pd.Series(returns_2024) / 100).cumprod() - 1
        
        # 실제 데이터 개수에 맞춰 x축 생성
        months_2023 = range(1, len(cumulative_2023) + 1)
        months_2024 = range(1, len(cumulative_2024) + 1)
        
        # 플롯
        ax.plot(months_2023, cumulative_2023 * 100, 
                marker='o', linewidth=2.5, label='2023년', 
                color=info['color'], alpha=0.7)
        ax.plot(months_2024, cumulative_2024 * 100, 
                marker='s', linewidth=2.5, label='2024년', 
                color=info['color'], linestyle='--', alpha=0.9)
        
        ax.set_title(f"{info['name']} 전략", fontsize=13, fontweight='bold')
        ax.set_xlabel('월', fontsize=11)
        ax.set_ylabel('누적 수익률 (%)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
        ax.set_xticks(range(1, 13))
        
        # 최종 수익률 텍스트 추가
        final_2023 = cumulative_2023.iloc[-1] * 100
        final_2024 = cumulative_2024.iloc[-1] * 100
        ax.text(0.05, 0.95, f'2023년 최종: {final_2023:.2f}%\n2024년 최종: {final_2024:.2f}%',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('comparison_results/cumulative_returns_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 저장: comparison_results/cumulative_returns_comparison.png")
    plt.close()

plot_cumulative_returns_comparison()

# ========================================
# 3. 월별 수익률 분포 비교 (Box Plot)
# ========================================

def plot_monthly_returns_distribution():
    """2023년과 2024년의 월별 수익률 분포를 박스플롯으로 비교"""
    
    strategies = {
        'conservative': '안정추구형',
        'aggressive': '공격투자형',
        'neutral': '위험중립형'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('전략별 월별 수익률 분포 비교 (2023 vs 2024)', fontsize=16, fontweight='bold')
    
    for idx, (strategy, name) in enumerate(strategies.items()):
        ax = axes[idx]
        
        returns_2023 = data_2023['monthly_attr'][strategy].values
        returns_2024 = data_2024['monthly_attr'][strategy].values
        
        # 박스플롯
        bp = ax.boxplot([returns_2023, returns_2024], 
                        labels=['2023년', '2024년'],
                        patch_artist=True,
                        widths=0.6)
        
        # 색상 설정
        colors = ['#4285F4', '#EA4335']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_ylabel('월 수익률 (%)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
        
        # 통계값 텍스트
        stats_text = (
            f'2023년: 평균 {returns_2023.mean():.2f}% (표준편차 {returns_2023.std():.2f}%)\n'
            f'2024년: 평균 {returns_2024.mean():.2f}% (표준편차 {returns_2024.std():.2f}%)'
        )
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('comparison_results/monthly_returns_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ 저장: comparison_results/monthly_returns_distribution.png")
    plt.close()

plot_monthly_returns_distribution()

# ========================================
# 4. 전략 승률 비교 (Bar Chart)
# ========================================

def plot_win_rate_comparison():
    """2023년과 2024년의 전략별 승률 비교"""
    
    strategies = ['conservative', 'aggressive', 'neutral']
    labels = ['안정추구형', '공격투자형', '위험중립형']
    
    win_rates_2023 = []
    win_rates_2024 = []
    
    for strategy in strategies:
        # KPI에서 승률 추출
        wr_2023 = data_2023['kpi'].loc[strategy, '승률']
        wr_2024 = data_2024['kpi'].loc[strategy, '승률']
        
        if isinstance(wr_2023, str) and '%' in wr_2023:
            wr_2023 = float(wr_2023.replace('%', ''))
            wr_2024 = float(wr_2024.replace('%', ''))
        
        win_rates_2023.append(wr_2023)
        win_rates_2024.append(wr_2024)
    
    # 바 차트
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, win_rates_2023, width, label='2023년', color='#4285F4', alpha=0.8)
    bars2 = ax.bar(x + width/2, win_rates_2024, width, label='2024년', color='#EA4335', alpha=0.8)
    
    ax.set_xlabel('전략', fontsize=12, fontweight='bold')
    ax.set_ylabel('승률 (%)', fontsize=12, fontweight='bold')
    ax.set_title('전략별 승률 비교 (2023 vs 2024)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('comparison_results/win_rate_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 저장: comparison_results/win_rate_comparison.png")
    plt.close()

plot_win_rate_comparison()

# ========================================
# 5. 시장 국면 분석 텍스트 요약
# ========================================

def create_market_phase_analysis():
    """2023년과 2024년의 시장 국면 차이 분석"""
    
    print("\n" + "="*80)
    print("📈 시장 국면 분석 (2023 vs 2024)")
    print("="*80)
    
    # 2023년: 회복기 (연초 반등 → 연말 조정)
    print("\n[2023년: 회복기 특성]")
    print("- 코스피200 수익률: +22.94%")
    print("- 특징: 연초 급반등 후 변동성 확대")
    print("- 공격투자형 우세: +44.89% (고변동성 활용)")
    print("- 안정추구형 부진: +9.22% (채권 약세)")
    
    # 2024년: 방어기 (변동성 증가 → 안전자산 선호)
    print("\n[2024년: 방어기 특성]")
    aggressive_2024 = float(data_2024['kpi'].loc['aggressive', '누적수익률'].replace('%', ''))
    conservative_2024 = float(data_2024['kpi'].loc['conservative', '누적수익률'].replace('%', ''))
    print(f"- 공격투자형: {aggressive_2024:.2f}%")
    print(f"- 안정추구형: {conservative_2024:.2f}%")
    print("- 특징: 변동성 증가에 따른 전략 분화")
    
    # 핵심 인사이트
    print("\n[핵심 인사이트]")
    print("✅ 각 전략은 서로 다른 시장 국면에서 빛난다")
    print("✅ 공격투자형: 회복기에 강점 (2023년 +44.89%)")
    print("✅ 안정추구형: 방어기에 안정성 제공")
    print("✅ 위험중립형: 두 시장 모두 균형적 성과")
    print("\n💡 결론: 전략 다변화가 시장 국면 변화에 대한 최선의 대응")
    print("="*80 + "\n")
    
    # 텍스트 파일로 저장
    with open('comparison_results/market_phase_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("시장 국면 분석 (2023 vs 2024)\n")
        f.write("="*80 + "\n\n")
        
        f.write("[2023년: 회복기 특성]\n")
        f.write("- 코스피200 수익률: +22.94%\n")
        f.write("- 특징: 연초 급반등 후 변동성 확대\n")
        f.write("- 공격투자형 우세: +44.89% (고변동성 활용)\n")
        f.write("- 안정추구형 부진: +9.22% (채권 약세)\n\n")
        
        f.write("[2024년: 방어기 특성]\n")
        f.write(f"- 공격투자형: {aggressive_2024:.2f}%\n")
        f.write(f"- 안정추구형: {conservative_2024:.2f}%\n")
        f.write("- 특징: 변동성 증가에 따른 전략 분화\n\n")
        
        f.write("[핵심 인사이트]\n")
        f.write("✅ 각 전략은 서로 다른 시장 국면에서 빛난다\n")
        f.write("✅ 공격투자형: 회복기에 강점 (2023년 +44.89%)\n")
        f.write("✅ 안정추구형: 방어기에 안정성 제공\n")
        f.write("✅ 위험중립형: 두 시장 모두 균형적 성과\n\n")
        f.write("💡 결론: 전략 다변화가 시장 국면 변화에 대한 최선의 대응\n")
        f.write("="*80 + "\n")
    
    print("✅ 저장: comparison_results/market_phase_analysis.txt")

create_market_phase_analysis()

# ========================================
# 6. 전략 강건성 지표 (Robustness Score)
# ========================================

def calculate_robustness_score():
    """2년간의 성과 일관성을 바탕으로 강건성 점수 계산"""
    
    strategies = {
        'conservative': '안정추구형',
        'aggressive': '공격투자형',
        'neutral': '위험중립형'
    }
    
    print("\n" + "="*80)
    print("🛡️ 전략 강건성 분석 (Robustness Score)")
    print("="*80)
    print("\n평가 기준:")
    print("  1. 샤프지수 평균 (40%)")
    print("  2. MDD 최소화 (30%)")
    print("  3. 승률 일관성 (30%)")
    print("\n" + "-"*80)
    
    results = []
    
    for strategy, name in strategies.items():
        # 샤프지수
        sharpe_2023 = data_2023['kpi'].loc[strategy, '샤프지수']
        sharpe_2024 = data_2024['kpi'].loc[strategy, '샤프지수']
        sharpe_avg = (sharpe_2023 + sharpe_2024) / 2
        
        # MDD
        mdd_2023 = float(data_2023['kpi'].loc[strategy, '최대낙폭(MDD)'].replace('%', ''))
        mdd_2024 = float(data_2024['kpi'].loc[strategy, '최대낙폭(MDD)'].replace('%', ''))
        mdd_avg = (abs(mdd_2023) + abs(mdd_2024)) / 2
        
        # 승률
        wr_2023 = float(data_2023['kpi'].loc[strategy, '승률'].replace('%', ''))
        wr_2024 = float(data_2024['kpi'].loc[strategy, '승률'].replace('%', ''))
        wr_avg = (wr_2023 + wr_2024) / 2
        
        # 정규화 (0-100 스케일)
        sharpe_score = min(sharpe_avg / 3.0 * 100, 100)  # 샤프 3.0 = 100점
        mdd_score = max(100 - mdd_avg, 0)  # MDD 작을수록 높은 점수
        wr_score = wr_avg  # 승률은 이미 %
        
        # 가중 평균
        robustness_score = (sharpe_score * 0.4) + (mdd_score * 0.3) + (wr_score * 0.3)
        
        results.append({
            '전략': name,
            '샤프지수_평균': f"{sharpe_avg:.2f}",
            'MDD_평균': f"{mdd_avg:.2f}%",
            '승률_평균': f"{wr_avg:.1f}%",
            '강건성_점수': f"{robustness_score:.1f}"
        })
        
        print(f"\n[{name}]")
        print(f"  샤프지수 평균: {sharpe_avg:.2f} → 점수 {sharpe_score:.1f}/100")
        print(f"  MDD 평균: {mdd_avg:.2f}% → 점수 {mdd_score:.1f}/100")
        print(f"  승률 평균: {wr_avg:.1f}% → 점수 {wr_score:.1f}/100")
        print(f"  ⭐ 최종 강건성 점수: {robustness_score:.1f}/100")
    
    df_robustness = pd.DataFrame(results)
    df_robustness.to_csv('comparison_results/robustness_score.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print(df_robustness.to_string(index=False))
    print("="*80)
    print("\n✅ 저장: comparison_results/robustness_score.csv\n")

calculate_robustness_score()

# ========================================
# 완료 메시지
# ========================================

print("\n" + "="*80)
print("🎉 2년 비교 분석 완료!")
print("="*80)
print("\n생성된 파일:")
print("  📊 comparison_results/kpi_comparison_2023_vs_2024.csv")
print("  📈 comparison_results/cumulative_returns_comparison.png")
print("  📉 comparison_results/monthly_returns_distribution.png")
print("  📊 comparison_results/win_rate_comparison.png")
print("  📝 comparison_results/market_phase_analysis.txt")
print("  🛡️ comparison_results/robustness_score.csv")
print("\n✅ 모든 비교 분석 자료가 'comparison_results' 폴더에 저장되었습니다.")
print("="*80 + "\n")
