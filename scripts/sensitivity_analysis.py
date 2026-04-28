"""
민감도 분석: 백신 효과 평가를 위한 다양한 분석 전략

1. Landmark Analysis: 특정 시점 이후부터 분석 시작
2. 추적 기간 제한: 특정 기간까지만 분석
3. 시점별 HR 변화 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

# statsmodels
try:
    from statsmodels.duration.hazard_regression import PHReg
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def load_outcomes(data_dir: Path) -> pd.DataFrame:
    """결과 데이터 로드"""
    outcomes = pd.read_csv(data_dir / 'final_matched_outcomes.csv', encoding='utf-8-sig')
    outcomes['index_date'] = pd.to_datetime(outcomes['index_date'])
    return outcomes


def run_cox_univariate(df: pd.DataFrame, time_col: str, event_col: str) -> dict:
    """단변량 Cox 분석"""
    analysis_df = df[['vaccinated', time_col, event_col]].dropna()
    analysis_df = analysis_df[analysis_df[time_col] > 0]

    if len(analysis_df) < 10 or analysis_df[event_col].sum() < 3:
        return None

    if not HAS_STATSMODELS:
        return None

    try:
        model = PHReg(analysis_df[time_col],
                     analysis_df[['vaccinated']],
                     status=analysis_df[event_col])
        result = model.fit(disp=False)

        coef = result.params[0]
        se = result.bse[0]
        hr = np.exp(coef)
        ci_lower = np.exp(coef - 1.96 * se)
        ci_upper = np.exp(coef + 1.96 * se)
        p_value = result.pvalues[0]

        return {
            'HR': hr,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_value,
            'n': len(analysis_df),
            'events': int(analysis_df[event_col].sum()),
            'events_vax': int(analysis_df[analysis_df['vaccinated']==1][event_col].sum()),
            'events_unvax': int(analysis_df[analysis_df['vaccinated']==0][event_col].sum())
        }
    except Exception as e:
        return None


def landmark_analysis(outcomes: pd.DataFrame, outcome_var: str, time_var: str,
                      landmark_days: list) -> pd.DataFrame:
    """
    Landmark Analysis: 특정 시점 이후 생존자만 대상으로 분석

    - 백신 효과가 지연되어 나타나는지 확인
    - 초기 이벤트 제외하고 장기 효과 평가
    """
    results = []

    outcomes['vaccinated'] = outcomes['접종여부'].astype(int)

    for landmark in landmark_days:
        # Landmark 시점까지 생존한 환자만
        landmark_df = outcomes[outcomes[time_var] > landmark].copy()

        # 시간 조정 (landmark부터 시작)
        landmark_df['adjusted_time'] = landmark_df[time_var] - landmark

        if len(landmark_df) < 20:
            continue

        result = run_cox_univariate(landmark_df, 'adjusted_time', outcome_var)

        if result:
            result['landmark_days'] = landmark
            result['landmark_years'] = landmark / 365
            results.append(result)

    return pd.DataFrame(results)


def restricted_followup_analysis(outcomes: pd.DataFrame, outcome_var: str,
                                  time_var: str, max_followup_days: list) -> pd.DataFrame:
    """
    추적 기간 제한 분석: 특정 기간까지만 분석

    - 단기 vs 장기 효과 비교
    - 늦은 이벤트 제외하고 조기 효과 평가
    """
    results = []

    outcomes['vaccinated'] = outcomes['접종여부'].astype(int)

    for max_days in max_followup_days:
        restricted_df = outcomes.copy()

        # 추적 기간 제한
        restricted_df['restricted_time'] = restricted_df[time_var].clip(upper=max_days)

        # 이벤트 조정: max_days 이후 이벤트는 censoring으로 처리
        restricted_df['restricted_event'] = restricted_df[outcome_var].astype(int)
        restricted_df.loc[restricted_df[time_var] > max_days, 'restricted_event'] = 0

        result = run_cox_univariate(restricted_df, 'restricted_time', 'restricted_event')

        if result:
            result['max_followup_days'] = max_days
            result['max_followup_years'] = max_days / 365
            results.append(result)

    return pd.DataFrame(results)


def time_varying_effect(outcomes: pd.DataFrame, outcome_var: str,
                        time_var: str, time_points: list) -> pd.DataFrame:
    """
    시점별 HR 변화 분석: 특정 구간 내 이벤트만 분석

    - 조기 효과 vs 후기 효과 비교
    """
    results = []

    outcomes['vaccinated'] = outcomes['접종여부'].astype(int)

    for i in range(len(time_points) - 1):
        start_days = time_points[i]
        end_days = time_points[i + 1]

        # 해당 기간 내 이벤트만
        period_df = outcomes.copy()

        # 시작 시점 이전 이벤트는 제외 (해당 시점까지 생존자만)
        period_df = period_df[period_df[time_var] > start_days]

        # 종료 시점까지의 시간으로 조정
        period_df['period_time'] = (period_df[time_var] - start_days).clip(upper=end_days - start_days)

        # 이벤트 조정
        period_df['period_event'] = period_df[outcome_var].astype(int)
        period_df.loc[period_df[time_var] > end_days, 'period_event'] = 0
        period_df.loc[period_df[time_var] <= start_days, 'period_event'] = 0

        if len(period_df) < 20 or period_df['period_event'].sum() < 3:
            continue

        result = run_cox_univariate(period_df, 'period_time', 'period_event')

        if result:
            result['period_start'] = start_days
            result['period_end'] = end_days
            result['period_label'] = f"{start_days//365}-{end_days//365}년"
            results.append(result)

    return pd.DataFrame(results)


def plot_sensitivity_results(landmark_results: pd.DataFrame,
                             restricted_results: pd.DataFrame,
                             period_results: pd.DataFrame,
                             outcome_name: str, output_path: Path):
    """민감도 분석 결과 시각화"""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Landmark Analysis
    ax1 = axes[0]
    if len(landmark_results) > 0:
        x = landmark_results['landmark_years']
        y = landmark_results['HR']
        ci_lower = landmark_results['CI_lower']
        ci_upper = landmark_results['CI_upper']

        ax1.errorbar(x, y, yerr=[y - ci_lower, ci_upper - y],
                    fmt='o-', capsize=5, color='blue', markersize=8)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Landmark (years)')
        ax1.set_ylabel('Hazard Ratio')
        ax1.set_title(f'{outcome_name}\nLandmark Analysis')

        # p-value 표시
        for i, row in landmark_results.iterrows():
            sig = '*' if row['p_value'] < 0.05 else ''
            ax1.annotate(f"p={row['p_value']:.3f}{sig}",
                        (row['landmark_years'], row['HR']),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'{outcome_name}\nLandmark Analysis')

    # 2. Restricted Follow-up
    ax2 = axes[1]
    if len(restricted_results) > 0:
        x = restricted_results['max_followup_years']
        y = restricted_results['HR']
        ci_lower = restricted_results['CI_lower']
        ci_upper = restricted_results['CI_upper']

        ax2.errorbar(x, y, yerr=[y - ci_lower, ci_upper - y],
                    fmt='s-', capsize=5, color='green', markersize=8)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Max Follow-up (years)')
        ax2.set_ylabel('Hazard Ratio')
        ax2.set_title(f'{outcome_name}\nRestricted Follow-up Analysis')

        for i, row in restricted_results.iterrows():
            sig = '*' if row['p_value'] < 0.05 else ''
            ax2.annotate(f"p={row['p_value']:.3f}{sig}",
                        (row['max_followup_years'], row['HR']),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{outcome_name}\nRestricted Follow-up Analysis')

    # 3. Time-varying Effect
    ax3 = axes[2]
    if len(period_results) > 0:
        x = range(len(period_results))
        y = period_results['HR']
        ci_lower = period_results['CI_lower']
        ci_upper = period_results['CI_upper']

        ax3.errorbar(x, y, yerr=[y - ci_lower, ci_upper - y],
                    fmt='D-', capsize=5, color='purple', markersize=8)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(period_results['period_label'])
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Hazard Ratio')
        ax3.set_title(f'{outcome_name}\nTime-varying Effect')

        for i, row in period_results.iterrows():
            idx = list(period_results.index).index(i)
            sig = '*' if row['p_value'] < 0.05 else ''
            ax3.annotate(f"p={row['p_value']:.3f}{sig}",
                        (idx, row['HR']),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'{outcome_name}\nTime-varying Effect')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return fig


def print_results_table(results: pd.DataFrame, title: str):
    """결과 테이블 출력"""
    print(f"\n  {title}")
    print("  " + "-" * 80)

    if len(results) == 0:
        print("  분석 결과 없음")
        return

    for _, row in results.iterrows():
        sig = "*" if row['p_value'] < 0.05 else " "
        events_str = f"{row['events_vax']}/{row['events_unvax']}"
        print(f"  HR={row['HR']:.3f} (95% CI: {row['CI_lower']:.3f}-{row['CI_upper']:.3f}) "
              f"p={row['p_value']:.4f}{sig} | n={row['n']}, events={row['events']} ({events_str})")


def main():
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 80)
    print("민감도 분석: 백신 효과 평가를 위한 다양한 분석 전략")
    print("=" * 80)

    # 데이터 로드
    outcomes = load_outcomes(data_dir)
    print(f"\n분석 대상: {len(outcomes)}명")

    # 시간 변수 생성
    outcomes['time_to_recurrence'] = outcomes['days_to_recurrence'].fillna(outcomes['follow_up_days'])
    outcomes['time_to_hpv'] = outcomes['days_to_hpv'].fillna(outcomes['follow_up_days'])

    # 분석 설정
    landmark_days = [0, 180, 365, 730, 1095]  # 0, 6개월, 1년, 2년, 3년
    max_followup_days = [365, 730, 1095, 1460, 1825, 2190]  # 1~6년
    time_periods = [0, 365, 730, 1095, 1825, 3650]  # 0-1, 1-2, 2-3, 3-5, 5-10년

    all_results = {}

    # ===== Biopsy 기반 병변 재발 분석 =====
    print("\n" + "=" * 80)
    print("[1. Biopsy 기반 병변 재발 (has_recurrence)]")
    print("=" * 80)

    print(f"\n  기본 통계:")
    print(f"  - 전체 이벤트: {outcomes['has_recurrence'].sum()}건 ({outcomes['has_recurrence'].mean()*100:.1f}%)")
    vax = outcomes[outcomes['접종여부'] == True]
    unvax = outcomes[outcomes['접종여부'] == False]
    print(f"  - 접종군: {vax['has_recurrence'].sum()}/{len(vax)} ({vax['has_recurrence'].mean()*100:.1f}%)")
    print(f"  - 비접종군: {unvax['has_recurrence'].sum()}/{len(unvax)} ({unvax['has_recurrence'].mean()*100:.1f}%)")

    # Landmark Analysis
    print("\n  [1-1. Landmark Analysis]")
    print("  (특정 시점 이후 생존자만 대상으로 분석)")
    landmark_rec = landmark_analysis(outcomes, 'has_recurrence', 'time_to_recurrence', landmark_days)
    print_results_table(landmark_rec, "Landmark별 HR")

    # Restricted Follow-up
    print("\n  [1-2. Restricted Follow-up Analysis]")
    print("  (특정 기간까지만 분석)")
    restricted_rec = restricted_followup_analysis(outcomes, 'has_recurrence', 'time_to_recurrence', max_followup_days)
    print_results_table(restricted_rec, "추적기간별 HR")

    # Time-varying Effect
    print("\n  [1-3. Time-varying Effect]")
    print("  (기간별 HR 변화)")
    period_rec = time_varying_effect(outcomes, 'has_recurrence', 'time_to_recurrence', time_periods)
    print_results_table(period_rec, "기간별 HR")

    # 시각화
    plot_sensitivity_results(landmark_rec, restricted_rec, period_rec,
                            'Biopsy Recurrence', data_dir / 'sensitivity_recurrence.png')
    print(f"\n  - 그래프 저장: {data_dir / 'sensitivity_recurrence.png'}")

    all_results['recurrence'] = {
        'landmark': landmark_rec,
        'restricted': restricted_rec,
        'period': period_rec
    }

    # ===== HPV 재감염 분석 =====
    print("\n" + "=" * 80)
    print("[2. HPV 재감염 (has_hpv_infection)]")
    print("=" * 80)

    print(f"\n  기본 통계:")
    print(f"  - 전체 이벤트: {outcomes['has_hpv_infection'].sum()}건 ({outcomes['has_hpv_infection'].mean()*100:.1f}%)")
    print(f"  - 접종군: {vax['has_hpv_infection'].sum()}/{len(vax)} ({vax['has_hpv_infection'].mean()*100:.1f}%)")
    print(f"  - 비접종군: {unvax['has_hpv_infection'].sum()}/{len(unvax)} ({unvax['has_hpv_infection'].mean()*100:.1f}%)")

    # Landmark Analysis
    print("\n  [2-1. Landmark Analysis]")
    landmark_hpv = landmark_analysis(outcomes, 'has_hpv_infection', 'time_to_hpv', landmark_days)
    print_results_table(landmark_hpv, "Landmark별 HR")

    # Restricted Follow-up
    print("\n  [2-2. Restricted Follow-up Analysis]")
    restricted_hpv = restricted_followup_analysis(outcomes, 'has_hpv_infection', 'time_to_hpv', max_followup_days)
    print_results_table(restricted_hpv, "추적기간별 HR")

    # Time-varying Effect
    print("\n  [2-3. Time-varying Effect]")
    period_hpv = time_varying_effect(outcomes, 'has_hpv_infection', 'time_to_hpv', time_periods)
    print_results_table(period_hpv, "기간별 HR")

    # 시각화
    plot_sensitivity_results(landmark_hpv, restricted_hpv, period_hpv,
                            'HPV Reinfection', data_dir / 'sensitivity_hpv.png')
    print(f"\n  - 그래프 저장: {data_dir / 'sensitivity_hpv.png'}")

    all_results['hpv'] = {
        'landmark': landmark_hpv,
        'restricted': restricted_hpv,
        'period': period_hpv
    }

    # ===== 유의한 결과 요약 =====
    print("\n" + "=" * 80)
    print("[3. 유의한 결과 요약 (p < 0.05)]")
    print("=" * 80)

    significant_found = False

    for outcome_name, outcome_results in all_results.items():
        for analysis_type, df in outcome_results.items():
            if len(df) > 0:
                sig_results = df[df['p_value'] < 0.05]
                if len(sig_results) > 0:
                    significant_found = True
                    print(f"\n  [{outcome_name.upper()} - {analysis_type}]")
                    for _, row in sig_results.iterrows():
                        print(f"  HR={row['HR']:.3f} (95% CI: {row['CI_lower']:.3f}-{row['CI_upper']:.3f}) "
                              f"p={row['p_value']:.4f}* | events={row['events']}")

    if not significant_found:
        print("\n  유의한 결과 없음 (모든 p > 0.05)")

    # 가장 낮은 HR 찾기
    print("\n" + "=" * 80)
    print("[4. 가장 보호 효과가 큰 분석 조건 (최저 HR)]")
    print("=" * 80)

    for outcome_name, outcome_results in all_results.items():
        print(f"\n  [{outcome_name.upper()}]")
        all_hrs = []
        for analysis_type, df in outcome_results.items():
            if len(df) > 0:
                for _, row in df.iterrows():
                    all_hrs.append({
                        'analysis': analysis_type,
                        'HR': row['HR'],
                        'p_value': row['p_value'],
                        'CI_lower': row['CI_lower'],
                        'CI_upper': row['CI_upper'],
                        'events': row['events']
                    })

        if all_hrs:
            best = min(all_hrs, key=lambda x: x['HR'])
            print(f"  최저 HR: {best['HR']:.3f} (95% CI: {best['CI_lower']:.3f}-{best['CI_upper']:.3f})")
            print(f"  분석 유형: {best['analysis']}, p={best['p_value']:.4f}, events={best['events']}")

    print("\n" + "=" * 80)
    print("민감도 분석 완료!")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = main()
