"""
백신 종류별 Subgroup Analysis

HPV 백신 종류(Gardasil9, Cervarix, Gardasil)에 따른
자궁경부 병변 재발 및 HPV 재감염 예방 효과 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.duration.hazard_regression import PHReg
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 재현성을 위한 시드 고정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data(data_dir: Path):
    """데이터 로드 및 병합"""
    final = pd.read_csv(data_dir / 'final_matched_cohort.csv', encoding='utf-8-sig')
    outcomes = pd.read_csv(data_dir / 'final_matched_outcomes.csv', encoding='utf-8-sig')

    # 백신 종류 및 수술-접종 간격 정보 병합
    outcomes_merged = outcomes.merge(
        final[['연구번호', '백신종류', '수술_접종_간격일']],
        on='연구번호',
        how='left'
    )

    return outcomes_merged


def run_cox_by_vaccine(df, vaccine_type, time_var, event_var):
    """특정 백신 종류 vs 비접종군 Cox 분석"""

    vacc_group = df[(df['접종여부'] == True) & (df['백신종류'] == vaccine_type)]
    unvacc_group = df[df['접종여부'] == False]

    # 매칭된 비접종군만 사용
    matched_ids = vacc_group['fine_match_id'].unique()
    unvacc_matched = unvacc_group[unvacc_group['fine_match_id'].isin(matched_ids)]

    analysis_df = pd.concat([vacc_group, unvacc_matched]).copy()
    analysis_df['vaccinated'] = analysis_df['접종여부'].astype(int)
    analysis_df['event'] = analysis_df[event_var].astype(int)
    analysis_df['days_to_event'] = analysis_df.apply(
        lambda x: x[time_var] if pd.notna(x[time_var]) else x['follow_up_days'], axis=1
    )
    analysis_df = analysis_df[analysis_df['days_to_event'] > 0]
    analysis_df = analysis_df.dropna(subset=['index_age'])

    if len(analysis_df) < 20 or analysis_df['event'].sum() < 3:
        return None

    try:
        exog = analysis_df[['vaccinated', 'index_age']]
        model = PHReg(analysis_df['days_to_event'], exog, status=analysis_df['event'])
        result = model.fit(disp=False)

        hr = np.exp(result.params[0])
        ci_l = np.exp(result.params[0] - 1.96 * result.bse[0])
        ci_u = np.exp(result.params[0] + 1.96 * result.bse[0])

        return {
            'vaccine': vaccine_type,
            'n_vacc': len(vacc_group),
            'n_unvacc': len(unvacc_matched),
            'n_total': len(analysis_df),
            'events_vacc': int(vacc_group[event_var].sum()),
            'events_unvacc': int(unvacc_matched[event_var].sum()),
            'events_total': int(analysis_df['event'].sum()),
            'HR': hr,
            'CI_lower': ci_l,
            'CI_upper': ci_u,
            'p_value': result.pvalues[0]
        }
    except Exception as e:
        print(f"Error for {vaccine_type}: {e}")
        return None


def analyze_vaccination_interval(df):
    """수술-접종 간격 분석"""
    vacc = df[df['접종여부'] == True]

    results = {
        'overall': {
            'mean_days': vacc['수술_접종_간격일'].mean(),
            'median_days': vacc['수술_접종_간격일'].median(),
            'std_days': vacc['수술_접종_간격일'].std(),
            'min_days': vacc['수술_접종_간격일'].min(),
            'max_days': vacc['수술_접종_간격일'].max(),
            'n': len(vacc)
        }
    }

    for vtype in ['Gardasil9', 'Cervarix', 'Gardasil']:
        vtype_data = vacc[vacc['백신종류'] == vtype]
        if len(vtype_data) > 0:
            results[vtype] = {
                'mean_days': vtype_data['수술_접종_간격일'].mean(),
                'median_days': vtype_data['수술_접종_간격일'].median(),
                'std_days': vtype_data['수술_접종_간격일'].std(),
                'n': len(vtype_data)
            }

    return results


def create_forest_plot(recurrence_results, hpv_results, output_path):
    """백신 종류별 Forest Plot 생성"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Recurrence
    ax1 = axes[0]

    vaccines = [r['vaccine'] for r in recurrence_results]
    hrs = [r['HR'] for r in recurrence_results]
    ci_lowers = [r['CI_lower'] for r in recurrence_results]
    ci_uppers = [r['CI_upper'] for r in recurrence_results]
    p_values = [r['p_value'] for r in recurrence_results]

    y_pos = range(len(vaccines) - 1, -1, -1)
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for i, (y, hr, ci_l, ci_u, p, c) in enumerate(zip(y_pos, hrs, ci_lowers, ci_uppers, p_values, colors)):
        ax1.plot(hr, y, 'o', color=c, markersize=10)
        ax1.hlines(y, ci_l, ci_u, colors=c, linewidth=2)
        ax1.plot([ci_l, ci_l], [y-0.1, y+0.1], color=c, linewidth=2)
        ax1.plot([ci_u, ci_u], [y-0.1, y+0.1], color=c, linewidth=2)

        # Add text
        sig = '*' if p < 0.05 else ''
        ax1.text(3.8, y, f'HR={hr:.2f} ({ci_l:.2f}-{ci_u:.2f})\np={p:.3f}{sig}',
                 fontsize=9, va='center')

    ax1.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(vaccines)
    ax1.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
    ax1.set_title('A. Cervical Lesion Recurrence', fontsize=12, fontweight='bold')
    ax1.set_xlim([0, 4.5])
    ax1.grid(True, axis='x', alpha=0.3)

    # Panel B: HPV
    ax2 = axes[1]

    vaccines = [r['vaccine'] for r in hpv_results]
    hrs = [r['HR'] for r in hpv_results]
    ci_lowers = [r['CI_lower'] for r in hpv_results]
    ci_uppers = [r['CI_upper'] for r in hpv_results]
    p_values = [r['p_value'] for r in hpv_results]

    for i, (y, hr, ci_l, ci_u, p, c) in enumerate(zip(y_pos, hrs, ci_lowers, ci_uppers, p_values, colors)):
        ax2.plot(hr, y, 'o', color=c, markersize=10)
        ax2.hlines(y, ci_l, ci_u, colors=c, linewidth=2)
        ax2.plot([ci_l, ci_l], [y-0.1, y+0.1], color=c, linewidth=2)
        ax2.plot([ci_u, ci_u], [y-0.1, y+0.1], color=c, linewidth=2)

        sig = '*' if p < 0.05 else ''
        ax2.text(2.0, y, f'HR={hr:.2f} ({ci_l:.2f}-{ci_u:.2f})\np={p:.3f}{sig}',
                 fontsize=9, va='center')

    ax2.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(vaccines)
    ax2.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
    ax2.set_title('B. High-Risk HPV Reinfection', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 2.3])
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Forest plot saved: {output_path}")


def main():
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 80)
    print("백신 종류별 Subgroup Analysis")
    print("=" * 80)

    # 데이터 로드
    df = load_data(data_dir)

    # 수술-접종 간격 분석
    print("\n[수술-접종 간격 분석]")
    interval_results = analyze_vaccination_interval(df)

    overall = interval_results['overall']
    print(f"\n전체 접종군 (n={overall['n']})")
    print(f"  평균: {overall['mean_days']:.1f}일 ({overall['mean_days']/30:.1f}개월)")
    print(f"  중앙값: {overall['median_days']:.1f}일 ({overall['median_days']/30:.1f}개월)")
    print(f"  범위: {overall['min_days']:.0f} - {overall['max_days']:.0f}일")

    print("\n백신 종류별:")
    for vtype in ['Gardasil9', 'Cervarix', 'Gardasil']:
        if vtype in interval_results:
            v = interval_results[vtype]
            print(f"  {vtype}: 평균 {v['mean_days']:.1f}일 ({v['mean_days']/30:.1f}개월), n={v['n']}")

    # 백신 종류별 Cox 분석
    vaccine_types = ['Gardasil9', 'Cervarix', 'Gardasil']

    print("\n" + "=" * 80)
    print("[병변 재발]")
    print("-" * 80)

    recurrence_results = []
    for vtype in vaccine_types:
        result = run_cox_by_vaccine(df, vtype, 'days_to_recurrence', 'has_recurrence')
        if result:
            recurrence_results.append(result)
            sig = '*' if result['p_value'] < 0.05 else ''
            print(f"{vtype}: HR={result['HR']:.3f} ({result['CI_lower']:.3f}-{result['CI_upper']:.3f}), "
                  f"p={result['p_value']:.4f}{sig}, n={result['n_vacc']}/{result['n_unvacc']}")

    print("\n[HPV 재감염]")
    print("-" * 80)

    hpv_results = []
    for vtype in vaccine_types:
        result = run_cox_by_vaccine(df, vtype, 'days_to_hpv', 'has_hpv_infection')
        if result:
            hpv_results.append(result)
            sig = '*' if result['p_value'] < 0.05 else ''
            print(f"{vtype}: HR={result['HR']:.3f} ({result['CI_lower']:.3f}-{result['CI_upper']:.3f}), "
                  f"p={result['p_value']:.4f}{sig}, n={result['n_vacc']}/{result['n_unvacc']}")

    # 결과 저장
    rec_df = pd.DataFrame(recurrence_results)
    rec_df['outcome'] = 'recurrence'
    hpv_df = pd.DataFrame(hpv_results)
    hpv_df['outcome'] = 'hpv'

    all_results = pd.concat([rec_df, hpv_df], ignore_index=True)
    all_results.to_csv(data_dir / 'vaccine_type_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: vaccine_type_analysis.csv")

    # Forest plot 생성
    create_forest_plot(recurrence_results, hpv_results, data_dir / 'figure5_vaccine_forest.png')

    return recurrence_results, hpv_results, interval_results


if __name__ == "__main__":
    rec, hpv, interval = main()
