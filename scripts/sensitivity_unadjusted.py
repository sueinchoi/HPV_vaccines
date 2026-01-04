"""
민감도 분석: Age 보정 vs 비보정 비교

동일한 분석을 age 보정 유무에 따라 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir: Path) -> pd.DataFrame:
    """데이터 로드"""
    outcomes = pd.read_csv(data_dir / 'final_matched_outcomes.csv', encoding='utf-8-sig')
    outcomes['index_date'] = pd.to_datetime(outcomes['index_date'])
    return outcomes


def run_cox(df: pd.DataFrame, time_var: str, event_var: str,
            covariates: List[str] = None) -> Dict:
    """Cox 분석 수행"""
    try:
        from statsmodels.duration.hazard_regression import PHReg
    except ImportError:
        return {'HR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': np.nan}

    analysis_df = df.copy()
    analysis_df['vaccinated'] = analysis_df['접종여부'].astype(int)
    analysis_df['event'] = analysis_df[event_var].astype(int)
    analysis_df['duration'] = analysis_df[time_var].astype(float)

    analysis_df = analysis_df[analysis_df['duration'] > 0]
    analysis_df = analysis_df.dropna(subset=['duration', 'event', 'vaccinated'])

    if len(analysis_df) < 20 or analysis_df['event'].sum() < 5:
        return {'HR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': np.nan,
                'n': len(analysis_df), 'events': int(analysis_df['event'].sum())}

    exog_cols = ['vaccinated']
    if covariates:
        for cov in covariates:
            if cov in analysis_df.columns:
                analysis_df = analysis_df.dropna(subset=[cov])
                exog_cols.append(cov)

    if len(analysis_df) < 20:
        return {'HR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': np.nan}

    try:
        exog = analysis_df[exog_cols]
        model = PHReg(analysis_df['duration'], exog, status=analysis_df['event'])
        result = model.fit(disp=False)

        coef = result.params[0]
        se = result.bse[0]
        hr = np.exp(coef)
        ci_lower = np.exp(coef - 1.96 * se)
        ci_upper = np.exp(coef + 1.96 * se)
        p_val = result.pvalues[0]

        return {
            'HR': hr, 'CI_lower': ci_lower, 'CI_upper': ci_upper, 'p_value': p_val,
            'n': len(analysis_df), 'events': int(analysis_df['event'].sum())
        }
    except Exception as e:
        return {'HR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': np.nan}


def run_analysis_pair(df: pd.DataFrame, event_var: str, time_var: str,
                      analysis_name: str, condition: str) -> List[Dict]:
    """보정/비보정 분석 쌍으로 수행"""
    results = []

    # 시간 변수 준비
    df_analysis = df.copy()
    df_analysis['days_to_event'] = df_analysis.apply(
        lambda x: x[time_var] if pd.notna(x[time_var]) else x['follow_up_days'],
        axis=1
    )

    # Unadjusted
    unadj = run_cox(df_analysis, 'days_to_event', event_var, covariates=[])
    unadj['adjustment'] = 'Unadjusted'
    unadj['analysis'] = analysis_name
    unadj['condition'] = condition
    results.append(unadj)

    # Age-adjusted
    adj = run_cox(df_analysis, 'days_to_event', event_var, covariates=['index_age'])
    adj['adjustment'] = 'Age-adjusted'
    adj['analysis'] = analysis_name
    adj['condition'] = condition
    results.append(adj)

    return results


def main():
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 90)
    print("민감도 분석: Age 보정 vs 비보정 비교")
    print("=" * 90)

    df = load_data(data_dir)
    print(f"\n전체 코호트: {len(df)}명")

    all_results = []

    for outcome_name, event_var, time_var in [
        ('Recurrence', 'has_recurrence', 'days_to_recurrence'),
        ('HPV_Reinfection', 'has_hpv_infection', 'days_to_hpv')
    ]:
        print(f"\n{'='*90}")
        print(f"[{outcome_name}]")
        print("=" * 90)

        # 기본 데이터
        df_base = df.copy()
        df_base['days_to_event'] = df_base.apply(
            lambda x: x[time_var] if pd.notna(x[time_var]) else x['follow_up_days'],
            axis=1
        )

        # 1. Baseline
        results = run_analysis_pair(df, event_var, time_var, 'Baseline', 'All')
        for r in results:
            r['outcome'] = outcome_name
        all_results.extend(results)

        # 2. Landmark
        for lm in [180, 365]:
            lm_df = df[
                (df[event_var] == 0) | (df[time_var] > lm)
            ].copy()
            lm_df['follow_up_days'] = lm_df['follow_up_days'] - lm
            lm_df = lm_df[lm_df['follow_up_days'] > 0]
            lm_df[time_var] = lm_df[time_var].apply(lambda x: x - lm if pd.notna(x) else None)

            results = run_analysis_pair(lm_df, event_var, time_var, 'Landmark', f'{lm}일')
            for r in results:
                r['outcome'] = outcome_name
            all_results.extend(results)

        # 3. Restricted FU
        for my in [2, 3, 5]:
            max_days = my * 365
            rf_df = df.copy()
            rf_df['restricted_event'] = rf_df.apply(
                lambda x: 1 if x[event_var] == 1 and pd.notna(x[time_var]) and x[time_var] <= max_days else 0,
                axis=1
            )
            rf_df['restricted_time'] = rf_df.apply(
                lambda x: min(x[time_var], max_days) if pd.notna(x[time_var]) and x[event_var] == 1 else min(x['follow_up_days'], max_days),
                axis=1
            )

            results = run_analysis_pair(rf_df, 'restricted_event', 'restricted_time', 'Restricted FU', f'{my}년')
            for r in results:
                r['outcome'] = outcome_name
            all_results.extend(results)

        # 4. Exclude Early
        for md in [180, 365]:
            ee_df = df[
                (df[event_var] == 0) | (df[time_var] >= md)
            ].copy()

            results = run_analysis_pair(ee_df, event_var, time_var, 'Exclude Early', f'{md}일 이전 제외')
            for r in results:
                r['outcome'] = outcome_name
            all_results.extend(results)

        # 5. Age subgroups (unadjusted only makes sense here)
        for label, mask in [('<35세', df['index_age'] < 35),
                           ('35-44세', (df['index_age'] >= 35) & (df['index_age'] < 45)),
                           ('≥45세', df['index_age'] >= 45)]:
            sg_df = df[mask].copy()
            sg_df['days_to_event'] = sg_df.apply(
                lambda x: x[time_var] if pd.notna(x[time_var]) else x['follow_up_days'],
                axis=1
            )

            unadj = run_cox(sg_df, 'days_to_event', event_var, covariates=[])
            unadj['adjustment'] = 'Unadjusted'
            unadj['analysis'] = 'Subgroup-Age'
            unadj['condition'] = label
            unadj['outcome'] = outcome_name
            all_results.append(unadj)

    # 결과 정리
    results_df = pd.DataFrame(all_results)

    # 출력
    for outcome in ['Recurrence', 'HPV_Reinfection']:
        print(f"\n{'='*90}")
        print(f"[{outcome}] Age 보정 vs 비보정 비교")
        print("=" * 90)
        print(f"\n{'Analysis':<15} {'Condition':<18} {'Adjustment':<14} {'HR':>7} {'95% CI':>18} {'p-value':>10}")
        print("-" * 90)

        oc_df = results_df[results_df['outcome'] == outcome]

        for analysis in oc_df['analysis'].unique():
            an_df = oc_df[oc_df['analysis'] == analysis]
            for condition in an_df['condition'].unique():
                cond_df = an_df[an_df['condition'] == condition]
                for _, row in cond_df.iterrows():
                    hr = row.get('HR', np.nan)
                    ci_l = row.get('CI_lower', np.nan)
                    ci_u = row.get('CI_upper', np.nan)
                    p = row.get('p_value', np.nan)
                    adj = row['adjustment']

                    if pd.isna(hr):
                        continue

                    sig = "*" if p < 0.05 else ""
                    ci_str = f"({ci_l:.3f}-{ci_u:.3f})"
                    print(f"{analysis:<15} {condition:<18} {adj:<14} {hr:>7.3f} {ci_str:>18} {p:>9.4f} {sig}")

    # 요약: 유의한 결과
    print(f"\n{'='*90}")
    print("유의한 보호 효과 (HR<1, p<0.05)")
    print("=" * 90)

    sig_prot = results_df[(results_df['HR'] < 1) & (results_df['p_value'] < 0.05)]
    if len(sig_prot) > 0:
        for _, row in sig_prot.iterrows():
            print(f"  [{row['outcome']}] {row['analysis']}/{row['condition']} ({row['adjustment']})")
            print(f"    HR={row['HR']:.3f}, p={row['p_value']:.4f}")
    else:
        print("  없음")

    # 경계 유의
    print(f"\n경계 유의 (HR<1, 0.05<p<0.15)")
    print("-" * 90)
    borderline = results_df[(results_df['HR'] < 1) &
                            (results_df['p_value'] >= 0.05) &
                            (results_df['p_value'] < 0.15)]
    if len(borderline) > 0:
        borderline = borderline.sort_values('p_value')
        for _, row in borderline.iterrows():
            print(f"  [{row['outcome']}] {row['analysis']}/{row['condition']} ({row['adjustment']})")
            print(f"    HR={row['HR']:.3f}, p={row['p_value']:.4f}")

    # 비보정 vs 보정 비교
    print(f"\n{'='*90}")
    print("Unadjusted vs Age-adjusted 비교 (동일 조건)")
    print("=" * 90)

    for outcome in ['Recurrence', 'HPV_Reinfection']:
        print(f"\n[{outcome}]")
        oc_df = results_df[results_df['outcome'] == outcome]

        for analysis in ['Baseline', 'Landmark', 'Restricted FU', 'Exclude Early']:
            an_df = oc_df[oc_df['analysis'] == analysis]
            for condition in an_df['condition'].unique():
                cond_df = an_df[an_df['condition'] == condition]
                unadj_row = cond_df[cond_df['adjustment'] == 'Unadjusted']
                adj_row = cond_df[cond_df['adjustment'] == 'Age-adjusted']

                if len(unadj_row) > 0 and len(adj_row) > 0:
                    unadj_hr = unadj_row.iloc[0]['HR']
                    adj_hr = adj_row.iloc[0]['HR']
                    unadj_p = unadj_row.iloc[0]['p_value']
                    adj_p = adj_row.iloc[0]['p_value']

                    if pd.notna(unadj_hr) and pd.notna(adj_hr):
                        diff = unadj_hr - adj_hr
                        better = "Unadj" if unadj_p < adj_p else "Adj"
                        print(f"  {analysis}/{condition}: Unadj HR={unadj_hr:.3f}(p={unadj_p:.3f}) vs Adj HR={adj_hr:.3f}(p={adj_p:.3f}) → {better} better")

    results_df.to_csv(data_dir / 'sensitivity_adjusted_vs_unadjusted.csv', index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: sensitivity_adjusted_vs_unadjusted.csv")

    return results_df


if __name__ == "__main__":
    results = main()
