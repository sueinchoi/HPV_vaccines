"""
민감도 분석: Recurrence & HPV Reinfection 모두 분석

다양한 분석 전략을 통해 백신의 보호 효과를 탐색:
1. Landmark 분석 (면역 형성 시간 고려)
2. 추적 기간 제한 분석
3. 하위그룹 분석 (연령, 수술방법)
4. 조기 이벤트 제외 분석
5. 복합 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir: Path) -> pd.DataFrame:
    """데이터 로드"""
    outcomes = pd.read_csv(data_dir / 'final_matched_outcomes.csv', encoding='utf-8-sig')
    outcomes['index_date'] = pd.to_datetime(outcomes['index_date'])
    return outcomes


def run_cox_simple(df: pd.DataFrame, time_var: str, event_var: str,
                   covariates: List[str] = None) -> Dict:
    """간단한 Cox 분석 수행"""
    try:
        from statsmodels.duration.hazard_regression import PHReg
    except ImportError:
        return {'HR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': np.nan}

    analysis_df = df.copy()
    analysis_df['vaccinated'] = analysis_df['접종여부'].astype(int)
    analysis_df['event'] = analysis_df[event_var].astype(int)
    analysis_df['duration'] = analysis_df[time_var].astype(float)

    # 유효한 데이터만
    analysis_df = analysis_df[analysis_df['duration'] > 0]
    analysis_df = analysis_df.dropna(subset=['duration', 'event', 'vaccinated'])

    if len(analysis_df) < 20 or analysis_df['event'].sum() < 5:
        return {'HR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': np.nan,
                'n': len(analysis_df), 'events': int(analysis_df['event'].sum())}

    # 공변량 준비
    exog_cols = ['vaccinated']
    if covariates:
        for cov in covariates:
            if cov in analysis_df.columns:
                if cov == '수술방법':
                    analysis_df['surgery_hysterectomy'] = (analysis_df[cov] == '자궁절제술').astype(int)
                    exog_cols.append('surgery_hysterectomy')
                else:
                    analysis_df = analysis_df.dropna(subset=[cov])
                    exog_cols.append(cov)

    if len(analysis_df) < 20:
        return {'HR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': np.nan,
                'n': len(analysis_df), 'events': int(analysis_df['event'].sum())}

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

        vax = analysis_df[analysis_df['vaccinated'] == 1]
        unvax = analysis_df[analysis_df['vaccinated'] == 0]

        return {
            'HR': hr, 'CI_lower': ci_lower, 'CI_upper': ci_upper, 'p_value': p_val,
            'n': len(analysis_df), 'events': int(analysis_df['event'].sum()),
            'n_vax': len(vax), 'events_vax': int(vax['event'].sum()),
            'n_unvax': len(unvax), 'events_unvax': int(unvax['event'].sum())
        }
    except Exception as e:
        return {'HR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': np.nan,
                'error': str(e)}


def run_sensitivity_analysis(df: pd.DataFrame, event_var: str, time_var: str,
                             outcome_name: str) -> pd.DataFrame:
    """특정 outcome에 대한 전체 민감도 분석 수행"""

    results = []

    # 시간 변수 준비
    df_analysis = df.copy()
    df_analysis['days_to_event'] = df_analysis.apply(
        lambda x: x[time_var] if pd.notna(x[time_var]) else x['follow_up_days'],
        axis=1
    )

    # =========================================================================
    # 1. 기본 분석
    # =========================================================================
    base_result = run_cox_simple(df_analysis, 'days_to_event', event_var, ['index_age'])
    base_result['analysis'] = 'Baseline'
    base_result['condition'] = 'Age-adjusted'
    base_result['outcome'] = outcome_name
    results.append(base_result)

    # =========================================================================
    # 2. Landmark 분석
    # =========================================================================
    landmarks = [90, 180, 365, 730]
    for lm in landmarks:
        # Landmark 시점에 이벤트가 없거나 이후에 발생한 환자만
        lm_df = df_analysis[
            (df_analysis[event_var] == 0) |
            (df_analysis[time_var] > lm)
        ].copy()

        lm_df['time_from_landmark'] = lm_df['follow_up_days'] - lm
        lm_df = lm_df[lm_df['time_from_landmark'] > 0]

        lm_df['days_to_event'] = lm_df.apply(
            lambda x: x[time_var] - lm if pd.notna(x[time_var]) and x[event_var] == 1 else x['time_from_landmark'],
            axis=1
        )

        lm_result = run_cox_simple(lm_df, 'days_to_event', event_var, ['index_age'])
        lm_result['analysis'] = 'Landmark'
        lm_result['condition'] = f'{lm}일 ({lm//30}개월)'
        lm_result['outcome'] = outcome_name
        results.append(lm_result)

    # =========================================================================
    # 3. 추적 기간 제한 분석
    # =========================================================================
    max_years_list = [1, 2, 3, 5]
    for my in max_years_list:
        max_days = my * 365
        rf_df = df_analysis.copy()

        rf_df['restricted_time'] = rf_df['follow_up_days'].clip(upper=max_days)
        rf_df['restricted_event'] = rf_df.apply(
            lambda x: 1 if x[event_var] == 1 and pd.notna(x[time_var]) and x[time_var] <= max_days else 0,
            axis=1
        )
        rf_df['days_to_event'] = rf_df.apply(
            lambda x: min(x[time_var], max_days) if pd.notna(x[time_var]) and x[event_var] == 1 else x['restricted_time'],
            axis=1
        )

        rf_result = run_cox_simple(rf_df, 'days_to_event', 'restricted_event', ['index_age'])
        rf_result['analysis'] = 'Restricted FU'
        rf_result['condition'] = f'{my}년'
        rf_result['outcome'] = outcome_name
        results.append(rf_result)

    # =========================================================================
    # 4. 하위그룹 분석 - 연령
    # =========================================================================
    age_groups = [('<35세', df_analysis['index_age'] < 35),
                  ('35-44세', (df_analysis['index_age'] >= 35) & (df_analysis['index_age'] < 45)),
                  ('≥45세', df_analysis['index_age'] >= 45)]

    for label, mask in age_groups:
        sg_df = df_analysis[mask].copy()
        sg_result = run_cox_simple(sg_df, 'days_to_event', event_var, [])
        sg_result['analysis'] = 'Subgroup-Age'
        sg_result['condition'] = label
        sg_result['outcome'] = outcome_name
        results.append(sg_result)

    # =========================================================================
    # 5. 조기 이벤트 제외
    # =========================================================================
    min_days_list = [90, 180, 270, 365]
    for md in min_days_list:
        ee_df = df_analysis[
            (df_analysis[event_var] == 0) |
            (df_analysis[time_var] >= md)
        ].copy()

        ee_result = run_cox_simple(ee_df, 'days_to_event', event_var, ['index_age'])
        ee_result['analysis'] = 'Exclude Early'
        ee_result['condition'] = f'{md}일 이전 제외'
        ee_result['outcome'] = outcome_name
        results.append(ee_result)

    # =========================================================================
    # 6. 복합 분석
    # =========================================================================
    combinations = [
        (180, 3, 180, 'LM 6mo + FU 3yr + Excl <6mo'),
        (180, 5, 270, 'LM 6mo + FU 5yr + Excl <9mo'),
        (365, 5, 180, 'LM 1yr + FU 5yr + Excl <6mo'),
        (90, 3, 180, 'LM 3mo + FU 3yr + Excl <6mo'),
    ]

    for lm, my, md, label in combinations:
        max_days = my * 365

        # Landmark 적용
        comb_df = df_analysis[
            (df_analysis[event_var] == 0) |
            (df_analysis[time_var] > lm)
        ].copy()

        # 조기 이벤트 제외
        comb_df = comb_df[
            (comb_df[event_var] == 0) |
            (comb_df[time_var] >= md)
        ]

        # 시간 재계산
        comb_df['time_from_landmark'] = comb_df['follow_up_days'] - lm
        comb_df = comb_df[comb_df['time_from_landmark'] > 0]

        # 추적 기간 제한
        comb_df['restricted_time'] = comb_df['time_from_landmark'].clip(upper=max_days - lm)
        comb_df['restricted_event'] = comb_df.apply(
            lambda x: 1 if x[event_var] == 1 and pd.notna(x[time_var]) and x[time_var] <= max_days else 0,
            axis=1
        )
        comb_df['days_to_event'] = comb_df.apply(
            lambda x: x[time_var] - lm if x['restricted_event'] == 1 else x['restricted_time'],
            axis=1
        )

        comb_result = run_cox_simple(comb_df, 'days_to_event', 'restricted_event', ['index_age'])
        comb_result['analysis'] = 'Combined'
        comb_result['condition'] = label
        comb_result['outcome'] = outcome_name
        results.append(comb_result)

    return pd.DataFrame(results)


def print_results(results_df: pd.DataFrame, outcome_name: str):
    """결과 출력"""
    outcome_results = results_df[results_df['outcome'] == outcome_name].copy()

    print(f"\n{'='*80}")
    print(f"[{outcome_name}] 민감도 분석 결과")
    print(f"{'='*80}")

    # 분석 유형별 출력
    for analysis_type in outcome_results['analysis'].unique():
        type_results = outcome_results[outcome_results['analysis'] == analysis_type]
        print(f"\n  [{analysis_type}]")

        for _, row in type_results.iterrows():
            hr = row.get('HR', np.nan)
            ci_l = row.get('CI_lower', np.nan)
            ci_u = row.get('CI_upper', np.nan)
            p = row.get('p_value', np.nan)
            n = row.get('n', 0)
            events = row.get('events', 0)

            if pd.isna(hr):
                print(f"    {row['condition']}: 분석 불가 (n={n}, events={events})")
            else:
                sig = "*" if p < 0.05 else ""
                direction = "↓" if hr < 1 else "↑"
                print(f"    {row['condition']}: HR={hr:.3f} ({ci_l:.3f}-{ci_u:.3f}), p={p:.4f} {sig} {direction}")

    # HR < 1인 결과 요약
    protective = outcome_results[outcome_results['HR'] < 1].copy()
    print(f"\n  [보호 효과 방향 (HR<1): {len(protective)}개]")
    if len(protective) > 0:
        protective = protective.sort_values('HR')
        for _, row in protective.head(5).iterrows():
            hr = row['HR']
            p = row.get('p_value', np.nan)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"    - {row['analysis']}/{row['condition']}: HR={hr:.3f}, p={p:.4f} {sig}")


def main():
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 80)
    print("민감도 분석: Recurrence & HPV Reinfection")
    print("=" * 80)

    # 데이터 로드
    df = load_data(data_dir)
    print(f"\n전체 코호트: {len(df)}명")
    print(f"  - 접종군: {(df['접종여부']==True).sum()}명")
    print(f"  - 비접종군: {(df['접종여부']==False).sum()}명")

    print(f"\n[Recurrence]")
    print(f"  - 전체: {df['has_recurrence'].sum()}건 ({df['has_recurrence'].mean()*100:.1f}%)")
    vax_r = df[df['접종여부']==True]['has_recurrence']
    unvax_r = df[df['접종여부']==False]['has_recurrence']
    print(f"  - 접종군: {vax_r.sum()}건 ({vax_r.mean()*100:.1f}%)")
    print(f"  - 비접종군: {unvax_r.sum()}건 ({unvax_r.mean()*100:.1f}%)")

    print(f"\n[HPV Reinfection]")
    print(f"  - 전체: {df['has_hpv_infection'].sum()}건 ({df['has_hpv_infection'].mean()*100:.1f}%)")
    vax_h = df[df['접종여부']==True]['has_hpv_infection']
    unvax_h = df[df['접종여부']==False]['has_hpv_infection']
    print(f"  - 접종군: {vax_h.sum()}건 ({vax_h.mean()*100:.1f}%)")
    print(f"  - 비접종군: {unvax_h.sum()}건 ({unvax_h.mean()*100:.1f}%)")

    all_results = []

    # =========================================================================
    # 1. Recurrence 분석
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. RECURRENCE (Biopsy 기반 재발) 분석 시작...")
    print("=" * 80)

    recur_results = run_sensitivity_analysis(
        df, 'has_recurrence', 'days_to_recurrence', 'Recurrence'
    )
    all_results.append(recur_results)
    print_results(recur_results, 'Recurrence')

    # =========================================================================
    # 2. HPV Reinfection 분석
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. HPV REINFECTION 분석 시작...")
    print("=" * 80)

    hpv_results = run_sensitivity_analysis(
        df, 'has_hpv_infection', 'days_to_hpv', 'HPV_Reinfection'
    )
    all_results.append(hpv_results)
    print_results(hpv_results, 'HPV_Reinfection')

    # =========================================================================
    # 결과 통합 및 저장
    # =========================================================================
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(data_dir / 'sensitivity_analysis_both_outcomes.csv',
                           index=False, encoding='utf-8-sig')

    # =========================================================================
    # 최종 요약
    # =========================================================================
    print("\n" + "=" * 80)
    print("최종 요약: 통계적으로 유의한 보호 효과 (HR<1, p<0.05)")
    print("=" * 80)

    sig_protective = combined_results[(combined_results['HR'] < 1) & (combined_results['p_value'] < 0.05)]

    if len(sig_protective) > 0:
        print(f"\n*** 총 {len(sig_protective)}개 분석에서 유의한 보호 효과 발견 ***\n")
        for _, row in sig_protective.iterrows():
            print(f"  [{row['outcome']}] {row['analysis']} - {row['condition']}")
            print(f"    HR = {row['HR']:.3f} (95% CI: {row['CI_lower']:.3f}-{row['CI_upper']:.3f})")
            print(f"    p = {row['p_value']:.4f}")
            print(f"    n = {row.get('n', 'N/A')}, events = {row.get('events', 'N/A')}\n")
    else:
        print("\n통계적으로 유의한 보호 효과 (HR<1, p<0.05)를 보이는 분석이 없습니다.")

    # 경계 유의 (0.05 < p < 0.10)
    borderline = combined_results[
        (combined_results['HR'] < 1) &
        (combined_results['p_value'] >= 0.05) &
        (combined_results['p_value'] < 0.10)
    ]

    if len(borderline) > 0:
        print(f"\n[경계 유의 (0.05 < p < 0.10): {len(borderline)}개]")
        for _, row in borderline.iterrows():
            print(f"  [{row['outcome']}] {row['analysis']} - {row['condition']}: HR={row['HR']:.3f}, p={row['p_value']:.4f}")

    # 전체 분포
    print(f"\n[전체 분석 결과 분포]")
    for outcome in ['Recurrence', 'HPV_Reinfection']:
        oc_results = combined_results[combined_results['outcome'] == outcome]
        hr_lt1 = (oc_results['HR'] < 1).sum()
        hr_ge1 = (oc_results['HR'] >= 1).sum()
        print(f"  {outcome}:")
        print(f"    - HR < 1 (보호): {hr_lt1}개, HR >= 1 (위험↑): {hr_ge1}개")
        if not oc_results['HR'].isna().all():
            print(f"    - HR 범위: {oc_results['HR'].min():.3f} ~ {oc_results['HR'].max():.3f}")

    print(f"\n결과 저장: {data_dir / 'sensitivity_analysis_both_outcomes.csv'}")
    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)

    return combined_results


if __name__ == "__main__":
    results = main()
