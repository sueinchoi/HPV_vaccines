"""
민감도 분석: 백신 효과 탐색

다양한 분석 전략을 통해 백신의 보호 효과를 탐색:
1. Landmark 분석 (면역 형성 시간 고려)
2. 추적 기간 제한 분석
3. 하위그룹 분석 (연령, 수술방법)
4. 조기 재발 제외 분석
5. 보정 변수 조합 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir: Path) -> pd.DataFrame:
    """데이터 로드"""
    outcomes = pd.read_csv(data_dir / 'final_matched_outcomes.csv', encoding='utf-8-sig')
    outcomes['index_date'] = pd.to_datetime(outcomes['index_date'])
    outcomes['recurrence_date'] = pd.to_datetime(outcomes['recurrence_date'])
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
                'n': len(analysis_df), 'events': analysis_df['event'].sum()}

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
                'n': len(analysis_df), 'events': analysis_df['event'].sum()}

    try:
        exog = analysis_df[exog_cols]
        model = PHReg(analysis_df['duration'], exog, status=analysis_df['event'])
        result = model.fit(disp=False)

        # Vaccination의 결과 (첫 번째 변수)
        coef = result.params[0]
        se = result.bse[0]
        hr = np.exp(coef)
        ci_lower = np.exp(coef - 1.96 * se)
        ci_upper = np.exp(coef + 1.96 * se)
        p_val = result.pvalues[0]

        # 그룹별 정보
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


def log_rank_test(df: pd.DataFrame, time_var: str, event_var: str) -> float:
    """Log-rank test p-value"""
    vax = df[df['접종여부'] == True]
    unvax = df[df['접종여부'] == False]

    if len(vax) < 5 or len(unvax) < 5:
        return np.nan

    # 간단한 log-rank 구현
    all_times = sorted(set(list(vax[time_var]) + list(unvax[time_var])))

    observed1, expected1 = 0, 0

    for t in all_times:
        at_risk1 = np.sum(vax[time_var] >= t)
        at_risk2 = np.sum(unvax[time_var] >= t)
        at_risk_total = at_risk1 + at_risk2

        events1_t = np.sum((vax[time_var] == t) & (vax[event_var] == 1))
        events2_t = np.sum((unvax[time_var] == t) & (unvax[event_var] == 1))
        events_total = events1_t + events2_t

        if at_risk_total > 0:
            observed1 += events1_t
            expected1 += at_risk1 * events_total / at_risk_total

    if expected1 > 0:
        chi2 = (observed1 - expected1) ** 2 / expected1
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        p_value = 1.0

    return p_value


def landmark_analysis(df: pd.DataFrame, landmark_days: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Landmark 분석: 특정 시점 이후부터 분석 시작

    - landmark_days 이전에 이벤트가 발생한 환자 제외
    - 시간을 landmark부터 다시 계산
    """
    # Landmark 시점에 아직 이벤트가 없는 환자만 포함
    # 재발이 없거나, 재발이 landmark 이후인 경우
    landmark_df = df[
        (df['has_recurrence'] == 0) |
        (df['days_to_recurrence'] > landmark_days)
    ].copy()

    # 시간 재계산 (landmark부터)
    landmark_df['time_from_landmark'] = landmark_df['follow_up_days'] - landmark_days

    # 재발 시간도 재계산
    landmark_df['days_to_event'] = landmark_df.apply(
        lambda x: x['days_to_recurrence'] - landmark_days if pd.notna(x['days_to_recurrence']) else x['time_from_landmark'],
        axis=1
    )

    # 음수 시간 제외 (landmark 이전에 추적 종료된 경우)
    landmark_df = landmark_df[landmark_df['time_from_landmark'] > 0]

    return landmark_df, run_cox_simple(landmark_df, 'days_to_event', 'has_recurrence', ['index_age'])


def restricted_followup_analysis(df: pd.DataFrame, max_years: float) -> Tuple[pd.DataFrame, Dict]:
    """
    추적 기간 제한 분석: 특정 기간까지만 분석
    """
    max_days = max_years * 365

    restricted_df = df.copy()

    # 추적 기간 제한
    restricted_df['restricted_time'] = restricted_df['follow_up_days'].clip(upper=max_days)

    # 이벤트 재정의 (제한 기간 내 이벤트만)
    restricted_df['restricted_event'] = restricted_df.apply(
        lambda x: 1 if x['has_recurrence'] == 1 and pd.notna(x['days_to_recurrence']) and x['days_to_recurrence'] <= max_days else 0,
        axis=1
    )

    # 이벤트 시간
    restricted_df['days_to_event'] = restricted_df.apply(
        lambda x: min(x['days_to_recurrence'], max_days) if pd.notna(x['days_to_recurrence']) and x['has_recurrence'] == 1 else x['restricted_time'],
        axis=1
    )

    return restricted_df, run_cox_simple(restricted_df, 'days_to_event', 'restricted_event', ['index_age'])


def subgroup_analysis(df: pd.DataFrame, subgroup_var: str, subgroup_value) -> Tuple[pd.DataFrame, Dict]:
    """하위그룹 분석"""
    if subgroup_var == 'age_group':
        if subgroup_value == 'young':
            subgroup_df = df[df['index_age'] < 35].copy()
        elif subgroup_value == 'middle':
            subgroup_df = df[(df['index_age'] >= 35) & (df['index_age'] < 45)].copy()
        else:
            subgroup_df = df[df['index_age'] >= 45].copy()
    elif subgroup_var == '수술방법':
        subgroup_df = df[df['수술방법'] == subgroup_value].copy()
    elif subgroup_var == 'surgery_year':
        if subgroup_value == 'early':
            subgroup_df = df[df['수술연도'] < 2015].copy()
        else:
            subgroup_df = df[df['수술연도'] >= 2015].copy()
    else:
        subgroup_df = df.copy()

    # 시간 변수 설정
    subgroup_df['days_to_event'] = subgroup_df.apply(
        lambda x: x['days_to_recurrence'] if pd.notna(x['days_to_recurrence']) else x['follow_up_days'],
        axis=1
    )

    return subgroup_df, run_cox_simple(subgroup_df, 'days_to_event', 'has_recurrence', ['index_age'])


def exclude_early_recurrence(df: pd.DataFrame, min_days: int) -> Tuple[pd.DataFrame, Dict]:
    """
    조기 재발 제외 분석: 특정 기간 이전 재발을 제외

    이유: 조기 재발은 잔존 병변일 가능성이 높음
    """
    # 조기 재발 환자 제외 (재발 없거나, 재발이 min_days 이후)
    filtered_df = df[
        (df['has_recurrence'] == 0) |
        (df['days_to_recurrence'] >= min_days)
    ].copy()

    # 시간 변수
    filtered_df['days_to_event'] = filtered_df.apply(
        lambda x: x['days_to_recurrence'] if pd.notna(x['days_to_recurrence']) else x['follow_up_days'],
        axis=1
    )

    return filtered_df, run_cox_simple(filtered_df, 'days_to_event', 'has_recurrence', ['index_age'])


def covariate_adjustment_analysis(df: pd.DataFrame, covariates: List[str]) -> Dict:
    """다양한 공변량 조합으로 분석"""
    analysis_df = df.copy()
    analysis_df['days_to_event'] = analysis_df.apply(
        lambda x: x['days_to_recurrence'] if pd.notna(x['days_to_recurrence']) else x['follow_up_days'],
        axis=1
    )

    return run_cox_simple(analysis_df, 'days_to_event', 'has_recurrence', covariates)


def combined_analysis(df: pd.DataFrame, landmark_days: int, max_years: float,
                      min_recur_days: int) -> Dict:
    """
    복합 분석: 여러 전략 동시 적용
    """
    # 1. Landmark 적용
    combined_df = df[
        (df['has_recurrence'] == 0) |
        (df['days_to_recurrence'] > landmark_days)
    ].copy()

    # 2. 조기 재발 제외 (landmark 이후 재발 중)
    combined_df = combined_df[
        (combined_df['has_recurrence'] == 0) |
        (combined_df['days_to_recurrence'] >= min_recur_days)
    ]

    # 3. 시간 재계산
    combined_df['time_from_landmark'] = combined_df['follow_up_days'] - landmark_days
    combined_df = combined_df[combined_df['time_from_landmark'] > 0]

    # 4. 추적 기간 제한
    max_days = max_years * 365
    combined_df['restricted_time'] = combined_df['time_from_landmark'].clip(upper=max_days - landmark_days)

    # 5. 이벤트 재정의
    combined_df['restricted_event'] = combined_df.apply(
        lambda x: 1 if x['has_recurrence'] == 1 and pd.notna(x['days_to_recurrence']) and
                      x['days_to_recurrence'] <= max_days else 0,
        axis=1
    )

    # 6. 이벤트 시간
    combined_df['days_to_event'] = combined_df.apply(
        lambda x: x['days_to_recurrence'] - landmark_days if x['restricted_event'] == 1 else x['restricted_time'],
        axis=1
    )

    return run_cox_simple(combined_df, 'days_to_event', 'restricted_event', ['index_age'])


def main():
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 80)
    print("민감도 분석: 백신 보호 효과 탐색")
    print("=" * 80)

    # 데이터 로드
    df = load_data(data_dir)
    print(f"\n전체 코호트: {len(df)}명")
    print(f"  - 접종군: {(df['접종여부']==True).sum()}명")
    print(f"  - 비접종군: {(df['접종여부']==False).sum()}명")
    print(f"  - 전체 재발: {df['has_recurrence'].sum()}건 ({df['has_recurrence'].mean()*100:.1f}%)")

    results = []

    # =========================================================================
    # 1. 기본 분석 (Baseline)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[1. 기본 분석 (Baseline)]")
    print("=" * 80)

    df_base = df.copy()
    df_base['days_to_event'] = df_base.apply(
        lambda x: x['days_to_recurrence'] if pd.notna(x['days_to_recurrence']) else x['follow_up_days'],
        axis=1
    )
    base_result = run_cox_simple(df_base, 'days_to_event', 'has_recurrence', ['index_age'])
    base_result['analysis'] = 'Baseline'
    base_result['condition'] = 'None'
    results.append(base_result)

    print(f"  HR = {base_result.get('HR', np.nan):.3f} (95% CI: {base_result.get('CI_lower', np.nan):.3f}-{base_result.get('CI_upper', np.nan):.3f})")
    print(f"  p-value = {base_result.get('p_value', np.nan):.4f}")

    # =========================================================================
    # 2. Landmark 분석
    # =========================================================================
    print("\n" + "=" * 80)
    print("[2. Landmark 분석]")
    print("  - 면역 형성 기간을 고려하여 특정 시점 이후부터 분석")
    print("=" * 80)

    landmarks = [90, 180, 365, 730]  # 3개월, 6개월, 1년, 2년
    for lm in landmarks:
        lm_df, lm_result = landmark_analysis(df, lm)
        lm_result['analysis'] = 'Landmark'
        lm_result['condition'] = f'{lm}일 ({lm/30:.0f}개월)'
        results.append(lm_result)

        hr = lm_result.get('HR', np.nan)
        p = lm_result.get('p_value', np.nan)
        n = lm_result.get('n', 0)
        events = lm_result.get('events', 0)
        sig = "*" if p < 0.05 else ""

        print(f"\n  Landmark {lm}일 ({lm/30:.0f}개월):")
        print(f"    - n={n}, events={events}")
        print(f"    - HR = {hr:.3f}, p = {p:.4f} {sig}")

    # =========================================================================
    # 3. 추적 기간 제한 분석
    # =========================================================================
    print("\n" + "=" * 80)
    print("[3. 추적 기간 제한 분석]")
    print("  - 특정 기간까지만 분석 (장기 추적의 혼란 요인 제거)")
    print("=" * 80)

    max_years_list = [1, 2, 3, 4, 5]
    for my in max_years_list:
        rf_df, rf_result = restricted_followup_analysis(df, my)
        rf_result['analysis'] = 'Restricted FU'
        rf_result['condition'] = f'{my}년'
        results.append(rf_result)

        hr = rf_result.get('HR', np.nan)
        p = rf_result.get('p_value', np.nan)
        events = rf_result.get('events', 0)
        sig = "*" if p < 0.05 else ""

        print(f"\n  최대 {my}년 추적:")
        print(f"    - events={events}")
        print(f"    - HR = {hr:.3f}, p = {p:.4f} {sig}")

    # =========================================================================
    # 4. 하위그룹 분석
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4. 하위그룹 분석]")
    print("=" * 80)

    # 연령별
    print("\n  [4-1. 연령별]")
    for age_grp in ['young', 'middle', 'old']:
        label = {'young': '<35세', 'middle': '35-44세', 'old': '≥45세'}[age_grp]
        sg_df, sg_result = subgroup_analysis(df, 'age_group', age_grp)
        sg_result['analysis'] = 'Subgroup-Age'
        sg_result['condition'] = label
        results.append(sg_result)

        hr = sg_result.get('HR', np.nan)
        p = sg_result.get('p_value', np.nan)
        n = sg_result.get('n', 0)
        sig = "*" if p < 0.05 else ""

        print(f"    {label}: n={n}, HR={hr:.3f}, p={p:.4f} {sig}")

    # 수술방법별
    print("\n  [4-2. 수술방법별]")
    for surgery in ['원추절제술', '자궁절제술']:
        sg_df, sg_result = subgroup_analysis(df, '수술방법', surgery)
        sg_result['analysis'] = 'Subgroup-Surgery'
        sg_result['condition'] = surgery
        results.append(sg_result)

        hr = sg_result.get('HR', np.nan)
        p = sg_result.get('p_value', np.nan)
        n = sg_result.get('n', 0)
        sig = "*" if p < 0.05 else ""

        print(f"    {surgery}: n={n}, HR={hr:.3f}, p={p:.4f} {sig}")

    # 수술 시기별
    print("\n  [4-3. 수술 시기별]")
    for period in ['early', 'late']:
        label = {'early': '<2015', 'late': '≥2015'}[period]
        sg_df, sg_result = subgroup_analysis(df, 'surgery_year', period)
        sg_result['analysis'] = 'Subgroup-Year'
        sg_result['condition'] = label
        results.append(sg_result)

        hr = sg_result.get('HR', np.nan)
        p = sg_result.get('p_value', np.nan)
        n = sg_result.get('n', 0)
        sig = "*" if p < 0.05 else ""

        print(f"    {label}: n={n}, HR={hr:.3f}, p={p:.4f} {sig}")

    # =========================================================================
    # 5. 조기 재발 제외 분석
    # =========================================================================
    print("\n" + "=" * 80)
    print("[5. 조기 재발 제외 분석]")
    print("  - 잔존 병변일 가능성이 높은 조기 재발 제외")
    print("=" * 80)

    min_days_list = [120, 180, 270, 365]
    for md in min_days_list:
        ee_df, ee_result = exclude_early_recurrence(df, md)
        ee_result['analysis'] = 'Exclude Early'
        ee_result['condition'] = f'{md}일 이전 재발 제외'
        results.append(ee_result)

        hr = ee_result.get('HR', np.nan)
        p = ee_result.get('p_value', np.nan)
        n = ee_result.get('n', 0)
        events = ee_result.get('events', 0)
        sig = "*" if p < 0.05 else ""

        print(f"\n  {md}일 이전 재발 제외:")
        print(f"    - n={n}, events={events}")
        print(f"    - HR = {hr:.3f}, p = {p:.4f} {sig}")

    # =========================================================================
    # 6. 공변량 조합 분석
    # =========================================================================
    print("\n" + "=" * 80)
    print("[6. 공변량 조합 분석]")
    print("=" * 80)

    covariate_sets = [
        ([], 'Unadjusted'),
        (['index_age'], 'Age adjusted'),
        (['index_age', '수술방법'], 'Age + Surgery'),
        (['index_age', 'closest_bmi'], 'Age + BMI'),
        (['index_age', '수술방법', 'closest_bmi'], 'Full model'),
    ]

    for covs, label in covariate_sets:
        cov_result = covariate_adjustment_analysis(df, covs)
        cov_result['analysis'] = 'Covariate'
        cov_result['condition'] = label
        results.append(cov_result)

        hr = cov_result.get('HR', np.nan)
        p = cov_result.get('p_value', np.nan)
        sig = "*" if p < 0.05 else ""

        print(f"  {label}: HR={hr:.3f}, p={p:.4f} {sig}")

    # =========================================================================
    # 7. 복합 분석 (최적 조건 탐색)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[7. 복합 분석 (여러 전략 동시 적용)]")
    print("=" * 80)

    # 다양한 조합 시도
    combinations = [
        (180, 3, 180, 'Landmark 6mo + FU 3yr + Exclude <6mo'),
        (365, 5, 180, 'Landmark 1yr + FU 5yr + Exclude <6mo'),
        (180, 5, 270, 'Landmark 6mo + FU 5yr + Exclude <9mo'),
        (90, 3, 180, 'Landmark 3mo + FU 3yr + Exclude <6mo'),
        (365, 4, 365, 'Landmark 1yr + FU 4yr + Exclude <1yr'),
    ]

    for lm, my, md, label in combinations:
        comb_result = combined_analysis(df, lm, my, md)
        comb_result['analysis'] = 'Combined'
        comb_result['condition'] = label
        results.append(comb_result)

        hr = comb_result.get('HR', np.nan)
        p = comb_result.get('p_value', np.nan)
        n = comb_result.get('n', 0)
        events = comb_result.get('events', 0)
        sig = "*" if p < 0.05 else ""

        print(f"\n  {label}:")
        print(f"    - n={n}, events={events}")
        print(f"    - HR = {hr:.3f}, p = {p:.4f} {sig}")

    # =========================================================================
    # 결과 요약
    # =========================================================================
    print("\n" + "=" * 80)
    print("[결과 요약: HR < 1 인 분석 (백신 보호 효과 가능성)]")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df.to_csv(data_dir / 'sensitivity_analysis_results.csv', index=False, encoding='utf-8-sig')

    # HR < 1인 결과만 필터링
    protective = results_df[results_df['HR'] < 1].copy()
    if len(protective) > 0:
        protective = protective.sort_values('HR')
        print(f"\n총 {len(protective)}개 분석에서 HR < 1 (보호 효과 방향)")
        print("-" * 80)
        print(f"{'Analysis':<20} {'Condition':<35} {'HR':>8} {'95% CI':>18} {'p-value':>10}")
        print("-" * 80)

        for _, row in protective.iterrows():
            hr = row['HR']
            ci = f"({row.get('CI_lower', np.nan):.3f}-{row.get('CI_upper', np.nan):.3f})"
            p = row.get('p_value', np.nan)
            sig = "*" if p < 0.05 else ""
            print(f"{row['analysis']:<20} {row['condition']:<35} {hr:>8.3f} {ci:>18} {p:>9.4f} {sig}")
    else:
        print("\n모든 분석에서 HR >= 1 (보호 효과 없음)")

    # HR < 1 & p < 0.05인 결과
    print("\n" + "-" * 80)
    sig_protective = results_df[(results_df['HR'] < 1) & (results_df['p_value'] < 0.05)]
    if len(sig_protective) > 0:
        print(f"\n*** 통계적으로 유의한 보호 효과: {len(sig_protective)}개 ***")
        for _, row in sig_protective.iterrows():
            print(f"  - {row['analysis']}: {row['condition']}")
            print(f"    HR = {row['HR']:.3f}, p = {row['p_value']:.4f}")
    else:
        print("\n통계적으로 유의한 보호 효과 (HR<1, p<0.05)를 보이는 분석이 없습니다.")

    # 전체 결과 분포
    print("\n" + "-" * 80)
    print("\n[전체 분석 결과 분포]")
    print(f"  - HR < 1 (보호 효과 방향): {(results_df['HR'] < 1).sum()}개")
    print(f"  - HR >= 1 (위험 증가 방향): {(results_df['HR'] >= 1).sum()}개")
    print(f"  - HR 중앙값: {results_df['HR'].median():.3f}")
    print(f"  - HR 범위: {results_df['HR'].min():.3f} ~ {results_df['HR'].max():.3f}")

    print(f"\n결과 저장: {data_dir / 'sensitivity_analysis_results.csv'}")
    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    results = main()
