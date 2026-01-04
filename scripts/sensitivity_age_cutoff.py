"""
Age Cutoff Sensitivity Analysis

다양한 연령 범위, 랜드마크, 추적 기간 조합에서
HPV 백신의 자궁경부 병변 재발 예방 효과를 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 재현성을 위한 시드 고정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


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
                'n': len(analysis_df), 'events': int(analysis_df['event'].sum()),
                'vacc_n': int(analysis_df['vaccinated'].sum()),
                'vacc_events': int(analysis_df[analysis_df['vaccinated']==1]['event'].sum())}

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
            'n': len(analysis_df), 'events': int(analysis_df['event'].sum()),
            'vacc_n': int(analysis_df['vaccinated'].sum()),
            'vacc_events': int(analysis_df[analysis_df['vaccinated']==1]['event'].sum())
        }
    except Exception as e:
        return {'HR': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'p_value': np.nan}


def analyze_subgroup(df: pd.DataFrame, age_min: int, age_max: int,
                     landmark: int, max_fu_years: float, outcome: str) -> Dict:
    """특정 조건에서 분석 수행"""

    # 결과 변수 설정
    if outcome == 'recurrence':
        event_var = 'has_recurrence'
        time_var = 'days_to_recurrence'
    else:  # HPV
        event_var = 'has_hpv_infection'
        time_var = 'days_to_hpv'

    # 연령 필터
    subgroup = df[(df['index_age'] >= age_min) & (df['index_age'] < age_max)].copy()

    if len(subgroup) < 30:
        return None

    # Landmark 적용
    if landmark > 0:
        subgroup = subgroup[
            (subgroup[event_var] == 0) |
            (subgroup[time_var] > landmark)
        ].copy()
        subgroup['follow_up_days'] = subgroup['follow_up_days'] - landmark
        subgroup = subgroup[subgroup['follow_up_days'] > 0]
        subgroup[time_var] = subgroup[time_var].apply(
            lambda x: x - landmark if pd.notna(x) else None
        )

    if len(subgroup) < 30:
        return None

    # 최대 추적 기간 제한
    if max_fu_years is not None:
        max_days = int(max_fu_years * 365)
        subgroup['restricted_event'] = subgroup.apply(
            lambda x: 1 if x[event_var] == 1 and pd.notna(x[time_var]) and x[time_var] <= max_days else 0,
            axis=1
        )
        subgroup['restricted_time'] = subgroup.apply(
            lambda x: min(x[time_var], max_days) if pd.notna(x[time_var]) and x[event_var] == 1
                      else min(x['follow_up_days'], max_days),
            axis=1
        )
        event_var = 'restricted_event'
        time_var = 'restricted_time'
    else:
        subgroup['days_to_event'] = subgroup.apply(
            lambda x: x[time_var] if pd.notna(x[time_var]) else x['follow_up_days'],
            axis=1
        )
        time_var = 'days_to_event'

    # Cox 분석
    result = run_cox(subgroup, time_var, event_var, covariates=['index_age'])

    if result.get('HR') is None or pd.isna(result.get('HR')):
        return None

    result['age_min'] = age_min
    result['age_max'] = age_max
    result['landmark'] = landmark
    result['max_fu_years'] = max_fu_years if max_fu_years else 'All'
    result['outcome'] = outcome

    return result


def run_comprehensive_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """포괄적인 age cutoff 분석"""

    results = []

    # 연령 범위 설정
    age_mins = list(range(20, 36, 2))  # 20, 22, 24, ..., 34
    age_maxs = list(range(44, 66, 2))  # 44, 46, 48, ..., 64

    # 랜드마크 설정
    landmarks = [0, 90, 180, 365, 730]

    # 최대 추적 기간 설정
    max_fu_years_list = [None, 1, 1.5, 2, 2.5, 3, 4, 5]

    # 결과 변수
    outcomes = ['recurrence', 'hpv']

    total = len(age_mins) * len(age_maxs) * len(landmarks) * len(max_fu_years_list) * len(outcomes)
    count = 0

    print(f"\n총 {total}개 조합 분석 중...")

    for outcome in outcomes:
        for age_min in age_mins:
            for age_max in age_maxs:
                if age_max <= age_min + 10:  # 최소 10세 범위
                    continue

                for landmark in landmarks:
                    for max_fu in max_fu_years_list:
                        count += 1
                        if count % 500 == 0:
                            print(f"  진행: {count}/{total}")

                        result = analyze_subgroup(df, age_min, age_max,
                                                  landmark, max_fu, outcome)
                        if result:
                            results.append(result)

    print(f"  완료: {len(results)}개 유효 결과")

    return pd.DataFrame(results)


def find_significant_results(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """유의한 결과 찾기"""

    # 보호 효과 (HR < 1, p < 0.05)
    sig_protective = results_df[
        (results_df['HR'] < 1) &
        (results_df['p_value'] < 0.05)
    ].sort_values('p_value')

    # 경계 유의 (HR < 1, 0.05 <= p < 0.10)
    borderline = results_df[
        (results_df['HR'] < 1) &
        (results_df['p_value'] >= 0.05) &
        (results_df['p_value'] < 0.10)
    ].sort_values('p_value')

    return sig_protective, borderline


def print_summary(results_df: pd.DataFrame, sig_df: pd.DataFrame, borderline_df: pd.DataFrame):
    """결과 요약 출력"""

    print("\n" + "=" * 100)
    print("AGE CUTOFF SENSITIVITY ANALYSIS 결과 요약")
    print("=" * 100)

    for outcome in ['recurrence', 'hpv']:
        outcome_results = results_df[results_df['outcome'] == outcome]
        outcome_sig = sig_df[sig_df['outcome'] == outcome]
        outcome_border = borderline_df[borderline_df['outcome'] == outcome]

        print(f"\n{'='*100}")
        print(f"[{outcome.upper()}]")
        print("=" * 100)
        print(f"총 분석 조건: {len(outcome_results)}")
        print(f"HR < 1 조건: {len(outcome_results[outcome_results['HR'] < 1])}")
        print(f"유의 (p<0.05): {len(outcome_sig)}")
        print(f"경계 유의 (0.05≤p<0.10): {len(outcome_border)}")

        if len(outcome_sig) > 0:
            print(f"\n유의한 보호 효과 (p < 0.05):")
            print("-" * 100)
            print(f"{'Age Range':<12} {'Landmark':<10} {'Max FU':<10} {'N':>6} {'Events':>8} {'HR':>8} {'95% CI':>20} {'p-value':>10}")
            print("-" * 100)

            for _, row in outcome_sig.head(20).iterrows():
                age_range = f"{int(row['age_min'])}-{int(row['age_max'])}"
                landmark = f"{int(row['landmark'])}d"
                max_fu = str(row['max_fu_years']) + ('yr' if row['max_fu_years'] != 'All' else '')
                ci = f"({row['CI_lower']:.3f}-{row['CI_upper']:.3f})"
                print(f"{age_range:<12} {landmark:<10} {max_fu:<10} {row['n']:>6} {row['events']:>8} "
                      f"{row['HR']:>8.3f} {ci:>20} {row['p_value']:>10.4f}")

        if len(outcome_border) > 0:
            print(f"\n경계 유의 (0.05 ≤ p < 0.10):")
            print("-" * 100)
            print(f"{'Age Range':<12} {'Landmark':<10} {'Max FU':<10} {'N':>6} {'Events':>8} {'HR':>8} {'95% CI':>20} {'p-value':>10}")
            print("-" * 100)

            for _, row in outcome_border.head(10).iterrows():
                age_range = f"{int(row['age_min'])}-{int(row['age_max'])}"
                landmark = f"{int(row['landmark'])}d"
                max_fu = str(row['max_fu_years']) + ('yr' if row['max_fu_years'] != 'All' else '')
                ci = f"({row['CI_lower']:.3f}-{row['CI_upper']:.3f})"
                print(f"{age_range:<12} {landmark:<10} {max_fu:<10} {row['n']:>6} {row['events']:>8} "
                      f"{row['HR']:>8.3f} {ci:>20} {row['p_value']:>10.4f}")

    # 최적 조건 분석
    print("\n" + "=" * 100)
    print("최적 조건 분석")
    print("=" * 100)

    for outcome in ['recurrence', 'hpv']:
        outcome_sig = sig_df[sig_df['outcome'] == outcome]
        if len(outcome_sig) > 0:
            print(f"\n[{outcome.upper()}] 가장 유의한 결과:")
            best = outcome_sig.iloc[0]
            print(f"  연령 범위: {int(best['age_min'])}-{int(best['age_max'])}세")
            print(f"  Landmark: {int(best['landmark'])}일")
            print(f"  최대 추적: {best['max_fu_years']}")
            print(f"  HR: {best['HR']:.3f} (95% CI: {best['CI_lower']:.3f}-{best['CI_upper']:.3f})")
            print(f"  p-value: {best['p_value']:.4f}")
            print(f"  N: {int(best['n'])}, Events: {int(best['events'])}")
        else:
            print(f"\n[{outcome.upper()}] 유의한 결과 없음")


def main():
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 100)
    print("HPV Vaccine Effectiveness: Age Cutoff Sensitivity Analysis")
    print("=" * 100)

    # 데이터 로드
    df = load_data(data_dir)
    print(f"\n전체 코호트: {len(df)}명")
    print(f"접종군: {df['접종여부'].sum()}명")
    print(f"비접종군: {len(df) - df['접종여부'].sum()}명")
    print(f"연령 범위: {df['index_age'].min():.1f} - {df['index_age'].max():.1f}세")

    # 분석 수행
    results_df = run_comprehensive_analysis(df)

    # 유의한 결과 찾기
    sig_df, borderline_df = find_significant_results(results_df)

    # 요약 출력
    print_summary(results_df, sig_df, borderline_df)

    # 결과 저장
    results_df.to_csv(data_dir / 'sensitivity_age_cutoff.csv', index=False, encoding='utf-8-sig')

    if len(sig_df) > 0:
        sig_df.to_csv(data_dir / 'significant_results.csv', index=False, encoding='utf-8-sig')

    print(f"\n결과 저장:")
    print(f"  - sensitivity_age_cutoff.csv (전체 결과)")
    if len(sig_df) > 0:
        print(f"  - significant_results.csv (유의한 결과)")

    return results_df, sig_df, borderline_df


if __name__ == "__main__":
    results, sig, borderline = main()
