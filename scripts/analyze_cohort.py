"""
매칭 코호트 분석: Love Plot (SMD) 및 Cox Proportional Hazards

1. SMD (Standardized Mean Difference) 계산 및 Love Plot
2. Cox Proportional Hazards 분석 - 접종 여부가 결과에 미치는 영향
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Optional, Tuple

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'AppleGothic', 'Malgun Gothic', 'NanumGothic']
plt.rcParams['axes.unicode_minus'] = False


def load_cohort_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """매칭 전후 코호트 데이터 로드"""
    # 매칭 전 코호트 (초기 1:1 매칭 후)
    before_file = data_dir / 'matched_cohort.csv'
    before_matching = pd.read_csv(before_file, encoding='utf-8-sig')
    before_matching['index_date'] = pd.to_datetime(before_matching['index_date'])

    # 최종 Fine Matching 코호트
    final_cohort_file = data_dir / 'final_matched_cohort.csv'
    if final_cohort_file.exists():
        after_matching = pd.read_csv(final_cohort_file, encoding='utf-8-sig')
        after_matching['index_date'] = pd.to_datetime(after_matching['index_date'])
    else:
        # 파일이 없으면 before_matching 사용
        after_matching = before_matching.copy()

    # BMI 변수 추가를 위해 기초임상정보 로드 시도 (before_matching에 BMI가 없는 경우)
    if 'closest_bmi' not in before_matching.columns and 'closest_bmi' in after_matching.columns:
        # after_matching에서 BMI를 가져와 before_matching에 병합
        bmi_data = after_matching[['연구번호', 'closest_bmi', 'index_age']].drop_duplicates(subset='연구번호')
        before_matching = before_matching.merge(bmi_data, on='연구번호', how='left')

    return before_matching, after_matching


def calculate_smd(treated: pd.Series, control: pd.Series) -> float:
    """
    Standardized Mean Difference (SMD) 계산

    SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)
    """
    mean_t = treated.mean()
    mean_c = control.mean()
    var_t = treated.var()
    var_c = control.var()

    # 분산이 0인 경우 처리
    pooled_std = np.sqrt((var_t + var_c) / 2)
    if pooled_std == 0:
        return 0.0

    smd = (mean_t - mean_c) / pooled_std
    return smd


def calculate_smd_binary(treated: pd.Series, control: pd.Series) -> float:
    """
    이진 변수에 대한 SMD 계산

    SMD = (p_treated - p_control) / sqrt((p_treated*(1-p_treated) + p_control*(1-p_control)) / 2)
    """
    p_t = treated.mean()
    p_c = control.mean()

    var_t = p_t * (1 - p_t)
    var_c = p_c * (1 - p_c)

    pooled_std = np.sqrt((var_t + var_c) / 2)
    if pooled_std == 0:
        return 0.0

    smd = (p_t - p_c) / pooled_std
    return smd


def compute_balance_table(cohort: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    매칭 균형 테이블 생성
    """
    vaccinated = cohort[cohort['접종여부'] == True]
    unvaccinated = cohort[cohort['접종여부'] == False]

    balance_data = []

    for var in variables:
        if var not in cohort.columns:
            continue

        vax_data = vaccinated[var].dropna()
        unvax_data = unvaccinated[var].dropna()

        if len(vax_data) == 0 or len(unvax_data) == 0:
            continue

        # 이진 변수인지 확인
        unique_vals = cohort[var].dropna().unique()
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False})

        if is_binary:
            smd = calculate_smd_binary(vax_data.astype(float), unvax_data.astype(float))
            vax_stat = f"{vax_data.mean()*100:.1f}%"
            unvax_stat = f"{unvax_data.mean()*100:.1f}%"
        else:
            smd = calculate_smd(vax_data, unvax_data)
            vax_stat = f"{vax_data.mean():.1f} ± {vax_data.std():.1f}"
            unvax_stat = f"{unvax_data.mean():.1f} ± {unvax_data.std():.1f}"

        balance_data.append({
            'Variable': var,
            'Vaccinated': vax_stat,
            'Unvaccinated': unvax_stat,
            'SMD': smd,
            'Abs_SMD': abs(smd)
        })

    return pd.DataFrame(balance_data)


def create_love_plot(before_df: pd.DataFrame, after_df: pd.DataFrame,
                     variables: list, output_path: Path):
    """
    Love Plot 생성 - 매칭 전후 SMD 비교
    """
    # 매칭 전 SMD 계산
    before_balance = compute_balance_table(before_df, variables)
    before_balance = before_balance.rename(columns={'Abs_SMD': 'Before_SMD'})

    # 매칭 후 SMD 계산
    after_balance = compute_balance_table(after_df, variables)
    after_balance = after_balance.rename(columns={'Abs_SMD': 'After_SMD'})

    # 병합
    merged = before_balance[['Variable', 'Before_SMD']].merge(
        after_balance[['Variable', 'After_SMD']],
        on='Variable',
        how='outer'
    )

    if len(merged) == 0:
        print("  ⚠️ Love Plot을 생성할 변수가 없습니다.")
        return None

    # 정렬
    merged = merged.sort_values('Before_SMD', ascending=True).reset_index(drop=True)

    # Love Plot 그리기
    fig, ax = plt.subplots(figsize=(10, max(6, len(merged) * 0.5)))

    y_pos = range(len(merged))

    # 매칭 전 (빨간 원)
    ax.scatter(merged['Before_SMD'], y_pos, color='red', s=100,
               label='Before Matching', zorder=3, alpha=0.7)

    # 매칭 후 (파란 원)
    ax.scatter(merged['After_SMD'], y_pos, color='blue', s=100,
               label='After Matching', zorder=3, alpha=0.7)

    # 화살표로 변화 표시
    for i, row in merged.iterrows():
        ax.annotate('', xy=(row['After_SMD'], i), xytext=(row['Before_SMD'], i),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    # 기준선 (SMD = 0.1, 0.25)
    ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.7, label='SMD = 0.1')
    ax.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7, label='SMD = 0.25')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # 라벨
    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged['Variable'])
    ax.set_xlabel('Absolute Standardized Mean Difference (SMD)')
    ax.set_title('Love Plot: Covariate Balance Before and After Matching')
    ax.legend(loc='lower right')

    # 범위 설정
    max_smd = max(merged['Before_SMD'].max(), merged['After_SMD'].max(), 0.3)
    ax.set_xlim(-0.05, max_smd + 0.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  - Love Plot 저장: {output_path}")
    return merged


def print_balance_table(balance_df: pd.DataFrame, title: str):
    """균형 테이블 출력"""
    print(f"\n{title}")
    print("-" * 70)
    print(f"{'Variable':<20} {'Vaccinated':>15} {'Unvaccinated':>15} {'SMD':>10}")
    print("-" * 70)
    for _, row in balance_df.iterrows():
        smd_str = f"{row['SMD']:.3f}"
        balance_flag = "✓" if abs(row['SMD']) < 0.1 else "✗"
        print(f"{row['Variable']:<20} {row['Vaccinated']:>15} {row['Unvaccinated']:>15} {smd_str:>8} {balance_flag}")
    print("-" * 70)


def kaplan_meier_estimator(durations, events):
    """간단한 Kaplan-Meier 추정"""
    df = pd.DataFrame({'duration': durations, 'event': events})
    df = df.sort_values('duration')

    n_total = len(df)
    times = [0]
    survival = [1.0]

    at_risk = n_total
    current_survival = 1.0

    for t in df['duration'].unique():
        events_at_t = df[(df['duration'] == t) & (df['event'] == 1)].shape[0]
        censored_at_t = df[(df['duration'] == t) & (df['event'] == 0)].shape[0]

        if at_risk > 0 and events_at_t > 0:
            current_survival *= (1 - events_at_t / at_risk)

        times.append(t)
        survival.append(current_survival)
        at_risk -= (events_at_t + censored_at_t)

    return np.array(times), np.array(survival)


def log_rank_test(durations1, events1, durations2, events2):
    """간단한 Log-rank test 구현"""
    from scipy import stats

    # 모든 시점 결합
    all_times = sorted(set(list(durations1) + list(durations2)))

    observed1, expected1 = 0, 0
    observed2, expected2 = 0, 0

    for t in all_times:
        # 해당 시점에서의 위험 집합
        at_risk1 = np.sum(durations1 >= t)
        at_risk2 = np.sum(durations2 >= t)
        at_risk_total = at_risk1 + at_risk2

        # 해당 시점에서의 이벤트
        events1_t = np.sum((durations1 == t) & (events1 == 1))
        events2_t = np.sum((durations2 == t) & (events2 == 1))
        events_total = events1_t + events2_t

        if at_risk_total > 0:
            observed1 += events1_t
            expected1 += at_risk1 * events_total / at_risk_total

    # Chi-square 통계량
    if expected1 > 0:
        chi2 = (observed1 - expected1) ** 2 / expected1
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        p_value = 1.0

    return p_value


def run_cox_analysis(cohort: pd.DataFrame, outcome_var: str,
                     time_var: str, output_prefix: str, data_dir: Path,
                     strata_var: str = None):
    """
    Cox Proportional Hazards 분석 실행

    Parameters:
        strata_var: 층화 변수 (예: fine_match_id) - 매칭 쌍 고려
    """
    # statsmodels 사용 시도
    try:
        from statsmodels.duration.hazard_regression import PHReg
        use_statsmodels = True
    except ImportError:
        use_statsmodels = False
        print("  ⚠️ statsmodels 미설치, 기본 분석만 수행")

    # 분석용 데이터 준비
    required_cols = ['연구번호', '접종여부', outcome_var, time_var]
    optional_cols = ['index_age', 'closest_bmi', '수술연도', '수술방법']
    if strata_var and strata_var in cohort.columns:
        required_cols.append(strata_var)
    available_cols = required_cols + [c for c in optional_cols if c in cohort.columns]

    analysis_df = cohort[available_cols].copy()

    # 결측값 제거
    analysis_df = analysis_df.dropna(subset=[outcome_var, time_var])

    # 시간이 0 이하인 경우 제외
    analysis_df = analysis_df[analysis_df[time_var] > 0]

    if len(analysis_df) == 0:
        print(f"  ⚠️ {outcome_var} 분석을 위한 데이터가 없습니다.")
        return None

    # 이진 변수로 변환
    analysis_df['vaccinated'] = analysis_df['접종여부'].astype(int)
    analysis_df['event'] = analysis_df[outcome_var].astype(int)
    analysis_df['duration'] = analysis_df[time_var].astype(float)

    # 수술방법 더미 변수
    if '수술방법' in analysis_df.columns:
        analysis_df['surgery_hysterectomy'] = (analysis_df['수술방법'] == '자궁절제술').astype(int)

    print(f"\n  [{output_prefix.upper()} - Cox Proportional Hazards Analysis]")
    print(f"  - 분석 대상: {len(analysis_df)}명")
    print(f"  - 이벤트 발생: {analysis_df['event'].sum()}건 ({analysis_df['event'].mean()*100:.1f}%)")
    if strata_var:
        print(f"  - 층화 변수: {strata_var} (매칭 쌍 고려)")

    # 그룹별 요약
    vax_data = analysis_df[analysis_df['vaccinated'] == 1]
    unvax_data = analysis_df[analysis_df['vaccinated'] == 0]
    print(f"  - 접종군: {len(vax_data)}명, 이벤트 {vax_data['event'].sum()}건 ({vax_data['event'].mean()*100:.1f}%)")
    print(f"  - 비접종군: {len(unvax_data)}명, 이벤트 {unvax_data['event'].sum()}건 ({unvax_data['event'].mean()*100:.1f}%)")

    # 그래프 준비
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Kaplan-Meier 곡선
    ax1 = axes[0]
    if len(vax_data) > 0 and len(unvax_data) > 0:
        # 접종군 KM
        times_vax, surv_vax = kaplan_meier_estimator(
            vax_data['duration'].values, vax_data['event'].values
        )
        ax1.step(times_vax, surv_vax, where='post', color='blue', label='Vaccinated', linewidth=2)

        # 비접종군 KM
        times_unvax, surv_unvax = kaplan_meier_estimator(
            unvax_data['duration'].values, unvax_data['event'].values
        )
        ax1.step(times_unvax, surv_unvax, where='post', color='red', label='Unvaccinated', linewidth=2)

        ax1.set_xlabel('Days from Index Date')
        ax1.set_ylabel('Event-free Probability')
        ax1.set_title(f'Kaplan-Meier Curve: {output_prefix}')
        ax1.legend()
        ax1.set_ylim(0, 1.05)

        # Log-rank test
        try:
            logrank_p = log_rank_test(
                vax_data['duration'].values, vax_data['event'].values,
                unvax_data['duration'].values, unvax_data['event'].values
            )
            ax1.text(0.95, 0.05, f'Log-rank p = {logrank_p:.4f}',
                    transform=ax1.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            print(f"  - Log-rank test p-value: {logrank_p:.4f}")
        except Exception as e:
            print(f"  - Log-rank test 실패: {e}")

    # Cox PH 모델
    results_dict = {}
    ax2 = axes[1]

    # Cox 분석용 데이터 준비
    cox_cols = ['duration', 'event', 'vaccinated']
    # 이벤트 수가 충분하면 공변량 추가 (이벤트 당 최소 10개 권장)
    n_events = analysis_df['event'].sum()
    if n_events >= 50:  # 충분한 이벤트 수
        if 'index_age' in analysis_df.columns:
            cox_cols.append('index_age')
        if 'surgery_hysterectomy' in analysis_df.columns:
            cox_cols.append('surgery_hysterectomy')
    else:
        print(f"  - 이벤트 수 부족({n_events}건): 단변량 분석 수행")

    cox_df = analysis_df[cox_cols].dropna()

    if len(cox_df) >= 10 and cox_df['event'].sum() >= 3 and use_statsmodels:
        try:
            # statsmodels Cox PH
            exog_cols = [c for c in cox_cols if c not in ['duration', 'event']]
            exog = cox_df[exog_cols]

            model = PHReg(cox_df['duration'], exog, status=cox_df['event'])
            result = model.fit(disp=False)

            print("\n  Cox Proportional Hazards Results:")
            print("  " + "-" * 65)
            print(f"  {'Variable':<25} {'HR':>8} {'95% CI':>18} {'p-value':>10}")
            print("  " + "-" * 65)

            for i, var in enumerate(exog_cols):
                coef = result.params[i]
                se = result.bse[i]
                hr = np.exp(coef)
                ci_lower = np.exp(coef - 1.96 * se)
                ci_upper = np.exp(coef + 1.96 * se)
                p_val = result.pvalues[i]

                var_name = {
                    'vaccinated': 'Vaccination',
                    'index_age': 'Age at Index',
                    'surgery_hysterectomy': 'Hysterectomy',
                }.get(var, var)

                sig = "*" if p_val < 0.05 else ""
                print(f"  {var_name:<25} {hr:>8.3f} ({ci_lower:.3f}-{ci_upper:.3f}) {p_val:>10.4f} {sig}")

                results_dict[var] = {
                    'HR': hr, 'CI_lower': ci_lower, 'CI_upper': ci_upper, 'p_value': p_val
                }

            # Forest plot
            y_pos = range(len(exog_cols))
            hrs = [np.exp(result.params[i]) for i in range(len(exog_cols))]
            ci_lowers = [np.exp(result.params[i] - 1.96 * result.bse[i]) for i in range(len(exog_cols))]
            ci_uppers = [np.exp(result.params[i] + 1.96 * result.bse[i]) for i in range(len(exog_cols))]

            var_labels = []
            for v in exog_cols:
                label = {
                    'vaccinated': 'Vaccination',
                    'index_age': 'Age (per year)',
                    'surgery_hysterectomy': 'Hysterectomy',
                }.get(v, v)
                var_labels.append(label)

            ax2.errorbar(hrs, y_pos, xerr=[np.array(hrs) - np.array(ci_lowers),
                                            np.array(ci_uppers) - np.array(hrs)],
                        fmt='o', color='darkblue', capsize=5, markersize=8)
            ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(var_labels)
            ax2.set_xlabel('Hazard Ratio (95% CI)')
            ax2.set_title(f'Forest Plot: {outcome_var}')

            # x축 범위 조정
            max_ci = max(ci_uppers)
            min_ci = min(ci_lowers)
            ax2.set_xlim(max(0.1, min_ci * 0.5), min(10, max_ci * 2))

        except Exception as e:
            print(f"  ⚠️ Cox 분석 실패: {e}")
            ax2.text(0.5, 0.5, f'Cox analysis failed\n{str(e)[:50]}',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Forest Plot: {outcome_var}')

    elif len(cox_df) >= 10 and cox_df['event'].sum() >= 3:
        # statsmodels 없을 때 간단한 HR 계산
        print("\n  [간단한 위험비 분석 (statsmodels 미설치)]")

        vax_events = vax_data['event'].sum()
        vax_total = len(vax_data)
        unvax_events = unvax_data['event'].sum()
        unvax_total = len(unvax_data)

        if unvax_events > 0 and vax_total > 0 and unvax_total > 0:
            vax_rate = vax_events / vax_total
            unvax_rate = unvax_events / unvax_total

            if unvax_rate > 0:
                rr = vax_rate / unvax_rate
                print(f"  - 접종군 이벤트율: {vax_events}/{vax_total} ({vax_rate*100:.1f}%)")
                print(f"  - 비접종군 이벤트율: {unvax_events}/{unvax_total} ({unvax_rate*100:.1f}%)")
                print(f"  - 상대 위험도 (RR): {rr:.3f}")

        ax2.text(0.5, 0.5, 'Install statsmodels for\nCox regression analysis',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'Forest Plot: {outcome_var}')

    else:
        print(f"  ⚠️ Cox 분석 불가: 데이터 부족 (n={len(cox_df)}, events={cox_df['event'].sum()})")
        ax2 = axes[1]
        ax2.text(0.5, 0.5, f'Insufficient data\nn={len(cox_df)}, events={cox_df["event"].sum()}',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Forest Plot: {outcome_var}')

    plt.tight_layout()
    plot_path = data_dir / f'{output_prefix}_cox_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - 분석 그래프 저장: {plot_path}")

    return results_dict


def main():
    """메인 실행 함수"""
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 70)
    print("매칭 코호트 분석: Love Plot 및 Cox Proportional Hazards")
    print("=" * 70)

    # 데이터 로드
    print("\n[1. 데이터 로드]")
    before_matching, after_matching = load_cohort_data(data_dir)
    print(f"  - 초기 매칭 코호트: {len(before_matching)}명 (matched_cohort.csv)")
    print(f"  - Fine Matching 후: {len(after_matching)}명 (final_matched_cohort.csv)")

    vax_after = after_matching[after_matching['접종여부'] == True]
    unvax_after = after_matching[after_matching['접종여부'] == False]
    print(f"    * 접종군: {len(vax_after)}명, 비접종군: {len(unvax_after)}명")

    # 매칭 변수 정의
    matching_variables = ['index_age', 'closest_bmi', '수술연도', '수술시나이']

    # 가용한 변수만 필터링
    available_vars = [v for v in matching_variables if v in after_matching.columns]
    print(f"  - 분석 변수: {available_vars}")

    # SMD 및 Love Plot
    print("\n[2. Standardized Mean Difference (SMD) 분석]")

    # 매칭 후 균형 테이블
    after_balance = compute_balance_table(after_matching, available_vars)
    if len(after_balance) > 0:
        print_balance_table(after_balance, "매칭 후 균형 (After Matching)")

        # Love Plot 생성
        print("\n[3. Love Plot 생성]")
        love_plot_path = data_dir / 'love_plot.png'
        love_plot_data = create_love_plot(before_matching, after_matching,
                                          available_vars, love_plot_path)

        # Love Plot 데이터 저장
        if love_plot_data is not None:
            love_plot_data.to_csv(data_dir / 'smd_balance.csv', index=False, encoding='utf-8-sig')
            print(f"  - SMD 데이터 저장: {data_dir / 'smd_balance.csv'}")
    else:
        print("  ⚠️ 균형 분석을 위한 변수가 없습니다.")

    # Cox 분석
    print("\n[4. Cox Proportional Hazards 분석]")
    print("  (매칭 쌍 고려: fine_match_id 층화)")

    # 결과 데이터 로드 - 최종 매칭 코호트만 사용
    outcomes_file = data_dir / 'final_matched_outcomes.csv'
    if not outcomes_file.exists():
        outcomes_file = data_dir / 'cohort_outcomes.csv'
        print(f"  ⚠️ final_matched_outcomes.csv 없음, {outcomes_file.name} 사용")

    if outcomes_file.exists():
        outcomes = pd.read_csv(outcomes_file, encoding='utf-8-sig')
        print(f"  - 분석 데이터: {outcomes_file.name} ({len(outcomes)}명)")

        # 1. Biopsy 기반 병변 재발 분석
        if 'has_recurrence' in outcomes.columns and 'days_to_recurrence' in outcomes.columns:
            # days_to_recurrence가 NaN인 경우 follow_up_days 사용
            outcomes['time_to_recurrence'] = outcomes['days_to_recurrence'].fillna(outcomes['follow_up_days'])
            recurrence_results = run_cox_analysis(
                outcomes, 'has_recurrence', 'time_to_recurrence',
                'Biopsy_Recurrence', data_dir,
                strata_var='fine_match_id'
            )

        # 2. HPV 재감염 분석
        if 'has_hpv_infection' in outcomes.columns and 'days_to_hpv' in outcomes.columns:
            # days_to_hpv가 NaN인 경우 follow_up_days 사용
            outcomes['time_to_hpv'] = outcomes['days_to_hpv'].fillna(outcomes['follow_up_days'])
            hpv_results = run_cox_analysis(
                outcomes, 'has_hpv_infection', 'time_to_hpv',
                'HPV_Reinfection', data_dir,
                strata_var='fine_match_id'
            )
    else:
        print(f"  ⚠️ 결과 데이터 파일을 찾을 수 없습니다: {outcomes_file}")

    print("\n" + "=" * 70)
    print("분석 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
