"""
Analyze Cohort B under v3 primary definition (≥2 dose + 3-mo landmark).

Outputs:
  Data/CohortB_HR_v3.csv             — co-primary HRs + sensitivities
  Data/CohortB_SustainedClearance.csv — duration of sustained clearance per arm
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from extract_pathology_outcomes import detect_high_risk_hpv  # noqa: E402

PATH_FILE = ROOT / 'Data' / '한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV'
COH_FILE = ROOT / 'Data' / 'primary_cohort_v3.csv'
OUT_HR = ROOT / 'Data' / 'CohortB_HR_v3.csv'
OUT_SC = ROOT / 'Data' / 'CohortB_SustainedClearance.csv'

LANDMARK_DAYS = 90


def load_mol() -> dict[int, pd.DataFrame]:
    path = pd.read_csv(PATH_FILE, encoding='cp949', low_memory=False)
    path['실시일자'] = pd.to_datetime(path['실시일자'], format='%Y%m%d', errors='coerce')
    mol = path[path['병리검사구분'].isin(['분자병리', 'HPV'])].copy()
    mol = mol.dropna(subset=['실시일자', '판독결과'])
    res = mol['판독결과'].apply(detect_high_risk_hpv)
    mol['hpv_pos'] = res.apply(lambda d: d['is_high_risk_hpv_positive'])
    mol['hpv_types_list'] = res.apply(lambda d: d['detected_hpv_types'])
    return {pid: g.sort_values('실시일자') for pid, g in mol.groupby('연구번호')}


def has_prevac_hr(mol_by_pid, pid, idx_dt) -> bool:
    sub = mol_by_pid.get(pid)
    if sub is None:
        return False
    return bool((sub[sub['실시일자'] < idx_dt]['hpv_pos'] == True).any())


def first_two_consecutive_neg(mol_by_pid, pid, idx_dt):
    sub = mol_by_pid.get(pid)
    if sub is None:
        return None
    sub = sub[sub['실시일자'] > idx_dt]
    if len(sub) < 2:
        return None
    pos = sub['hpv_pos'].values
    dates = sub['실시일자'].values
    for i in range(len(pos) - 1):
        if (not pos[i]) and (not pos[i + 1]):
            return pd.Timestamp(dates[i])
    return None


def fit_cox(df, t_col, e_col, label):
    fit = df[[t_col, e_col, 'index_age', 'fine_match_id']].copy()
    fit['vaccinated'] = df['접종여부'].astype(int)
    fit[e_col] = fit[e_col].astype(int)
    fit = fit[fit[t_col] > 0].copy()
    n_vac = int(fit['vaccinated'].sum())
    n_non = int((1 - fit['vaccinated']).sum())
    ev_vac = int(fit.loc[fit['vaccinated'] == 1, e_col].sum())
    ev_non = int(fit.loc[fit['vaccinated'] == 0, e_col].sum())
    py_vac = float(fit.loc[fit['vaccinated'] == 1, t_col].sum() / 365.25)
    py_non = float(fit.loc[fit['vaccinated'] == 0, t_col].sum() / 365.25)
    cph = CoxPHFitter()
    cph.fit(fit, duration_col=t_col, event_col=e_col, cluster_col='fine_match_id')
    s = cph.summary.loc['vaccinated']
    return {
        'analysis': label,
        'n_vac': n_vac,
        'n_non': n_non,
        'events_vac': ev_vac,
        'events_non': ev_non,
        'PY_vac': round(py_vac, 1),
        'PY_non': round(py_non, 1),
        'IR_vac_per1000PY': round(1000 * ev_vac / py_vac, 2) if py_vac > 0 else np.nan,
        'IR_non_per1000PY': round(1000 * ev_non / py_non, 2) if py_non > 0 else np.nan,
        'HR': round(s['exp(coef)'], 3),
        'CI_lower': round(s['exp(coef) lower 95%'], 3),
        'CI_upper': round(s['exp(coef) upper 95%'], 3),
        'p': round(s['p'], 4),
    }


def apply_landmark(df, event_col, event_date_col, lm_days=LANDMARK_DAYS):
    df = df.copy()
    df['lm_zero'] = df['index_date'] + pd.Timedelta(days=lm_days)
    pre_lm = (df[event_col] == True) & (df[event_date_col] < df['lm_zero'])
    bad_fids = df[(df['접종여부'] == True) & pre_lm]['fine_match_id'].unique()
    df = df[~df['fine_match_id'].isin(bad_fids)].copy()
    df = df[~pre_lm.loc[df.index]].copy()
    df['t'] = np.where(
        df[event_col] == True,
        (df[event_date_col] - df['lm_zero']).dt.days,
        (df['최종추적일자'] - df['lm_zero']).dt.days,
    )
    return df[df['t'] > 0].copy()


def main():
    mol_by_pid = load_mol()

    df = pd.read_csv(COH_FILE, encoding='utf-8-sig')
    df['index_date'] = pd.to_datetime(df['index_date'])
    df['최종추적일자'] = pd.to_datetime(df['최종추적일자'])
    df['recurrence_date'] = pd.to_datetime(df['recurrence_date'], errors='coerce')

    rows = []

    # ===== P1: lesion recurrence (CIN2+) =====
    p1 = apply_landmark(df, 'has_recurrence', 'recurrence_date')
    rows.append(fit_cox(p1, 't', 'has_recurrence',
                         'P1 — Lesion recurrence (CIN2+); ≥2 dose + 3mo landmark'))

    # ===== P2: hr-HPV clearance (2-consecutive-negatives) =====
    df['prevac_hr'] = df.apply(
        lambda r: has_prevac_hr(mol_by_pid, r['연구번호'], r['index_date']), axis=1
    )
    fids_with_vac_hr = df[(df['접종여부'] == True) & (df['prevac_hr'] == True)][
        'fine_match_id'
    ].unique()
    clr = df[df['fine_match_id'].isin(fids_with_vac_hr) & (df['prevac_hr'] == True)].copy()
    clr['first_neg_date'] = clr.apply(
        lambda r: first_two_consecutive_neg(mol_by_pid, r['연구번호'], r['index_date']),
        axis=1,
    )
    clr['has_clearance'] = clr['first_neg_date'].notna()
    clr2 = apply_landmark(clr, 'has_clearance', 'first_neg_date')
    rows.append(fit_cox(clr2, 't', 'has_clearance',
                         'P2 — hr-HPV clearance; ≥2 dose + 3mo landmark'))

    # ===== Sensitivity: ≥1 dose (any), no landmark =====
    out_all = pd.read_csv(ROOT / 'Data' / 'final_matched_outcomes.csv', encoding='utf-8-sig')
    out_all['index_date'] = pd.to_datetime(out_all['index_date'])
    out_all['최종추적일자'] = pd.to_datetime(out_all['최종추적일자'])
    out_all['recurrence_date'] = pd.to_datetime(out_all['recurrence_date'], errors='coerce')
    out_all['t'] = np.where(
        out_all['has_recurrence'],
        (out_all['recurrence_date'] - out_all['index_date']).dt.days,
        (out_all['최종추적일자'] - out_all['index_date']).dt.days,
    )
    rows.append(fit_cox(out_all[out_all['t'] > 0].copy(), 't', 'has_recurrence',
                         'Sens — Lesion recurrence; ≥1 dose, no landmark'))

    # ===== Sensitivity: ≥3 dose, no landmark =====
    doses = (
        pd.read_csv(ROOT / 'Data' / 'primary_cohort_v3.csv', encoding='utf-8-sig')[
            ['연구번호', 'dose_count']
        ]
        .drop_duplicates('연구번호')
        .set_index('연구번호')['dose_count']
    )
    # Need full final_matched_outcomes with dose merged
    o3 = out_all.merge(doses.rename('dose_count'), left_on='연구번호',
                        right_index=True, how='left')
    o3['dose_count'] = o3['dose_count'].fillna(0).astype(int)
    bad3 = o3[(o3['접종여부'] == True) & (o3['dose_count'] < 3)]['fine_match_id'].unique()
    o3 = o3[~o3['fine_match_id'].isin(bad3)].copy()
    rows.append(fit_cox(o3[o3['t'] > 0].copy(), 't', 'has_recurrence',
                         'Sens — Lesion recurrence; ≥3 dose, no landmark'))

    # ===== Sustained clearance duration — Kaplan–Meier =====
    # Among patients who achieved clearance (first_neg_date observed),
    # time-to-reversion is analysed by KM. Time origin = first_neg_date.
    # Event = first subsequent HR-HPV+ molecular pathology record (reversion).
    # Censoring = last follow-up date (clearance still sustained).
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    def time_and_event(row):
        if not row['has_clearance']:
            return pd.Series({'sus_days': np.nan, 'reversion': np.nan})
        pid = row['연구번호']
        fd = row['first_neg_date']
        sub = mol_by_pid.get(pid)
        post_pos = sub[(sub['실시일자'] > fd) & (sub['hpv_pos'] == True)] if sub is not None else pd.DataFrame()
        if len(post_pos) > 0:
            ev_date = post_pos['실시일자'].min()
            return pd.Series({'sus_days': (ev_date - fd).days, 'reversion': 1})
        else:
            return pd.Series({'sus_days': (row['최종추적일자'] - fd).days, 'reversion': 0})

    clr2[['sus_days', 'reversion']] = clr2.apply(time_and_event, axis=1)
    clr2['sus_yrs'] = clr2['sus_days'] / 365.25

    sc_rows = []
    sub_groups = {}
    for grp_val, label in [(True, 'Vaccinated'), (False, 'Non-vaccinated')]:
        sub = clr2[(clr2['접종여부'] == grp_val) & (clr2['has_clearance'])].copy()
        sub = sub[sub['sus_days'] > 0].copy()
        if len(sub) == 0:
            continue
        sub_groups[label] = sub
        kmf = KaplanMeierFitter()
        kmf.fit(sub['sus_yrs'], event_observed=sub['reversion'], label=label)
        med = kmf.median_survival_time_
        # IQR from KM: time when S(t) = 0.75 and 0.25
        try:
            ci = kmf.median_survival_times_.iloc[0]
        except Exception:
            ci = None
        # Compute 25th and 75th percentiles from KM survival function
        sf = kmf.survival_function_.iloc[:, 0]

        def percentile_t(target):
            # smallest t where S(t) <= target
            below = sf[sf <= target]
            return float(below.index[0]) if len(below) > 0 else np.nan

        q25 = percentile_t(0.75)  # 25th percentile of reversion time = S(t)=0.75
        q75 = percentile_t(0.25)  # 75th percentile = S(t)=0.25
        med_str = (f'{med:.2f}' if not (isinstance(med, float) and (np.isinf(med) or np.isnan(med)))
                   else 'NR (not reached)')
        sc_rows.append(
            {
                'group': label,
                'n_clearance_events': int(len(sub)),
                'reversion_events': int(sub['reversion'].sum()),
                'censored': int((sub['reversion'] == 0).sum()),
                'KM_median_sustained_years': med_str,
                'KM_q25_years': round(q25, 2) if not np.isnan(q25) else 'NR',
                'KM_q75_years': round(q75, 2) if not np.isnan(q75) else 'NR',
            }
        )

    # Log-rank test (vac vs non-vac, sustained clearance time)
    if 'Vaccinated' in sub_groups and 'Non-vaccinated' in sub_groups:
        v_, n_ = sub_groups['Vaccinated'], sub_groups['Non-vaccinated']
        lr = logrank_test(v_['sus_yrs'], n_['sus_yrs'],
                           event_observed_A=v_['reversion'],
                           event_observed_B=n_['reversion'])
        sc_rows.append(
            {
                'group': 'Log-rank (vac vs non-vac)',
                'n_clearance_events': '',
                'reversion_events': '',
                'censored': '',
                'KM_median_sustained_years': f'χ²={lr.test_statistic:.2f}',
                'KM_q25_years': '',
                'KM_q75_years': f'p={lr.p_value:.3f}',
            }
        )

    pd.DataFrame(sc_rows).to_csv(OUT_SC, index=False, encoding='utf-8-sig')
    print(f'Wrote {OUT_SC.relative_to(ROOT)}')

    res = pd.DataFrame(rows)
    res.to_csv(OUT_HR, index=False, encoding='utf-8-sig')
    print(f'Wrote {OUT_HR.relative_to(ROOT)}')
    print()
    print(res.to_string(index=False))
    print()
    print('=== Sustained clearance ===')
    print(pd.DataFrame(sc_rows).to_string(index=False))


if __name__ == '__main__':
    main()
