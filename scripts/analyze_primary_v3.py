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

    # ===== Sustained clearance duration =====
    def sustained(row):
        if not row['has_clearance']:
            return None
        pid = row['연구번호']
        fd = row['first_neg_date']
        sub = mol_by_pid.get(pid)
        if sub is None:
            return (row['최종추적일자'] - fd).days
        post = sub[(sub['실시일자'] > fd) & (sub['hpv_pos'] == True)]
        end = post['실시일자'].min() if len(post) else row['최종추적일자']
        return (end - fd).days

    clr2['sustained_days'] = clr2.apply(sustained, axis=1)
    clr2['sustained_yrs'] = clr2['sustained_days'] / 365.25
    sc_rows = []
    for grp_val, label in [(True, 'Vaccinated'), (False, 'Non-vaccinated')]:
        sub = clr2[(clr2['접종여부'] == grp_val) & (clr2['has_clearance'])].copy()
        if len(sub) == 0:
            continue
        d = sub['sustained_yrs'].dropna()
        # Reversion: any HR+ test after first_neg_date
        rev = 0
        for _, r in sub.iterrows():
            pid = r['연구번호']
            fd = r['first_neg_date']
            sb = mol_by_pid.get(pid)
            if sb is None:
                continue
            if (sb[(sb['실시일자'] > fd)]['hpv_pos'] == True).any():
                rev += 1
        sc_rows.append(
            {
                'group': label,
                'n_clearance_events': int(len(sub)),
                'median_sustained_years': round(d.median(), 2),
                'IQR_lower_years': round(d.quantile(0.25), 2),
                'IQR_upper_years': round(d.quantile(0.75), 2),
                'reversion_n': rev,
                'reversion_pct': round(100 * rev / len(sub), 1),
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
