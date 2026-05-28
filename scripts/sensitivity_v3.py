"""
Recompute key sensitivity analyses (Sens-A, Sens-B, Sens-E) under the
v3 primary cohort (≥2 dose + 3-mo landmark).

Sens-A: single-negative clearance event definition (vs 2-consecutive-neg primary)
Sens-B: time-stratified clearance HR (0–6, 6–12, 12–24, ≥24 mo)
Sens-E: lesion recurrence with minimum disease-free interval (3/6/12 mo)

Outputs:
  Data/Sensitivity_HPV_Clearance_SingleNegative_v3.csv
  Data/Sensitivity_HPV_Clearance_TimeStratified_v3.csv
  Data/Sensitivity_Recurrence_DFInterval_v3.csv
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

LM_DAYS = 90


def load_mol():
    path = pd.read_csv(
        ROOT / 'Data' / '한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV',
        encoding='cp949',
        low_memory=False,
    )
    path['실시일자'] = pd.to_datetime(path['실시일자'], format='%Y%m%d', errors='coerce')
    mol = path[path['병리검사구분'].isin(['분자병리', 'HPV'])].dropna(
        subset=['실시일자', '판독결과']
    )
    res = mol['판독결과'].apply(detect_high_risk_hpv)
    mol = mol.assign(hpv_pos=res.apply(lambda d: d['is_high_risk_hpv_positive']))
    return {pid: g.sort_values('실시일자') for pid, g in mol.groupby('연구번호')}


def cox_hr(df, t_col, e_col):
    fit = df[[t_col, e_col, 'index_age', 'fine_match_id']].copy()
    fit['vaccinated'] = df['접종여부'].astype(int)
    fit[e_col] = fit[e_col].astype(int)
    fit = fit[fit[t_col] > 0].copy()
    n_v = int(fit['vaccinated'].sum())
    n_c = int((1 - fit['vaccinated']).sum())
    ev_v = int(fit.loc[fit['vaccinated'] == 1, e_col].sum())
    ev_c = int(fit.loc[fit['vaccinated'] == 0, e_col].sum())
    if ev_v + ev_c < 3:
        return n_v, n_c, ev_v, ev_c, np.nan, np.nan, np.nan, np.nan
    cph = CoxPHFitter()
    cph.fit(fit, duration_col=t_col, event_col=e_col, cluster_col='fine_match_id')
    s = cph.summary.loc['vaccinated']
    return (
        n_v, n_c, ev_v, ev_c,
        float(s['exp(coef)']),
        float(s['exp(coef) lower 95%']),
        float(s['exp(coef) upper 95%']),
        float(s['p']),
    )


def main():
    mol = load_mol()

    df = pd.read_csv(ROOT / 'Data' / 'primary_cohort_v3.csv', encoding='utf-8-sig')
    df['index_date'] = pd.to_datetime(df['index_date'])
    df['최종추적일자'] = pd.to_datetime(df['최종추적일자'])
    df['recurrence_date'] = pd.to_datetime(df['recurrence_date'], errors='coerce')
    df['lm_zero'] = df['index_date'] + pd.Timedelta(days=LM_DAYS)

    def prevac_hr(pid, idx):
        sub = mol.get(pid)
        return False if sub is None else bool(
            (sub[sub['실시일자'] < idx]['hpv_pos'] == True).any()
        )

    df['prevac_hr'] = df.apply(
        lambda r: prevac_hr(r['연구번호'], r['index_date']), axis=1
    )
    fids = df[(df['접종여부'] == True) & (df['prevac_hr'] == True)]['fine_match_id'].unique()
    clr = df[df['fine_match_id'].isin(fids) & (df['prevac_hr'] == True)].copy()

    # ----- Sens-A: single-negative vs two-consecutive-negative clearance -----
    # Use the SAME matched-set integrity rule as the primary analysis:
    # search event from index_date, apply landmark by dropping sets with
    # vaccinated case having early event AND removing the early-event row.
    def first_two_neg(pid, idx):
        sub = mol.get(pid)
        if sub is None:
            return None
        sub = sub[sub['실시일자'] > idx]
        if len(sub) < 2:
            return None
        pos = sub['hpv_pos'].values
        dates = sub['실시일자'].values
        for i in range(len(pos) - 1):
            if (not pos[i]) and (not pos[i + 1]):
                return pd.Timestamp(dates[i])
        return None

    def first_single_neg(pid, idx):
        sub = mol.get(pid)
        if sub is None:
            return None
        sub = sub[(sub['실시일자'] > idx) & (sub['hpv_pos'] == False)]
        return pd.Timestamp(sub.iloc[0]['실시일자']) if len(sub) else None

    def apply_landmark_to_clr(c, event_col, event_date_col):
        c = c.copy()
        pre_lm = (c[event_col] == True) & (c[event_date_col] < c['lm_zero'])
        bad = c[(c['접종여부'] == True) & pre_lm]['fine_match_id'].unique()
        c = c[~c['fine_match_id'].isin(bad)].copy()
        c = c[~pre_lm.loc[c.index]].copy()
        # Time origin at index date; landmark applied by excluding pre-landmark events
        c['t'] = np.where(
            c[event_col] == True,
            (c[event_date_col] - c['index_date']).dt.days,
            (c['최종추적일자'] - c['index_date']).dt.days,
        )
        return c[c['t'] > 0].copy()

    sens_a_rows = []
    for label, fn in [
        ('Two consecutive negatives (PRIMARY, v3)', first_two_neg),
        ('Single negative test (sensitivity, v3)', first_single_neg),
    ]:
        c = clr.copy()
        c['ev_date'] = c.apply(lambda r: fn(r['연구번호'], r['index_date']), axis=1)
        c['event'] = c['ev_date'].notna()
        c2 = apply_landmark_to_clr(c, 'event', 'ev_date')
        n_v, n_c, ev_v, ev_c, hr, lo, hi, p = cox_hr(c2, 't', 'event')
        sens_a_rows.append(
            {
                'definition': label,
                'n_v': n_v, 'n_c': n_c, 'ev_v': ev_v, 'ev_c': ev_c,
                'HR': round(hr, 3) if not np.isnan(hr) else 'NA',
                'CIlo': round(lo, 3) if not np.isnan(lo) else 'NA',
                'CIhi': round(hi, 3) if not np.isnan(hi) else 'NA',
                'p': round(p, 4) if not np.isnan(p) else 'NA',
            }
        )
    pd.DataFrame(sens_a_rows).to_csv(
        ROOT / 'Data' / 'Sensitivity_HPV_Clearance_SingleNegative_v3.csv',
        index=False, encoding='utf-8-sig',
    )
    print('Wrote Data/Sensitivity_HPV_Clearance_SingleNegative_v3.csv')

    # ----- Sens-B: time-stratified clearance HR under v3 -----
    # Build clearance dataset with same primary integrity rules as analyze_primary_v3
    c = clr.copy()
    c['ev_date'] = c.apply(lambda r: first_two_neg(r['연구번호'], r['index_date']), axis=1)
    c['event'] = c['ev_date'].notna()
    c = apply_landmark_to_clr(c, 'event', 'ev_date')
    c['event'] = c['event'].astype(int)

    sens_b_rows = []
    # Time windows measured from index date (consistent with primary analysis).
    # Patients are at risk only from the 3-month landmark by construction; the
    # first window therefore effectively spans (90 d, 180 d] after index.
    windows = [
        ('Overall (post-index, ≥90 d)', 0, np.inf),
        ('0–6 months (from index)', 0, 180),
        ('6–12 months (from index)', 180, 365),
        ('12–24 months (from index)', 365, 730),
        ('≥24 months (from index)', 730, np.inf),
    ]
    for label, lo_d, hi_d in windows:
        sub = c.copy()
        if np.isfinite(hi_d):
            # In window: event must be in (lo_d, hi_d], or follow-up reaches hi_d
            in_win = (sub['t'] > lo_d) & (sub['t'] <= hi_d)
            sub['win_t'] = np.where(in_win,
                                    sub['t'] - lo_d,
                                    np.where(sub['t'] > hi_d, hi_d - lo_d, np.nan))
            sub['win_event'] = ((sub['event'] == 1) & (sub['t'] > lo_d) & (sub['t'] <= hi_d)).astype(int)
        else:
            in_or_past = sub['t'] > lo_d
            sub['win_t'] = np.where(in_or_past, sub['t'] - lo_d, np.nan)
            sub['win_event'] = ((sub['event'] == 1) & (sub['t'] > lo_d)).astype(int)
        sub = sub.dropna(subset=['win_t']).copy()
        n_v, n_c, ev_v, ev_c, hr, lwr, upr, p = cox_hr(sub, 'win_t', 'win_event')
        sens_b_rows.append(
            {
                'period': label,
                'n_v': n_v, 'n_c': n_c, 'ev_v': ev_v, 'ev_c': ev_c,
                'HR': round(hr, 3) if not np.isnan(hr) else 'NA',
                'CIlo': round(lwr, 3) if not np.isnan(lwr) else 'NA',
                'CIhi': round(upr, 3) if not np.isnan(upr) else 'NA',
                'p': round(p, 4) if not np.isnan(p) else 'NA',
            }
        )
    pd.DataFrame(sens_b_rows).to_csv(
        ROOT / 'Data' / 'Sensitivity_HPV_Clearance_TimeStratified_v3.csv',
        index=False, encoding='utf-8-sig',
    )
    print('Wrote Data/Sensitivity_HPV_Clearance_TimeStratified_v3.csv')

    # ----- Sens-E: recurrence with min disease-free interval (DFI) -----
    sens_e_rows = []
    base = df.copy()
    base['rec_pre_lm'] = (base['has_recurrence'] == True) & (base['recurrence_date'] < base['lm_zero'])
    bad = base[(base['접종여부'] == True) & base['rec_pre_lm']]['fine_match_id'].unique()
    base = base[~base['fine_match_id'].isin(bad)].copy()
    base = base[~base['rec_pre_lm']].copy()
    base['t'] = np.where(
        base['has_recurrence'] == True,
        (base['recurrence_date'] - base['lm_zero']).dt.days,
        (base['최종추적일자'] - base['lm_zero']).dt.days,
    )

    for dfi_days, label in [
        (0,  'Primary (no DFI buffer; landmark 90 d only)'),
        (90, 'DFI ≥3 months from landmark'),
        (180, 'DFI ≥6 months from landmark'),
        (365, 'DFI ≥12 months from landmark'),
    ]:
        sub = base.copy()
        # Drop events occurring before DFI
        early_event = (sub['has_recurrence'] == True) & (sub['t'] <= dfi_days)
        bad_sets = sub[(sub['접종여부'] == True) & early_event]['fine_match_id'].unique()
        sub = sub[~sub['fine_match_id'].isin(bad_sets)].copy()
        sub = sub[~early_event.loc[sub.index]].copy()
        sub['t_dfi'] = sub['t'] - dfi_days
        sub = sub[sub['t_dfi'] > 0].copy()
        n_v, n_c, ev_v, ev_c, hr, lo, hi, p = cox_hr(sub, 't_dfi', 'has_recurrence')
        sens_e_rows.append(
            {
                'definition': label,
                'n_v': n_v, 'n_c': n_c, 'ev_v': ev_v, 'ev_c': ev_c,
                'HR': round(hr, 3) if not np.isnan(hr) else 'NA',
                'CIlo': round(lo, 3) if not np.isnan(lo) else 'NA',
                'CIhi': round(hi, 3) if not np.isnan(hi) else 'NA',
                'p': round(p, 4) if not np.isnan(p) else 'NA',
            }
        )
    pd.DataFrame(sens_e_rows).to_csv(
        ROOT / 'Data' / 'Sensitivity_Recurrence_DFInterval_v3.csv',
        index=False, encoding='utf-8-sig',
    )
    print('Wrote Data/Sensitivity_Recurrence_DFInterval_v3.csv')

    print('\n=== Summary ===')
    print('Sens-A (single neg vs 2-cons-neg, v3):')
    print(pd.DataFrame(sens_a_rows).to_string(index=False))
    print('\nSens-B (time-stratified clearance, v3):')
    print(pd.DataFrame(sens_b_rows).to_string(index=False))
    print('\nSens-E (recurrence DFI, v3):')
    print(pd.DataFrame(sens_e_rows).to_string(index=False))


if __name__ == '__main__':
    main()
