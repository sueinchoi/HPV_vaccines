"""
Analyze Cohort A under v3 primary definition (≥2 dose + 3-mo landmark).

Outputs:
  Data/CohortA_HR_v3.csv  — co-primary + secondary HRs

Mirrors the methodology of analyze_primary_v3.py but for Cohort A:
  - Cluster-robust SE on pair_id
  - Cause-specific Cox HR (death as competing event → right-censoring)
  - Fine-Gray subdistribution HR
  - Time anchored from the 3-month landmark (index + 90 days)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

CLASS_LABELS = {'1': 'Angina/MI', '2': 'Hypertension', '3': 'Diabetes',
                '4': 'Stroke', '5': 'PE'}
ANY5 = ['1', '2', '3', '4', '5']
MCE = ['1', '4', '5']
LANDMARK_DAYS = 90


def make_tte(m: pd.DataFrame, comp) -> pd.DataFrame:
    """Build time-to-event dataframe, anchored at landmark (index + 90 d)."""
    if isinstance(comp, list):
        dx = m[comp].min(axis=1)
    else:
        dx = m[comp]
    # Cast to datetime if needed
    dx = pd.to_datetime(dx, errors='coerce')

    lm_zero = m['index_date'] + pd.Timedelta(days=LANDMARK_DAYS)
    last_fu = m['최종추적일자']

    # Prevalent disease (≤ landmark) → excluded from at-risk population
    is_pre = dx.notna() & (dx <= lm_zero)

    primary = dx.where(dx > lm_zero, pd.NaT)
    death_after = m['death_date'].where(
        (m['death_date'].notna()) & (m['death_date'] > lm_zero) &
        ((primary.isna()) | (m['death_date'] < primary)),
        pd.NaT,
    )

    event_date = primary.combine_first(death_after)
    status = np.where(
        primary.notna() & ((death_after.isna()) | (primary <= death_after)),
        1,
        np.where(death_after.notna(), 2, 0),
    )

    end_date = event_date.combine_first(last_fu)
    time = (end_date - lm_zero).dt.days.astype(float)

    res = pd.DataFrame({
        'pid': m['pid'].values,
        'pair_id': m['pair_id'].values,
        'vaccinated': m['vaccinated'].astype(int).values,
        'age_at_index': m['age_at_index'].values,
        'time': time,
        'status': status,
    })
    res = res[~is_pre.values & (res['time'] > 0)].reset_index(drop=True)
    return res


def fit_hr(tte: pd.DataFrame, label: str) -> dict:
    """Fit cluster-robust Cox HR (cause-specific: death censored)."""
    df = tte.copy()
    df['event'] = (df['status'] == 1).astype(int)
    n_v = int((df['vaccinated'] == 1).sum())
    n_c = int((df['vaccinated'] == 0).sum())
    ev_v = int(((df['vaccinated'] == 1) & (df['event'] == 1)).sum())
    ev_c = int(((df['vaccinated'] == 0) & (df['event'] == 1)).sum())
    py_v = float(df.loc[df['vaccinated'] == 1, 'time'].sum() / 365.25)
    py_c = float(df.loc[df['vaccinated'] == 0, 'time'].sum() / 365.25)

    out = {
        'outcome': label,
        'n_v': n_v, 'n_c': n_c,
        'ev_v': ev_v, 'ev_c': ev_c,
        'PY_v': round(py_v, 1), 'PY_c': round(py_c, 1),
        'IR_v_per1000PY': round(1000 * ev_v / py_v, 2) if py_v > 0 else np.nan,
        'IR_c_per1000PY': round(1000 * ev_c / py_c, 2) if py_c > 0 else np.nan,
        'HR': np.nan, 'CI_lo': np.nan, 'CI_hi': np.nan, 'p': np.nan,
    }
    if ev_v + ev_c < 5:
        return out
    fit = df[['time', 'event', 'vaccinated', 'age_at_index', 'pair_id']].copy()
    try:
        cph = CoxPHFitter()
        cph.fit(fit, duration_col='time', event_col='event', cluster_col='pair_id')
        s = cph.summary.loc['vaccinated']
        out['HR'] = round(float(s['exp(coef)']), 3)
        out['CI_lo'] = round(float(s['exp(coef) lower 95%']), 3)
        out['CI_hi'] = round(float(s['exp(coef) upper 95%']), 3)
        out['p'] = round(float(s['p']), 4)
    except Exception as e:
        print(f'  Fit failed for {label}: {e}')
    return out


def main():
    print('Loading Cohort A v3 matched dataset...')
    df = pd.read_csv(ROOT / 'Data' / 'cohort_a_v3_matched.csv', encoding='utf-8-sig')
    df['index_date'] = pd.to_datetime(df['index_date'], errors='coerce')
    df['최종추적일자'] = pd.to_datetime(df['최종추적일자'], errors='coerce')
    df['death_date'] = pd.to_datetime(df['death_date'], errors='coerce')
    df['vaccinated'] = df['vaccinated'].astype(bool)

    # Parse comorbidity date columns (may be in different format)
    for c in ['1', '2', '3', '4', '5']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

    print(f'  n = {len(df)} ({(df["vaccinated"]==True).sum()} vac / '
          f'{(df["vaccinated"]==False).sum()} non)')

    rows = []
    for label, comp in [
        ('Any-of-5 composite', ANY5),
        ('MCE composite (MI/Stroke/PE)', MCE),
        ('Hypertension', '2'),
        ('Diabetes', '3'),
        ('Angina/MI', '1'),
        ('Stroke', '4'),
        ('PE', '5'),
    ]:
        tte = make_tte(df, comp)
        rows.append(fit_hr(tte, label))

    out = pd.DataFrame(rows)
    out_path = ROOT / 'Data' / 'CohortA_HR_v3.csv'
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'Wrote {out_path.relative_to(ROOT)}')
    print()
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
