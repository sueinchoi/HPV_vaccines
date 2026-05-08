"""
Landmark sensitivity analysis for the post-index hr-HPV detection outcome.

Rationale: the operational definition of "post-index hr-HPV detection" treats
any positive hr-HPV pathology record after the index date as an event, but
because all Cohort B women were HPV-positive at the qualifying surgery, the
earliest positives more likely reflect persistence or reactivation of the
pre-surgical infection than new acquisition. To partition the two, we
re-estimate the age-adjusted Cox HR after advancing the at-risk clock to
6, 12, and 24 months post-index. Women who already had a positive record
before the landmark, and women whose follow-up ended before the landmark,
are excluded from the corresponding analysis.

Output: Data/Sensitivity_HPV_Landmark.csv
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

B = pd.read_csv('Data/final_matched_outcomes.csv', encoding='utf-8-sig')
B['vac']             = B['접종여부'].astype(bool).astype(int)
B['follow_up_days']  = pd.to_numeric(B['follow_up_days'], errors='coerce')
B['index_age']       = pd.to_numeric(B['index_age'], errors='coerce')
B['days_to_hpv']     = pd.to_numeric(B['days_to_hpv'], errors='coerce')


def landmark_cox(df, lm_days):
    """Conditional analysis: among women still HPV-negative at t = lm_days,
    model time from the landmark to first hr-HPV+ or end of follow-up."""
    d = df.copy()
    eligible = (d['follow_up_days'] >= lm_days) & (
        d['days_to_hpv'].isna() | (d['days_to_hpv'] >= lm_days))
    d = d[eligible].copy()

    has_event = d['has_hpv_infection'].astype(bool) & (d['days_to_hpv'] >= lm_days)
    d['time']  = np.where(has_event, d['days_to_hpv'] - lm_days,
                          d['follow_up_days'] - lm_days)
    d['event'] = has_event.astype(int)
    d = d[d['time'] > 0]

    n_v = int((d['vac'] == 1).sum())
    n_c = int((d['vac'] == 0).sum())
    e_v = int(((d['vac'] == 1) & (d['event'] == 1)).sum())
    e_c = int(((d['vac'] == 0) & (d['event'] == 1)).sum())
    res = dict(landmark_days=lm_days, n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if e_v + e_c < 5 or n_v < 2 or n_c < 2:
        return res
    cph = CoxPHFitter().fit(
        d[['time', 'event', 'vac', 'index_age', 'fine_match_id']],
        duration_col='time', event_col='event',
        cluster_col='fine_match_id', robust=True)
    r = cph.summary.loc['vac']
    res.update(HR=float(r['exp(coef)']),
               CIlo=float(r['exp(coef) lower 95%']),
               CIhi=float(r['exp(coef) upper 95%']),
               p=float(r['p']))
    return res


rows = []
for lm_d, lab in [(0,    'Primary (any post-index detection)'),
                  (180,  'Landmark 6 months (likely persistent)'),
                  (365,  'Landmark 12 months (clearance window)'),
                  (730,  'Landmark 24 months (true new acquisition)')]:
    r = landmark_cox(B, lm_d)
    r['definition'] = lab
    rows.append(r)
    if not np.isnan(r['HR']):
        print(f"{lab:42s}  n_v={r['n_v']:>3} n_c={r['n_c']:>3}  "
              f"events {r['ev_v']}/{r['ev_c']}  "
              f"HR={r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  p={r['p']:.3f}")
    else:
        print(f"{lab:42s}  insufficient events")

out = pd.DataFrame(rows)[['landmark_days', 'definition', 'n_v', 'n_c',
                           'ev_v', 'ev_c', 'HR', 'CIlo', 'CIhi', 'p']]
out.to_csv('Data/Sensitivity_HPV_Landmark.csv',
           index=False, encoding='utf-8-sig')
print('\nSaved: Data/Sensitivity_HPV_Landmark.csv')
