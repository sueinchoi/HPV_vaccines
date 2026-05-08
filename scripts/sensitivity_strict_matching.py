"""
Sensitivity analysis: strict (fixed-ratio) Cohort B matching vs the primary
variable-ratio (1:up-to-4) approach.

Strict 1:4 retains only those vaccinated cases that received the full 4
controls in the original fine-matching step; cases with 1–3 controls and
their attached non-vaccinated participants are dropped. Cox HRs (age-
adjusted, cluster-robust on fine_match_id) are re-estimated under the
strict definition for both primary outcomes.

Output: Data/Sensitivity_StrictMatching.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from lifelines import CoxPHFitter

B = pd.read_csv('Data/final_matched_outcomes.csv', encoding='utf-8-sig')
B['vac'] = B['접종여부'].astype(bool).astype(int)
B['follow_up_days'] = pd.to_numeric(B['follow_up_days'], errors='coerce')
B['index_age'] = pd.to_numeric(B['index_age'], errors='coerce')

n_ctl_per_match = (B.groupby('fine_match_id')['vac']
                    .apply(lambda s: int((s == 0).sum())))
full4_ids = n_ctl_per_match[n_ctl_per_match == 4].index
B_strict = B[B['fine_match_id'].isin(full4_ids)].copy()


def cox_hr(d, ev_col):
    df = d[['follow_up_days', ev_col, 'vac', 'index_age', 'fine_match_id']].dropna().rename(
        columns={'follow_up_days': 'time', ev_col: 'event'})
    df['event'] = df['event'].astype(int)
    n_v = int((df['vac'] == 1).sum()); n_c = int((df['vac'] == 0).sum())
    e_v = int(((df['vac'] == 1) & (df['event'] == 1)).sum())
    e_c = int(((df['vac'] == 0) & (df['event'] == 1)).sum())
    res = dict(n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if df['event'].sum() < 3:
        return res
    cph = CoxPHFitter().fit(df, duration_col='time', event_col='event',
                            cluster_col='fine_match_id', robust=True)
    r = cph.summary.loc['vac']
    res.update(HR=float(r['exp(coef)']),
               CIlo=float(r['exp(coef) lower 95%']),
               CIhi=float(r['exp(coef) upper 95%']),
               p=float(r['p']))
    return res


def detectable_hr_80(events_total):
    if events_total < 4:
        return np.nan
    z = 1.96 + 0.84
    return float(np.exp(z * 2 / np.sqrt(events_total)))


rows = []
for ev_label, ev_col in [('Lesion recurrence', 'has_recurrence'),
                          ('HPV reinfection',  'has_hpv_infection')]:
    for design, sub in [('variable-ratio (1:up-to-4, primary)', B),
                         ('strict 1:4 (sensitivity)', B_strict)]:
        r = cox_hr(sub, ev_col)
        e_total = r['ev_v'] + r['ev_c']
        r.update(outcome=ev_label, design=design,
                 detectable_HR_80pct_power=detectable_hr_80(e_total))
        rows.append(r)

out = pd.DataFrame(rows)[
    ['outcome', 'design', 'n_v', 'n_c', 'ev_v', 'ev_c',
     'HR', 'CIlo', 'CIhi', 'p', 'detectable_HR_80pct_power']]
out.to_csv('Data/Sensitivity_StrictMatching.csv',
           index=False, encoding='utf-8-sig')
print(out.to_string(index=False))
print('\nSaved: Data/Sensitivity_StrictMatching.csv')
