"""
Generate PH check Schoenfeld plot for the hr-HPV clearance co-primary
outcome on the n = 292 pre-vaccine hr-HPV-positive subset using the
two-consecutive-negative event definition.

Output: Data/PH_check_B_clearance.png (matches the file referenced in
docs/Submission_File_Manifest.md SupFigS6b and in docs/Manuscript_Draft.md
Supplementary Figure S5(b)).
"""
import warnings; warnings.filterwarnings('ignore')
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

sys.path.insert(0, 'scripts')
from extract_pathology_outcomes import detect_high_risk_hpv

PATH_FILE = 'Data/한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV'

print('Loading...')
patho = pd.read_csv(PATH_FILE, encoding='cp949', low_memory=False)
hpv = patho[patho['병리검사구분'].isin(['분자병리', 'HPV'])].copy()
hpv['실시일자_dt'] = pd.to_datetime(
    hpv['실시일자'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')

B = pd.read_csv('Data/final_matched_cohort.csv', encoding='utf-8-sig')
B['index_date'] = pd.to_datetime(B['index_date'])
B['최종추적일자'] = pd.to_datetime(B['최종추적일자'])
B['vac'] = B['접종여부'].astype(bool).astype(int)
B['index_age'] = pd.to_numeric(B['index_age'], errors='coerce')

# ---- Identify pre-vaccine hr-HPV+ baseline + first two-consecutive-negative
hpv_b = hpv[hpv['연구번호'].isin(B['연구번호'])].copy()
hpv_b = hpv_b.merge(B[['연구번호', 'index_date']], on='연구번호')

def is_pos(s):
    r = detect_high_risk_hpv(s) if pd.notna(s) else None
    if isinstance(r, dict):
        return r.get('is_high_risk_hpv_positive', False)
    return bool(r)

# pre-vaccine hr-HPV+ baseline
pre = hpv_b[hpv_b['실시일자_dt'] < hpv_b['index_date']].copy()
pre['is_hr_pos'] = pre['판독결과'].apply(is_pos)
pre_pos = pre[pre['is_hr_pos']]['연구번호'].unique()

# post-index records, sorted by date per patient
post = hpv_b[hpv_b['실시일자_dt'] >= hpv_b['index_date']].copy()
post['is_hr_pos'] = post['판독결과'].apply(is_pos)
post = post.sort_values(['연구번호', '실시일자_dt'])

# first two consecutive negatives
def first_two_consec_neg(g):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        if (not g.loc[i, 'is_hr_pos']) and (not g.loc[i + 1, 'is_hr_pos']):
            return g.loc[i, '실시일자_dt']
    return pd.NaT

first_neg = post.groupby('연구번호').apply(first_two_consec_neg).rename('first_neg_date').reset_index()

# Build analytic cohort: pre-vaccine HPV+ subset
C = B[B['연구번호'].isin(pre_pos)].merge(first_neg, on='연구번호', how='left')

# matched-set integrity: drop fine_match_ids whose vaccinated case is not in pre_pos
vac_match_ids = C[C['vac'] == 1]['fine_match_id'].unique()
C = C[C['fine_match_id'].isin(vac_match_ids)]
# drop unvaccinated rows that are themselves not in pre_pos
C = C[(C['vac'] == 1) | (C['연구번호'].isin(pre_pos))]

C['event'] = C['first_neg_date'].notna().astype(int)
C['time'] = np.where(
    C['event'] == 1,
    (C['first_neg_date'] - C['index_date']).dt.days,
    (C['최종추적일자'] - C['index_date']).dt.days
).astype(float)
C = C[C['time'] > 0].dropna(subset=['index_age'])

print(f'Clearance analytic cohort: n={len(C)} '
      f'(vac={int((C["vac"]==1).sum())}, ctl={int((C["vac"]==0).sum())}, '
      f'events={int(C["event"].sum())})')

# ---- Fit Cox + PH check
d = C[['time', 'event', 'vac', 'index_age']].rename(columns={'index_age': 'age'})
cph = CoxPHFitter()
cph.fit(d, duration_col='time', event_col='event', show_progress=False)
print(cph.summary[['coef', 'exp(coef)', 'p']])

# generate diagnostic plots
plt.close('all')
cph.check_assumptions(d, p_value_threshold=0.05, show_plots=True)
fig = plt.gcf()
fig.suptitle('PH check: Cohort B — hr-HPV clearance (n = 292, two-consecutive-negative event)')
fig.tight_layout()
out_path = 'Data/PH_check_B_clearance.png'
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f'Saved: {out_path}')
