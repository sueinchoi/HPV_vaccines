"""
Refined HPV outcome: NEW high-risk type acquisition.

Definition: among Cohort B women with at least one pre-surgery molecular
pathology record, the event is the first post-index pathology record whose
detected hr-HPV type set contains AT LEAST one type not present in the
patient's pre-surgery type set. This excludes pure persistence/reactivation
of pre-existing types and operationalises "new acquisition".

Notes on type resolution:
- Pre-surgery "Positive(Other)" reports assign all 12 non-16/18 types; a
  patient with such a baseline therefore can only register a novel event
  if a post-index type 16 or 18 appears, which is conservative (the
  unknown specific "Other" type may genuinely be new but cannot be
  distinguished).
- Pre-surgery "Negative": any post-index hr-HPV type counts as new.
- Pool labels (P1=33+58, P2=56+59+66, P3=35+39+68) and DNA-chip pooled
  reporting are honoured by the existing detect_high_risk_hpv function.

Patients without a pre-surgery molecular pathology record are excluded
(baseline cannot be defined). Matched-set integrity is preserved by also
dropping non-vaccinated controls whose matched vaccinated case was
dropped.

Output: Data/Sensitivity_HPV_NovelType.csv
"""
import warnings; warnings.filterwarnings('ignore')
import sys, pandas as pd, numpy as np
sys.path.insert(0, 'scripts')
from extract_pathology_outcomes import detect_high_risk_hpv
from lifelines import CoxPHFitter

PATH_FILE = 'Data/한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV'

# ---------- Load ----------
print('Loading raw pathology file...')
patho = pd.read_csv(PATH_FILE, encoding='cp949', low_memory=False)
hpv = patho[patho['병리검사구분'].isin(['분자병리','HPV'])].copy()
hpv['실시일자_dt'] = pd.to_datetime(
    hpv['실시일자'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')

B = pd.read_csv('Data/final_matched_cohort.csv', encoding='utf-8-sig')
B['첫수술일자']  = pd.to_datetime(B['첫수술일자'])
B['index_date'] = pd.to_datetime(B['index_date'])
B['최종추적일자'] = pd.to_datetime(B['최종추적일자'])
B['vac']        = B['접종여부'].astype(bool).astype(int)
B['index_age']  = pd.to_numeric(B['index_age'], errors='coerce')
B['follow_up_days'] = (B['최종추적일자'] - B['index_date']).dt.days

print(f'Cohort B size: {len(B):,}')

# ---------- Pre-surgery type set per patient ----------
print('\nBuilding pre-surgery type set per patient...')
hpv_b = hpv[hpv['연구번호'].isin(B['연구번호'])].copy()
hpv_b = hpv_b.merge(B[['연구번호','첫수술일자','index_date']], on='연구번호')
pre = hpv_b[hpv_b['실시일자_dt'] < hpv_b['첫수술일자']].copy()

# Pool ALL pre-surgery records (not only the last) — most permissive baseline
def union_types(series):
    s = set()
    for r in series:
        s.update(r['detected_hpv_types'] if isinstance(r['detected_hpv_types'], list) else [])
    s.discard('high_risk_unspecified')   # symbolic markers we won't use here
    s.discard('unspecified')
    return s

pre['detect'] = pre['판독결과'].apply(detect_high_risk_hpv)
pre_types = pre.groupby('연구번호')['detect'].apply(
    lambda series: set().union(
        *(set(t for t in r['detected_hpv_types'] if isinstance(t, int))
          for r in series))).rename('pre_types')

print(f'  Patients with any pre-surgery HPV test: {len(pre_types):,}')

# Restrict Cohort B to patients with pre-test, drop matched-set partners
B = B.merge(pre_types, on='연구번호', how='left')
keep_mask = B['pre_types'].notna()
print(f'  Cohort B with pre-test:    {keep_mask.sum():,} of {len(B):,}')

# Preserve matched-set structure: drop a fine_match_id only when the
# vaccinated case lacks a pre-test (controls of vac cases without pre-tests
# are also dropped; controls without pre-tests but with a vac case that has
# one are kept since they belong to a valid cluster)
vac_with_pre = set(B.loc[(B['vac']==1) & B['pre_types'].notna(), 'fine_match_id'])
B_pre = B[B['fine_match_id'].isin(vac_with_pre)].copy()
print(f'  After matched-set filter: {len(B_pre):,} (vac '
      f'{int((B_pre["vac"]==1).sum())} / non-vac {int((B_pre["vac"]==0).sum())})')

# For non-vaccinated participants without pre_types data, treat baseline as
# unknown -> exclude them from the analytic dataset (matched-set integrity
# preserved by the cluster-robust SE on remaining members).
B_pre = B_pre[B_pre['pre_types'].notna()].copy()
print(f'  After excluding non-vac without pre-test: {len(B_pre):,} '
      f'(vac {int((B_pre["vac"]==1).sum())} / non-vac {int((B_pre["vac"]==0).sum())})')

# ---------- For each patient, find FIRST post-index record with a NOVEL type ----------
print('\nSearching for first post-index novel-type detection per patient...')
hpv_b = hpv_b.merge(pre_types, on='연구번호', how='left')
post = hpv_b[hpv_b['실시일자_dt'] > hpv_b['index_date']].copy()
post = post[post['연구번호'].isin(B_pre['연구번호'])]

post['detect'] = post['판독결과'].apply(detect_high_risk_hpv)
post['post_types'] = post['detect'].apply(
    lambda r: set(t for t in r['detected_hpv_types'] if isinstance(t, int)))
post['novel'] = post.apply(
    lambda r: bool(r['post_types'] - (r['pre_types'] if isinstance(r['pre_types'], set) else set())),
    axis=1)

# First novel-type record per patient
post = post.sort_values(['연구번호','실시일자_dt'])
first_novel = (post[post['novel']].groupby('연구번호')
               .agg(novel_date=('실시일자_dt','first'),
                    novel_types=('post_types','first'))
               .reset_index())

B_pre = B_pre.merge(first_novel, on='연구번호', how='left')
B_pre['has_novel'] = B_pre['novel_date'].notna()
B_pre['days_to_novel'] = (B_pre['novel_date'] - B_pre['index_date']).dt.days
B_pre['time'] = np.where(B_pre['has_novel'],
                          B_pre['days_to_novel'], B_pre['follow_up_days'])
B_pre['event'] = B_pre['has_novel'].astype(int)
B_pre = B_pre[B_pre['time'] > 0]

# ---------- Descriptive ----------
print(f'\nNovel-type events: {int(B_pre["event"].sum()):,} of {len(B_pre):,}')
print(f'  Vaccinated:     {int(((B_pre["vac"]==1) & (B_pre["event"]==1)).sum())} / '
      f'{int((B_pre["vac"]==1).sum())}')
print(f'  Non-vaccinated: {int(((B_pre["vac"]==0) & (B_pre["event"]==1)).sum())} / '
      f'{int((B_pre["vac"]==0).sum())}')

# ---------- Cox HR ----------
def cox_hr(df, label):
    d = df[['time','event','vac','index_age','fine_match_id']].dropna()
    n_v = int((d['vac']==1).sum()); n_c = int((d['vac']==0).sum())
    e_v = int(((d['vac']==1) & (d['event']==1)).sum())
    e_c = int(((d['vac']==0) & (d['event']==1)).sum())
    res = dict(definition=label, n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if d['event'].sum() < 5 or n_v < 2 or n_c < 2:
        return res
    try:
        cph = CoxPHFitter().fit(d, duration_col='time', event_col='event',
                                cluster_col='fine_match_id', robust=True)
        r = cph.summary.loc['vac']
        res.update(HR=float(r['exp(coef)']),
                   CIlo=float(r['exp(coef) lower 95%']),
                   CIhi=float(r['exp(coef) upper 95%']),
                   p=float(r['p']))
    except Exception as e:
        print(f'  fit failed: {e}')
    return res


print('\n===== Novel-type acquisition outcome =====\n')
results = [cox_hr(B_pre, 'Novel hr-HPV type (any post-index type not in pre-surgery set)')]

# Landmark variants — robustness against incomplete clearance
for lm_d, lab in [(180, 'Novel + 6-month landmark'),
                  (365, 'Novel + 12-month landmark')]:
    sub = B_pre.copy()
    eligible = (sub['follow_up_days'] >= lm_d) & (
        sub['days_to_novel'].isna() | (sub['days_to_novel'] >= lm_d))
    sub = sub[eligible].copy()
    has_event = sub['has_novel'] & (sub['days_to_novel'] >= lm_d)
    sub['time']  = np.where(has_event, sub['days_to_novel'] - lm_d,
                            sub['follow_up_days'] - lm_d)
    sub['event'] = has_event.astype(int)
    sub = sub[sub['time'] > 0]
    results.append(cox_hr(sub, lab))

print(f'  {"Definition":68s}  n_v   n_c   ev_v  ev_c   HR (95% CI)         p')
for r in results:
    if not np.isnan(r['HR']):
        print(f"  {r['definition']:68s}  {r['n_v']:>3}  {r['n_c']:>3}   "
              f"{r['ev_v']:>3}  {r['ev_c']:>4}   "
              f"{r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  {r['p']:.3f}")
    else:
        print(f"  {r['definition']:68s}  insufficient events")

out = pd.DataFrame(results)[
    ['definition','n_v','n_c','ev_v','ev_c','HR','CIlo','CIhi','p']]
out.to_csv('Data/Sensitivity_HPV_NovelType.csv',
           index=False, encoding='utf-8-sig')
print('\nSaved: Data/Sensitivity_HPV_NovelType.csv')
