"""
HPV regression / clearance outcome.

Among Cohort B women with documented pre-surgery hr-HPV positivity,
the event is the first post-index molecular pathology record explicitly
reported as HPV-NEGATIVE (i.e., the post-index regression of the
pre-surgical infection). Higher HR = faster clearance in the vaccinated
arm = favourable vaccine effect.

Three definitions:
  C1. Any HPV-negative test post-index ("any-clearance")
  C2. HPV-16 specific clearance (pre-surgery 16+ -> post-index 16-)
  C3. HPV-18 specific clearance (pre-surgery 18+ -> post-index 18-)

Matched-set integrity: drop fine_match_ids whose vaccinated case
lacks a pre-surgery hr-HPV+ test, and drop non-vaccinated controls
who lack the corresponding pre-surgery status.

Output: Data/Sensitivity_HPV_Clearance.csv
"""
import warnings; warnings.filterwarnings('ignore')
import sys, pandas as pd, numpy as np
sys.path.insert(0, 'scripts')
from extract_pathology_outcomes import detect_high_risk_hpv
from lifelines import CoxPHFitter, KaplanMeierFitter

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

# ---------- Pre-surgery type set per patient ----------
print('\nBuilding pre-surgery type set per patient...')
hpv_b = hpv[hpv['연구번호'].isin(B['연구번호'])].copy()
hpv_b = hpv_b.merge(B[['연구번호','첫수술일자','index_date']], on='연구번호')
pre = hpv_b[hpv_b['실시일자_dt'] < hpv_b['첫수술일자']].copy()
pre['detect'] = pre['판독결과'].apply(detect_high_risk_hpv)

pre_summary = pre.groupby('연구번호').apply(lambda g: pd.Series({
    'pre_pos_any':  any(r['is_high_risk_hpv_positive'] for r in g['detect']),
    'pre_types':    set().union(*[set(t for t in r['detected_hpv_types']
                                      if isinstance(t, int)) for r in g['detect']]),
})).reset_index()
pre_summary['pre_16_pos'] = pre_summary['pre_types'].apply(lambda s: int(16 in s))
pre_summary['pre_18_pos'] = pre_summary['pre_types'].apply(lambda s: int(18 in s))

B = B.merge(pre_summary, on='연구번호', how='left')
print(f'  Pre-surgery hr-HPV+:  {(B["pre_pos_any"] == True).sum()} patients')
print(f'  Pre-surgery HPV-16+:  {(B["pre_16_pos"] == 1).sum()}')
print(f'  Pre-surgery HPV-18+:  {(B["pre_18_pos"] == 1).sum()}')

# ---------- Find first post-index HPV-NEGATIVE test per patient ----------
print('\nScanning post-index records for first HPV-negative result...')
post = hpv_b[hpv_b['실시일자_dt'] > hpv_b['index_date']].copy()
post['detect'] = post['판독결과'].apply(detect_high_risk_hpv)
post['post_pos']    = post['detect'].apply(lambda r: r['is_high_risk_hpv_positive'])
post['post_types']  = post['detect'].apply(
    lambda r: set(t for t in r['detected_hpv_types'] if isinstance(t, int)))
post = post.sort_values(['연구번호','실시일자_dt'])

# C1. Any HPV-negative test (overall clearance)
neg = post[~post['post_pos']]   # explicit negative or no positive types detected
first_neg = neg.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
first_neg.columns = ['연구번호','first_neg_date']
B = B.merge(first_neg, on='연구번호', how='left')

# C2. HPV-16 negative (does not contain 16 even if positive for others)
neg16 = post[~post['post_types'].apply(lambda s: 16 in s)]
first_neg16 = neg16.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
first_neg16.columns = ['연구번호','first_neg16_date']
B = B.merge(first_neg16, on='연구번호', how='left')

# C3. HPV-18 negative
neg18 = post[~post['post_types'].apply(lambda s: 18 in s)]
first_neg18 = neg18.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
first_neg18.columns = ['연구번호','first_neg18_date']
B = B.merge(first_neg18, on='연구번호', how='left')

# ---------- Cox HR helper ----------
def cox_hr(df, event_date_col, label):
    d = df.copy()
    d['has_event']  = d[event_date_col].notna()
    d['days_to_ev'] = (d[event_date_col] - d['index_date']).dt.days
    d['time']  = np.where(d['has_event'], d['days_to_ev'], d['follow_up_days'])
    d['event'] = d['has_event'].astype(int)
    d = d[d['time'] > 0].dropna(subset=['index_age','fine_match_id'])
    n_v = int((d['vac']==1).sum()); n_c = int((d['vac']==0).sum())
    e_v = int(((d['vac']==1) & (d['event']==1)).sum())
    e_c = int(((d['vac']==0) & (d['event']==1)).sum())
    res = dict(definition=label, n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if d['event'].sum() < 5 or n_v < 2 or n_c < 2:
        return res
    try:
        cph = CoxPHFitter().fit(
            d[['time','event','vac','index_age','fine_match_id']],
            duration_col='time', event_col='event',
            cluster_col='fine_match_id', robust=True)
        r = cph.summary.loc['vac']
        res.update(HR=float(r['exp(coef)']),
                   CIlo=float(r['exp(coef) lower 95%']),
                   CIhi=float(r['exp(coef) upper 95%']),
                   p=float(r['p']))
    except Exception as e:
        print(f'  fit failed for {label}: {e}')
    return res


# ---------- Build analytic samples preserving matched-set integrity ----------
def restrict(B_full, vac_cond):
    """Drop fine_match_ids whose vaccinated case fails vac_cond, then drop
    non-vac members who fail the same cond."""
    keep_ids = set(B_full.loc[(B_full['vac']==1) & vac_cond(B_full), 'fine_match_id'])
    sub = B_full[B_full['fine_match_id'].isin(keep_ids)].copy()
    sub = sub[vac_cond(sub) | (sub['vac']==1)]
    # for non-vac, also require they meet the same baseline criterion
    sub = sub[ (sub['vac']==1) | vac_cond(sub) ]
    return sub

print('\n===== HPV clearance / regression outcome =====\n')
results = []

# C1: any-clearance (pre-surgery hr-HPV+ baseline)
sub1 = restrict(B, lambda d: d['pre_pos_any'] == True)
print(f'C1 cohort: {len(sub1)} (vac {int((sub1["vac"]==1).sum())} / '
      f'non-vac {int((sub1["vac"]==0).sum())})')
results.append(cox_hr(sub1, 'first_neg_date',
                      'C1. Any-clearance (pre-surgery hr-HPV+ baseline)'))

# C2: HPV-16 clearance (pre-surgery 16+ baseline)
sub2 = restrict(B, lambda d: d['pre_16_pos'] == 1)
print(f'C2 cohort: {len(sub2)} (vac {int((sub2["vac"]==1).sum())} / '
      f'non-vac {int((sub2["vac"]==0).sum())})')
results.append(cox_hr(sub2, 'first_neg16_date',
                      'C2. HPV-16 clearance (pre-surgery 16+ baseline)'))

# C3: HPV-18 clearance
sub3 = restrict(B, lambda d: d['pre_18_pos'] == 1)
print(f'C3 cohort: {len(sub3)} (vac {int((sub3["vac"]==1).sum())} / '
      f'non-vac {int((sub3["vac"]==0).sum())})')
results.append(cox_hr(sub3, 'first_neg18_date',
                      'C3. HPV-18 clearance (pre-surgery 18+ baseline)'))

print()
print(f'  {"Definition":62s}  n_v   n_c   ev_v  ev_c   HR (95% CI)         p')
for r in results:
    if not np.isnan(r['HR']):
        print(f"  {r['definition']:62s}  {r['n_v']:>3}  {r['n_c']:>3}   "
              f"{r['ev_v']:>3}  {r['ev_c']:>4}   "
              f"{r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  {r['p']:.3f}")
    else:
        print(f"  {r['definition']:62s}  insufficient events / fit failed")

# Median time to clearance
print('\nMedian time to first HPV-negative (any-clearance) by group:')
sub1 = sub1.copy()
sub1['days_to_neg'] = (sub1['first_neg_date'] - sub1['index_date']).dt.days
for grp, lbl in [(1,'Vaccinated'), (0,'Non-vaccinated')]:
    days = sub1.loc[(sub1['vac']==grp) & sub1['days_to_neg'].notna(), 'days_to_neg']
    if len(days):
        print(f'  {lbl:18s} median {days.median():.0f} days '
              f'(IQR {days.quantile(0.25):.0f}–{days.quantile(0.75):.0f}, n={len(days)})')

out = pd.DataFrame(results)[
    ['definition','n_v','n_c','ev_v','ev_c','HR','CIlo','CIhi','p']]
out.to_csv('Data/Sensitivity_HPV_Clearance.csv',
           index=False, encoding='utf-8-sig')
print('\nSaved: Data/Sensitivity_HPV_Clearance.csv')
print('\nNote: HR > 1 = faster clearance in vaccinated arm = favourable')
