"""
Re-analyze Cohort B with HPV regression / clearance as the PRIMARY outcome,
and using each patient's PRE-VACCINE (= pre-index) molecular pathology
records as the baseline for defining hr-HPV status. The pseudo-index date
for non-vaccinated controls (surgery date + matched vaccinated partner's
surgery-to-vaccine interval) is treated as the equivalent baseline anchor
because that is the temporal point at which the vaccine effect would
begin in a counterfactual world.

Primary outcomes:
  P1. Lesion recurrence (HSIL/CIN3+)        — unchanged
  P2. hr-HPV clearance / regression          — NEW PRIMARY
       Among women with pre-index hr-HPV+, time from index to first
       post-index molecular pathology record explicitly negative for
       hr-HPV. HR > 1 = faster clearance in vaccinated (favourable).

Sensitivity outcomes (using same pre-vaccine baseline):
  S1. Novel-type acquisition (HPV+ for type not in pre-index set)
  S2. HPV-16-specific clearance (pre 16+ -> post 16-)
  S3. HPV-18-specific clearance (pre 18+ -> post 18-)

Output: Data/CohortB_PrimaryClearance_Results.csv
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
Bo = pd.read_csv('Data/final_matched_outcomes.csv', encoding='utf-8-sig')
B = B.merge(Bo[['연구번호','has_recurrence','recurrence_date','days_to_recurrence']],
            on='연구번호')
B['index_date']  = pd.to_datetime(B['index_date'])
B['최종추적일자'] = pd.to_datetime(B['최종추적일자'])
B['recurrence_date'] = pd.to_datetime(B['recurrence_date'], errors='coerce')
B['vac']        = B['접종여부'].astype(bool).astype(int)
B['index_age']  = pd.to_numeric(B['index_age'], errors='coerce')
B['follow_up_days'] = (B['최종추적일자'] - B['index_date']).dt.days

# ---------- PRE-VACCINE / PRE-INDEX baseline ----------
print('\nBuilding pre-INDEX baseline type set per patient...')
hpv_b = hpv[hpv['연구번호'].isin(B['연구번호'])].copy()
hpv_b = hpv_b.merge(B[['연구번호','index_date']], on='연구번호')
pre = hpv_b[hpv_b['실시일자_dt'] < hpv_b['index_date']].copy()
pre['detect'] = pre['판독결과'].apply(detect_high_risk_hpv)

pre_summary = pre.groupby('연구번호').apply(lambda g: pd.Series({
    'pre_pos_any':  any(r['is_high_risk_hpv_positive'] for r in g['detect']),
    'pre_types':    set().union(*[set(t for t in r['detected_hpv_types']
                                      if isinstance(t, int)) for r in g['detect']]),
    'n_pre_records': len(g),
    'last_pre_date': g['실시일자_dt'].max(),
})).reset_index()
pre_summary['pre_16_pos'] = pre_summary['pre_types'].apply(lambda s: int(16 in s))
pre_summary['pre_18_pos'] = pre_summary['pre_types'].apply(lambda s: int(18 in s))

B = B.merge(pre_summary, on='연구번호', how='left')
print(f'  Patients with any pre-INDEX HPV test:  {B["pre_pos_any"].notna().sum()}')
print(f'  Pre-index hr-HPV+:                     {(B["pre_pos_any"] == True).sum()}')
print(f'  Pre-index HPV-16+:                     {(B["pre_16_pos"] == 1).sum()}')
print(f'  Pre-index HPV-18+:                     {(B["pre_18_pos"] == 1).sum()}')

# ---------- Post-index records per patient ----------
print('\nScanning post-index pathology records...')
post = hpv_b[hpv_b['실시일자_dt'] > hpv_b['index_date']].copy()
post['detect'] = post['판독결과'].apply(detect_high_risk_hpv)
post['post_pos']    = post['detect'].apply(lambda r: r['is_high_risk_hpv_positive'])
post['post_types']  = post['detect'].apply(
    lambda r: set(t for t in r['detected_hpv_types'] if isinstance(t, int)))
post = post.sort_values(['연구번호','실시일자_dt'])

# C1: first HPV-negative test per patient (any-clearance)
neg = post[~post['post_pos']]
first_neg = neg.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
first_neg.columns = ['연구번호','first_neg_date']
B = B.merge(first_neg, on='연구번호', how='left')

# C2/C3: type-specific clearance (first record without that type)
neg16 = post[~post['post_types'].apply(lambda s: 16 in s)]
first_neg16 = neg16.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
first_neg16.columns = ['연구번호','first_neg16_date']
B = B.merge(first_neg16, on='연구번호', how='left')

neg18 = post[~post['post_types'].apply(lambda s: 18 in s)]
first_neg18 = neg18.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
first_neg18.columns = ['연구번호','first_neg18_date']
B = B.merge(first_neg18, on='연구번호', how='left')

# Novel-type: first post-index record with a type not in pre_types
def has_novel(row, post_records_by_pid):
    if row['연구번호'] not in post_records_by_pid: return None
    pre_set = row['pre_types'] if isinstance(row['pre_types'], set) else set()
    for _, r in post_records_by_pid[row['연구번호']].iterrows():
        if r['post_types'] - pre_set:
            return r['실시일자_dt']
    return None

post_by_pid = {pid: g for pid, g in post.groupby('연구번호')}
B['first_novel_date'] = B.apply(lambda r: has_novel(r, post_by_pid), axis=1)
B['first_novel_date'] = pd.to_datetime(B['first_novel_date'])

# ---------- Helper: Cox HR ----------
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


def restrict(B_full, vac_cond):
    """Drop fine_match_ids whose vaccinated case fails vac_cond, then drop
    non-vac members who fail the same cond."""
    keep_ids = set(B_full.loc[(B_full['vac']==1) & vac_cond(B_full), 'fine_match_id'])
    sub = B_full[B_full['fine_match_id'].isin(keep_ids)].copy()
    sub = sub[(sub['vac']==1) | vac_cond(sub)]
    return sub


# ---------- PRIMARY OUTCOMES ----------
print('\n===== PRIMARY OUTCOMES (Cohort B, pre-index baseline) =====\n')
results = []

# P1. Lesion recurrence (unchanged)
B_full = B.copy()
B_full['rec_event_date'] = B_full['recurrence_date']
results.append(cox_hr(B_full, 'rec_event_date',
                      'P1. Lesion recurrence (HSIL/CIN3+)  [HR<1 favourable]'))

# P2. hr-HPV clearance among pre-index hr-HPV+ women
sub_clear = restrict(B, lambda d: d['pre_pos_any'] == True)
print(f'P2 cohort (pre-INDEX hr-HPV+ baseline): {len(sub_clear)} '
      f'(vac {int((sub_clear["vac"]==1).sum())} / non-vac {int((sub_clear["vac"]==0).sum())})\n')
results.append(cox_hr(sub_clear, 'first_neg_date',
                      'P2. Any hr-HPV clearance  [HR>1 favourable]'))

# ---------- SENSITIVITY OUTCOMES ----------
# S1. Novel-type acquisition
sub_novel = restrict(B, lambda d: d['pre_pos_any'].notna())
results.append(cox_hr(sub_novel, 'first_novel_date',
                      'S1. Novel-type acquisition  [HR<1 favourable]'))

# S2/S3. Type-specific clearance
sub_16 = restrict(B, lambda d: d['pre_16_pos'] == 1)
results.append(cox_hr(sub_16, 'first_neg16_date',
                      'S2. HPV-16 clearance (pre-index 16+)'))

sub_18 = restrict(B, lambda d: d['pre_18_pos'] == 1)
results.append(cox_hr(sub_18, 'first_neg18_date',
                      'S3. HPV-18 clearance (pre-index 18+)'))

# ---------- Print + save ----------
print(f'  {"Outcome":62s}  n_v   n_c   ev_v  ev_c   HR (95% CI)         p')
for r in results:
    if not np.isnan(r['HR']):
        print(f"  {r['definition']:62s}  {r['n_v']:>3}  {r['n_c']:>3}   "
              f"{r['ev_v']:>3}  {r['ev_c']:>4}   "
              f"{r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  {r['p']:.3f}")
    else:
        print(f"  {r['definition']:62s}  insufficient events")

# Median time to clearance
sub_clear = sub_clear.copy()
sub_clear['days_to_neg'] = (sub_clear['first_neg_date'] - sub_clear['index_date']).dt.days
print('\nMedian time to first HPV-negative (P2 any-clearance):')
for grp, lbl in [(1,'Vaccinated'), (0,'Non-vaccinated')]:
    days = sub_clear.loc[(sub_clear['vac']==grp) & sub_clear['days_to_neg'].notna(),
                          'days_to_neg']
    if len(days):
        print(f'  {lbl:18s}  median {days.median():.0f} d '
              f'(IQR {days.quantile(0.25):.0f}–{days.quantile(0.75):.0f}, '
              f'n_event={len(days)})')

out = pd.DataFrame(results)[
    ['definition','n_v','n_c','ev_v','ev_c','HR','CIlo','CIhi','p']]
out.to_csv('Data/CohortB_PrimaryClearance_Results.csv',
           index=False, encoding='utf-8-sig')
print('\nSaved: Data/CohortB_PrimaryClearance_Results.csv')

# Also export the analytic dataset for downstream figure generation
sub_clear[['연구번호','vac','index_age','fine_match_id',
           'index_date','follow_up_days','first_neg_date',
           'pre_types','pre_pos_any']].to_csv(
    'Data/CohortB_Clearance_Analytic.csv',
    index=False, encoding='utf-8-sig')
print('Saved: Data/CohortB_Clearance_Analytic.csv')
