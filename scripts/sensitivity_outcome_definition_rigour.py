"""
Two outcome-definition sensitivity analyses to address standard epidemiological
critiques of the primary definitions.

S18. Two-consecutive-negative HPV clearance
  Standard epidemiological definitions of HPV clearance (Bouvard IARC 2009,
  Insinga 2010) require two consecutive negative tests rather than a single
  negative, because a single negative may reflect imperfect assay sensitivity
  or transient viral-load fluctuation. We re-define the clearance event as the
  date of the FIRST of two consecutive post-index molecular pathology records
  explicitly negative for hr-HPV; patients with a single negative followed by
  a positive (or no further test) are NOT considered to have cleared.

S19. ≥6-month disease-free interval for lesion recurrence
  Post-treatment HPV vaccine meta-analyses (Lichter 2020, Petras 2023) require
  a minimum 6-month disease-free interval before counting any subsequent
  histological abnormality as a "recurrence" rather than persistent disease.
  Because the index date in this cohort is set at vaccination (or pseudo-
  vaccine date) which already lies a median 125 days post-surgery, an
  additional 6-month minimum from the index date provides a cumulative
  ~10-month disease-free buffer before counting an event.

Outputs:
  Data/Sensitivity_HPV_Clearance_TwoNegative.csv
  Data/Sensitivity_Recurrence_DFInterval.csv
"""
import warnings; warnings.filterwarnings('ignore')
import sys, pandas as pd, numpy as np
sys.path.insert(0, 'scripts')
from extract_pathology_outcomes import detect_high_risk_hpv
from lifelines import CoxPHFitter

PATH_FILE = 'Data/한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV'

# ---------- Load ----------
print('Loading...')
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
B['follow_up_days']     = (B['최종추적일자'] - B['index_date']).dt.days
B['days_to_recurrence'] = pd.to_numeric(B['days_to_recurrence'], errors='coerce')

# Pre-vaccine baseline + post-index records
hpv_b = hpv[hpv['연구번호'].isin(B['연구번호'])].merge(
    B[['연구번호','index_date']], on='연구번호')
pre = hpv_b[hpv_b['실시일자_dt'] < hpv_b['index_date']].copy()
pre['detect'] = pre['판독결과'].apply(detect_high_risk_hpv)
pre_summary = pre.groupby('연구번호').apply(lambda g: pd.Series({
    'pre_pos_any': any(r['is_high_risk_hpv_positive'] for r in g['detect']),
})).reset_index()
B = B.merge(pre_summary, on='연구번호', how='left')

post = hpv_b[hpv_b['실시일자_dt'] > hpv_b['index_date']].copy()
post['detect']   = post['판독결과'].apply(detect_high_risk_hpv)
post['post_pos'] = post['detect'].apply(lambda r: r['is_high_risk_hpv_positive'])
post = post.sort_values(['연구번호','실시일자_dt'])


def restrict(df, vac_cond):
    keep_ids = set(df.loc[(df['vac']==1) & vac_cond(df), 'fine_match_id'])
    sub = df[df['fine_match_id'].isin(keep_ids)].copy()
    sub = sub[(sub['vac']==1) | vac_cond(sub)]
    return sub


def cox_hr(d, time_col, event_col, label):
    df = d[[time_col, event_col, 'vac', 'index_age', 'fine_match_id']].dropna().rename(
        columns={time_col:'time', event_col:'event'})
    df['event'] = df['event'].astype(int)
    df = df[df['time'] > 0]
    n_v = int((df['vac']==1).sum()); n_c = int((df['vac']==0).sum())
    e_v = int(((df['vac']==1)&(df['event']==1)).sum())
    e_c = int(((df['vac']==0)&(df['event']==1)).sum())
    res = dict(definition=label, n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if df['event'].sum() < 4 or n_v < 2 or n_c < 2:
        return res
    try:
        cph = CoxPHFitter().fit(df, duration_col='time', event_col='event',
                                cluster_col='fine_match_id', robust=True)
        r = cph.summary.loc['vac']
        res.update(HR=float(r['exp(coef)']),
                   CIlo=float(r['exp(coef) lower 95%']),
                   CIhi=float(r['exp(coef) upper 95%']),
                   p=float(r['p']))
    except Exception as e:
        print(f'  {label}: fit failed ({e})')
    return res

# =====================================================================
# S18. Two-consecutive-negative HPV clearance
# =====================================================================
print('\n[S18] Two-consecutive-negative HPV clearance')
sub_clear_full = restrict(B, lambda d: d['pre_pos_any'] == True)

def first_two_consecutive_negatives(pid_records):
    """Return the date of the FIRST of two consecutive negative records, or None."""
    records = pid_records.sort_values('실시일자_dt').reset_index(drop=True)
    for i in range(len(records) - 1):
        if (not records.loc[i, 'post_pos']) and (not records.loc[i+1, 'post_pos']):
            return records.loc[i, '실시일자_dt']
    return None

print('  Computing two-consecutive-negative dates per patient...')
two_neg = post.groupby('연구번호').apply(first_two_consecutive_negatives)
two_neg = two_neg.dropna().rename('first_two_neg_date').reset_index()
sub_clear_full = sub_clear_full.merge(two_neg, on='연구번호', how='left')

sub_clear_full['days_to_2neg']  = (sub_clear_full['first_two_neg_date']
                                    - sub_clear_full['index_date']).dt.days
sub_clear_full['clear2_event']  = sub_clear_full['first_two_neg_date'].notna().astype(int)
sub_clear_full['clear2_time']   = np.where(sub_clear_full['clear2_event']==1,
                                            sub_clear_full['days_to_2neg'],
                                            sub_clear_full['follow_up_days'])

# Compare with single-negative result
single_neg = post[~post['post_pos']]
first_neg = single_neg.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
first_neg.columns = ['연구번호','first_neg_date']
sub_clear_full = sub_clear_full.merge(first_neg, on='연구번호', how='left')
sub_clear_full['days_to_1neg']  = (sub_clear_full['first_neg_date']
                                    - sub_clear_full['index_date']).dt.days
sub_clear_full['clear1_event']  = sub_clear_full['first_neg_date'].notna().astype(int)
sub_clear_full['clear1_time']   = np.where(sub_clear_full['clear1_event']==1,
                                            sub_clear_full['days_to_1neg'],
                                            sub_clear_full['follow_up_days'])

# Cox fits
rows = []
rows.append(cox_hr(sub_clear_full, 'clear1_time', 'clear1_event',
                    'Single-negative (primary)'))
rows.append(cox_hr(sub_clear_full, 'clear2_time', 'clear2_event',
                    'Two-consecutive-negatives (S18)'))

s18 = pd.DataFrame(rows)[['definition','n_v','n_c','ev_v','ev_c','HR','CIlo','CIhi','p']]
s18.to_csv('Data/Sensitivity_HPV_Clearance_TwoNegative.csv',
           index=False, encoding='utf-8-sig')
print(f'  {"Definition":40s}  n_v   n_c   ev_v  ev_c   HR (95% CI)         p')
for _, r in s18.iterrows():
    if not pd.isna(r['HR']):
        print(f"  {r['definition']:40s}  {int(r['n_v']):>3}   {int(r['n_c']):>3}   "
              f"{int(r['ev_v']):>3}  {int(r['ev_c']):>4}   "
              f"{r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  {r['p']:.3f}")
print('Saved: Data/Sensitivity_HPV_Clearance_TwoNegative.csv')

# =====================================================================
# S19. ≥6-month disease-free interval for lesion recurrence
# =====================================================================
print('\n[S19] ≥6-month disease-free interval for lesion recurrence')
B = B.copy()  # full cohort
B['rec_event'] = B['has_recurrence'].astype(int)
B['rec_time']  = np.where(B['rec_event']==1, B['days_to_recurrence'], B['follow_up_days'])

rows = []
# Primary (no minimum)
rows.append(cox_hr(B, 'rec_time', 'rec_event',
                    'Primary (no minimum disease-free interval)'))

for min_days, lab in [(90,'≥3 months'), (180,'≥6 months (S19)'), (365,'≥12 months')]:
    d = B.copy()
    # Patients with event before min_days are excluded (likely persistent
    # disease); patients censored before min_days are also excluded.
    eligible = (d['rec_time'] >= min_days)
    d = d[eligible].copy()
    has_event_after = (d['rec_event']==1) & (d['rec_time'] >= min_days)
    d['rec_event_dfi'] = has_event_after.astype(int)
    d['rec_time_dfi']  = d['rec_time'] - min_days
    rows.append(cox_hr(d, 'rec_time_dfi', 'rec_event_dfi',
                        f'Minimum disease-free interval {lab}'))

s19 = pd.DataFrame(rows)[['definition','n_v','n_c','ev_v','ev_c','HR','CIlo','CIhi','p']]
s19.to_csv('Data/Sensitivity_Recurrence_DFInterval.csv',
           index=False, encoding='utf-8-sig')
print(f'  {"Definition":52s}  n_v   n_c   ev_v  ev_c   HR (95% CI)         p')
for _, r in s19.iterrows():
    if not pd.isna(r['HR']):
        print(f"  {r['definition']:52s}  {int(r['n_v']):>3}   {int(r['n_c']):>3}   "
              f"{int(r['ev_v']):>3}  {int(r['ev_c']):>4}   "
              f"{r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  {r['p']:.3f}")
    else:
        print(f"  {r['definition']:52s}  insufficient events")
print('Saved: Data/Sensitivity_Recurrence_DFInterval.csv')
