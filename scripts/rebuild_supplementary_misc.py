"""
Rebuild remaining Cohort B supplementary tables to use corrected event time
and pre-vaccine baseline (the new convention).

Tables affected:
  S3   restricted follow-up + adjusted-vs-unadjusted sensitivity
       (Data/sensitivity_analysis_both_outcomes.csv)
       — co-primary outcomes: lesion recurrence + hr-HPV clearance
  S4   age-stratified HRs for lesion recurrence × follow-up window
       (Data/CohortB_age_fu_forest.csv)
       — corrected event time (days_to_recurrence when event)
  S15  novel-type acquisition with pre-VACCINE baseline
       (Data/Sensitivity_HPV_NovelType.csv)
  S16  HPV clearance with pre-VACCINE baseline
       (Data/Sensitivity_HPV_Clearance.csv)
"""
import warnings; warnings.filterwarnings('ignore')
import sys, pandas as pd, numpy as np
sys.path.insert(0, 'scripts')
from extract_pathology_outcomes import detect_high_risk_hpv
from lifelines import CoxPHFitter

PATH_FILE = 'Data/한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV'

# ---------- Load + build pre-vaccine baseline ----------
print('Loading raw pathology...')
patho = pd.read_csv(PATH_FILE, encoding='cp949', low_memory=False)
hpv = patho[patho['병리검사구분'].isin(['분자병리','HPV'])].copy()
hpv['실시일자_dt'] = pd.to_datetime(
    hpv['실시일자'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')

B = pd.read_csv('Data/final_matched_cohort.csv', encoding='utf-8-sig')
Bo = pd.read_csv('Data/final_matched_outcomes.csv', encoding='utf-8-sig')
B = B.merge(Bo[['연구번호','has_recurrence','recurrence_date','days_to_recurrence',
                'has_hpv_infection','hpv_infection_date','days_to_hpv']],
            on='연구번호')
B['index_date']  = pd.to_datetime(B['index_date'])
B['최종추적일자'] = pd.to_datetime(B['최종추적일자'])
B['recurrence_date']    = pd.to_datetime(B['recurrence_date'], errors='coerce')
B['hpv_infection_date'] = pd.to_datetime(B['hpv_infection_date'], errors='coerce')
B['vac']            = B['접종여부'].astype(bool).astype(int)
B['index_age']      = pd.to_numeric(B['index_age'], errors='coerce')
B['follow_up_days'] = (B['최종추적일자'] - B['index_date']).dt.days
B['days_to_recurrence'] = pd.to_numeric(B['days_to_recurrence'], errors='coerce')
B['days_to_hpv']        = pd.to_numeric(B['days_to_hpv'], errors='coerce')

hpv_b = hpv[hpv['연구번호'].isin(B['연구번호'])].merge(
    B[['연구번호','index_date']], on='연구번호')
pre = hpv_b[hpv_b['실시일자_dt'] < hpv_b['index_date']].copy()
pre['detect'] = pre['판독결과'].apply(detect_high_risk_hpv)
pre_summary = pre.groupby('연구번호').apply(lambda g: pd.Series({
    'pre_pos_any': any(r['is_high_risk_hpv_positive'] for r in g['detect']),
    'pre_types':   set().union(*[set(t for t in r['detected_hpv_types']
                                     if isinstance(t, int)) for r in g['detect']]),
})).reset_index()
pre_summary['pre_16_pos'] = pre_summary['pre_types'].apply(lambda s: int(16 in s))
pre_summary['pre_18_pos'] = pre_summary['pre_types'].apply(lambda s: int(18 in s))
B = B.merge(pre_summary, on='연구번호', how='left')

# Post-index records per patient
post = hpv_b[hpv_b['실시일자_dt'] > hpv_b['index_date']].copy()
post['detect']     = post['판독결과'].apply(detect_high_risk_hpv)
post['post_pos']   = post['detect'].apply(lambda r: r['is_high_risk_hpv_positive'])
post['post_types'] = post['detect'].apply(
    lambda r: set(t for t in r['detected_hpv_types'] if isinstance(t, int)))
post = post.sort_values(['연구번호','실시일자_dt'])
post_by_pid = {pid: g for pid, g in post.groupby('연구번호')}

# First HPV-negative test per patient
neg = post[~post['post_pos']]
first_neg = neg.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
first_neg.columns = ['연구번호','first_neg_date']
B = B.merge(first_neg, on='연구번호', how='left')
neg16 = post[~post['post_types'].apply(lambda s: 16 in s)]
B = B.merge(neg16.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
            .rename(columns={'실시일자_dt':'first_neg16_date'}),
            on='연구번호', how='left')
neg18 = post[~post['post_types'].apply(lambda s: 18 in s)]
B = B.merge(neg18.groupby('연구번호').first().reset_index()[['연구번호','실시일자_dt']]
            .rename(columns={'실시일자_dt':'first_neg18_date'}),
            on='연구번호', how='left')

# Novel-type using pre-vaccine baseline
def first_novel(row):
    if row['연구번호'] not in post_by_pid: return None
    pre_set = row['pre_types'] if isinstance(row['pre_types'], set) else set()
    for _, r in post_by_pid[row['연구번호']].iterrows():
        if r['post_types'] - pre_set:
            return r['실시일자_dt']
    return None
B['first_novel_date'] = pd.to_datetime(B.apply(first_novel, axis=1))


def restrict(df, vac_cond):
    keep_ids = set(df.loc[(df['vac']==1) & vac_cond(df), 'fine_match_id'])
    sub = df[df['fine_match_id'].isin(keep_ids)].copy()
    sub = sub[(sub['vac']==1) | vac_cond(sub)]
    return sub


def cox_hr(d, time_col, event_col, age_adjust=True):
    cols = [time_col, event_col, 'vac', 'fine_match_id']
    if age_adjust: cols.insert(2, 'index_age')
    df = d[cols].dropna().rename(columns={time_col:'time', event_col:'event'})
    df['event'] = df['event'].astype(int)
    df = df[df['time'] > 0]
    n_v = int((df['vac']==1).sum()); n_c = int((df['vac']==0).sum())
    e_v = int(((df['vac']==1) & (df['event']==1)).sum())
    e_c = int(((df['vac']==0) & (df['event']==1)).sum())
    res = dict(n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if df['event'].sum() < 5: return res
    try:
        cph = CoxPHFitter().fit(df, duration_col='time', event_col='event',
                                cluster_col='fine_match_id', robust=True)
        r = cph.summary.loc['vac']
        res.update(HR=float(r['exp(coef)']),
                   CIlo=float(r['exp(coef) lower 95%']),
                   CIhi=float(r['exp(coef) upper 95%']),
                   p=float(r['p']))
    except Exception:
        pass
    return res


# ============================================================
# S15. Novel-type acquisition with PRE-VACCINE baseline
# ============================================================
print('\n[S15] Novel-type acquisition (pre-vaccine baseline)')
sub_novel = restrict(B, lambda d: d['pre_pos_any'].notna())
sub_novel = sub_novel.copy()
sub_novel['days_to_novel'] = (sub_novel['first_novel_date']-sub_novel['index_date']).dt.days
sub_novel['novel_event']   = sub_novel['first_novel_date'].notna().astype(int)
sub_novel['novel_time']    = np.where(sub_novel['novel_event']==1,
                                       sub_novel['days_to_novel'],
                                       sub_novel['follow_up_days'])

s15_rows = []
r0 = cox_hr(sub_novel, 'novel_time', 'novel_event')
s15_rows.append({'definition':'Novel hr-HPV type (any post-index type not in pre-vaccine set)', **r0})

for lm_d, lab in [(180,'+ 6-month landmark'), (365,'+ 12-month landmark')]:
    sub = sub_novel.copy()
    eligible = (sub['follow_up_days'] >= lm_d) & (
        sub['days_to_novel'].isna() | (sub['days_to_novel'] >= lm_d))
    sub = sub[eligible].copy()
    has_event = sub['novel_event'].astype(bool) & (sub['days_to_novel'] >= lm_d)
    sub['time']  = np.where(has_event, sub['days_to_novel']-lm_d, sub['follow_up_days']-lm_d)
    sub['event'] = has_event.astype(int)
    sub = sub[sub['time'] > 0]
    r = cox_hr(sub, 'time', 'event')
    s15_rows.append({'definition':f'Novel + {lab.strip("+ ")} (pre-vaccine baseline)', **r})

s15 = pd.DataFrame(s15_rows)[['definition','n_v','n_c','ev_v','ev_c','HR','CIlo','CIhi','p']]
s15.to_csv('Data/Sensitivity_HPV_NovelType.csv', index=False, encoding='utf-8-sig')
print(s15.to_string(index=False))


# ============================================================
# S16. HPV clearance with PRE-VACCINE baseline
# ============================================================
print('\n[S16] HPV clearance (pre-vaccine baseline)')
sub_clear = restrict(B, lambda d: d['pre_pos_any'] == True)
sub_clear = sub_clear.copy()
sub_clear['days_to_clear'] = (sub_clear['first_neg_date']-sub_clear['index_date']).dt.days
sub_clear['clear_event']   = sub_clear['first_neg_date'].notna().astype(int)
sub_clear['clear_time']    = np.where(sub_clear['clear_event']==1,
                                       sub_clear['days_to_clear'],
                                       sub_clear['follow_up_days'])

sub16 = restrict(B, lambda d: d['pre_16_pos'] == 1)
sub16 = sub16.copy()
sub16['days_to_neg16'] = (sub16['first_neg16_date']-sub16['index_date']).dt.days
sub16['n16_event']     = sub16['first_neg16_date'].notna().astype(int)
sub16['n16_time']      = np.where(sub16['n16_event']==1,
                                   sub16['days_to_neg16'], sub16['follow_up_days'])
sub18 = restrict(B, lambda d: d['pre_18_pos'] == 1)
sub18 = sub18.copy()
sub18['days_to_neg18'] = (sub18['first_neg18_date']-sub18['index_date']).dt.days
sub18['n18_event']     = sub18['first_neg18_date'].notna().astype(int)
sub18['n18_time']      = np.where(sub18['n18_event']==1,
                                   sub18['days_to_neg18'], sub18['follow_up_days'])

s16_rows = [
    {'definition':'C1. Any-clearance (pre-vaccine hr-HPV+ baseline)',
     **cox_hr(sub_clear, 'clear_time', 'clear_event')},
    {'definition':'C2. HPV-16 clearance (pre-vaccine 16+ baseline)',
     **cox_hr(sub16, 'n16_time', 'n16_event')},
    {'definition':'C3. HPV-18 clearance (pre-vaccine 18+ baseline)',
     **cox_hr(sub18, 'n18_time', 'n18_event')},
]
s16 = pd.DataFrame(s16_rows)[['definition','n_v','n_c','ev_v','ev_c','HR','CIlo','CIhi','p']]
s16.to_csv('Data/Sensitivity_HPV_Clearance.csv', index=False, encoding='utf-8-sig')
print(s16.to_string(index=False))


# ============================================================
# S4. Age-stratified HRs for lesion recurrence × follow-up window
#    (use proper event time)
# ============================================================
print('\n[S4] Age-stratified lesion recurrence (corrected time)')
B['rec_event'] = B['has_recurrence'].astype(int)
B['rec_time']  = np.where(B['rec_event']==1, B['days_to_recurrence'], B['follow_up_days'])
B['age_grp']   = pd.cut(B['index_age'], bins=[-np.inf, 40, 50, np.inf],
                          labels=['<40 years', '40–49 years', '≥50 years'])

s4_rows = []
windows = [('1 yr', 365), ('2 yr', 730), ('4 yr', 1461),
           ('Full follow-up', None)]
strata  = [('All ages', None),
           ('<40 years', '<40 years'),
           ('40–49 years', '40–49 years'),
           ('≥50 years', '≥50 years')]

for stratum_lab, stratum_key in strata:
    sub_full = B if stratum_key is None else B[B['age_grp']==stratum_key]
    for fu_lab, fu_d in windows:
        d = sub_full.copy()
        if fu_d is not None:
            # Censor at the follow-up window
            has_event_in_window = d['rec_event'].astype(bool) & (d['rec_time'] <= fu_d)
            d['rec_event_w'] = has_event_in_window.astype(int)
            d['rec_time_w']  = np.minimum(d['rec_time'], fu_d)
            r = cox_hr(d, 'rec_time_w', 'rec_event_w')
        else:
            r = cox_hr(d, 'rec_time', 'rec_event')
        s4_rows.append({'stratum':stratum_lab, 'fu_label':fu_lab,
                        'fu_yr': fu_d/365.25 if fu_d else None, **r})
s4 = pd.DataFrame(s4_rows)[['stratum','fu_label','fu_yr','n_v','n_c','ev_v','ev_c',
                             'HR','CIlo','CIhi','p']]
s4.columns = ['stratum','fu_label','fu_yr','n_vac','n_ctl','events_vac','events_ctl',
              'HR','CI_lo','CI_hi','p']
s4 = s4[['n_vac','n_ctl','events_vac','events_ctl','HR','CI_lo','CI_hi','p',
        'stratum','fu_label','fu_yr']]
# add total n column to mirror existing schema
s4.insert(0, 'n', s4['n_vac'] + s4['n_ctl'])
s4.to_csv('Data/CohortB_age_fu_forest.csv', index=False, encoding='utf-8-sig')
print(s4.head(10).to_string(index=False))


# ============================================================
# S3. Restricted follow-up sensitivity for both co-primary outcomes
# ============================================================
print('\n[S3] Restricted follow-up sensitivity (corrected time)')
s3_rows = []
# Lesion recurrence (full Cohort B)
for fu_lab, fu_d in [('Baseline (full follow-up)', None),
                      ('Restricted to 3 years', 365*3),
                      ('Restricted to 5 years', 365*5)]:
    d = B.copy()
    if fu_d is not None:
        d['ev_w'] = (d['rec_event'].astype(bool) & (d['rec_time'] <= fu_d)).astype(int)
        d['t_w']  = np.minimum(d['rec_time'], fu_d)
        r = cox_hr(d, 't_w', 'ev_w')
    else:
        r = cox_hr(d, 'rec_time', 'rec_event')
    s3_rows.append({'outcome':'Lesion recurrence (≥CIN2)', 'analysis':fu_lab,
                    'adjustment':'Age-adjusted', **r})

# Unadjusted version (Baseline)
d_full = B.copy()
r_un = cox_hr(d_full, 'rec_time', 'rec_event', age_adjust=False)
s3_rows.append({'outcome':'Lesion recurrence (≥CIN2)', 'analysis':'Baseline',
                'adjustment':'Unadjusted', **r_un})

# hr-HPV clearance (pre-vaccine HPV+ subset)
for fu_lab, fu_d in [('Baseline (full follow-up)', None),
                      ('Restricted to 3 years', 365*3),
                      ('Restricted to 5 years', 365*5)]:
    d = sub_clear.copy()
    if fu_d is not None:
        d['ev_w'] = (d['clear_event'].astype(bool) & (d['clear_time'] <= fu_d)).astype(int)
        d['t_w']  = np.minimum(d['clear_time'], fu_d)
        r = cox_hr(d, 't_w', 'ev_w')
    else:
        r = cox_hr(d, 'clear_time', 'clear_event')
    s3_rows.append({'outcome':'hr-HPV clearance', 'analysis':fu_lab,
                    'adjustment':'Age-adjusted', **r})
r_un_c = cox_hr(sub_clear, 'clear_time', 'clear_event', age_adjust=False)
s3_rows.append({'outcome':'hr-HPV clearance', 'analysis':'Baseline',
                'adjustment':'Unadjusted', **r_un_c})

s3 = pd.DataFrame(s3_rows)[['outcome','analysis','adjustment','n_v','n_c',
                             'ev_v','ev_c','HR','CIlo','CIhi','p']]
s3.columns = ['outcome','analysis','adjustment','n_vac','n_ctl','events_vac','events_ctl',
              'HR','CI_lo','CI_hi','p']
s3.to_csv('Data/sensitivity_analysis_both_outcomes.csv',
          index=False, encoding='utf-8-sig')
# also keep the legacy file in sync
s3.to_csv('Data/sensitivity_analysis_results.csv',
          index=False, encoding='utf-8-sig')
print(s3.to_string(index=False))

print('\nDone. Supplementary tables S3, S4, S15, S16 rebuilt.')
