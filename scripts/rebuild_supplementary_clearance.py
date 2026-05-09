"""
Rebuild Cohort B supplementary tables to use the new co-primary HPV
outcome (clearance / regression among pre-vaccine hr-HPV+ women)
instead of the legacy "HPV reinfection" / post-index detection
endpoint.

Tables affected:
  S6  CohortB_vaccine_interaction.csv         vaccine-type interaction
  S7  CohortB_HR_revised.csv                  cluster-robust HR + PY
  S14 Sensitivity_VaccineType_ByCalendar.csv  vaccine-type by calendar
  S3  CohortB_HR_revised.csv  also includes Schoenfeld diagnostics
"""
import warnings; warnings.filterwarnings('ignore')
import sys, pandas as pd, numpy as np
sys.path.insert(0, 'scripts')
from extract_pathology_outcomes import detect_high_risk_hpv
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import chi2

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
B['index_year']         = B['index_date'].dt.year

# Vaccine type (inherit to controls)
vt_by_match = B.loc[B['vac']==1].groupby('fine_match_id')['백신종류'].first()
B['vacc_type'] = B.apply(
    lambda r: r['백신종류'] if r['vac']==1 else vt_by_match.get(r['fine_match_id'], np.nan),
    axis=1)

# Pre-vaccine HPV+ baseline + first post-index neg test
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
neg = post[~post['post_pos']]
first_neg = neg.sort_values(['연구번호','실시일자_dt']).groupby('연구번호').first().reset_index()[
    ['연구번호','실시일자_dt']].rename(columns={'실시일자_dt':'first_neg_date'})
B = B.merge(first_neg, on='연구번호', how='left')

# Outcome times
B['rec_event']  = B['has_recurrence'].astype(int)
B['rec_time']   = np.where(B['rec_event']==1, B['days_to_recurrence'], B['follow_up_days'])
B['days_to_clear'] = (B['first_neg_date'] - B['index_date']).dt.days
B['clear_event']   = B['first_neg_date'].notna().astype(int)
B['clear_time']    = np.where(B['clear_event']==1, B['days_to_clear'], B['follow_up_days'])

# ---------- Helpers ----------
def restrict(df, vac_cond):
    keep_ids = set(df.loc[(df['vac']==1) & vac_cond(df), 'fine_match_id'])
    sub = df[df['fine_match_id'].isin(keep_ids)].copy()
    sub = sub[(sub['vac']==1) | vac_cond(sub)]
    return sub

def fit_basic(df, ev_col, time_col):
    d = df[[time_col, ev_col, 'vac', 'index_age', 'fine_match_id']].dropna().rename(
        columns={time_col:'time', ev_col:'event'})
    d['event'] = d['event'].astype(int)
    d = d[d['time'] > 0]
    n_v = int((d['vac']==1).sum()); n_c = int((d['vac']==0).sum())
    py_v = float(d.loc[d['vac']==1,'time'].sum() / 365.25)
    py_c = float(d.loc[d['vac']==0,'time'].sum() / 365.25)
    e_v = int(((d['vac']==1) & (d['event']==1)).sum())
    e_c = int(((d['vac']==0) & (d['event']==1)).sum())
    res = dict(n_v=n_v, n_c=n_c, py_v=py_v, py_c=py_c,
               ir_v=e_v/py_v*1000 if py_v else np.nan,
               ir_c=e_c/py_c*1000 if py_c else np.nan,
               ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan,
               PH_p_vaccinated=np.nan, PH_p_age=np.nan)
    if d['event'].sum() < 5: return res
    cph = CoxPHFitter().fit(d, duration_col='time', event_col='event',
                            cluster_col='fine_match_id', robust=True)
    r = cph.summary.loc['vac']
    res.update(HR=float(r['exp(coef)']),
               CIlo=float(r['exp(coef) lower 95%']),
               CIhi=float(r['exp(coef) upper 95%']),
               p=float(r['p']))
    try:
        ph = proportional_hazard_test(cph, d, time_transform='rank')
        ph_df = ph.summary
        if 'vac' in ph_df.index:
            res['PH_p_vaccinated'] = float(ph_df.loc['vac','p'])
        if 'index_age' in ph_df.index:
            res['PH_p_age'] = float(ph_df.loc['index_age','p'])
    except Exception:
        pass
    return res

def vacc_interaction(df, ev_col, time_col):
    d = df[[time_col, ev_col, 'vac', 'index_age', 'vacc_type', 'fine_match_id']].dropna().copy()
    d = d.rename(columns={time_col:'time', ev_col:'event'})
    d['event'] = d['event'].astype(int)
    d = d[d['time'] > 0]
    d['type_Cervarix'] = (d['vacc_type']=='Cervarix').astype(int)
    d['type_Gardasil'] = (d['vacc_type']=='Gardasil').astype(int)
    d['vac_x_Cervarix'] = d['vac'] * d['type_Cervarix']
    d['vac_x_Gardasil'] = d['vac'] * d['type_Gardasil']
    full_cols = ['time','event','vac','index_age','type_Cervarix','type_Gardasil',
                 'vac_x_Cervarix','vac_x_Gardasil','fine_match_id']
    red_cols  = ['time','event','vac','index_age','type_Cervarix','type_Gardasil','fine_match_id']
    if d['event'].sum() < 5:
        return None
    try:
        full = CoxPHFitter().fit(d[full_cols], duration_col='time', event_col='event',
                                 cluster_col='fine_match_id', robust=True)
        red  = CoxPHFitter().fit(d[red_cols], duration_col='time', event_col='event',
                                 cluster_col='fine_match_id', robust=True)
    except Exception:
        return None
    lrt = 2*(full.log_likelihood_ - red.log_likelihood_)
    lrt_p = float(1 - chi2.cdf(lrt, df=2))
    sm = full.summary; cov = full.variance_matrix_
    coef_v = sm.loc['vac','coef']; se_v = sm.loc['vac','se(coef)']
    coef_c = coef_v + sm.loc['vac_x_Cervarix','coef']
    coef_g = coef_v + sm.loc['vac_x_Gardasil','coef']
    se_c = np.sqrt(cov.loc['vac','vac'] + cov.loc['vac_x_Cervarix','vac_x_Cervarix']
                  + 2*cov.loc['vac','vac_x_Cervarix'])
    se_g = np.sqrt(cov.loc['vac','vac'] + cov.loc['vac_x_Gardasil','vac_x_Gardasil']
                  + 2*cov.loc['vac','vac_x_Gardasil'])
    def ci(c,s): return (float(np.exp(c)), float(np.exp(c-1.96*s)), float(np.exp(c+1.96*s)))
    return {
        'lrt_chi2':float(lrt),'lrt_p':lrt_p,
        'Gardasil9_HR':ci(coef_v,se_v)[0],'Gardasil9_CI_lo':ci(coef_v,se_v)[1],'Gardasil9_CI_hi':ci(coef_v,se_v)[2],
        'Cervarix_HR':ci(coef_c,se_c)[0], 'Cervarix_CI_lo':ci(coef_c,se_c)[1], 'Cervarix_CI_hi':ci(coef_c,se_c)[2],
        'Gardasil_HR':ci(coef_g,se_g)[0], 'Gardasil_CI_lo':ci(coef_g,se_g)[1], 'Gardasil_CI_hi':ci(coef_g,se_g)[2],
    }


# ============================================================
# S7. Cluster-robust HR with PY, IR, Schoenfeld diagnostics
#    - Lesion recurrence (full Cohort B)
#    - hr-HPV clearance (pre-vaccine HPV+ subset)
#    - Post-index detection (full cohort, sensitivity)
# ============================================================
print('\n[S7] Cluster-robust HRs with person-years & PH diagnostics')
sub_clear = restrict(B, lambda d: d['pre_pos_any'] == True)

s7_rows = []
r1 = fit_basic(B, 'rec_event', 'rec_time')
s7_rows.append({'outcome':'Lesion recurrence (≥CIN2)', **r1})
r2 = fit_basic(sub_clear, 'clear_event', 'clear_time')
s7_rows.append({'outcome':'hr-HPV clearance (pre-vaccine HPV+, co-primary)', **r2})
# Post-index detection sensitivity using days_to_hpv when event happens
B['hpv_event'] = pd.read_csv('Data/final_matched_outcomes.csv',
                              encoding='utf-8-sig')['has_hpv_infection'].astype(int).values
days_to_hpv = pd.to_numeric(pd.read_csv(
    'Data/final_matched_outcomes.csv', encoding='utf-8-sig')['days_to_hpv'], errors='coerce').values
B['hpv_time'] = np.where(B['hpv_event']==1, days_to_hpv, B['follow_up_days'])
r3 = fit_basic(B, 'hpv_event', 'hpv_time')
s7_rows.append({'outcome':'Post-index hr-HPV detection (sensitivity)', **r3})

s7 = pd.DataFrame(s7_rows)[
    ['outcome','n_v','n_c','py_v','py_c','ir_v','ir_c','ev_v','ev_c',
     'HR','CIlo','CIhi','p','PH_p_vaccinated','PH_p_age']]
s7.columns = ['outcome','n_vac','n_ctl','py_vac','py_ctl','ir_vac_per1000py','ir_ctl_per1000py',
              'events_vac','events_ctl','HR_clusterRobust','CI_lo','CI_hi','p_unadjusted',
              'PH_p_vaccinated','PH_p_age']
s7.to_csv('Data/CohortB_HR_revised.csv', index=False, encoding='utf-8-sig')
print(s7.to_string(index=False))
print('Saved: Data/CohortB_HR_revised.csv')


# ============================================================
# S6. Vaccine-type interaction analysis on the new outcomes
# ============================================================
print('\n[S6] Vaccine-type interaction on new outcomes')
s6_rows = []
for ev_label, ev_col, time_col, sub in [
    ('Lesion recurrence (≥CIN2)', 'rec_event', 'rec_time', B),
    ('hr-HPV clearance (pre-vaccine HPV+)', 'clear_event', 'clear_time', sub_clear),
    ('Post-index hr-HPV detection (sensitivity)', 'hpv_event', 'hpv_time', B),
]:
    res = vacc_interaction(sub, ev_col, time_col)
    if res is None:
        s6_rows.append({'outcome':ev_label, 'LRT_interaction_chi2':np.nan,
                        'LRT_df':2, 'LRT_p':np.nan,
                        **{k:np.nan for k in ['Gardasil9_HR','Gardasil9_CI_lo','Gardasil9_CI_hi',
                            'Cervarix_HR','Cervarix_CI_lo','Cervarix_CI_hi',
                            'Gardasil_HR','Gardasil_CI_lo','Gardasil_CI_hi']}})
    else:
        s6_rows.append({'outcome':ev_label, 'LRT_interaction_chi2':res['lrt_chi2'],
                        'LRT_df':2, 'LRT_p':res['lrt_p'],
                        'Gardasil9_HR':res['Gardasil9_HR'],
                        'Gardasil9_CI_lo':res['Gardasil9_CI_lo'],
                        'Gardasil9_CI_hi':res['Gardasil9_CI_hi'],
                        'Cervarix_HR':res['Cervarix_HR'],
                        'Cervarix_CI_lo':res['Cervarix_CI_lo'],
                        'Cervarix_CI_hi':res['Cervarix_CI_hi'],
                        'Gardasil_HR':res['Gardasil_HR'],
                        'Gardasil_CI_lo':res['Gardasil_CI_lo'],
                        'Gardasil_CI_hi':res['Gardasil_CI_hi']})
s6 = pd.DataFrame(s6_rows)
s6.to_csv('Data/CohortB_vaccine_interaction.csv', index=False, encoding='utf-8-sig')
for _, row in s6.iterrows():
    print(f"  {row['outcome']:50s}  LRT chi2={row['LRT_interaction_chi2']}  "
          f"p={row['LRT_p']}")
print('Saved: Data/CohortB_vaccine_interaction.csv')


# ============================================================
# S14. Vaccine-type x calendar-period (HPV CLEARANCE outcome)
# ============================================================
print('\n[S14] Vaccine-type x calendar period on HPV clearance')
strata = [
    ('Full clearance subset (primary)',          sub_clear),
    ('Index year ≤ 2015 (Gardasil-4v era)',      sub_clear[sub_clear['index_year'] <= 2015]),
    ('Index year 2016–2018 (mixed era)',
                                                  sub_clear[(sub_clear['index_year'] >= 2016) & (sub_clear['index_year'] <= 2018)]),
    ('Index year ≥ 2019 (Gardasil-9v era)',      sub_clear[sub_clear['index_year'] >= 2019]),
]
s14_rows = []
for label, sub in strata:
    n_total = len(sub)
    n_v_g9 = ((sub['vac']==1) & (sub['vacc_type']=='Gardasil9')).sum()
    n_v_cv = ((sub['vac']==1) & (sub['vacc_type']=='Cervarix')).sum()
    n_v_gd = ((sub['vac']==1) & (sub['vacc_type']=='Gardasil')).sum()
    res = vacc_interaction(sub, 'clear_event', 'clear_time')
    base = dict(stratum=label, n_total=n_total,
                n_vac_G9=int(n_v_g9), n_vac_Cv=int(n_v_cv), n_vac_G4=int(n_v_gd))
    if res is None:
        base.update({k:np.nan for k in ['lrt_chi2','lrt_p',
                'Gardasil9_HR','Gardasil9_lo','Gardasil9_hi',
                'Cervarix_HR','Cervarix_lo','Cervarix_hi',
                'Gardasil_HR','Gardasil_lo','Gardasil_hi']})
    else:
        base.update(lrt_chi2=res['lrt_chi2'], lrt_p=res['lrt_p'],
                    Gardasil9_HR=res['Gardasil9_HR'],
                    Gardasil9_lo=res['Gardasil9_CI_lo'],
                    Gardasil9_hi=res['Gardasil9_CI_hi'],
                    Cervarix_HR=res['Cervarix_HR'],
                    Cervarix_lo=res['Cervarix_CI_lo'],
                    Cervarix_hi=res['Cervarix_CI_hi'],
                    Gardasil_HR=res['Gardasil_HR'],
                    Gardasil_lo=res['Gardasil_CI_lo'],
                    Gardasil_hi=res['Gardasil_CI_hi'])
    s14_rows.append(base)
s14 = pd.DataFrame(s14_rows)
s14.to_csv('Data/Sensitivity_VaccineType_ByCalendar.csv',
           index=False, encoding='utf-8-sig')
for _, r in s14.iterrows():
    print(f"  {r['stratum']:40s}  n_total={r['n_total']}, "
          f"LRT p = {r['lrt_p']}")
print('Saved: Data/Sensitivity_VaccineType_ByCalendar.csv')

print('\nDone. Supplementary tables S6, S7, S14 rebuilt with co-primary clearance outcome.')
