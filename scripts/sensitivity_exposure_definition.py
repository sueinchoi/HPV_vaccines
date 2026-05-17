"""
Sensitivity analyses on the HPV-vaccine EXPOSURE definition.

Three analyses:
  S1. Dose threshold — re-fit primary Cox HRs requiring ≥2 doses and ≥3 doses
      to qualify as 'vaccinated', for both Cohort A (chronic comorbidities)
      and Cohort B (lesion recurrence / HPV reinfection).
  S2. Prescription-code cross-check — does the drug-code (처방코드) field
      identify the same set of patients as the name-string match? Are there
      patients caught by code only, or by name only?
  S3. Mixed-type recipients — how many patients received >1 vaccine type
      across their dose history? How does this affect the vaccine-type
      subgroup attribution?

Outputs (Data/):
  Sensitivity_DoseThreshold_HR.csv    — HRs by dose-threshold per outcome
  Sensitivity_PrescriptionCode.csv    — code vs name ascertainment overlap
  Sensitivity_MixedVaccineType.csv    — mixed-type recipient breakdown
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

RX_FILE     = 'Data/한국 HPV 코호트 자료를 이용한 자_처방정보.csv'
COHORT_FILE = 'Data/한국 HPV 코호트 자료를 이용한 자_코호트.csv'
B_OUT       = 'Data/final_matched_outcomes.csv'
B_COH       = 'Data/final_matched_cohort.csv'

# ---------------------------------------------------------------------------
# Common: load and label the raw prescription file
# ---------------------------------------------------------------------------
print('Loading prescription file...')
rx = pd.read_csv(RX_FILE, encoding='cp949', low_memory=False)
mask_name = (rx['처방명'].astype(str).str.contains('Gardasil|Cervarix|HPV vaccine',
                                                  case=False, na=False) |
             rx['처방한글명'].astype(str).str.contains('가다실|서바릭스', na=False))
rx_vac = rx[mask_name].copy()
rx_vac['처방일자'] = pd.to_datetime(rx_vac['처방일자'].astype('Int64').astype(str),
                                  format='%Y%m%d', errors='coerce')

def vaccine_type(name, knm):
    s = f"{str(name).lower()} {str(knm)}"
    if ('gardasil 9' in s) or ('gardasil9' in s) or ('9가' in s) or ('9-valent' in s):
        return 'Gardasil9'
    if ('gardasil' in s) or ('가다실' in s):
        return 'Gardasil'    # 4-valent
    if ('cervarix' in s) or ('서바릭스' in s):
        return 'Cervarix'    # 2-valent
    return 'Other'

rx_vac['vt'] = rx_vac.apply(lambda r: vaccine_type(r['처방명'], r['처방한글명']), axis=1)

# Doses per patient × type
doses_long = (rx_vac.groupby(['연구번호', 'vt']).size()
              .unstack('vt', fill_value=0)
              .rename_axis(None, axis=1))
for c in ['Gardasil9', 'Gardasil', 'Cervarix', 'Other']:
    if c not in doses_long.columns:
        doses_long[c] = 0
doses_long['total_doses'] = doses_long[['Gardasil9','Gardasil','Cervarix','Other']].sum(axis=1)

print(f'  vaccinated patients (any dose): {len(doses_long):,}')
print(f'  total prescription rows for HPV vaccines: {len(rx_vac):,}')

# ===========================================================================
# S2. PRESCRIPTION-CODE cross-check
# ===========================================================================
print('\n[S2] Prescription-code cross-check')
# Build a code-based mask. Common KCD/EDI hint: HPV vaccines have specific drug codes;
# we don't have a published list, so we operationally define 'vaccine code' as any
# 처방코드 that appears in the name-matched rows ≥ X% of the time.
# Then compare populations.
codes_in_name_match = rx_vac['처방코드'].value_counts()
print(f'  distinct 처방코드 values caught by name match: {codes_in_name_match.shape[0]}')
print(f'  top 10 codes:')
print(codes_in_name_match.head(10).to_string())

code_set = set(rx_vac['처방코드'].dropna().astype(str).unique())
mask_code = rx['처방코드'].astype(str).isin(code_set)
rx_by_code = rx[mask_code]
n_pts_name = rx_vac['연구번호'].nunique()
n_pts_code = rx_by_code['연구번호'].nunique()
both = set(rx_vac['연구번호']).intersection(set(rx_by_code['연구번호']))
only_name = set(rx_vac['연구번호']) - set(rx_by_code['연구번호'])
only_code = set(rx_by_code['연구번호']) - set(rx_vac['연구번호'])
print(f'  patients identified by NAME match:         {n_pts_name:,}')
print(f'  patients identified by CODE match:         {n_pts_code:,}')
print(f'  intersection (both):                       {len(both):,}')
print(f'  name-only (caught by string but not code): {len(only_name):,}')
print(f'  code-only (caught by code but not string): {len(only_code):,}')

s2 = pd.DataFrame([
    {'method':'name match (current)', 'n_patients':n_pts_name},
    {'method':'code match (codes seen in name-matched rows)', 'n_patients':n_pts_code},
    {'method':'intersection', 'n_patients':len(both)},
    {'method':'name-only',    'n_patients':len(only_name)},
    {'method':'code-only',    'n_patients':len(only_code)},
])
s2.to_csv('Data/Sensitivity_PrescriptionCode.csv', index=False, encoding='utf-8-sig')
print('  Saved: Data/Sensitivity_PrescriptionCode.csv')

# ===========================================================================
# S3. MIXED-VACCINE-TYPE recipients
# ===========================================================================
print('\n[S3] Mixed vaccine-type recipients')
def mixed_label(row):
    types = [t for t in ['Gardasil9','Gardasil','Cervarix'] if row[t] > 0]
    if len(types) == 0:
        return 'Other only'
    if len(types) == 1:
        return f'Single: {types[0]}'
    return 'Mixed: ' + '+'.join(types)

doses_long['type_pattern'] = doses_long.apply(mixed_label, axis=1)
pattern_counts = doses_long['type_pattern'].value_counts()
print(pattern_counts.to_string())
n_mixed = doses_long['type_pattern'].str.startswith('Mixed').sum()
print(f'  total patients with >1 vaccine type: {n_mixed:,} '
      f'({n_mixed/len(doses_long)*100:.1f}% of {len(doses_long):,})')

# Per-patient first vaccine type vs last vaccine type — to see how vaccine-type
# attribution would differ depending on chosen rule
rx_sorted = rx_vac.sort_values(['연구번호','처방일자'])
first_type = rx_sorted.groupby('연구번호').first()['vt'].rename('first_type')
last_type  = rx_sorted.groupby('연구번호').last()['vt'].rename('last_type')
attribution = pd.concat([first_type, last_type], axis=1)
attribution['changed'] = attribution['first_type'] != attribution['last_type']
n_changed = int(attribution['changed'].sum())
print(f'  patients whose FIRST and LAST vaccine type differ: {n_changed} '
      f'({n_changed/len(attribution)*100:.1f}%)')

s3 = pattern_counts.rename_axis('pattern').reset_index(name='n_patients')
s3.loc[len(s3)] = ['__ANY mixed (>1 type)', n_mixed]
s3.loc[len(s3)] = ['__first ≠ last vaccine type', n_changed]
s3.to_csv('Data/Sensitivity_MixedVaccineType.csv', index=False, encoding='utf-8-sig')
print('  Saved: Data/Sensitivity_MixedVaccineType.csv')

# ===========================================================================
# S1. DOSE-THRESHOLD sensitivity for Cohort B (efficacy)
# ===========================================================================
print('\n[S1] Dose-threshold sensitivity — Cohort B')
B = pd.read_csv(B_OUT, encoding='utf-8-sig')
Bc = pd.read_csv(B_COH, encoding='utf-8-sig')
B = B.merge(Bc[['연구번호','백신종류']], on='연구번호', how='left')
B['vac'] = B['접종여부'].astype(bool).astype(int)
B['follow_up_days'] = pd.to_numeric(B['follow_up_days'], errors='coerce')
B['index_age'] = pd.to_numeric(B['index_age'], errors='coerce')

# Attach per-patient total dose count (only meaningful for vaccinated)
B = B.merge(doses_long[['total_doses']], left_on='연구번호',
            right_index=True, how='left')
B['total_doses'] = B['total_doses'].fillna(0).astype(int)

print('  vaccinated dose distribution in Cohort B:')
print(B.loc[B['vac']==1, 'total_doses'].value_counts().sort_index().to_string())

def cox_HR(d, ev_col):
    """Cox HR for vaccinated vs non-vaccinated, age-adjusted, cluster on match id.
    Time is days_to_event when an event occurred, follow_up_days otherwise."""
    df = d.copy()
    if ev_col == 'has_recurrence':
        df['time'] = np.where(df['has_recurrence'].astype(bool),
                               pd.to_numeric(df['days_to_recurrence'], errors='coerce'),
                               df['follow_up_days'])
    elif ev_col == 'has_hpv_infection':
        df['time'] = np.where(df['has_hpv_infection'].astype(bool),
                               pd.to_numeric(df['days_to_hpv'], errors='coerce'),
                               df['follow_up_days'])
    else:
        df['time'] = df['follow_up_days']
    df = df[['time', ev_col, 'vac', 'index_age', 'fine_match_id']].dropna()
    df = df.rename(columns={ev_col:'event'})
    df['event'] = df['event'].astype(int)
    df = df[df['time'] > 0]
    n_v = int((df['vac']==1).sum()); n_c = int((df['vac']==0).sum())
    e_v = int(((df['vac']==1) & (df['event']==1)).sum())
    e_c = int(((df['vac']==0) & (df['event']==1)).sum())
    res = dict(n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if e_v + e_c < 3 or n_v < 2 or n_c < 2 or df['event'].sum() < 3:
        return res
    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col='time', event_col='event',
                cluster_col='fine_match_id', robust=True)
        r = cph.summary.loc['vac']
        res.update(HR=float(r['exp(coef)']),
                   CIlo=float(r['exp(coef) lower 95%']),
                   CIhi=float(r['exp(coef) upper 95%']),
                   p=float(r['p']))
    except Exception as e:
        print(f'    Cox fit failed for {ev_col}: {e}')
    return res

cohortB_rows = []
# Lesion recurrence — full Cohort B
for thr_label, thr in [('≥1 dose (primary)', 1),
                        ('≥2 doses', 2),
                        ('≥3 doses (complete)', 3)]:
    bad_match_ids = set(
        B.loc[(B['vac']==1) & (B['total_doses'] < thr), 'fine_match_id'])
    sub = B[~B['fine_match_id'].isin(bad_match_ids)].copy()
    r = cox_HR(sub, 'has_recurrence')
    r.update(cohort='B', outcome='Lesion recurrence',
             definition=thr_label, threshold=thr)
    cohortB_rows.append(r)
    if not np.isnan(r['HR']):
        print(f'  Lesion recurrence    {thr_label:22s} '
              f'n_v={r["n_v"]:>3}/n_c={r["n_c"]:>3}  '
              f"HR={r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  p={r['p']:.3f}")

# hr-HPV clearance — pre-vaccine HPV-positive subset (n = 292)
BC = pd.read_csv('Data/CohortB_Clearance_Analytic.csv', encoding='utf-8-sig')
BC['index_date']     = pd.to_datetime(BC['index_date'])
BC['first_neg_date'] = pd.to_datetime(BC['first_neg_date'], errors='coerce')
BC['vac']            = BC['vac'].astype(int)
BC['index_age']      = pd.to_numeric(BC['index_age'], errors='coerce')
BC['follow_up_days'] = pd.to_numeric(BC['follow_up_days'], errors='coerce')
BC['has_clearance']  = BC['first_neg_date'].notna().astype(int)
BC['days_to_clear']  = (BC['first_neg_date'] - BC['index_date']).dt.days
BC = BC.merge(doses_long[['total_doses']], left_on='연구번호',
              right_index=True, how='left')
BC['total_doses'] = BC['total_doses'].fillna(0).astype(int)


def cox_HR_clearance(d):
    df = d.copy()
    df['time'] = np.where(df['has_clearance'].astype(bool),
                          df['days_to_clear'],
                          df['follow_up_days'])
    df = df[['time', 'has_clearance', 'vac', 'index_age', 'fine_match_id']].dropna(
        ).rename(columns={'has_clearance': 'event'})
    df['event'] = df['event'].astype(int)
    df = df[df['time'] > 0]
    n_v = int((df['vac']==1).sum()); n_c = int((df['vac']==0).sum())
    e_v = int(((df['vac']==1) & (df['event']==1)).sum())
    e_c = int(((df['vac']==0) & (df['event']==1)).sum())
    res = dict(n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if e_v + e_c < 3 or n_v < 2 or n_c < 2:
        return res
    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col='time', event_col='event',
                cluster_col='fine_match_id', robust=True)
        r = cph.summary.loc['vac']
        res.update(HR=float(r['exp(coef)']),
                   CIlo=float(r['exp(coef) lower 95%']),
                   CIhi=float(r['exp(coef) upper 95%']),
                   p=float(r['p']))
    except Exception as e:
        print(f'    Cox fit failed for clearance: {e}')
    return res


for thr_label, thr in [('≥1 dose (primary)', 1),
                        ('≥2 doses', 2),
                        ('≥3 doses (complete)', 3)]:
    bad_match_ids = set(
        BC.loc[(BC['vac']==1) & (BC['total_doses'] < thr), 'fine_match_id'])
    sub = BC[~BC['fine_match_id'].isin(bad_match_ids)].copy()
    r = cox_HR_clearance(sub)
    r.update(cohort='B', outcome='hr-HPV clearance',
             definition=thr_label, threshold=thr)
    cohortB_rows.append(r)
    if not np.isnan(r['HR']):
        print(f'  hr-HPV clearance     {thr_label:22s} '
              f'n_v={r["n_v"]:>3}/n_c={r["n_c"]:>3}  '
              f"HR={r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  p={r['p']:.3f}")

# ===========================================================================
# S1b. Dose-threshold sensitivity for Cohort A (safety)
# ===========================================================================
print('\n[S1] Dose-threshold sensitivity — Cohort A')
# Re-run abbreviated Cohort A pipeline (PSM 1:1 matched dataset reproduced here
# would be overkill; we use the lighter approach of recomputing the *same*
# matched dataset by importing the heavy build function from make_main_figures.)
import sys
sys.path.insert(0, 'scripts')
from make_main_figures import build_cohort_a_matched, make_tte, ANY5, MCE

m = build_cohort_a_matched()
m = m.merge(doses_long[['total_doses']], left_on='pid',
            right_index=True, how='left')
m['total_doses'] = m['total_doses'].fillna(0).astype(int)

def cox_HR_A(tte, threshold):
    """Cox on the Cohort A pair-matched outcome dataframe with optional dose threshold."""
    if threshold > 1:
        # Drop vaccinated participants who don't meet the threshold AND drop their
        # paired controls together (preserve 1:1 matched structure).
        v = tte[tte['vaccinated']==1]
        # Need to know dose count per pid — merge from m
        v = v.merge(m[['pid','total_doses']], on='pid', how='left')
        bad_pairs = set(v.loc[v['total_doses'] < threshold, 'pair_id'])
        tte = tte[~tte['pair_id'].isin(bad_pairs)].reset_index(drop=True)
    n_v = int((tte['vaccinated']==1).sum()); n_c = int((tte['vaccinated']==0).sum())
    e_v = int(((tte['status']==1)&(tte['vaccinated']==1)).sum())
    e_c = int(((tte['status']==1)&(tte['vaccinated']==0)).sum())
    res = dict(n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if e_v + e_c < 5 or e_v < 1 or e_c < 1:
        return res
    d = tte.copy(); d['event'] = (d['status']==1).astype(int)
    try:
        cph = CoxPHFitter()
        cph.fit(d[['time','event','vaccinated','pair_id']],
                duration_col='time', event_col='event',
                cluster_col='pair_id', robust=True)
        r = cph.summary.loc['vaccinated']
        res.update(HR=float(r['exp(coef)']),
                   CIlo=float(r['exp(coef) lower 95%']),
                   CIhi=float(r['exp(coef) upper 95%']),
                   p=float(r['p']))
    except Exception as e:
        print(f'    Cox fit failed: {e}')
    return res

cohortA_rows = []
for ev_label, ev_def in [('Any-of-5 composite', ANY5),
                          ('MCE composite',     MCE),
                          ('Diabetes',          '3'),
                          ('Hypertension',      '2'),
                          ('Angina/MI',         '1')]:
    tte = make_tte(m, ev_def)
    for thr_label, thr in [('≥1 dose (primary)', 1),
                            ('≥2 doses', 2),
                            ('≥3 doses (complete)', 3)]:
        r = cox_HR_A(tte, thr)
        r.update(cohort='A', outcome=ev_label, definition=thr_label, threshold=thr)
        cohortA_rows.append(r)
        if not np.isnan(r['HR']):
            print(f'  {ev_label:20s} {thr_label:22s} '
                  f'n_v={r["n_v"]:>4}/n_c={r["n_c"]:>4}  '
                  f"HR={r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  p={r['p']:.3f}")
        else:
            print(f'  {ev_label:20s} {thr_label:22s} insufficient events')

# Save combined dose-threshold output
out = pd.DataFrame(cohortA_rows + cohortB_rows)[
    ['cohort','outcome','definition','threshold',
     'n_v','n_c','ev_v','ev_c','HR','CIlo','CIhi','p']]
out.to_csv('Data/Sensitivity_DoseThreshold_HR.csv',
           index=False, encoding='utf-8-sig')
print('\nSaved: Data/Sensitivity_DoseThreshold_HR.csv')

# ---------------------------------------------------------------------------
print('\n=== Sensitivity analyses complete ===')
print('  Data/Sensitivity_DoseThreshold_HR.csv')
print('  Data/Sensitivity_PrescriptionCode.csv')
print('  Data/Sensitivity_MixedVaccineType.csv')
