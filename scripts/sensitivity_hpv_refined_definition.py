"""
Refined HPV-acquisition outcome using pre-surgery pathology data.

Rationale: the primary "post-index hr-HPV detection" outcome conflates
persistence of the pre-surgical infection with new acquisition. By using
the last pre-surgery molecular pathology record as a baseline, we can
construct three more rigorous outcome variants:

  D1. New any-hr-HPV acquisition  -- restricted to women with a
      hr-HPV-NEGATIVE pre-surgery test; post-index hr-HPV+ counts as
      new acquisition.
  D2. New HPV-16 acquisition     -- women with pre-surgery test NOT
      positive for HPV 16; post-index HPV-16+ counts as new.
  D3. New HPV-18 acquisition     -- analogous for HPV 18.

The local pathology test reporting often pools non-16/18 high-risk
types ("Positive(Other)" or pool labels P1/P2/P3) so finer type-switch
definitions are not reliably achievable in this cohort.

Output:
  Data/Sensitivity_HPV_RefinedDefinition.csv
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
Bo = pd.read_csv('Data/final_matched_outcomes.csv', encoding='utf-8-sig')
B = B.merge(Bo[['연구번호','has_hpv_infection','hpv_infection_date',
                'hpv_types','days_to_hpv','follow_up_days']], on='연구번호')
B['첫수술일자']  = pd.to_datetime(B['첫수술일자'])
B['index_date'] = pd.to_datetime(B['index_date'])
B['hpv_infection_date'] = pd.to_datetime(B['hpv_infection_date'], errors='coerce')
B['vac']            = B['접종여부'].astype(bool).astype(int)
B['index_age']      = pd.to_numeric(B['index_age'], errors='coerce')
B['follow_up_days'] = pd.to_numeric(B['follow_up_days'], errors='coerce')
B['days_to_hpv']    = pd.to_numeric(B['days_to_hpv'], errors='coerce')

# ---------- Pre-surgery HPV typing per patient ----------
print('Extracting last pre-surgery HPV result per patient...')
hpv_b = hpv[hpv['연구번호'].isin(B['연구번호'])].copy()
hpv_b = hpv_b.merge(B[['연구번호','첫수술일자']], on='연구번호')
pre = hpv_b[hpv_b['실시일자_dt'] < hpv_b['첫수술일자']].copy()
pre_res = pre['판독결과'].apply(detect_high_risk_hpv)
pre['hpv_pos']  = pre_res.apply(lambda x: x['is_high_risk_hpv_positive'])
pre['types']    = pre_res.apply(lambda x: set(t for t in x['detected_hpv_types']
                                              if isinstance(t, int)))
pre = pre.sort_values(['연구번호','실시일자_dt'])
last_pre = pre.groupby('연구번호').agg(
    pre_pos=('hpv_pos','last'),
    pre_types=('types','last'),
    pre_test_date=('실시일자_dt','last')).reset_index()

B = B.merge(last_pre, on='연구번호', how='left')
B['has_pre_test'] = B['pre_pos'].notna()
print(f'  Patients with pre-surgery HPV test: {B["has_pre_test"].sum()}')
print(f'  Pre-surgery HPV-: {(B["pre_pos"] == False).sum()}')
print(f'  Pre-surgery HPV+: {(B["pre_pos"] == True).sum()}')

# Type-specific pre-surgery flags
def has_type(t_set, target):
    if not isinstance(t_set, set): return np.nan
    return int(target in t_set)
B['pre_16'] = B['pre_types'].apply(lambda s: has_type(s, 16))
B['pre_18'] = B['pre_types'].apply(lambda s: has_type(s, 18))

# ---------- Post-index type info ----------
def parse_type_list(s):
    if pd.isna(s) or s in ('', '[]'): return set()
    s2 = str(s).strip("[]").replace("'unspecified'","")
    out = set()
    for tok in s2.split(','):
        tok = tok.strip().strip("'\"")
        if tok.isdigit():
            out.add(int(tok))
    return out
B['post_types'] = B['hpv_types'].apply(parse_type_list)
B['post_pos']   = B['has_hpv_infection'].astype(bool).astype(int)
B['post_16'] = B['post_types'].apply(lambda s: int(16 in s))
B['post_18'] = B['post_types'].apply(lambda s: int(18 in s))


def cox_hr(df, time_col, event_col, label):
    d = df[[time_col, event_col, 'vac', 'index_age', 'fine_match_id']].dropna().rename(
        columns={time_col:'time', event_col:'event'})
    d['event'] = d['event'].astype(int)
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
        print(f'  fit failed for {label}: {e}')
    return res


print('\n===== Refined HPV outcome definitions =====\n')
results = []

# Primary (all patients, any post-index hr-HPV+ — for reference)
B['time_any'] = np.where(B['has_hpv_infection'].astype(bool),
                          B['days_to_hpv'], B['follow_up_days'])
B['event_any'] = B['has_hpv_infection'].astype(bool).astype(int)
results.append(cox_hr(B, 'time_any', 'event_any',
                      'Primary (all patients, any post-index hr-HPV+)'))

# D1. New any-hr-HPV among pre-surgery hr-HPV-NEGATIVE women
B_neg = B[B['pre_pos'] == False].copy()
B_neg['time_any'] = np.where(B_neg['has_hpv_infection'].astype(bool),
                              B_neg['days_to_hpv'], B_neg['follow_up_days'])
B_neg['event_any'] = B_neg['has_hpv_infection'].astype(bool).astype(int)
results.append(cox_hr(B_neg, 'time_any', 'event_any',
                      'D1. New any-hr-HPV (pre-surgery hr-HPV NEGATIVE only)'))

# D2. New HPV-16 acquisition (pre-16 negative → post-16 positive)
B_16 = B[(B['pre_16'] == 0)].copy()  # pre-surgery NOT 16+
# Event: post-index 16+; time = days_to_hpv if positive (and event), else follow_up_days
B_16['event_16'] = ((B_16['has_hpv_infection'].astype(bool)) & (B_16['post_16'] == 1)).astype(int)
B_16['time_16']  = np.where(B_16['event_16'].astype(bool),
                             B_16['days_to_hpv'], B_16['follow_up_days'])
results.append(cox_hr(B_16, 'time_16', 'event_16',
                      'D2. New HPV-16 (pre-surgery 16-negative only)'))

# D3. New HPV-18 acquisition
B_18 = B[(B['pre_18'] == 0)].copy()
B_18['event_18'] = ((B_18['has_hpv_infection'].astype(bool)) & (B_18['post_18'] == 1)).astype(int)
B_18['time_18']  = np.where(B_18['event_18'].astype(bool),
                             B_18['days_to_hpv'], B_18['follow_up_days'])
results.append(cox_hr(B_18, 'time_18', 'event_18',
                      'D3. New HPV-18 (pre-surgery 18-negative only)'))

# D2+D3 combined: new 16 OR 18
B_1618 = B[(B['pre_16'] == 0) & (B['pre_18'] == 0)].copy()
B_1618['event_1618'] = ((B_1618['has_hpv_infection'].astype(bool)) &
                        ((B_1618['post_16']==1) | (B_1618['post_18']==1))).astype(int)
B_1618['time_1618']  = np.where(B_1618['event_1618'].astype(bool),
                                 B_1618['days_to_hpv'], B_1618['follow_up_days'])
results.append(cox_hr(B_1618, 'time_1618', 'event_1618',
                      'D4. New HPV-16/18 (pre-surgery 16- AND 18-negative only)'))

# ---------- Print and save ----------
print(f'  {"Definition":62s}  n_v   n_c   ev_v  ev_c   HR (95% CI)         p')
for r in results:
    if not np.isnan(r['HR']):
        print(f"  {r['definition']:62s}  {r['n_v']:>3}  {r['n_c']:>3}   "
              f"{r['ev_v']:>3}  {r['ev_c']:>4}   "
              f"{r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  {r['p']:.3f}")
    else:
        print(f"  {r['definition']:62s}  {r['n_v']:>3}  {r['n_c']:>3}   "
              f"{r['ev_v']:>3}  {r['ev_c']:>4}   insufficient events")

out = pd.DataFrame(results)[
    ['definition','n_v','n_c','ev_v','ev_c','HR','CIlo','CIhi','p']]
out.to_csv('Data/Sensitivity_HPV_RefinedDefinition.csv',
           index=False, encoding='utf-8-sig')
print('\nSaved: Data/Sensitivity_HPV_RefinedDefinition.csv')
