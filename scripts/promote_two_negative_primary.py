"""
Promote the two-consecutive-negatives definition to the PRIMARY clearance
co-primary outcome, with the single-negative definition demoted to a
sensitivity comparison.

Rationale: standard epidemiological convention (Bouvard IARC 2009, Insinga
2010) defines hr-HPV clearance as two consecutive negative tests, because a
single negative may reflect imperfect assay sensitivity (~5-10% false-
negative rate for hr-HPV PCR/chip assays) or transient viral-load
fluctuation rather than true clearance.

Concretely this script:
  1. Recomputes per-patient clearance events as the date of the FIRST of two
     consecutive post-index hr-HPV-negative records.
  2. Rewrites Data/CohortB_Clearance_Analytic.csv with these new event
     dates so downstream figures and tables (Table 3, Figure 3, Figure 4,
     Supplementary S6/S7/S14/S17) automatically pick up the new primary.
  3. Renames the prior sensitivity file
       Data/Sensitivity_HPV_Clearance_TwoNegative.csv
     to
       Data/Sensitivity_HPV_Clearance_SingleNegative.csv
     and reverses its row order: the single-negative variant is now the
     sensitivity.

Run after this script:
  python3 scripts/rebuild_table3.py
  python3 scripts/rebuild_supplementary_clearance.py
  python3 scripts/sensitivity_clearance_time_stratified.py
  python3 -c "import sys; sys.path.insert(0,'scripts'); \
              from make_main_figures import figure3, figure4_subgroup; \
              figure3(); figure4_subgroup()"
"""
import warnings; warnings.filterwarnings('ignore')
import sys, pandas as pd, numpy as np, os, shutil
sys.path.insert(0, 'scripts')
from extract_pathology_outcomes import detect_high_risk_hpv

PATH_FILE = 'Data/한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV'

print('Loading raw pathology...')
patho = pd.read_csv(PATH_FILE, encoding='cp949', low_memory=False)
hpv = patho[patho['병리검사구분'].isin(['분자병리','HPV'])].copy()
hpv['실시일자_dt'] = pd.to_datetime(
    hpv['실시일자'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')

B = pd.read_csv('Data/final_matched_cohort.csv', encoding='utf-8-sig')
B['index_date']  = pd.to_datetime(B['index_date'])
B['최종추적일자'] = pd.to_datetime(B['최종추적일자'])
B['vac']        = B['접종여부'].astype(bool).astype(int)
B['index_age']  = pd.to_numeric(B['index_age'], errors='coerce')
B['follow_up_days'] = (B['최종추적일자'] - B['index_date']).dt.days

# Pre-vaccine baseline
hpv_b = hpv[hpv['연구번호'].isin(B['연구번호'])].merge(
    B[['연구번호','index_date']], on='연구번호')
pre = hpv_b[hpv_b['실시일자_dt'] < hpv_b['index_date']].copy()
pre['detect'] = pre['판독결과'].apply(detect_high_risk_hpv)
pre_summary = pre.groupby('연구번호').apply(lambda g: pd.Series({
    'pre_pos_any': any(r['is_high_risk_hpv_positive'] for r in g['detect']),
    'pre_types':   set().union(*[set(t for t in r['detected_hpv_types']
                                     if isinstance(t, int)) for r in g['detect']]),
})).reset_index()
B = B.merge(pre_summary, on='연구번호', how='left')

# Restrict matched-set integrity to clearance subset
keep_ids = set(B.loc[(B['vac']==1) & (B['pre_pos_any']==True), 'fine_match_id'])
clear_subset = B[B['fine_match_id'].isin(keep_ids)].copy()
clear_subset = clear_subset[(clear_subset['vac']==1) | (clear_subset['pre_pos_any']==True)]
print(f'Clearance subset (pre-vaccine HPV+): {len(clear_subset)} '
      f'(vac {int((clear_subset["vac"]==1).sum())} / non-vac {int((clear_subset["vac"]==0).sum())})')

# Post-index records
post = hpv_b[hpv_b['실시일자_dt'] > hpv_b['index_date']].copy()
post = post.merge(B[['연구번호','vac','fine_match_id','follow_up_days']], on='연구번호')
post['detect']   = post['판독결과'].apply(detect_high_risk_hpv)
post['post_pos'] = post['detect'].apply(lambda r: r['is_high_risk_hpv_positive'])
post = post.sort_values(['연구번호','실시일자_dt'])

# ---------- Compute first-of-two-consecutive-negative date per patient ----------
def first_two_consecutive_negatives(g):
    """Return date of first of two consecutive negative records (or None)."""
    g = g.sort_values('실시일자_dt').reset_index(drop=True)
    for i in range(len(g) - 1):
        if (not g.loc[i, 'post_pos']) and (not g.loc[i+1, 'post_pos']):
            return g.loc[i, '실시일자_dt']
    return None

print('Computing two-consecutive-negative dates...')
two_neg = post.groupby('연구번호').apply(first_two_consecutive_negatives)
two_neg = two_neg.dropna().rename('first_two_neg_date').reset_index()

# Also compute single-negative for record-keeping (kept as sensitivity)
neg_only = post[~post['post_pos']]
single_neg = neg_only.sort_values(['연구번호','실시일자_dt']).groupby('연구번호').first()[
    ['실시일자_dt']].rename(columns={'실시일자_dt':'first_single_neg_date'}).reset_index()

clear_subset = clear_subset.merge(two_neg, on='연구번호', how='left')
clear_subset = clear_subset.merge(single_neg, on='연구번호', how='left')

# PRIMARY clearance event = two consecutive negatives
clear_subset['first_neg_date']    = clear_subset['first_two_neg_date']
clear_subset['days_to_two_neg']   = (clear_subset['first_two_neg_date']
                                       - clear_subset['index_date']).dt.days
clear_subset['days_to_single_neg'] = (clear_subset['first_single_neg_date']
                                       - clear_subset['index_date']).dt.days

# Save updated analytic dataset (now keyed on two-negative as primary)
out_cols = ['연구번호','vac','index_age','fine_match_id',
            'index_date','follow_up_days',
            'first_neg_date',          # <— NOW = two-consecutive-negatives
            'first_two_neg_date',      # explicit copy
            'first_single_neg_date',   # for sensitivity reference
            'pre_types','pre_pos_any']
clear_subset[out_cols].to_csv(
    'Data/CohortB_Clearance_Analytic.csv', index=False, encoding='utf-8-sig')
print('Updated: Data/CohortB_Clearance_Analytic.csv (primary = two consecutive negatives)')

# ---------- Demote the two-negative file to a "Single-negative sensitivity" ----------
# The previous Sensitivity_HPV_Clearance_TwoNegative.csv compared single vs two-neg.
# Now that two-neg is primary, the sensitivity is the single-negative variant.
src = 'Data/Sensitivity_HPV_Clearance_TwoNegative.csv'
dst = 'Data/Sensitivity_HPV_Clearance_SingleNegative.csv'
if os.path.exists(src):
    df = pd.read_csv(src, encoding='utf-8-sig')
    # Reverse row order so that the now-primary (two-neg) is shown first as
    # the reference, with single-negative as the sensitivity comparison.
    df = df.sort_values('definition', key=lambda s: s.str.contains('Two')).reset_index(drop=True)
    # Relabel
    df['definition'] = df['definition'].str.replace(
        'Two-consecutive-negatives (S18)', 'Two consecutive negatives (primary)', regex=False)
    df['definition'] = df['definition'].str.replace(
        'Single-negative (primary)', 'Single negative test (sensitivity)', regex=False)
    df.to_csv(dst, index=False, encoding='utf-8-sig')
    if os.path.exists(src):
        os.remove(src)
    print(f'Renamed and relabelled: {src} -> {dst}')

print('\nNext steps (run in order):')
print('  python3 scripts/rebuild_table3.py')
print('  python3 scripts/rebuild_supplementary_clearance.py')
print('  python3 scripts/sensitivity_clearance_time_stratified.py')
print('  python3 -c "import sys; sys.path.insert(0,\'scripts\'); '
      'from make_main_figures import figure3, figure4_subgroup; '
      'figure3(); figure4_subgroup()"')
