"""
Regenerate the CohortB_clearance block of
Data/Table1_BaselineCharacteristics_unified.csv using the canonical v3
clearance subset (n = 233 women: 92 vaccinated / 141 fine-matched controls,
i.e. pre-vaccine hr-HPV-positive primary-cohort subset with matched-set
integrity preserved and the 3-month landmark applied).

The legacy block (n = 292: 110 / 182) was the pre-landmark CohortB_Clearance_Analytic.csv
subset and is now superseded.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from extract_pathology_outcomes import detect_high_risk_hpv  # noqa: E402
from analyze_primary_v3 import apply_landmark, first_two_consecutive_neg  # noqa: E402


def smd_cont(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def smd_bin(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    pp1, pp2 = p1 / n1, p2 / n2
    pooled = np.sqrt((pp1 * (1 - pp1) + pp2 * (1 - pp2)) / 2)
    return (pp1 - pp2) / pooled if pooled > 0 else np.nan


def p_cont(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return ttest_ind(a, b, equal_var=False).pvalue


def p_bin(a_yes, a_n, b_yes, b_n):
    if a_n == 0 or b_n == 0:
        return np.nan
    return fisher_exact([[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]])[1]


def fmt_p(p):
    if pd.isna(p):
        return '-'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def fmt_smd(s):
    return '-' if pd.isna(s) else f'{abs(s):.3f}'


def fmt_cont(s):
    s = pd.Series(s).dropna()
    return '-' if len(s) == 0 else f'{s.mean():.2f} ± {s.std():.2f}'


def fmt_pct(n_yes, n_total):
    return f'{int(n_yes)} ({100 * n_yes / n_total:.1f}%)' if n_total > 0 else '-'


def has_prevac_hr(mol_by_pid, pid, idx_dt):
    sub = mol_by_pid.get(pid)
    if sub is None:
        return False
    return bool((sub[sub['실시일자'] < idx_dt]['hpv_pos'] == True).any())


print('Loading primary v3 cohort and pathology...')
df = pd.read_csv(ROOT / 'Data' / 'primary_cohort_v3.csv', encoding='utf-8-sig')
df['index_date'] = pd.to_datetime(df['index_date'])
df['최종추적일자'] = pd.to_datetime(df['최종추적일자'])

path = pd.read_csv(
    ROOT / 'Data' / '한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV',
    encoding='cp949', low_memory=False,
)
path['실시일자'] = pd.to_datetime(path['실시일자'], format='%Y%m%d', errors='coerce')
mol = path[path['병리검사구분'].isin(['분자병리', 'HPV'])].dropna(subset=['실시일자', '판독결과']).copy()
res = mol['판독결과'].apply(detect_high_risk_hpv)
mol['hpv_pos'] = res.apply(lambda d: d['is_high_risk_hpv_positive'])
mol_by_pid = {pid: g.sort_values('실시일자') for pid, g in mol.groupby('연구번호')}

df['prevac_hr'] = df.apply(
    lambda r: has_prevac_hr(mol_by_pid, r['연구번호'], r['index_date']), axis=1
)

# Matched-set integrity: keep fine_match_id where vaccinated case has pre-vaccine hr-HPV+
fids_keep = df[(df['접종여부'] == True) & (df['prevac_hr'] == True)]['fine_match_id'].unique()
clr = df[df['fine_match_id'].isin(fids_keep) & (df['prevac_hr'] == True)].copy()

# Compute clearance event date and apply 3-mo landmark (drop early-event matched sets)
clr['first_neg_date'] = clr.apply(
    lambda r: first_two_consecutive_neg(mol_by_pid, r['연구번호'], r['index_date']), axis=1,
)
clr['has_clearance'] = clr['first_neg_date'].notna()
clr = apply_landmark(clr, 'has_clearance', 'first_neg_date')

print(f'Clearance subset (post-landmark, primary analytic): n={len(clr)} '
      f'(vac {(clr["접종여부"]==True).sum()} / non-vac {(clr["접종여부"]==False).sum()})')

# Surgical record + Seoul + smoking + BP — load from raw files
mc = pd.read_csv(ROOT / 'Data' / 'final_matched_cohort.csv', encoding='utf-8-sig')
mc['첫수술일자'] = pd.to_datetime(mc['첫수술일자'])
surg_map = mc.set_index('연구번호')['첫수술일자']
clr['첫수술일자'] = clr['연구번호'].map(surg_map)

cohort = pd.read_csv(
    ROOT / 'Data' / '한국 HPV 코호트 자료를 이용한 자_코호트.csv',
    encoding='cp949', low_memory=False,
)
cohort['region_top'] = cohort['주소'].astype(str).str.split().str[0]
cohort['is_seoul'] = (cohort['region_top'] == '서울').astype(int)
clr = clr.merge(
    cohort[['연구번호', 'is_seoul']].drop_duplicates('연구번호'), on='연구번호', how='left'
)

v = clr[clr['접종여부'] == True]
nv = clr[clr['접종여부'] == False]
n_v, n_nv = len(v), len(nv)

rows = []
header_label = f'— ≥2 DOSE + 3-MO LANDMARK + PRE-VACCINE hr-HPV+ CLEARANCE SUBSET (n={n_v + n_nv}: {n_v} vac / {n_nv} non-vac) —'
rows.append({'block': 'CohortB_clearance', 'Variable': header_label,
             'Vac': '', 'Non': '', 'p': '', 'SMD': ''})


def add_cont(label, col):
    a, b = v[col], nv[col]
    rows.append({'block': 'CohortB_clearance', 'Variable': label,
                 'Vac': fmt_cont(a), 'Non': fmt_cont(b),
                 'p': fmt_p(p_cont(a, b)), 'SMD': fmt_smd(smd_cont(a, b))})


def add_bin(label, col):
    a_yes = int(v[col].sum())
    b_yes = int(nv[col].sum())
    rows.append({'block': 'CohortB_clearance', 'Variable': label,
                 'Vac': fmt_pct(a_yes, n_v), 'Non': fmt_pct(b_yes, n_nv),
                 'p': fmt_p(p_bin(a_yes, n_v, b_yes, n_nv)),
                 'SMD': fmt_smd(smd_bin(a_yes, n_v, b_yes, n_nv))})


def add_cat(prefix, col, levels):
    for lvl in levels:
        a_yes = (v[col] == lvl).sum()
        b_yes = (nv[col] == lvl).sum()
        rows.append({'block': 'CohortB_clearance', 'Variable': f'{prefix}: {lvl}',
                     'Vac': fmt_pct(a_yes, n_v), 'Non': fmt_pct(b_yes, n_nv),
                     'p': fmt_p(p_bin(a_yes, n_v, b_yes, n_nv)),
                     'SMD': fmt_smd(smd_bin(a_yes, n_v, b_yes, n_nv))})


add_cont('Age at index, years', 'index_age')
add_cont('BMI, kg/m²', 'closest_bmi')
add_bin('Seoul residence', 'is_seoul')
clr['fu_yrs'] = (clr['최종추적일자'] - clr['index_date']).dt.days / 365.25
v = clr[clr['접종여부'] == True]; nv = clr[clr['접종여부'] == False]
add_cont('Follow-up, years (from index)', 'fu_yrs')
add_cont('Surgery year', '수술연도')
for lvl, label in [('원추절제술', 'Conization'), ('자궁절제술', 'Hysterectomy')]:
    a_yes = (v['수술방법'].astype(str) == lvl).sum()
    b_yes = (nv['수술방법'].astype(str) == lvl).sum()
    rows.append({'block': 'CohortB_clearance', 'Variable': f'Surgery type: {label}',
                 'Vac': fmt_pct(a_yes, n_v), 'Non': fmt_pct(b_yes, n_nv),
                 'p': fmt_p(p_bin(a_yes, n_v, b_yes, n_nv)),
                 'SMD': fmt_smd(smd_bin(a_yes, n_v, b_yes, n_nv))})

# ------------------------------ patch unified CSV ------------------------------
unified = ROOT / 'Data' / 'Table1_BaselineCharacteristics_unified.csv'
with unified.open(encoding='utf-8-sig') as f:
    reader = list(csv.reader(f))

start = end = None
for i, r in enumerate(reader):
    if r and r[0] == 'CohortB_clearance':
        if start is None:
            start = i
        end = i

new_section = [[r['block'], r['Variable'], r['Vac'], r['Non'], r['p'], r['SMD']]
               for r in rows]

if start is not None:
    out = reader[:start] + new_section + reader[end + 1:]
    print(f'Replaced CohortB_clearance block (rows {start}..{end}) with {len(new_section)} v3 rows.')
else:
    out = reader + new_section
    print(f'Appended {len(new_section)} CohortB_clearance v3 rows.')

with unified.open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    for r in out:
        w.writerow(r)

print(f'Wrote {unified}')
