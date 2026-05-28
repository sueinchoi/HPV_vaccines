"""Build Table 1A full baseline for Cohort A v3 primary cohort.

Adds the standard demographic + anthropometric + smoking + comorbidity +
follow-up variables that were present in the legacy Cohort A baseline,
computed on the v3 primary cohort (n = 2,776).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

CLASS_LABELS = {'1': 'Angina/MI', '2': 'Hypertension', '3': 'Diabetes',
                '4': 'Stroke', '5': 'PE'}
SMOKE_MAP = {'비흡연': 'Never', '과거흡연': 'Former',
             '현재흡연': 'Current', '확인불능': 'Unknown'}


def smd_cont(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    p = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / p if p > 0 else np.nan


def smd_bin(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    pp1, pp2 = p1 / n1, p2 / n2
    p = np.sqrt((pp1 * (1 - pp1) + pp2 * (1 - pp2)) / 2)
    return (pp1 - pp2) / p if p > 0 else np.nan


def fmt_cont(s):
    s = pd.Series(s).dropna()
    return '-' if len(s) == 0 else f'{s.mean():.2f} ± {s.std():.2f}'


def fmt_smd(s):
    return '-' if pd.isna(s) else f'{abs(s):.3f}'


def fmt_p(p):
    if pd.isna(p):
        return '-'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def p_cont(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return ttest_ind(a, b, equal_var=False).pvalue


def p_bin(a_yes, a_n, b_yes, b_n):
    if a_n == 0 or b_n == 0:
        return np.nan
    return fisher_exact([[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]])[1]


print('Loading Cohort A v3 matched cohort...')
df = pd.read_csv(ROOT / 'Data' / 'cohort_a_v3_matched.csv', encoding='utf-8-sig')
df['vaccinated'] = df['vaccinated'].astype(bool)
df['index_date'] = pd.to_datetime(df['index_date'], errors='coerce')
df['최종추적일자'] = pd.to_datetime(df['최종추적일자'], errors='coerce')
df['death_date'] = pd.to_datetime(df['death_date'], errors='coerce')
df['fu_days'] = (df['최종추적일자'] - df['index_date']).dt.days
print(f'n = {len(df)} (vac {(df["vaccinated"]).sum()} / non {(~df["vaccinated"]).sum()})')

# Birth year
df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
df['birth_year'] = df['birth_date'].dt.year
df['index_year'] = df['index_date'].dt.year

# Closest clinical info (already computed in matched dataset)
# Columns: height, weight, sbp, dbp, bmi, smoke
# Verify columns are present
for c in ['height', 'weight', 'bmi', 'sbp', 'dbp', 'smoke']:
    if c not in df.columns:
        print(f'WARNING: {c} not in matched dataset, using NaN')
        df[c] = np.nan

# Pre-existing comorbidities (before index)
print('Computing pre-index comorbidities...')
diag = pd.read_excel(
    ROOT / 'Data' / '한국 HPV 코호트 자료를 이용한 자_진단정보_기저질환추가_unlocked.xlsx'
)
diag['진단일자'] = pd.to_datetime(diag['진단일자'].astype(str), format='%Y%m%d', errors='coerce')
diag_col = [c for c in diag.columns if c.startswith('기저질환분류')][0]
diag['cls'] = diag[diag_col].astype(str).str.strip()


def has_pre_comor(pid, idx_dt, cls):
    sub = diag[(diag['연구번호'] == pid) & (diag['cls'] == cls) & (diag['진단일자'] <= idx_dt)]
    return int(len(sub) > 0)


for cls, lbl in CLASS_LABELS.items():
    df[f'comor_{lbl}'] = df.apply(
        lambda r: has_pre_comor(r['pid'], r['index_date'], cls), axis=1
    )
df['comor_any'] = df[[f'comor_{lbl}' for lbl in CLASS_LABELS.values()]].max(axis=1)

# Deaths
df['died'] = ((df['death_date'].notna()) & (df['death_date'] <= df['최종추적일자'])).astype(int)

v = df[df['vaccinated']]
nv = df[~df['vaccinated']]
n_v, n_nv = len(v), len(nv)
print(f'Building rows for n_v={n_v}, n_nv={n_nv}')

rows = []


def add_section(label):
    rows.append(['CohortA_post_v3', label, '', '', '', ''])


def add_cont(name, col):
    a, b = v[col], nv[col]
    rows.append([
        'CohortA_post_v3', f'  {name}',
        fmt_cont(a), fmt_cont(b),
        fmt_p(p_cont(a, b)), fmt_smd(smd_cont(a, b)),
    ])


def add_bin(name, col):
    a_yes = int(v[col].sum())
    b_yes = int(nv[col].sum())
    rows.append([
        'CohortA_post_v3', f'  {name}',
        f'{a_yes} ({100 * a_yes / n_v:.1f}%)',
        f'{b_yes} ({100 * b_yes / n_nv:.1f}%)',
        fmt_p(p_bin(a_yes, n_v, b_yes, n_nv)),
        fmt_smd(smd_bin(a_yes, n_v, b_yes, n_nv)),
    ])


def add_cat(name, col, levels):
    rows.append(['CohortA_post_v3', name, '', '', '', ''])
    for lvl in levels:
        a_yes = int((v[col] == lvl).sum())
        b_yes = int((nv[col] == lvl).sum())
        rows.append([
            'CohortA_post_v3', f'  {lvl}',
            f'{a_yes} ({100 * a_yes / n_v:.1f}%)',
            f'{b_yes} ({100 * b_yes / n_nv:.1f}%)',
            fmt_p(p_bin(a_yes, n_v, b_yes, n_nv)),
            fmt_smd(smd_bin(a_yes, n_v, b_yes, n_nv)),
        ])


add_section('Demographics')
add_cont('Age at index, years', 'age_at_index')
add_cont('Birth year', 'birth_year')
add_cont('Index year', 'index_year')
add_bin('Residence in Seoul', 'is_seoul')

add_section('Anthropometry & vital signs (closest to index, ±1 yr)')
add_cont('Height, cm', 'height')
add_cont('Weight, kg', 'weight')
add_cont('BMI, kg/m²', 'bmi')
add_cont('Systolic BP, mmHg', 'sbp')
add_cont('Diastolic BP, mmHg', 'dbp')

add_cat('Smoking status', 'smoke', ['Never', 'Former', 'Current', 'Unknown'])

add_section('Pre-existing comorbidities (before index)')
for lbl in CLASS_LABELS.values():
    add_bin(lbl, f'comor_{lbl}')
add_bin('Composite (any of 5)', 'comor_any')

add_section('Follow-up')
add_cont('Follow-up, days', 'fu_days')
add_bin('Mortality during follow-up', 'died')

# Write back to unified CSV (replace any existing CohortA_post_v3 rows)
unified = ROOT / 'Data' / 'Table1_BaselineCharacteristics_unified.csv'
with open(unified, encoding='utf-8-sig') as f:
    existing = list(csv.reader(f))
header = existing[0]
existing_data = [r for r in existing[1:] if r and r[0] != 'CohortA_post_v3']
with open(unified, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(existing_data)
    w.writerows(rows)
print(f'Wrote {len(rows)} CohortA_post_v3 rows to {unified.name}')
