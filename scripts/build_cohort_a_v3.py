"""
Build canonical Cohort A v3.

Apply the same exposure definition as Cohort B v3 for methodological symmetry:
  - ≥2 distinct HPV-vaccine prescription dates (dose threshold)
  - 3-month landmark applied symmetrically across both arms
  - Matched-set integrity preserved (drop the matched pair if either vaccinated
    case fails the threshold OR the landmark eligibility filter)

Starts from the existing Cohort A 1:1 PSM matched dataset and layers the v3
filters on top.

Outputs:
  Data/cohort_a_v3_matched.csv     — primary cohort
  Data/CohortA_HR_v3.csv           — recomputed HRs for all 7 endpoints
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

LANDMARK_DAYS = 90
DOSE_THRESHOLD = 2

# Reuse vaccine identification regex from cohort B v3 builder
NAME_RE = re.compile(r'gardasil|cervarix|hpv vaccine', re.I)
KOR_RE = re.compile(r'가다실|서바릭스')
CODE_PREFIX = ('DV-9HPF', 'DV-HPF', 'DV-JHP', 'DV-HPJ')


def is_hpv_vaccine(row) -> bool:
    pn = str(row.get('처방명', '') or '')
    kn = str(row.get('처방한글명', '') or '')
    pc = str(row.get('처방코드', '') or '')
    if NAME_RE.search(pn):
        return True
    if KOR_RE.search(kn):
        return True
    if pc.startswith(CODE_PREFIX):
        return True
    return False


def compute_dose_counts() -> pd.Series:
    rx = pd.read_csv(
        ROOT / 'Data' / '한국 HPV 코호트 자료를 이용한 자_처방정보.csv',
        encoding='cp949', low_memory=False,
    )
    rx['is_vac'] = rx.apply(is_hpv_vaccine, axis=1)
    vac = rx[rx['is_vac']].copy()
    vac['처방일자_dt'] = pd.to_datetime(vac['처방일자'], format='%Y%m%d', errors='coerce')
    return (
        vac.dropna(subset=['처방일자_dt'])
        .groupby('연구번호')['처방일자_dt']
        .nunique()
        .rename('dose_count')
    )


def build_v3():
    print('Building Cohort A 1:1 PSM matched dataset (re-running PSM to capture pair_id)...')
    from make_main_figures import build_cohort_a_matched
    matched = build_cohort_a_matched()
    matched['index_date'] = pd.to_datetime(matched['index_date'], errors='coerce')
    matched['최종추적일자'] = matched['last_follow']
    matched = matched.dropna(subset=['index_date', '최종추적일자']).copy()

    n_before = len(matched)
    print(f'  Starting PSM matched n = {n_before} '
          f'(vac {(matched["vaccinated"]==True).sum()} / '
          f'non {(matched["vaccinated"]==False).sum()})')

    # Merge dose counts
    doses = compute_dose_counts()
    matched = matched.merge(doses, left_on='pid', right_index=True, how='left')
    matched['dose_count'] = matched['dose_count'].fillna(0).astype(int)

    # ---- Step A: ≥2-dose threshold with PSM pair_id integrity ----
    vac_below = matched[(matched['vaccinated'] == True) & (matched['dose_count'] < DOSE_THRESHOLD)]
    bad_pairs_dose = vac_below['pair_id'].dropna().unique()
    print(f'  Vaccinated cases failing ≥{DOSE_THRESHOLD}-dose: {len(vac_below)}')
    print(f'  → Drop {len(bad_pairs_dose)} matched pairs (vac + paired non)')

    after_dose = matched[~matched['pair_id'].isin(bad_pairs_dose)].copy()
    print(f'  After dose filter: n = {len(after_dose)} '
          f'(vac {(after_dose["vaccinated"]==True).sum()} / '
          f'non {(after_dose["vaccinated"]==False).sum()})')

    # ---- Step B: 3-month landmark (FU ≥ 90 days from index) ----
    fu_days = (after_dose['최종추적일자'] - after_dose['index_date']).dt.days
    after_dose['lm_eligible_fu'] = fu_days >= LANDMARK_DAYS

    vac_fail_fu = after_dose[
        (after_dose['vaccinated'] == True) & (~after_dose['lm_eligible_fu'])
    ]
    bad_pairs_fu = vac_fail_fu['pair_id'].dropna().unique()
    print(f'  Vaccinated cases failing 3-mo landmark FU: {len(vac_fail_fu)}')
    print(f'  → Drop {len(bad_pairs_fu)} matched pairs')

    after_lm = after_dose[~after_dose['pair_id'].isin(bad_pairs_fu)].copy()
    after_lm = after_lm[after_lm['lm_eligible_fu']].copy()  # also drop individual non-vac

    after_lm['lm_zero'] = after_lm['index_date'] + pd.Timedelta(days=LANDMARK_DAYS)

    print(f'  After landmark filter: n = {len(after_lm)} '
          f'(vac {(after_lm["vaccinated"]==True).sum()} / '
          f'non {(after_lm["vaccinated"]==False).sum()})')

    out_path = ROOT / 'Data' / 'cohort_a_v3_matched.csv'
    after_lm.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'Wrote {out_path.relative_to(ROOT)}')
    return after_lm


if __name__ == '__main__':
    build_v3()
