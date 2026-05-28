"""
Build canonical Cohort B primary cohort v3.

Primary definition:
  - Exposure: HPV vaccine, ≥ 2 doses (distinct prescription dates)
  - 3-month landmark: index → index + 90 days; symmetric across arms
  - Matched-set integrity: drop matched controls of any vaccinated case
    that fails the ≥2-dose threshold OR fails to survive 90 days OR has
    an outcome event before landmark.

Outputs: Data/primary_cohort_v3.csv  (one row per patient, with dose_count and
landmark eligibility annotations; outcome flags filled in by analyze_primary_v3.py)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RX_FILE = ROOT / 'Data' / '한국 HPV 코호트 자료를 이용한 자_처방정보.csv'
COH_FILE = ROOT / 'Data' / 'final_matched_outcomes.csv'
OUT_FILE = ROOT / 'Data' / 'primary_cohort_v3.csv'

LANDMARK_DAYS = 90
DOSE_THRESHOLD = 2

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
    rx = pd.read_csv(RX_FILE, encoding='cp949', low_memory=False)
    rx['is_vac'] = rx.apply(is_hpv_vaccine, axis=1)
    vac = rx[rx['is_vac']].copy()
    vac['처방일자_dt'] = pd.to_datetime(vac['처방일자'], format='%Y%m%d', errors='coerce')
    doses = (
        vac.dropna(subset=['처방일자_dt'])
        .groupby('연구번호')['처방일자_dt']
        .nunique()
        .rename('dose_count')
    )
    return doses


def main() -> pd.DataFrame:
    doses = compute_dose_counts()
    out = pd.read_csv(COH_FILE, encoding='utf-8-sig')
    out['index_date'] = pd.to_datetime(out['index_date'])
    out['최종추적일자'] = pd.to_datetime(out['최종추적일자'])
    out['recurrence_date'] = pd.to_datetime(out['recurrence_date'], errors='coerce')

    out = out.merge(doses, left_on='연구번호', right_index=True, how='left')
    out['dose_count'] = out['dose_count'].fillna(0).astype(int)

    # Step A — ≥2-dose threshold (matched-set integrity)
    vac_below = out[(out['접종여부'] == True) & (out['dose_count'] < DOSE_THRESHOLD)]
    bad_fids_dose = vac_below['fine_match_id'].dropna().unique()
    after_dose = out[~out['fine_match_id'].isin(bad_fids_dose)].copy()
    print(f'After ≥{DOSE_THRESHOLD}-dose filter: '
          f'n={len(after_dose)} ({(after_dose["접종여부"]).sum()}/{(~after_dose["접종여부"]).sum()})')

    # Step B — 3-month landmark FU eligibility (FU ≥ 90 d from index)
    after_dose['lm_zero'] = after_dose['index_date'] + pd.Timedelta(days=LANDMARK_DAYS)
    fu_days = (after_dose['최종추적일자'] - after_dose['index_date']).dt.days
    after_dose['lm_eligible_fu'] = fu_days >= LANDMARK_DAYS

    vac_fail = after_dose[(after_dose['접종여부'] == True) & (~after_dose['lm_eligible_fu'])]
    bad_fids_fu = vac_fail['fine_match_id'].dropna().unique()
    after_fu = after_dose[~after_dose['fine_match_id'].isin(bad_fids_fu)].copy()
    after_fu = after_fu[after_fu['lm_eligible_fu']].copy()
    print(f'After 3-mo landmark FU filter: '
          f'n={len(after_fu)} ({(after_fu["접종여부"]).sum()}/{(~after_fu["접종여부"]).sum()})')

    # Step C — exclude patients with event before landmark; preserve matched-set
    # integrity (drop the whole set if the vaccinated case has a pre-landmark event)
    rec_pre_lm = (after_fu['has_recurrence'] == True) & \
                 (after_fu['recurrence_date'] < after_fu['lm_zero'])
    vac_early = after_fu[(after_fu['접종여부'] == True) & rec_pre_lm]
    bad_fids_ev = vac_early['fine_match_id'].dropna().unique()
    primary = after_fu[~after_fu['fine_match_id'].isin(bad_fids_ev)].copy()
    # Drop individual non-vac with pre-landmark recurrence
    rec_pre_lm_p = (primary['has_recurrence'] == True) & \
                   (primary['recurrence_date'] < primary['lm_zero'])
    primary = primary[~rec_pre_lm_p].copy()
    print(f'After pre-landmark event removal (at-risk at landmark): '
          f'n={len(primary)} ({(primary["접종여부"]).sum()}/{(~primary["접종여부"]).sum()})')

    primary.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')

    print(f'\nWrote {OUT_FILE.relative_to(ROOT)}')
    print(
        f'Primary cohort: total={len(primary)}, '
        f'vaccinated={(primary["접종여부"]).sum()}, '
        f'non-vaccinated={(~primary["접종여부"]).sum()}'
    )
    return primary


if __name__ == '__main__':
    main()
