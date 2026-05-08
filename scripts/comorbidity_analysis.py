"""
기저질환 1-5 분류 × 백신 접종 여부 비교
- Baseline (index date 이전 진단)
- New onset (index date 이후 진단, 사전 진단자 제외)
- Fisher's exact test
"""
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import fisher_exact
import openpyxl

CLASS_LABELS = {
    '1': 'Angina/MI (협심증/심근경색)',
    '2': 'Hypertension (고혈압)',
    '3': 'Diabetes (당뇨)',
    '4': 'Stroke (뇌출혈/뇌경색)',
    '5': 'PE (폐색전증)',
}

# 1) 매칭 코호트
cohort = pd.read_csv('Data/final_matched_cohort.csv', encoding='utf-8-sig')
cohort['index_date'] = pd.to_datetime(cohort['index_date'])
cohort_ids = set(cohort['연구번호'])
print(f'Matched cohort: {len(cohort)} (vaccinated={cohort["접종여부"].sum()}, control={(~cohort["접종여부"].astype(bool)).sum()})')

# 2) 기저질환 xlsx 스트리밍
wb = openpyxl.load_workbook(
    'Data/한국 HPV 코호트 자료를 이용한 자_진단정보_기저질환추가_unlocked.xlsx',
    read_only=True, data_only=True)
ws = wb.active

records = []
for row in ws.iter_rows(min_row=2, values_only=True):
    pid = row[0]
    cls = row[5]
    diag_date = row[8]  # 진단일자 (YYYYMMDD)
    if pid not in cohort_ids: continue
    if cls is None or str(cls).strip() == '': continue
    cls = str(cls).strip()
    if cls not in CLASS_LABELS: continue
    try:
        d = pd.to_datetime(str(diag_date), format='%Y%m%d', errors='coerce')
    except Exception:
        d = pd.NaT
    records.append((pid, cls, d))

como = pd.DataFrame(records, columns=['연구번호','class','diag_date'])
print(f'Comorbidity rows in cohort: {len(como)}')

# 3) 환자별 각 class 최초 진단일
first_diag = como.groupby(['연구번호','class'], as_index=False)['diag_date'].min()
first_wide = first_diag.pivot(index='연구번호', columns='class', values='diag_date')

# 4) cohort에 merge (없는 class도 NaT 컬럼으로 채움)
df = cohort[['연구번호','접종여부','index_date']].merge(first_wide, on='연구번호', how='left')
for cls in CLASS_LABELS:
    if cls not in df.columns:
        df[cls] = pd.NaT

def fisher(a_yes, a_n, b_yes, b_n):
    table = [[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]]
    odds, p = fisher_exact(table)
    return odds, p

vac = df[df['접종여부'] == True]
ctl = df[df['접종여부'] == False]
print(f'\nVaccinated n={len(vac)}, Control n={len(ctl)}')

print('\n=== A) Baseline 기저질환 (index_date 이전 진단) ===')
print(f'{"분류":35s} {"접종군 n(%)":18s} {"비접종군 n(%)":18s} {"OR":>7s} {"p":>9s}')
for cls, label in CLASS_LABELS.items():
    if cls not in df.columns:
        continue
    vac_yes = ((vac[cls].notna()) & (vac[cls] <= vac['index_date'])).sum()
    ctl_yes = ((ctl[cls].notna()) & (ctl[cls] <= ctl['index_date'])).sum()
    odds, p = fisher(vac_yes, len(vac), ctl_yes, len(ctl))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:35s} {vac_yes:3d} ({100*vac_yes/len(vac):5.2f}%)  {ctl_yes:3d} ({100*ctl_yes/len(ctl):5.2f}%) {odds:7.3f} {p:9.4f}{sig}')

print('\n=== B) New onset 기저질환 (index_date 이후 진단, baseline 진단자 제외) ===')
print(f'{"분류":35s} {"접종군 n(%)":18s} {"비접종군 n(%)":18s} {"OR":>7s} {"p":>9s}')
for cls, label in CLASS_LABELS.items():
    if cls not in df.columns:
        continue
    # exclude pre-index diagnosis
    vac_eligible = vac[~((vac[cls].notna()) & (vac[cls] <= vac['index_date']))]
    ctl_eligible = ctl[~((ctl[cls].notna()) & (ctl[cls] <= ctl['index_date']))]
    vac_yes = ((vac_eligible[cls].notna()) & (vac_eligible[cls] > vac_eligible['index_date'])).sum()
    ctl_yes = ((ctl_eligible[cls].notna()) & (ctl_eligible[cls] > ctl_eligible['index_date'])).sum()
    odds, p = fisher(vac_yes, len(vac_eligible), ctl_yes, len(ctl_eligible))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:35s} {vac_yes:3d}/{len(vac_eligible):4d} ({100*vac_yes/len(vac_eligible):5.2f}%)  {ctl_yes:3d}/{len(ctl_eligible):4d} ({100*ctl_yes/len(ctl_eligible):5.2f}%) {odds:7.3f} {p:9.4f}{sig}')

print('\n=== D) Composite: 1-5 중 하나라도 (Baseline) ===')
vac_any = vac[list(CLASS_LABELS)].apply(lambda r: any((r[c] is not pd.NaT) and pd.notna(r[c]) and r[c] <= vac.loc[r.name,'index_date'] for c in CLASS_LABELS), axis=1).sum()
ctl_any = ctl[list(CLASS_LABELS)].apply(lambda r: any((r[c] is not pd.NaT) and pd.notna(r[c]) and r[c] <= ctl.loc[r.name,'index_date'] for c in CLASS_LABELS), axis=1).sum()
odds, p = fisher(vac_any, len(vac), ctl_any, len(ctl))
print(f'Any baseline comorbidity: {vac_any} ({100*vac_any/len(vac):.2f}%) vs {ctl_any} ({100*ctl_any/len(ctl):.2f}%)  OR={odds:.3f}  p={p:.4f}')

# Save results CSV
rows = []
for label_grp, vac_subset, ctl_subset, mode in [
    ('Baseline (pre-index)', vac, ctl, 'pre'),
    ('Any-time prevalence', vac, ctl, 'any'),
]:
    for cls, label in CLASS_LABELS.items():
        if mode == 'pre':
            v_yes = ((vac_subset[cls].notna()) & (vac_subset[cls] <= vac_subset['index_date'])).sum()
            c_yes = ((ctl_subset[cls].notna()) & (ctl_subset[cls] <= ctl_subset['index_date'])).sum()
        else:
            v_yes = vac_subset[cls].notna().sum()
            c_yes = ctl_subset[cls].notna().sum()
        odds, p = fisher(v_yes, len(vac_subset), c_yes, len(ctl_subset))
        rows.append({
            'period': label_grp,
            'class': cls,
            'comorbidity': label,
            'vaccinated_n': v_yes,
            'vaccinated_total': len(vac_subset),
            'vaccinated_pct': round(100*v_yes/len(vac_subset),2),
            'control_n': c_yes,
            'control_total': len(ctl_subset),
            'control_pct': round(100*c_yes/len(ctl_subset),2),
            'OR': round(odds,3) if not np.isnan(odds) and not np.isinf(odds) else None,
            'p_value': round(p,4),
            'significant': p < 0.05,
        })
pd.DataFrame(rows).to_csv('Data/comorbidity_by_vaccine.csv', index=False, encoding='utf-8-sig')
print('\nSaved: Data/comorbidity_by_vaccine.csv')

print('\n=== C) Any-time prevalence (모든 진단, 시점 무관) ===')
print(f'{"분류":35s} {"접종군 n(%)":18s} {"비접종군 n(%)":18s} {"OR":>7s} {"p":>9s}')
for cls, label in CLASS_LABELS.items():
    if cls not in df.columns:
        continue
    vac_yes = vac[cls].notna().sum()
    ctl_yes = ctl[cls].notna().sum()
    odds, p = fisher(vac_yes, len(vac), ctl_yes, len(ctl))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:35s} {vac_yes:3d} ({100*vac_yes/len(vac):5.2f}%)  {ctl_yes:3d} ({100*ctl_yes/len(ctl):5.2f}%) {odds:7.3f} {p:9.4f}{sig}')
