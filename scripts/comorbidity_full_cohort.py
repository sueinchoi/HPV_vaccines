"""
전체 환자 (매칭 X) 대상 백신 접종군 vs 비접종군 기저질환 비교

기준:
- 전체 코호트: 32,969명 (한국 HPV 코호트 전체)
- 접종군: 처방정보에서 Gardasil/Gardasil9/Cervarix 처방 받은 환자 = 2,156명
- 비접종군: 나머지 = 30,813명
- 기저질환: 진단정보_기저질환추가 1-5 분류
- Baseline 기준일: 첫 백신 접종일 (접종군) / 코호트 등록 시점 적당한 기준 없음 → "any-time prevalence"로 비교
"""
import pandas as pd
import numpy as np
import openpyxl
from scipy.stats import fisher_exact

CLASS_LABELS = {
    '1': 'Angina/MI (협심증/심근경색)',
    '2': 'Hypertension (고혈압)',
    '3': 'Diabetes (당뇨)',
    '4': 'Stroke (뇌출혈/뇌경색)',
    '5': 'PE (폐색전증)',
}

# 1) 전체 코호트
cohort = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_코호트.csv',
                    encoding='cp949', low_memory=False)
print(f'Total cohort: {len(cohort)}')

# 2) 처방정보에서 백신 접종 환자 식별
rx = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_처방정보.csv',
                encoding='cp949', low_memory=False)
vaccine_mask = (
    rx['처방명'].astype(str).str.contains('Gardasil|Cervarix|HPV vaccine', case=False, na=False) |
    rx['처방한글명'].astype(str).str.contains('가다실|서바릭스', na=False)
)
rx_vac = rx[vaccine_mask].copy()
rx_vac['처방일자'] = pd.to_datetime(rx_vac['처방일자'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
first_vac_date = rx_vac.groupby('연구번호')['처방일자'].min().reset_index()
first_vac_date.columns = ['연구번호', 'first_vaccine_date']
print(f'Vaccinated patients: {len(first_vac_date)}')

cohort = cohort.merge(first_vac_date, on='연구번호', how='left')
cohort['vaccinated'] = cohort['first_vaccine_date'].notna()
print(f'  Vaccinated: {cohort["vaccinated"].sum()}, Control: {(~cohort["vaccinated"]).sum()}')

# 3) 기저질환 추출
wb = openpyxl.load_workbook(
    'Data/한국 HPV 코호트 자료를 이용한 자_진단정보_기저질환추가_unlocked.xlsx',
    read_only=True, data_only=True)
ws = wb.active
records = []
for row in ws.iter_rows(min_row=2, values_only=True):
    pid, cls, diag_date = row[0], row[5], row[8]
    if cls is None or str(cls).strip() == '': continue
    cls = str(cls).strip()
    if cls not in CLASS_LABELS: continue
    d = pd.to_datetime(str(diag_date), format='%Y%m%d', errors='coerce')
    records.append((pid, cls, d))
como = pd.DataFrame(records, columns=['연구번호','class','diag_date'])
first_diag = como.groupby(['연구번호','class'])['diag_date'].min().unstack('class')
print(f'Comorbidity records (cohort 전체): {len(records)}, unique pids: {first_diag.shape[0]}')

# 4) Merge & analyze
df = cohort[['연구번호','vaccinated','first_vaccine_date']].merge(
    first_diag, on='연구번호', how='left')
for cls in CLASS_LABELS:
    if cls not in df.columns:
        df[cls] = pd.NaT

vac = df[df['vaccinated']].copy()
ctl = df[~df['vaccinated']].copy()
print(f'\nFinal: vaccinated={len(vac)}, control={len(ctl)}')

def fisher(a_yes, a_n, b_yes, b_n):
    odds, p = fisher_exact([[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]])
    return odds, p

# A) Any-time prevalence
print('\n=== A) Any-time prevalence (시점 무관) ===')
print(f'{"분류":40s} {"접종군":>20s} {"비접종군":>20s} {"OR":>8s} {"p":>10s}')
results = []
for cls, label in CLASS_LABELS.items():
    v_yes = vac[cls].notna().sum()
    c_yes = ctl[cls].notna().sum()
    odds, p = fisher(v_yes, len(vac), c_yes, len(ctl))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:40s} {v_yes:5d} ({100*v_yes/len(vac):5.2f}%) {c_yes:6d} ({100*c_yes/len(ctl):5.2f}%) {odds:8.3f} {p:10.4g}{sig}')
    results.append({'period':'any-time','class':cls,'comorbidity':label,
                    'vac_n':v_yes,'vac_total':len(vac),'vac_pct':round(100*v_yes/len(vac),3),
                    'ctl_n':c_yes,'ctl_total':len(ctl),'ctl_pct':round(100*c_yes/len(ctl),3),
                    'OR':round(odds,3) if not (np.isnan(odds) or np.isinf(odds)) else None,
                    'p_value':round(p,5),'significant':p<0.05})

# B) Baseline (접종군: 백신 접종일 이전 / 비접종군: 기준일 모호 → 사용자 안내 위해 출생 후 모든 진단)
# More fair: compare diagnosis BEFORE first vaccine date for vaccinated. For control, use full history.
# For symmetry, compare PREVALENCE at any time = (A). Skip B.

# C) New onset after first vaccine date (접종군), 비접종군 비교 baseline 모호 → 코호트 등록 시점 통계는 정확히 안 됨
# For full cohort, focus on (A) any-time prevalence as primary unmatched comparison

# Composite
v_any = (vac[list(CLASS_LABELS)].notna().any(axis=1)).sum()
c_any = (ctl[list(CLASS_LABELS)].notna().any(axis=1)).sum()
odds, p = fisher(v_any, len(vac), c_any, len(ctl))
sig = ' *' if p < 0.05 else ''
print(f'{"Composite (any of 1-5)":40s} {v_any:5d} ({100*v_any/len(vac):5.2f}%) {c_any:6d} ({100*c_any/len(ctl):5.2f}%) {odds:8.3f} {p:10.4g}{sig}')
results.append({'period':'any-time','class':'composite','comorbidity':'Any of 1-5',
                'vac_n':v_any,'vac_total':len(vac),'vac_pct':round(100*v_any/len(vac),3),
                'ctl_n':c_any,'ctl_total':len(ctl),'ctl_pct':round(100*c_any/len(ctl),3),
                'OR':round(odds,3) if not (np.isnan(odds) or np.isinf(odds)) else None,
                'p_value':round(p,5),'significant':p<0.05})

pd.DataFrame(results).to_csv('Data/comorbidity_full_cohort_unmatched.csv', index=False, encoding='utf-8-sig')
print('\nSaved: Data/comorbidity_full_cohort_unmatched.csv')

# D) 기준 분리: 접종군 = 백신접종일 이전 진단, 비접종군 = 임의 시점 진단 → 시점 mismatch 우려
# Fair pre-vaccine baseline: 접종군은 first_vaccine_date 이전 진단만, 비접종군은 전체기간 (더 많은 기회)
print('\n=== B) 백신 이전 진단 vs 비접종군 전체기간 (시점 비대칭, 참고용) ===')
print(f'{"분류":40s} {"접종군(백신前)":>20s} {"비접종군":>20s} {"OR":>8s} {"p":>10s}')
results_b = []
for cls, label in CLASS_LABELS.items():
    v_yes = ((vac[cls].notna()) & (vac[cls] <= vac['first_vaccine_date'])).sum()
    c_yes = ctl[cls].notna().sum()
    odds, p = fisher(v_yes, len(vac), c_yes, len(ctl))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:40s} {v_yes:5d} ({100*v_yes/len(vac):5.2f}%) {c_yes:6d} ({100*c_yes/len(ctl):5.2f}%) {odds:8.3f} {p:10.4g}{sig}')
    results_b.append({'period':'pre-vaccine_vs_anytime','class':cls,'comorbidity':label,
                    'vac_n':v_yes,'vac_total':len(vac),'vac_pct':round(100*v_yes/len(vac),3),
                    'ctl_n':c_yes,'ctl_total':len(ctl),'ctl_pct':round(100*c_yes/len(ctl),3),
                    'OR':round(odds,3) if not (np.isnan(odds) or np.isinf(odds)) else None,
                    'p_value':round(p,5),'significant':p<0.05})

pd.DataFrame(results + results_b).to_csv('Data/comorbidity_full_cohort_unmatched.csv', index=False, encoding='utf-8-sig')

# E) Post-vaccine 분석
# 접종군: first_vaccine_date 이후 진단
# 비접종군 baseline 부재 → 두 가지 방식
#   (E1) 비대칭: 비접종군 전체기간 진단 (관찰 시간 더 길어 OR 보수적으로 작게 추정)
#   (E2) 대칭화: 접종군의 백신일 분포에서 무작위 pseudo-index 부여 후 그 이후 진단
print('\n=== E1) Post-vaccine 진단 vs 비접종군 전체기간 (시점 비대칭, 참고용) ===')
print(f'{"분류":40s} {"접종군(백신後)":>20s} {"비접종군":>20s} {"OR":>8s} {"p":>10s}')
results_e = []
for cls, label in CLASS_LABELS.items():
    v_yes = ((vac[cls].notna()) & (vac[cls] > vac['first_vaccine_date'])).sum()
    c_yes = ctl[cls].notna().sum()
    odds, p = fisher(v_yes, len(vac), c_yes, len(ctl))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:40s} {v_yes:5d} ({100*v_yes/len(vac):5.2f}%) {c_yes:6d} ({100*c_yes/len(ctl):5.2f}%) {odds:8.3f} {p:10.4g}{sig}')
    results_e.append({'period':'post-vaccine_vs_anytime','class':cls,'comorbidity':label,
                    'vac_n':v_yes,'vac_total':len(vac),'vac_pct':round(100*v_yes/len(vac),3),
                    'ctl_n':c_yes,'ctl_total':len(ctl),'ctl_pct':round(100*c_yes/len(ctl),3),
                    'OR':round(odds,3) if not (np.isnan(odds) or np.isinf(odds)) else None,
                    'p_value':round(p,5),'significant':p<0.05})

# E2) Pseudo-index for control (random sample from vac date distribution)
np.random.seed(42)
vac_dates = vac['first_vaccine_date'].dropna().values
ctl = ctl.copy()
ctl['pseudo_index'] = pd.to_datetime(np.random.choice(vac_dates, size=len(ctl)))
print('\n=== E2) Post-index 진단 (대칭화: 비접종군에 백신일 분포에서 pseudo-index 부여) ===')
print(f'{"분류":40s} {"접종군(백신後)":>20s} {"비접종군(pseudo後)":>22s} {"OR":>8s} {"p":>10s}')
for cls, label in CLASS_LABELS.items():
    v_yes = ((vac[cls].notna()) & (vac[cls] > vac['first_vaccine_date'])).sum()
    c_yes = ((ctl[cls].notna()) & (ctl[cls] > ctl['pseudo_index'])).sum()
    odds, p = fisher(v_yes, len(vac), c_yes, len(ctl))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:40s} {v_yes:5d} ({100*v_yes/len(vac):5.2f}%) {c_yes:6d} ({100*c_yes/len(ctl):5.2f}%) {odds:8.3f} {p:10.4g}{sig}')
    results_e.append({'period':'post-vaccine_vs_pseudo-post','class':cls,'comorbidity':label,
                    'vac_n':v_yes,'vac_total':len(vac),'vac_pct':round(100*v_yes/len(vac),3),
                    'ctl_n':c_yes,'ctl_total':len(ctl),'ctl_pct':round(100*c_yes/len(ctl),3),
                    'OR':round(odds,3) if not (np.isnan(odds) or np.isinf(odds)) else None,
                    'p_value':round(p,5),'significant':p<0.05})

# E3) Pre-existing 진단자 제외 후 신규 발생만 (incident analysis, pseudo-index symmetric)
print('\n=== E3) New-onset only (baseline 진단자 제외, 대칭 pseudo-index) ===')
print(f'{"분류":40s} {"접종군 신규":>20s} {"비접종군 신규":>22s} {"OR":>8s} {"p":>10s}')
for cls, label in CLASS_LABELS.items():
    # vaccinated: exclude pre-vaccine cases
    v_eligible = vac[~((vac[cls].notna()) & (vac[cls] <= vac['first_vaccine_date']))]
    v_yes = ((v_eligible[cls].notna()) & (v_eligible[cls] > v_eligible['first_vaccine_date'])).sum()
    # control: exclude pre-pseudo cases
    c_eligible = ctl[~((ctl[cls].notna()) & (ctl[cls] <= ctl['pseudo_index']))]
    c_yes = ((c_eligible[cls].notna()) & (c_eligible[cls] > c_eligible['pseudo_index'])).sum()
    odds, p = fisher(v_yes, len(v_eligible), c_yes, len(c_eligible))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:40s} {v_yes:5d}/{len(v_eligible)} ({100*v_yes/len(v_eligible):5.2f}%) {c_yes:6d}/{len(c_eligible)} ({100*c_yes/len(c_eligible):5.2f}%) {odds:8.3f} {p:10.4g}{sig}')
    results_e.append({'period':'new-onset_post-index','class':cls,'comorbidity':label,
                    'vac_n':v_yes,'vac_total':len(v_eligible),'vac_pct':round(100*v_yes/len(v_eligible),3),
                    'ctl_n':c_yes,'ctl_total':len(c_eligible),'ctl_pct':round(100*c_yes/len(c_eligible),3),
                    'OR':round(odds,3) if not (np.isnan(odds) or np.isinf(odds)) else None,
                    'p_value':round(p,5),'significant':p<0.05})

pd.DataFrame(results + results_b + results_e).to_csv('Data/comorbidity_full_cohort_unmatched.csv', index=False, encoding='utf-8-sig')
print('\nSaved: Data/comorbidity_full_cohort_unmatched.csv (with post-vaccine sections)')
