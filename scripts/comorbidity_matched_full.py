"""
전체 코호트에서 접종군 1:4 매칭 후 기저질환 비교

매칭 변수 (가능한 변수만):
- 생년월 (Index 시점 나이 ±5세)
- 성별 (전부 F)
- Index date 시점 관찰 가능 (사망/추적종료 이후 아님)

Index date 부여:
- 접종군: first_vaccine_date
- 비접종군: 매칭된 접종군의 first_vaccine_date를 index_date로 부여

비교: index_date 이후 신규 발생 기저질환 (5개 분류)
"""
import pandas as pd
import numpy as np
import openpyxl
from scipy.stats import fisher_exact

RANDOM_SEED = 42
MATCH_RATIO = 4  # 1:4
AGE_TOL_YEARS = 5

CLASS_LABELS = {
    '1': 'Angina/MI (협심증/심근경색)',
    '2': 'Hypertension (고혈압)',
    '3': 'Diabetes (당뇨)',
    '4': 'Stroke (뇌출혈/뇌경색)',
    '5': 'PE (폐색전증)',
}

rng = np.random.default_rng(RANDOM_SEED)

# 1) 코호트
cohort = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_코호트.csv',
                    encoding='cp949', low_memory=False)
cohort['birth_date'] = pd.to_datetime(cohort['생년월'].astype('Int64').astype(str),
                                     format='%Y%m%d', errors='coerce')
cohort['death_date'] = pd.to_datetime(cohort['사망일자'].astype('Int64').astype(str),
                                     format='%Y%m%d', errors='coerce')
cohort['last_follow'] = pd.to_datetime(cohort['최종추적일자'].astype('Int64').astype(str),
                                       format='%Y%m%d', errors='coerce')
print(f'Total cohort: {len(cohort)}')

# 2) 접종군 식별
rx = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_처방정보.csv',
                encoding='cp949', low_memory=False)
mask = (rx['처방명'].astype(str).str.contains('Gardasil|Cervarix|HPV vaccine', case=False, na=False) |
        rx['처방한글명'].astype(str).str.contains('가다실|서바릭스', na=False))
rx_vac = rx[mask].copy()
rx_vac['처방일자'] = pd.to_datetime(rx_vac['처방일자'].astype('Int64').astype(str),
                                  format='%Y%m%d', errors='coerce')
first_vac = rx_vac.groupby('연구번호')['처방일자'].min().reset_index()
first_vac.columns = ['연구번호','first_vaccine_date']
cohort = cohort.merge(first_vac, on='연구번호', how='left')

vaccinated = cohort[cohort['first_vaccine_date'].notna()].copy()
controls_pool = cohort[cohort['first_vaccine_date'].isna()].copy()
print(f'Vaccinated pool: {len(vaccinated)}, Control pool: {len(controls_pool)}')

# 생년월/사망/추적 결측 제거
vaccinated = vaccinated.dropna(subset=['birth_date'])
controls_pool = controls_pool.dropna(subset=['birth_date'])
print(f'After dropping no-DOB: vac={len(vaccinated)}, ctl={len(controls_pool)}')

# 3) 1:4 매칭 (greedy without replacement)
vaccinated = vaccinated.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
controls_pool = controls_pool.reset_index(drop=True)

controls_sorted = controls_pool.sort_values('birth_date').reset_index(drop=True)
used_ctl_idx = set()
TOL = pd.Timedelta(days=AGE_TOL_YEARS*365.25)

matched_pairs = []  # list of dicts
for _, vrow in vaccinated.iterrows():
    v_pid = vrow['연구번호']
    v_birth = vrow['birth_date']
    v_idx_date = vrow['first_vaccine_date']

    # candidate controls: |birth - v_birth| ≤ 5y AND alive at v_idx_date AND last_follow ≥ v_idx_date
    lo = v_birth - TOL
    hi = v_birth + TOL
    cand = controls_sorted[(controls_sorted['birth_date'] >= lo) &
                           (controls_sorted['birth_date'] <= hi)]
    alive = (cand['death_date'].isna()) | (cand['death_date'] > v_idx_date)
    observable = cand['last_follow'] >= v_idx_date
    cand = cand[alive & observable]
    cand = cand[~cand.index.isin(used_ctl_idx)]

    if len(cand) == 0:
        matched_pairs.append({'vac_pid': v_pid, 'idx_date': v_idx_date, 'matched_n': 0})
        continue

    n_take = min(MATCH_RATIO, len(cand))
    chosen = rng.choice(cand.index.values, size=n_take, replace=False)
    used_ctl_idx.update(chosen)
    for ci in chosen:
        c = cand.loc[ci]
        matched_pairs.append({
            'vac_pid': v_pid,
            'ctl_pid': c['연구번호'],
            'idx_date': v_idx_date,
            'v_birth': v_birth,
            'c_birth': c['birth_date'],
            'age_diff_years': abs((v_birth - c['birth_date']).days)/365.25,
        })

mp = pd.DataFrame(matched_pairs)
matched_with_ctl = mp[mp['ctl_pid'].notna()]
print(f'\nMatched: vaccinated with ≥1 control = {matched_with_ctl["vac_pid"].nunique()}')
print(f'Total matched controls = {len(matched_with_ctl)}')
print(f'Mean age diff: {matched_with_ctl["age_diff_years"].mean():.2f} years')
print(f'Match ratio distribution: {matched_with_ctl.groupby("vac_pid").size().value_counts().to_dict()}')

# 4) 분석용 long-format
vac_used = matched_with_ctl['vac_pid'].unique()
vac_df = vaccinated[vaccinated['연구번호'].isin(vac_used)][['연구번호','first_vaccine_date']].rename(
    columns={'연구번호':'pid','first_vaccine_date':'index_date'})
vac_df['vaccinated'] = True
ctl_df = matched_with_ctl[['ctl_pid','idx_date']].rename(
    columns={'ctl_pid':'pid','idx_date':'index_date'})
ctl_df['vaccinated'] = False
analysis = pd.concat([vac_df, ctl_df], ignore_index=True)
print(f'\nAnalysis cohort: vac={analysis["vaccinated"].sum()}, ctl={(~analysis["vaccinated"]).sum()}')

# 5) 기저질환
wb = openpyxl.load_workbook(
    'Data/한국 HPV 코호트 자료를 이용한 자_진단정보_기저질환추가_unlocked.xlsx',
    read_only=True, data_only=True)
ws = wb.active
recs = []
for row in ws.iter_rows(min_row=2, values_only=True):
    pid, cls, dd = row[0], row[5], row[8]
    if cls is None or str(cls).strip() == '': continue
    cls = str(cls).strip()
    if cls not in CLASS_LABELS: continue
    d = pd.to_datetime(str(dd), format='%Y%m%d', errors='coerce')
    recs.append((pid, cls, d))
como = pd.DataFrame(recs, columns=['pid','class','diag_date'])
first_diag = como.groupby(['pid','class'])['diag_date'].min().unstack('class')
df = analysis.merge(first_diag, left_on='pid', right_index=True, how='left')
for cls in CLASS_LABELS:
    if cls not in df.columns:
        df[cls] = pd.NaT

vac = df[df['vaccinated']].copy()
ctl = df[~df['vaccinated']].copy()

def fisher(a_yes, a_n, b_yes, b_n):
    odds, p = fisher_exact([[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]])
    return odds, p

# 6) Baseline (index 이전)
print(f'\n=== A) Baseline (index_date 이전 진단) — n_vac={len(vac)}, n_ctl={len(ctl)} ===')
print(f'{"분류":40s} {"접종군":>20s} {"비접종군":>20s} {"OR":>8s} {"p":>10s}')
results = []
for cls, label in CLASS_LABELS.items():
    v_yes = ((vac[cls].notna()) & (vac[cls] <= vac['index_date'])).sum()
    c_yes = ((ctl[cls].notna()) & (ctl[cls] <= ctl['index_date'])).sum()
    odds, p = fisher(v_yes, len(vac), c_yes, len(ctl))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:40s} {v_yes:5d} ({100*v_yes/len(vac):5.2f}%) {c_yes:6d} ({100*c_yes/len(ctl):5.2f}%) {odds:8.3f} {p:10.4g}{sig}')
    results.append({'period':'A_baseline','class':cls,'comorbidity':label,
        'vac_n':v_yes,'vac_total':len(vac),'vac_pct':round(100*v_yes/len(vac),3),
        'ctl_n':c_yes,'ctl_total':len(ctl),'ctl_pct':round(100*c_yes/len(ctl),3),
        'OR':round(odds,3) if not (np.isnan(odds) or np.isinf(odds)) else None,
        'p_value':round(p,5),'significant':p<0.05})

# 7) New onset (index 이후, baseline 진단자 제외)
print(f'\n=== B) New-onset (index_date 이후, baseline 제외) ===')
print(f'{"분류":40s} {"접종군":>20s} {"비접종군":>22s} {"OR":>8s} {"p":>10s}')
for cls, label in CLASS_LABELS.items():
    v_elig = vac[~((vac[cls].notna()) & (vac[cls] <= vac['index_date']))]
    c_elig = ctl[~((ctl[cls].notna()) & (ctl[cls] <= ctl['index_date']))]
    v_yes = ((v_elig[cls].notna()) & (v_elig[cls] > v_elig['index_date'])).sum()
    c_yes = ((c_elig[cls].notna()) & (c_elig[cls] > c_elig['index_date'])).sum()
    odds, p = fisher(v_yes, len(v_elig), c_yes, len(c_elig))
    sig = ' *' if p < 0.05 else ''
    print(f'{label:40s} {v_yes:4d}/{len(v_elig):5d} ({100*v_yes/len(v_elig):5.2f}%) {c_yes:5d}/{len(c_elig):5d} ({100*c_yes/len(c_elig):5.2f}%) {odds:8.3f} {p:10.4g}{sig}')
    results.append({'period':'B_new_onset','class':cls,'comorbidity':label,
        'vac_n':v_yes,'vac_total':len(v_elig),'vac_pct':round(100*v_yes/len(v_elig),3),
        'ctl_n':c_yes,'ctl_total':len(c_elig),'ctl_pct':round(100*c_yes/len(c_elig),3),
        'OR':round(odds,3) if not (np.isnan(odds) or np.isinf(odds)) else None,
        'p_value':round(p,5),'significant':p<0.05})

# 8) Composite
v_any_pre = (vac[list(CLASS_LABELS)].apply(
    lambda r: any(pd.notna(r[c]) and r[c] <= vac.loc[r.name,'index_date'] for c in CLASS_LABELS), axis=1)).sum()
c_any_pre = (ctl[list(CLASS_LABELS)].apply(
    lambda r: any(pd.notna(r[c]) and r[c] <= ctl.loc[r.name,'index_date'] for c in CLASS_LABELS), axis=1)).sum()
odds, p = fisher(v_any_pre, len(vac), c_any_pre, len(ctl))
print(f'\nComposite Baseline (any of 1-5):  vac {v_any_pre}/{len(vac)} ({100*v_any_pre/len(vac):.2f}%) vs ctl {c_any_pre}/{len(ctl)} ({100*c_any_pre/len(ctl):.2f}%)  OR={odds:.3f}, p={p:.4f}')

v_any_post = (vac[list(CLASS_LABELS)].apply(
    lambda r: any(pd.notna(r[c]) and r[c] > vac.loc[r.name,'index_date'] for c in CLASS_LABELS), axis=1)).sum()
c_any_post = (ctl[list(CLASS_LABELS)].apply(
    lambda r: any(pd.notna(r[c]) and r[c] > ctl.loc[r.name,'index_date'] for c in CLASS_LABELS), axis=1)).sum()
odds, p = fisher(v_any_post, len(vac), c_any_post, len(ctl))
print(f'Composite Post-index (any of 1-5): vac {v_any_post}/{len(vac)} ({100*v_any_post/len(vac):.2f}%) vs ctl {c_any_post}/{len(ctl)} ({100*c_any_post/len(ctl):.2f}%)  OR={odds:.3f}, p={p:.4f}')

# 9) 매칭 코호트 + 결과 저장
analysis.to_csv('Data/full_cohort_age_matched.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(results).to_csv('Data/comorbidity_age_matched.csv', index=False, encoding='utf-8-sig')
print('\nSaved:')
print('  Data/full_cohort_age_matched.csv (매칭 코호트)')
print('  Data/comorbidity_age_matched.csv (결과)')
