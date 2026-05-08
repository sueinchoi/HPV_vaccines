"""
연령 매칭 코호트 (full_cohort_age_matched.csv)의 SMD 점검

- Before matching: 전체 접종군(2,156) vs 전체 비접종군(30,813)
- After matching: 접종군(2,155) vs 매칭된 비접종군(8,620)
- |SMD| < 0.1 = 잘 균형
"""
import pandas as pd
import numpy as np

# 1) 전체 코호트 + 백신 정보
cohort = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_코호트.csv',
                    encoding='cp949', low_memory=False)
cohort['birth_date'] = pd.to_datetime(cohort['생년월'].astype('Int64').astype(str),
                                     format='%Y%m%d', errors='coerce')
cohort['death_date'] = pd.to_datetime(cohort['사망일자'].astype('Int64').astype(str),
                                     format='%Y%m%d', errors='coerce')
cohort['last_follow'] = pd.to_datetime(cohort['최종추적일자'].astype('Int64').astype(str),
                                       format='%Y%m%d', errors='coerce')

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
cohort = cohort.dropna(subset=['birth_date'])

# 2) 매칭 코호트 로드 → 환자별 index_date
matched = pd.read_csv('Data/full_cohort_age_matched.csv', encoding='utf-8-sig')
matched['index_date'] = pd.to_datetime(matched['index_date'])
print(f'Matched: vac={matched["vaccinated"].sum()}, ctl={(~matched["vaccinated"]).sum()}')

# 3) 매칭 코호트에 인구학 변수 결합
m = matched.merge(cohort[['연구번호','birth_date','death_date','last_follow','성별','first_vaccine_date']],
                  left_on='pid', right_on='연구번호', how='left')
m['age_at_index'] = (m['index_date'] - m['birth_date']).dt.days / 365.25
m['birth_year'] = m['birth_date'].dt.year
m['index_year'] = m['index_date'].dt.year
m['followup_days'] = (m['last_follow'] - m['index_date']).dt.days
m['died'] = m['death_date'].notna() & (m['death_date'] >= m['index_date'])
m['female'] = (m['성별'] == 'F').astype(int)

# 4) Before matching: 매칭 안 한 전체 접종군 vs 전체 비접종군
# 비접종군 pseudo-index를 줄 수 없으므로, 인덱스 무관 변수만 비교
before_vac = cohort[cohort['first_vaccine_date'].notna()].copy()
before_ctl = cohort[cohort['first_vaccine_date'].isna()].copy()
before_vac['birth_year'] = before_vac['birth_date'].dt.year
before_ctl['birth_year'] = before_ctl['birth_date'].dt.year
before_vac['female'] = (before_vac['성별'] == 'F').astype(int)
before_ctl['female'] = (before_ctl['성별'] == 'F').astype(int)
# 접종군의 백신일 시점 나이 = (vac_date - birth) / 365.25 (only for vaccinated)
before_vac['age_at_vacdate'] = (before_vac['first_vaccine_date'] - before_vac['birth_date']).dt.days/365.25
# 비접종군은 last_follow 시점 나이로 대체
before_ctl['age_at_lastfollow'] = (before_ctl['last_follow'] - before_ctl['birth_date']).dt.days/365.25

def smd_cont(a, b):
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt((v1 + v2) / 2)
    return (m1 - m2) / pooled if pooled > 0 else np.nan

def smd_bin(a, b):
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    p1, p2 = a.mean(), b.mean()
    pooled = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return (p1 - p2) / pooled if pooled > 0 else np.nan

print('\n=== Before matching: 전체 접종군(2156) vs 전체 비접종군(30813) ===')
print(f'{"variable":30s} {"vac (mean/%)":>15s} {"ctl (mean/%)":>15s} {"|SMD|":>8s}')
print(f'{"birth_year":30s} {before_vac["birth_year"].mean():>15.1f} {before_ctl["birth_year"].mean():>15.1f} {abs(smd_cont(before_vac["birth_year"], before_ctl["birth_year"])):>8.3f}')
print(f'{"female (%)":30s} {100*before_vac["female"].mean():>14.1f}% {100*before_ctl["female"].mean():>14.1f}% {abs(smd_bin(before_vac["female"], before_ctl["female"])):>8.3f}')
print('  (참고) 접종군의 백신일 시점 나이 vs 비접종군의 추적종료 시점 나이 — 시점 다름:')
print(f'    age @ ref     vac={before_vac["age_at_vacdate"].mean():.2f}, ctl={before_ctl["age_at_lastfollow"].mean():.2f}')

print('\n=== After matching: 접종군(2155) vs 매칭 비접종군(8620) (모두 index_date 시점) ===')
mvac = m[m['vaccinated']]
mctl = m[~m['vaccinated']]
rows = []
def report(name, ser_v, ser_c, kind='cont'):
    if kind == 'cont':
        s = smd_cont(ser_v, ser_c)
        v_str = f'{ser_v.mean():.2f}±{ser_v.std():.2f}'
        c_str = f'{ser_c.mean():.2f}±{ser_c.std():.2f}'
    else:
        s = smd_bin(ser_v, ser_c)
        v_str = f'{100*ser_v.mean():.1f}%'
        c_str = f'{100*ser_c.mean():.1f}%'
    flag = ' ✓' if abs(s) < 0.1 else (' ⚠' if abs(s) < 0.25 else ' ✗')
    print(f'{name:30s} {v_str:>20s} {c_str:>20s} |SMD|={abs(s):>6.3f}{flag}')
    rows.append({'variable':name,'vac':v_str,'ctl':c_str,'SMD':round(s,4),'abs_SMD':round(abs(s),4)})

print(f'{"variable":30s} {"vac (mean±SD/%)":>20s} {"ctl (mean±SD/%)":>20s}  {"|SMD|":>8s}')
report('age_at_index (years)', mvac['age_at_index'], mctl['age_at_index'], 'cont')
report('birth_year', mvac['birth_year'], mctl['birth_year'], 'cont')
report('index_year', mvac['index_year'], mctl['index_year'], 'cont')
report('followup_days', mvac['followup_days'], mctl['followup_days'], 'cont')
report('female (binary)', mvac['female'], mctl['female'], 'bin')
report('died_during_followup', mvac['died'].astype(int), mctl['died'].astype(int), 'bin')

pd.DataFrame(rows).to_csv('Data/smd_age_matched.csv', index=False, encoding='utf-8-sig')
print('\n범례: ✓ |SMD|<0.1 (잘 균형)  ⚠ 0.1≤|SMD|<0.25 (경미한 불균형)  ✗ ≥0.25 (불균형)')
print('\nSaved: Data/smd_age_matched.csv')
