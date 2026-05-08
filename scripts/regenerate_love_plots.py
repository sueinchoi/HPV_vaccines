"""
Regenerate Supplementary Figure S1 (Cohort A) and S2 (Cohort B) Love plots
with a unified visual style and English-only labels.
"""
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': ['DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10.5,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#cccccc',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

RANDOM_SEED = 42
SMOKE_MAP = {'비흡연':'Never','과거흡연':'Former','현재흡연':'Current','확인불능':'Unknown'}

# --------------------- Helpers ---------------------
def smd_cont(a, b):
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    if len(a)<2 or len(b)<2: return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1))/2)
    return (a.mean() - b.mean())/pooled if pooled>0 else np.nan

def smd_bin(a, b):
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    if len(a)==0 or len(b)==0: return np.nan
    p1, p2 = a.mean(), b.mean()
    pooled = np.sqrt((p1*(1-p1) + p2*(1-p2))/2)
    return (p1 - p2)/pooled if pooled>0 else np.nan

def closest_vec(query_df, ci, value_col, window_days=365):
    ci_v = ci[['연구번호','기록일자_dt', value_col]].dropna(subset=[value_col,'기록일자_dt']).copy()
    ci_v = ci_v.sort_values('기록일자_dt').rename(columns={'연구번호':'pid','기록일자_dt':'rec_date'})
    q = query_df.sort_values('index_date').reset_index().rename(columns={'index':'orig_idx'})
    fw = pd.merge_asof(q[['orig_idx','pid','index_date']], ci_v,
        left_on='index_date', right_on='rec_date', by='pid', direction='forward', tolerance=pd.Timedelta(days=window_days))
    bw = pd.merge_asof(q[['orig_idx','pid','index_date']], ci_v,
        left_on='index_date', right_on='rec_date', by='pid', direction='backward', tolerance=pd.Timedelta(days=window_days))
    fw['_d'] = (fw['rec_date']-fw['index_date']).abs()
    bw['_d'] = (bw['rec_date']-bw['index_date']).abs()
    use_fw = (fw['_d'].fillna(pd.Timedelta(days=window_days*10)) <= bw['_d'].fillna(pd.Timedelta(days=window_days*10)))
    return pd.Series(np.where(use_fw, fw[value_col].values, bw[value_col].values),
                    index=fw['orig_idx'].values).reindex(query_df.index).astype(float)

def smoke_vec(query_df, ci):
    smk = ci[['연구번호','기록일자_dt','흡연여부']].dropna(subset=['흡연여부','기록일자_dt']).copy()
    smk = smk.sort_values('기록일자_dt').rename(columns={'연구번호':'pid','기록일자_dt':'rec_date'})
    q = query_df.sort_values('index_date').reset_index().rename(columns={'index':'orig_idx'})
    bw = pd.merge_asof(q[['orig_idx','pid','index_date']], smk,
        left_on='index_date', right_on='rec_date', by='pid', direction='backward')
    return pd.Series(bw['흡연여부'].map(SMOKE_MAP).fillna('Unknown').values,
                    index=bw['orig_idx'].values).reindex(query_df.index).fillna('Unknown')

# --------------------- Unified plotting style ---------------------
def love_plot(variables, before_smds, after_smds, out_path):
    """Standardised love-plot style — no in-figure title (carried by figure legend)."""
    n = len(variables)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.5*n + 1.4)))
    y = np.arange(n)
    # connecting lines
    for i in range(n):
        if not (np.isnan(before_smds[i]) or np.isnan(after_smds[i])):
            ax.plot([before_smds[i], after_smds[i]], [i, i],
                   color='#cccccc', lw=1, zorder=1)
    # Before (red circles)
    ax.scatter(before_smds, y, s=110, color='#9b2226', edgecolor='black',
              linewidth=0.8, label='Before matching', zorder=3)
    # After (blue squares)
    ax.scatter(after_smds, y, s=110, color='#1f6f8b', edgecolor='black',
              linewidth=0.8, label='After matching', marker='s', zorder=3)

    # Reference lines
    ax.axvline(0, color='black', lw=0.6, alpha=0.4)
    ax.axvline(0.10, color='#2b8a3e', linestyle='--', alpha=0.7, lw=1.3,
              label='|SMD| = 0.10 (good balance)')
    ax.axvline(0.25, color='#e8590c', linestyle='--', alpha=0.5, lw=1.1,
              label='|SMD| = 0.25 (acceptable)')

    ax.set_yticks(y)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Absolute Standardized Mean Difference (|SMD|)')
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.25, linestyle=':')
    ax.set_xlim(-0.05, max(1.3, max(np.nan_to_num(before_smds, nan=0)) + 0.1))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out_path}')

# ============================================================
# Cohort A — PSM Love plot (rebuild)
# ============================================================
print('[Cohort A] Loading source...')
cohort = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_코호트.csv', encoding='cp949', low_memory=False)
cohort['birth_date'] = pd.to_datetime(cohort['생년월'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
cohort['death_date'] = pd.to_datetime(cohort['사망일자'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
cohort['last_follow'] = pd.to_datetime(cohort['최종추적일자'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
cohort['is_seoul'] = (cohort['주소'].astype(str).str.split().str[0]=='서울').astype(int)
rx = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_처방정보.csv', encoding='cp949', low_memory=False)
mask = (rx['처방명'].astype(str).str.contains('Gardasil|Cervarix|HPV vaccine', case=False, na=False) |
        rx['처방한글명'].astype(str).str.contains('가다실|서바릭스', na=False))
rx_vac = rx[mask].copy()
rx_vac['처방일자'] = pd.to_datetime(rx_vac['처방일자'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
first_vac = rx_vac.groupby('연구번호')['처방일자'].min().reset_index()
first_vac.columns = ['연구번호','first_vaccine_date']
cohort = cohort.merge(first_vac, on='연구번호', how='left').dropna(subset=['birth_date'])
ci = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_기초임상정보.csv', encoding='cp949', low_memory=False)
ci['기록일자_dt'] = pd.to_datetime(ci['기록일자'].astype(str).str.strip(), format='%Y%m%d', errors='coerce')

rng = np.random.default_rng(RANDOM_SEED)
df = cohort.copy()
df['vaccinated'] = df['first_vaccine_date'].notna()
vac_dates = df.loc[df['vaccinated'], 'first_vaccine_date'].dropna().values
df.loc[~df['vaccinated'], 'index_date'] = pd.to_datetime(rng.choice(vac_dates, size=(~df['vaccinated']).sum()))
df.loc[df['vaccinated'], 'index_date'] = df.loc[df['vaccinated'], 'first_vaccine_date']
df = df.rename(columns={'연구번호':'pid'}).reset_index(drop=True)
df['age_at_index'] = (df['index_date'] - df['birth_date']).dt.days/365.25
df = df[(df['death_date'].isna()) | (df['death_date'] > df['index_date'])]
df = df[df['last_follow'] > df['index_date']].reset_index(drop=True)
q = df[['pid','index_date']].copy()
df['height'] = closest_vec(q, ci, '키')
df['weight'] = closest_vec(q, ci, '몸무게')
df['sbp'] = closest_vec(q, ci, '수축기혈압')
df['dbp'] = closest_vec(q, ci, '이완기혈압')
df['bmi'] = df['weight']/(df['height']/100)**2
df['smoke'] = smoke_vec(q, ci).values
for c in ['bmi','sbp','dbp']:
    df[f'{c}_miss'] = df[c].isna().astype(int)
    df[c] = df[c].fillna(df[c].mean())
sm = pd.get_dummies(df['smoke'], prefix='smoke').astype(int)
df = pd.concat([df, sm], axis=1)
ps_features = ['age_at_index','bmi','bmi_miss','sbp','sbp_miss','dbp','dbp_miss','is_seoul',
              'smoke_Never','smoke_Former','smoke_Current']
ps_features = [c for c in ps_features if c in df.columns]
X = df[ps_features].astype(float).values
y = df['vaccinated'].astype(int).values
Xs = StandardScaler().fit_transform(X)
lr = LogisticRegression(max_iter=2000, C=1e6, solver='lbfgs')
lr.fit(Xs, y)
df['ps'] = lr.predict_proba(Xs)[:,1]
df['logit_ps'] = np.log(df['ps']/(1-df['ps']))
caliper = 0.2 * df['logit_ps'].std()

# 1:1 matching
vac_idx = df.index[df['vaccinated']].tolist()
ctl_idx = np.array(df.index[~df['vaccinated']].tolist())
ctl_logit = df.loc[ctl_idx,'logit_ps'].values
order = np.argsort(ctl_logit)
ctl_sorted = ctl_idx[order]; ctl_logit_sorted = ctl_logit[order]
used = np.zeros(len(ctl_sorted), dtype=bool)
matched = []
vac_order = np.array(vac_idx); rng2 = np.random.default_rng(RANDOM_SEED); rng2.shuffle(vac_order)
for vi in vac_order:
    target = df.loc[vi,'logit_ps']
    lo = np.searchsorted(ctl_logit_sorted, target-caliper)
    hi = np.searchsorted(ctl_logit_sorted, target+caliper, side='right')
    best_j, best_d = -1, caliper+1
    for j in range(lo, hi):
        if used[j]: continue
        d = abs(ctl_logit_sorted[j]-target)
        if d < best_d: best_d=d; best_j=j
    if best_j>=0:
        used[best_j] = True
        matched.append((vi, ctl_sorted[best_j]))
matched_idx = [i for v,c in matched for i in (v,c)]
df['matched'] = False
df.loc[matched_idx, 'matched'] = True

# Variable labels (English)
var_labels_A = {
    'age_at_index':   'Age at index (years)',
    'bmi':            'BMI (kg/m²)',
    'bmi_miss':       'BMI missing',
    'sbp':            'Systolic BP (mmHg)',
    'sbp_miss':       'SBP missing',
    'dbp':            'Diastolic BP (mmHg)',
    'dbp_miss':       'DBP missing',
    'is_seoul':       'Residence in Seoul',
    'smoke_Never':    'Smoking: Never',
    'smoke_Former':   'Smoking: Former',
    'smoke_Current':  'Smoking: Current',
    'logit_ps':       'Logit (propensity score)',
}
binary_A = {'bmi_miss','sbp_miss','dbp_miss','is_seoul','smoke_Never','smoke_Former','smoke_Current'}
vars_A = list(var_labels_A.keys())

vac_pre = df[df['vaccinated']]; ctl_pre = df[~df['vaccinated']]
vac_post = df[df['matched'] & df['vaccinated']]; ctl_post = df[df['matched'] & ~df['vaccinated']]

before_A = []; after_A = []
for v in vars_A:
    if v in binary_A:
        before_A.append(abs(smd_bin(vac_pre[v].astype(float), ctl_pre[v].astype(float))))
        after_A.append(abs(smd_bin(vac_post[v].astype(float), ctl_post[v].astype(float))))
    else:
        before_A.append(abs(smd_cont(vac_pre[v], ctl_pre[v])))
        after_A.append(abs(smd_cont(vac_post[v], ctl_post[v])))

love_plot(
    variables=[var_labels_A[v] for v in vars_A],
    before_smds=before_A, after_smds=after_A,
    out_path='Data/SupFigS1_loveplot_cohortA.png'
)

# --- Sup Fig S3: PS density (uses df from above) ---
print('[Sup Fig S3] Rendering PS density...')
fig_s3, ax_s3 = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={'wspace':0.22, 'left':0.07,
                                          'right':0.985, 'top':0.93, 'bottom':0.13})
ps_pre_v  = df.loc[df['vaccinated'], 'ps']
ps_pre_c  = df.loc[~df['vaccinated'], 'ps']
ps_post_v = df.loc[df['matched'] & df['vaccinated'], 'ps']
ps_post_c = df.loc[df['matched'] & ~df['vaccinated'], 'ps']
for ax_, (vac_, ctl_, ttl, tag) in zip(ax_s3, [
        (ps_pre_v,  ps_pre_c,  '(a) Pre-matching', 'a'),
        (ps_post_v, ps_post_c, f'(b) Post-matching (1:1, n = {len(ps_post_v):,} per group)', 'b'),
    ]):
    ax_.hist(vac_, bins=40, alpha=0.6, color='#9b2226', label='Vaccinated', density=True)
    ax_.hist(ctl_, bins=40, alpha=0.6, color='#1f6f8b', label='Non-vaccinated', density=True)
    ax_.set_xlabel('Propensity score', fontsize=12)
    ax_.set_ylabel('Density', fontsize=12)
    ax_.set_title(ttl, fontsize=12, fontweight='bold', pad=6)
    ax_.text(-0.13, 1.07, tag, transform=ax_.transAxes,
             fontsize=14, fontweight='bold', va='top', ha='left')
    ax_.legend(loc='upper right')
    ax_.grid(axis='y', alpha=0.25, linestyle=':')
    ax_.tick_params(labelsize=11)
plt.savefig('Data/SupFigS3_ps_density.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: Data/SupFigS3_ps_density.png')

# ============================================================
# Cohort B — Fine matching Love plot
# ============================================================
print('\n[Cohort B] Loading...')
B_post = pd.read_csv('Data/final_matched_cohort.csv', encoding='utf-8-sig')
B_post['vaccinated'] = B_post['접종여부'].astype(bool)

# Pre-matching pool: all surgery patients with exposure status
surg = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_수술처방_수술종류구분완료.csv',
                  encoding='cp949', low_memory=False)
surg['수술처방일자'] = pd.to_datetime(surg['수술처방일자'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
surg = surg[surg['수술 종류'].astype(str).isin(['1','3'])].copy()
SURG_TYPE = {'1':'원추절제술','3':'자궁절제술'}
surg['수술방법'] = surg['수술 종류'].astype(str).map(SURG_TYPE)
first_surg = surg.sort_values('수술처방일자').groupby('연구번호').first()[['수술처방일자','수술방법']].reset_index()
first_surg.columns = ['연구번호','first_surg_date','first_surg_type']

cohort_b_pre = first_surg.merge(
    cohort[['연구번호','birth_date','first_vaccine_date']], on='연구번호', how='left').dropna(subset=['birth_date'])
cohort_b_pre['vaccinated'] = cohort_b_pre['first_vaccine_date'].notna()
cohort_b_pre['age_at_surgery'] = (cohort_b_pre['first_surg_date'] - cohort_b_pre['birth_date']).dt.days/365.25
cohort_b_pre['surgery_year'] = cohort_b_pre['first_surg_date'].dt.year
cohort_b_pre['hysterectomy'] = (cohort_b_pre['first_surg_type']=='자궁절제술').astype(int)

# Closest BMI to surgery date as proxy (limited but consistent)
cohort_b_pre['index_date'] = cohort_b_pre['first_surg_date']
qB = cohort_b_pre[['연구번호','index_date']].rename(columns={'연구번호':'pid'})
cohort_b_pre['height'] = closest_vec(qB.assign(pid=cohort_b_pre['연구번호'].values), ci, '키')
cohort_b_pre['weight'] = closest_vec(qB.assign(pid=cohort_b_pre['연구번호'].values), ci, '몸무게')
cohort_b_pre['bmi'] = cohort_b_pre['weight']/(cohort_b_pre['height']/100)**2

# Post-matching: rebuild same variables from final_matched_cohort
B_post['hysterectomy'] = (B_post['수술방법']=='자궁절제술').astype(int)
B_post['surgery_year'] = pd.to_numeric(B_post['수술연도'], errors='coerce')
B_post['age_at_surgery'] = pd.to_numeric(B_post['수술시나이'], errors='coerce')
B_post['age_at_index'] = pd.to_numeric(B_post['index_age'], errors='coerce')
B_post['bmi'] = pd.to_numeric(B_post['closest_bmi'], errors='coerce')

# Pre-matching equivalent of age_at_index can't be computed without index_date — use surgery age as proxy
cohort_b_pre['age_at_index'] = cohort_b_pre['age_at_surgery']  # surgery == index proxy pre-match

var_labels_B = {
    'age_at_surgery': 'Age at surgery (years)',
    'age_at_index':   'Age at index (years)',
    'bmi':            'BMI (kg/m²)',
    'surgery_year':   'Year of surgery',
    'hysterectomy':   'Hysterectomy (vs conization)',
}
binary_B = {'hysterectomy'}
vars_B = list(var_labels_B.keys())

vacB_pre = cohort_b_pre[cohort_b_pre['vaccinated']]
ctlB_pre = cohort_b_pre[~cohort_b_pre['vaccinated']]
vacB_post = B_post[B_post['vaccinated']]
ctlB_post = B_post[~B_post['vaccinated']]

before_B = []; after_B = []
for v in vars_B:
    if v in binary_B:
        before_B.append(abs(smd_bin(vacB_pre[v].astype(float), ctlB_pre[v].astype(float))))
        after_B.append(abs(smd_bin(vacB_post[v].astype(float), ctlB_post[v].astype(float))))
    else:
        before_B.append(abs(smd_cont(vacB_pre[v], ctlB_pre[v])))
        after_B.append(abs(smd_cont(vacB_post[v], ctlB_post[v])))

love_plot(
    variables=[var_labels_B[v] for v in vars_B],
    before_smds=before_B, after_smds=after_B,
    out_path='Data/SupFigS2_loveplot_cohortB.png'
)

print('\nDone.')
