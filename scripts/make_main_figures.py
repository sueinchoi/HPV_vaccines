"""
Publication-level main figures with unified style.

Outputs (300 dpi PNG, English-only):
  Figure1_CohortSelection.png      — CONSORT-like flow diagram
  Figure2_CohortA_CIF_HR.png       — Aalen-Johansen CIF + forest (Cohort A safety)
  Figure3_CohortB_KaplanMeier.png  — KM curves with number-at-risk (Cohort B efficacy)
  Figure4_CohortB_Subgroup.png     — JAMA-style table+forest combining
                                     Overall / by age / by vaccine type
                                     for both primary outcomes (replaces
                                     the previous Figure 4 + Figure 5).
"""
import pandas as pd
import numpy as np
import warnings
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter, AalenJohansenFitter, CoxTimeVaryingFitter
from scipy.stats import chi2
warnings.filterwarnings('ignore')

# =====================================================================
# Unified style (single source of truth)
# =====================================================================
plt.rcParams.update({
    'font.family': ['DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.titlepad': 6,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#cccccc',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Colour palette (colourblind-safe; Okabe-Ito hybrid)
COL_VAC      = '#C44536'   # warm red — vaccinated
COL_CTL      = '#2A6F97'   # cool blue — non-vaccinated
COL_REF      = '#6A994E'   # green — reference / good balance
COL_REF2     = '#E36414'   # orange — secondary reference
COL_GREY     = '#6c757d'
COL_LIGHTGREY= '#dee2e6'
COL_HIGHLIGHT= '#FCEBC2'   # soft yellow highlight bar

# Standard sizes
PANEL_LABEL_SIZE = 14
PANEL_LABEL_KW = dict(fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
LINE_W = 1.9
MARKER_S = 8

CLASS_LABELS = {'1':'Angina/MI','2':'Hypertension','3':'Diabetes','4':'Stroke','5':'PE'}
ANY5 = ['1','2','3','4','5']
MCE = ['1','4','5']
SMOKE_MAP = {'비흡연':'Never','과거흡연':'Former','현재흡연':'Current','확인불능':'Unknown'}

def panel_label(ax, label, dx=-0.18, dy=1.10):
    ax.text(dx, dy, label, transform=ax.transAxes, **PANEL_LABEL_KW)

def style_axes(ax):
    ax.grid(axis='y', alpha=0.25, linestyle=':', lw=0.6)
    ax.tick_params(length=3, width=0.8, which='major')

# =====================================================================
# Helpers
# =====================================================================
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

# =====================================================================
# Figure 1 — Cohort selection flow
# =====================================================================
def figure1():
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11.5)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#ffffff', ec=COL_GREY, fs=11, weight='normal'):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.06',
                              facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                fontweight=weight, color='#222')

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                           mutation_scale=14, color='#444', lw=1.2)
        ax.add_patch(a)

    box(6, 10.7, 5.4, 0.75,
        'Source population: Korean HPV cohort\n'
        '(prospective enrolment 2009–2024)   N = 32,969',
        fc='#e8f4f8', ec='#1f6f8b', fs=11.5, weight='bold')

    arrow(6, 10.32, 6, 10.05)
    box(6, 9.65, 5.8, 0.65,
        'HPV vaccine ascertained from prescription file\n'
        f'Vaccinated  n = 2,156   /   Unvaccinated candidates  n = 30,813',
        fc='#fff3cd', ec='#856404', fs=11)

    arrow(6, 9.32, 3.0, 8.7)
    arrow(6, 9.32, 9.0, 8.7)

    # Cohort A header (left)
    box(3.0, 8.35, 4.9, 0.6,
        'COHORT A — Long-term safety analysis',
        fc='#d4edda', ec='#155724', fs=11.5, weight='bold')

    # Cohort B header (right)
    box(9.0, 8.35, 4.9, 0.6,
        'COHORT B — Post-surgical efficacy analysis',
        fc='#fde2e4', ec='#9b2226', fs=11.5, weight='bold')

    # === Cohort A steps ===
    arrow(3.0, 8.05, 3.0, 7.75)
    box(3.0, 7.35, 4.9, 0.75,
        'Pseudo index date for unvaccinated\n(random sample from vaccine-date distribution)',
        fc='#eaf6ee', ec='#155724', fs=10.5)

    arrow(3.0, 6.97, 3.0, 6.7)
    box(3.0, 6.15, 4.9, 1.05,
        'Eligibility: alive at index date, ≥ 1 day follow-up\n'
        'Logistic propensity score model:\n'
        'age, BMI, SBP, DBP, smoking, residence',
        fc='#eaf6ee', ec='#155724', fs=10.5)

    arrow(3.0, 5.62, 3.0, 5.35)
    box(3.0, 4.8, 4.9, 1.05,
        '1:1 nearest-neighbour matching on logit(PS)\n'
        'Caliper = 0.2 × SD(logit PS), no replacement',
        fc='#eaf6ee', ec='#155724', fs=10.5)

    arrow(3.0, 4.27, 3.0, 4.0)
    box(3.0, 3.15, 4.9, 1.5,
        'Final Cohort A:  n = 4,102\n'
        'Vaccinated 2,051   /   Unvaccinated 2,051\n\n'
        'Outcomes: 5 chronic conditions and the\n'
        'Any-of-5 and MCE composite endpoints',
        fc='#a8d5b5', ec='#155724', fs=11, weight='bold')

    # === Cohort B steps ===
    arrow(9.0, 8.05, 9.0, 7.75)
    box(9.0, 7.35, 4.9, 0.75,
        'Cervical surgery (conization or hysterectomy)\nN = 6,890',
        fc='#fdedee', ec='#9b2226', fs=10.5)

    arrow(9.0, 6.97, 9.0, 6.7)
    box(9.0, 6.15, 4.9, 1.15,
        '1:up-to-5 variable-ratio match (greedy, no replacement):\n'
        'surgery method (exact), year (±1 yr), age (±5 yr)\n'
        'Vaccinated 411  /  Unvaccinated 1,815\n'
        '(mean 4.42 controls per case)',
        fc='#fdedee', ec='#9b2226', fs=10.5)

    arrow(9.0, 5.57, 9.0, 5.35)
    box(9.0, 4.8, 4.9, 1.05,
        'Index date filter: ≤ 31 Dec 2020,  ≥ 2 follow-up records\n'
        'Vaccinated 411  /  Unvaccinated 1,797   (excluded 18)',
        fc='#fdedee', ec='#9b2226', fs=10.5)

    arrow(9.0, 4.27, 9.0, 4.0)
    box(9.0, 3.05, 4.9, 1.65,
        'Final Cohort B:  n = 1,108\n'
        'Vaccinated 241   /   Unvaccinated 867\n'
        '(Fine 1:up-to-4 variable-ratio match\n'
        'on age, BMI, surgery year; mean 3.60)\n\n'
        'Outcomes: lesion recurrence, HPV reinfection',
        fc='#f4a4a8', ec='#9b2226', fs=11, weight='bold')

    plt.tight_layout()
    plt.savefig('Data/Figure1_CohortSelection.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: Data/Figure1_CohortSelection.png')

# =====================================================================
# Re-run Cohort A PSM matching to get the matched dataframe
# =====================================================================
def build_cohort_a_matched():
    print('  Building Cohort A matched dataset...')
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

    rng = np.random.default_rng(42)
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
    lr = LogisticRegression(max_iter=2000, C=1e6, solver='lbfgs').fit(Xs, y)
    df['ps'] = lr.predict_proba(Xs)[:,1]
    df['logit_ps'] = np.log(df['ps']/(1-df['ps']))
    caliper = 0.2 * df['logit_ps'].std()
    vac_idx = df.index[df['vaccinated']].tolist()
    ctl_idx = np.array(df.index[~df['vaccinated']].tolist())
    ctl_logit = df.loc[ctl_idx,'logit_ps'].values
    order = np.argsort(ctl_logit)
    ctl_sorted = ctl_idx[order]; ctl_logit_sorted = ctl_logit[order]
    used = np.zeros(len(ctl_sorted), dtype=bool)
    matched = []
    vac_order = np.array(vac_idx); rng2 = np.random.default_rng(42); rng2.shuffle(vac_order)
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
    pair_records = []
    for pid_, (vi, cii) in enumerate(matched):
        pair_records.append((vi, pid_)); pair_records.append((cii, pid_))
    pair_df = pd.DataFrame(pair_records, columns=['orig_idx','pair_id'])
    out = df.loc[pair_df['orig_idx'].values].copy()
    out['pair_id'] = pair_df['pair_id'].values
    out = out.reset_index(drop=True)

    # comorbidities
    wb = openpyxl.load_workbook('Data/한국 HPV 코호트 자료를 이용한 자_진단정보_기저질환추가_unlocked.xlsx',
                              read_only=True, data_only=True)
    ws = wb.active
    recs=[]
    for row in ws.iter_rows(min_row=2, values_only=True):
        pid, cls, dd = row[0], row[5], row[8]
        if cls is None or str(cls).strip()=='': continue
        cls = str(cls).strip()
        if cls not in CLASS_LABELS: continue
        recs.append((pid, cls, pd.to_datetime(str(dd), format='%Y%m%d', errors='coerce')))
    como = pd.DataFrame(recs, columns=['pid','class','diag_date'])
    first_diag = como.groupby(['pid','class'])['diag_date'].min().unstack('class')
    for c in CLASS_LABELS:
        if c not in first_diag.columns: first_diag[c] = pd.NaT
    out = out.merge(first_diag, left_on='pid', right_index=True, how='left')
    return out

def make_tte(m, cls_or_list):
    if isinstance(cls_or_list, list):
        dx = m[cls_or_list].min(axis=1)
    else:
        dx = m[cls_or_list]
    is_pre = dx.notna() & (dx <= m['index_date'])
    primary = dx.where(dx > m['index_date'], pd.NaT)
    death_after = m['death_date'].where(
        (m['death_date'].notna()) & (m['death_date'] > m['index_date']) &
        ((primary.isna()) | (m['death_date'] < primary)), pd.NaT)
    event_date = primary.combine_first(death_after)
    status = np.where(primary.notna() & ((death_after.isna()) | (primary <= death_after)), 1,
            np.where(death_after.notna(), 2, 0))
    end_date = event_date.combine_first(m['last_follow'])
    time = (end_date - m['index_date']).dt.days.astype(float)
    res = pd.DataFrame({'pid':m['pid'].values,'pair_id':m['pair_id'].values,
                       'vaccinated':m['vaccinated'].astype(int).values,
                       'time':time,'status':status})
    res = res[~is_pre.values & (res['time']>0)].reset_index(drop=True)
    return res

# =====================================================================
# Figure 2 — Cohort A: CIF panels + forest plot
# =====================================================================
def figure2(m):
    print('  Computing CIFs and HRs for Figure 2...')
    outcomes = [('Any-of-5', ANY5), ('MCE composite', MCE),
               ('Diabetes', '3'), ('Hypertension', '2'),
               ('Angina/MI', '1')]  # 5 outcomes (Stroke/PE excluded — no events)

    # Compute CIFs and HRs
    cifs = {}
    hr_results = []
    for label, comp in outcomes + [('Stroke','4'), ('PE','5')]:
        tte = make_tte(m, comp)
        cifs[label] = {}
        for grp_val, grp_name in [(1,'vac'),(0,'ctl')]:
            sub = tte[tte['vaccinated']==grp_val]
            if (sub['status']==1).sum() < 1:
                cifs[label][grp_name] = None
                continue
            try:
                aj = AalenJohansenFitter()
                aj.fit(sub['time'].values, sub['status'].values, event_of_interest=1)
                cifs[label][grp_name] = aj
            except Exception:
                cifs[label][grp_name] = None
        # HR
        e_v = int(((tte['status']==1)&(tte['vaccinated']==1)).sum())
        e_c = int(((tte['status']==1)&(tte['vaccinated']==0)).sum())
        n_v = int((tte['vaccinated']==1).sum()); n_c = int((tte['vaccinated']==0).sum())
        if e_v + e_c >= 5 and e_v >= 1 and e_c >= 1:
            d = tte.copy(); d['event'] = (d['status']==1).astype(int)
            cph = CoxPHFitter()
            try:
                cph.fit(d[['time','event','vaccinated','pair_id']],
                       duration_col='time', event_col='event',
                       cluster_col='pair_id', robust=True)
                sm = cph.summary
                hr = float(sm.loc['vaccinated','exp(coef)'])
                lo = float(sm.loc['vaccinated','exp(coef) lower 95%'])
                hi = float(sm.loc['vaccinated','exp(coef) upper 95%'])
                p = float(sm.loc['vaccinated','p'])
            except Exception:
                hr=lo=hi=p=np.nan
        else:
            hr=lo=hi=p=np.nan
        hr_results.append({'label':label,'e_v':e_v,'n_v':n_v,'e_c':e_c,'n_c':n_c,
                          'hr':hr,'lo':lo,'hi':hi,'p':p})
    hr_df = pd.DataFrame(hr_results)

    # === Plot ===
    fig = plt.figure(figsize=(15, 9.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38,
                 left=0.06, right=0.98, top=0.96, bottom=0.08)

    # CIF panels: Any-of-5, MCE, Diabetes (top row), Hypertension, Angina/MI (bottom row left)
    panel_specs = [
        ('a', 'Any-of-5 composite',  'Any-of-5',     gs[0,0]),
        ('b', 'MCE composite',        'MCE composite',gs[0,1]),
        ('c', 'Diabetes',             'Diabetes',     gs[0,2]),
        ('d', 'Hypertension',         'Hypertension', gs[1,0]),
        ('e', 'Angina / Myocardial infarction', 'Angina/MI', gs[1,1]),
    ]
    max_year = 10
    for plabel, title, key, gs_pos in panel_specs:
        ax = fig.add_subplot(gs_pos)
        for grp_name, color, lbl in [('vac', COL_VAC, 'Vaccinated'),
                                     ('ctl', COL_CTL, 'Non-vaccinated')]:
            aj = cifs.get(key, {}).get(grp_name)
            if aj is None: continue
            cif = aj.cumulative_density_; ci_band = aj.confidence_interval_
            col_name = cif.columns[0]
            t = cif.index.values / 365.25
            y = cif[col_name].values
            ax.step(t, y, where='post', color=color, label=lbl, lw=LINE_W)
            try:
                lo = ci_band.iloc[:,0].values; hi = ci_band.iloc[:,1].values
                ax.fill_between(t, lo, hi, alpha=0.15, color=color, step='post')
            except Exception:
                pass
        ax.set_xlim(0, max_year); ax.set_ylim(bottom=0)
        ax.set_xlabel('Years from index date')
        ax.set_ylabel('Cumulative incidence')
        ax.set_title(title)
        if plabel == 'a':
            ax.legend(loc='lower right', fontsize=9)
        style_axes(ax)
        panel_label(ax, plabel)

    # Forest plot panel (bottom-right)
    ax_f = fig.add_subplot(gs[1, 2])
    forest_order = ['Any-of-5','MCE composite','Hypertension','Diabetes','Angina/MI','Stroke','PE']
    sub = hr_df.set_index('label').reindex(forest_order)
    y_pos = np.arange(len(sub))
    has_hr = sub['hr'].notna().values
    ax_f.errorbar(sub.loc[has_hr,'hr'], y_pos[has_hr],
                 xerr=[sub.loc[has_hr,'hr']-sub.loc[has_hr,'lo'],
                       sub.loc[has_hr,'hi']-sub.loc[has_hr,'hr']],
                 fmt='o', color=COL_VAC, ecolor='#444', capsize=3,
                 markersize=MARKER_S, lw=1.2)
    # mark "insufficient events"
    for i, has in enumerate(has_hr):
        if not has:
            ax_f.text(1.0, i, '— insufficient events —', ha='center', va='center',
                     fontsize=9, color=COL_GREY, style='italic')
    ax_f.axvline(1, color='black', linestyle='--', alpha=0.5, lw=1)
    ax_f.set_xscale('log')
    ax_f.set_xlim(0.05, 30)
    ax_f.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 10])
    ax_f.set_xticklabels(['0.1','0.25','0.5','1','2','4','10'])
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(forest_order)
    ax_f.set_ylim(len(forest_order)-0.3, -0.7)  # explicit padding both ends
    ax_f.set_xlabel('Hazard ratio (95% CI)')
    ax_f.set_title('Hazard ratios — vaccinated vs non-vaccinated')
    style_axes(ax_f); ax_f.grid(axis='x', alpha=0.25, linestyle=':')
    panel_label(ax_f, 'f')

    plt.savefig('Data/Figure2_CohortA_CIF_HR.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: Data/Figure2_CohortA_CIF_HR.png')

# =====================================================================
# Figure 3 — Cohort B Kaplan-Meier with at-risk tables
# =====================================================================
def figure3():
    print('  Building Figure 3 (Cohort B KM)...')
    B = pd.read_csv('Data/final_matched_outcomes.csv', encoding='utf-8-sig')
    B['index_date'] = pd.to_datetime(B['index_date'])
    B['recurrence_date'] = pd.to_datetime(B['recurrence_date'], errors='coerce')
    B['hpv_infection_date'] = pd.to_datetime(B['hpv_infection_date'], errors='coerce')
    B['vac'] = B['접종여부'].astype(bool).astype(int)
    B['follow_up_days'] = pd.to_numeric(B['follow_up_days'], errors='coerce')
    B['index_age'] = pd.to_numeric(B['index_age'], errors='coerce')

    fig = plt.figure(figsize=(14, 8.4))
    gs = GridSpec(2, 2, figure=fig,
                 height_ratios=[3.4, 1.1], hspace=0.32, wspace=0.26,
                 left=0.07, right=0.98, top=0.96, bottom=0.07)

    outcomes = [('a', 'Lesion recurrence (HSIL/CIN3+)', 'recurrence_date', 'has_recurrence', 0),
                ('b', 'New high-risk HPV infection',     'hpv_infection_date','has_hpv_infection', 1)]
    max_year = 10

    for plabel, title, ev_date, ev_col, col_pos in outcomes:
        ax = fig.add_subplot(gs[0, col_pos])
        sub = B.copy()
        sub['time'] = np.where(sub[ev_date].notna(),
                              (sub[ev_date]-sub['index_date']).dt.days,
                              sub['follow_up_days'])
        sub['event'] = sub[ev_col].astype(int)
        sub = sub[sub['time'] > 0]

        kmf_v = KaplanMeierFitter(); kmf_c = KaplanMeierFitter()
        v = sub[sub['vac']==1]; c = sub[sub['vac']==0]
        kmf_v.fit(v['time']/365.25, v['event'], label='Vaccinated')
        kmf_c.fit(c['time']/365.25, c['event'], label='Non-vaccinated')

        kmf_v.plot_survival_function(ax=ax, ci_alpha=0.15, color=COL_VAC, lw=LINE_W)
        kmf_c.plot_survival_function(ax=ax, ci_alpha=0.15, color=COL_CTL, lw=LINE_W)

        # Cox HR
        d = sub[['time','event','vac','index_age','fine_match_id']].dropna()
        cph = CoxPHFitter()
        cph.fit(d, duration_col='time', event_col='event',
               cluster_col='fine_match_id', robust=True)
        sm = cph.summary
        hr = sm.loc['vac','exp(coef)']
        lo, hi = sm.loc['vac','exp(coef) lower 95%'], sm.loc['vac','exp(coef) upper 95%']
        p = sm.loc['vac','p']
        ax.text(0.97, 0.97,
               f"HR = {hr:.2f} (95% CI {lo:.2f}–{hi:.2f})\np = {p:.3f}",
               transform=ax.transAxes, fontsize=10.5, va='top', ha='right',
               bbox=dict(facecolor='white', edgecolor=COL_LIGHTGREY, boxstyle='round,pad=0.45'))

        ax.set_xlim(0, max_year)
        ax.set_xticks(range(0, max_year+1, 2))
        if ev_col == 'has_hpv_infection':
            ax.set_ylim(0.0, 1.02)
        else:
            ax.set_ylim(0.7, 1.02)
        ax.set_xlabel('')  # at-risk row carries the time axis
        ax.set_ylabel('Event-free probability')
        ax.set_title(title)
        ax.legend(loc='lower left', fontsize=10)
        style_axes(ax)
        panel_label(ax, plabel)

        # At-risk table — own x-axis, with row labels in left margin
        ax_tab = fig.add_subplot(gs[1, col_pos])
        ax_tab.set_xlim(-3.6, max_year)   # negative space for row labels
        ax_tab.set_ylim(0, 3)
        ax_tab.axis('off')
        years_list = list(range(0, max_year+1, 2))
        ax_tab.text(-3.5, 2.4, 'No. at risk', fontsize=10.5, fontweight='bold', ha='left')
        for yr in years_list:
            n_v_ = int((v['time']/365.25 >= yr).sum())
            n_c_ = int((c['time']/365.25 >= yr).sum())
            ax_tab.text(yr, 2.4, str(yr), fontsize=10.5, ha='center', fontweight='bold')
            ax_tab.text(yr, 1.4, str(n_v_), fontsize=10, ha='center', color=COL_VAC)
            ax_tab.text(yr, 0.4, str(n_c_), fontsize=10, ha='center', color=COL_CTL)
        ax_tab.text(-3.5, 1.4, 'Vaccinated',     fontsize=10, color=COL_VAC, ha='left')
        ax_tab.text(-3.5, 0.4, 'Non-vaccinated', fontsize=10, color=COL_CTL, ha='left')
        ax_tab.text(max_year/2, -0.6, 'Years from index date',
                   fontsize=11, ha='center', color='#222')

    plt.savefig('Data/Figure3_CohortB_KaplanMeier.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: Data/Figure3_CohortB_KaplanMeier.png')

# =====================================================================
# Figure 4 — Combined subgroup forest (replaces old Figure 4 + Figure 5)
#   JAMA-style table+forest. Two panels (two primary outcomes).
#   Subgroups: Overall, by age at index, by vaccine type.
# =====================================================================
def figure4_subgroup():
    print('  Building Figure 4 (combined subgroup forest)...')
    B = pd.read_csv('Data/final_matched_outcomes.csv', encoding='utf-8-sig')
    Bc = pd.read_csv('Data/final_matched_cohort.csv', encoding='utf-8-sig')
    B = B.merge(Bc[['연구번호','백신종류']], on='연구번호', how='left')
    B['vac'] = B['접종여부'].astype(bool).astype(int)
    B['follow_up_days'] = pd.to_numeric(B['follow_up_days'], errors='coerce')
    B['index_age'] = pd.to_numeric(B['index_age'], errors='coerce')

    # Inherit vaccine type from matched vaccinated participant to controls
    vt_by_match = B.loc[B['vac']==1].groupby('fine_match_id')['백신종류'].first()
    B['vacc_type'] = B.apply(
        lambda r: r['백신종류'] if r['vac']==1 else vt_by_match.get(r['fine_match_id'], np.nan),
        axis=1)
    B['age_grp'] = pd.cut(B['index_age'], bins=[-np.inf, 40, 50, np.inf],
                          labels=['<40', '40-49', '≥50'])

    def hr_subset(d, ev_col):
        n_v = int((d['vac']==1).sum()); n_c = int((d['vac']==0).sum())
        ev_v = int(((d['vac']==1) & (d[ev_col]==1)).sum())
        ev_c = int(((d['vac']==0) & (d[ev_col]==1)).sum())
        out = dict(n_v=n_v, n_c=n_c, ev_v=ev_v, ev_c=ev_c,
                   HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
        if ev_v + ev_c < 3 or n_v < 2 or n_c < 2:
            return out
        df_fit = d[['follow_up_days', ev_col, 'vac', 'index_age', 'fine_match_id']].dropna().rename(
            columns={'follow_up_days':'time', ev_col:'event'})
        df_fit['event'] = df_fit['event'].astype(int)
        if df_fit['event'].sum() < 3:
            return out
        try:
            cph = CoxPHFitter()
            cph.fit(df_fit, duration_col='time', event_col='event',
                   cluster_col='fine_match_id', robust=True)
            r = cph.summary.loc['vac']
            out.update(HR=float(r['exp(coef)']),
                       CIlo=float(r['exp(coef) lower 95%']),
                       CIhi=float(r['exp(coef) upper 95%']),
                       p=float(r['p']))
        except Exception:
            pass
        return out

    def age_interaction_p(ev_col):
        d = B[['follow_up_days', ev_col, 'vac', 'index_age', 'age_grp', 'fine_match_id']].dropna().copy()
        d = d.rename(columns={'follow_up_days':'time', ev_col:'event'})
        d['event'] = d['event'].astype(int)
        d['ag_4049'] = (d['age_grp']=='40-49').astype(int)
        d['ag_50p']  = (d['age_grp']=='≥50').astype(int)
        d['vac_x_4049'] = d['vac'] * d['ag_4049']
        d['vac_x_50p']  = d['vac'] * d['ag_50p']
        full_cols = ['time','event','vac','index_age','ag_4049','ag_50p',
                     'vac_x_4049','vac_x_50p','fine_match_id']
        red_cols  = ['time','event','vac','index_age','ag_4049','ag_50p','fine_match_id']
        try:
            full = CoxPHFitter().fit(d[full_cols], duration_col='time', event_col='event',
                                     cluster_col='fine_match_id', robust=True)
            red  = CoxPHFitter().fit(d[red_cols], duration_col='time', event_col='event',
                                     cluster_col='fine_match_id', robust=True)
            lrt = 2*(full.log_likelihood_ - red.log_likelihood_)
            return float(1 - chi2.cdf(lrt, df=2))
        except Exception:
            return np.nan

    vint = pd.read_csv('Data/CohortB_vaccine_interaction.csv', encoding='utf-8-sig')

    # ----- Build per-outcome row list -----
    outcomes = [('a', 'Lesion recurrence (HSIL/CIN3+)', 'has_recurrence', 'Lesion recurrence'),
                ('b', 'New high-risk HPV infection',     'has_hpv_infection', 'HPV reinfection')]

    panel_data = {}
    for plabel, ttl, ev_col, vint_key in outcomes:
        rows = []
        rows.append(('Overall', hr_subset(B, ev_col), 'data'))
        rows.append(('', None, 'spacer'))
        rows.append(('By age at index', None, 'header'))
        for grp_key, grp_lab in [('<40', '<40 years'),
                                  ('40-49', '40–49 years'),
                                  ('≥50', '≥50 years')]:
            rows.append((grp_lab, hr_subset(B[B['age_grp']==grp_key], ev_col), 'data'))
        ap = age_interaction_p(ev_col)
        rows.append((f'P for interaction = {ap:.3f}' if not np.isnan(ap) else
                     'P for interaction = NA', None, 'pval'))
        rows.append(('', None, 'spacer'))
        rows.append(('By vaccine type', None, 'header'))
        for vt_key, vt_lab in [('Gardasil9', 'Gardasil 9 (9-valent)'),
                                ('Cervarix',  'Cervarix (2-valent)'),
                                ('Gardasil',  'Gardasil (4-valent)')]:
            rows.append((vt_lab, hr_subset(B[B['vacc_type']==vt_key], ev_col), 'data'))
        vp = float(vint.loc[vint['outcome']==vint_key, 'LRT_p'].values[0])
        rows.append((f'P for interaction = {vp:.3f}', None, 'pval'))
        panel_data[plabel] = (ttl, rows)

    # ----- Plot -----
    n_rows = max(len(rows) for _, rows in panel_data.values())
    fig_h = max(7.5, 0.45 * n_rows + 2.8)
    fig, axes = plt.subplots(1, 2, figsize=(20.0, fig_h),
                             gridspec_kw={'left':0.025, 'right':0.995,
                                          'top':0.93, 'bottom':0.06, 'wspace':0.05})

    XCOL = {
        'label':     0.005,   # subgroup label
        'vac':       0.32,    # vaccinated events/N
        'ctl':       0.48,    # control events/N
        'forest_lo': 0.58,    # forest x range begin
        'forest_hi': 0.84,    # forest x range end
        'hrtxt':     0.86,    # HR (CI) text start
    }
    HEADER_Y = -2.0   # column header row
    SUBHEAD_Y = -1.1  # 'No. events / N' subline
    DIVIDER_Y = -0.4  # divider between header and rows
    X_LO, X_HI = 0.1, 10
    XTICKS = [0.1, 0.25, 0.5, 1, 2, 4, 10]

    def x_to_axes(hr):
        return XCOL['forest_lo'] + (np.log10(hr) - np.log10(X_LO))/\
               (np.log10(X_HI) - np.log10(X_LO)) * (XCOL['forest_hi'] - XCOL['forest_lo'])

    for idx, (plabel, (ttl, rows)) in enumerate(panel_data.items()):
        ax = axes[idx]
        n = len(rows)
        ax.set_xlim(0, 1); ax.set_ylim(n + 1.6, HEADER_Y - 0.6)
        ax.axis('off')

        # ---- Column headers ----
        ax.text(XCOL['label'], HEADER_Y, 'Subgroup', fontweight='bold',
                fontsize=11, ha='left', va='center')
        ax.text(XCOL['vac'], HEADER_Y, 'Vaccinated', fontweight='bold',
                fontsize=11, ha='center', va='center')
        ax.text(XCOL['vac'], SUBHEAD_Y, 'No. events / N', fontsize=10,
                ha='center', va='center', color='#444')
        ax.text(XCOL['ctl'], HEADER_Y, 'Non-vaccinated', fontweight='bold',
                fontsize=11, ha='center', va='center')
        ax.text(XCOL['ctl'], SUBHEAD_Y, 'No. events / N', fontsize=10,
                ha='center', va='center', color='#444')
        ax.text((XCOL['forest_lo']+XCOL['forest_hi'])/2, HEADER_Y,
                'Hazard ratio (95% CI)', fontweight='bold',
                fontsize=11, ha='center', va='center')
        ax.text(XCOL['hrtxt']+0.06, HEADER_Y, 'HR (95% CI)',
                fontweight='bold', fontsize=11, ha='left', va='center')

        # divider
        ax.plot([0, 1], [DIVIDER_Y, DIVIDER_Y], color='black', lw=0.9, clip_on=False)

        # ---- Forest backbone (null line) ----
        x_null = x_to_axes(1)
        ax.plot([x_null, x_null], [DIVIDER_Y, n - 0.4],
                color='black', linestyle='--', alpha=0.45, lw=1)

        # ---- Rows ----
        for y, (label, data, kind) in enumerate(rows):
            if kind == 'header':
                ax.text(XCOL['label'], y, label, fontstyle='italic',
                        fontweight='bold', fontsize=10.5, ha='left', va='center')
            elif kind == 'pval':
                ax.text(XCOL['label']+0.025, y, label, fontsize=9.5,
                        fontstyle='italic', ha='left', va='center', color='#444')
            elif kind == 'spacer':
                continue
            elif kind == 'data' and data is not None:
                indent = 0.025 if label != 'Overall' else 0.0
                weight = 'bold' if label == 'Overall' else 'normal'
                ax.text(XCOL['label']+indent, y, label, fontsize=10.5,
                        fontweight=weight, ha='left', va='center')
                ax.text(XCOL['vac'], y, f"{data['ev_v']} / {data['n_v']}",
                        fontsize=10.5, ha='center', va='center')
                ax.text(XCOL['ctl'], y, f"{data['ev_c']} / {data['n_c']}",
                        fontsize=10.5, ha='center', va='center')
                if not np.isnan(data['HR']):
                    sig = (data['CIlo'] > 1) or (data['CIhi'] < 1)
                    color = (COL_VAC if (data['HR']<1 and sig)
                             else COL_CTL if (data['HR']>1 and sig)
                             else '#333')
                    x_hr   = x_to_axes(data['HR'])
                    x_clo  = x_to_axes(max(X_LO, data['CIlo']))
                    x_chi  = x_to_axes(min(X_HI, data['CIhi']))
                    # CI line
                    ax.plot([x_clo, x_chi], [y, y], color=color, lw=1.7)
                    # caps
                    cap = 0.18
                    ax.plot([x_clo, x_clo], [y-cap, y+cap], color=color, lw=1.7)
                    ax.plot([x_chi, x_chi], [y-cap, y+cap], color=color, lw=1.7)
                    # marker
                    weight_overall = (label == 'Overall')
                    ms = 11 if weight_overall else (9 if sig else 8)
                    marker = 's' if (sig or weight_overall) else 'o'
                    ax.plot(x_hr, y, marker, color=color, markersize=ms,
                            markeredgecolor='black', markeredgewidth=0.6)
                    ax.text(XCOL['hrtxt']+0.06, y,
                            f"{data['HR']:.2f} ({data['CIlo']:.2f}–{data['CIhi']:.2f})",
                            fontsize=10, ha='left', va='center')
                else:
                    ax.text(XCOL['hrtxt']+0.06, y, '— insufficient events —',
                            fontsize=9.5, ha='left', va='center',
                            color=COL_GREY, style='italic')

        # ---- Forest x-axis ticks (below last row) ----
        x_axis_y = n - 0.1
        ax.plot([XCOL['forest_lo'], XCOL['forest_hi']], [x_axis_y, x_axis_y],
                color='black', lw=0.9)
        for tk in XTICKS:
            xt = x_to_axes(tk)
            ax.plot([xt, xt], [x_axis_y, x_axis_y + 0.12], color='black', lw=0.9)
            ax.text(xt, x_axis_y + 0.45, str(tk),
                    fontsize=9.5, ha='center', va='center')
        ax.text((XCOL['forest_lo']+XCOL['forest_hi'])/2, x_axis_y + 1.05,
                'Hazard ratio (log scale)', fontsize=10.5, ha='center', va='center')

        # ---- Panel label & outcome subtitle ----
        ax.text(-0.005, HEADER_Y - 0.3, plabel, fontsize=15, fontweight='bold',
                ha='left', va='bottom', clip_on=False)
        ax.text(0.5, HEADER_Y - 0.3, ttl, fontsize=12.5, fontweight='bold',
                ha='center', va='bottom')

    plt.savefig('Data/Figure4_CohortB_Subgroup.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: Data/Figure4_CohortB_Subgroup.png')

# =====================================================================
# Figure 5 — Cohort B age × follow-up forest
# =====================================================================
def figure5():
    print('  Building Figure 5 (age × FU)...')
    df = pd.read_csv('Data/CohortB_age_fu_forest.csv', encoding='utf-8-sig')

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.6), sharey=True,
                            gridspec_kw={'wspace':0.10, 'left':0.12, 'right':0.985,
                                         'top':0.92, 'bottom':0.14})
    target_fu_keys   = ['1 yr', '2 yr', '4 yr', 'Full follow-up']
    target_fu_titles = ['Up to 1 yr', 'Up to 2 yr', 'Up to 4 yr', 'Full follow-up']
    # Pre-specified age strata only — post-hoc 30–52 row removed per JAMA editorial
    # standards (data-driven cutoff, not multiple-comparison adjusted).
    strata_labels = ['All ages', '<40 years', '40–49 years', '≥50 years']

    for i, (ax, fu_key, fu_title) in enumerate(zip(axes, target_fu_keys, target_fu_titles)):
        sub = df[df['fu_label']==fu_key].set_index('stratum').reindex(strata_labels)
        for j, (stratum, row) in enumerate(sub.iterrows()):
            if pd.isna(row['HR']):
                ax.text(1, j, 'insufficient events', ha='center', va='center',
                       fontsize=9, color=COL_GREY, style='italic')
                continue
            sig = row['p'] < 0.05
            color = COL_VAC if (row['HR']<1 and sig) else (COL_CTL if (row['HR']>1 and sig) else COL_GREY)
            marker = 's' if sig else 'o'
            ms = MARKER_S+2 if sig else MARKER_S
            ax.errorbar(row['HR'], j, xerr=[[row['HR']-row['CI_lo']],[row['CI_hi']-row['HR']]],
                       fmt=marker, color=color, ecolor='#555', capsize=3, markersize=ms, lw=1.4)
            # HR text positioned just above marker for readability
            ax.text(row['HR'], j-0.42,
                   f"{row['HR']:.2f} ({row['CI_lo']:.2f}–{row['CI_hi']:.2f})\np = {row['p']:.3f}",
                   fontsize=9, ha='center', va='bottom')
        ax.axvline(1, color='black', linestyle='--', alpha=0.5, lw=1)
        ax.set_xscale('log'); ax.set_xlim(0.025, 30)
        ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 10])
        ax.set_xticklabels(['0.1','0.25','0.5','1','2','4','10'])
        ax.set_yticks(range(len(strata_labels)))
        ax.invert_yaxis()
        ax.set_ylim(len(strata_labels)-0.3, -0.7)  # padding for HR text
        ax.set_title(fu_title)
        ax.set_xlabel('Hazard ratio (95% CI)')
        style_axes(ax); ax.grid(axis='x', alpha=0.25, linestyle=':')
        panel_label(ax, chr(ord('a')+i))
        # Hide tick labels on non-first panels (sharey shares the labels object,
        # so we use tick_params instead of set_yticklabels which would overwrite axes[0])
        if i > 0:
            ax.tick_params(axis='y', labelleft=False)

    # Set strata labels ONCE on axes[0] (last so it isn't overwritten by sharey)
    axes[0].set_yticklabels(strata_labels, fontsize=11)
    axes[0].tick_params(axis='y', labelleft=True)

    plt.savefig('Data/Figure5_CohortB_AgeFollowUp.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: Data/Figure5_CohortB_AgeFollowUp.png')


# =====================================================================
if __name__ == '__main__':
    print('[Figure 1]')
    figure1()
    print('[Figure 2]')
    m_A = build_cohort_a_matched()
    figure2(m_A)
    print('[Figure 3]')
    figure3()
    print('[Figure 4 — combined subgroup forest]')
    figure4_subgroup()
    print('\nAll main figures regenerated at 300 dpi with unified style.')
