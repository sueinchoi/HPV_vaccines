"""
Publication-level main figures with unified style.

Outputs (300 dpi PNG, English-only):
  Figure1_CohortSelection.png      — CONSORT-like flow diagram
  Figure2_CohortA_CIF_HR.png       — Aalen-Johansen CIF + forest (Cohort A safety)
  Figure3_CohortB_CIF.png          — Cumulative incidence (1−KM) curves with
                                     number-at-risk tables (Cohort B efficacy)
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
    """Concise CONSORT flow. Academic monochrome palette (black borders,
    white / light-gray fills, black text) for journal submission. Detailed
    methodology lives in docs/Figure1_Note.md; the boxes carry only step
    labels and n's."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10.5)
    ax.axis('off')

    EDGE = '#1a1a1a'
    FILL_WHITE = '#ffffff'
    FILL_LIGHT = '#f2f2f2'
    FILL_FINAL = '#e6e6e6'

    def box(x, y, w, h, text, fc=FILL_WHITE, ec=EDGE, fs=12, weight='normal'):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.06',
                              facecolor=fc, edgecolor=ec, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                fontweight=weight, color='#000000')

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                           mutation_scale=12, color=EDGE, lw=1.0)
        ax.add_patch(a)

    # Source
    box(6, 9.7, 5.0, 0.7,
        'Source population\nN = 32,969',
        fc=FILL_WHITE, fs=12, weight='bold')

    arrow(6, 9.35, 6, 9.05)
    box(6, 8.65, 5.8, 0.7,
        'HPV vaccine prescription ascertained\n'
        'Vaccinated 2,156    Unvaccinated 30,813',
        fc=FILL_WHITE, fs=12)

    arrow(6, 8.30, 3.0, 7.65)
    arrow(6, 8.30, 9.0, 7.65)

    # Cohort A header
    box(3.0, 7.30, 4.6, 0.6,
        'Cohort A — chronic-disease safety',
        fc=FILL_LIGHT, fs=12, weight='bold')

    # Cohort B header
    box(9.0, 7.30, 4.6, 0.6,
        'Cohort B — post-surgical efficacy',
        fc=FILL_LIGHT, fs=12, weight='bold')

    # === Cohort A steps (concise) ===
    arrow(3.0, 7.00, 3.0, 6.70)
    box(3.0, 6.30, 4.6, 0.7,
        'Eligibility (index ≤ 31 Dec 2024)\n+ pseudo index date for controls',
        fc=FILL_WHITE, fs=11)

    arrow(3.0, 5.95, 3.0, 5.65)
    box(3.0, 5.20, 4.6, 0.8,
        '1:1 propensity-score matching\n(caliper 0.2 × SD logit PS)',
        fc=FILL_WHITE, fs=11)

    arrow(3.0, 4.80, 3.0, 4.55)
    box(3.0, 4.18, 4.6, 0.55,
        'Post-PSM intermediate\nVac 2,053 / Non-vac 2,053',
        fc=FILL_WHITE, fs=11)

    arrow(3.0, 3.90, 3.0, 3.68)
    box(3.0, 3.32, 4.6, 0.55,
        'Primary exposure filter\n≥2 doses + 3-month landmark',
        fc=FILL_WHITE, fs=11)

    arrow(3.0, 3.04, 3.0, 2.82)
    box(3.0, 2.35, 4.6, 0.95,
        'Final analytic Cohort A\nn = 2,776\nVac 1,396  /  Non-vac 1,380',
        fc=FILL_FINAL, fs=12, weight='bold')

    # === Cohort B steps (concise) ===
    arrow(9.0, 7.00, 9.0, 6.78)
    box(9.0, 6.45, 4.6, 0.55,
        'Cervical surgery (conization/hysterectomy)\nn = 6,890',
        fc=FILL_WHITE, fs=11)

    arrow(9.0, 6.17, 9.0, 5.95)
    box(9.0, 5.62, 4.6, 0.55,
        '1:up-to-5 initial match\nVac 411 / Non-vac 1,815',
        fc=FILL_WHITE, fs=11)

    arrow(9.0, 5.35, 9.0, 5.13)
    box(9.0, 4.80, 4.6, 0.55,
        '1:up-to-4 fine match\nVac 241 / Non-vac 867',
        fc=FILL_WHITE, fs=11)

    arrow(9.0, 4.52, 9.0, 4.30)
    box(9.0, 3.97, 4.6, 0.55,
        'Primary exposure filter\n≥2 doses + 3-month landmark',
        fc=FILL_WHITE, fs=11)

    arrow(9.0, 3.69, 9.0, 3.47)
    box(9.0, 3.00, 4.6, 0.95,
        'Final analytic Cohort B\nn = 912\nVac 203  /  Non-vac 709',
        fc=FILL_FINAL, fs=12, weight='bold')

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
    # Symmetric ≥1y-potential-FU eligibility filter (admin censor 2025-12-31):
    # cap index date at 2024-12-31 so every retained subject has at least one
    # year of potential observation before administrative censoring. Mirrors
    # the eligibility filter used in build_final_cohort.py (Cohort B).
    df = df[df['index_date'] <= pd.Timestamp('2024-12-31')].reset_index(drop=True)
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
    legend_handles_recorded = False
    legend_handles = []
    legend_labels = []
    from matplotlib.ticker import PercentFormatter as _PercentFormatter, FixedLocator as _FixedLocator
    # Per user request, unify y-axis ranges:
    #   panels a (Any-of-5) and c (Diabetes):    0, 1, 2, 3, 4, 5  (%)
    #   panels b (MCE), d (Hypertension), e (Angina/MI):  0, 0.5, 1, 1.5, 2  (%)
    y_high = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    y_low  = [0.000, 0.005, 0.010, 0.015, 0.020]
    Y_TICKS_BY_PANEL = {'a': y_high, 'b': y_low, 'c': y_high, 'd': y_low, 'e': y_low}
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
            line, = ax.step(t, y, where='post', color=color, label=lbl, lw=LINE_W)
            if not legend_handles_recorded:
                legend_handles.append(line)
                legend_labels.append(lbl)
            try:
                lo = ci_band.iloc[:,0].values; hi = ci_band.iloc[:,1].values
                ax.fill_between(t, lo, hi, alpha=0.15, color=color, step='post')
            except Exception:
                pass
        legend_handles_recorded = True
        ax.set_xlim(0, max_year)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Cumulative incidence (%)')
        ax.yaxis.set_major_formatter(_PercentFormatter(xmax=1.0, decimals=1))
        ticks = Y_TICKS_BY_PANEL.get(plabel)
        if ticks is not None:
            ax.set_ylim(ticks[0], ticks[-1])
            ax.yaxis.set_major_locator(_FixedLocator(ticks))
        else:
            ax.set_ylim(bottom=0)
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
    style_axes(ax_f); ax_f.grid(axis='x', alpha=0.25, linestyle=':')
    panel_label(ax_f, 'f')

    # ---- Shared horizontal legend (Vaccinated / Non-vaccinated only) below all panels ----
    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='lower center', ncol=len(legend_handles),
                   bbox_to_anchor=(0.5, -0.04),
                   fontsize=11, frameon=False, columnspacing=2.5,
                   handletextpad=0.6, handlelength=2.0)
        plt.subplots_adjust(bottom=0.12)

    plt.savefig('Data/Figure2_CohortA_CIF_HR.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: Data/Figure2_CohortA_CIF_HR.png')

# =====================================================================
# Figure 3 — Cohort B Kaplan-Meier with at-risk tables
# =====================================================================
def figure3():
    """Cohort B co-primary outcomes:
       a) Cumulative incidence of lesion recurrence (whole cohort)
       b) Cumulative clearance probability of pre-existing hr-HPV
          (subset with documented pre-vaccine hr-HPV+ baseline)
    """
    print('  Building Figure 3 (Cohort B co-primary CIF)...')
    LANDMARK_DAYS = 90
    # Load v3 primary cohort (≥2 dose + 3-mo landmark + matched-set integrity).
    # The 3-month landmark filter is applied below so that the cumulative
    # incidence curves are flat in [0, 90 d] from the index date by
    # construction — no events are at risk in the landmark window.
    B = pd.read_csv('Data/primary_cohort_v3.csv', encoding='utf-8-sig')
    B['index_date']      = pd.to_datetime(B['index_date'])
    B['최종추적일자']     = pd.to_datetime(B['최종추적일자'])
    B['recurrence_date'] = pd.to_datetime(B['recurrence_date'], errors='coerce')
    B['vac']             = B['접종여부'].astype(bool).astype(int)
    B['index_age']       = pd.to_numeric(B['index_age'], errors='coerce')
    B['lm_zero']         = B['index_date'] + pd.Timedelta(days=LANDMARK_DAYS)

    # Apply landmark to recurrence: drop matched sets where the vaccinated
    # case had a recurrence before the landmark, then drop any remaining
    # rows whose event was pre-landmark, so the analytic sample equals the
    # canonical P1 (n = 912: 203 vac / 709 ctl).
    rec_pre_lm = (B['has_recurrence']==True) & (B['recurrence_date'] < B['lm_zero'])
    bad_p1 = B[(B['vac']==1) & rec_pre_lm]['fine_match_id'].unique()
    P1 = B[~B['fine_match_id'].isin(bad_p1)].copy()
    P1 = P1[~rec_pre_lm.loc[P1.index]].copy()

    # Compute clearance subset (pre-vaccine hr-HPV+, matched-set integrity,
    # landmark filter) — same definition as analyze_primary_v3.py P2.
    import sys as _sys; _sys.path.insert(0, 'scripts')
    from extract_pathology_outcomes import detect_high_risk_hpv as _det
    _path = pd.read_csv(
        'Data/한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV',
        encoding='cp949', low_memory=False)
    _path['실시일자'] = pd.to_datetime(_path['실시일자'], format='%Y%m%d', errors='coerce')
    _mol = _path[_path['병리검사구분'].isin(['분자병리','HPV'])].dropna(subset=['실시일자','판독결과'])
    _res = _mol['판독결과'].apply(_det)
    _mol = _mol.assign(hpv_pos=_res.apply(lambda d: d['is_high_risk_hpv_positive']))
    _mol_by_pid = {pid: g.sort_values('실시일자') for pid, g in _mol.groupby('연구번호')}

    def _prevac_hr(pid, idx_dt):
        sub = _mol_by_pid.get(pid)
        return False if sub is None else bool((sub[sub['실시일자']<idx_dt]['hpv_pos']==True).any())
    def _first_two_neg(pid, idx_dt):
        sub = _mol_by_pid.get(pid)
        if sub is None: return None
        sub = sub[sub['실시일자']>idx_dt]
        if len(sub)<2: return None
        pos = sub['hpv_pos'].values; dates = sub['실시일자'].values
        for i in range(len(pos)-1):
            if (not pos[i]) and (not pos[i+1]):
                return pd.Timestamp(dates[i])
        return None

    B['prevac_hr'] = B.apply(lambda r: _prevac_hr(r['연구번호'], r['index_date']), axis=1)
    fids_clr = B.loc[(B['vac']==1) & B['prevac_hr'], 'fine_match_id'].unique()
    BC = B[B['fine_match_id'].isin(fids_clr) & B['prevac_hr']].copy()
    BC['first_neg_date'] = BC.apply(
        lambda r: _first_two_neg(r['연구번호'], r['index_date']), axis=1)
    BC['has_clearance']  = BC['first_neg_date'].notna()
    clr_pre_lm = BC['has_clearance'] & (BC['first_neg_date'] < BC['lm_zero'])
    bad_p2 = BC[(BC['vac']==1) & clr_pre_lm]['fine_match_id'].unique()
    BC = BC[~BC['fine_match_id'].isin(bad_p2)].copy()
    BC = BC[~clr_pre_lm.loc[BC.index]].copy()

    fig = plt.figure(figsize=(14, 8.4))
    gs = GridSpec(2, 2, figure=fig,
                 height_ratios=[3.4, 1.1], hspace=0.32, wspace=0.26,
                 left=0.07, right=0.98, top=0.96, bottom=0.07)
    max_year = 10

    # ---- Panel a: Lesion recurrence (cumulative incidence rises) ----
    ax_a = fig.add_subplot(gs[0, 0])
    sub_a = P1.copy()
    sub_a['time'] = np.where(sub_a['has_recurrence']==True,
                              (sub_a['recurrence_date']-sub_a['index_date']).dt.days,
                              (sub_a['최종추적일자']    -sub_a['index_date']).dt.days)
    sub_a['event'] = (sub_a['has_recurrence']==True).astype(int)
    sub_a = sub_a[sub_a['time'] > 0]
    v_a = sub_a[sub_a['vac']==1]; c_a = sub_a[sub_a['vac']==0]
    kmf_v = KaplanMeierFitter().fit(v_a['time']/365.25, v_a['event'], label='Vaccinated')
    kmf_c = KaplanMeierFitter().fit(c_a['time']/365.25, c_a['event'], label='Non-vaccinated')
    kmf_v.plot_cumulative_density(ax=ax_a, ci_alpha=0.15, color=COL_VAC, lw=LINE_W)
    kmf_c.plot_cumulative_density(ax=ax_a, ci_alpha=0.15, color=COL_CTL, lw=LINE_W)

    cph = CoxPHFitter().fit(
        sub_a[['time','event','vac','index_age','fine_match_id']].dropna(),
        duration_col='time', event_col='event', cluster_col='fine_match_id', robust=True)
    sm = cph.summary; hr=sm.loc['vac','exp(coef)']
    lo=sm.loc['vac','exp(coef) lower 95%']; hi=sm.loc['vac','exp(coef) upper 95%']
    pv=sm.loc['vac','p']
    ax_a.text(0.97, 0.04,
              f"HR = {hr:.2f} (95% CI {lo:.2f}–{hi:.2f})\np = {pv:.3f}",
              transform=ax_a.transAxes, fontsize=10.5, va='bottom', ha='right',
              bbox=dict(facecolor='white', edgecolor=COL_LIGHTGREY, boxstyle='round,pad=0.45'))
    ax_a.set_xlim(0, max_year); ax_a.set_ylim(0.0, 0.20)
    ax_a.set_xticks(range(0, max_year+1, 2))
    ax_a.set_xlabel(''); ax_a.set_ylabel('Cumulative incidence')
    ax_a.legend(loc='upper left', fontsize=10)
    style_axes(ax_a); panel_label(ax_a, 'a')

    # ---- Panel b: hr-HPV clearance (cumulative clearance probability) ----
    ax_b = fig.add_subplot(gs[0, 1])
    sub_b = BC.copy()
    sub_b['time']  = np.where(sub_b['has_clearance'],
                               (sub_b['first_neg_date']-sub_b['index_date']).dt.days,
                               (sub_b['최종추적일자']    -sub_b['index_date']).dt.days)
    sub_b['event'] = sub_b['has_clearance'].astype(int)
    sub_b = sub_b[sub_b['time'] > 0]
    v_b = sub_b[sub_b['vac']==1]; c_b = sub_b[sub_b['vac']==0]
    kmf_v2 = KaplanMeierFitter().fit(v_b['time']/365.25, v_b['event'], label='Vaccinated')
    kmf_c2 = KaplanMeierFitter().fit(c_b['time']/365.25, c_b['event'], label='Non-vaccinated')
    kmf_v2.plot_cumulative_density(ax=ax_b, ci_alpha=0.15, color=COL_VAC, lw=LINE_W)
    kmf_c2.plot_cumulative_density(ax=ax_b, ci_alpha=0.15, color=COL_CTL, lw=LINE_W)

    cph2 = CoxPHFitter().fit(
        sub_b[['time','event','vac','index_age','fine_match_id']].dropna(),
        duration_col='time', event_col='event', cluster_col='fine_match_id', robust=True)
    sm2 = cph2.summary; hr2=sm2.loc['vac','exp(coef)']
    lo2=sm2.loc['vac','exp(coef) lower 95%']; hi2=sm2.loc['vac','exp(coef) upper 95%']
    pv2=sm2.loc['vac','p']
    ax_b.text(0.55, 0.05,
              f"HR = {hr2:.2f} (95% CI {lo2:.2f}–{hi2:.2f})\np = {pv2:.3f}",
              transform=ax_b.transAxes, fontsize=10.5, va='bottom', ha='center',
              bbox=dict(facecolor='white', edgecolor=COL_LIGHTGREY, boxstyle='round,pad=0.45'))
    ax_b.set_xlim(0, max_year); ax_b.set_ylim(0.0, 0.85)
    ax_b.set_xticks(range(0, max_year+1, 2))
    ax_b.set_xlabel(''); ax_b.set_ylabel('Cumulative clearance probability')
    ax_b.legend(loc='upper right', fontsize=10)
    style_axes(ax_b); panel_label(ax_b, 'b')

    # ---- At-risk tables (one per panel) ----
    for col_pos, (v_d, c_d) in enumerate([(v_a, c_a), (v_b, c_b)]):
        ax_tab = fig.add_subplot(gs[1, col_pos])
        ax_tab.set_xlim(-3.6, max_year); ax_tab.set_ylim(0, 3); ax_tab.axis('off')
        years_list = list(range(0, max_year+1, 2))
        ax_tab.text(-3.5, 2.4, 'No. at risk', fontsize=10.5, fontweight='bold', ha='left')
        for yr in years_list:
            n_v_ = int((v_d['time']/365.25 >= yr).sum())
            n_c_ = int((c_d['time']/365.25 >= yr).sum())
            ax_tab.text(yr, 2.4, str(yr), fontsize=10.5, ha='center', fontweight='bold')
            ax_tab.text(yr, 1.4, str(n_v_), fontsize=10, ha='center', color=COL_VAC)
            ax_tab.text(yr, 0.4, str(n_c_), fontsize=10, ha='center', color=COL_CTL)
        ax_tab.text(-3.5, 1.4, 'Vaccinated',     fontsize=10, color=COL_VAC, ha='left')
        ax_tab.text(-3.5, 0.4, 'Non-vaccinated', fontsize=10, color=COL_CTL, ha='left')
        ax_tab.text(max_year/2, -0.6, 'Years from index date',
                   fontsize=11, ha='center', color='#222')

    plt.savefig('Data/Figure3_CohortB_CIF.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: Data/Figure3_CohortB_CIF.png')

# =====================================================================
# Figure 4 — Combined subgroup forest (replaces old Figure 4 + Figure 5)
#   JAMA-style table+forest. Two panels (two primary outcomes).
#   Subgroups: Overall, by age at index, by vaccine type.
# =====================================================================
def figure4_subgroup():
    """Subgroup forest plot for Cohort B co-primary outcomes.
    Subgroups: Overall and by age at index (<40 / 40–49 / ≥50 yrs).
    """
    print('  Building Figure 4 (combined subgroup forest, v3 primary)...')
    LANDMARK_DAYS = 90
    B = pd.read_csv('Data/primary_cohort_v3.csv', encoding='utf-8-sig')
    B['index_date']      = pd.to_datetime(B['index_date'])
    B['최종추적일자']      = pd.to_datetime(B['최종추적일자'])
    B['recurrence_date'] = pd.to_datetime(B['recurrence_date'], errors='coerce')
    B['lm_zero'] = B['index_date'] + pd.Timedelta(days=LANDMARK_DAYS)
    Bc = pd.read_csv('Data/final_matched_cohort.csv', encoding='utf-8-sig')
    B = B.merge(Bc[['연구번호','백신종류']], on='연구번호', how='left')
    B['vac'] = B['접종여부'].astype(bool).astype(int)
    B['index_age'] = pd.to_numeric(B['index_age'], errors='coerce')
    # Apply landmark to recurrence event-time
    B['rec_pre_lm'] = (B['has_recurrence']==True) & (B['recurrence_date'] < B['lm_zero'])
    bad_p1 = B[(B['vac']==1) & B['rec_pre_lm']]['fine_match_id'].unique()
    B = B[~B['fine_match_id'].isin(bad_p1)].copy()
    B = B[~B['rec_pre_lm']].copy()
    B['rec_time_lm'] = np.where(B['has_recurrence']==True,
                                 (B['recurrence_date'] - B['lm_zero']).dt.days,
                                 (B['최종추적일자']      - B['lm_zero']).dt.days)
    B['follow_up_days'] = (B['최종추적일자'] - B['lm_zero']).dt.days  # for clearance default

    # Inherit vaccine type from matched vaccinated participant to controls
    vt_by_match = B.loc[B['vac']==1].groupby('fine_match_id')['백신종류'].first()
    B['vacc_type'] = B.apply(
        lambda r: r['백신종류'] if r['vac']==1 else vt_by_match.get(r['fine_match_id'], np.nan),
        axis=1)
    B['age_grp'] = pd.cut(B['index_age'], bins=[-np.inf, 40, 50, np.inf],
                          labels=['<40', '40-49', '≥50'])

    # rec_time = landmark-adjusted (already set above as rec_time_lm)
    B['rec_time'] = B['rec_time_lm']

    # Add clearance event/time per patient — recompute from molecular pathology
    # with pre-vaccine baseline (HR+ before index) + 2-consecutive-negative event +
    # 3-mo landmark.
    import sys as _sys; _sys.path.insert(0, 'scripts')
    from extract_pathology_outcomes import detect_high_risk_hpv
    _path = pd.read_csv(
        'Data/한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV',
        encoding='cp949', low_memory=False)
    _path['실시일자'] = pd.to_datetime(_path['실시일자'], format='%Y%m%d', errors='coerce')
    _mol = _path[_path['병리검사구분'].isin(['분자병리','HPV'])].dropna(subset=['실시일자','판독결과'])
    _res = _mol['판독결과'].apply(detect_high_risk_hpv)
    _mol = _mol.assign(hpv_pos=_res.apply(lambda d: d['is_high_risk_hpv_positive']))
    _mol_by_pid = {pid: g.sort_values('실시일자') for pid, g in _mol.groupby('연구번호')}
    def _prevac_hr(pid, idx_dt):
        sub = _mol_by_pid.get(pid)
        return False if sub is None else bool((sub[sub['실시일자']<idx_dt]['hpv_pos']==True).any())
    def _first_two_neg(pid, idx_dt):
        sub = _mol_by_pid.get(pid)
        if sub is None: return None
        sub = sub[sub['실시일자']>idx_dt]
        if len(sub)<2: return None
        pos = sub['hpv_pos'].values; dates = sub['실시일자'].values
        for i in range(len(pos)-1):
            if (not pos[i]) and (not pos[i+1]):
                return pd.Timestamp(dates[i])
        return None
    B['prevac_hr'] = B.apply(lambda r: _prevac_hr(r['연구번호'], r['index_date']), axis=1)
    B['first_neg_date'] = B.apply(
        lambda r: _first_two_neg(r['연구번호'], r['index_date']) if r['prevac_hr'] else None,
        axis=1)
    B['has_clearance'] = B['first_neg_date'].notna()
    B['clr_pre_lm'] = B['has_clearance'] & (B['first_neg_date'] < B['lm_zero'])
    # Matched-set integrity for clearance: drop fids where the vaccinated case
    # had a clearance event before the landmark.
    bad_p2 = B[(B['vac']==1) & B['clr_pre_lm']]['fine_match_id'].unique()
    B = B[~B['fine_match_id'].isin(bad_p2)].copy()
    # Use time-from-index (not time-from-lm_zero) to match analyze_primary_v3.py;
    # also drop any remaining rows with pre-landmark clearance events so the
    # clearance analytic sample equals the canonical P2 subset (n = 233).
    B = B[~B['clr_pre_lm']].copy()
    B['clear_event'] = B['has_clearance'].astype(int)
    B['clear_time']  = np.where(
        B['has_clearance'],
        (B['first_neg_date'] - B['index_date']).dt.days,
        (B['최종추적일자']     - B['index_date']).dt.days)
    # Restrict clearance analyses to the canonical P2 analytic subset:
    # matched-set integrity (only fine_match_ids whose vaccinated case is
    # itself pre-vaccine hr-HPV+) + individual pre-vaccine hr-HPV+ (vac or
    # ctl). This matches the P2 cohort definition in analyze_primary_v3.py
    # and yields n = 233 (92 vac / 141 ctl) with HR 1.82 instead of the
    # loose n = 402 (92 / 310, HR 2.44) that drops only the individual-row
    # filter.
    fids_clr = set(B.loc[(B['vac']==1) & (B['prevac_hr'].astype(bool)),
                          'fine_match_id'].unique())
    keep_clr = B['fine_match_id'].isin(fids_clr) & B['prevac_hr'].astype(bool)
    B.loc[~keep_clr, 'clear_event'] = np.nan
    B.loc[~keep_clr, 'clear_time']  = np.nan

    def hr_subset(d, ev_col, time_col='follow_up_days'):
        """Generic Cox HR with cluster on fine_match_id and age adjustment.
        For lesion recurrence: ev_col='has_recurrence', time_col='follow_up_days'.
        For clearance:        ev_col='clear_event',    time_col='clear_time'.
        """
        df_fit = d[[time_col, ev_col, 'vac', 'index_age', 'fine_match_id']].dropna().rename(
            columns={time_col:'time', ev_col:'event'})
        df_fit['event'] = df_fit['event'].astype(int)
        df_fit = df_fit[df_fit['time'] > 0]
        n_v = int((df_fit['vac']==1).sum()); n_c = int((df_fit['vac']==0).sum())
        ev_v = int(((df_fit['vac']==1) & (df_fit['event']==1)).sum())
        ev_c = int(((df_fit['vac']==0) & (df_fit['event']==1)).sum())
        out = dict(n_v=n_v, n_c=n_c, ev_v=ev_v, ev_c=ev_c,
                   HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
        if ev_v + ev_c < 3 or n_v < 2 or n_c < 2 or df_fit['event'].sum() < 3:
            return out
        try:
            cph = CoxPHFitter().fit(df_fit, duration_col='time', event_col='event',
                                    cluster_col='fine_match_id', robust=True)
            r = cph.summary.loc['vac']
            out.update(HR=float(r['exp(coef)']),
                       CIlo=float(r['exp(coef) lower 95%']),
                       CIhi=float(r['exp(coef) upper 95%']),
                       p=float(r['p']))
        except Exception:
            pass
        return out

    def age_interaction_p(d, ev_col, time_col='follow_up_days'):
        df = d[[time_col, ev_col, 'vac', 'index_age', 'age_grp', 'fine_match_id']].dropna().copy()
        df = df.rename(columns={time_col:'time', ev_col:'event'})
        df['event'] = df['event'].astype(int)
        df = df[df['time'] > 0]
        df['ag_4049']      = (df['age_grp']=='40-49').astype(int)
        df['ag_50p']       = (df['age_grp']=='≥50').astype(int)
        df['vac_x_4049']   = df['vac'] * df['ag_4049']
        df['vac_x_50p']    = df['vac'] * df['ag_50p']
        full_cols = ['time','event','vac','index_age','ag_4049','ag_50p',
                     'vac_x_4049','vac_x_50p','fine_match_id']
        red_cols  = ['time','event','vac','index_age','ag_4049','ag_50p','fine_match_id']
        try:
            full = CoxPHFitter().fit(df[full_cols], duration_col='time', event_col='event',
                                     cluster_col='fine_match_id', robust=True)
            red  = CoxPHFitter().fit(df[red_cols], duration_col='time', event_col='event',
                                     cluster_col='fine_match_id', robust=True)
            lrt = 2*(full.log_likelihood_ - red.log_likelihood_)
            return float(1 - chi2.cdf(lrt, df=2))
        except Exception:
            return np.nan

    def vacc_interaction_p(d, ev_col, time_col='follow_up_days'):
        df = d[[time_col, ev_col, 'vac', 'index_age', 'vacc_type', 'fine_match_id']].dropna().copy()
        df = df.rename(columns={time_col:'time', ev_col:'event'})
        df['event'] = df['event'].astype(int)
        df = df[df['time'] > 0]
        df['type_Cervarix']   = (df['vacc_type']=='Cervarix').astype(int)
        df['type_Gardasil']   = (df['vacc_type']=='Gardasil').astype(int)
        df['vac_x_Cervarix']  = df['vac'] * df['type_Cervarix']
        df['vac_x_Gardasil']  = df['vac'] * df['type_Gardasil']
        full_cols = ['time','event','vac','index_age','type_Cervarix','type_Gardasil',
                     'vac_x_Cervarix','vac_x_Gardasil','fine_match_id']
        red_cols  = ['time','event','vac','index_age','type_Cervarix','type_Gardasil','fine_match_id']
        try:
            full = CoxPHFitter().fit(df[full_cols], duration_col='time', event_col='event',
                                     cluster_col='fine_match_id', robust=True)
            red  = CoxPHFitter().fit(df[red_cols], duration_col='time', event_col='event',
                                     cluster_col='fine_match_id', robust=True)
            lrt = 2*(full.log_likelihood_ - red.log_likelihood_)
            return float(1 - chi2.cdf(lrt, df=2))
        except Exception:
            return np.nan

    # ----- Outcome specifications -----
    # Panel a: lesion recurrence on full Cohort B    (HR < 1 favourable)
    # Panel b: hr-HPV clearance on pre-vaccine HPV+   (HR > 1 favourable)
    outcome_specs = [
        ('a', 'Lesion recurrence (≥CIN2)',           'has_recurrence', 'rec_time', B,
         'protective_lt1'),
        ('b', 'hr-HPV clearance (pre-vaccine HPV+)', 'clear_event',    'clear_time',
         B[B['clear_event'].notna()].copy(), 'protective_gt1'),
    ]

    panel_data = {}
    for plabel, ttl, ev_col, time_col, sub, direction in outcome_specs:
        rows = []
        rows.append(('Overall', hr_subset(sub, ev_col, time_col), 'data'))
        rows.append(('', None, 'spacer'))
        rows.append(('Age groups', None, 'header'))
        for grp_key, grp_lab in [('<40', '<40 years'),
                                  ('40-49', '40–49 years'),
                                  ('≥50', '≥50 years')]:
            rows.append((grp_lab,
                         hr_subset(sub[sub['age_grp']==grp_key], ev_col, time_col),
                         'data'))
        ap = age_interaction_p(sub, ev_col, time_col)
        rows.append((f'P for interaction = {ap:.3f}' if not np.isnan(ap) else
                     'P for interaction = NA', None, 'pval'))
        panel_data[plabel] = (ttl, rows, direction)

    # ----- Plot -----
    n_rows = max(len(rows) for _, rows, _ in panel_data.values())
    fig_h = max(7.5, 0.45 * n_rows + 2.8)
    fig, axes = plt.subplots(1, 2, figsize=(24.0, fig_h),
                             gridspec_kw={'left':0.025, 'right':0.995,
                                          'top':0.93, 'bottom':0.06, 'wspace':0.12})

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

    for idx, (plabel, (ttl, rows, direction)) in enumerate(panel_data.items()):
        ax = axes[idx]
        n = len(rows)
        favourable_lt1 = (direction == 'protective_lt1')
        ax.set_xlim(0, 1); ax.set_ylim(n + 1.6, HEADER_Y - 2.4)
        ax.axis('off')

        # ---- Column headers ----
        ax.text(XCOL['label'], HEADER_Y, 'Subgroup', fontweight='bold',
                fontsize=13, ha='left', va='center')
        ax.text(XCOL['vac'], HEADER_Y, 'Vaccinated', fontweight='bold',
                fontsize=13, ha='center', va='center')
        ax.text(XCOL['vac'], SUBHEAD_Y, 'No. events / N', fontsize=11,
                ha='center', va='center', color='#444')
        ax.text(XCOL['ctl'], HEADER_Y, 'Non-vaccinated', fontweight='bold',
                fontsize=13, ha='center', va='center')
        ax.text(XCOL['ctl'], SUBHEAD_Y, 'No. events / N', fontsize=11,
                ha='center', va='center', color='#444')
        ax.text((XCOL['forest_lo']+XCOL['forest_hi'])/2, HEADER_Y,
                'Hazard ratio (95% CI)', fontweight='bold',
                fontsize=13, ha='center', va='center')
        ax.text(XCOL['hrtxt']+0.06, HEADER_Y, 'HR (95% CI)',
                fontweight='bold', fontsize=13, ha='left', va='center')

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
                        fontweight='bold', fontsize=12.5, ha='left', va='center')
            elif kind == 'pval':
                ax.text(XCOL['label']+0.025, y, label, fontsize=11.5,
                        fontstyle='italic', ha='left', va='center', color='#444')
            elif kind == 'spacer':
                continue
            elif kind == 'data' and data is not None:
                indent = 0.025 if label != 'Overall' else 0.0
                weight = 'bold' if label == 'Overall' else 'normal'
                ax.text(XCOL['label']+indent, y, label, fontsize=12.5,
                        fontweight=weight, ha='left', va='center')
                ax.text(XCOL['vac'], y, f"{data['ev_v']} / {data['n_v']}",
                        fontsize=12.5, ha='center', va='center')
                ax.text(XCOL['ctl'], y, f"{data['ev_c']} / {data['n_c']}",
                        fontsize=12.5, ha='center', va='center')
                if not np.isnan(data['HR']):
                    sig = (data['CIlo'] > 1) or (data['CIhi'] < 1)
                    favourable = ((data['HR']<1) == favourable_lt1)
                    color = (COL_VAC if (favourable and sig)
                             else COL_CTL if ((not favourable) and sig)
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
                            fontsize=12, ha='left', va='center')
                else:
                    ax.text(XCOL['hrtxt']+0.06, y, '— insufficient events —',
                            fontsize=11.5, ha='left', va='center',
                            color=COL_GREY, style='italic')

        # ---- Forest x-axis ticks (below last row) ----
        x_axis_y = n - 0.1
        ax.plot([XCOL['forest_lo'], XCOL['forest_hi']], [x_axis_y, x_axis_y],
                color='black', lw=0.9)
        for tk in XTICKS:
            xt = x_to_axes(tk)
            ax.plot([xt, xt], [x_axis_y, x_axis_y + 0.12], color='black', lw=0.9)
            ax.text(xt, x_axis_y + 0.45, str(tk),
                    fontsize=11.5, ha='center', va='center')
        ax.text((XCOL['forest_lo']+XCOL['forest_hi'])/2, x_axis_y + 1.05,
                'Hazard ratio (log scale)', fontsize=10.5, ha='center', va='center')

        # ---- Panel label (a, b) — placed with extra clearance above the header row ----
        ax.text(-0.005, HEADER_Y - 1.8, plabel, fontsize=16, fontweight='bold',
                ha='left', va='bottom', clip_on=False)

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
    print('[Figure 2 — Cohort A v3 primary (≥2 dose + 3-mo landmark)]')
    m_A = build_cohort_a_matched()
    # Apply v3 primary filters: ≥2 dose + 3-mo landmark with pair_id integrity
    import re as _re
    _NAME = _re.compile(r'gardasil|cervarix|hpv vaccine', _re.I)
    _KOR = _re.compile(r'가다실|서바릭스')
    _CODE = ('DV-9HPF','DV-HPF','DV-JHP','DV-HPJ')
    def _is_vac(row):
        return bool(_NAME.search(str(row.get('처방명','')))) or \
               bool(_KOR.search(str(row.get('처방한글명','')))) or \
               str(row.get('처방코드','')).startswith(_CODE)
    _rx = pd.read_csv('Data/한국 HPV 코호트 자료를 이용한 자_처방정보.csv',
                       encoding='cp949', low_memory=False)
    _rx = _rx[_rx.apply(_is_vac, axis=1)].copy()
    _rx['처방일자_dt'] = pd.to_datetime(_rx['처방일자'], format='%Y%m%d', errors='coerce')
    _doses = _rx.dropna(subset=['처방일자_dt']).groupby('연구번호')['처방일자_dt'].nunique()
    m_A = m_A.merge(_doses.rename('dose_count'), left_on='pid', right_index=True, how='left')
    m_A['dose_count'] = m_A['dose_count'].fillna(0).astype(int)
    # ≥2-dose filter
    _bad_d = m_A[(m_A['vaccinated']==True) & (m_A['dose_count']<2)]['pair_id'].unique()
    m_A = m_A[~m_A['pair_id'].isin(_bad_d)].copy()
    # 3-mo landmark
    LM = pd.Timedelta(days=90)
    _fu = (m_A['last_follow'] - m_A['index_date']).dt.days
    _bad_lm = m_A[(m_A['vaccinated']==True) & (_fu < 90)]['pair_id'].unique()
    m_A = m_A[~m_A['pair_id'].isin(_bad_lm)].copy()
    m_A = m_A[(m_A['last_follow']-m_A['index_date']).dt.days >= 90].copy()
    # Shift index_date to landmark for time-to-event analysis
    m_A['index_date'] = m_A['index_date'] + LM
    print(f'  Cohort A v3 cohort: n={len(m_A)} '
          f'(vac {(m_A["vaccinated"]==True).sum()} / non {(m_A["vaccinated"]==False).sum()})')
    figure2(m_A)
    print('[Figure 3]')
    figure3()
    print('[Figure 4 — combined subgroup forest]')
    figure4_subgroup()
    print('\nAll main figures regenerated at 300 dpi with unified style.')
