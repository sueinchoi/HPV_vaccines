"""
Figure 3 v3 — Cohort B co-primary cumulative incidence under
≥2-dose + 3-month landmark primary definition.

Panel a: Lesion recurrence (≥CIN2)
Panel b: hr-HPV clearance (two consecutive negatives) among pre-vaccine hr-HPV+

Time axis is years from landmark (index + 90 days).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from extract_pathology_outcomes import detect_high_risk_hpv  # noqa: E402

# Match the styling of make_main_figures.py
plt.rcParams.update({
    'font.family': ['DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'savefig.dpi': 300,
})
COL_VAC = '#C44536'
COL_CTL = '#2A6F97'
COL_LIGHTGREY = '#dee2e6'
LINE_W = 1.9
LANDMARK_DAYS = 90


def style_axes(ax):
    ax.grid(axis='y', alpha=0.25, linestyle=':', lw=0.6)
    ax.tick_params(length=3, width=0.8, which='major')


def panel_label(ax, label):
    # Top-left placement, consistent across panels
    ax.text(0.02, 1.05, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')


def build_p1_p2():
    df = pd.read_csv(ROOT / 'Data' / 'primary_cohort_v3.csv', encoding='utf-8-sig')
    df['index_date'] = pd.to_datetime(df['index_date'])
    df['최종추적일자'] = pd.to_datetime(df['최종추적일자'])
    df['recurrence_date'] = pd.to_datetime(df['recurrence_date'], errors='coerce')
    df['lm_zero'] = df['index_date'] + pd.Timedelta(days=LANDMARK_DAYS)

    # P1 — recurrence (time-from-index; landmark applied by excluding pre-landmark events)
    df['rec_pre_lm'] = (df['has_recurrence'] == True) & (df['recurrence_date'] < df['lm_zero'])
    bad_fids = df[(df['접종여부'] == True) & df['rec_pre_lm']]['fine_match_id'].unique()
    p1 = df[~df['fine_match_id'].isin(bad_fids)].copy()
    p1 = p1[~p1['rec_pre_lm']].copy()
    p1['time_days'] = np.where(
        p1['has_recurrence'] == True,
        (p1['recurrence_date'] - p1['index_date']).dt.days,
        (p1['최종추적일자'] - p1['index_date']).dt.days,
    )
    p1['event'] = p1['has_recurrence'].astype(int)
    p1['vac'] = p1['접종여부'].astype(int)
    p1 = p1[p1['time_days'] > 0].copy()

    # P2 — clearance
    path = pd.read_csv(
        ROOT / 'Data' / '한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV',
        encoding='cp949', low_memory=False,
    )
    path['실시일자'] = pd.to_datetime(path['실시일자'], format='%Y%m%d', errors='coerce')
    mol = path[path['병리검사구분'].isin(['분자병리', 'HPV'])].dropna(
        subset=['실시일자', '판독결과']
    )
    res = mol['판독결과'].apply(detect_high_risk_hpv)
    mol = mol.assign(hpv_pos=res.apply(lambda d: d['is_high_risk_hpv_positive']))
    mol_by_pid = {pid: g.sort_values('실시일자') for pid, g in mol.groupby('연구번호')}

    def prevac_hr(pid, idx_dt):
        sub = mol_by_pid.get(pid)
        if sub is None:
            return False
        return bool((sub[sub['실시일자'] < idx_dt]['hpv_pos'] == True).any())

    def first_two_neg(pid, idx_dt):
        sub = mol_by_pid.get(pid)
        if sub is None:
            return None
        sub = sub[sub['실시일자'] > idx_dt]
        if len(sub) < 2:
            return None
        pos = sub['hpv_pos'].values
        dates = sub['실시일자'].values
        for i in range(len(pos) - 1):
            if (not pos[i]) and (not pos[i + 1]):
                return pd.Timestamp(dates[i])
        return None

    df['prevac_hr'] = df.apply(
        lambda r: prevac_hr(r['연구번호'], r['index_date']), axis=1
    )
    fids = df[(df['접종여부'] == True) & (df['prevac_hr'] == True)]['fine_match_id'].unique()
    clr = df[df['fine_match_id'].isin(fids) & (df['prevac_hr'] == True)].copy()
    clr['first_neg_date'] = clr.apply(
        lambda r: first_two_neg(r['연구번호'], r['index_date']), axis=1
    )
    clr['has_clearance'] = clr['first_neg_date'].notna()
    clr['lm_zero'] = clr['index_date'] + pd.Timedelta(days=LANDMARK_DAYS)
    clr['clr_pre_lm'] = clr['has_clearance'] & (clr['first_neg_date'] < clr['lm_zero'])
    bad2 = clr[(clr['접종여부'] == True) & clr['clr_pre_lm']]['fine_match_id'].unique()
    clr = clr[~clr['fine_match_id'].isin(bad2)].copy()
    clr = clr[~clr['clr_pre_lm']].copy()
    clr['time_days'] = np.where(
        clr['has_clearance'],
        (clr['first_neg_date'] - clr['index_date']).dt.days,
        (clr['최종추적일자'] - clr['index_date']).dt.days,
    )
    clr['event'] = clr['has_clearance'].astype(int)
    clr['vac'] = clr['접종여부'].astype(int)
    clr = clr[clr['time_days'] > 0].copy()

    return p1, clr


def cox_hr(df):
    fit = df[['time_days', 'event', 'vac', 'index_age', 'fine_match_id']].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(fit, duration_col='time_days', event_col='event',
            cluster_col='fine_match_id', robust=True)
    s = cph.summary.loc['vac']
    return s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%'], s['p']


def main():
    p1, p2 = build_p1_p2()
    print(f'P1 n={len(p1)} ({(p1["vac"]==1).sum()} vac / {(p1["vac"]==0).sum()} non)')
    print(f'P2 n={len(p2)} ({(p2["vac"]==1).sum()} vac / {(p2["vac"]==0).sum()} non)')

    fig = plt.figure(figsize=(14, 8.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.4, 1.1],
                  hspace=0.32, wspace=0.26,
                  left=0.07, right=0.98, top=0.96, bottom=0.07)
    max_year = 10

    # ---- Panel a: Lesion recurrence ----
    ax_a = fig.add_subplot(gs[0, 0])
    v_a = p1[p1['vac'] == 1]
    c_a = p1[p1['vac'] == 0]
    kmf_v = KaplanMeierFitter().fit(v_a['time_days'] / 365.25, v_a['event'], label='Vaccinated')
    kmf_c = KaplanMeierFitter().fit(c_a['time_days'] / 365.25, c_a['event'], label='Non-vaccinated')
    kmf_v.plot_cumulative_density(ax=ax_a, ci_alpha=0.15, color=COL_VAC, lw=LINE_W)
    kmf_c.plot_cumulative_density(ax=ax_a, ci_alpha=0.15, color=COL_CTL, lw=LINE_W)
    hr, lo, hi, pv = cox_hr(p1)
    ax_a.text(0.97, 0.96,
              f"HR = {hr:.2f} (95% CI {lo:.2f}–{hi:.2f})\np = {pv:.3f}",
              transform=ax_a.transAxes, fontsize=10.5, va='top', ha='right',
              bbox=dict(facecolor='white', edgecolor=COL_LIGHTGREY,
                        boxstyle='round,pad=0.45'))
    ax_a.set_xlim(0, max_year)
    ax_a.set_ylim(0.0, 0.20)
    ax_a.set_xticks(range(0, max_year + 1, 2))
    ax_a.set_ylabel('Cumulative incidence')
    leg_a = ax_a.get_legend()
    if leg_a is not None: leg_a.remove()
    style_axes(ax_a)
    panel_label(ax_a, 'a')

    # ---- Panel b: hr-HPV clearance ----
    ax_b = fig.add_subplot(gs[0, 1])
    v_b = p2[p2['vac'] == 1]
    c_b = p2[p2['vac'] == 0]
    kmf_v2 = KaplanMeierFitter().fit(v_b['time_days'] / 365.25, v_b['event'], label='Vaccinated')
    kmf_c2 = KaplanMeierFitter().fit(c_b['time_days'] / 365.25, c_b['event'], label='Non-vaccinated')
    kmf_v2.plot_cumulative_density(ax=ax_b, ci_alpha=0.15, color=COL_VAC, lw=LINE_W)
    kmf_c2.plot_cumulative_density(ax=ax_b, ci_alpha=0.15, color=COL_CTL, lw=LINE_W)
    hr2, lo2, hi2, pv2 = cox_hr(p2)
    ax_b.text(0.97, 0.96,
              f"HR = {hr2:.2f} (95% CI {lo2:.2f}–{hi2:.2f})\np = {pv2:.3f}",
              transform=ax_b.transAxes, fontsize=10.5, va='top', ha='right',
              bbox=dict(facecolor='white', edgecolor=COL_LIGHTGREY,
                        boxstyle='round,pad=0.45'))
    ax_b.set_xlim(0, max_year)
    ax_b.set_ylim(0.0, 0.85)
    ax_b.set_xticks(range(0, max_year + 1, 2))
    ax_b.set_ylabel('Cumulative clearance probability')
    leg_b = ax_b.get_legend()
    if leg_b is not None: leg_b.remove()
    style_axes(ax_b)
    panel_label(ax_b, 'b')

    # ---- At-risk tables ----
    for col_pos, (v_d, c_d) in enumerate([(v_a, c_a), (v_b, c_b)]):
        ax_tab = fig.add_subplot(gs[1, col_pos])
        ax_tab.set_xlim(-3.6, max_year)
        ax_tab.set_ylim(0, 3)
        ax_tab.axis('off')
        ax_tab.text(-3.5, 2.4, 'No. at risk', fontsize=10.5, fontweight='bold', ha='left')
        for yr in range(0, max_year + 1, 2):
            n_v_ = int((v_d['time_days'] / 365.25 >= yr).sum())
            n_c_ = int((c_d['time_days'] / 365.25 >= yr).sum())
            ax_tab.text(yr, 2.4, str(yr), fontsize=10.5, ha='center', fontweight='bold')
            ax_tab.text(yr, 1.4, str(n_v_), fontsize=10, ha='center', color=COL_VAC)
            ax_tab.text(yr, 0.4, str(n_c_), fontsize=10, ha='center', color=COL_CTL)
        ax_tab.text(-3.5, 1.4, 'Vaccinated', fontsize=10, color=COL_VAC, ha='left')
        ax_tab.text(-3.5, 0.4, 'Non-vaccinated', fontsize=10, color=COL_CTL, ha='left')
        ax_tab.text(max_year / 2, -0.6, 'Years',
                    fontsize=11, ha='center', color='#222')

    # Shared horizontal legend (Vaccinated / Non-vaccinated) below all panels
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COL_VAC, lw=LINE_W, label='Vaccinated'),
        Line2D([0], [0], color=COL_CTL, lw=LINE_W, label='Non-vaccinated'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.01),
               fontsize=11, frameon=False, columnspacing=2.5)

    plt.savefig(ROOT / 'Data' / 'Figure3_CohortB_CIF.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: Data/Figure3_CohortB_CIF.png')


if __name__ == '__main__':
    main()
