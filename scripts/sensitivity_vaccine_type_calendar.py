"""
Sensitivity analysis: is the apparent quadrivalent-specific protective signal
on post-index hr-HPV detection (HR 0.49, LRT p = 0.037) attributable to
calendar-period confounding?

Rationale: in Korea, Gardasil 4-valent dominated 2007-2016, then was
replaced by Gardasil 9-valent. Quadrivalent recipients therefore differed
from 9-valent recipients in calendar year of vaccination, surgery-to-
vaccination interval, and the surveillance environment under which their
post-index HPV testing occurred. If the quadrivalent finding is real and
mechanistic (vaccine antigen coverage), it should persist when restricted
to a calendar window that contains all three products. If it is calendar-
confounded, it should attenuate or disappear.

Outputs (Data/):
  Sensitivity_VaccineType_ByCalendar.csv  HR table by calendar-period stratum
  Sensitivity_VaccineType_Descriptives.csv  vaccine-type x index-year n's,
                                            surgery-to-vaccine interval
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from lifelines import CoxPHFitter
from scipy.stats import chi2

B  = pd.read_csv('Data/final_matched_outcomes.csv', encoding='utf-8-sig')
Bc = pd.read_csv('Data/final_matched_cohort.csv', encoding='utf-8-sig')
B = B.merge(Bc[['연구번호','백신종류','첫수술일자','수술_접종_간격일']],
            on='연구번호', how='left')
B['vac']            = B['접종여부'].astype(bool).astype(int)
B['follow_up_days'] = pd.to_numeric(B['follow_up_days'], errors='coerce')
B['index_age']      = pd.to_numeric(B['index_age'], errors='coerce')
B['index_date']     = pd.to_datetime(B['index_date'])
B['index_year']     = B['index_date'].dt.year
B['surg_to_vac_days'] = pd.to_numeric(B['수술_접종_간격일'], errors='coerce')

# Inherit vaccine type from matched vaccinated participant to controls
vt_by_match = B.loc[B['vac']==1].groupby('fine_match_id')['백신종류'].first()
B['vacc_type'] = B.apply(
    lambda r: r['백신종류'] if r['vac']==1
    else vt_by_match.get(r['fine_match_id'], np.nan), axis=1)

# ---------------------------------------------------------------------------
# Descriptives: vaccine type x index year, surgery-to-vaccine interval
# ---------------------------------------------------------------------------
print('=== Vaccine-type × index-year (vaccinated arm only) ===')
vac_only = B[B['vac']==1]
year_bins = [-np.inf, 2014, 2016, 2018, np.inf]
year_labels = ['≤2014', '2015–2016', '2017–2018', '≥2019']
vac_only = vac_only.copy()
vac_only['period'] = pd.cut(vac_only['index_year'], bins=year_bins, labels=year_labels)
xtab = pd.crosstab(vac_only['vacc_type'], vac_only['period'], margins=True)
print(xtab.to_string()); print()

print('Index year median (vaccinated arm) by vaccine type:')
desc = vac_only.groupby('vacc_type')['index_year'].agg(['median','min','max','count'])
print(desc.to_string()); print()

print('Surgery-to-vaccine interval (days), vaccinated arm by vaccine type:')
sv = vac_only.groupby('vacc_type')['surg_to_vac_days'].agg(
    ['median','mean','min','max','count'])
print(sv.to_string()); print()

# Save descriptives
desc_out = (vac_only.groupby('vacc_type')
            .agg(n=('vacc_type','size'),
                 median_index_year=('index_year','median'),
                 min_index_year=('index_year','min'),
                 max_index_year=('index_year','max'),
                 median_surg_to_vac_days=('surg_to_vac_days','median'),
                 mean_surg_to_vac_days=('surg_to_vac_days','mean'))
            .reset_index())
desc_out.to_csv('Data/Sensitivity_VaccineType_Descriptives.csv',
                index=False, encoding='utf-8-sig')
print(f'Saved: Data/Sensitivity_VaccineType_Descriptives.csv\n')

# ---------------------------------------------------------------------------
# Calendar-period stratified vaccine-type interaction model
# ---------------------------------------------------------------------------
def fit_interaction(d, ev_col):
    """Returns dict with type-specific HRs from a single Cox interaction model."""
    d = d.dropna(subset=['vacc_type','follow_up_days','index_age']).copy()
    d['type_Cervarix'] = (d['vacc_type']=='Cervarix').astype(int)
    d['type_Gardasil'] = (d['vacc_type']=='Gardasil').astype(int)
    d['vac_x_Cervarix'] = d['vac'] * d['type_Cervarix']
    d['vac_x_Gardasil'] = d['vac'] * d['type_Gardasil']
    cols_full = ['follow_up_days', ev_col, 'vac', 'index_age',
                 'type_Cervarix','type_Gardasil',
                 'vac_x_Cervarix','vac_x_Gardasil', 'fine_match_id']
    cols_red  = ['follow_up_days', ev_col, 'vac', 'index_age',
                 'type_Cervarix','type_Gardasil', 'fine_match_id']
    d_full = d[cols_full].dropna().rename(columns={'follow_up_days':'time', ev_col:'event'})
    d_red  = d[cols_red].dropna().rename(columns={'follow_up_days':'time', ev_col:'event'})
    d_full['event'] = d_full['event'].astype(int)
    d_red['event']  = d_red['event'].astype(int)
    if d_full['event'].sum() < 5:
        return None
    try:
        cph_full = CoxPHFitter().fit(d_full, duration_col='time', event_col='event',
                                     cluster_col='fine_match_id', robust=True)
        cph_red  = CoxPHFitter().fit(d_red,  duration_col='time', event_col='event',
                                     cluster_col='fine_match_id', robust=True)
    except Exception as e:
        return None
    lrt = 2*(cph_full.log_likelihood_ - cph_red.log_likelihood_)
    lrt_p = float(1 - chi2.cdf(lrt, df=2))
    sm  = cph_full.summary
    cov = cph_full.variance_matrix_
    coef_v = sm.loc['vac','coef']; se_v = sm.loc['vac','se(coef)']
    coef_c = coef_v + sm.loc['vac_x_Cervarix','coef']
    coef_g = coef_v + sm.loc['vac_x_Gardasil','coef']
    se_c = np.sqrt(cov.loc['vac','vac'] + cov.loc['vac_x_Cervarix','vac_x_Cervarix']
                  + 2*cov.loc['vac','vac_x_Cervarix'])
    se_g = np.sqrt(cov.loc['vac','vac'] + cov.loc['vac_x_Gardasil','vac_x_Gardasil']
                  + 2*cov.loc['vac','vac_x_Gardasil'])
    def hrci(c, s):
        return float(np.exp(c)), float(np.exp(c-1.96*s)), float(np.exp(c+1.96*s))
    return {
        'lrt_chi2':float(lrt),'lrt_p':lrt_p,
        'Gardasil9_HR':hrci(coef_v, se_v)[0], 'Gardasil9_lo':hrci(coef_v, se_v)[1],
        'Gardasil9_hi':hrci(coef_v, se_v)[2],
        'Cervarix_HR':hrci(coef_c, se_c)[0],   'Cervarix_lo':hrci(coef_c, se_c)[1],
        'Cervarix_hi':hrci(coef_c, se_c)[2],
        'Gardasil_HR':hrci(coef_g, se_g)[0],   'Gardasil_lo':hrci(coef_g, se_g)[1],
        'Gardasil_hi':hrci(coef_g, se_g)[2],
    }

# Strata: full + by index-year cutoff
strata = [
    ('Full Cohort B (primary)', B),
    ('Index year ≤ 2015 (Gardasil-4v era)',  B[B['index_year'] <= 2015]),
    ('Index year 2016–2018 (mixed era)',     B[(B['index_year'] >= 2016) & (B['index_year'] <= 2018)]),
    ('Index year ≥ 2019 (Gardasil-9v era)',  B[B['index_year'] >= 2019]),
]

print('===== Vaccine-type interaction by calendar period (HPV reinfection) =====\n')
rows = []
for label, sub in strata:
    n_total = len(sub)
    n_v_g9  = ((sub['vac']==1) & (sub['vacc_type']=='Gardasil9')).sum()
    n_v_cv  = ((sub['vac']==1) & (sub['vacc_type']=='Cervarix')).sum()
    n_v_gd  = ((sub['vac']==1) & (sub['vacc_type']=='Gardasil')).sum()
    print(f'{label}: total {n_total}, vac (G9 / Cv / G4) = {n_v_g9} / {n_v_cv} / {n_v_gd}')
    res = fit_interaction(sub, 'has_hpv_infection')
    if res is None:
        print('  → insufficient events / fit failed\n')
        rows.append({'stratum':label, 'n_total':n_total,
                     'n_vac_G9':n_v_g9,'n_vac_Cv':n_v_cv,'n_vac_G4':n_v_gd,
                     **{k:np.nan for k in ['lrt_chi2','lrt_p',
                        'Gardasil9_HR','Gardasil9_lo','Gardasil9_hi',
                        'Cervarix_HR','Cervarix_lo','Cervarix_hi',
                        'Gardasil_HR','Gardasil_lo','Gardasil_hi']}})
        continue
    rows.append({'stratum':label, 'n_total':n_total,
                 'n_vac_G9':n_v_g9,'n_vac_Cv':n_v_cv,'n_vac_G4':n_v_gd, **res})
    print(f"  Gardasil 9    HR {res['Gardasil9_HR']:.2f} ({res['Gardasil9_lo']:.2f}–{res['Gardasil9_hi']:.2f})")
    print(f"  Cervarix 2v   HR {res['Cervarix_HR']:.2f} ({res['Cervarix_lo']:.2f}–{res['Cervarix_hi']:.2f})")
    print(f"  Gardasil 4v   HR {res['Gardasil_HR']:.2f} ({res['Gardasil_lo']:.2f}–{res['Gardasil_hi']:.2f})")
    print(f"  LRT for vaccine-type heterogeneity:  χ²(2) = {res['lrt_chi2']:.2f}, p = {res['lrt_p']:.3f}\n")

out = pd.DataFrame(rows)
out.to_csv('Data/Sensitivity_VaccineType_ByCalendar.csv',
           index=False, encoding='utf-8-sig')
print('Saved: Data/Sensitivity_VaccineType_ByCalendar.csv')
