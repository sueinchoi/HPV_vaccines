# Manuscript Submission — File Manifest and Naming Guide

**Manuscript**: Long-term safety and post-surgical efficacy of HPV vaccination in women with cervical intraepithelial lesions: a prospective–retrospective cohort study using Korean clinical data warehouse
**Last updated**: 2026-05-02
**Repository root**: `/Users/sueinchoi/GitHub/HPV_vaccines/`

---

## 0. Submission folder layout

When preparing the submission, copy the source files from `Data/` into a clean `submission/` folder using the standardised filenames in column **Submit as**. Convert to the format requested by the journal (see §4).

```
submission/
├── manuscript.docx                    # main text
├── tables/
│   ├── Table1_BaselineCharacteristics.docx
│   ├── Table2_CohortA_HazardRatios.docx
│   └── Table3_CohortB_HazardRatios.docx
├── figures/
│   ├── Figure1_CohortSelection.png
│   ├── Figure2_CohortA_CIF_HR.png
│   ├── Figure3_CohortB_CIF.png
│   └── Figure4_CohortB_Subgroup.png        # combined subgroup (replaces legacy Fig 4 + Fig 5)
└── supplementary/
    ├── SupTableS1_Baseline_PreMatching.docx
    ├── SupTableS2_PropensityScoreCoefficients.docx
    ├── SupTableS3_AgeFollowUpSubgroup.docx
    ├── SupTableS4_NumberAtRisk.docx
    ├── SupTableS5_VaccineTypeDetails.docx
    ├── SupTableS6_ClusterRobustHRs.docx
    ├── SupTableS7_PseudoIndexSensitivity.docx
    ├── SupTableS8_DoseThreshold_SensC.docx
    ├── SupTableS9_StrictMatching_SensD.docx
    ├── SupTableS10_TimeStratifiedClearance_SensB.docx
    ├── SupTableS11_SingleNegativeClearance_SensA.docx
    ├── SupTableS12_DiseaseFreeInterval_SensE.docx
    ├── SupFigS1_LovePlot_CohortA.png
    ├── SupFigS2_LovePlot_CohortB.png
    ├── SupFigS3_PropensityScoreDensity.png
    ├── SupFigS4_AgeSubgroupKM.png
    ├── SupFigS5a_PHCheck_CohortA_AnyOf5.png
    ├── SupFigS5b_PHCheck_CohortA_Diabetes.png
    ├── SupFigS6a_PHCheck_CohortB_Recurrence.png
    └── SupFigS6b_PHCheck_CohortB_HPVClearance.png
```

---

## 1. Main Tables

| # | Title | Source file (in `Data/`) | Submit as | Format |
|---|---|---|---|---|
| **Table 1** | Baseline characteristics of analytic cohorts (post-matching). Demographics, anthropometry, vital signs, smoking, comorbidities, and follow-up for Cohort A and Cohort B with absolute standardised mean differences. | `Table1_BaselineCharacteristics_unified.docx` (post-matching blocks (b) and (d) only — keep page breaks; remove pre-matching blocks (a)·(c) before submission) | `Table1_BaselineCharacteristics.docx` | DOCX |
| **Table 2** | Cohort A — Cluster-robust cause-specific and Fine–Gray subdistribution hazard ratios with person-years and incidence rates for the Any-of-5 composite, MCE composite, and five individual chronic conditions. | `Data/Table2_CohortA_HazardRatios.csv` (canonical) rebuilt to docx by `scripts/build_docx_artifacts.py` | `Table2_CohortA_HazardRatios.docx` | DOCX |
| **Table 3** | Cohort B — Age-adjusted Cox cluster-robust hazard ratios with person-years and incidence rates for the two co-primary outcomes under the ≥2-dose + 3-month landmark primary: biopsy-confirmed lesion recurrence (primary n = 912) and hr-HPV clearance defined as two consecutive post-index negatives among women with documented pre-vaccine hr-HPV positivity (primary n = 233). Includes the KM-estimated sustained-clearance summary (median + 5-year reversion-free probability) and Sens-C dose-threshold sensitivity rows (≥1-dose, no-landmark legacy; ≥3-dose, no-landmark). | `Table3_CohortB_HR_v3.docx` (or `Data/Table3_CohortB_HR.csv` rebuilt to docx) | `Table3_CohortB_HazardRatios.docx` | DOCX |

> **Note for Table 3 **: Use the values from `Data/CohortB_HR_v3.csv` and `Data/CohortB_SustainedClearance.csv`:
> - Lesion recurrence (P1): HR 1.01 (95% CI 0.49–2.06), p = 0.99
> - hr-HPV clearance (P2): HR 1.82 (95% CI 1.07–3.11), p = 0.027
> - Sustained clearance KM: vaccinated 10.79 y, non-vaccinated 5.67 y (log-rank p = 0.317); 5-year reversion-free probability 56.9% vs 53.3%
> 
> Sens-C rows (≥1-dose, no-landmark; ≥3-dose, no-landmark) are included beneath the primary rows for transparency. Bonferroni-adjusted columns and 97.5% CIs are not used.

---

## 2. Main Figures

| # | Title | Source file (in `Data/`) | Submit as | Format / Resolution |
|---|---|---|---|---|
| **Figure 1** | Cohort selection flow diagram — single source population (N = 32,969) → Cohort A (whole-cohort safety, n = 4,102) and Cohort B (post-surgical efficacy, primary n = 912). | `Figure1_cohort_selection.png` | `Figure1_CohortSelection.png` | PNG, 200 dpi |
| **Figure 2** | Cohort A — Aalen–Johansen cumulative incidence functions and cause-specific / Fine–Gray hazard ratio forest plot for Any-of-5, MCE, and five individual comorbidities. | `cohort_a_psm_cif_hr.png` | `Figure2_CohortA_CIF_HR.png` | PNG, 150 dpi |
| **Figure 3** | Cohort B — cumulative incidence (1 − Kaplan–Meier) curves for the two co-primary outcomes, anchored at the 3-month landmark (index + 90 days): biopsy-confirmed lesion recurrence (primary n = 912) and hr-HPV clearance among women with documented pre-vaccine hr-HPV positivity (primary n = 233). | `Figure3_CohortB_CIF.png` | `Figure3_CohortB_CIF.png` | PNG, 300 dpi |
| **Figure 4** | Cohort B — JAMA-style combined table-with-forest plot of both co-primary outcomes, with subgroups by age at index (<40 / 40–49 / ≥50). Replaces the legacy separate Figure 4 (vaccine type) and Figure 5 (age × FU) — both are now panels in this single figure. | `Figure4_CohortB_Subgroup.png` | `Figure4_CohortB_Subgroup.png` | PNG, 300 dpi |

---

## 3. Supplementary Materials

### 3.1 Supplementary Tables

| # | Title | Source file (in `Data/`) | Submit as | Format |
|---|---|---|---|---|
| **S1** | Pre-matching baseline characteristics (full variable list as Table 1, before matching for both cohorts). | `Table1_BaselineCharacteristics_unified.docx` (pre-matching blocks (a) and (c) only) | `SupTableS1_Baseline_PreMatching.docx` | DOCX |
| **S2** | Cohort A propensity score model — logistic regression coefficients (original-scale and standardised), odds ratios, and intercept. | `SupTableS2_ps_coefficients.docx` | `SupTableS2_PropensityScoreCoefficients.docx` | DOCX |
| **S3** | Cohort B subgroup analysis — pre-specified age strata (<40, 40–49, ≥50, plus 30–52 post-hoc) × follow-up windows (1, 2, 4 yr, full) for lesion recurrence. | `SupTableS4_revised_age_fu_forest.docx` | `SupTableS3_AgeFollowUpSubgroup.docx` | DOCX |
| **S4** | Number-at-risk tables for Kaplan–Meier and Aalen–Johansen curves by group at yearly intervals (0–8 yr). | `SupTableS5_number_at_risk.docx` | `SupTableS4_NumberAtRisk.docx` | DOCX |
| **S5** | Per-vaccine-type detailed results — pairwise subgroup hazard ratios and single-model interaction-derived hazard ratios with the LRT for vaccine-type heterogeneity. Sensitivity outcome rows (post-index hr-HPV detection) included for transparency. | `vaccine_type_analysis.csv` + `CohortB_vaccine_interaction.csv` | `SupTableS5_VaccineTypeDetails.docx` | DOCX |
| **S6** | Cluster-robust hazard ratios with person-years, incidence rates, and Schoenfeld residual p-values for both cohorts (Cohort A Any-of-5 / MCE / 5 individual chronic conditions; Cohort B lesion recurrence + hr-HPV clearance). Built from `Data/Table2_CohortA_HazardRatios.csv` + `Data/Table3_CohortB_HR.csv`. | inline in `docs/HPV_supplementary.docx` | `SupTableS6_ClusterRobustHRs.docx` | DOCX |
| **S7** | Pseudo-index assignment sensitivity analysis for Cohort A (Any-of-5 endpoint) — random sample, calendar-year-matched, and risk-set sampling strategies. | `CohortA_pseudoindex_sensitivity.csv` | `SupTableS7_PseudoIndexSensitivity.docx` | DOCX |
| **S8** *(Sens-C)* | Dose-threshold sensitivity for both cohorts — re-fitted hazard ratios under ≥1, ≥2, and ≥3 (complete schedule) dose definitions with matched-set integrity preserved. | `Sensitivity_DoseThreshold_HR.csv` | `SupTableS8_DoseThreshold_SensC.docx` | DOCX |
| **S9** *(Sens-D)* | Strict 1:4 fine-matching sensitivity for Cohort B lesion recurrence — variable-ratio (1:up-to-4, primary) vs strict (sensitivity) specifications. | `Sensitivity_StrictMatching.csv` | `SupTableS9_StrictMatching_SensD.docx` | DOCX |
| **S10** *(Sens-B)* | Time-stratified hr-HPV clearance hazard ratios decomposed into 0–6, 6–12, 12–24, and ≥24-month windows post-index (clearance subset, primary n = 233). | `Sensitivity_HPV_Clearance_TimeStratified.csv` | `SupTableS10_TimeStratifiedClearance_SensB.docx` | DOCX |
| **S11** *(Sens-A)* | Single-negative HPV clearance sensitivity — alternative event definition using the FIRST single post-index hr-HPV-negative record vs the two-consecutive-negative primary. | `Sensitivity_HPV_Clearance_SingleNegative.csv` | `SupTableS11_SingleNegativeClearance_SensA.docx` | DOCX |
| **S12** *(Sens-E)* | Disease-free-interval sensitivity for lesion recurrence — minimum 3-, 6-, and 12-month buffer from the index date before counting a recurrence event. | `Sensitivity_Recurrence_DFInterval.csv` | `SupTableS12_DiseaseFreeInterval_SensE.docx` | DOCX |

### 3.2 Supplementary Figures

| # | Title | Source file (in `Data/`) | Submit as | Format / Resolution |
|---|---|---|---|---|
| **S1** | Love plot — covariate balance before and after 1:1 propensity score matching (Cohort A). | `SupFigS1_loveplot_cohortA.png` | `SupFigS1_LovePlot_CohortA.png` | PNG, 200 dpi |
| **S2** | Love plot — covariate balance before and after fine variable-ratio (1:up-to-4) matching (Cohort B). | `SupFigS2_loveplot_cohortB.png` | `SupFigS2_LovePlot_CohortB.png` | PNG, 200 dpi |
| **S3** | Propensity score density distributions before and after matching (Cohort A). | `SupFigS3_ps_density.png` | `SupFigS3_PropensityScoreDensity.png` | PNG, 200 dpi |
| **S4** | Subgroup forest plot by age strata for Cohort B (legacy single-figure view; superseded by Figure 5 — retain as supplementary). | `figure3_subgroup_km.png` | `SupFigS4_AgeSubgroupKM.png` | PNG (existing) |
| **S5a** | Schoenfeld residual diagnostic plot — Cohort A, Any-of-5 endpoint. | `PH_check_A_0.png` | `SupFigS5a_PHCheck_CohortA_AnyOf5.png` | PNG, 130 dpi |
| **S5b** | Schoenfeld residual diagnostic plot — Cohort A, Diabetes endpoint. | `PH_check_A_1.png` | `SupFigS5b_PHCheck_CohortA_Diabetes.png` | PNG, 130 dpi |
| **S6a** | Schoenfeld residual diagnostic plot — Cohort B, lesion recurrence. | `PH_check_B_has_recurrence.png` | `SupFigS6a_PHCheck_CohortB_Recurrence.png` | PNG, 130 dpi |
| **S6b** | Schoenfeld residual diagnostic plot — Cohort B, hr-HPV clearance co-primary (primary n = 233, two-consecutive-negative event). | `PH_check_B_clearance.png` | `SupFigS6b_PHCheck_CohortB_HPVClearance.png` | PNG, 130 dpi |

---

## 4. Format conversion checklist

Most journals require:

| Asset type | Preferred format | Acceptable | Notes |
|---|---|---|---|
| Main figures | TIFF (LZW) ≥ 300 dpi or EPS/PDF (vector) | High-res PNG | Keep colour mode RGB unless print journal asks CMYK. Embed all fonts. |
| Main tables | DOCX | RTF / typeset in manuscript | Avoid landscape unless necessary; use journal's table style. |
| Supplementary figures | PNG/PDF | TIFF | Compress if > 5 MB. |
| Supplementary tables | DOCX, CSV, or XLSX | PDF | Single supplementary document (`Supplementary_Material.docx`) often preferred — combine with explicit captions. |

**Conversion commands (macOS / Linux)**

```bash
# PNG → TIFF (300 dpi)
sips -s format tiff --setProperty dpiHeight 300 --setProperty dpiWidth 300 figure.png --out figure.tiff

# CSV → DOCX (preserve simple table layout)
pandoc -f csv -t docx file.csv -o file.docx

# Combine supplementary into a single docx
pandoc SupTableS*.md SupFigS*.md -o Supplementary_Material.docx
```

---

## 5. Files NOT used in the final submission

The following analytic outputs in `Data/` were intermediate or superseded; **do not include** them in the submission package:

- `Table1_BaselineCharacteristics.docx` (early version — superseded by `_unified.docx`)
- `Table1_BaselineCharacteristics.csv` (early version)
- `Table3_CohortB_HR.csv` is now the canonical v3 source (rebuilt to `Data/Table3_CohortB_HR.docx`; legacy `Methodology_Revisions.docx` / `CohortB_HR_revised.csv` are pre-cluster-robust and superseded)
- `cohort_a_psm_loveplot.png` (early Cohort A love plot — superseded by `SupFigS1_loveplot_cohortA.png`)
- `love_plot.png` (early Cohort B love plot with Korean labels — superseded by `SupFigS2_loveplot_cohortB.png`)
- `figure2_forest_plot.png` (legacy single-outcome forest — superseded by Figure 4 layout)
- `figure4_baseline.png` (legacy baseline figure — superseded by Table 1)
- `figure6_vaccine_group_forest.png` (legacy duplicate of Figure 4)
- `subgroup_fine_tuning.csv` (intermediate diagnostic output)
- `sensitivity_age_cutoff.csv` (full ≈6,900-row grid search; cite in manuscript footnote but do not attach as supplementary table — the pre-specified subset is in S4)
- `outcomes_summary.csv`, `patient_outcomes_summary.csv`, `cohort_outcomes.csv` (intermediate outputs)
- `pathology_sample.csv`, `한국 HPV 코호트 자료를 이용한 자_*.csv` (raw source data — do not share)

---

## 6. Manuscript text — exact in-text references

When citing files in the manuscript text, use the **figure/table numbers only** (e.g., "Figure 2", "Supplementary Table S4"). The submission package's standardised filenames in column **Submit as** above ensure that the editorial system can match each callout to its file.

The current `docs/Manuscript_Draft.md` references are already consistent with the numbering above. Before final submission, run:

```bash
grep -nE "(Figure|Table)\s*[0-9SR]" docs/Manuscript_Draft.md | sort -u
```

to verify every callout has a corresponding entry in this manifest.

---

## 7. Reproducibility chain

For each manuscript asset, the script that produces it is recorded below for the journal's reproducibility statement and for any future revisions.

| Asset | Script |
|---|---|
| Table 1, Sup Table S1 | `scripts/baseline_table1_unified.py` |
| Table 2, Sup Table S7, Figure 2 | `scripts/cohort_a_psm_hr_cif.py` and `scripts/methodology_revision.py` |
| Table 3, Sup Table S7 | `scripts/methodology_revision.py` |
| Sup Table S2 (PS coefficients) | `scripts/cohort_a_psm.py` and `scripts/generate_missing_artifacts.py` |
| Sup Table S3 | `scripts/sensitivity_analysis.py` (and v2/v3 variants) |
| Sup Table S4, Figure 5 | `scripts/age_fu_forest.py` |
| Sup Table S5 | `scripts/generate_missing_artifacts.py` |
| Sup Table S6 (interaction) | `scripts/methodology_revision.py` |
| Sup Table S8 (pseudo-index sensitivity) | `scripts/methodology_revision.py` |
| Sup Fig S1, S2 | `scripts/regenerate_love_plots.py` |
| Sup Fig S3 | `scripts/generate_missing_artifacts.py` |
| Sup Fig S5a/b, S6a/b (PH plots) | `scripts/methodology_revision.py` |
| Figure 1 | `scripts/generate_missing_artifacts.py` |
| Figure 3, 4, Sup Fig S4 | `scripts/analyze_cohort.py`, `scripts/vaccine_type_analysis.py` |

All scripts use `random_seed = 42` for reproducibility.

---

## 8. Pre-submission checklist

- [ ] All figures saved at journal-required resolution and format
- [ ] Filenames match the **Submit as** column (case-sensitive)
- [ ] Each figure caption ≤ 50 words; each table footnote present
- [ ] Bonferroni-adjusted columns and 97.5% CIs removed from Table 3
- [ ] Cohort A median follow-up updated where needed (currently reported as approximately 5.9 years mean for Cohort A and 4.88 yr median for Cohort B vaccinated, 5.02 yr for controls)
- [ ] Korean-labelled assets (`love_plot.png`) replaced with English-only counterparts
- [ ] Old Bonferroni columns removed from any inherited tables
- [ ] PH check supplementary figures attached
- [ ] Reproducibility appendix lists Python 3.14.3, lifelines 0.30.3, scikit-learn 1.8.0, scipy 1.17.1, pandas 2.3.3, numpy 2.4.2
- [ ] IRB number filled in (currently placeholder)
- [ ] Author affiliations and ORCIDs added
- [ ] Conflict-of-interest and funding statements complete
